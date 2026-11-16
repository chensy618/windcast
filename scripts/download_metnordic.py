"""Download MET Nordic v4 analyses (1 km) on the windcast reference grid from the MET Norway
thredds archive (OPeNDAP subset), the target of every experiment.

    python scripts/download_metnordic.py --start 2018-01-01 --end 2023-12-31

The reference grid is rows 1677:1991, cols 874:1217 of the native 2321 x 1796 MET Nordic
grid (314 x 343 cells, 1 km LCC), the same cells as the mtno E1 cache. One file per
analysis time (00 / 06 / 12 / 18 UTC): OUT/YYYY/mn_YYYYMMDDTHH.nc. Wind is stored as
earth-relative (u10, v10) from speed / meteorological direction. Precipitation and
radiation are the archive's 1-hour accumulations ending at the analysis time.
OUT/static.nc holds lat, lon, altitude and land fraction. Existing files are skipped.
"""
import argparse
import time
from pathlib import Path

import netCDF4
import numpy as np
import pandas as pd
import xarray as xr

URL = ("https://thredds.met.no/thredds/dodsC/metpparchivev4/{t:%Y/%m/%d}/"
       "met_analysis_1_0km_nordic_{t:%Y%m%dT%H}Z.nc")
OUT = Path("/cluster/work/projects/nn8106k/siyan/windcast/metnordic")
ROWS, COLS = slice(1677, 1991), slice(874, 1217)
FIELDS = {"t2m": "air_temperature_2m", "rh2m": "relative_humidity_2m",
          "msl": "air_pressure_at_sea_level", "tcc": "cloud_area_fraction",
          "tp1h": "precipitation_amount",
          "ssrd1h": "integral_of_surface_downwelling_shortwave_flux_in_air_wrt_time",
          "strd1h": "integral_of_surface_downwelling_longwave_flux_in_air_wrt_time"}


def get(ds, name, static=False):
    v = ds[name]
    a = v[ROWS, COLS] if static or v.ndim == 2 else v[0, ROWS, COLS]
    return np.ma.filled(np.ma.asarray(a).astype("f4"), np.nan)


def fetch(t: pd.Timestamp, out: Path):
    ds = netCDF4.Dataset(URL.format(t=t))
    try:
        data = {k: get(ds, v) for k, v in FIELDS.items()}
        s, d = get(ds, "wind_speed_10m"), np.deg2rad(get(ds, "wind_direction_10m"))
    finally:
        ds.close()
    data["u10"], data["v10"] = -s * np.sin(d), -s * np.cos(d)
    tmp = out.with_suffix(".tmp")
    xr.Dataset({k: (("y", "x"), a) for k, a in data.items()},
               attrs={"time": str(t), "source": URL.format(t=t)}
               ).to_netcdf(tmp, encoding={k: {"zlib": True, "complevel": 3} for k in data})
    tmp.rename(out)


def with_timeout(fn, args, timeout: float = 180.0):
    """Run fn(*args) in a forked child; OPeNDAP reads can hang forever on a stalled server
    (netCDF sets no read timeout), so a child that does not finish in `timeout` s is
    killed and the attempt counts as failed."""
    import multiprocessing as mp
    ctx = mp.get_context("fork")
    q = ctx.Queue()

    def run():
        try:
            fn(*args)
            q.put(None)
        except Exception as e:  # sent back to the parent
            q.put(f"{type(e).__name__}: {e}")

    p = ctx.Process(target=run)
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.kill()
        p.join()
        raise TimeoutError(f"no answer in {timeout:.0f} s")
    err = q.get() if not q.empty() else f"child exit code {p.exitcode}"
    if err:
        raise RuntimeError(err)


def static(out: Path, t: pd.Timestamp):
    ds = netCDF4.Dataset(URL.format(t=t))
    try:
        data = {k: get(ds, k, static=True) for k in ("latitude", "longitude")}
        for k in ("altitude", "land_area_fraction"):
            if k in ds.variables:
                data[k] = get(ds, k, static=True)
    finally:
        ds.close()
    xr.Dataset({k: (("y", "x"), a) for k, a in data.items()}).to_netcdf(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2018-01-01")
    p.add_argument("--end", default="2025-12-31")
    p.add_argument("--hours", type=int, nargs="+", default=[0, 6, 12, 18])
    p.add_argument("--out", type=Path, default=OUT)
    a = p.parse_args()
    times = pd.date_range(a.start, f"{a.end} 23:00", freq="h")
    times = times[times.hour.isin(a.hours)]
    a.out.mkdir(parents=True, exist_ok=True)
    if not (a.out / "static.nc").exists():
        static(a.out / "static.nc", times[0])
    n_ok = n_fail = 0
    for t in times:
        out = a.out / f"{t:%Y}" / f"mn_{t:%Y%m%dT%H}.nc"
        if out.exists():
            continue
        out.parent.mkdir(exist_ok=True)
        for attempt in range(4):
            try:
                t0 = time.time()
                with_timeout(fetch, (t, out))
                n_ok += 1
                print(f"{t} ok {time.time() - t0:.0f}s", flush=True)
                break
            except Exception as e:
                msg = str(e).splitlines()[0][:120]
                if attempt == 3 or "file not found" in msg:
                    n_fail += 1
                    print(f"{t} FAIL {msg}", flush=True)
                    with open(a.out / "missing.txt", "a") as f:
                        f.write(f"{t:%Y%m%dT%H} {msg}\n")
                    break
                time.sleep(30 * (attempt + 1))
    print(f"done: {n_ok} ok, {n_fail} failed", flush=True)


if __name__ == "__main__":
    main()
