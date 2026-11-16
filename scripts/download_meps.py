"""Download MEPS 2.5 km deterministic forecasts over Northern Norway from the MET Norway
thredds archive (OPeNDAP subset: window, leads, variables), for forecast downscaling (track A).

    python scripts/download_meps.py --start 2018-01-01 --end 2025-12-31 [--inits 0 12]

One file per run: OUT/YYYY/meps_YYYYMMDDTHH.nc with (lead, y, x) fields plus lat/lon:
wind (u10, v10, gust), t2m, rh2m, msl, total cloud, accumulated precipitation, BLH.
MEPS x/y_wind_10m are grid-relative; they are rotated to earth-relative (u10 east, v10
north), as CARRA / MET Nordic (checked against wind_speed / wind_direction in meps_det).
Existing files are skipped if they hold every requested lead.
"""
import argparse
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import netCDF4
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from windcast.geo import to_plane  # noqa: E402

BASE = "https://thredds.met.no/thredds/dodsC/meps25epsarchive/{d:%Y/%m/%d}/"
# meps_det from 2020-02; before it the control member (ensemble_member 0) of meps_subset
URLS = [BASE + "meps_det_2_5km_{d:%Y%m%d}T{h:02d}Z.nc",
        BASE + "meps_subset_2_5km_{d:%Y%m%d}T{h:02d}Z.nc"]
LCC_LON0, LCC_N = 15.0, np.sin(np.deg2rad(63.0))  # MEPS Lambert, tangent at 63N
OUT = Path("/cluster/work/projects/nn8106k/siyan/windcast/meps")
# Northern Norway (Nordland, Troms, Finnmark: the 113 Frost stations span x -374..414,
# y -299..364 km on the windcast.geo plane) + 50 km; contains the 1 km Tromsø–Senja grid.
# About 333 x 378 MEPS cells on both MEPS grids (checked: inside, off the grid edges).
WIN_KM = (-424, 464, -349, 414)
VARS = {  # output name: MEPS variable (any extra level / member dims: index 0)
    "u10": "x_wind_10m", "v10": "y_wind_10m", "fg10": "wind_speed_of_gust",
    "t2m": "air_temperature_2m", "rh2m": "relative_humidity_2m",
    "msl": "air_pressure_at_sea_level", "tcc": "cloud_area_fraction",
    "tp_acc": "precipitation_amount_acc",  # accumulated since init, kg m-2
    "blh": "atmosphere_boundary_layer_thickness",
}


def window(ds) -> tuple[slice, slice]:
    lon, lat = ds["longitude"][:], ds["latitude"][:]
    xy = to_plane(lon, lat)
    m = ((xy[..., 0] >= WIN_KM[0]) & (xy[..., 0] <= WIN_KM[1])
         & (xy[..., 1] >= WIN_KM[2]) & (xy[..., 1] <= WIN_KM[3]))
    iy, ix = np.where(m)
    return slice(iy.min(), iy.max() + 1), slice(ix.min(), ix.max() + 1)


def rotate(ux, uy, lon):
    """Grid-relative (x, y) wind on the MEPS Lambert grid -> earth-relative (east, north)."""
    a = np.deg2rad(LCC_N * (((lon - LCC_LON0) + 180) % 360 - 180))
    return ux * np.cos(a) + uy * np.sin(a), -ux * np.sin(a) + uy * np.cos(a)


def open_run(d: date, h: int):
    err = None
    for u in URLS:
        try:
            return netCDF4.Dataset(u.format(d=d, h=h)), u.format(d=d, h=h)
        except OSError as e:
            err = e
    raise err


def as_slices(leads) -> list:
    """Sorted lead indices -> the fewest evenly spaced slices covering them. OPeNDAP reads
    an irregular index list one element per request, which trips the archive's rate
    limit (HTTP 429); a slice is one request."""
    leads = sorted(leads)
    out, i = [], 0
    while i < len(leads):
        j = i + 1
        if j < len(leads):
            step = leads[j] - leads[i]
            while j + 1 < len(leads) and leads[j + 1] - leads[j] == step:
                j += 1
            out.append(slice(leads[i], leads[j] + 1, step))
            i = j + 1
        else:
            out.append(slice(leads[i], leads[i] + 1))
            i = j
    return out


def read(var, lt, sy, sx):
    """var[lt, (level / member 0), sy, sx] as float32 with NaN for missing; lt is a list of
    lead indices, read as a few strided slices."""
    parts = []
    for s in as_slices(lt):
        idx = tuple(s if n == "time" else sy if n == "y" else sx if n == "x" else 0
                    for n in var.dimensions)
        parts.append(np.ma.filled(var[idx].astype("f4"), np.nan))
    return np.concatenate(parts, axis=0)


def fetch(d: date, h: int, leads, windows: dict, out: Path):
    ds, url = open_run(d, h)
    try:
        shape = (len(ds.dimensions["y"]), len(ds.dimensions["x"]))
        if shape not in windows:  # the MEPS grid changed in 2020
            windows[shape] = window(ds)
            print(f"grid {shape}: window y {windows[shape][0]} x {windows[shape][1]}", flush=True)
        sy, sx = windows[shape]
        n = len(ds["time"][:])
        lt = [l for l in leads if l < n]
        lon, lat = ds["longitude"][sy, sx], ds["latitude"][sy, sx]
        raw = {k: read(ds[v], lt, sy, sx) for k, v in VARS.items()}
        raw["u10"], raw["v10"] = rotate(raw["u10"], raw["v10"], lon)
        data = {k: (("lead", "y", "x"), a.astype("f4")) for k, a in raw.items()}
        ref = float(ds["forecast_reference_time"][:])
    finally:
        ds.close()
    xr.Dataset(
        data,
        coords={"lead": ("lead", np.asarray(lt, "i2")),
                "lon": (("y", "x"), lon.astype("f8")), "lat": (("y", "x"), lat.astype("f8"))},
        attrs={"forecast_reference_time": ref, "source": url},
    ).to_netcdf(out.with_suffix(".tmp"), encoding={k: {"zlib": True, "complevel": 3} for k in VARS})
    out.with_suffix(".tmp").rename(out)


def has_leads(path: Path, leads) -> bool:
    """An existing file counts as done only if it holds every requested lead (or the run
    is shorter: then it holds all the run has, which is not checked here)."""
    try:
        with netCDF4.Dataset(path) as ds:
            have = set(int(x) for x in ds["lead"][:])
        return set(leads) <= have or max(have) < max(leads)
    except OSError:
        return False


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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2018-01-01")
    p.add_argument("--end", default="2025-12-31")
    p.add_argument("--inits", type=int, nargs="+", default=[0, 12])
    # hourly 0-12 h (obs-forecast: past runs vs hourly station obs), then the 6-hourly
    # leads shared with the global ML models (fusion), to MEPS's +66 h
    p.add_argument("--leads", type=int, nargs="+", default=list(range(0, 13)) + list(range(18, 67, 6)))
    p.add_argument("--out", type=Path, default=OUT)
    a = p.parse_args()
    d0, d1 = date.fromisoformat(a.start), date.fromisoformat(a.end)
    windows = {}
    d, n_ok, n_fail = d0, 0, 0
    while d <= d1:
        for h in a.inits:
            out = a.out / f"{d:%Y}" / f"meps_{d:%Y%m%d}T{h:02d}.nc"
            if out.exists() and has_leads(out, a.leads):
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            for attempt in range(4):
                try:
                    t0 = time.time()
                    if not windows:  # both grids' windows, computed once in the parent
                        for g in (date(2018, 1, 1), date(2023, 1, 1)):
                            ds, _ = open_run(g, 0)
                            windows[(ds.dimensions["y"].size, ds.dimensions["x"].size)] = window(ds)
                            ds.close()
                    with_timeout(fetch, (d, h, a.leads, windows, out))
                    n_ok += 1
                    time.sleep(1.0)  # stay under the archive's rate limit
                    print(f"{d} {h:02d}Z ok {time.time() - t0:.0f}s", flush=True)
                    break
                except Exception as e:  # missing runs, server hiccups
                    msg = str(e).splitlines()[0][:120]
                    if attempt == 3 or "NetCDF: file not found" in msg or "404" in msg:
                        n_fail += 1
                        print(f"{d} {h:02d}Z FAIL {msg}", flush=True)
                        with open(a.out / "missing.txt", "a") as f:
                            f.write(f"{d:%Y%m%d}T{h:02d} {msg}\n")
                        break
                    # 429: the server rate-limits this client; back off for minutes
                    time.sleep((120 if "429" in msg or "Access failure" in msg else 30) * (attempt + 1))
        d += timedelta(days=1)
    print(f"done: {n_ok} ok, {n_fail} failed", flush=True)


if __name__ == "__main__":
    main()
