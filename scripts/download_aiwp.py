"""Download machine-learning weather prediction forecasts from the NOAA AI Weather Prediction
(AIWP) archive on AWS (noaa-oar-mlwp-data, CIRA / NOAA-GSL; no use restrictions), cut to the
0.25 deg box of mainland Norway (57.5-74 N, 4-36 E), for the MLWP / NWP comparison (track A).

    python scripts/download_aiwp.py --models graphcast pangu fourcastnet --start 2024-02-28 --end 2025-09-16

Models (all initialised from the IFS analysis, the same in real time: the files of a 00 UTC run
appear about 8.5 h later):
    graphcast    GRAP_v100_IFS   GraphCast Operational (13 levels)
    pangu        PANG_v100_IFS   Pangu-Weather
    fourcastnet  FOUR_v200_IFS   FourCastNet v2-small
    aurora       AURO_v100_IFS   Aurora (from 2025-01 only)
Each archive file is one run, 0-240 h every 6 h, global 0.25 deg, chunked per (time, level), so
only the needed chunks are read (netCDF byte-range access): 0-66 h of u10, v10, t2, msl and t
at 850 hPa. One output file per run, in the layout of scripts/download_ecmwf.py:
OUT/<model>/YYYY/<model>_YYYYMMDDTHH.nc with (lead, lat, lon) fields 10u, 10v, 2t, msl, t850 on
lat 74 .. 57.5 N (descending) x lon 4 .. 36 E. Existing files are skipped.
"""
import argparse
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path

import netCDF4
import numpy as np

URL = "https://noaa-oar-mlwp-data.s3.amazonaws.com/{code}/{d:%Y}/{d:%m%d}/{code}_{d:%Y%m%d}{h:02d}_f000_f240_06.nc"
CODES = {"graphcast": "GRAP_v100_IFS", "pangu": "PANG_v100_IFS", "fourcastnet": "FOUR_v200_IFS",
         "aurora": "AURO_v100_IFS"}
OUT = Path("/cluster/work/projects/nn8106k/siyan/windcast")
LEADS = list(range(0, 67, 6))
LAT, LON = (57.5, 74.0), (4.0, 36.0)   # mainland Norway (57.9-71.2 N) + margin
FIELDS = {"10u": ("u10", None, "m s**-1"), "10v": ("v10", None, "m s**-1"), "2t": ("t2", None, "K"),
          "msl": ("msl", None, "Pa"), "t850": ("t", 850, "K")}


def fetch(model, d, h, out):
    url = URL.format(code=CODES[model], d=d, h=h) + "#mode=bytes"
    t0 = time.time()
    with netCDF4.Dataset(url) as ds:
        lat, lon = ds["latitude"][:], ds["longitude"][:]
        iy = np.where((lat >= LAT[0] - 1e-6) & (lat <= LAT[1] + 1e-6))[0]
        ix = np.where((lon >= LON[0] - 1e-6) & (lon <= LON[1] + 1e-6))[0]
        assert lat[iy[0]] > lat[iy[-1]], "latitude must descend"
        hrs = np.rint((ds["time"][:] - ds["time"][0]) / 3600.0 if ds["time"].units.startswith("seconds")
                      else (ds["time"][:] - ds["time"][0]) * (1 if "hour" in ds["time"].units else 24)).astype(int)
        jt = [int(np.where(hrs == L)[0][0]) for L in LEADS]
        data = {}
        for k, (v, lev, _) in FIELDS.items():
            var = ds[v]
            if lev is None:
                a = var[jt[0]:jt[-1] + 1, iy[0]:iy[-1] + 1, ix[0]:ix[-1] + 1]
            else:
                kl = int(np.where(ds["level"][:] == lev)[0][0])
                a = var[jt[0]:jt[-1] + 1, kl, iy[0]:iy[-1] + 1, ix[0]:ix[-1] + 1]
            data[k] = np.ma.filled(a.astype(np.float32), np.nan)
        glat, glon = lat[iy].astype(float), lon[ix].astype(float)
    tmp = out.with_suffix(".tmp")
    with netCDF4.Dataset(tmp, "w") as o:
        o.model, o.product, o.init, o.regridded = model, CODES[model], f"{d:%Y-%m-%d}T{h:02d}", 0
        o.createDimension("lead", len(LEADS))
        o.createDimension("lat", len(glat))
        o.createDimension("lon", len(glon))
        o.createVariable("lead", "i2", ("lead",))[:] = LEADS
        o.createVariable("lat", "f8", ("lat",))[:] = glat
        o.createVariable("lon", "f8", ("lon",))[:] = glon
        for k, a in data.items():
            v = o.createVariable(k, "f4", ("lead", "lat", "lon"), zlib=True, complevel=3)
            v.units = FIELDS[k][2]
            v[:] = a
    tmp.rename(out)
    return time.time() - t0


def job(args):
    model, d, h, out = args
    for attempt in range(4):
        try:
            return f"{model} {d} {h:02d}Z ok {fetch(model, d, h, out):.0f}s"
        except Exception as e:
            msg = str(e).splitlines()[0][:120]
            if "404" in msg or "file not found" in msg.lower() or attempt == 3:
                return f"{model} {d} {h:02d}Z FAIL {msg}"
            time.sleep(30 * (attempt + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["graphcast", "pangu", "fourcastnet"], choices=list(CODES))
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--inits", type=int, nargs="+", default=[0, 12])
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    d0, d1 = date.fromisoformat(a.start), date.fromisoformat(a.end)
    jobs = []
    d = d0
    while d <= d1:
        for h in a.inits:
            for m in a.models:
                out = OUT / m / f"{d:%Y}" / f"{m}_{d:%Y%m%d}T{h:02d}.nc"
                if not out.exists():
                    out.parent.mkdir(parents=True, exist_ok=True)
                    jobs.append((m, d, h, out))
        d += timedelta(days=1)
    print(f"{len(jobs)} runs to fetch", flush=True)
    n_ok = n_fail = 0
    with ProcessPoolExecutor(a.workers) as pool:
        for f in as_completed([pool.submit(job, j) for j in jobs]):
            msg = f.result()
            n_ok += " ok " in msg
            n_fail += " FAIL " in msg
            print(msg, flush=True)
    print(f"done: {n_ok} ok, {n_fail} failed")


if __name__ == "__main__":
    main()
