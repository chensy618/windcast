"""MEPS static fields on the download window, one file per MEPS grid (the grid changed in
2020-02): OUT/static_<ny>x<nx>.nc with lon, lat, orography (m) and land fraction (NaN before 2020-02).

    python scripts/meps_static.py
"""
import sys
from datetime import date
from pathlib import Path

import netCDF4
import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from download_meps import OUT, open_run, read, window  # noqa: E402

G = 9.80665
RUNS = [date(2018, 1, 1), date(2023, 1, 1)]   # one run per grid

for d in RUNS:
    ds, url = open_run(d, 0)
    try:
        shape = (len(ds.dimensions["y"]), len(ds.dimensions["x"]))
        sy, sx = window(ds)
        lon, lat = ds["longitude"][sy, sx], ds["latitude"][sy, sx]
        oro = read(ds["surface_geopotential"], [0], sy, sx)[0] / G
        lsm = (read(ds["land_area_fraction"], [0], sy, sx)[0] if "land_area_fraction" in ds.variables
               else np.full(oro.shape, np.nan, "f4"))   # not in meps_subset (before 2020-02)
    finally:
        ds.close()
    out = OUT / f"static_{shape[0]}x{shape[1]}.nc"
    xr.Dataset({"lon": (("y", "x"), np.asarray(lon, "f8")), "lat": (("y", "x"), np.asarray(lat, "f8")),
                "orography": (("y", "x"), oro), "land_fraction": (("y", "x"), lsm)},
               attrs={"source": url, "window_y": f"{sy.start}:{sy.stop}", "window_x": f"{sx.start}:{sx.stop}"}
               ).to_netcdf(out)
    print(out, shape, f"oro {np.nanmin(oro):.0f}..{np.nanmax(oro):.0f} m, land {np.nanmean(lsm):.2f}")
