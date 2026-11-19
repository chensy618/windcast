"""The 28 terrain descriptors of scripts/build_terrain_descriptors.py on a 1 km grid over all of
mainland Norway, computed here from the elevation itself (track A, all of Norway).

    python scripts/build_terrain_norway.py          # CPU node: about 60 GB, internet for Copernicus

Why a new build: the northern descriptors come from the mtno "nno" cache, whose 16 terrain
products are 1 km box means of a 200 m product (WeatherFNO) that covers only a 1024 km box around
the north. Here the same definitions are applied to the Kartverket 50 m DTM (scripts/download_dtm.py;
Copernicus GLO-90 where Kartverket has no land: sea, Sweden, Finland, Russia).

Grid: regular 1 km on the windcast.geo plane, x -926 .. 464 km, y -1088 .. 416 km (the MEPS
"norway" window of scripts/download_meps.py), rows going north; 1504 x 1390 cells.
200 m fields (the resolution of the product behind the northern descriptors), then 5 x 5 block
means onto the 1 km cells:
    elevation, slope (rad), aspect_sin / aspect_cos (downslope bearing; 0 on flat cells),
    tpi_1km / tpi_5km (elevation minus its mean over a 1 / 5 km square window), std_1km
    (elevation std over a 1 km square window), land_fraction (share of 200 m cells with
    elevation > 0), sx_<dir> (steepest rise toward <dir> over the next 5 cells, i.e. 1 km along
    the axes, rad; positive = sheltered from winds from <dir>)
The window shapes and the shelter steps are the ones that reproduce the northern descriptors
best (checked on their 728 284 cells: square windows match their mean and spread; the shelter
takes the same number of cells in every direction, also diagonally).
Then on the 1 km grid, exactly as in build_terrain_descriptors.py:
    elev_rel_10km / elev_rel_30km, land_10km / land_30km, sx10_<dir>

Writes
    ROOT/static/terrain_descriptors_1km_norway.npz  fields (F, H, W), names, lat, lon, xy
    ROOT/cache/station_terrain_norway.npz           at the nearest cell of each station of
                                                    ROOT/stations/frost_no/stations.parquet
and prints, for the northern overlap, the correlation of every descriptor with the mtno-based one.
"""
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif")

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from rasterio.merge import merge
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject, transform_bounds
from scipy.ndimage import gaussian_filter, uniform_filter
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from windcast.geo import PROJ4, to_plane  # noqa: E402

ROOT = Path("/cluster/work/projects/nn8106k/siyan/windcast")
DTM = ROOT / "static" / "dtm50"
COP = ("/vsicurl/https://copernicus-dem-90m.s3.amazonaws.com/Copernicus_DSM_COG_30_{ns}{a:02d}_00_{ew}{o:03d}_00_DEM/"
       "Copernicus_DSM_COG_30_{ns}{a:02d}_00_{ew}{o:03d}_00_DEM.tif")
X0, X1, Y0, Y1 = -926.0, 464.0, -1088.0, 416.0      # km, 1 km cell edges
FINE = 0.2                                          # km
B = int(round(1.0 / FINE))                          # 200 m cells per 1 km cell (per side)
DIRS = ("n", "ne", "e", "se", "s", "sw", "w", "nw")
STEPS = ((1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1))   # (row, col); rows north


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def elevation_200m():
    """(H, W) float32 on the 200 m grid, rows north -> south (raster order)."""
    W, H = int(round((X1 - X0) / FINE)), int(round((Y1 - Y0) / FINE))
    dst_t = from_origin(X0 * 1000, Y1 * 1000, FINE * 1000, FINE * 1000)
    srcs = [rasterio.open(f) for f in sorted(DTM.glob("dtm50_*.tif"))]
    mos, mos_t = merge(srcs, nodata=-9999.0)
    crs = srcs[0].crs
    for s in srcs:
        s.close()
    log("DTM mosaic", mos.shape)
    kv = np.full((H, W), np.nan, np.float32)
    reproject(mos[0].astype(np.float32), kv, src_transform=mos_t, src_crs=crs, src_nodata=-9999.0,
              dst_transform=dst_t, dst_crs=PROJ4, dst_nodata=np.nan, resampling=Resampling.average)
    del mos
    log("DTM on the plane; Kartverket land share", float(np.mean(kv > 0)))
    # Copernicus GLO-90, tile by tile into its own window of the destination
    inv = Transformer.from_crs(PROJ4, "EPSG:4326", always_xy=True)
    xs = np.linspace(X0, X1, 60) * 1000
    ys = np.linspace(Y0, Y1, 60) * 1000
    gx, gy = np.meshgrid(xs, ys)
    lon, lat = inv.transform(gx, gy)
    cop = np.zeros((H, W), np.float32)
    n = 0
    for a in range(int(np.floor(lat.min())), int(np.floor(lat.max())) + 1):
        for o in range(int(np.floor(lon.min())), int(np.floor(lon.max())) + 1):
            url = COP.format(ns="N" if a >= 0 else "S", a=abs(a), ew="E" if o >= 0 else "W", o=abs(o))
            try:
                src = rasterio.open(url)
            except rasterio.errors.RasterioIOError:      # sea-only tiles do not exist
                continue
            with src:
                l, b, r, t = transform_bounds(src.crs, PROJ4, *src.bounds)
                c0 = max(int(np.floor((l / 1000 - X0) / FINE)) - 2, 0)
                c1 = min(int(np.ceil((r / 1000 - X0) / FINE)) + 2, W)
                r0 = max(int(np.floor((Y1 - t / 1000) / FINE)) - 2, 0)
                r1 = min(int(np.ceil((Y1 - b / 1000) / FINE)) + 2, H)
                if c1 <= c0 or r1 <= r0:
                    continue
                tmp = np.full((r1 - r0, c1 - c0), np.nan, np.float32)
                reproject(rasterio.band(src, 1), tmp,
                          dst_transform=from_origin((X0 + c0 * FINE) * 1000, (Y1 - r0 * FINE) * 1000, FINE * 1000, FINE * 1000),
                          dst_crs=PROJ4, dst_nodata=np.nan, resampling=Resampling.average)
                ok = np.isfinite(tmp)
                sub = cop[r0:r1, c0:c1]
                sub[ok] = tmp[ok]
                n += 1
    log("Copernicus tiles", n)
    return np.where(np.nan_to_num(kv, nan=0.0) > 0, kv, np.maximum(cop, 0.0)).astype(np.float32)


def box_mean(a, km):
    """Mean over a square window of side `km` (200 m cells), edges extended."""
    return uniform_filter(a.astype(np.float64), int(round(km / FINE)), mode="nearest").astype(np.float32)


def shelter(e, dx_km, radius_km):
    """Steepest rise over the next radius_km / dx_km cells toward each of the 8 directions (the
    same number of cells diagonally, as in build_terrain_descriptors.py), as an angle (rad); rows north."""
    k_max = int(round(radius_km / dx_km))
    pad = np.pad(e, k_max, mode="edge")
    ny, nx = e.shape
    out = []
    for dy, dx in STEPS:
        step = np.hypot(dy, dx) * dx_km * 1000.0
        best = np.full(e.shape, -np.pi / 2, np.float32)
        for k in range(1, k_max + 1):
            nb = pad[k_max + k * dy: k_max + k * dy + ny, k_max + k * dx: k_max + k * dx + nx]
            np.maximum(best, np.arctan((nb - e) / (k * step)), out=best)
        out.append(best)
    return out


def block(a):
    """200 m (rows north) -> 1 km block means."""
    H, W = a.shape
    return a.reshape(H // B, B, W // B, B).mean((1, 3), dtype=np.float64).astype(np.float32)


def main():
    e = elevation_200m()[::-1].copy()                  # rows north
    log("200 m grid", e.shape)
    names, fields = [], []

    def add(n, a200):
        names.append(n)
        fields.append(block(a200))

    add("elevation", e)
    gy, gx = np.gradient(e, FINE * 1000.0)              # rows north: gy is d/dnorth, gx d/deast
    add("slope", np.arctan(np.hypot(gx, gy)))
    bearing = np.arctan2(-gx, -gy)                       # downslope direction, clockwise from north
    flat = np.hypot(gx, gy) < 1e-6
    add("aspect_sin", np.where(flat, 0.0, np.sin(bearing)).astype(np.float32))
    add("aspect_cos", np.where(flat, 0.0, np.cos(bearing)).astype(np.float32))
    del gy, gx, bearing, flat
    m1 = box_mean(e, 1.0)
    add("tpi_1km", e - m1)
    add("tpi_5km", e - box_mean(e, 5.0))
    add("std_1km", np.sqrt(np.maximum(box_mean(e * e, 1.0) - m1 * m1, 0.0)))
    del m1
    add("land_fraction", (e > 0).astype(np.float32))
    log("slope / aspect / tpi / std / land done")
    for d, f in zip(DIRS, shelter(e, FINE, 1.0)):
        add(f"sx_{d}", f)
    log("sx (1 km) done")
    del e

    # on the 1 km grid, as in build_terrain_descriptors.py
    elev, land = fields[names.index("elevation")], fields[names.index("land_fraction")]
    for s, tag in ((5, "10km"), (12, "30km")):
        fields.append(elev - gaussian_filter(elev, s, mode="nearest"))
        names.append(f"elev_rel_{tag}")
    for s, tag in ((5, "10km"), (12, "30km")):
        fields.append(gaussian_filter(land, s, mode="nearest"))
        names.append(f"land_{tag}")
    fields += shelter(elev, 1.0, 10.0)
    names += [f"sx10_{d}" for d in DIRS]
    F = np.stack(fields).astype(np.float32)

    H1, W1 = elev.shape
    xc = X0 + np.arange(W1) + 0.5
    yc = Y0 + np.arange(H1) + 0.5
    gxk, gyk = np.meshgrid(xc, yc)
    xy = np.stack([gxk, gyk], -1)
    lon, lat = Transformer.from_crs(PROJ4, "EPSG:4326", always_xy=True).transform(gxk * 1000, gyk * 1000)
    out = ROOT / "static" / "terrain_descriptors_1km_norway.npz"
    np.savez_compressed(out, fields=F, names=np.array(names), lat=lat.astype(np.float32), lon=lon.astype(np.float32),
                        xy=xy.astype(np.float32))
    log(out, F.shape)

    # stations
    st = pd.read_parquet(ROOT / "stations" / "frost_no" / "stations.parquet")
    p = to_plane(st.lon.values, st.lat.values)
    dist, k = cKDTree(xy.reshape(-1, 2)).query(p)
    i, j = np.unravel_index(k, xy.shape[:2])
    np.savez_compressed(ROOT / "cache" / "station_terrain_norway.npz", station=st.station.values, names=np.array(names),
                        values=F[:, i, j].T, ij=np.stack([i, j], -1), dist_km=dist)
    log(f"{len(st)} stations, max distance to the cell centre {dist.max():.2f} km")

    # check against the mtno-based northern descriptors on their cells
    old = np.load(ROOT / "static" / "terrain_descriptors_1km.npz", allow_pickle=True)
    oxy = old["xy"].reshape(-1, 2)
    d, kk = cKDTree(xy.reshape(-1, 2)).query(oxy)
    ok = d < 0.5
    on = [str(x) for x in old["names"]]
    print("descriptor        corr   mean(new)  mean(old)   (northern overlap, %d cells)" % ok.sum())
    for n in names:
        a = F[names.index(n)].reshape(-1)[kk[ok]]
        b = old["fields"][on.index(n)].reshape(-1)[ok]
        print(f"  {n:14s} {np.corrcoef(a, b)[0, 1]:6.3f} {a.mean():10.3f} {b.mean():10.3f}")


if __name__ == "__main__":
    main()
