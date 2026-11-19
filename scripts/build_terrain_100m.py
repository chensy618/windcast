"""100 m terrain on the windcast plane, and from it (a) 1 km-cell statistics on the 1 km grid of
scripts/build_terrain_descriptors.py and (b) station-point values (track A, A2).

    python scripts/build_terrain_100m.py              # CPU node: about 120 GB, internet for Copernicus
    python scripts/build_terrain_100m.py --res 300    # the same at 300 m (radii 900 m / 3 km), tag c300 / p300

At --res R (m) every name and radius below scales with R: R = 100 gives the fields listed, R = 300
gives slope300, tpi900, std900, sx900_<dir>, sx3k_<dir> and the files terrain_c300_1km.npz,
station_terrain_c300.npz, station_terrain_p300.npz (h300 = elevation).

Elevation: the Kartverket 50 m DTM (bare earth, scripts/download_dtm.py) over Norway; where it
has no land (<= 0 m: sea, and outside Norway, where the WCS returns 0 / -1), the Copernicus
GLO-90 DEM (public COGs on AWS) if that is above 0 m, else 0 (sea). Both are averaged / resampled
to a regular 100 m grid on the windcast LAEA plane (windcast.geo), rows going north, covering the
1 km grid plus 2 km.

100 m fields:
    slope100            slope (rad)
    tpi300              elevation minus its mean within 300 m (disk)
    std300              elevation std within 300 m
    sx300_<dir>         steepest rise within 300 m toward <dir> (rad; positive = sheltered
    sx1k_<dir>          from winds from <dir>); 1 km; 100 m steps, plane directions

(a) Each 100 m cell belongs to the nearest 1 km cell centre (the 1 km grid is slightly rotated
on the plane). Per 1 km cell: the mean of every 100 m field (c100_<name>), the std of tpi300
(c100_tpi300_std: sub-cell relief) and of elevation (c100_elev_std).
    ROOT/static/terrain_c100_1km.npz       fields (F, H, W), names
    ROOT/cache/station_terrain_c100.npz    station, names, values (S, F) at the station's 1 km cell
(b) The 100 m fields at the station point (the station's 100 m cell), with h100 = elevation:
    ROOT/cache/station_terrain_p100.npz    station, names, values (S, F)
Stations: the base set (multimodel_2024-03-01_2025-09-12.npz).
"""
import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")   # no directory listing per remote COG
os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif")

import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.warp import Resampling, reproject
from scipy.ndimage import convolve
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from windcast.geo import PROJ4, to_plane  # noqa: E402

ROOT = Path("/cluster/work/projects/nn8106k/siyan/windcast")
DTM = ROOT / "static" / "dtm50"
COP = ("/vsicurl/https://copernicus-dem-90m.s3.amazonaws.com/Copernicus_DSM_COG_30_N{a:02d}_00_E{o:03d}_00_DEM/"
       "Copernicus_DSM_COG_30_N{a:02d}_00_E{o:03d}_00_DEM.tif")
DIRS = ("n", "ne", "e", "se", "s", "sw", "w", "nw")
STEPS = ((1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1))   # (row, col); rows north
DX = 100.0          # grid spacing (m), set by --res


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def disk(r_m):
    k = int(np.ceil(r_m / DX))
    yy, xx = np.mgrid[-k:k + 1, -k:k + 1]
    m = (np.hypot(yy, xx) * DX <= r_m).astype(np.float32)
    return m / m.sum()


def elevation(x0, y1, W, H, lat_rng, lon_rng):
    """(H, W) float32, rows north -> south (raster order), on the LAEA 100 m grid with top-left (x0, y1) m."""
    from rasterio.transform import from_origin
    dst_t = from_origin(x0, y1, DX, DX)
    # Kartverket: mosaic at 50 m (UTM 33N), then average onto the plane
    files = sorted(DTM.glob("dtm50_*.tif"))
    srcs = [rasterio.open(f) for f in files]
    mos, mos_t = merge(srcs, nodata=-9999.0)
    crs = srcs[0].crs
    for s in srcs:
        s.close()
    log("DTM mosaic", mos.shape)
    kv = np.full((H, W), np.nan, np.float32)
    reproject(mos[0].astype(np.float32), kv, src_transform=mos_t, src_crs=crs, src_nodata=-9999.0,
              dst_transform=dst_t, dst_crs=PROJ4, dst_nodata=np.nan, resampling=Resampling.average)
    del mos
    log("DTM on the plane, land (> 0 m):", float(np.mean(kv > 0)))
    # Copernicus GLO-90 where Kartverket has no land
    cop = np.zeros((H, W), np.float32)
    n = 0
    for a in range(lat_rng[0], lat_rng[1] + 1):
        for o in range(lon_rng[0], lon_rng[1] + 1):
            try:
                with rasterio.open(COP.format(a=a, o=o)) as src:
                    tmp = np.full((H, W), np.nan, np.float32)
                    reproject(rasterio.band(src, 1), tmp, dst_transform=dst_t, dst_crs=PROJ4,
                              dst_nodata=np.nan, resampling=Resampling.average)
                    ok = np.isfinite(tmp)
                    cop[ok] = tmp[ok]
                    n += 1
            except rasterio.errors.RasterioIOError:      # sea-only tiles do not exist
                pass
    log("Copernicus tiles", n)
    e = np.where(np.nan_to_num(kv, nan=0.0) > 0, kv, np.maximum(cop, 0.0)).astype(np.float32)
    return e


def shelter(e, r_m):
    """steepest rise within r_m toward each direction, at every cell (rows north)"""
    out = []
    pad_k = int(np.ceil(r_m / DX))
    pad = np.pad(e, pad_k, mode="edge")
    ny, nx = e.shape
    for dy, dx in STEPS:
        step = np.hypot(dy, dx) * DX
        best = np.full(e.shape, -np.pi / 2, np.float32)
        for k in range(1, int(r_m / step) + 1):
            nb = pad[pad_k + k * dy: pad_k + k * dy + ny, pad_k + k * dx: pad_k + k * dx + nx]
            np.maximum(best, np.arctan((nb - e) / (k * step)), out=best)
        out.append(best)
    return out


def rname(r_m):
    return f"{r_m // 1000}k" if r_m % 1000 == 0 else str(r_m)


def main():
    global DX
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=int, default=100, help="grid spacing (m)")
    res = ap.parse_args().res
    DX = float(res)
    r1, r2 = 3 * res, 10 * res            # 300 m / 1 km at 100 m
    tag = str(res)
    g = np.load(ROOT / "static" / "terrain_descriptors_1km.npz", allow_pickle=True)
    xy, lat, lon = g["xy"].astype(np.float64), g["lat"], g["lon"]
    H1, W1 = lat.shape
    x0, x1 = xy[..., 0].min() - 2.0, xy[..., 0].max() + 2.0
    y0, y1 = xy[..., 1].min() - 2.0, xy[..., 1].max() + 2.0
    kd = DX / 1000.0                      # spacing (km)
    W, H = int(np.ceil((x1 - x0) / kd)), int(np.ceil((y1 - y0) / kd))
    log(f"{res} m grid {H} x {W}")
    lat_rng = (int(np.floor(lat.min())) - 1, int(np.floor(lat.max())) + 1)
    lon_rng = (int(np.floor(lon.min())) - 1, int(np.floor(lon.max())) + 1)
    e = elevation(x0 * 1000, y1 * 1000, W, H, lat_rng, lon_rng)[::-1].copy()    # rows north
    # 100 m cell centres on the plane (km), rows north
    xc = x0 + (np.arange(W) + 0.5) * kd
    yc = (y1 - H * kd) + (np.arange(H) + 0.5) * kd

    gy, gx = np.gradient(e, DX)
    fields = {f"slope{tag}": np.arctan(np.hypot(gx, gy)).astype(np.float32)}
    del gy, gx
    m300 = convolve(e, disk(r1), mode="nearest")
    fields[f"tpi{rname(r1)}"] = (e - m300).astype(np.float32)
    fields[f"std{rname(r1)}"] = np.sqrt(np.maximum(convolve(e * e, disk(r1), mode="nearest") - m300 * m300, 0)).astype(np.float32)
    del m300
    log("slope / tpi / std done")
    for r_m, pre in ((r1, f"sx{rname(r1)}"), (r2, f"sx{rname(r2)}")):
        for d, f in zip(DIRS, shelter(e, r_m)):
            fields[f"{pre}_{d}"] = f
        log(pre, "done")

    # (a) 1 km-cell statistics: each 100 m cell -> nearest 1 km cell centre
    tree = cKDTree(xy.reshape(-1, 2))
    lab = np.empty((H, W), np.int64)
    dist = np.empty((H, W), np.float32)
    for i in range(H):
        dd, ii = tree.query(np.stack([xc, np.full(W, yc[i])], -1))
        lab[i], dist[i] = ii, dd
    inside = dist < 0.75                      # drop the margin beyond the grid edge
    L = lab[inside]
    cnt = np.bincount(L, minlength=H1 * W1).astype(np.float64)
    log("cells per 1 km cell: median", float(np.median(cnt[cnt > 0])))

    def cmean(a):
        return (np.bincount(L, a[inside].astype(np.float64), minlength=H1 * W1) / np.maximum(cnt, 1)).reshape(H1, W1)

    def cstd(a):
        m = cmean(a).reshape(-1)
        m2 = np.bincount(L, a[inside].astype(np.float64) ** 2, minlength=H1 * W1) / np.maximum(cnt, 1)
        return np.sqrt(np.maximum(m2 - m * m, 0)).reshape(H1, W1)
    names, out = [], []
    for k, a in fields.items():
        names.append(f"c{tag}_{k}")
        out.append(cmean(a))
    names += [f"c{tag}_tpi{rname(r1)}_std", f"c{tag}_elev_std"]
    out += [cstd(fields[f"tpi{rname(r1)}"]), cstd(e)]
    F = np.stack(out).astype(np.float32)
    np.savez_compressed(ROOT / "static" / f"terrain_c{tag}_1km.npz", fields=F, names=np.array(names))
    log("1 km statistics", F.shape)

    # stations: their 1 km cell (a) and their 100 m cell (b)
    st = np.load(ROOT / "cache" / "multimodel_2024-03-01_2025-09-12.npz", allow_pickle=True)
    t = np.load(ROOT / "cache" / "station_terrain.npz", allow_pickle=True)
    ij = t["ij"]
    vals_c = F[:, ij[:, 0], ij[:, 1]].T
    np.savez_compressed(ROOT / "cache" / f"station_terrain_c{tag}.npz", station=st["station"], names=np.array(names),
                        values=vals_c)
    p = to_plane(st["lon_st"], st["lat_st"])
    jc = np.clip(((p[:, 0] - x0) / kd).astype(int), 0, W - 1)
    ic = np.clip(((p[:, 1] - (y1 - H * kd)) / kd).astype(int), 0, H - 1)
    pnames = [f"h{tag}"] + list(fields)
    vals_p = np.stack([e[ic, jc]] + [fields[k][ic, jc] for k in fields], -1)
    np.savez_compressed(ROOT / "cache" / f"station_terrain_p{tag}.npz", station=st["station"], names=np.array(pnames),
                        values=vals_p.astype(np.float32))
    print(f"h{tag} vs station masl" + ": corr %.3f, median |diff| %.1f m" % (
        np.corrcoef(vals_p[:, 0], st["masl"].astype(float))[0, 1], np.median(np.abs(vals_p[:, 0] - st["masl"].astype(float)))))
    if res == 100:
        old = np.load(ROOT / "cache" / "station_terrain_100m.npz", allow_pickle=True)
        on = [str(x) for x in old["names"]]
        for k in ("h100", "tpi300", "std300", "sx1k_w"):
            print(f"  {k}: corr with the mtno-patch version %.3f" % np.corrcoef(vals_p[:, pnames.index(k)], old["values"][:, on.index(k)])[0, 1])
    for n_, v in zip(names, vals_c.T):
        print(f"  {n_:18s} {v.min():9.3f} {np.median(v):9.3f} {v.max():9.3f}")


if __name__ == "__main__":
    main()
