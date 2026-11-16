"""Build the 1 km reference grid file used by every experiment.

    python scripts/build_grid.py

OUT/grid.npz:
    lat, lon, xy            (H, W[, 2]) MET Nordic 1 km cells (314 x 343), xy in km (windcast.geo)
    altitude, land          (H, W) MET Nordic static fields
    terrain, terrain_names  (16, H, W) terrain box-mean to the 1 km cells. Its source zarr has
                            been purged from /cluster/work, so it is taken from the mtno E1 cache
                            (same cells; lat / lon checked to be identical)
    meps_<ny>x<nx>_idx / _w (H, W, 4) bilinear interpolation from the MEPS download window of
                            that MEPS grid (regular in its Lambert projection), flat indices into
                            the (y, x) window
    meps_<ny>x<nx>_oro      (H, W) MEPS orography interpolated to the 1 km cells (m)
"""
from pathlib import Path
import sys

import numpy as np
import xarray as xr
from pyproj import Transformer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from windcast.geo import to_plane  # noqa: E402

ROOT = Path("/cluster/work/projects/nn8106k/siyan/windcast")
MTNO_GRID = Path("/cluster/work/projects/nn8106k/siyan/mtno/cache/e1_tromso_senja/grid.npz")


# MEPS Lambert projection per grid (from the files' projection_lambert.proj4)
MEPS_LCC = {
    "949x889": "+proj=lcc +lat_0=63 +lon_0=15 +lat_1=63 +lat_2=63 +no_defs +R=6.371e+06",
    "1069x949": "+proj=lcc +lat_0=63.3 +lon_0=15 +lat_1=63.3 +lat_2=63.3 +no_defs +R=6.371e+06",
}


def bilinear(proj, src_lon, src_lat, lon, lat):
    """Bilinear weights from the MEPS window (ny, nx), regular in MEPS_LCC, to points
    (lon, lat) (M,): flat indices (M, 4) and weights (M, 4)."""
    tr = Transformer.from_crs("+proj=longlat +R=6371000 +no_defs", proj, always_xy=True)
    sx, sy = tr.transform(src_lon, src_lat)
    x0, dx = sx[0, 0], sx[0, 1] - sx[0, 0]
    y0, dy = sy[0, 0], sy[1, 0] - sy[0, 0]
    assert np.allclose(sx, x0 + dx * np.arange(sx.shape[1])[None], atol=1.0), "not regular in x"
    assert np.allclose(sy, y0 + dy * np.arange(sy.shape[0])[:, None], atol=1.0), "not regular in y"
    px, py = tr.transform(lon, lat)
    fi, fj = (py - y0) / dy, (px - x0) / dx
    i, j = np.floor(fi).astype(int), np.floor(fj).astype(int)
    if (i < 0).any() or (j < 0).any() or (i + 1 >= sx.shape[0]).any() or (j + 1 >= sx.shape[1]).any():
        raise ValueError("target cells outside the MEPS window")
    a, b = fi - i, fj - j
    nx = sx.shape[1]
    idx = np.stack([i * nx + j, i * nx + j + 1, (i + 1) * nx + j, (i + 1) * nx + j + 1], -1)
    w = np.stack([(1 - a) * (1 - b), (1 - a) * b, a * (1 - b), a * b], -1)
    return idx.astype(np.int32), w.astype(np.float32)


def main():
    st = xr.open_dataset(ROOT / "metnordic" / "static.nc")
    lat, lon = st.latitude.values, st.longitude.values
    H, W = lat.shape
    xy = to_plane(lon, lat).astype(np.float32)
    old = np.load(MTNO_GRID)
    assert np.array_equal(old["lat"], lat) and np.array_equal(old["lon"], lon), "grid mismatch"
    out = {"lat": lat, "lon": lon, "xy": xy,
           "altitude": st.altitude.values.astype(np.float32),
           "land": st.land_area_fraction.values.astype(np.float32),
           "terrain": old["terrain"].astype(np.float32), "terrain_names": old["terrain_names"]}
    for f in sorted((ROOT / "meps").glob("static_*.nc")):
        m = xr.open_dataset(f)
        key = "meps_" + f.stem.split("_", 1)[1]
        idx, w = bilinear(MEPS_LCC[f.stem.split("_", 1)[1]], m.lon.values, m.lat.values, lon.reshape(-1), lat.reshape(-1))
        out[key + "_idx"] = idx.reshape(H, W, 4)
        out[key + "_w"] = w.reshape(H, W, 4)
        out[key + "_oro"] = (m.orography.values.reshape(-1)[idx] * w).sum(-1).reshape(H, W)
        d = out[key + "_oro"] - out["altitude"]
        print(f"{key}: window {m.sizes['y']} x {m.sizes['x']}, "
              f"orography - MET Nordic altitude: mean {d.mean():+.1f} m, sd {d.std():.1f} m")
    np.savez(ROOT / "grid.npz", **out)
    print(ROOT / "grid.npz", H, W, list(out))


if __name__ == "__main__":
    main()
