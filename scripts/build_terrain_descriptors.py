"""Terrain and surface descriptors on the 1 km grid of northern Norway, and their values at the
stations (track A, A2).

    python scripts/build_terrain_descriptors.py
    python scripts/build_terrain_descriptors.py --stations ext|all   # station values only, for
                       # an extended set (multimodel_..._<tag>.npz) -> station_terrain_<tag>.npz

Every descriptor is computed from fields that exist at every 1 km cell, so a model that uses
them can be applied anywhere on the grid, not only at stations.

Source: the 1 km grid of the mtno "nno" cache (1033 x 964 cells, 62.6-72.4 N, 5-33 E, the
windcast LAEA plane). Its 16 terrain products are box means of the Kartverket DTM features:
    elevation, slope, aspect_sin / aspect_cos (downslope bearing), tpi_1km / tpi_5km
    (elevation minus its 1 / 5 km mean), std_1km, land_fraction, and sx_<dir> for the 8
    directions: the steepest terrain rise within 1 km toward <dir>, as an angle (rad);
    positive = sheltered from winds blowing FROM <dir> (Winstral-type exposure).
Derived here, on the 1 km grid:
    elev_rel_10km / elev_rel_30km  elevation minus its Gaussian mean (sigma 5 / 12 km); the
                                   30 km version is roughly the height above the terrain a
                                   0.25 deg model sees
    land_10km / land_30km          land fraction, Gaussian mean (sigma 5 / 12 km): coast,
                                   fjord or open sea
    sx10_<dir>                     as sx_<dir>, but the steepest rise within 10 km

Writes
    ROOT/static/terrain_descriptors_1km.npz  fields (F, H, W), names, lat, lon, xy
    ROOT/cache/station_terrain.npz           station, names, values (S, F) at the nearest
                                             cell, ij, dist_km (for the A1 stations)
"""
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree

ROOT = Path("/cluster/work/projects/nn8106k/siyan/windcast")
NNO = Path("/cluster/work/projects/nn8106k/siyan/mtno/cache/nno/grid.npz")
STATIONS = ROOT / "cache" / "multimodel_2024-03-01_2025-09-12.npz"
DIRS = ("n", "ne", "e", "se", "s", "sw", "w", "nw")
STEPS = ((1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1))   # (row, col); rows go north


def shelter(elev, dx_km, radius_km):
    """Steepest rise within radius_km toward each of the 8 directions, as an angle (rad)."""
    k_max = int(round(radius_km / dx_km))
    pad = np.pad(elev, k_max, mode="edge")
    ny, nx = elev.shape
    out = []
    for dy, dx in STEPS:
        best = np.full(elev.shape, -np.pi / 2, np.float32)
        step = np.hypot(dy, dx) * dx_km * 1000.0
        for k in range(1, k_max + 1):
            nb = pad[k_max + k * dy: k_max + k * dy + ny, k_max + k * dx: k_max + k * dx + nx]
            best = np.maximum(best, np.arctan((nb - elev) / (k * step)))
        out.append(best)
    return out


def main():
    g = np.load(NNO, allow_pickle=True)
    xy = g["xy"]
    rows_north = xy[1, 0, 1] > xy[0, 0, 1]
    assert rows_north, "rows must go north for the shelter directions"
    dx_km = float(np.mean(np.diff(xy[0, :, 0])))
    names = [str(n) for n in g["terrain_names"]]
    fields = [f for f in g["terrain"].astype(np.float32)]
    elev, land = fields[names.index("elevation")], fields[names.index("land_fraction")]
    for s, tag in ((5, "10km"), (12, "30km")):
        fields.append(elev - gaussian_filter(elev, s / dx_km, mode="nearest"))
        names.append(f"elev_rel_{tag}")
    for s, tag in ((5, "10km"), (12, "30km")):
        fields.append(gaussian_filter(land, s / dx_km, mode="nearest"))
        names.append(f"land_{tag}")
    fields += shelter(elev, dx_km, 10.0)
    names += [f"sx10_{d}" for d in DIRS]
    F = np.stack(fields).astype(np.float32)
    out = ROOT / "static" / "terrain_descriptors_1km.npz"
    np.savez_compressed(out, fields=F, names=np.array(names), lat=g["lat"], lon=g["lon"], xy=xy)
    print(out, F.shape)

    station_values(F, names, xy, STATIONS, ROOT / "cache" / "station_terrain.npz")


def station_values(F, names, xy, stations, out2):
    from windcast.geo import to_plane
    st = np.load(stations, allow_pickle=True)
    p = to_plane(st["lon_st"], st["lat_st"])
    dist, k = cKDTree(xy.reshape(-1, 2)).query(p)
    i, j = np.unravel_index(k, xy.shape[:2])
    vals = F[:, i, j].T
    np.savez_compressed(out2, station=st["station"], names=np.array(names), values=vals,
                        ij=np.stack([i, j], -1), dist_km=dist)
    print(out2, vals.shape, f"max distance to cell centre {dist.max():.2f} km")
    for n, v in zip(names, vals.T):
        print(f"  {n:14s} {v.min():9.3f} {np.median(v):9.3f} {v.max():9.3f}")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    if len(sys.argv) == 3 and sys.argv[1] == "--stations" and sys.argv[2] in ("ext", "all"):
        tag = sys.argv[2]
        t = np.load(ROOT / "static" / "terrain_descriptors_1km.npz", allow_pickle=True)
        station_values(t["fields"], [str(x) for x in t["names"]], t["xy"],
                       ROOT / "cache" / f"multimodel_2024-03-01_2025-09-12_{tag}.npz",
                       ROOT / "cache" / f"station_terrain_{tag}.npz")
    else:
        main()
