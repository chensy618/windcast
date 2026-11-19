"""Model-group versions of the multimodel samples (track A): the same runs, stations and
observations as cache/multimodel_2024-03-01_2025-09-12.npz, with the forecasts of another set
of models, for comparing input groups with the same operator + MLP head.

    python scripts/build_group_samples.py --group nwp      # IFS, GFS, MEPS, AROME-Arctic
    python scripts/build_group_samples.py --group mlwp     # AIFS, GraphCast, Pangu, FourCastNet

Fields: 10u, 10v, 2t, msl (the ones every model has), on the 0.25 deg grid of the base file
(49 x 121, 74 -> 62 N, 6 -> 36 E):
* global models on 0.25 deg (IFS, AIFS, GFS, the MLWP models): the same grid points (GFS and
  the MLWP models are on a larger box, which is cut);
* the 2.5 km Lambert models (MEPS, AROME-Arctic): the mean of their cells inside each 0.25 deg
  cell; cells outside the model's window (MEPS covers 64-72 N only) take the IFS value. At the
  stations, their nearest 2.5 km cell (not the 0.25 deg mean).
Runs where one of the models is missing get keep = False.

OUT ROOT/cache/multimodel_2024-03-01_2025-09-12_<group>.npz        same keys as the base file
    ROOT/cache/multimodel_2024-03-01_2025-09-12_<group>_grid.npy   (N, L, M, 4, 49, 121) float16
"""
import argparse
import sys
from pathlib import Path

import netCDF4
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from windcast.geo import to_plane  # noqa: E402

ROOT = Path("/cluster/work/projects/nn8106k/siyan/windcast")
TAG = "2024-03-01_2025-09-12"
GROUPS = {"nwp": ["ifs", "gfs", "meps", "arome_arctic"],
          "mlwp": ["aifs", "graphcast", "pangu", "fourcastnet"]}
FIELDS = ["10u", "10v", "2t", "msl"]
LAMBERT_VARS = {"10u": "u10", "10v": "v10", "2t": "t2m", "msl": "msl"}   # MEPS / AROME names
LAMBERT = {"meps", "arome_arctic"}


def path(m, t):
    d = ROOT / "ecmwf" / m if m in ("ifs", "aifs") else ROOT / m
    return d / f"{t:%Y}" / f"{m}_{t:%Y%m%dT%H}.nc"


class Lambert:
    """Mapping of one 2.5 km window onto the 0.25 deg grid (cell mean) and the stations (nearest)."""

    def __init__(self, ds, lat, lon, st_lon, st_lat):
        glon, glat = ds["lon"][:], ds["lat"][:]
        self.shape = glon.shape
        i = np.rint((glat - lat[0]) / (lat[1] - lat[0])).astype(int).ravel()
        j = np.rint((glon - lon[0]) / (lon[1] - lon[0])).astype(int).ravel()
        ok = (i >= 0) & (i < len(lat)) & (j >= 0) & (j < len(lon))
        self.src = np.where(ok)[0]
        self.cell = i[ok] * len(lon) + j[ok]
        self.count = np.bincount(self.cell, minlength=len(lat) * len(lon))
        self.HW = (len(lat), len(lon))
        dist, self.st_idx = cKDTree(to_plane(glon.ravel(), glat.ravel())).query(to_plane(st_lon, st_lat))
        assert dist.max() < 2.0, f"station outside the window ({dist.max():.1f} km)"

    def to_grid(self, v, fill):
        """v (L, y, x) -> (L, H, W): mean over the native cells of each 0.25 deg cell, else fill."""
        out = fill.copy()
        has = self.count > 0
        for k in range(v.shape[0]):
            s = np.bincount(self.cell, v[k].ravel()[self.src], minlength=has.size)
            out[k].reshape(-1)[has] = s[has] / self.count[has]
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", required=True, choices=list(GROUPS))
    a = ap.parse_args()
    models = GROUPS[a.group]
    base = dict(np.load(ROOT / "cache" / f"multimodel_{TAG}.npz", allow_pickle=True))
    inits, leads = pd.DatetimeIndex(base["inits"]), [int(x) for x in base["leads"]]
    lat, lon = base["lat"].astype(float), base["lon"].astype(float)
    dlat, dlon = lat[1] - lat[0], lon[1] - lon[0]
    fi, fj = (base["lat_st"] - lat[0]) / dlat, (base["lon_st"] - lon[0]) / dlon
    i0, j0 = np.floor(fi).astype(int), np.floor(fj).astype(int)
    wa, wb = fi - i0, fj - j0
    Wb = np.stack([(1 - wa) * (1 - wb), (1 - wa) * wb, wa * (1 - wb), wa * wb], -1)
    II, JJ = np.stack([i0, i0, i0 + 1, i0 + 1], -1), np.stack([j0, j0 + 1, j0, j0 + 1], -1)

    N, L, M, C, S = len(inits), len(leads), len(models), len(FIELDS), len(base["station"])
    out_grid = ROOT / "cache" / f"multimodel_{TAG}_{a.group}_grid.npy"
    grid = np.lib.format.open_memmap(out_grid, mode="w+", dtype=np.float16, shape=(N, L, M, C, len(lat), len(lon)))
    st_fc = np.full((N, L, M, S, 2), np.nan, np.float32)
    keep = base["keep"].copy()
    lam, cut = {}, {}
    for n, t in enumerate(inits):
        ifs = None
        for k, m in enumerate(models):
            f = path(m, t)
            if not f.exists():
                keep[n] = False
                continue
            try:
                ds = netCDF4.Dataset(f)
            except OSError:
                keep[n] = False
                continue
            with ds:
                fl = [int(x) for x in ds["lead"][:]]
                if any(ld not in fl for ld in leads):
                    keep[n] = False
                    continue
                jl = [fl.index(ld) for ld in leads]
                if m in LAMBERT:
                    if m not in lam or lam[m].shape != ds["lon"].shape:
                        lam[m] = Lambert(ds, lat, lon, base["lon_st"], base["lat_st"])
                    for c, fn in enumerate(FIELDS):
                        v = np.ma.filled(ds[LAMBERT_VARS[fn]][:].astype(np.float32), np.nan)[jl]
                        if fn == "msl":
                            v = v / 100.0
                        fill = ifs[:, c] if ifs is not None else np.full((L,) + lam[m].HW, np.nan, np.float32)
                        grid[n, :, k, c] = lam[m].to_grid(v, fill)
                        if c < 2:
                            st_fc[n, :, k, :, c] = v.reshape(L, -1)[:, lam[m].st_idx]
                else:
                    if m not in cut:
                        glat, glon = ds["lat"][:].astype(float), ds["lon"][:].astype(float)
                        ia = [int(np.argmin(abs(glat - x))) for x in lat]
                        ja = [int(np.argmin(abs(glon - x))) for x in lon]
                        assert np.allclose(glat[ia], lat) and np.allclose(glon[ja], lon), m
                        cut[m] = (slice(ia[0], ia[-1] + 1), slice(ja[0], ja[-1] + 1))
                    sy, sx = cut[m]
                    vals = []
                    for c, fn in enumerate(FIELDS):
                        v = np.ma.filled(ds[fn][:, sy, sx].astype(np.float32), np.nan)[jl]
                        if fn == "msl":
                            v = v / 100.0
                        vals.append(v)
                        if c < 2:
                            st_fc[n, :, k, :, c] = (v[:, II, JJ] * Wb).sum(-1)
                    vals = np.stack(vals, 1)                     # (L, C, H, W)
                    grid[n, :, k] = vals
                    if m == "ifs":
                        ifs = vals
        if n % 100 == 0:
            print(n, t, flush=True)
    grid.flush()
    keep &= np.isfinite(st_fc).all((1, 2, 3, 4))
    keep &= np.isfinite(np.asarray(grid[:, :, :, :2, ::8, ::8], np.float32)).all((1, 2, 3, 4, 5))
    print(f"{a.group} {models}: runs kept {keep.sum()} / {N} (base {base['keep'].sum()})")
    out = {**base, "models": np.array(models), "fields": np.array(FIELDS), "keep": keep, "st_fc": st_fc}
    np.savez_compressed(ROOT / "cache" / f"multimodel_{TAG}_{a.group}.npz", **out)
    print("done", out_grid)


if __name__ == "__main__":
    main()
