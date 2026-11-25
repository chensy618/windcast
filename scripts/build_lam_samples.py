"""2.5 km version of the multimodel samples (track A): MEPS and AROME-Arctic only, on a common
2.5 km grid, never on the 0.25 deg grid. Same runs, stations and observations as
cache/multimodel_2024-03-01_2025-09-12.npz.

    python scripts/build_lam_samples.py

Grid: 2.5 km cells of the working plane (windcast.geo, LAEA 20 E 68 N), x -410 .. 450 km,
y -340 .. 405 km (345 x 299 cells; the 99 stations span x -374 .. 414, y -299 .. 364). Each
model is put on it by its nearest native 2.5 km cell (both models are 2.5 km, so this keeps
their resolution; distance <= 1.8 km). MEPS covers the whole grid; AROME-Arctic misses 3.8 %
of it (the south-east corner), where it takes the MEPS value of the same cell. Winds are
earth-relative (scripts/download_meps.py).
Fields (all surface fields of the downloaded files; FIELDS): 10u, 10v, 2t, msl (hPa) -- the four
of the other groups, first -- then fg10 (gust), rh2m, tcc, tp6 (log1p of the precipitation of the
6 h before the valid time, kg m-2; 0 at lead 0) and blh (boundary-layer height, m). The trainer
picks a subset with --fields.
--fields6: the six fields of the IFS + AIFS samples instead, the same for all four models: 10u, 10v,
2t, msl, sp (hPa; ROOT/<model>_sfc2, download_meps.py --vars sp skt) and t850 (K; ROOT/<model>_pl,
download_meps.py --pl) -> multimodel_<TAG>_lam6.npz / _lam6_grid.npy.
At the stations: the nearest native cell of each model (as in build_group_samples.py).
Runs where one of the two models is missing get keep = False.

OUT ROOT/cache/multimodel_2024-03-01_2025-09-12_lam.npz        keys of the base file, + grid_x, grid_y (km)
    ROOT/cache/multimodel_2024-03-01_2025-09-12_lam_grid.npy   (N, L, 2, 9, 299, 345) float16 (~45 GB)
"""
import sys
from pathlib import Path

import netCDF4
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from windcast.geo import to_plane  # noqa: E402
from build_group_samples import LAMBERT_VARS, ROOT, TAG, NoGrid, path  # noqa: E402

MODELS = ["meps", "arome_arctic"]
FIELDS = ["10u", "10v", "2t", "msl", "fg10", "rh2m", "tcc", "tp6", "blh"]
FILE_VARS = {**LAMBERT_VARS, "fg10": "fg10", "rh2m": "rh2m", "tcc": "tcc", "tp6": "tp_acc", "blh": "blh"}
GRID_X = np.arange(-410.0, 450.01, 2.5)
GRID_Y = np.arange(-340.0, 405.01, 2.5)
MAX_KM = 2.5   # a grid cell further than this from every native cell is outside the model's window
FIELDS6 = ["10u", "10v", "2t", "msl", "sp", "t850"]
EXTRA = {"sp": "_sfc2", "t850": "_pl"}          # fields from the extra downloads: directory suffix


class Nearest:
    """Native 2.5 km window -> the common grid (nearest cell) and the stations (nearest cell)."""

    def __init__(self, ds, st_xy):
        p = to_plane(ds["lon"][:], ds["lat"][:]).reshape(-1, 2)
        self.shape = ds["lon"].shape
        XX, YY = np.meshgrid(GRID_X, GRID_Y)
        tree = cKDTree(p)
        dist, self.idx = tree.query(np.stack([XX.ravel(), YY.ravel()], -1))
        self.inside = dist <= MAX_KM
        d_st, self.st_idx = tree.query(st_xy)
        self.st_in = d_st < 2.0   # outside (AROME-Arctic, 3 Swedish / Finnish stations): the MEPS value


def main():
    # ext: + the Swedish / Finnish stations; the grid does not depend on the stations, so the
    # base grid file is linked and only the station values are read
    global FIELDS
    args = sys.argv[1:]
    six = "--fields6" in args
    args = [x for x in args if x != "--fields6"]
    sfx = "_ext" if args == ["--stations", "ext"] else ""
    assert sfx or not args, "usage: build_lam_samples.py [--stations ext] [--fields6]"
    gsfx = "_lam6" if six else "_lam"
    if six:
        FIELDS = FIELDS6
    base = dict(np.load(ROOT / "cache" / f"multimodel_{TAG}{sfx}.npz", allow_pickle=True))
    inits, leads = pd.DatetimeIndex(base["inits"]), [int(x) for x in base["leads"]]
    st_xy = base["xy"].astype(float)
    N, L, M, C, S = len(inits), len(leads), len(MODELS), len(FIELDS), len(base["station"])
    H, W = len(GRID_Y), len(GRID_X)
    out_grid = ROOT / "cache" / f"multimodel_{TAG}{gsfx}_grid.npy"
    if sfx:
        grid = NoGrid()
        base_keep = np.load(ROOT / "cache" / f"multimodel_{TAG}{gsfx}.npz")["keep"]
    else:
        grid = np.lib.format.open_memmap(out_grid, mode="w+", dtype=np.float16, shape=(N, L, M, C, H, W))
    st_fc = np.full((N, L, M, S, 2), np.nan, np.float32)
    keep = base["keep"].copy()
    near = {}
    for n, t in enumerate(inits):
        vals = {}
        for k, m in enumerate(MODELS):
            f = path(m, t)
            try:
                ds = netCDF4.Dataset(f)
                xds = {fn: netCDF4.Dataset(f.parents[1].with_name(f.parents[1].name + d) / f.parent.name / f.name)
                       for fn, d in EXTRA.items() if fn in FIELDS}
            except (OSError, FileNotFoundError):
                break
            xl = {fn: [int(x) for x in x_["lead"][:]] for fn, x_ in xds.items()}
            if any(ld not in xl[fn] for fn in xds for ld in leads) or \
                    any(x_["lon"].shape != ds["lon"].shape for x_ in xds.values()):
                ds.close(); [x_.close() for x_ in xds.values()]
                break
            with ds:
                fl = [int(x) for x in ds["lead"][:]]
                if any(ld not in fl for ld in leads):
                    break
                if any(ld - 6 not in fl for ld in leads if ld >= 6):
                    break
                jl = [fl.index(ld) for ld in leads]
                if m not in near or near[m].shape != ds["lon"].shape:
                    near[m] = Nearest(ds, st_xy)
                    print(f"{m} grid {ds['lon'].shape}: {near[m].inside.mean():.3f} of the 2.5 km grid inside",
                          flush=True)
                read = lambda fn: np.ma.filled(ds[FILE_VARS[fn]][:].astype(np.float32), np.nan)
                v = []
                for fn in FIELDS:
                    if fn in xds:     # extra download: its own lead axis
                        a = np.ma.filled(xds[fn][fn][:].astype(np.float32), np.nan)[[xl[fn].index(ld) for ld in leads]]
                        v.append(a / 100.0 if fn == "sp" else a)
                        continue
                    a = read(fn)
                    if fn == "tp6":   # accumulated since init -> the 6 h before the valid time
                        prev = [fl.index(ld - 6) if ld >= 6 else fl.index(ld) for ld in leads]
                        a = np.log1p(np.clip(a[jl] - a[prev], 0.0, None))
                    else:
                        a = a[jl]
                    v.append(a)
                v = np.stack(v, 1).reshape(L, C, -1)                            # (L, C, y*x)
                v[:, FIELDS.index("msl")] /= 100.0
                st_fc[n, :, k] = v[:, :2, near[m].st_idx].transpose(0, 2, 1)
                out_ = ~near[m].st_in
                if out_.any():
                    assert m != "meps", "station outside MEPS"
                    st_fc[n, :, k, out_] = st_fc[n, :, 0, out_]
                vals[m] = (v[:, :, near[m].idx], near[m].inside)
            for x_ in xds.values():
                x_.close()
        if len(vals) < M:
            keep[n] = False
            continue
        meps = vals["meps"][0]
        for k, m in enumerate(MODELS):
            g, inside = vals[m]
            g = np.where(inside, g, meps)                                       # AROME outside: MEPS
            grid[n, :, k] = g.reshape(L, C, H, W)
        if n % 100 == 0:
            print(n, t, flush=True)
    keep &= np.isfinite(st_fc).all((1, 2, 3, 4))
    if sfx:
        keep &= base_keep
        link = ROOT / "cache" / f"multimodel_{TAG}{sfx}{gsfx}_grid.npy"
        link.unlink(missing_ok=True)
        link.symlink_to(out_grid.name)
    else:
        grid.flush()
        keep &= np.isfinite(np.asarray(grid[:, :, :, :, ::8, ::8], np.float32)).all((1, 2, 3, 4, 5))
    print(f"lam {MODELS}: runs kept {keep.sum()} / {N} (base {base['keep'].sum()})")
    out = {**base, "models": np.array(MODELS), "fields": np.array(FIELDS), "keep": keep, "st_fc": st_fc,
           "grid_x": GRID_X, "grid_y": GRID_Y}
    np.savez_compressed(ROOT / "cache" / f"multimodel_{TAG}{sfx}{gsfx}.npz", **out)
    print("done", out_grid)


if __name__ == "__main__":
    main()
