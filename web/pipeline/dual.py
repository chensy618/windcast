"""WindCast on the website: the four-field dual-scale GNO (windcast/training/train_dual_gno.py) at the
1 km cells of one run.

    python web/pipeline/dual.py --prepare                  # once after training (CPU, ~100 GB RAM)
    python web/pipeline/dual.py --check 2025-08-19T12      # inputs rebuilt from the files == training cache

Model: run tag RUN_TAG (width 16, 8 x 16 modes, GNO readout 60 / 10 km, MLP head starting at the
four-model mean; inputs 10u, 10v, 2t, msl of IFS, AIFS, MEPS, AROME-Arctic) of the extended
period (train 2024-03 .. 2025-12, PERIOD), the mean of its 15 models (5 station folds x 3 seeds).
The spread is the s.d. of their wind speeds.

Inputs, as in training (DualData) and in scripts/eval_metnordic_grid.py:
* global branch: IFS and AIFS on the 0.25 deg grid of the samples (49 x 121);
* regional branch: MEPS and AROME-Arctic on the common 2.5 km grid of scripts/build_lam_samples.py
  (nearest native cell; AROME-Arctic outside its window: the MEPS value);
* point features of a cell: its 28 terrain descriptors (z-scored with the station mean / s.d.),
  the 10 m wind of the four models (IFS / AIFS bilinear, MEPS / AROME the nearest 2.5 km cell),
  and the 5 time features;
* every channel is z-scored with the statistics of the training data (STATS, from --prepare).
Cells outside the 2.5 km grid get no forecast.
"""
import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import netCDF4
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "scripts")]
PERIOD = "2024-03-01_2026-11-14"
from build_group_samples import LAMBERT_VARS, path as group_path  # noqa: E402
from build_lam_samples import GRID_X, GRID_Y, Nearest  # noqa: E402
from windcast.geo import to_plane  # noqa: E402

# torch and the training module are imported only where the models run (forecast), so that
# --prepare and --check work in a CPU environment without torch
ROOT = Path("/cluster/work/projects/nn8106k/siyan/windcast")
DATA_G = ROOT / "cache" / f"multimodel_{PERIOD}"           # as train_dual_gno.DATA_G / DATA_R (+ _lam)
TERRAIN = ROOT / "cache" / "station_terrain.npz"
TRAIN = ("2024-03-01", "2025-12-31")                        # train_station_operator.PERIODS[PERIOD]["train"]
RUN_TAG = "rg60_rr10_r16m8x16_g16m8x16_z0_head-mlp"
RUNS = ROOT / f"runs_dual_{PERIOD}"
STATS = ROOT / "cache" / f"web_dual_stats_{PERIOD}.npz"
FIELDS = ["10u", "10v", "2t", "msl"]                       # train_dual_gno.FIELDS
G_MODELS, R_MODELS = ["ifs", "aifs"], ["meps", "arome_arctic"]
MODELS = G_MODELS + R_MODELS
CHUNK = 8000                                                # query cells per GNO call


# ---------------------------------------------------------------------- statistics (once)
def prepare():
    """The normalisation of DualData (train_dual_gno.py), without loading the grids: channel mean /
    s.d. over every 4th (global) / 16th (regional) training run, the point-wind and residual
    statistics over all training samples, the station mean / s.d. of the terrain descriptors."""
    dg = dict(np.load(f"{DATA_G}.npz", allow_pickle=True))
    dr = dict(np.load(f"{DATA_G}_lam.npz", allow_pickle=True))
    assert (dg["inits"] == dr["inits"]).all() and (dg["station"] == dr["station"]).all()
    inits = pd.DatetimeIndex(dg["inits"])
    keep = dg["keep"] & dr["keep"]
    tr = np.where((inits >= TRAIN[0]) & (inits <= pd.Timestamp(TRAIN[1]) + pd.Timedelta(hours=12)) & keep)[0]
    fc = np.concatenate([dg["st_fc"], dr["st_fc"]], 2)                       # (N, L, 4, S, 2)
    ti = pd.Index(dg["time"]).get_indexer(inits)
    obs = np.stack([dg["obs"][ti + ld] for ld in dg["leads"]], 1)
    y = obs - fc.mean(2)
    N, L, M, S_, _ = fc.shape
    f8 = fc.transpose(0, 1, 3, 2, 4).reshape(N, L, S_, -1)[tr]

    def grid_stats(stem, d, step):
        mm = np.load(f"{stem}_grid.npy", mmap_mode="r")
        names = [str(x) for x in d["fields"]]
        g = np.stack([np.asarray(mm[n], np.float32)[:, :, [names.index(f) for f in FIELDS]] for n in tr[::step]])
        return (g.mean((0, 1, 4, 5)).reshape(1, -1, 1, 1),                     # (1, M * C, 1, 1)
                g.std((0, 1, 4, 5)).reshape(1, -1, 1, 1) + 1e-6)

    gmu, gsd = grid_stats(DATA_G, dg, 4)
    rmu, rsd = grid_stats(f"{DATA_G}_lam", dr, 16)
    t = np.load(TERRAIN, allow_pickle=True)
    val = t["values"].astype(np.float64)
    np.savez(STATS, gmu=gmu, gsd=gsd, rmu=rmu, rsd=rsd, y_std=float(np.nanstd(y[tr])),
             f8_mu=np.nanmean(f8, (0, 1, 2)), f8_sd=np.nanstd(f8, (0, 1, 2)) + 1e-9,
             st_mu=val.mean(0), st_sd=val.std(0) + 1e-9, terrain_names=t["names"], lat=dg["lat"], lon=dg["lon"],
             leads=dg["leads"], grid_x=GRID_X, grid_y=GRID_Y, fields=np.array(FIELDS), n_train=len(tr))
    print(f"{len(tr)} training runs -> {STATS}")


# ---------------------------------------------------------------------- inputs of one run
def available(init):
    return all(group_path(m, init).exists() for m in MODELS)


def read_inputs(init, S):
    """Global grid (L, 2, 4, Hg, Wg) and regional grid (L, 2, 4, Hr, Wr), raw units (K, hPa, m/s)."""
    leads = [int(x) for x in S["leads"]]
    lat, lon = S["lat"].astype(float), S["lon"].astype(float)
    L = len(leads)
    gg = np.zeros((L, 2, len(FIELDS), len(lat), len(lon)), np.float32)
    for k, m in enumerate(G_MODELS):
        with netCDF4.Dataset(group_path(m, init)) as ds:
            fl = [int(x) for x in ds["lead"][:]]
            jl = [fl.index(ld) for ld in leads]
            glat, glon = ds["lat"][:].astype(float), ds["lon"][:].astype(float)
            ia = [int(np.argmin(abs(glat - x))) for x in lat]
            ja = [int(np.argmin(abs(glon - x))) for x in lon]
            assert np.allclose(glat[ia], lat) and np.allclose(glon[ja], lon), m
            for c, fn in enumerate(FIELDS):
                v = np.ma.filled(ds[fn][:, ia[0]:ia[-1] + 1, ja[0]:ja[-1] + 1].astype(np.float32), np.nan)[jl]
                if ia[0] > ia[-1]:
                    raise ValueError("latitude order")
                gg[:, k, c] = v / 100.0 if fn == "msl" else v
    rg = np.zeros((L, 2, len(FIELDS), len(GRID_Y), len(GRID_X)), np.float32)
    vals = {}
    for m in R_MODELS:
        with netCDF4.Dataset(group_path(m, init)) as ds:
            fl = [int(x) for x in ds["lead"][:]]
            jl = [fl.index(ld) for ld in leads]
            near = Nearest(ds, np.zeros((1, 2)))
            v = np.stack([np.ma.filled(ds[LAMBERT_VARS[fn]][:].astype(np.float32), np.nan)[jl] for fn in FIELDS], 1)
            v = v.reshape(L, len(FIELDS), -1)
            v[:, FIELDS.index("msl")] /= 100.0
            vals[m] = (v[:, :, near.idx], near.inside)
    for k, m in enumerate(R_MODELS):
        g, inside = vals[m]
        rg[:, k] = np.where(inside, g, vals["meps"][0]).reshape(L, len(FIELDS), len(GRID_Y), len(GRID_X))
    assert np.isfinite(gg).all() and np.isfinite(rg).all(), f"{init}: non-finite input fields"
    return gg, rg


def point_winds(gg, rg, S, lat, lon, xy):
    """10 m wind of the four models at the cells (L, P, 4, 2): IFS / AIFS bilinear, MEPS / AROME nearest."""
    la, lo = S["lat"].astype(float), S["lon"].astype(float)
    fi, fj = (lat - la[0]) / (la[1] - la[0]), (lon - lo[0]) / (lo[1] - lo[0])
    i0, j0 = np.floor(fi).astype(int), np.floor(fj).astype(int)
    a, b = fi - i0, fj - j0
    W = np.stack([(1 - a) * (1 - b), (1 - a) * b, a * (1 - b), a * b], -1)
    II, JJ = np.stack([i0, i0, i0 + 1, i0 + 1], -1), np.stack([j0, j0 + 1, j0, j0 + 1], -1)
    wg = (gg[:, :, :2][..., II, JJ] * W).sum(-1)                               # (L, 2, 2, P)
    ri = np.rint((xy[:, 1] - GRID_Y[0]) / (GRID_Y[1] - GRID_Y[0])).astype(int)
    rj = np.rint((xy[:, 0] - GRID_X[0]) / (GRID_X[1] - GRID_X[0])).astype(int)
    wr = rg[:, :, :2][..., ri, rj]                                              # (L, 2, 2, P)
    return np.concatenate([wg, wr], 1).transpose(0, 3, 1, 2)                   # (L, P, 4, 2)


def inside(S, lat, lon, xy):
    """Cells the dual model can forecast: inside the 2.5 km grid and the 0.25 deg grid."""
    la, lo = S["lat"].astype(float), S["lon"].astype(float)
    return ((xy[:, 0] >= GRID_X[0]) & (xy[:, 0] <= GRID_X[-1]) & (xy[:, 1] >= GRID_Y[0]) & (xy[:, 1] <= GRID_Y[-1])
            & (lat <= la.max()) & (lat >= la.min()) & (lon >= lo.min()) & (lon <= lo.max()))


# ---------------------------------------------------------------------- models
def neighbours(src, qry, radius, dev):
    """GNO.set_queries for the query cells qry, as tensors (idx, mask, rel)."""
    nb = cKDTree(src).query_ball_point(qry, radius)
    cnt = np.array([len(x) for x in nb])
    assert cnt.min() > 0
    K = int(cnt.max())
    idx = np.zeros((len(qry), K), np.int64)
    mask = np.zeros((len(qry), K), np.float32)
    for p, x in enumerate(nb):
        idx[p, :len(x)], mask[p, :len(x)] = x, 1.0
    rel = (src[idx] - qry[:, None]) / radius
    import torch
    return (torch.tensor(idx, device=dev), torch.tensor(mask, device=dev),
            torch.tensor(rel, dtype=torch.float32, device=dev))


def load_models(S, dev):
    import torch
    import windcast.training.train_dual_gno as tdg
    lat, lon = S["lat"].astype(float), S["lon"].astype(float)
    LON, LAT = np.meshgrid(lon, lat)
    GX, GY = np.meshgrid(GRID_X, GRID_Y)
    C = len(FIELDS)
    D = SimpleNamespace(Cg=2 * C, Cr=2 * C, Hg=len(lat), Wg=len(lon), Hr=len(GRID_Y), Wr=len(GRID_X),
                        n_point=len(S["st_mu"]) + 8 + 5, pos_g=to_plane(LON, LAT).reshape(-1, 2),
                        pos_r=np.stack([GX, GY], -1).reshape(-1, 2).astype(np.float64),
                        st_xy=np.zeros((1, 2)))
    models = []
    for s in range(3):
        for k in range(5):
            run = RUNS / f"dual_sfno-fno_g{k}_s{s}_{RUN_TAG}_lr1e-05"
            h = json.load(open(run / "history.json"))["args"]
            m = tdg.DualGNO(D, h["enc_g"], h["enc_r"], width=h["width"], width_r=h["width_r"],
                            modes_g=tuple(h["modes_g"]), modes_r=tuple(h["modes_r"]), rg=h["rg"], rr=h["rr"],
                            branches=h["branches"], readout=h["readout"], wind_frame=h["wind_frame"],
                            dropout=h.get("dropout", 0.0), zero_init=h.get("zero_init", False),
                            head=h.get("head", "linear"), head_hidden=h.get("head_hidden", 128))
            sd = {k_: v for k_, v in torch.load(run / "best.pt", weights_only=True, map_location="cpu").items()
                  if not k_.split(".")[-1] in ("idx", "mask", "rel")}
            m.load_state_dict(sd, strict=False)
            models.append(m.to(dev).eval())
    return D, models


def forecast(init, cells, dev):
    """{key: (base (L, P, 2), correction (L, P, 2), spread (L, P))} for each cell set
    (cells[key] = (lat, lon, xy, terrain (P, 28))); NaN outside the model's area."""
    import torch
    torch.set_grad_enabled(False)
    S = dict(np.load(STATS, allow_pickle=True))
    t0 = time.time()
    gg, rg = read_inputs(init, S)
    D, models = load_models(S, dev)
    L = len(S["leads"])
    leads = S["leads"].astype(float)
    vt = pd.Timestamp(init) + pd.to_timedelta(leads, unit="h")
    tf = torch.tensor(np.stack([leads / 66.0, np.sin(2 * np.pi * vt.hour / 24), np.cos(2 * np.pi * vt.hour / 24),
                                np.sin(2 * np.pi * vt.dayofyear / 366), np.cos(2 * np.pi * vt.dayofyear / 366)], -1),
                      dtype=torch.float32, device=dev)
    xg = torch.tensor((gg.reshape(L, -1, D.Hg, D.Wg) - S["gmu"]) / S["gsd"], dtype=torch.float32, device=dev)
    xr = torch.tensor((rg.reshape(L, -1, D.Hr, D.Wr) - S["rmu"]) / S["rsd"], dtype=torch.float32, device=dev)
    lat_g = [m.encode(m.enc_g, xg, tf, D.Hg, D.Wg).flatten(2).transpose(1, 2) for m in models]
    lat_r = [m.encode(m.enc_r, xr, tf, D.Hr, D.Wr).flatten(2).transpose(1, 2) for m in models]
    rg_km, rr_km = models[0].gno_g.radius, models[0].gno_r.radius
    out = {}
    for key, (lat, lon, xy, terr) in cells.items():
        P = len(lat)
        ok = inside(S, lat, lon, xy) & np.isfinite(terr).all(1)
        w = np.full((L, P, 4, 2), np.nan, np.float32)
        w[:, ok] = point_winds(gg, rg, S, lat[ok], lon[ok], xy[ok])
        base = w.mean(2)
        corr = np.full((L, P, 2), np.nan, np.float32)
        spread = np.full((L, P), np.nan, np.float32)
        sel_all = np.where(ok)[0]
        for c0 in range(0, len(sel_all), CHUNK):
            sel = sel_all[c0:c0 + CHUNK]
            nb_g = neighbours(D.pos_g, xy[sel], rg_km, dev)
            nb_r = neighbours(D.pos_r, xy[sel], rr_km, dev)
            f8 = np.nan_to_num((w[:, sel].reshape(L, len(sel), 8) - S["f8_mu"]) / S["f8_sd"]).astype(np.float32)
            tz = ((terr[sel] - S["st_mu"]) / S["st_sd"]).astype(np.float32)
            pt = torch.cat([torch.tensor(tz, device=dev)[None].expand(L, -1, -1), torch.tensor(f8, device=dev)], -1)
            outs = []
            for m, zg, zr in zip(models, lat_g, lat_r):
                for g, nb in ((m.gno_g, nb_g), (m.gno_r, nb_r)):
                    g.idx, g.mask, g.rel = nb
                h = [m.gno_g(zg), m.gno_r(zr), pt, tf[:, None].expand(-1, len(sel), -1)]
                outs.append(m.proj(torch.cat(h, -1)).float().cpu().numpy() * float(S["y_std"]))
            outs = np.stack(outs)                                               # (15, L, Pc, 2)
            corr[:, sel] = outs.mean(0)
            spread[:, sel] = np.linalg.norm(base[None, :, sel] + outs, axis=-1).std(0)
        out[key] = (base, corr, spread)
        print(f"dual {key}: {ok.sum()} / {P} cells, {time.time() - t0:.0f} s", flush=True)
    return out


# ---------------------------------------------------------------------- check against the training cache
def check(init):
    """The inputs rebuilt from the files for a run of the samples equal the training grids."""
    S = dict(np.load(STATS, allow_pickle=True))
    gg, rg = read_inputs(pd.Timestamp(init), S)
    for sfx, arr, name in (("", gg, "global"), ("_lam", rg, "regional")):
        d = np.load(f"{DATA_G}{sfx}.npz", allow_pickle=True)
        n = pd.DatetimeIndex(d["inits"]).get_loc(pd.Timestamp(init))
        mm = np.load(f"{DATA_G}{sfx}_grid.npy", mmap_mode="r")
        names = [str(x) for x in d["fields"]]
        ref = np.asarray(mm[n][:, :, [names.index(f) for f in FIELDS]], np.float32)
        diff = np.abs(arr - ref).reshape(arr.shape[0], arr.shape[1], len(FIELDS), -1).max(-1).max((0, 1))
        print(f"{name}: max |rebuilt - cache| per field {dict(zip(FIELDS, diff.round(3)))}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--check", default="")
    a = ap.parse_args()
    if a.prepare:
        prepare()
    if a.check:
        check(a.check)
