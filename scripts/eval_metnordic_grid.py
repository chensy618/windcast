"""Gridded evaluation against MET Nordic (GPU): the trained station-correction models applied at
1 km cells of the Tromso-Senja MET Nordic grid, away from the stations, over the test period.

    python scripts/eval_metnordic_grid.py --step 2

Points: land cells (land fraction > 0.5) of the MET Nordic reference grid (314 x 343, 1 km),
every --step cells in both directions, matched to the nearest cell of the 1 km terrain grid
(static/terrain_descriptors_1km.npz). Every input is built at the cell itself, as at a station:
  * the 10 m wind of IFS / AIFS (bilinear from 0.25 deg) and of MEPS / AROME-Arctic (nearest
    cell of the common 2.5 km grid; at the stations it is the nearest native cell);
  * the 28 terrain descriptors of the cell, z-scored with the station mean / std of training;
  * the potential-temperature / mountain-wind features (scripts/build_station_physics.py
    features(), with the cell elevation in place of the station height);
  * the encoders see the same grids as in training.
Each configuration is the mean of its 5 fold models (each trained without one station group);
per seed and over all seeds.

Truth: the MET Nordic v4 analysis 10 m wind at the valid time (6-hourly files of
scripts/download_metnordic.py). MET Nordic is built on MEPS, so it is not independent of the
regional models; the station check below measures how close it is to the observations.

Station check: at the stations inside the MET Nordic grid (nearest cell within 1 km), MET Nordic
vs the observation, next to the four-model mean and MEPS + AROME-Arctic, on the same samples.

--configs all: every trained four-model dual configuration of the original period (seeds that have
all 5 folds), not only the three of the report; the raw single models (raw_ifs, raw_aifs, raw_meps,
raw_arome) are always scored too.

OUT ROOT/results/metnordic/grid_eval.npz (--out for another name)
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import xarray as xr
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_station_physics import (MOUNTAIN, THETA, features, global_at, regional_at,  # noqa: E402
                                   terrain_at_stations)
from windcast.geo import to_plane  # noqa: E402
from windcast.station_correction import SX_DIRS, upwind  # noqa: E402
import windcast.training.train_station_operator as tso  # noqa: E402
from windcast.training.train_dual_gno import DualData, DualGNO, RUNS as RUNS_DUAL, TAG  # noqa: E402

ROOT = Path("/cluster/work/projects/nn8106k/siyan/windcast")
MN = ROOT / "metnordic"
SMALL = "r16m8x16_g16m8x16_z0"
DUAL_CFGS = {"dual_gno": "rg60_rr10", "dual_small_mlp": f"rg60_rr10_{SMALL}_head-mlp",
             "dual_small_mlp_phys": f"rg60_rr10_phys-all_{SMALL}_head-mlp"}
DUAL_ALL = {  # every four-model dual configuration of the original period
    "dual_bilinear": "bilinear", "dual_gno": "rg60_rr10", "dual_gno_phys": "rg60_rr10_phys-all",
    "dual_small_bilinear": f"bilinear_{SMALL}", "dual_small_gno": f"rg60_rr10_{SMALL}",
    "dual_small_bilinear_reg": "bilinear_r16m8x16_g16m8x16_do0.3_wd0.01_z0",
    "dual_small_gno_reg": "rg60_rr10_r16m8x16_g16m8x16_do0.3_wd0.01_z0",
    "dual_mlp_bilinear": "bilinear_z0_head-mlp", "dual_mlp_gno": "rg60_rr10_z0_head-mlp",
    "dual_small_mlp_bilinear": f"bilinear_{SMALL}_head-mlp", "dual_small_mlp": f"rg60_rr10_{SMALL}_head-mlp",
    "dual_small_mlp_theta": f"rg60_rr10_phys-theta_{SMALL}_head-mlp",
    "dual_small_mlp_mountain": f"rg60_rr10_phys-mountain_{SMALL}_head-mlp",
    "dual_small_mlp_phys": f"rg60_rr10_phys-all_{SMALL}_head-mlp",
    "dual_w16m4x8_mlp_phys": "rg60_rr10_phys-all_r16m4x8_g16m4x8_z0_head-mlp",
    "dual_w8m8x16_mlp_phys": "rg60_rr10_phys-all_r8m8x16_g8m8x16_z0_head-mlp",
    "dual_w8m4x8_mlp_phys": "rg60_rr10_phys-all_r8m4x8_g8m4x8_z0_head-mlp"}
DUAL_F6 = {  # --configs f6: six-field models (train_dual_gno.py --fields6) and their four-field twins
    "dual_small_mlp": f"rg60_rr10_{SMALL}_head-mlp", "dual_small_mlp_phys": f"rg60_rr10_phys-all_{SMALL}_head-mlp",
    "dual_small_mlp_f6": f"rg60_rr10_{SMALL}_head-mlp_f6",
    "dual_small_mlp_phys_f6": f"rg60_rr10_phys-all_{SMALL}_head-mlp_f6"}


def mn_points(step):
    s = xr.open_dataset(MN / "static.nc")
    lat, lon = s["latitude"].values, s["longitude"].values
    land = s["land_area_fraction"].values
    yy, xx = np.mgrid[0:lat.shape[0]:step, 0:lat.shape[1]:step]
    yy, xx = yy.ravel(), xx.ravel()
    ok = land[yy, xx] > 0.5
    return yy[ok], xx[ok], lat[yy[ok], xx[ok]].astype(float), lon[yy[ok], xx[ok]].astype(float)


def mn_wind(t):
    """MET Nordic 10 m wind (314, 343, 2) at time t, or None."""
    f = MN / f"{t:%Y}" / f"mn_{t:%Y%m%dT%H}.nc"
    if not f.exists():
        return None
    with xr.open_dataset(f) as d:
        return np.stack([d["u10"].values, d["v10"].values], -1).astype(np.float32)


def z_stats(a, tr):
    return np.nanmean(a[tr], (0, 1, 2)), np.nanstd(a[tr], (0, 1, 2)) + 1e-9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=int, default=2)
    ap.add_argument("--max-inits", type=int, default=0, help="smoke test")
    ap.add_argument("--configs", default="report", choices=["report", "all", "f6"],
                    help="f6: the six-field dual models and their four-field twins, on the six-field samples")
    ap.add_argument("--out", default="grid_eval.npz")
    ap.add_argument("--case-init", default="", help="evaluate only this test run (e.g. 2025-06-14T12); its +24 h map is the case")
    a = ap.parse_args()
    cfgs = {"report": DUAL_CFGS, "all": DUAL_ALL, "f6": DUAL_F6}[a.configs]
    dev = torch.device("cuda")
    t0 = time.time()

    # ------------------------------------------------------------------ points
    yy, xx, lat, lon = mn_points(a.step)
    xy = to_plane(lon, lat)
    T = np.load(ROOT / "static" / "terrain_descriptors_1km.npz", allow_pickle=True)
    tnames = [str(x) for x in T["names"]]
    H1, W1 = T["lat"].shape
    d, k = cKDTree(T["xy"].reshape(-1, 2)).query(xy)
    ti, tj = np.unravel_index(k, (H1, W1))
    tval = T["fields"][:, ti, tj].T.astype(np.float64)                         # (P, 28)
    r = 10
    ok = (d < 0.75) & np.isfinite(tval).all(1) & (ti >= r) & (ti < H1 - r) & (tj >= r) & (tj < W1 - r)
    yy, xx, lat, lon, xy, ti, tj, tval = (v[ok] for v in (yy, xx, lat, lon, xy, ti, tj, tval))
    P = len(lat)
    elev = tval[:, tnames.index("elevation")]
    print(f"{P} points (every {a.step} km, land, matched to the 1 km terrain grid)", flush=True)

    # ------------------------------------------------------------------ dual data (station stats)
    D = DualData(dev, phys="all", fields6=a.configs == "f6")
    dg = np.load(f"{ROOT}/cache/multimodel_{TAG}.npz", allow_pickle=True)
    dr = np.load(f"{ROOT}/cache/multimodel_{TAG}_lam.npz", allow_pickle=True)
    tr = D.split["train"]
    fc_st = np.concatenate([dg["st_fc"], dr["st_fc"]], 2)                         # (N, L, 4, S, 2)
    f8_st = fc_st.transpose(0, 1, 3, 2, 4).reshape(D.N, D.L, D.S, -1)
    f8_mu, f8_sd = z_stats(f8_st, tr)
    ph = np.load(ROOT / "cache" / f"station_physics_{TAG}.npz", allow_pickle=True)
    phn = [str(x) for x in ph["names"]]
    assert phn == THETA + MOUNTAIN == D.phys_names
    ph_mu, ph_sd = z_stats(ph["values"], tr)
    stt = np.load(tso.TERRAIN, allow_pickle=True)
    assert [str(x) for x in stt["names"]] == tnames
    st_val = stt["values"].astype(np.float64)
    st_mu, st_sd = st_val.mean(0), st_val.std(0) + 1e-9
    dist_st = cKDTree(D.st_xy).query(xy)[0]

    # station geometry, kept for the station check, then the dual data is moved to the points
    st_xy, st_lat, st_lon = D.st_xy.copy(), dg["lat_st"].astype(float), dg["lon_st"].astype(float)
    D.st_xy = xy.astype(np.float64)
    latg, long_ = dg["lat"].astype(float), dg["lon"].astype(float)
    fi = (lat - latg[0]) / (latg[1] - latg[0])
    fj = (lon - long_[0]) / (long_[1] - long_[0])
    D.sample_g = D._t(np.stack([fj / (D.Wg - 1) * 2 - 1, fi / (D.Hg - 1) * 2 - 1], -1))
    gx, gy = dr["grid_x"], dr["grid_y"]
    D.sample_r = D._t(np.stack([(xy[:, 0] - gx[0]) / (gx[-1] - gx[0]) * 2 - 1,
                                (xy[:, 1] - gy[0]) / (gy[-1] - gy[0]) * 2 - 1], -1))
    D.S = P
    static_z = D._t((tval - st_mu) / st_sd)                                    # (P, 28)

    # ------------------------------------------------------------------ dual models
    models, dual_seeds = {}, {}
    for name, tag in cfgs.items():
        dual_seeds[name] = [s for s in range(3) if all(
            (RUNS_DUAL / f"dual_sfno-fno_g{f}_s{s}_{tag}_lr1e-05" / "best.pt").exists() for f in range(5))]
        for s in dual_seeds[name]:
            for f in range(5):
                run = RUNS_DUAL / f"dual_sfno-fno_g{f}_s{s}_{tag}_lr1e-05"
                h = json.load(open(run / "history.json"))["args"]
                pidx = {"none": [], "all": list(range(len(phn))), "theta": [phn.index(q) for q in THETA],
                        "mountain": [phn.index(q) for q in MOUNTAIN]}[h["phys"]]
                D.n_point = 28 + 8 + 5 + len(pidx)
                # a four-field model on the six-field grids: the first four fields of each model
                nf = 6 if h.get("fields6") else 4
                Cg, Cr, F = D.Cg, D.Cr, len(D.fields)
                D.Cg, D.Cr = D.Cg // F * nf, D.Cr // F * nf
                m = DualGNO(D, h["enc_g"], h["enc_r"], width=h["width"], width_r=h["width_r"],
                            modes_g=tuple(h.get("modes_g", [16, 32])), modes_r=tuple(h["modes_r"]),
                            rg=h["rg"], rr=h["rr"], branches=h["branches"], readout=h["readout"],
                            wind_frame=h["wind_frame"], dropout=h.get("dropout", 0.0),
                            zero_init=h.get("zero_init", False), head=h.get("head", "linear")).to(dev)
                m.load_state_dict(torch.load(run / "best.pt", weights_only=True), strict=True)
                m.phys_idx = pidx
                m.gch = [i * F + c for i in range(Cg // F) for c in range(nf)] if nf != F else None
                m.rch = [i * F + c for i in range(Cr // F) for c in range(nf)] if nf != F else None
                D.Cg, D.Cr = Cg, Cr
                models.setdefault(name, []).append((s, m.eval()))
    print("dual models loaded", {k: len(v) for k, v in models.items()},
          {k: getattr(v[0][1], "gno_r").n_neighbours for k, v in models.items() if hasattr(v[0][1], "gno_r")},
          flush=True)

    # ------------------------------------------------------------------ station models per input group
    # point-wise MLP / SFNO + MLP (/ FNO + MLP) of train_station_operator.py --inputs terrain, input
    # group IFS + AIFS ('ecmwf'), MEPS + AROME-Arctic ('lam', 2.5 km grid) and all four ('mix');
    # baseline: the mean of the group's models. The grids are read per run (need_grid=False).
    SGROUPS = {"ecmwf": [0, 1], "lam": [2, 3], "mix": [0, 1, 2, 3]} if a.configs == "all" else {"mix": [0, 1, 2, 3]}
    SKINDS = ("point", "sfno", "fno") if a.configs == "all" else ("point", "sfno")
    sx = {p: tval[:, [tnames.index(f"{p}_{k}") for k in SX_DIRS]] for p in ("sx", "sx10")}
    sdata, smodels = {}, {}
    for grp, mi in SGROUPS.items():
        tso.DATA = ROOT / "cache" / f"multimodel_{TAG}"
        tso.GRID_FILE = ROOT / "cache" / f"multimodel_{TAG}_grid.npy"
        if grp != "ecmwf":
            tso.use_group(grp)
        Dk = tso.Data(dev, terrain=True, need_grid=False)
        assert Dk.M == len(mi) and (Dk.station == dg["station"]).all()
        if Dk.km_grid:
            fi_k, fj_k = (xy[:, 1] - Dk.lat[0]) / (Dk.lat[1] - Dk.lat[0]), (xy[:, 0] - Dk.lon[0]) / (Dk.lon[1] - Dk.lon[0])
        else:
            fi_k, fj_k = (lat - Dk.lat[0]) / (Dk.lat[1] - Dk.lat[0]), (lon - Dk.lon[0]) / (Dk.lon[1] - Dk.lon[0])
        Dk.pt_sample = torch.tensor(np.stack([fj_k / (Dk.W - 1) * 2 - 1, fi_k / (Dk.H - 1) * 2 - 1], -1),
                                    dtype=torch.float32, device=dev)
        Dk.pt_terrain = torch.tensor((tval - Dk.st_mu) / Dk.st_sd, dtype=torch.float32, device=dev)
        sdata[grp] = (Dk, mi)
        sfx = "" if grp == "ecmwf" else f"_{grp}"
        for kind in SKINDS:
            for s in range(3):
                for f in range(5):
                    m = tso.TerrainNet(kind, Dk).to(dev)
                    m.load_state_dict(torch.load(tso.RUNS / f"{kind}_terrain_g{f}_s{s}{sfx}_lr1e-05" / "best.pt",
                                                 weights_only=True))
                    smodels.setdefault(f"{kind}_{grp}", []).append((s, m.eval()))
        print(f"station models loaded: {grp} {Dk.M} models, grid {Dk.H} x {Dk.W} x {Dk.C} fields", flush=True)

    # ------------------------------------------------------------------ point geometry for inputs
    lat_c, lon_c = latg, long_
    fi, fj = (lat - lat_c[0]) / (lat_c[1] - lat_c[0]), (lon - lon_c[0]) / (lon_c[1] - lon_c[0])
    i0, j0 = np.floor(fi).astype(int), np.floor(fj).astype(int)
    wa, wb = fi - i0, fj - j0
    Wb = np.stack([(1 - wa) * (1 - wb), (1 - wa) * wb, wa * (1 - wb), wa * wb], -1)
    II, JJ = np.stack([i0, i0, i0 + 1, i0 + 1], -1), np.stack([j0, j0 + 1, j0, j0 + 1], -1)
    gmm = np.load(f"{ROOT}/cache/multimodel_{TAG}_grid.npy", mmap_mode="r")      # (N, L, 2, C, 49, 121)
    lmm = np.load(f"{ROOT}/cache/multimodel_{TAG}_lam_grid.npy", mmap_mode="r")  # (N, L, 2, C, Hr, Wr)
    lf = [str(x) for x in dr["fields"]]
    ri = np.rint((xy[:, 1] - gy[0]) / (gy[1] - gy[0])).astype(int)
    rj = np.rint((xy[:, 0] - gx[0]) / (gx[1] - gx[0])).astype(int)
    grad1, grad10, relief = terrain_at_stations(np.stack([ti, tj], -1))

    # ------------------------------------------------------------------ station check geometry
    s_static = xr.open_dataset(MN / "static.nc")
    mn_xy = to_plane(s_static["longitude"].values.astype(float), s_static["latitude"].values.astype(float))
    dS, kS = cKDTree(mn_xy.reshape(-1, 2)).query(st_xy)
    st_in = dS < 1.0
    sy, sxx = np.unravel_index(kS[st_in], mn_xy.shape[:2])
    ti_obs = pd.Index(dg["time"]).get_indexer(D.inits)
    obs_st = np.stack([dg["obs"][ti_obs + ld] for ld in D.leads], 1)[:, :, st_in]   # (N, L, S_in, 2)
    print(f"station check: {st_in.sum()} stations inside the MET Nordic grid", flush=True)

    # ------------------------------------------------------------------ loop over test runs
    test = D.split["test"]
    for Dk, _ in sdata.values():
        test = np.intersect1d(test, Dk.split["test"])
    if a.max_inits:
        test = test[:a.max_inits]
    if a.case_init:
        test = test[pd.DatetimeIndex(D.inits[test]) == pd.Timestamp(a.case_init)]
        assert len(test) == 1, f"{a.case_init} is not a test run"
    names = (["raw_ifs", "raw_aifs", "raw_meps", "raw_arome", "ifs_aifs", "meps_arome", "four_mean", "mlp", "sfno_mlp"]
             + list(smodels) + list(cfgs))
    seeds = {k: 3 for k in ["mlp", "sfno_mlp"] + list(smodels)}
    seeds.update({k: len(v) for k, v in dual_seeds.items()})
    se_pt = {k: np.zeros(P) for k in names}
    se_lead = {k: np.zeros(D.L) for k in names}
    se_seed = {k: np.zeros(v) for k, v in seeds.items()}
    cnt_pt, cnt_lead = np.zeros(P), np.zeros(D.L)
    se_sample = {k: np.full((len(test), D.L), np.nan, np.float32) for k in names}
    st_se = {k: 0.0 for k in ("metnordic", "four_mean", "meps_arome", "ifs_aifs")}
    st_cnt = 0
    case = None
    L = D.L
    for it, n in enumerate(test):
        times = [D.inits[n] + pd.Timedelta(hours=int(ld)) for ld in D.leads]
        full = [mn_wind(t) for t in times]
        if all(w is None for w in full):
            continue
        full = np.stack([w if w is not None else np.full(mn_xy.shape[:2] + (2,), np.nan, np.float32) for w in full])
        Y = full[:, yy, xx]                                                     # (L, P, 2)
        # ---- 10 m wind of the four models at the points
        gw = np.asarray(gmm[n][:, :, :2], np.float32)                           # (L, 2, 2, 49, 121)
        wg = (gw[..., II, JJ] * Wb).sum(-1)                                     # (L, 2, 2, P)
        rw = np.asarray(lmm[n][:, :, [lf.index("10u"), lf.index("10v")]], np.float32)[..., ri, rj]  # (L, 2, 2, P)
        fc = np.concatenate([wg, rw], 1).transpose(0, 1, 3, 2)                  # (L, 4, P, 2)
        base = fc.mean(1)                                                       # (L, P, 2)
        # ---- physics features
        glob = global_at([D.inits[n]], [int(x) for x in D.leads], latg, long_, lon, lat)
        sp = glob["sp"] / 100.0 if np.nanmedian(glob["sp"]) > 2000 else glob["sp"]
        reg = regional_at(dr, [n], xy, mm=lmm)                                  # (1, L, 4, P)
        t2r, mslr, tcc, blh = (reg[:, :, q] for q in range(4))
        feats = features(glob, sp, t2r, mslr, tcc, blh, base[None, ..., 0], base[None, ..., 1],
                         np.array([times], dtype="datetime64[ns]"), lon, lat, elev, grad1, grad10, relief,
                         tval, tnames)
        phys = np.stack([np.broadcast_to(feats[q], (1, L, P)) for q in phn], -1)[0]          # (L, P, 22)
        phys_z = np.nan_to_num((phys - ph_mu) / ph_sd).astype(np.float32)
        f8 = fc.transpose(0, 2, 1, 3).reshape(L, P, 8)
        f8_z = np.nan_to_num((f8 - f8_mu) / f8_sd).astype(np.float32)
        nn_, ll = np.full(L, n), np.arange(L)
        xg, xr_, tf, _, uv = D.batch(nn_, ll)
        pt0 = torch.cat([static_z[None].expand(L, -1, -1), D._t(f8_z)], -1)
        phys_t = D._t(phys_z)
        preds = {"ifs_aifs": fc[:, :2].mean(1), "meps_arome": fc[:, 2:].mean(1), "four_mean": base,
                 "raw_ifs": fc[:, 0], "raw_aifs": fc[:, 1], "raw_meps": fc[:, 2], "raw_arome": fc[:, 3]}
        uvb = D._t(base)
        with torch.no_grad():
            for name, ms in models.items():
                sl = dual_seeds[name]
                acc = np.zeros((len(sl), L, P, 2))
                for s, m in ms:
                    acc[sl.index(s)] += m(xg if m.gch is None else xg[:, m.gch], xr_ if m.rch is None else xr_[:, m.rch], tf, torch.cat([pt0, phys_t[..., m.phys_idx]], -1) if m.phys_idx else pt0, uvb).float().cpu().numpy() * D.y_std / 5
                for i in range(len(sl)):
                    se_seed[name][i] += np.nansum(np.sum((base + acc[i] - Y) ** 2, -1))
                preds[name] = base + acc.mean(0)
            # station models per group: dyn = the group's winds, mean speed, upwind sx, sx10
            for grp, (Dk, mi) in sdata.items():
                fcg = fc[:, mi]
                bg = fcg.mean(1)                                                # (L, P, 2)
                u, v = bg[..., 0], bg[..., 1]
                dyn = np.concatenate([fcg.transpose(0, 2, 1, 3).reshape(L, P, -1), np.hypot(u, v)[..., None],
                                      np.stack([upwind(sx[p], u, v) for p in ("sx", "sx10")], -1)], -1)
                dyn = torch.tensor(((dyn - Dk.dyn_mu) / Dk.dyn_sd).astype(np.float32), device=dev)
                ptm = torch.cat([dyn, Dk.pt_terrain[None].expand(L, -1, -1)], -1)
                tfm = torch.tensor(Dk.time_features(nn_, ll), dtype=torch.float32, device=dev)
                gr = np.asarray(Dk.grid[n][:, :, Dk.fsel], np.float32).reshape(L, -1, Dk.H, Dk.W)
                g = torch.tensor((gr - Dk.g_mean) / Dk.g_std, dtype=torch.float32, device=dev)
                for kind in SKINDS:
                    key = f"{kind}_{grp}"
                    acc = np.zeros((3, L, P, 2))
                    for s, m in smodels[key]:
                        acc[s] += m.head_at(m.latent(g, tfm), tfm, ptm, Dk.pt_sample).float().cpu().numpy() * Dk.y_std / 5
                    for s in range(3):
                        se_seed[key][s] += np.nansum(np.sum((bg + acc[s] - Y) ** 2, -1))
                    preds[key] = bg + acc.mean(0)
            preds["mlp"], preds["sfno_mlp"] = preds["point_mix"], preds["sfno_mix"]
        ok_ = np.isfinite(Y).all(-1)                                            # (L, P)
        for k in names:
            e = np.where(ok_, np.sum((preds[k] - Y) ** 2, -1), 0.0)
            se_pt[k] += e.sum(0)
            se_lead[k] += e.sum(1)
            se_sample[k][it] = np.where(ok_.any(1), e.sum(1) / np.maximum(ok_.sum(1), 1), np.nan)
        cnt_pt += ok_.sum(0)
        cnt_lead += ok_.sum(1)
        # ---- station check: MET Nordic at the station cells vs the observations
        mn_st = full[:, sy, sxx]                                                # (L, S_in, 2)
        o = obs_st[n]
        bst = D.base[n][:, st_in]
        fst = fc_st[n][:, :, st_in]
        okst = np.isfinite(o).all(-1) & np.isfinite(mn_st).all(-1)
        for key, f in (("metnordic", mn_st), ("four_mean", bst), ("meps_arome", fst[:, 2:].mean(1)),
                       ("ifs_aifs", fst[:, :2].mean(1))):
            st_se[key] += float(np.sum(np.sum((f - o) ** 2, -1)[okst]))
        st_cnt += int(okst.sum())
        # ---- one case map: the run with the strongest mean MET Nordic wind at +24 h
        j24 = list(D.leads).index(24)
        spd = float(np.nanmean(np.linalg.norm(Y[j24], axis=-1)))
        if case is None or spd > case["speed"]:
            case = dict(speed=spd, init=str(D.inits[n]), truth=Y[j24].copy(),
                        **{f"pred_{k}": preds[k][j24].astype(np.float32) for k in names})
        if it % 20 == 0:
            rm = {k: np.sqrt(se_pt[k].sum() / cnt_pt.sum()) for k in names}
            print(f"{it}/{len(test)} {D.inits[n]} {time.time() - t0:.0f} s  " +
                  "  ".join(f"{k} {v:.3f}" for k, v in rm.items()), flush=True)

    se_seed["mlp"], se_seed["sfno_mlp"] = se_seed["point_mix"].copy(), se_seed["sfno_mix"].copy()
    out = ROOT / "results" / "metnordic"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / a.out, names=np.array(names), dual_seeds=json.dumps(dual_seeds), lat=lat, lon=lon, xy=xy, elev=elev,
                        dist_station=dist_st, mn_y=yy, mn_x=xx, step=a.step, leads=D.leads,
                        test_inits=np.array(D.inits[test], dtype="datetime64[h]"),
                        cnt_pt=cnt_pt, cnt_lead=cnt_lead, st_xy=st_xy, st_in=st_in, st_cnt=st_cnt,
                        **{f"se_pt_{k}": v for k, v in se_pt.items()},
                        **{f"se_lead_{k}": v for k, v in se_lead.items()},
                        **{f"se_sample_{k}": v for k, v in se_sample.items()},
                        **{f"se_seed_{k}": v for k, v in se_seed.items()},
                        **{f"st_se_{k}": v for k, v in st_se.items()},
                        **{f"case_{k}": v for k, v in case.items()})
    tot = cnt_pt.sum()
    print("\nvector RMSE vs MET Nordic (m/s), all points / leads:")
    for k in names:
        extra = (" seeds " + " ".join(f"{np.sqrt(x / tot):.3f}" for x in se_seed[k])) if k in se_seed else ""
        print(f"  {k:22s} {np.sqrt(se_pt[k].sum() / tot):.3f}{extra}")
    print("station check (RMSE vs obs, m/s):", {k: round(np.sqrt(v / st_cnt), 3) for k, v in st_se.items()})
    print("->", out / a.out, f"{time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
