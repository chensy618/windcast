"""Applies the A2 terrain models on the 1 km grid of northern Norway (GPU).

    python scripts/predict_terrain_grid.py --run sfno_terrain
    python scripts/predict_terrain_grid.py --run sfno_terrainobs

--run is a run prefix in runs_a1 (<run>_g<k>_s0, k = 0..4, trained by
windcast.training.train_station_operator with --inputs terrain / both).

Every input is built at the 1 km cell itself:
  * the IFS and AIFS 10 m wind, bilinear from the 0.25 deg grid to the cell;
  * the 28 descriptors of the cell (scripts/build_terrain_descriptors.py), z-scored with the
    station mean / std used in training, and the upwind shelter for the model wind there;
  * with --inputs both, the other stations' obs-forecast weighted to the cell.

Writes ROOT/results/a2/grid_<run>.npz:
  * maps for two test cases (the strongest observed station wind at +24 h, and a median one):
    base (u, v) at every cell, correction (mean of the 5 fold models), lat, lon, land, and the
    station positions, observations and groups;
  * a check of the gridded product at withheld stations, over the whole test period: the
    correction of the fold model that did not see the station, at the station's 1 km cell,
    added to the model wind at that cell, against the station observation.
  * if the extended station set exists (scripts/download_nordic_obs.py): the same at the
    Swedish / Finnish stations, none of which any of the 5 fold models saw (mean of the 5):
    nordic_corr, nordic_base (N, L, S', 2), nordic_obs, nordic_station, nordic_masl.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from windcast.station_correction import SX_DIRS, upwind
from windcast.training.train_station_operator import ROOT, RUNS, Data, TerrainNet

DESC = ROOT / "static" / "terrain_descriptors_1km.npz"


def cell_sample(D, lat, lon):
    """grid_sample coordinates (P, 2) of points in the 0.25 deg grid."""
    fi = (lat - D.lat[0]) / (D.lat[1] - D.lat[0])
    fj = (lon - D.lon[0]) / (D.lon[1] - D.lon[0])
    return np.stack([fj / (D.W - 1) * 2 - 1, fi / (D.H - 1) * 2 - 1], -1).astype(np.float32)


class Cells:
    """Static inputs at a set of 1 km cells."""

    def __init__(self, D, fields, names, lat, lon, xy):
        assert names == D.terrain_names, "descriptor order differs from training"
        self.static = ((fields - D.st_mu) / D.st_sd).astype(np.float32)        # (P, 28)
        self.sx = {p: fields[:, [names.index(f"{p}_{k}") for k in SX_DIRS]] for p in ("sx", "sx10")}
        self.sample = torch.tensor(cell_sample(D, lat, lon), device=D.device)
        self.pos = torch.tensor(xy, dtype=torch.float32, device=D.device)
        self.P = len(lat)


@torch.no_grad()
def predict_cells(models, D, C, n, l, in_mask, chunk=200_000):
    """Corrections (k, B, P, 2) of each model in `models` and the base wind (B, P, 2), for the
    samples (n, l) at the cells C."""
    B = len(n)
    g, tf, X = D.batch(n, l, in_mask)
    raw = torch.tensor(np.asarray(D.grid[n, l], dtype=np.float32)[:, :, :2].reshape(B, -1, D.H, D.W),
                       device=D.device)                                        # (B, M*2, H, W)
    wind = F.grid_sample(raw, C.sample[None, None].expand(B, -1, -1, -1), align_corners=True,
                         padding_mode="border")[:, :, 0].permute(0, 2, 1).cpu().numpy()   # (B, P, M*2)
    base = wind.reshape(B, C.P, D.M, 2).mean(2)
    u, v = base[..., 0], base[..., 1]
    dyn = np.concatenate([wind, np.hypot(u, v)[..., None],
                          np.stack([upwind(C.sx[p], u, v) for p in ("sx", "sx10")], -1)], -1)
    dyn = ((dyn - D.dyn_mu) / D.dyn_sd).astype(np.float32)
    st = X.get("st")
    out = np.zeros((len(models), B, C.P, 2), np.float32)
    for i, m in enumerate(models):
        lat = m.latent(g, tf, st)
        for a in range(0, C.P, chunk):
            b = min(a + chunk, C.P)
            pt = torch.tensor(np.concatenate([dyn[:, a:b], np.broadcast_to(C.static[a:b], (B, b - a, C.static.shape[1]))], -1),
                              device=D.device)
            out[i, :, a:b] = m.head_at(lat, tf, pt, C.sample[a:b], st, C.pos[a:b]).float().cpu().numpy() * D.y_std
    return out, base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="e.g. sfno_terrain or sfno_terrainobs")
    ap.add_argument("--bs", type=int, default=16)
    a = ap.parse_args()
    kind, inputs = a.run.split("_")
    use_obs = inputs == "terrainobs"
    dev = torch.device("cuda")
    D = Data(dev, terrain=True, use_obs=use_obs)
    models = []
    for k in range(5):
        m = TerrainNet(kind, D).to(dev)
        m.load_state_dict(torch.load(RUNS / f"{a.run}_g{k}_s0" / "best.pt", weights_only=True))
        models.append(m.eval())
    t = np.load(DESC, allow_pickle=True)
    names = [str(x) for x in t["names"]]
    H1, W1 = t["lat"].shape
    allc = Cells(D, t["fields"].reshape(len(names), -1).T, names, t["lat"].reshape(-1), t["lon"].reshape(-1),
                 t["xy"].reshape(-1, 2))
    st = np.load(ROOT / "cache" / "station_terrain.npz", allow_pickle=True)
    ij = st["ij"]
    stc = Cells(D, t["fields"][:, ij[:, 0], ij[:, 1]].T, names, t["lat"][ij[:, 0], ij[:, 1]],
                t["lon"][ij[:, 0], ij[:, 1]], t["xy"][ij[:, 0], ij[:, 1]])

    # --- the gridded product at withheld stations, whole test period ---------------------------
    te = D.split["test"]
    avail_all = np.isfinite(D.omf).all((1, 3))                                  # (N, S)
    corr = np.full((D.N, D.L, D.S, 2), np.nan, np.float32)
    base_cell = np.full((D.N, D.L, D.S, 2), np.nan, np.float32)
    idx = (te[:, None] * D.L + np.arange(D.L)[None]).reshape(-1)
    for k in range(5):
        allowed = D.groups != k
        for b0 in range(0, len(idx), a.bs):
            sel = idx[b0:b0 + a.bs]
            n, l = sel // D.L, sel % D.L
            out, base = predict_cells([models[k]], D, stc, n, l, avail_all[n] & allowed[None])
            gk = D.groups == k
            corr[n[:, None], l[:, None], np.where(gk)[0][None]] = out[0][:, gk]
            base_cell[n[:, None], l[:, None], np.where(gk)[0][None]] = base[:, gk]
        print("withheld group", k, "done", flush=True)

    # --- the gridded product at the Swedish / Finnish stations (never seen) --------------------
    nordic = {}
    ext = ROOT / "cache" / "multimodel_2024-03-01_2025-09-12_ext.npz"
    if ext.exists():
        e = np.load(ext, allow_pickle=True)
        assert (e["inits"] == np.array(D.inits, dtype="datetime64[h]")).all()
        new = e["country"] != "NO"
        te_ = np.load(ROOT / "cache" / "station_terrain_ext.npz", allow_pickle=True)
        ijn = te_["ij"][new]
        nc = Cells(D, t["fields"][:, ijn[:, 0], ijn[:, 1]].T, names, t["lat"][ijn[:, 0], ijn[:, 1]],
                   t["lon"][ijn[:, 0], ijn[:, 1]], t["xy"][ijn[:, 0], ijn[:, 1]])
        Sn = int(new.sum())
        ncorr = np.full((D.N, D.L, Sn, 2), np.nan, np.float32)
        nbase = np.full((D.N, D.L, Sn, 2), np.nan, np.float32)
        for b0 in range(0, len(idx), a.bs):
            sel = idx[b0:b0 + a.bs]
            n, l = sel // D.L, sel % D.L
            out, base = predict_cells(models, D, nc, n, l, avail_all[n])
            ncorr[n, l], nbase[n, l] = out.mean(0), base
        ti = pd.Index(e["time"]).get_indexer(np.array(D.inits, dtype="datetime64[h]"))
        nobs = np.stack([e["obs"][ti + ld][:, new] for ld in D.leads], 1)
        nordic = dict(nordic_corr=ncorr, nordic_base=nbase, nordic_obs=nobs.astype(np.float32),
                      nordic_station=e["station"][new], nordic_masl=e["masl"][new],
                      nordic_lat=e["lat_st"][new], nordic_lon=e["lon_st"][new])
        print("nordic stations done:", Sn, flush=True)

    # --- maps for two test cases (mean of the 5 fold models) ------------------------------------
    j24 = list(D.leads).index(24)
    spd = np.linalg.norm(D.o_valid[te, j24], axis=-1)
    mean_spd = np.nanmean(spd, 1)
    order = np.argsort(mean_spd)
    cases = [te[order[-1]], te[order[len(order) // 2]]]
    maps = {}
    for c, n_ in zip(("strong", "median"), cases):
        n, l = np.array([n_]), np.array([j24])
        out, base = predict_cells(models, D, allc, n, l, avail_all[n])
        maps[c] = dict(init=str(D.inits[n_]), lead=24, base=base[0].reshape(H1, W1, 2).astype(np.float32),
                       corr=out.mean(0)[0].reshape(H1, W1, 2).astype(np.float32),
                       corr_sd=out.std(0)[0].reshape(H1, W1, 2).astype(np.float32),
                       obs=D.o_valid[n_, j24].astype(np.float32))
        print("map", c, maps[c]["init"], flush=True)
    out_f = ROOT / "results" / "a2" / f"grid_{a.run}.npz"
    np.savez_compressed(out_f, lat=t["lat"], lon=t["lon"], land=t["fields"][names.index("land_fraction")],
                        elevation=t["fields"][names.index("elevation")],
                        st_lat=t["lat"][ij[:, 0], ij[:, 1]], st_lon=t["lon"][ij[:, 0], ij[:, 1]], groups=D.groups,
                        corr_withheld=corr, base_cell=base_cell, **nordic,
                        **{f"{c}_{k}": v for c, m in maps.items() for k, v in m.items()})
    print("->", out_f)


if __name__ == "__main__":
    main()
