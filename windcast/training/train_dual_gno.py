"""Dual-branch neural operator with a dual-scale GNO fusion decoder (track A).

Global forecasts (IFS + AIFS, 0.25 deg) and regional forecasts (MEPS + AROME-Arctic, 2.5 km)
are each encoded on their native grid by a neural operator; a GNO per branch aggregates the
latent field around the target point x within its own radius (r_g, r_r); the two GNO features
are concatenated with the terrain, the local forecast winds, the time features and, with
--phys, the potential-temperature / mountain-wind features of scripts/build_station_physics.py,
and a single linear output projection gives the residual against the four-model mean:

    Z_g = E_g(X_g, tau)                      Z_r = E_r(X_r, tau)            (B, 32, H, W)
    h_g(x) = GNO_g(Z_g, x; r_g)              h_r(x) = GNO_r(Z_r, x; r_r)    (B, P, 32)
    h = [h_g, h_r, T(x), f(x, t), tau(t) (, phys(x, t))]
    du(x, t) = W_out h + b_out
    u(x, t) = mean(IFS, AIFS, MEPS, AROME-Arctic)(x, t) + du(x, t)

GNO: GNO(Z, x) = sum_{y in N(x)} k((y - x) / r, Z(y)) * Z(y) / |N(x)|, with N(x) the grid points
within r of x in the windcast LAEA plane (km). The kernel k (MLP) sees only the displacement and
the source features, never absolute coordinates, so the model has no station identity and the
same weights apply at any point. Each branch is averaged over its own neighbours (separate
normalisation). Neighbour lists are fixed (fixed grids and stations) and computed once.
--wind-frame rotates the displacement into (along, cross) the four-model mean 10 m wind at x
(unrotated where the wind is below 1 m/s). --readout bilinear replaces both GNOs by bilinear
sampling of the latent fields at x (control). --branches global|regional drops one branch.
--models ifs,aifs,meps,arome (any subset) restricts everything to those models: the encoder inputs,
the point forecast winds and the baseline (the residual is against the mean of the chosen models);
a branch without chosen models is dropped. Not combined with --phys (built from the four-model mean).

Encoder inputs: u10, v10, t2m, MSLP of both models of the branch (z-scored, train statistics) +
5 time channels (lead / 66, hour and day-of-year sin / cos of the valid time). No coordinate
channels, no observations.

Training: as train_station_operator.py --inputs terrain --setting excluding: spatial station
groups, --fold k is held out; loss = MSE of the normalised vector residual at the valid
training stations; AdamW + OneCycleLR; the best epoch on the validation period (held-out group)
is kept. Runs where any of the four models is missing are left out.

    python -m windcast.training.train_dual_gno --fold 0 --seed 0
    python -m windcast.training.train_dual_gno --fold 0 --phys all --rg 60 --rr 10

Writes RUNS/<run>/pred.npz: val / test residual (m/s) against the four-model mean for the
held-out group (NaN elsewhere), base (the four-model mean, N, L, S, 2), plus history.json.
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.spatial import cKDTree

from windcast.geo import to_plane
from windcast.training.train_station_operator import RUN_SFX, SPLIT, TAG, build_grid_model, groups_for

ROOT = Path("/cluster/work/projects/nn8106k/siyan/windcast")
DATA_G = ROOT / "cache" / f"multimodel_{TAG}"           # IFS + AIFS, 0.25 deg
DATA_R = ROOT / "cache" / f"multimodel_{TAG}_lam"       # MEPS + AROME-Arctic, 2.5 km
TERRAIN = ROOT / "cache" / "station_terrain.npz"
PHYS = ROOT / "cache" / f"station_physics_{TAG}.npz"
RUNS = ROOT / f"runs_dual{RUN_SFX}"
FIELDS = ["10u", "10v", "2t", "msl"]
MODELS = ["ifs", "aifs", "meps", "arome"]                 # the order of the four-model samples
FIELDS6 = ["10u", "10v", "2t", "msl", "sp", "t850"]       # --fields6: the same six fields for all four models
G_ENCODERS = ["sfno", "fno", "tfno", "uno"]
R_ENCODERS = ["fno", "tfno", "uno", "localno"]


class DualData:
    def __init__(self, device, phys="none", stations="base", models=tuple(MODELS), fields6=False, use_terrain=True):
        # stations 'ext': + the Swedish / Finnish stations (build_multimodel_samples.py --nordic,
        # build_lam_samples.py --stations ext, ...); their spatial groups are 5..7 (groups_for)
        sfx = "" if stations == "base" else f"_{stations}"
        data_g = DATA_G.with_name(DATA_G.name + sfx)
        data_r = DATA_G.with_name(DATA_G.name + sfx + ("_lam6" if fields6 else "_lam"))   # build_lam_samples.py [--fields6]
        self.fields = FIELDS6 if fields6 else FIELDS
        terrain = TERRAIN.with_name(f"station_terrain{sfx}.npz")
        phys_file = PHYS.with_name(PHYS.stem + sfx + ".npz")
        dg = dict(np.load(f"{data_g}.npz", allow_pickle=True))
        dr = dict(np.load(f"{data_r}.npz", allow_pickle=True))
        assert (dg["station"] == dr["station"]).all() and (dg["inits"] == dr["inits"]).all()
        self.device = device
        self.inits = pd.DatetimeIndex(dg["inits"])
        self.leads = dg["leads"]
        N, L, Mg, S, _ = dg["st_fc"].shape
        self.N, self.L, self.S = N, L, S
        keep = dg["keep"] & dr["keep"]
        assert [str(m) for m in dg["models"]] == ["ifs", "aifs"]
        assert [str(m) for m in dr["models"]] == ["meps", "arome_arctic"]
        self.models = [m for m in MODELS if m in models]
        assert self.models and len(self.models) == len(models), models
        self.msel_g = [i for i, m in enumerate(MODELS[:2]) if m in self.models]
        self.msel_r = [i for i, m in enumerate(MODELS[2:]) if m in self.models]

        # baseline (mean of the chosen models; same runs for every subset) and residual target
        fc = np.concatenate([dg["st_fc"], dr["st_fc"]], 2)[:, :, [MODELS.index(m) for m in self.models]]  # (N, L, M, S, 2)
        self.base = fc.mean(2)
        ti = pd.Index(dg["time"]).get_indexer(self.inits)
        o_valid = np.stack([dg["obs"][ti + ld] for ld in self.leads], 1)     # (N, L, S, 2)
        self.y = (o_valid - self.base).astype(np.float32)
        self.split = {k: np.where((self.inits >= a) & (self.inits <= pd.Timestamp(b) + pd.Timedelta(hours=12))
                                  & keep)[0] for k, (a, b) in SPLIT.items()}
        tr = self.split["train"]
        self.y_std = float(np.nanstd(self.y[tr]))

        # grids in memory (float16), z-scored per channel with train statistics
        self.Hg, self.Wg = len(dg["lat"]), len(dg["lon"])
        self.Hr, self.Wr = len(dr["grid_y"]), len(dr["grid_x"])
        self.grid_g = self.grid_r = None
        self.Cg = self.Cr = 0
        if self.msel_g:
            self.grid_g, self.gmu, self.gsd = self._grid(data_g, dg, tr, step=4, msel=self.msel_g)
            self.Cg = self.grid_g.shape[2] * self.grid_g.shape[3]
        if self.msel_r:
            self.grid_r, self.rmu, self.rsd = self._grid(data_r, dr, tr, step=16, msel=self.msel_r)
            self.Cr = self.grid_r.shape[2] * self.grid_r.shape[3]

        # geometry in the LAEA plane (km)
        LON, LAT = np.meshgrid(dg["lon"], dg["lat"])
        self.pos_g = to_plane(LON, LAT).reshape(-1, 2)                        # (Hg * Wg, 2)
        GX, GY = np.meshgrid(dr["grid_x"], dr["grid_y"])
        self.pos_r = np.stack([GX, GY], -1).reshape(-1, 2).astype(np.float64)
        self.st_xy = dg["xy"].astype(np.float64)                              # (S, 2)
        lat, lon = dg["lat"], dg["lon"]
        fi = (dg["lat_st"] - lat[0]) / (lat[1] - lat[0])
        fj = (dg["lon_st"] - lon[0]) / (lon[1] - lon[0])
        self.sample_g = self._t(np.stack([fj / (self.Wg - 1) * 2 - 1, fi / (self.Hg - 1) * 2 - 1], -1))
        gx, gy = dr["grid_x"], dr["grid_y"]
        fi = (self.st_xy[:, 1] - gy[0]) / (gy[-1] - gy[0])
        fj = (self.st_xy[:, 0] - gx[0]) / (gx[-1] - gx[0])
        self.sample_r = self._t(np.stack([fj * 2 - 1, fi * 2 - 1], -1))
        self.groups = groups_for(dg)
        self.n_groups = int(self.groups.max()) + 1
        self.station = dg["station"]

        # point features: terrain (S, 28) z-scored over the stations; the 8 forecast winds;
        # the physical features (z-scored with train statistics)
        t = np.load(terrain, allow_pickle=True)
        assert (t["station"] == dg["station"]).all()
        val = t["values"].astype(np.float64)
        self.pt_static = self._t((val - val.mean(0)) / (val.std(0) + 1e-9))
        if not use_terrain:                                                   # --no-terrain ablation
            self.pt_static = self.pt_static[:, :0]
        f8 = fc.transpose(0, 1, 3, 2, 4).reshape(N, L, S, -1)                 # (N, L, S, 2M)
        self.pt_fc_mu, self.pt_fc_sd = np.nanmean(f8[tr], (0, 1, 2)), np.nanstd(f8[tr], (0, 1, 2)) + 1e-9
        self.pt_fc = self._z(f8, tr)
        self.uv_mean = self.base.astype(np.float32)                           # for --wind-frame
        self.phys_names = []
        if phys != "none":
            assert len(self.models) == 4, "--phys is built from the four-model mean"
            p = np.load(phys_file, allow_pickle=True)
            assert (p["station"] == dg["station"]).all()
            names = [str(x) for x in p["names"]]
            sel = {"theta": [str(x) for x in p["theta"]], "mountain": [str(x) for x in p["mountain"]],
                   "all": names}[phys]
            self.phys_names = sel
            self.pt_phys = self._z(p["values"][..., [names.index(k) for k in sel]], tr)
        self.n_point = self.pt_static.shape[1] + 2 * len(self.models) + 5 + len(self.phys_names)

    def _t(self, a):
        return torch.tensor(np.asarray(a), dtype=torch.float32, device=self.device)

    @staticmethod
    def _z(a, tr):
        mu, sd = np.nanmean(a[tr], (0, 1, 2)), np.nanstd(a[tr], (0, 1, 2)) + 1e-9
        return np.nan_to_num((a - mu) / sd).astype(np.float32)

    def _grid(self, stem, d, tr, step, msel):
        mm = np.load(f"{stem}_grid.npy", mmap_mode="r")                      # (N, L, M, C, H, W)
        names = [str(x) for x in d.get("fields", FIELDS)]
        fsel = [names.index(f) for f in self.fields]
        t0 = time.time()
        arr = np.empty(mm.shape[:2] + (len(msel), len(fsel)) + mm.shape[4:], np.float16)
        for i in range(0, len(mm), 32):
            arr[i:i + 32] = mm[i:i + 32][:, :, msel][:, :, :, fsel]
        g = arr[tr[::step]].astype(np.float32)
        mu = g.mean((0, 1, 4, 5)).reshape(1, -1, 1, 1)                        # (1, M * C, 1, 1)
        sd = g.std((0, 1, 4, 5)).reshape(1, -1, 1, 1) + 1e-6
        print(f"{stem.name}: grid {arr.shape} in memory, {arr.nbytes / 1e9:.1f} GB, {time.time() - t0:.0f} s",
              flush=True)
        return arr, mu, sd

    def time_features(self, n, l):
        vt = self.inits[n] + pd.to_timedelta(self.leads[l], unit="h")
        return np.stack([self.leads[l] / 66.0, np.sin(2 * np.pi * vt.hour / 24), np.cos(2 * np.pi * vt.hour / 24),
                         np.sin(2 * np.pi * vt.dayofyear / 366), np.cos(2 * np.pi * vt.dayofyear / 366)], -1)

    def batch(self, n, l):
        """Global grid (B, Cg, Hg, Wg), regional grid (B, Cr, Hr, Wr), time (B, 5), point
        features (B, S, n_point - 5) and the mean 10 m wind at the stations (B, S, 2)."""
        B = len(n)
        xg = xr = None
        if self.grid_g is not None:
            xg = self._t((np.asarray(self.grid_g[n, l], np.float32).reshape(B, -1, self.Hg, self.Wg) - self.gmu) / self.gsd)
        if self.grid_r is not None:
            xr = self._t((np.asarray(self.grid_r[n, l], np.float32).reshape(B, -1, self.Hr, self.Wr) - self.rmu) / self.rsd)
        pt = [self.pt_static[None].expand(B, -1, -1), self._t(self.pt_fc[n, l])]
        if self.phys_names:
            pt.append(self._t(self.pt_phys[n, l]))
        return (xg, xr, self._t(self.time_features(n, l)), torch.cat(pt, -1),
                self._t(np.nan_to_num(self.uv_mean[n, l])))


class GNO(torch.nn.Module):
    """Kernel integral from a fixed source grid to fixed query points:
    out(x) = mean_{y in N(x)} k((y - x) / r, f(y)) * f(y), N(x) = {y : |y - x| <= r}."""

    def __init__(self, src, qry, radius, channels=32, hidden=64, wind_frame=False):
        super().__init__()
        self.radius, self.wind_frame = radius, wind_frame
        self.kernel = torch.nn.Sequential(torch.nn.Linear(3 + channels, hidden), torch.nn.GELU(),
                                          torch.nn.Linear(hidden, hidden), torch.nn.GELU(),
                                          torch.nn.Linear(hidden, channels))
        self.set_queries(src, qry)

    def set_queries(self, src, qry):
        """Neighbour lists of the query points qry (P, 2) km in the source grid src (n, 2) km,
        padded to the largest neighbourhood."""
        nb = cKDTree(src).query_ball_point(qry, self.radius)
        cnt = np.array([len(a) for a in nb])
        assert cnt.min() > 0, f"{(cnt == 0).sum()} query points without neighbours within {self.radius} km"
        K = int(cnt.max())
        idx = np.zeros((len(qry), K), np.int64)
        mask = np.zeros((len(qry), K), np.float32)
        for p, a in enumerate(nb):
            idx[p, :len(a)], mask[p, :len(a)] = a, 1.0
        rel = (src[idx] - qry[:, None]) / self.radius                         # (P, K, 2)
        dev = next(self.parameters()).device if len(list(self.parameters())) else None
        mk = lambda a, dt: torch.tensor(a, dtype=dt, device=dev)
        self.register_buffer("idx", mk(idx, torch.long), persistent=False)
        self.register_buffer("mask", mk(mask, torch.float32), persistent=False)
        self.register_buffer("rel", mk(rel, torch.float32), persistent=False)
        self.n_neighbours = (int(cnt.min()), float(cnt.mean()), K)

    def forward(self, f, wind=None):
        """f (B, n, C) source features, wind (B, P, 2) for --wind-frame -> (B, P, C)."""
        B = f.shape[0]
        fy = f[:, self.idx]                                                   # (B, P, K, C)
        rel = self.rel[None].expand(B, -1, -1, -1)
        if self.wind_frame:
            sp = wind.norm(dim=-1, keepdim=True)                              # (B, P, 1)
            e = torch.where(sp > 1.0, wind / sp.clamp(min=1e-6), torch.tensor([0.0, 1.0], device=f.device))
            ea, ec = e[:, :, None], torch.stack([-e[..., 1], e[..., 0]], -1)[:, :, None]
            rel = torch.stack([(rel * ea).sum(-1), (rel * ec).sum(-1)], -1)   # (along, cross)
        k = self.kernel(torch.cat([rel, rel.norm(dim=-1, keepdim=True), fy], -1))
        m = self.mask[None, :, :, None]
        return (k * fy * m).sum(2) / m.sum(2)


class DualGNO(torch.nn.Module):
    def __init__(self, D: DualData, enc_g="sfno", enc_r="fno", width=64, width_r=64, modes_g=(16, 32),
                 modes_r=(16, 32), layers=4, latent=32, rg=60.0, rr=10.0, branches="both",
                 readout="gno", wind_frame=False, dropout=0.0, zero_init=False, head="linear", head_hidden=128):
        super().__init__()
        self.D, self.branches, self.readout = D, branches, readout
        self.use_g, self.use_r = branches in ("both", "global"), branches in ("both", "regional")
        n_feat = D.n_point
        if self.use_g:
            self.enc_g = build_grid_model(enc_g, D.Cg + 5, D.Hg, D.Wg, width, modes_g, layers, n_out=latent)
            if readout == "gno":
                self.gno_g = GNO(D.pos_g, D.st_xy, rg, latent, wind_frame=wind_frame)
            n_feat += latent
        if self.use_r:
            self.enc_r = build_grid_model(enc_r, D.Cr + 5, D.Hr, D.Wr, width_r, modes_r, layers, n_out=latent)
            if readout == "gno":
                self.gno_r = GNO(D.pos_r, D.st_xy, rr, latent, wind_frame=wind_frame)
            n_feat += latent
        if head == "linear":                                                  # PDF: single linear projection
            self.proj = torch.nn.Linear(n_feat, 2)
        else:                                                                 # control: same MLP head as TerrainNet
            hh = head_hidden
            self.proj = torch.nn.Sequential(torch.nn.Linear(n_feat, hh), torch.nn.GELU(),
                                            torch.nn.Linear(hh, hh), torch.nn.GELU(), torch.nn.Linear(hh, 2))
        if zero_init:                                                         # start at the four-model mean
            last = self.proj if head == "linear" else self.proj[-1]
            torch.nn.init.zeros_(last.weight)
            torch.nn.init.zeros_(last.bias)
        self.drop = torch.nn.Dropout(dropout)                                 # on the operator latents only

    @staticmethod
    def encode(enc, x, tf, H, W):
        z = enc(torch.cat([x, tf[:, :, None, None].expand(-1, -1, H, W)], 1))
        if z.shape[-2:] != (H, W):
            z = F.interpolate(z, size=(H, W), mode="bilinear", align_corners=True)
        return z                                                              # (B, latent, H, W)

    def aggregate(self, z, gno, sample, wind):
        B = z.shape[0]
        if self.readout == "bilinear":
            return F.grid_sample(z, sample[None, None].expand(B, -1, -1, -1), align_corners=True,
                                 padding_mode="border")[:, :, 0].permute(0, 2, 1)
        return gno(z.flatten(2).transpose(1, 2), wind)                        # (B, S, latent)

    def forward(self, xg, xr, tf, pt, wind):
        D = self.D
        h = []
        if self.use_g:
            h.append(self.drop(self.aggregate(self.encode(self.enc_g, xg, tf, D.Hg, D.Wg), getattr(self, "gno_g", None),
                                    D.sample_g, wind)))
        if self.use_r:
            h.append(self.drop(self.aggregate(self.encode(self.enc_r, xr, tf, D.Hr, D.Wr), getattr(self, "gno_r", None),
                                    D.sample_r, wind)))
        B, S = pt.shape[:2]
        h += [pt, tf[:, None].expand(-1, S, -1)]
        return self.proj(torch.cat(h, -1))                                    # (B, S, 2), units of y_std


def run_epoch(model, D, idx, fold, train, opt=None, sched=None, bs=16, rng=None):
    model.train(train)
    tot, cnt = 0.0, 0
    allowed = D.groups != fold
    order = rng.permutation(len(idx)) if train else np.arange(len(idx))
    for b0 in range(0, len(idx), bs):
        sel = idx[order[b0:b0 + bs]]
        n, l = sel // D.L, sel % D.L
        yv = np.isfinite(D.y[n, l]).all(-1)
        tgt = yv & (allowed if train else D.groups == fold)
        if tgt.sum() == 0:
            continue
        y = torch.tensor(np.nan_to_num(D.y[n, l]) / D.y_std, device=D.device)
        w = torch.tensor(tgt, dtype=torch.float32, device=D.device)
        with torch.set_grad_enabled(train):
            pred = model(*D.batch(n, l))
            loss = (((pred - y) ** 2).sum(-1) * w).sum() / w.sum()
        if train:
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
        tot += loss.item() * w.sum().item()
        cnt += w.sum().item()
    return tot / max(cnt, 1)


@torch.no_grad()
def predict(model, D, inits_idx, fold, bs=16):
    model.eval()
    out = np.full((D.N, D.L, D.S, 2), np.nan, np.float32)
    idx = (inits_idx[:, None] * D.L + np.arange(D.L)[None]).reshape(-1)
    for b0 in range(0, len(idx), bs):
        sel = idx[b0:b0 + bs]
        n, l = sel // D.L, sel % D.L
        p = model(*D.batch(n, l)).float().cpu().numpy() * D.y_std
        p[:, D.groups != fold] = np.nan
        out[n, l] = p
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--enc-g", default="sfno", choices=G_ENCODERS)
    ap.add_argument("--enc-r", default="fno", choices=R_ENCODERS)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--width-r", type=int, default=64)
    ap.add_argument("--modes-r", type=int, nargs=2, default=[16, 32])
    ap.add_argument("--modes-g", type=int, nargs=2, default=[16, 32])
    ap.add_argument("--dropout", type=float, default=0.0, help="dropout on the operator latents")
    ap.add_argument("--wd", type=float, default=1e-4, help="AdamW weight decay")
    ap.add_argument("--head", default="linear", choices=["linear", "mlp"], help="output head (mlp = control)")
    ap.add_argument("--head-hidden", type=int, default=128, help="width of the two hidden layers of the MLP head")
    ap.add_argument("--zero-init", action="store_true", help="zero output projection (start at the mean)")
    ap.add_argument("--rg", type=float, default=60.0, help="global GNO radius (km)")
    ap.add_argument("--rr", type=float, default=10.0, help="regional GNO radius (km)")
    ap.add_argument("--branches", default="both", choices=["both", "global", "regional"])
    ap.add_argument("--readout", default="gno", choices=["gno", "bilinear"])
    ap.add_argument("--wind-frame", action="store_true", help="GNO displacement in (along, cross) the mean wind")
    ap.add_argument("--phys", default="none", choices=["none", "theta", "mountain", "all"],
                    help="potential-temperature / mountain-wind point features (build_station_physics.py)")
    ap.add_argument("--fields6", action="store_true",
                    help="encoder inputs: 10u, 10v, 2t, msl, sp, t850 of every model (default: the first four); name + _f6")
    ap.add_argument("--models", default=",".join(MODELS),
                    help="comma list, subset of ifs,aifs,meps,arome; run name + _m-<models> unless all four")
    ap.add_argument("--stations", default="base", choices=["base", "ext"],
                    help="ext: + the Swedish / Finnish stations (spatial groups 5..7); run name + _ext")
    ap.add_argument("--no-terrain", action="store_true",
                    help="ablation: drop the 28 terrain point features T(x); run name + _noterr")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--max-train", type=int, default=0, help="smoke test: limit samples")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    dev = torch.device("cuda")
    models = a.models.split(",")
    D = DualData(dev, phys=a.phys, stations=a.stations, models=models, fields6=a.fields6, use_terrain=not a.no_terrain)
    assert 0 <= a.fold < D.n_groups
    if len(D.models) < 4:                    # the branches follow the chosen models
        a.branches = "both" if D.msel_g and D.msel_r else ("global" if D.msel_g else "regional")

    name = f"dual_{a.enc_g}-{a.enc_r}_g{a.fold}_s{a.seed}"
    if a.branches != "both":
        name += f"_{a.branches}"
    if a.readout == "gno":
        name += (f"_rg{a.rg:g}" if a.branches != "regional" else "") + (f"_rr{a.rr:g}" if a.branches != "global" else "")
        name += "_wf" if a.wind_frame else ""
    else:
        name += "_bilinear"
    if a.phys != "none":
        name += f"_phys-{a.phys}"
    if a.width_r != 64 or a.modes_r != [16, 32]:
        name += f"_r{a.width_r}m{a.modes_r[0]}x{a.modes_r[1]}"
    if a.width != 64 or a.modes_g != [16, 32]:
        name += f"_g{a.width}m{a.modes_g[0]}x{a.modes_g[1]}"
    if a.dropout:
        name += f"_do{a.dropout:g}"
    if a.wd != 1e-4:
        name += f"_wd{a.wd:g}"
    if a.zero_init:
        name += "_z0"
    if a.head != "linear":
        name += f"_head-{a.head}"
        if a.head_hidden != 128:
            name += f"{a.head_hidden}"
    if a.stations != "base":
        name += f"_{a.stations}"
    if len(D.models) < 4:
        name += "_m-" + "-".join(D.models)
    if a.fields6:
        name += "_f6"
    if a.no_terrain:
        name += "_noterr"
    name += f"_lr{a.lr:g}"
    if a.max_train:
        name += "_smoke"
    out = RUNS / name
    out.mkdir(parents=True, exist_ok=True)

    model = DualGNO(D, a.enc_g, a.enc_r, width=a.width, width_r=a.width_r, modes_g=tuple(a.modes_g), modes_r=tuple(a.modes_r),
                    dropout=a.dropout, zero_init=a.zero_init, head=a.head, head_hidden=a.head_hidden,
                    rg=a.rg, rr=a.rr, branches=a.branches, readout=a.readout, wind_frame=a.wind_frame).to(dev)
    n_par = sum(p.numel() for p in model.parameters())
    nb = {k: getattr(model, k).n_neighbours for k in ("gno_g", "gno_r") if hasattr(model, k)}
    print(f"{name}: {n_par / 1e6:.2f} M parameters, {D.S} stations, point features {D.n_point} "
          f"(phys {D.phys_names}), neighbours (min, mean, max) {nb}, "
          f"train/val/test runs {[len(D.split[k]) for k in SPLIT]}", flush=True)

    def samples(k):
        i = D.split[k]
        return (i[:, None] * D.L + np.arange(D.L)[None]).reshape(-1)
    tr, va = samples("train"), samples("val")
    if a.max_train:
        tr, va = tr[:a.max_train], va[:a.max_train]
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * math.ceil(len(tr) / a.bs) + 10)
    hist, best, best_state = [], np.inf, None
    for ep in range(a.epochs):
        t0 = time.time()
        trl = run_epoch(model, D, tr, a.fold, True, opt, sched, bs=a.bs, rng=rng)
        val = run_epoch(model, D, va, a.fold, False, bs=a.bs, rng=rng)
        hist.append({"epoch": ep, "train": trl, "val": val, "sec": time.time() - t0})
        print(f"ep {ep:3d} train {trl:.4f} val {val:.4f} ({time.time() - t0:.0f} s)", flush=True)
        if val < best:
            best = val
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items() if torch.is_tensor(v)}
            torch.save(best_state, out / "best.pt")
    model.load_state_dict(best_state)
    pv = predict(model, D, D.split["val"], a.fold, bs=a.bs)
    pt = predict(model, D, D.split["test"], a.fold, bs=a.bs)
    np.savez_compressed(out / "pred.npz", val=pv, test=pt, base=D.base.astype(np.float32), groups=D.groups,
                        station=D.station, base_models=np.array(D.models), y_std=D.y_std)
    (out / "history.json").write_text(json.dumps({"args": vars(a), "n_params": n_par, "best_val": best,
                                                  "phys": D.phys_names, "neighbours": nb,
                                                  "history": hist}, indent=1))
    print("best val", best, "->", out)


if __name__ == "__main__":
    main()
