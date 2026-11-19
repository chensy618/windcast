"""Multi-model station correction with neural operators (track A, A1).

Predicts the residual y = obs(t + tau) - mean(IFS, AIFS)(t + tau) at the stations from
  * the IFS and AIFS fields at the valid time on their common 0.25 deg grid, and
  * the obs-forecast of each model at init time t at the reporting stations
    (obs(t) - forecast of the run from t - 12 h at +12 h; only information available at
    issue time).

Operators (neuraloperator 2.0):
  fno     grid -> grid (FNO); station obs-forecast rasterised onto the grid; output sampled
          bilinearly at the stations
  fnogno  grid -> station points (FNOGNO); same grid input
  gino    station points -> latent grid -> station points (GINO); the model fields enter as
          latent features on the grid

Inputs (--inputs):
  obs      (default) model fields + station obs-forecast, with latitude / longitude channels
  terrain  no station identity at all: no coordinate channels and no obs-forecast. The operator
           sees only the IFS / AIFS fields and time and outputs 32 latent channels, read at the
           point bilinearly; a point-wise network maps them, together with that point's 1 km
           terrain descriptors (scripts/build_terrain_descriptors.py), the upwind shelter for
           the model wind direction and the model wind there, to the correction. Everything it
           uses exists at every 1 km cell, so it can be applied where there is no station;
           stations only supply training targets. --model point drops the operator (the
           point-wise network alone, with the time features). Only --setting excluding.
  both     terrain + the other stations' obs-forecast at init time, which also exists at any
           point: rasterised onto the operator grid (5 more channels, as in "obs") and, for the
           point-wise network, inverse-distance weighted to the point (1 / (d^2 + (10 km)^2);
           4 values; the weight sum itself is not an input, since it is nearly constant per
           station and lets the network recognise training stations). Each training sample hides one whole training
           group (spatial block) from the input and scores only that group, so the hidden
           stations are as far from the input stations as the held-out group is (hiding a
           random 25 % instead leaves them close neighbours, which the held-out group never has).

Two-level operator (--fine localno, with --inputs terrain): the coarse operator's 32 latent
  channels are read bilinearly at every 1 km cell of a P x P km patch around each station
  (P = --patch, default 33), together with the bilinearly interpolated model fields, the wind
  speed and upwind shelter (1 km and 10 km) of the mean IFS / AIFS wind, the time features and
  the 28 terrain descriptors of every 1 km cell at full resolution (z-scored over the 1 km grid).
  A LocalNO on the 1 km patch (local integral (DISCO, radius --disco-km) and differential
  kernels, few Fourier modes) maps them to the (u, v) correction; the station is the patch's
  centre cell. No point-wise network: the 1 km detail enters only through the operator. All
  kernels are in km, so the same LocalNO applies to the whole 1 km grid.

Settings:
  including  the target station's own obs-forecast is an input; temporal split only
  excluding  station groups (5, spatially blocked) are held out: --fold k trains on the other
             4 groups (each training sample hides a random 25 % of the available stations
             from the input and scores only those) and predicts group k from the others

    python -m windcast.training.train_station_operator --model fno --setting including
    python -m windcast.training.train_station_operator --model gino --setting excluding --fold 2

Writes RUNS/<run>/pred.npz: predicted residual (m/s) for every (init, lead, station) in val
and test (NaN elsewhere), plus history.json.
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

from windcast.station_correction import SX_DIRS, station_groups, upwind

ROOT = Path("/cluster/work/projects/nn8106k/siyan/windcast")
DATA = ROOT / "cache" / "multimodel_2024-03-01_2025-09-12"
RUNS = ROOT / "runs_a1"
TERRAIN = ROOT / "cache" / "station_terrain.npz"
GRID_FILE = ROOT / "cache" / "multimodel_2024-03-01_2025-09-12_grid.npy"
# 100 m descriptors at the stations, by source (--terrain100):
#   mtno   the station point, from the mtno 50 m DTM patches (windcast tmp/terrain100.py)
#   point  the station point, from the 100 m grid of scripts/build_terrain_100m.py
#   cell   the statistics of the station's 1 km cell on that grid (also exist at every 1 km cell)
#   both   point + cell
TERRAIN100 = ROOT / "cache" / "station_terrain_100m.npz"
TERRAIN_GRID = ROOT / "static" / "terrain_descriptors_1km.npz"   # 1 km fields, for --fine
T100_FILES = {"mtno": [("station_terrain_100m.npz", ("sx300", "sx1k"))],
              "point": [("station_terrain_p100.npz", ("sx300", "sx1k"))],
              "cell": [("station_terrain_c100.npz", ("c100_sx300", "c100_sx1k"))]}
T100_FILES["both"] = T100_FILES["point"] + T100_FILES["cell"]
# the same from the 300 m grid (build_terrain_100m.py --res 300; radii 900 m / 3 km)
T100_FILES["point300"] = [("station_terrain_p300.npz", ("sx900", "sx3k"))]
T100_FILES["cell300"] = [("station_terrain_c300.npz", ("c300_sx900", "c300_sx3k"))]
T100_FILES["both300"] = T100_FILES["point300"] + T100_FILES["cell300"]
T100_SUFFIX = {"mtno": "_t100", "point": "_p100", "cell": "_c100", "both": "_pc100",
               "point300": "_p300", "cell300": "_c300", "both300": "_pc300"}
SPLIT = {"train": ("2024-03-01", "2024-12-31"), "val": ("2025-01-01", "2025-03-31"),
         "test": ("2025-04-01", "2025-09-12")}
N_GROUPS = 5  # station_groups() default
N_GROUPS_NORDIC = 3  # extra groups for the Swedish / Finnish stations (--stations ext)


def use_stations(tag):
    """'base': the 99 Frost stations of Nordland / Troms / Finnmark; 'ext': + the SMHI / FMI
    stations north of 65 N (scripts/download_nordic_obs.py, build_multimodel_samples.py
    --nordic); 'all': + the Frost stations of Trondelag (--other-counties). Sets the data and
    terrain files; see groups_for for the groups."""
    global DATA, TERRAIN, TERRAIN100
    sfx = "" if tag == "base" else f"_{tag}"
    TERRAIN100 = ROOT / "cache" / f"station_terrain_100m{sfx}.npz"
    DATA = ROOT / "cache" / f"multimodel_2024-03-01_2025-09-12{sfx}"
    TERRAIN = ROOT / "cache" / f"station_terrain{sfx}.npz"


def use_group(group):
    """Input models: 'ecmwf' (IFS + AIFS, default) or a group of scripts/build_group_samples.py
    ('nwp': IFS, GFS, MEPS, AROME-Arctic; 'mlwp': AIFS, GraphCast, Pangu, FourCastNet). Same runs,
    stations and obs; base stations only. The residual is against the mean of the group's models."""
    global DATA, GRID_FILE
    if group != "ecmwf":
        DATA = DATA.with_name(DATA.name + f"_{group}")
        GRID_FILE = GRID_FILE.with_name(DATA.name + "_grid.npy")


def groups_for(d):
    """Spatial groups: the base Frost stations keep the k-means groups of the base set (0..4,
    identical to A1 / A2); the Swedish / Finnish stations get their own k-means groups 5..7;
    added Frost stations of other counties (Trondelag) are group 8."""
    xy = d["xy"].astype(float)
    if "country" not in d:
        return station_groups(xy)
    nordic = d["country"] != "NO"
    added = d["added"] if "added" in d else nordic
    base = ~added
    g = np.empty(len(xy), int)
    g[base] = station_groups(xy[base])
    if nordic.any():
        g[nordic] = N_GROUPS + station_groups(xy[nordic], k=N_GROUPS_NORDIC)
    g[added & ~nordic] = N_GROUPS + N_GROUPS_NORDIC
    return g


def plane(lon, lat):
    """Equirectangular km around (21 E, 68 N), divided by 1000 (GNO coordinates)."""
    return np.stack([(lon - 21.0) * np.cos(np.deg2rad(68.0)) * 111.32,
                     (lat - 68.0) * 110.57], -1) / 1000.0


class Data:
    def __init__(self, device, terrain=False, use_obs=False, terrain100=None):
        d = dict(np.load(f"{DATA}.npz", allow_pickle=True))
        self.grid = np.load(GRID_FILE, mmap_mode="r")             # shared by both station sets
        self.inits = pd.DatetimeIndex(d["inits"])
        self.leads = d["leads"]
        N, L, M, S, _ = d["st_fc"].shape
        self.N, self.L, self.M, self.S = N, L, M, S
        lat, lon = d["lat"], d["lon"]
        self.H, self.W = len(lat), len(lon)
        tindex = pd.Index(d["time"])
        ti = tindex.get_indexer(self.inits)
        obs, fc = d["obs"], d["st_fc"]
        base = fc.mean(2)                                                   # (N, L, S, 2)
        o_valid = np.stack([obs[ti + ld] for ld in self.leads], 1)          # (N, L, S, 2)
        self.y = (o_valid - base).astype(np.float32)
        self.o_valid, self.base, self.st_fc = o_valid, base, fc
        prev = pd.Index(self.inits).get_indexer(self.inits - pd.Timedelta(hours=12))
        j12 = list(self.leads).index(12)
        omf = np.full((N, M, S, 2), np.nan, np.float32)                     # obs-forecast at t
        ok = prev >= 0
        omf[ok] = obs[ti[ok]][:, None] - fc[prev[ok], j12]
        self.omf = omf
        self.split = {k: np.where((self.inits >= a) & (self.inits <= pd.Timestamp(b) + pd.Timedelta(hours=12))
                                  & d["keep"])[0] for k, (a, b) in SPLIT.items()}
        tr = self.split["train"]
        # normalisation (train)
        g = np.asarray(self.grid[tr[::4]], dtype=np.float32)                # (n, L, M, C, H, W)
        self.g_mean = g.mean((0, 1, 4, 5)).reshape(1, -1, 1, 1)             # (1, M*C, 1, 1)
        self.g_std = g.std((0, 1, 4, 5)).reshape(1, -1, 1, 1) + 1e-6
        self.omf_std = float(np.nanstd(omf[tr]))
        self.y_std = float(np.nanstd(self.y[tr]))
        # geometry
        LON, LAT = np.meshgrid(lon, lat)
        self.grid_p = torch.tensor(plane(LON, LAT), dtype=torch.float32, device=device)       # (H, W, 2)
        self.st_p = torch.tensor(plane(d["lon_st"], d["lat_st"]), dtype=torch.float32, device=device)
        fi = (d["lat_st"] - lat[0]) / (lat[1] - lat[0])
        fj = (d["lon_st"] - lon[0]) / (lon[1] - lon[0])
        self.st_ij = (np.rint(fi).astype(int), np.rint(fj).astype(int))
        self.st_sample = torch.tensor(np.stack([fj / (self.W - 1) * 2 - 1, fi / (self.H - 1) * 2 - 1], -1),
                                      dtype=torch.float32, device=device)  # grid_sample (x, y)
        self.coord_ch = torch.tensor(np.stack([(LAT - LAT.mean()) / LAT.std(), (LON - LON.mean()) / LON.std()]),
                                     dtype=torch.float32, device=device)
        self.groups = groups_for(d)
        self.n_groups = int(self.groups.max()) + 1
        self.station = d["station"]
        self.device = device
        self.terrain, self.use_obs = terrain, use_obs
        self.lat, self.lon = lat, lon
        self.st_xy = torch.tensor(d["xy"], dtype=torch.float32, device=device)  # LAEA km (windcast.geo)
        if terrain:
            t = np.load(TERRAIN, allow_pickle=True)
            assert (t["station"] == d["station"]).all()
            names = [str(x) for x in t["names"]]
            val = t["values"].astype(np.float64)
            sx_sets = ["sx", "sx10"]
            for fname, sxs in T100_FILES.get(terrain100, []):     # 100 m descriptors after the 1 km ones
                t1 = np.load(TERRAIN.with_name(fname), allow_pickle=True)
                assert (t1["station"] == d["station"]).all()
                names = names + [str(x) for x in t1["names"]]
                val = np.concatenate([val, t1["values"].astype(np.float64)], 1)
                sx_sets += list(sxs)
            # z-scored over the stations; at an arbitrary 1 km cell the same mean / std is used
            st_z = (val - val.mean(0)) / (val.std(0) + 1e-9)
            u, v = base[..., 0], base[..., 1]
            dyn = [fc[:, :, m, :, c] for m in range(M) for c in range(2)] + [np.hypot(u, v)]
            for pre in sx_sets:
                dyn.append(upwind(val[:, [names.index(f"{pre}_{k}") for k in SX_DIRS]], u, v))
            dyn = np.stack(dyn, -1)                                             # (N, L, S, 7 or 9)
            mu, sd = np.nanmean(dyn[tr], (0, 1, 2)), np.nanstd(dyn[tr], (0, 1, 2)) + 1e-9
            self.pt_dyn = np.nan_to_num((dyn - mu) / sd).astype(np.float32)
            self.pt_static = st_z.astype(np.float32)                            # (S, 28)
            # kept so the same features can be built at any 1 km cell (scripts/predict_terrain_grid.py)
            self.terrain_names, self.st_mu, self.st_sd = names, val.mean(0), val.std(0) + 1e-9
            self.dyn_mu, self.dyn_sd = mu, sd
            self.n_point = self.pt_dyn.shape[-1] + self.pt_static.shape[-1] + (4 if use_obs else 0)

    def setup_fine(self, P):
        """1 km grid inputs for TwoLevelNet, kept on the GPU padded by P // 2 (edge values):
        static terrain (28, H1+, W1+) z-scored over the 1 km grid, the 8-sector shelter angles at
        1 km and 10 km (8, H1+, W1+) for the upwind shelter, and the grid_sample coordinates of
        every 1 km cell in the 0.25 deg grid (H1+, W1+, 2). patches() cuts P x P patches from
        them; the station patches (centred on each station's 1 km cell) are cut once here."""
        assert P % 2 == 1
        dev, r = self.device, P // 2
        t = np.load(TERRAIN_GRID, allow_pickle=True)
        names = [str(x) for x in t["names"]]
        Fg = t["fields"].astype(np.float32)                                     # (28, H1, W1)
        self.H1, self.W1 = Fg.shape[1:]
        pad = lambda a: np.pad(a, [(0, 0)] * (a.ndim - 2) + [(r, r), (r, r)], mode="edge")
        mu = np.nanmean(Fg, (1, 2))[:, None, None]
        sd = np.nanstd(Fg, (1, 2))[:, None, None] + 1e-6
        T = lambda a: torch.tensor(np.nan_to_num(pad(a)), dtype=torch.float32, device=dev)
        self.full_static = T((Fg - mu) / sd)
        sx = [Fg[[names.index(f"{pre}_{k}") for k in SX_DIRS]] for pre in ("sx", "sx10")]
        self.full_sx = [T(a) for a in sx]
        self.f_sx_sd = [float(np.nanstd(a)) + 1e-6 for a in sx]
        lat, lon = self.lat, self.lon
        fi = (t["lat"] - lat[0]) / (lat[1] - lat[0])
        fj = (t["lon"] - lon[0]) / (lon[1] - lon[0])
        smp = np.stack([fj / (self.W - 1) * 2 - 1, fi / (self.H - 1) * 2 - 1])  # (2, H1, W1)
        self.full_sample = T(smp)
        off = torch.arange(P, device=dev)
        self._off = (off[:, None], off[None, :])
        ij = np.load(TERRAIN, allow_pickle=True)["ij"]
        self.st_patch = self.patches(ij[:, 0], ij[:, 1])
        C = self.grid.shape[3]
        self.uv_ch = [(m * C, m * C + 1) for m in range(self.M)]     # (u, v) of each model in g
        self.P = P
        self.n_fine_in = self.M * C + 1 + 2 + Fg.shape[0] + 5

    def patches(self, ci, cj):
        """P x P patches centred on 1 km cells (ci, cj) (K,): dict of static (K, 28, P, P),
        sx (two (K, 8, P, P)), sample (K, P, P, 2)."""
        ci = torch.as_tensor(np.asarray(ci), device=self.device)[:, None, None] + self._off[0]
        cj = torch.as_tensor(np.asarray(cj), device=self.device)[:, None, None] + self._off[1]
        cut = lambda a: a[:, ci, cj].permute(1, 0, 2, 3)                        # (K, C, P, P)
        return {"static": cut(self.full_static), "sx": [cut(a) for a in self.full_sx],
                "sample": cut(self.full_sample).permute(0, 2, 3, 1)}

    def time_features(self, n, l):
        vt = self.inits[n] + pd.to_timedelta(self.leads[l], unit="h")
        return np.stack([self.leads[l] / 66.0, np.sin(2 * np.pi * vt.hour / 24), np.cos(2 * np.pi * vt.hour / 24),
                         np.sin(2 * np.pi * vt.dayofyear / 366), np.cos(2 * np.pi * vt.dayofyear / 366)], -1)

    def batch(self, n, l, in_mask):
        """n, l (B,) indices; in_mask (B, S) bool: stations whose obs-forecast is an input.
        Returns grid fields (B, MC, H, W), time (B, 5), station features (B, S, 5).
        With terrain inputs the station features are the point features (B, S, n_point)
        and in_mask is ignored."""
        dev = self.device
        g = np.asarray(self.grid[n, l], dtype=np.float32).reshape(len(n), -1, self.H, self.W)
        g = torch.tensor((g - self.g_mean) / self.g_std, device=dev)
        tf = torch.tensor(self.time_features(n, l), dtype=torch.float32, device=dev)
        if self.terrain:
            pt = np.concatenate([self.pt_dyn[n, l],
                                 np.broadcast_to(self.pt_static, (len(n),) + self.pt_static.shape)], -1)
            X = {"pt": torch.tensor(pt, device=dev)}
            if self.use_obs:
                X["st"] = self.station_obs(n, in_mask)
            return g, tf, X
        return g, tf, self.station_obs(n, in_mask)

    def station_obs(self, n, in_mask):
        """(B, S, M*2 + 1): normalised obs-forecast at init of the stations in in_mask, + flag."""
        dev = self.device
        om = self.omf[n].transpose(0, 2, 1, 3).reshape(len(n), self.S, -1) / self.omf_std  # (B, S, M*2)
        m = in_mask & np.isfinite(om).all(-1)
        om = np.where(m[..., None], np.nan_to_num(om), 0.0)
        st = torch.tensor(np.concatenate([om, m[..., None]], -1), dtype=torch.float32, device=dev)
        return st

    def raster(self, st):
        """Station features (B, S, F) -> grid (B, F, H, W): mean of the stations in each cell;
        the last channel (mask) becomes the station count indicator."""
        B, S, Fc = st.shape
        idx = torch.tensor(self.st_ij[0] * self.W + self.st_ij[1], device=st.device)
        val = st[..., :-1] * st[..., -1:]
        num = torch.zeros(B, self.H * self.W, Fc - 1, device=st.device).index_add_(1, idx, val)
        cnt = torch.zeros(B, self.H * self.W, 1, device=st.device).index_add_(1, idx, st[..., -1:])
        out = torch.cat([num / cnt.clamp(min=1), (cnt > 0).float()], -1)
        return out.permute(0, 2, 1).reshape(B, Fc, self.H, self.W)


GRID_MODELS = ["fno", "tfno", "uno", "sfno", "localno", "codano"]
POINT_MODELS = ["fnogno", "gino"]


class PadCrop(torch.nn.Module):
    """Replicate-pads the grid by `p` cells before a model with periodic convolutions and crops
    afterwards, so the periodic wrap only joins padding, not the two sides of the domain."""

    def __init__(self, net, p):
        super().__init__()
        self.net, self.p = net, p

    def forward(self, x):
        p = self.p
        return self.net(F.pad(x, (p, p, p, p), mode="replicate"))[..., p:-p, p:-p]


class CodanoHead(torch.nn.Module):
    """CODANO returns one output field per input variable; a 1x1 convolution maps them to (u, v)."""

    def __init__(self, codano, n_vars, n_out=2):
        super().__init__()
        self.codano, self.head = codano, torch.nn.Conv2d(n_vars, n_out, 1)

    def forward(self, x):
        return self.head(self.codano(x))


def build_grid_model(kind, n_in, H, W, width, modes, layers, n_out=2):
    import neuralop.models as nm
    if kind == "fno":
        return nm.FNO(n_modes=modes, in_channels=n_in, out_channels=n_out, hidden_channels=width, n_layers=layers,
                      domain_padding=0.1, positional_embedding=None)
    if kind == "tfno":   # Tucker-factorised spectral weights
        return nm.TFNO(n_modes=modes, in_channels=n_in, out_channels=n_out, hidden_channels=width, n_layers=layers,
                       domain_padding=0.1, positional_embedding=None, factorization="tucker", rank=0.42)
    if kind == "uno":    # U-shaped: full -> half -> half -> full resolution, with skips
        return nm.UNO(in_channels=n_in, out_channels=n_out, hidden_channels=width, n_layers=5,
                      uno_out_channels=[width, 2 * width, 2 * width, 2 * width, width],
                      uno_n_modes=[list(modes), [m // 2 for m in modes], [m // 2 for m in modes],
                                   [m // 2 for m in modes], list(modes)],
                      uno_scalings=[[1, 1], [0.5, 0.5], [1, 1], [2, 2], [1, 1]],
                      horizontal_skips_map={4: 0, 3: 1}, domain_padding=0.1, positional_embedding=None,
                      channel_mlp_skip="linear")
    if kind == "sfno":   # spherical harmonic convolutions (treats the grid as an equiangular sphere grid)
        return nm.SFNO(n_modes=modes, in_channels=n_in, out_channels=n_out, hidden_channels=width, n_layers=layers,
                       positional_embedding=None)
    if kind == "localno":  # FNO + local (DISCO) integral and differential kernels
        return PadCrop(nm.LocalNO(n_modes=modes, in_channels=n_in, out_channels=n_out, hidden_channels=width,
                                  default_in_shape=(H + 8, W + 8), n_layers=layers, positional_embedding=None,
                                  conv_padding_mode="periodic",
                                  domain_length=[2.0, 2.0 * (W + 8) / (H + 8)]), 4)   # equal spacing both ways
    if kind == "codano":   # codomain attention: every input channel is a token
        codano = nm.CODANO(output_variable_codimension=1, hidden_variable_codimension=32, lifting_channels=64,
                           projection_channels=64, n_layers=layers, n_modes=[list(modes)] * layers,
                           n_heads=[2] * layers, per_layer_scaling_factors=[[1, 1]] * layers,
                           attention_scaling_factors=[1] * layers, use_positional_encoding=False)
        return CodanoHead(codano, n_in, n_out)
    raise ValueError(kind)


class TerrainNet(torch.nn.Module):
    """Operator on the model fields and time only (no coordinates, no observations) -> 32 latent
    channels, read bilinearly at the point, + point features -> point-wise network -> (u, v)."""

    def __init__(self, kind, data: Data, width=64, modes=(16, 32), layers=4, latent=32, hidden=128):
        super().__init__()
        self.kind, self.D = kind, data
        n_field = data.g_mean.shape[1]
        n_in = n_field + 5 + (data.M * 2 + 1 if data.use_obs else 0)
        if kind == "point":
            self.op, n_head = None, data.n_point + 5
        else:
            self.op, n_head = build_grid_model(kind, n_in, data.H, data.W, width, modes, layers,
                                               n_out=latent), data.n_point + latent
        self.head = torch.nn.Sequential(torch.nn.Linear(n_head, hidden), torch.nn.GELU(),
                                        torch.nn.Linear(hidden, hidden), torch.nn.GELU(),
                                        torch.nn.Linear(hidden, 2))

    def latent(self, g, tf, st=None):
        """Operator output on the 0.25 deg grid (B, latent, H, W); None for the point model."""
        D = self.D
        if self.op is None:
            return None
        x = [g, tf[:, :, None, None].expand(-1, -1, D.H, D.W)]
        if D.use_obs:
            x.append(D.raster(st))
        out = self.op(torch.cat(x, 1))
        if out.shape[-2:] != (D.H, D.W):
            out = F.interpolate(out, size=(D.H, D.W), mode="bilinear", align_corners=True)
        return out

    def idw(self, st, pos):
        """Other stations' obs-forecast weighted to the points pos (P, 2) km -> (B, P, 4)."""
        d2 = torch.cdist(pos, self.D.st_xy) ** 2                                # (P, S)
        w = 1.0 / (d2 + 100.0)
        m = st[..., -1]
        num = torch.einsum("ps,bsf->bpf", w, st[..., :-1] * m[..., None])
        den = torch.einsum("ps,bs->bp", w, m)
        return num / den.clamp(min=1e-12)[..., None]

    def head_at(self, lat, tf, pt, sample, st=None, pos=None):
        """Point-wise network at P points: latent field (or None), point features (B, P, n),
        sample (P, 2) grid_sample coordinates in the 0.25 deg grid, pos (P, 2) km."""
        B, P = pt.shape[:2]
        if lat is None:
            z = tf[:, None].expand(-1, P, -1)
        else:
            z = F.grid_sample(lat, sample[None, None].expand(B, -1, -1, -1), align_corners=True,
                              padding_mode="border")[:, :, 0].permute(0, 2, 1)   # (B, P, latent)
        feats = [z, pt]
        if self.D.use_obs:
            feats.append(self.idw(st, pos))
        return self.head(torch.cat(feats, -1))

    def forward(self, g, tf, X):
        D = self.D
        st = X.get("st")
        return self.head_at(self.latent(g, tf, st), tf, X["pt"], D.st_sample, st, D.st_xy)


def upwind_t(sx, u, v):
    """torch upwind(): sx (S, 8, P, P), u / v (B, S, P, P) -> (B, S, P, P)"""
    pos = torch.remainder(torch.atan2(-u, -v), 2 * math.pi) / (math.pi / 4)
    i0 = torch.floor(pos)
    f = pos - i0
    i0 = i0.long() % 8
    sxb = sx[None].expand(u.shape[0], -1, -1, -1, -1)
    a = torch.gather(sxb, 2, i0[:, :, None]).squeeze(2)
    b = torch.gather(sxb, 2, ((i0 + 1) % 8)[:, :, None]).squeeze(2)
    return (1 - f) * a + f * b


class TwoLevelNet(torch.nn.Module):
    """Coarse operator on the 0.25 deg grid (latent channels) + LocalNO on 1 km patches
    (Data.setup_fine / patches); see the module docstring (--fine)."""

    def __init__(self, kind, data: Data, width=64, modes=(16, 32), layers=4, latent=32,
                 fine_width=32, fine_modes=(4, 4), disco_km=3.0):
        super().__init__()
        import neuralop.models as nm
        self.D = data
        n_field = data.g_mean.shape[1]
        self.op = build_grid_model(kind, n_field + 5, data.H, data.W, width, modes, layers, n_out=latent)
        P = data.P
        self.fine = nm.LocalNO(n_modes=fine_modes, in_channels=latent + data.n_fine_in, out_channels=2,
                               hidden_channels=fine_width, default_in_shape=(P, P), n_layers=layers,
                               positional_embedding=None, conv_padding_mode="replicate",
                               domain_length=[float(P), float(P)], radius_cutoff=disco_km)
        self.register_buffer("g_mean", torch.tensor(data.g_mean.reshape(-1), dtype=torch.float32), persistent=False)
        self.register_buffer("g_std", torch.tensor(data.g_std.reshape(-1), dtype=torch.float32), persistent=False)

    def fields(self, g, tf, patch):
        """Correction (B, K, 2, P, P) in units of y_std on the patches, and the mean model wind
        (u, v) there (B, K, 2, P, P) in m/s."""
        D, P = self.D, self.D.P
        B, K = g.shape[0], patch["sample"].shape[0]
        lat = self.op(torch.cat([g, tf[:, :, None, None].expand(-1, -1, D.H, D.W)], 1))
        if lat.shape[-2:] != (D.H, D.W):
            lat = F.interpolate(lat, size=(D.H, D.W), mode="bilinear", align_corners=True)
        smp = patch["sample"].reshape(1, K * P, P, 2).expand(B, -1, -1, -1)
        both = F.grid_sample(torch.cat([lat, g], 1), smp, align_corners=True, padding_mode="border")
        both = both.reshape(B, -1, K, P, P).permute(0, 2, 1, 3, 4)               # (B, K, ch, P, P)
        gi = both[:, :, lat.shape[1]:]
        uv = [(gi[:, :, a] * self.g_std[a] + self.g_mean[a], gi[:, :, b] * self.g_std[b] + self.g_mean[b])
              for a, b in D.uv_ch]
        u = sum(p[0] for p in uv) / len(uv)
        v = sum(p[1] for p in uv) / len(uv)
        dyn = [torch.hypot(u, v)[:, :, None] / 8.0] + \
              [upwind_t(sx, u, v)[:, :, None] / sd for sx, sd in zip(patch["sx"], D.f_sx_sd)]
        stat = patch["static"][None].expand(B, -1, -1, -1, -1)
        tch = tf[:, None, :, None, None].expand(-1, K, -1, P, P)
        inp = torch.cat([both] + dyn + [stat, tch], 2).reshape(B * K, -1, P, P)
        out = self.fine(inp).reshape(B, K, 2, P, P)
        return out, torch.stack([u, v], 2)

    def forward(self, g, tf, X):
        P = self.D.P
        return self.fields(g, tf, self.D.st_patch)[0][..., P // 2, P // 2]      # (B, S, 2)


def pretrain_meps(model, D, epochs, lr, bs=8, k=16, rng=None, crop=4):
    """Dense pretraining of the two-level operator: on random 1 km patches inside the MEPS
    window, the target is MEPS u10 / v10 (nearest 2.5 km cell, same init and lead) minus the
    mean of IFS and AIFS there, scored on every patch cell but the `crop` cells along the
    edge. Train-period runs only (no test-period information)."""
    tag = DATA.name.replace("multimodel_", "")
    mp = np.load(ROOT / "cache" / f"meps_uv_{tag}_map.npz")
    meps = np.load(ROOT / "cache" / f"meps_uv_{tag}.npy", mmap_mode="r")
    P, r = D.P, D.P // 2
    # patch centres whose whole patch lies inside the MEPS window
    ins = torch.tensor(mp["inside"], dtype=torch.float32)[None, None]
    full = (F.avg_pool2d(ins, P, stride=1, padding=r, count_include_pad=True)[0, 0] > 0.999).numpy()
    ci, cj = np.where(full)
    print(f"MEPS pretraining: {len(ci)} patch centres, {mp['have'][D.split['train']].sum()} train runs", flush=True)
    iy, ix = torch.tensor(mp["iy"]), torch.tensor(mp["ix"])
    tr = D.split["train"][mp["have"][D.split["train"]]]
    idx = (tr[:, None] * D.L + np.arange(D.L)[None]).reshape(-1)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * math.ceil(len(idx) / bs) + 10)
    sl = slice(crop, P - crop)
    hist = []
    for ep in range(epochs):
        t0, tot, cnt = time.time(), 0.0, 0
        model.train()
        order = rng.permutation(len(idx))
        for b0 in range(0, len(idx), bs):
            sel = idx[order[b0:b0 + bs]]
            n, l = sel // D.L, sel % D.L
            pick = rng.choice(len(ci), k, replace=False)
            pc, pj = ci[pick], cj[pick]
            # MEPS at the patch cells (B, K, 2, P, P)
            gy = (pc[:, None, None] - r + np.arange(P)[None, :, None]) + 0 * np.arange(P)[None, None, :]
            gx = (pj[:, None, None] - r + np.arange(P)[None, None, :]) + 0 * np.arange(P)[None, :, None]
            my, mx = iy[gy, gx].numpy(), ix[gy, gx].numpy()
            m = np.stack([np.asarray(meps[a, b], dtype=np.float32) for a, b in zip(n, l)])   # (B, 2, Hm, Wm)
            tgt = m[:, :, my, mx].transpose(0, 2, 1, 3, 4)                                   # (B, K, 2, P, P)
            g, tf, _ = D.batch(n, l, np.zeros((len(n), D.S), bool))
            patch = D.patches(pc, pj)
            out, base = model.fields(g, tf, patch)
            y = (torch.tensor(tgt, device=D.device) - base) / D.y_std
            w = torch.isfinite(y).all(2, keepdim=True).float()[..., sl, sl]
            err = ((out - torch.nan_to_num(y))[..., sl, sl] ** 2) * w
            loss = err.sum() / w.sum().clamp(min=1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item()
            cnt += 1
        hist.append({"epoch": ep, "train": tot / cnt, "sec": time.time() - t0})
        print(f"pretrain ep {ep:3d} loss {tot / cnt:.4f} ({time.time() - t0:.0f} s)", flush=True)
    return hist


class Net(torch.nn.Module):
    def __init__(self, kind, data: Data, width=64, modes=(16, 32), layers=4):
        super().__init__()
        from neuralop.models import FNOGNO, GINO
        self.kind, self.D = kind, data
        n_field = data.g_mean.shape[1]
        n_st = data.M * 2 + 1
        n_grid = n_field + 5 + 2 + n_st                                       # fields, time, coords, raster
        if kind in GRID_MODELS:
            self.net = build_grid_model(kind, n_grid, data.H, data.W, width, modes, layers)
        elif kind == "fnogno":
            self.net = FNOGNO(in_channels=n_grid, out_channels=2, gno_coord_dim=2, fno_n_modes=modes,
                              fno_hidden_channels=width, fno_n_layers=layers, gno_radius=0.06,
                              gno_use_open3d=False, gno_use_torch_scatter=False, gno_batched=True,
                              gno_channel_mlp_hidden_layers=[128, 64])
        elif kind == "gino":
            self.net = GINO(in_channels=n_st, out_channels=2, latent_feature_channels=n_field + 5 + 2,
                            gno_coord_dim=2, in_gno_radius=0.12, out_gno_radius=0.06, fno_in_channels=n_st,
                            fno_n_modes=modes, fno_hidden_channels=width, fno_n_layers=layers,
                            gno_use_open3d=False, gno_use_torch_scatter=False,
                            in_gno_channel_mlp_hidden_layers=[64, 64], out_gno_channel_mlp_hidden_layers=[128, 64])
        else:
            raise ValueError(kind)

    def forward(self, g, tf, st):
        D = self.D
        B = g.shape[0]
        tmap = tf[:, :, None, None].expand(-1, -1, D.H, D.W)
        cmap = D.coord_ch[None].expand(B, -1, -1, -1)
        if self.kind == "gino":
            lat = torch.cat([g, tmap, cmap], 1).permute(0, 2, 3, 1)           # (B, H, W, C)
            return self.net(input_geom=D.st_p[None], latent_queries=D.grid_p[None],
                            output_queries=D.st_p[None], x=st, latent_features=lat)   # (B, S, 2)
        x = torch.cat([g, tmap, cmap, D.raster(st)], 1)
        if self.kind == "fnogno":
            return self.net(in_p=D.grid_p, out_p=D.st_p, f=x.permute(0, 2, 3, 1))   # (B, S, 2)
        out = self.net(x)                                                       # (B, 2, h, w)
        if out.shape[-2:] != (D.H, D.W):
            out = F.interpolate(out, size=(D.H, D.W), mode="bilinear", align_corners=True)
        s = F.grid_sample(out, D.st_sample[None, None].expand(B, -1, -1, -1), align_corners=True)
        return s[:, :, 0].permute(0, 2, 1)                                      # (B, S, 2)


def run_epoch(model, D, idx, setting, fold, train, opt=None, sched=None, bs=16, rng=None, base=None, q=0.9):
    """One pass. With `base` (UQNO), `model` predicts the error radius of the frozen base model
    and is trained with the pinball loss at quantile q on the vector error |y - base|."""
    model.train(train)
    tot, cnt = 0.0, 0
    allowed = np.ones(D.S, bool) if setting == "including" else D.groups != fold
    order = rng.permutation(len(idx)) if train else np.arange(len(idx))
    for b0 in range(0, len(idx), bs):
        sel = idx[order[b0:b0 + bs]]
        n, l = sel // D.L, sel % D.L
        avail = np.isfinite(D.omf[n]).all((1, 3))                               # (B, S)
        yv = np.isfinite(D.y[n, l]).all(-1)
        if setting == "including":
            in_mask, tgt = avail & allowed, yv & allowed
        elif D.terrain and not D.use_obs:                                        # no observation inputs
            in_mask = avail
            tgt = yv & (allowed if train else D.groups == fold)
        elif train and D.terrain:                                                # "both": hide one whole group
            pick = rng.choice(np.setdiff1d(np.arange(D.n_groups), [fold]), size=len(n))
            hide = D.groups[None, :] == pick[:, None]
            in_mask = avail & allowed & ~hide
            tgt = yv & allowed & hide
        elif train:
            hide = rng.random((len(n), D.S)) < 0.25
            in_mask = avail & allowed & ~hide
            tgt = yv & allowed & hide
        else:                                                                    # predict group `fold`
            in_mask = avail & allowed
            tgt = yv & (D.groups == fold)
        if tgt.sum() == 0:
            continue
        g, tf, st = D.batch(n, l, in_mask)
        y = torch.tensor(np.nan_to_num(D.y[n, l]) / D.y_std, device=D.device)
        w = torch.tensor(tgt, dtype=torch.float32, device=D.device)
        with torch.set_grad_enabled(train), torch.autocast("cuda", dtype=torch.bfloat16, enabled=False):
            pred = model(g, tf, st)
            if base is None:
                loss = (((pred - y) ** 2).sum(-1) * w).sum() / w.sum()
            else:
                with torch.no_grad():
                    err = (base(g, tf, st) - y).norm(dim=-1)
                r = F.softplus(pred[..., 0])
                loss = (torch.maximum(q * (err - r), (q - 1) * (err - r)) * w).sum() / w.sum()
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
def predict(model, D, inits_idx, setting, fold, bs=32, radius=False):
    model.eval()
    out = np.full((D.N, D.L, D.S, 2), np.nan, np.float32)
    allowed = np.ones(D.S, bool) if setting == "including" else D.groups != fold
    pred_st = np.ones(D.S, bool) if setting == "including" else D.groups == fold
    idx = (inits_idx[:, None] * D.L + np.arange(D.L)[None]).reshape(-1)
    for b0 in range(0, len(idx), bs):
        sel = idx[b0:b0 + bs]
        n, l = sel // D.L, sel % D.L
        avail = np.isfinite(D.omf[n]).all((1, 3))
        g, tf, st = D.batch(n, l, avail & allowed[None])
        p = model(g, tf, st).float()
        p = (F.softplus(p[..., :1]).expand(-1, -1, 2) if radius else p).cpu().numpy() * D.y_std
        p[:, ~pred_st] = np.nan
        out[n, l] = p
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=GRID_MODELS + POINT_MODELS + ["uqno", "point"])
    ap.add_argument("--inputs", default="obs", choices=["obs", "terrain", "both"])
    ap.add_argument("--setting", required=True, choices=["including", "excluding"])
    ap.add_argument("--fold", type=int, default=-1)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--max-train", type=int, default=0, help="smoke test: limit samples")
    ap.add_argument("--terrain100", nargs="?", const="mtno", default=None, choices=list(T100_FILES),
                    help="add 100 m descriptors (see T100_FILES); run name + T100_SUFFIX")
    ap.add_argument("--fine", default=None, choices=["localno"],
                    help="two-level: coarse operator + LocalNO on 1 km patches (terrain inputs); run name + _lno")
    ap.add_argument("--patch", type=int, default=33, help="--fine: patch size (1 km cells, odd)")
    ap.add_argument("--disco-km", type=float, default=3.0, help="--fine: local integral radius (km)")
    ap.add_argument("--pretrain-meps", type=int, default=0, metavar="EPOCHS",
                    help="--fine: first pretrain on dense MEPS targets (pretrain_meps); run name + _mpt<EPOCHS>")
    ap.add_argument("--pretrain-lr", type=float, default=3e-4)
    ap.add_argument("--stations", default="base", choices=["base", "ext", "all"],
                    help="ext: + Swedish / Finnish stations (groups 5-7); all: + Trondelag (group 8); "
                         "run name + _<stations>")
    ap.add_argument("--group", default="ecmwf", choices=["ecmwf", "nwp", "mlwp"],
                    help="input models (use_group); run name + _<group>")
    a = ap.parse_args()
    use_stations(a.stations)
    assert a.group == "ecmwf" or a.stations == "base"
    use_group(a.group)
    assert a.setting == "including" or 0 <= a.fold < {"base": N_GROUPS, "ext": N_GROUPS + N_GROUPS_NORDIC,
                                                       "all": N_GROUPS + N_GROUPS_NORDIC + 1}[a.stations]
    terrain = a.inputs in ("terrain", "both")
    assert not terrain or (a.setting == "excluding" and a.model in GRID_MODELS + ["point"])
    assert terrain or a.model != "point"
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    dev = torch.device("cuda")
    assert not a.terrain100 or terrain
    D = Data(dev, terrain=terrain, use_obs=a.inputs == "both", terrain100=a.terrain100)
    if a.fine:
        assert a.inputs == "terrain" and a.model in GRID_MODELS and not a.terrain100
        D.setup_fine(a.patch)
    if terrain:
        name = f"{a.model}_{'terrain' if a.inputs == 'terrain' else 'terrainobs'}_g{a.fold}_s{a.seed}"
    else:
        name = f"{a.model}_{a.setting}" + (f"_g{a.fold}" if a.setting == "excluding" else "") + f"_s{a.seed}"
    if a.stations != "base":
        name += f"_{a.stations}"
    if a.group != "ecmwf":
        name += f"_{a.group}"
    if a.terrain100:
        name += T100_SUFFIX[a.terrain100]
    assert not a.pretrain_meps or a.fine
    if a.fine:
        name += "_lno" + (f"_mpt{a.pretrain_meps}" if a.pretrain_meps else "")
        name += (f"_p{a.patch}" if a.patch != 33 else "") + (f"_r{a.disco_km:g}" if a.disco_km != 3.0 else "")
    if a.lr != 1e-3:
        name += f"_lr{a.lr:g}"
    out = RUNS / name
    out.mkdir(parents=True, exist_ok=True)
    base = None
    if a.model == "uqno":
        # UQNO (Ma et al. 2024): the base operator is the trained FNO of the same setting / group,
        # frozen; the residual operator starts as a copy of it and learns the error radius.
        src = RUNS / name.replace("uqno_", "fno_")
        base = Net("fno", D, width=a.width).to(dev)
        base.load_state_dict(torch.load(src / "best.pt", weights_only=True))
        base.eval()
        for p_ in base.parameters():
            p_.requires_grad_(False)
        model = Net("fno", D, width=a.width).to(dev)
        model.load_state_dict({k: v for k, v in base.state_dict().items() if torch.is_tensor(v)})
    elif a.fine:
        model = TwoLevelNet(a.model, D, width=a.width, disco_km=a.disco_km).to(dev)
    elif terrain:
        model = TerrainNet(a.model, D, width=a.width).to(dev)
    else:
        model = Net(a.model, D, width=a.width).to(dev)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"{name}: {n_par / 1e6:.2f} M parameters, {D.S} stations, "
          f"train/val/test runs {[len(D.split[k]) for k in SPLIT]}", flush=True)

    def samples(k):
        i = D.split[k]
        return (i[:, None] * D.L + np.arange(D.L)[None]).reshape(-1)
    tr, va = samples("train"), samples("val")
    if a.max_train:
        tr, va = tr[:a.max_train], va[:a.max_train]
    hist_pt = pretrain_meps(model, D, a.pretrain_meps, a.pretrain_lr, rng=rng) if a.pretrain_meps else []
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * math.ceil(len(tr) / a.bs) + 10)
    hist, best, best_state = [], np.inf, None
    for ep in range(a.epochs):
        t0 = time.time()
        trl = run_epoch(model, D, tr, a.setting, a.fold, True, opt, sched, bs=a.bs, rng=rng, base=base)
        val = run_epoch(model, D, va, a.setting, a.fold, False, bs=a.bs, rng=rng, base=base)
        hist.append({"epoch": ep, "train": trl, "val": val, "sec": time.time() - t0})
        print(f"ep {ep:3d} train {trl:.4f} val {val:.4f} ({time.time() - t0:.0f} s)", flush=True)
        if val < best:
            best = val
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items() if torch.is_tensor(v)}
            torch.save(best_state, out / "best.pt")
    model.load_state_dict(best_state)
    if base is None:
        pv = predict(model, D, D.split["val"], a.setting, a.fold)
        pt = predict(model, D, D.split["test"], a.setting, a.fold)
        np.savez_compressed(out / "pred.npz", val=pv, test=pt, groups=D.groups, station=D.station)
    else:
        # split-conformal calibration on the validation set: scale the radius so that 90 % of
        # validation errors fall inside it, then apply the same scale on the test set
        pv = predict(base, D, D.split["val"], a.setting, a.fold)
        pt = predict(base, D, D.split["test"], a.setting, a.fold)
        rv = predict(model, D, D.split["val"], a.setting, a.fold, radius=True)[..., 0]
        rt = predict(model, D, D.split["test"], a.setting, a.fold, radius=True)[..., 0]
        ev = np.linalg.norm(D.y - pv, axis=-1)
        m = np.isfinite(ev) & np.isfinite(rv)
        scale = float(np.quantile(ev[m] / rv[m], 0.9))
        et = np.linalg.norm(D.y - pt, axis=-1)
        mt = np.isfinite(et) & np.isfinite(rt)
        cover = float(np.mean(et[mt] <= scale * rt[mt]))
        print(f"UQNO: conformal scale {scale:.3f}, test coverage of the 90 % radius {cover:.3f}, "
              f"mean radius {np.mean(scale * rt[mt]):.2f} m/s", flush=True)
        np.savez_compressed(out / "pred.npz", val=pv, test=pt, radius_val=scale * rv, radius_test=scale * rt,
                            scale=scale, groups=D.groups, station=D.station)
    (out / "history.json").write_text(json.dumps({"args": vars(a), "n_params": n_par, "best_val": best,
                                                  "history": hist, "pretrain": hist_pt}, indent=1))
    print("best val", best, "->", out)


if __name__ == "__main__":
    main()
