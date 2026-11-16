"""Forecast samples: one (init time, lead) per sample, on the 1 km reference grid.

Inputs x (C, H, W), normalised with the train statistics:
    driver   MEPS fields at the valid time, bilinear to 1 km: u10 v10 fg10 t2m rh2m msl tcc
             tp6 (6 h precipitation ending at the valid time) blh
    static   terrain (16), MET Nordic altitude, land fraction, MEPS orography - altitude
    time     lead / 66 h, sin / cos of the valid hour and of the valid day of year
base (V, H, W): the driver's interpolated target variables, physical units; the model
predicts a residual on top of it, so an untrained model returns the driver.
y (V, H, W): MET Nordic analysis at the valid time, physical units.
"""
import json
from pathlib import Path

import netCDF4
import numpy as np
import pandas as pd

DRIVER = ["u10", "v10", "fg10", "t2m", "rh2m", "msl", "tcc", "tp6", "blh"]
TARGET = ["u10", "v10", "t2m", "rh2m", "msl", "tcc"]
N_TIME = 5
LEAD_MAX = 66.0


def meps_path(root: Path, t: pd.Timestamp) -> Path:
    return root / "meps" / f"{t:%Y}" / f"meps_{t:%Y%m%dT%H}.nc"


def mn_path(root: Path, t: pd.Timestamp) -> Path:
    return root / "metnordic" / f"{t:%Y}" / f"mn_{t:%Y%m%dT%H}.nc"


class Grid:
    """The 1 km reference grid, terrain and MEPS -> 1 km interpolation (grid.npz)."""

    def __init__(self, root: Path):
        g = np.load(root / "grid.npz")
        self.H, self.W = g["lat"].shape
        self.interp = {k[5:-4]: (g[k], g[k[:-4] + "_w"]) for k in g.files
                       if k.startswith("meps_") and k.endswith("_idx")}
        oro = {k[5:-4]: g[k] for k in g.files if k.startswith("meps_") and k.endswith("_oro")}
        self.static = {s: np.concatenate([g["terrain"], g["altitude"][None], g["land"][None],
                                          (oro[s] - g["altitude"])[None]]).astype(np.float32)
                       for s in oro}
        self.static_names = [*map(str, g["terrain_names"]), "altitude", "land", "dz_meps"]

    def to_1km(self, shape: str, a: np.ndarray) -> np.ndarray:
        """(..., ny, nx) MEPS window -> (..., H, W)."""
        idx, w = self.interp[shape]
        flat = a.reshape(*a.shape[:-2], -1)
        return (flat[..., idx] * w).sum(-1)


def read_meps(path: Path, grid: Grid, lead: int) -> dict:
    """Driver fields at `lead` (hours) interpolated to 1 km, physical units."""
    with netCDF4.Dataset(path) as ds:
        leads = list(ds["lead"][:])
        j = leads.index(lead)
        j6 = leads.index(lead - 6) if lead >= 6 else None
        # meps_det files are on the 2020-02 grid, meps_subset files on the older one
        shape = "1069x949" if "meps_det" in ds.getncattr("source") else "949x889"
        out = {k: np.asarray(ds[k][j], np.float32) for k in DRIVER if k != "tp6"}
        acc = np.asarray(ds["tp_acc"][j], np.float32)
        prev = np.asarray(ds["tp_acc"][j6], np.float32) if j6 is not None else np.zeros_like(acc)
    out["tp6"] = np.maximum(acc - prev, 0.0)
    stack = np.stack([out[k] for k in DRIVER])
    stack = grid.to_1km(shape, np.nan_to_num(stack))
    return {"x": stack.astype(np.float32), "shape": shape}


def read_mn(path: Path) -> np.ndarray:
    with netCDF4.Dataset(path) as ds:
        return np.stack([np.asarray(ds[k][:], np.float32) for k in TARGET])


def time_features(valid: pd.Timestamp, lead: int, H: int, W: int) -> np.ndarray:
    h = 2 * np.pi * valid.hour / 24
    d = 2 * np.pi * (valid.dayofyear - 1) / 365.25
    f = [lead / LEAD_MAX, np.sin(h), np.cos(h), np.sin(d), np.cos(d)]
    return np.broadcast_to(np.asarray(f, np.float32)[:, None, None], (N_TIME, H, W))


class ForecastDataset:
    """Map-style dataset (DataLoader-compatible); torch is imported only in __getitem__, so
    raw() works on the x86 login node without torch."""

    def __init__(self, cfg: dict, split: str, stats: dict | None = None):
        self.root = Path(cfg["data"]["root"])
        self.grid = Grid(self.root)
        self.leads = list(cfg["data"]["leads"])
        a, b = cfg["split"][split]
        inits = pd.date_range(a, f"{b} 23:00", freq="h")
        inits = inits[inits.hour.isin(cfg["data"]["inits"])]
        self.samples = []
        for t in inits:
            if not meps_path(self.root, t).exists():
                continue
            for L in self.leads:
                if mn_path(self.root, t + pd.Timedelta(hours=L)).exists():
                    self.samples.append((t, L))
        self.stats = stats
        self.persistence = bool(cfg["data"].get("persistence", False))

    def __len__(self):
        return len(self.samples)

    def raw(self, i: int) -> dict:
        t, L = self.samples[i]
        v = t + pd.Timedelta(hours=L)
        m = read_meps(meps_path(self.root, t), self.grid, L)
        y = read_mn(mn_path(self.root, v))
        base = m["x"][[DRIVER.index(k) for k in TARGET]]
        return {"driver": m["x"], "static": self.grid.static[m["shape"]],
                "time": time_features(v, L, self.grid.H, self.grid.W),
                "base": base, "y": y, "init": t, "lead": L}

    def __getitem__(self, i: int) -> dict:
        import torch
        r = self.raw(i)
        s = self.stats
        x = np.concatenate([(r["driver"] - s["driver_mean"]) / s["driver_std"],
                            (r["static"] - s["static_mean"]) / s["static_std"], r["time"]])
        out = {"x": torch.from_numpy(np.ascontiguousarray(x, np.float32)),
               "base": torch.from_numpy(r["base"]), "y": torch.from_numpy(r["y"]),
               "lead": r["lead"], "index": i}
        if self.persistence:  # MET Nordic analysis at init, for the persistence baseline
            p = mn_path(self.root, r["init"])
            out["persist"] = torch.from_numpy(read_mn(p) if p.exists() else
                                              np.full_like(r["y"], np.nan))
        return out


def load_stats(path: Path) -> dict:
    """stats.json -> arrays shaped (C, 1, 1) for broadcasting; name lists are dropped."""
    s = json.loads(Path(path).read_text())
    return {k: np.asarray(v, np.float32)[:, None, None] for k, v in s.items()
            if k.endswith(("_mean", "_std"))}
