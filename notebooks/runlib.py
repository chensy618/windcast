"""Shared loading and scoring for the notebooks 01_training and 02_experiments.

Every station-correction run of windcast/training/train_station_operator.py writes
RUNS/<run>/history.json (arguments, train / validation loss per epoch) and pred.npz (the
predicted correction at every station for the validation and test runs). This module reads them
and scores the held-out predictions the same way as web/pipeline/verification.py:

* test runs: inits 2025-04-01 .. 2025-09-12 present in both the ECMWF and the NWP samples;
* the 99 Frost stations of Nordland, Troms and Finnmark, in 5 spatially blocked groups; every
  station is predicted by the run that did not train on its group;
* vector RMSE of the 10 m wind (u, v) against the observations, per lead, and its change in %
  against the IFS + AIFS mean (negative = better).
"""
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from windcast.station_correction import station_groups  # noqa: E402

ROOT = Path("/cluster/work/projects/nn8106k/siyan/windcast")
RUNS = ROOT / "runs_a1"
CACHE = ROOT / "cache" / "multimodel_2024-03-01_2025-09-12"
SPLIT = {"train": ("2024-03-01", "2024-12-31"), "val": ("2025-01-01", "2025-03-31"),
         "test": ("2025-04-01", "2025-09-12")}
N_BASE = 99          # the base stations come first in every station set


def _period(inits, keep, name):
    a, b = SPLIT[name]
    return (inits >= a) & (inits <= pd.Timestamp(b) + pd.Timedelta(hours=12)) & keep


class Sample:
    """One cache (station set x input group): observations and the uncorrected input mean."""

    def __init__(self, sfx=""):
        d = np.load(f"{CACHE}{sfx}.npz", allow_pickle=True)
        self.inits = pd.DatetimeIndex(d["inits"])
        self.leads = np.array(d["leads"])
        ti = pd.Index(d["time"]).get_indexer(self.inits)
        self.obs = np.stack([d["obs"][ti + ld] for ld in self.leads], 1)        # (N, L, S, 2)
        self.base = d["st_fc"].mean(2)                                          # (N, L, S, 2)
        self.keep = d["keep"]
        self.station = np.array([str(s) for s in d["station"]])
        self.xy = d["xy"].astype(float)
        y = self.obs - self.base
        # the training target's scale (train runs, all stations): loss = (vector error / y_std)^2
        self.y_std = float(np.nanstd(y[self.period("train")]))

    def period(self, name):
        return _period(self.inits, self.keep, name)


class Scorer:
    """Held-out test scores on the 99 base stations."""

    def __init__(self):
        self.ecmwf, self.nwp = Sample(""), Sample("_nwp")
        self.leads = self.ecmwf.leads
        self.groups = station_groups(self.ecmwf.xy)
        self.te = self.ecmwf.period("test") & self.nwp.keep
        self.obs = self.ecmwf.obs[self.te]
        self.base = {"ecmwf": self.ecmwf.base[self.te], "nwp": self.nwp.base[self.te]}
        self.r0 = self.rmse(self.base["ecmwf"])                                  # reference, per lead

    def rmse(self, p):
        """Vector RMSE (m/s) per lead over test runs and base stations."""
        m = np.isfinite(self.obs[..., 0]) & np.isfinite(p[..., 0])
        e = ((self.obs - p) ** 2).sum(-1)
        return np.array([np.sqrt(e[:, j][m[:, j]].mean()) for j in range(len(self.leads))])

    def change(self, p):
        """Change of the vector RMSE against the IFS + AIFS mean, % per lead."""
        return (self.rmse(p) / self.r0 - 1) * 100

    def held_out(self, pattern, seed=0, group="ecmwf"):
        """Corrected test forecast at the base stations from the 5 runs `pattern` (with {k} for
        the fold and {s} for the seed): station group k is taken from the run of fold k."""
        acc = np.full(self.base[group].shape, np.nan, np.float32)
        for k in range(5):
            p = np.load(RUNS / pattern.format(k=k, s=seed) / "pred.npz")
            acc[:, :, self.groups == k] = p["test"][self.te][:, :, :N_BASE][:, :, self.groups == k]
        return self.base[group] + acc

    def seeds(self, pattern):
        return sorted({s for _, s in run_names(pattern).values()})

    def score(self, pattern, group="ecmwf"):
        """(seeds, leads) change in % for every seed that has all 5 folds."""
        out = []
        for s in self.seeds(pattern):
            if all((RUNS / pattern.format(k=k, s=s) / "pred.npz").exists() for k in range(5)):
                out.append(self.change(self.held_out(pattern, s, group)))
        return np.array(out)


def run_names(pattern):
    """{run name: (fold, seed)} of the runs whose name is exactly `pattern` with {k} / {s} filled."""
    rx = re.compile(re.escape(pattern).replace(r"\{k\}", r"(?P<k>\d+)").replace(r"\{s\}", r"(?P<s>\d+)"))
    out = {}
    for d in glob.glob(str(RUNS / pattern.format(k="*", s="*"))):
        m = rx.fullmatch(Path(d).name)
        if m and (Path(d) / "history.json").exists():
            out[m.group(0)] = (int(m.group("k")), int(m.group("s")))
    return out


def histories(pattern):
    """All runs matching `pattern` ({k} fold, {s} seed): one row per run with its curves."""
    rows = []
    for name in sorted(run_names(pattern)):
        f = RUNS / name / "history.json"
        h = json.loads(Path(f).read_text())
        a = h["args"]
        rows.append(dict(run=Path(f).parent.name, fold=a["fold"], seed=a["seed"], lr=a["lr"],
                         n_params=h["n_params"], best_val=h["best_val"],
                         train=np.array([x["train"] for x in h["history"]]),
                         val=np.array([x["val"] for x in h["history"]]),
                         sec=np.array([x["sec"] for x in h["history"]])))
    return pd.DataFrame(rows)
