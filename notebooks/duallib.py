"""Loading and scoring for notebooks/03_dual_gno.ipynb (dual-scale GNO experiments).

Every number is a held-out test score on the 99 Frost stations of Nordland, Troms and Finnmark:
each station is predicted by the fold that did not train on its spatial group (5 folds stitched),
on the common test samples of a period (test runs, observation present, and the IFS + AIFS,
MEPS + AROME-Arctic and four-model samples all available). Same scoring as
scripts/report_dual_gno/make_results.py, so the numbers match docs/report_dual_gno/report.tex.

Runs with the Swedish and Finnish training stations (`ext`) have 178 stations, the 99 Norwegian
ones first and in the same order; they are scored on those 99 with folds 0-4.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/cluster/work/projects/nn8106k/siyan/windcast")
RESULTS = ROOT / "results" / "report_dual_gno"
N_NO = 99
SMALL = "r16m8x16_g16m8x16_z0"                       # width 16, 8 x 16 modes, zero-initialised head

PERIODS = {
    "original": dict(tag="2024-03-01_2025-09-12", test0="2025-04-01", sfx="",
                     label="original (train 2024-03 – 2024-12, test 2025-04 – 2025-09)"),
    "extended": dict(tag="2024-03-01_2026-11-14", test0="2026-04-01", sfx="_2024-03-01_2026-11-14",
                     label="extended (train 2024-03 – 2025-12, test 2026-04 – 2026-11)"),
}


class Period:
    def __init__(self, name):
        p = PERIODS[name]
        self.name, self.label = name, p["label"]
        cache = ROOT / "cache" / f"multimodel_{p['tag']}"
        self.runs_a1, self.runs_dual = ROOT / f"runs_a1{p['sfx']}", ROOT / f"runs_dual{p['sfx']}"
        self.base, self.raw, keep = {}, {}, None
        for g, s in [("ecmwf", ""), ("lam", "_lam"), ("mix", "_mix")]:
            d = np.load(f"{cache}{s}.npz", allow_pickle=True)
            self.base[g] = d["st_fc"].mean(2)
            keep = d["keep"] if keep is None else keep & d["keep"]
            if g != "mix":                       # raw single models: ifs, aifs, meps, arome
                for i, m in enumerate(d["models"]):
                    self.raw[{"arome_arctic": "arome"}.get(str(m), str(m))] = d["st_fc"][:, :, i]
        e = np.load(f"{cache}.npz", allow_pickle=True)
        self.inits = pd.DatetimeIndex(e["inits"])
        self.leads = np.array(e["leads"])
        ti = pd.Index(e["time"]).get_indexer(self.inits)
        self.obs = np.stack([e["obs"][ti + ld] for ld in self.leads], 1)            # (N, L, 99, 2)
        self.mask = ((self.inits >= p["test0"]) & keep)[:, None, None] & np.isfinite(self.obs).all(-1)
        self.ref = self.rmse(self.base["ecmwf"])                                   # IFS + AIFS mean
        self.groups = None

    # ------------------------------------------------------------ scores
    def rmse(self, f, m=None):
        m = self.mask if m is None else m
        return float(np.sqrt(np.mean(np.sum((f - self.obs) ** 2, -1)[m])))

    def pct(self, f, m=None, ref=None):
        """RMSE change in % against the IFS + AIFS mean (or `ref`, a forecast) on mask m."""
        r = self.rmse(self.base["ecmwf"] if ref is None else ref, m) if (m is not None or ref is not None) else self.ref
        return 100 * (self.rmse(f, m) / r - 1)

    def pct_lead(self, f):
        return np.array([self.pct(f, self.mask & (np.arange(len(self.leads)) == j)[None, :, None])
                         for j in range(len(self.leads))])

    def rmse_lead(self, f):
        return np.array([self.rmse(f, self.mask & (np.arange(len(self.leads)) == j)[None, :, None])
                         for j in range(len(self.leads))])

    # ------------------------------------------------------------ runs
    def _stitch(self, dirs):
        if not all((d / "pred.npz").exists() for d in dirs):
            return None
        out = None
        for d in dirs:
            z = np.load(d / "pred.npz", allow_pickle=True)
            t = z["test"][:, :, :N_NO]
            out = t if out is None else np.where(np.isfinite(t), t, out)
            if self.groups is None:
                self.groups = np.asarray(z["groups"])[:N_NO]
        return self.base["mix"] + out

    def dual_run(self, pattern, seed):
        """Corrected forecast of the 5 dual runs `pattern` ({k} fold, {s} seed) on their own
        baseline (the mean of the run's input models, saved in pred.npz), or None."""
        dirs = [self.runs_dual / pattern.format(k=k, s=seed) for k in range(5)]
        if not all((d / "pred.npz").exists() for d in dirs):
            return None
        out = None
        for d in dirs:
            z = np.load(d / "pred.npz", allow_pickle=True)
            t = z["test"][:, :, :N_NO]
            out = t if out is None else np.where(np.isfinite(t), t, out)
        if self.groups is None:
            self.groups = np.asarray(z["groups"])[:N_NO]
        return z["base"][:, :, :N_NO] + out

    def a1(self, model, seed, ext=False, lr="_lr1e-05"):
        """Point-wise MLP (`point`) or SFNO + MLP (`sfno`), four-model input group."""
        e = "_ext" if ext else ""
        return self._stitch([self.runs_a1 / f"{model}_terrain_g{k}_s{seed}{e}_mix{lr}" for k in range(5)])

    def dual(self, tag, seed, ext=False):
        """Dual-scale model; `tag` is the run name between the seed and `_lr1e-05`."""
        e = "_ext" if ext else ""
        return self._stitch([self.runs_dual / f"dual_sfno-fno_g{k}_s{seed}_{tag}{e}_lr1e-05" for k in range(5)])

    def seeds(self, fn, *args, **kw):
        """{seed: corrected forecast} for the seeds 0-2 that have all 5 folds."""
        return {s: f for s in range(3) if (f := fn(*args, s, **kw)) is not None}

    def history(self, kind, name):
        root = self.runs_dual if kind == "dual" else self.runs_a1
        return json.loads((root / name / "history.json").read_text())

    def best_epoch(self, kind, pattern, seeds):
        """Mean (over folds and seeds) of the kept epoch; pattern has {k} and {s}."""
        ep = []
        for s in seeds:
            for k in range(5):
                h = self.history(kind, pattern.format(k=k, s=s))["history"]
                ep.append(int(np.argmin([x["val"] for x in h])))
        return float(np.mean(ep))

    def n_params(self, kind, name):
        return self.history(kind, name).get("n_params")


def summarise(P, kind, label, tag_or_model, ext=False, lr="_lr1e-05"):
    """One table row: seeds, mean and seed sd of the change, parameters, kept epoch."""
    if kind == "dual":
        fs = P.seeds(P.dual, tag_or_model, ext=ext)
        pat = f"dual_sfno-fno_g{{k}}_s{{s}}_{tag_or_model}{'_ext' if ext else ''}_lr1e-05"
    else:
        fs = P.seeds(P.a1, tag_or_model, ext=ext, lr=lr)
        pat = f"{tag_or_model}_terrain_g{{k}}_s{{s}}{'_ext' if ext else ''}_mix{lr}"
    if not fs:
        return None
    v = [P.pct(f) for f in fs.values()]
    return dict(model=label, seeds=len(v), mean=np.mean(v), sd=np.std(v) if len(v) > 1 else np.nan,
                per_seed=v, params=P.n_params(kind, pat.format(k=0, s=min(fs))),
                epoch=P.best_epoch(kind, pat, fs), forecast=fs)
