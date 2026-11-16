"""Scores per lead time and variable. Wind is scored as a vector ((u, v) together), as
speed and as direction; the other targets as RMSE and bias. Every evaluation scores the
model, the raw driver and (when available) persistence of the analysis at init."""
from collections import defaultdict

import numpy as np
import torch

from windcast.data.dataset import TARGET

IU, IV = TARGET.index("u10"), TARGET.index("v10")
SCALARS = [k for k in TARGET if k not in ("u10", "v10")]


def huber_loss(pred_n: torch.Tensor, y_n: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    """Mean Huber loss on normalised (pred, target), NaN cells of the target ignored."""
    m = torch.isfinite(y_n)
    return torch.nn.functional.huber_loss(pred_n[m], y_n[m], delta=delta)


class Scores:
    """Running sums, keyed by (forecast name, lead)."""

    def __init__(self):
        self.s = defaultdict(lambda: defaultdict(float))

    @torch.no_grad()
    def add(self, name: str, pred: torch.Tensor, y: torch.Tensor, leads):
        p, t = pred.double(), y.double()
        for b, L in enumerate(leads):
            L = int(L)
            s = self.s[(name, L)]
            d = p[b] - t[b]
            ok = torch.isfinite(d).all(0)
            n = ok.sum().item()
            s["n"] += n
            du, dv = d[IU][ok], d[IV][ok]
            s["se_wind"] += (du ** 2 + dv ** 2).sum().item()
            sp, st = p[b, [IU, IV]].norm(dim=0)[ok], t[b, [IU, IV]].norm(dim=0)[ok]
            s["se_spd"] += ((sp - st) ** 2).sum().item()
            s["e_spd"] += (sp - st).sum().item()
            ang = torch.rad2deg(torch.atan2(p[b, IU], p[b, IV]) - torch.atan2(t[b, IU], t[b, IV]))
            ang = ((ang + 180) % 360 - 180)[ok]
            m = st > 2
            s["ae_dir"] += ang.abs()[m].sum().item()
            s["n_dir"] += m.sum().item()
            for k in SCALARS:
                e = d[TARGET.index(k)][ok]
                s[f"se_{k}"] += (e ** 2).sum().item()
                s[f"e_{k}"] += e.sum().item()

    def result(self) -> dict:
        """{name: {lead: {metric: value}}} plus an 'all' entry pooled over leads."""
        out = defaultdict(dict)
        pooled = defaultdict(lambda: defaultdict(float))
        for (name, L), s in self.s.items():
            out[name][L] = self._finish(s)
            for k, v in s.items():
                pooled[name][k] += v
        for name, s in pooled.items():
            out[name]["all"] = self._finish(s)
        return {k: dict(sorted(v.items(), key=lambda kv: (kv[0] == "all", kv[0])))
                for k, v in out.items()}

    @staticmethod
    def _finish(s) -> dict:
        n = max(s["n"], 1)
        r = {"rmse_wind": np.sqrt(s["se_wind"] / n), "rmse_spd": np.sqrt(s["se_spd"] / n),
             "bias_spd": s["e_spd"] / n, "mae_dir": s["ae_dir"] / max(s["n_dir"], 1)}
        for k in SCALARS:
            r[f"rmse_{k}"] = np.sqrt(s[f"se_{k}"] / n)
            r[f"bias_{k}"] = s[f"e_{k}"] / n
        return {k: float(v) for k, v in r.items()}
