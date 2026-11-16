"""Train-split normalisation statistics (runs on the login node, no torch needed).

    python scripts/compute_stats.py --config configs/f1_meps.yaml [--n 600]

Writes data.stats (default {data.root}/stats.json): mean / std per driver and static channel, and the std of
the target residual (MET Nordic - driver) per target variable, which scales the model
output and the loss so that every variable weighs alike.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from windcast.data.dataset import DRIVER, TARGET, ForecastDataset  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--n", type=int, default=600)
    a = p.parse_args()
    cfg = yaml.safe_load(open(a.config))
    ds = ForecastDataset(cfg, "train")
    rng = np.random.default_rng(0)
    pick = rng.choice(len(ds), min(a.n, len(ds)), replace=False)
    sums = {}
    for k, i in enumerate(pick):
        r = ds.raw(int(i))
        for name, arr in (("driver", r["driver"]), ("static", r["static"]),
                          ("residual", r["y"] - r["base"])):
            a2 = arr.reshape(arr.shape[0], -1).astype(np.float64)
            s = sums.setdefault(name, [0.0, 0.0, 0])
            s[0] = s[0] + np.nansum(a2, 1)
            s[1] = s[1] + np.nansum(a2 ** 2, 1)
            s[2] = s[2] + np.isfinite(a2).sum(1)
        if k % 100 == 0:
            print(f"{k}/{len(pick)}", flush=True)
    out = {}
    for name, (s1, s2, n) in sums.items():
        mean = s1 / n
        std = np.sqrt(np.maximum(s2 / n - mean ** 2, 0))
        out[f"{name}_mean"] = mean.tolist()
        out[f"{name}_std"] = np.where(std > 1e-6, std, 1.0).tolist()
    out["driver_names"], out["target_names"] = DRIVER, TARGET
    out["static_names"] = ds.grid.static_names
    out["n_samples"] = int(len(pick))
    path = Path(cfg["data"].get("stats", Path(cfg["data"]["root"]) / "stats.json"))
    path.write_text(json.dumps(out, indent=1))
    print(path, {k: np.round(out["residual_std"], 3).tolist() for k in ["residual_std"]})


if __name__ == "__main__":
    main()
