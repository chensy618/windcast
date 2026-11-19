"""Held-out verification of every website configuration, for the "How good is it" panel.

    python web/pipeline/verification.py --out web/site/data       # once, after training

Test period 2025-04-01 .. 2025-09-12, runs present in both the ECMWF and the NWP samples; every
station is predicted by the models that did not see its station group (3 seeds, mean RMSE).
Vector RMSE of (u, v) against the observations; change in % against the IFS + AIFS mean (negative
= better). Also the uncorrected NWP 4-model mean.

OUT/static/verification.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from windcast.station_correction import station_groups  # noqa: E402

ROOT = Path("/cluster/work/projects/nn8106k/siyan/windcast")
TAG = ROOT / "cache" / "multimodel_2024-03-01_2025-09-12"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=REPO / "web" / "site" / "data")
    a = ap.parse_args()
    e, n = (dict(np.load(f"{TAG}{s}.npz", allow_pickle=True)) for s in ("", "_nwp"))
    inits, leads = pd.DatetimeIndex(e["inits"]), [int(x) for x in e["leads"]]
    ti = pd.Index(e["time"]).get_indexer(inits)
    obs = np.stack([e["obs"][ti + ld] for ld in leads], 1)
    te = (inits >= "2025-04-01") & (inits <= pd.Timestamp("2025-09-12 12:00")) & e["keep"] & n["keep"]
    groups = station_groups(e["xy"].astype(float))
    base = {"ecmwf": e["st_fc"].mean(2), "nwp": n["st_fc"].mean(2)}

    def rmse(p):
        m = te[:, None, None] & np.isfinite(obs[..., 0]) & np.isfinite(p[..., 0])
        return np.array([np.sqrt(((obs[:, j] - p[:, j]) ** 2).sum(-1)[m[:, j]].mean()) for j in range(len(leads))])

    r0 = rmse(base["ecmwf"])
    out = {"leads": leads, "period": "2025-04-01 to 2025-09-12", "n_stations": int(len(groups)),
           "reference": "IFS + AIFS mean", "rmse_reference": r0.round(3).tolist(), "configs": {}}
    rn = rmse(base["nwp"])
    out["configs"]["nwp-raw"] = dict(rmse=rn.round(3).tolist(), change_pct=((rn / r0 - 1) * 100).round(2).tolist())
    for g in ("ecmwf", "nwp"):
        for h, name in (("sfno", "sfno"), ("fno", "fno"), ("point", "mlp")):
            rs = []
            for s in range(3):
                acc = np.full_like(base[g], np.nan)
                for k in range(5):
                    p = ROOT / "runs_a1" / f"{h}_terrain_g{k}_s{s}{'_nwp' if g == 'nwp' else ''}_lr1e-05" / "pred.npz"
                    acc[:, :, groups == k] = np.load(p)["test"][:, :, groups == k]
                rs.append(rmse(base[g] + acc))
            r = np.mean(rs, 0)
            out["configs"][f"{g}-{name}"] = dict(rmse=r.round(3).tolist(), change_pct=((r / r0 - 1) * 100).round(2).tolist())
            print(f"{g}-{name}: mean change {np.mean((r / r0 - 1) * 100):.2f} %")
    (a.out / "static").mkdir(parents=True, exist_ok=True)
    (a.out / "static" / "verification.json").write_text(json.dumps(out, separators=(",", ":")))


if __name__ == "__main__":
    main()
