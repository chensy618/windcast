"""Held-out verification of every website configuration, for the "How good is it" panel.

    python web/pipeline/verification.py --out web/site/data       # once, after training

Test period 2025-04-01 .. 2025-09-12, runs of the IFS + AIFS samples; every
station is predicted by the models that did not see its station group (3 seeds, mean RMSE).
Vector RMSE of (u, v) against the observations; change in % against the IFS + AIFS mean (negative
= better).
dual-gno (WindCast): the original-period twin of the deployed model (same run tag, trained on
2024-03 .. 2024-12, runs_dual), so that it is scored on the same test period as the others; its
baseline is the IFS + AIFS + MEPS + AROME-Arctic mean.
Per station ("stations"): the same RMSE at each station alone, mean over the leads, with the number
of observations it rests on (the panel's "This station" view).

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
DUAL_TAG = "rg60_rr10_r16m8x16_g16m8x16_z0_head-mlp"      # web/pipeline/dual.py RUN_TAG


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=REPO / "web" / "site" / "data")
    a = ap.parse_args()
    e = dict(np.load(f"{TAG}.npz", allow_pickle=True))
    inits, leads = pd.DatetimeIndex(e["inits"]), [int(x) for x in e["leads"]]
    ti = pd.Index(e["time"]).get_indexer(inits)
    obs = np.stack([e["obs"][ti + ld] for ld in leads], 1)
    te = (inits >= "2025-04-01") & (inits <= pd.Timestamp("2025-09-12 12:00")) & e["keep"]
    groups = station_groups(e["xy"].astype(float))
    base = {"ecmwf": e["st_fc"].mean(2)}

    def rmse(p):
        m = te[:, None, None] & np.isfinite(obs[..., 0]) & np.isfinite(p[..., 0])
        return np.array([np.sqrt(((obs[:, j] - p[:, j]) ** 2).sum(-1)[m[:, j]].mean()) for j in range(len(leads))])

    def rmse_st(p):                                  # (stations,): per-lead RMSE at each station, mean over leads
        m = te[:, None, None] & np.isfinite(obs[..., 0]) & np.isfinite(p[..., 0])
        se = np.where(m, ((obs - p) ** 2).sum(-1), 0.0)
        n = m.sum(0)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.sqrt(se.sum(0) / n).mean(0), n.sum(0)

    st_ids = [str(x) for x in e["station"]]
    s0, n0 = rmse_st(base["ecmwf"])
    per_st = {"reference": s0}
    r0 = rmse(base["ecmwf"])
    out = {"leads": leads, "period": "2025-04-01 to 2025-09-12", "n_stations": int(len(groups)),
           "reference": "IFS + AIFS mean", "rmse_reference": r0.round(3).tolist(), "configs": {}}
    for g in ("ecmwf",):
        for h, name in (("sfno", "sfno"), ("fno", "fno"), ("point", "mlp")):
            rs, ss = [], []
            for s in range(3):
                acc = np.full_like(base[g], np.nan)
                for k in range(5):
                    p = ROOT / "runs_a1" / f"{h}_terrain_g{k}_s{s}{'_nwp' if g == 'nwp' else ''}_lr1e-05" / "pred.npz"
                    acc[:, :, groups == k] = np.load(p)["test"][:, :, groups == k]
                rs.append(rmse(base[g] + acc))
                ss.append(rmse_st(base[g] + acc)[0])
            r = np.mean(rs, 0)
            per_st[f"{g}-{name}"] = np.mean(ss, 0)
            out["configs"][f"{g}-{name}"] = dict(rmse=r.round(3).tolist(), change_pct=((r / r0 - 1) * 100).round(2).tolist())
            print(f"{g}-{name}: mean change {np.mean((r / r0 - 1) * 100):.2f} %")
    # WindCast: the dual-scale GNO, held-out predictions of its 5 folds x 3 seeds
    S = len(st_ids)
    rs, ss = [], []
    for s in range(3):
        acc = None
        for k in range(5):
            z = np.load(ROOT / "runs_dual" / f"dual_sfno-fno_g{k}_s{s}_{DUAL_TAG}_lr1e-05" / "pred.npz", allow_pickle=True)
            t = z["test"][:, :, :S]
            acc = t if acc is None else np.where(np.isfinite(t), t, acc)
        p = z["base"][:, :, :S] + acc
        rs.append(rmse(p))
        ss.append(rmse_st(p)[0])
    r = np.mean(rs, 0)
    per_st["dual-gno"] = np.mean(ss, 0)
    out["configs"]["dual-gno"] = dict(rmse=r.round(3).tolist(), change_pct=((r / r0 - 1) * 100).round(2).tolist())
    print(f"dual-gno: mean change {np.mean((r / r0 - 1) * 100):.2f} %")
    out["stations"] = {sid: dict(n=int(n0[i]), **{c: (round(float(v[i]), 3) if np.isfinite(v[i]) else None)
                                                   for c, v in per_st.items()})
                       for i, sid in enumerate(st_ids) if n0[i] > 0}
    (a.out / "static").mkdir(parents=True, exist_ok=True)
    (a.out / "static" / "verification.json").write_text(json.dumps(out, separators=(",", ":")))


if __name__ == "__main__":
    main()
