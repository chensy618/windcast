"""Station results of the unified-input report (docs/report_unified_inputs/report.tex): every forecast
against the observations at the held-out stations, on the common test samples of the four- and
six-field samples.

    /cluster/projects/nn8106k/siyan/envs/windcast-x86/bin/python scripts/report_unified_inputs/station_results.py

* all 99 stations, original and extended period, with the MET Nordic analysis at the valid time
  (bilinear from its 1 km grid) as one more forecast; every forecast on the samples where the
  analysis exists.
Trained models: mean over the seeds of each seed's RMSE (and the per-seed values). Six-field runs
that are not finished yet are left out.

OUT ROOT/results/report_unified_inputs/stations.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import cKDTree

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "notebooks"))
import duallib as D  # noqa: E402
from windcast.geo import to_plane  # noqa: E402

OUT = D.ROOT / "results" / "report_unified_inputs"
S = D.SMALL
DUAL = {"dual_4": f"rg60_rr10_{S}_head-mlp", "dual_4_phys": f"rg60_rr10_phys-all_{S}_head-mlp",
        "dual_6": f"rg60_rr10_{S}_head-mlp_f6", "dual_6_phys": f"rg60_rr10_phys-all_{S}_head-mlp_f6"}


def bilinear_weights(mxy, st_xy):
    """Bilinear interpolation on the curvilinear 1 km grid mxy (H, W, 2) km: corner indices and
    weights (S, 4) of every station, from the local grid axes at its nearest cell."""
    H, W = mxy.shape[:2]
    _, k = cKDTree(mxy.reshape(-1, 2)).query(st_xy)
    i, j = np.unravel_index(k, (H, W))
    i, j = np.clip(i, 1, H - 2), np.clip(j, 1, W - 2)
    ex = (mxy[i, j + 1] - mxy[i, j - 1]) / 2                       # (S, 2) km per column
    ey = (mxy[i + 1, j] - mxy[i - 1, j]) / 2                       # km per row
    d = st_xy - mxy[i, j]
    det = ex[:, 0] * ey[:, 1] - ex[:, 1] * ey[:, 0]
    dj = (d[:, 0] * ey[:, 1] - d[:, 1] * ey[:, 0]) / det
    di = (ex[:, 0] * d[:, 1] - ex[:, 1] * d[:, 0]) / det
    fi, fj = i + di, j + dj
    i0, j0 = np.floor(fi).astype(int), np.floor(fj).astype(int)
    a, b = fi - i0, fj - j0
    II = np.stack([i0, i0, i0 + 1, i0 + 1], -1)
    JJ = np.stack([j0, j0 + 1, j0, j0 + 1], -1)
    Wt = np.stack([(1 - a) * (1 - b), (1 - a) * b, a * (1 - b), a * b], -1)
    return II, JJ, Wt


def metnordic_at_stations(P):
    """MET Nordic analysis (N, L, 99, 2) at the valid times, bilinear from the 1 km grid of the
    station window (download_metnordic.py --window stations; NaN where a time is missing)."""
    MN = D.ROOT / "metnordic_st"
    s = xr.open_dataset(MN / "static.nc")
    mxy = to_plane(s.longitude.values.astype(float), s.latitude.values.astype(float))
    e = np.load(D.ROOT / "cache" / f"multimodel_{D.PERIODS['original']['tag']}.npz", allow_pickle=True)
    II, JJ, Wt = bilinear_weights(mxy, e["xy"].astype(float))
    mn = np.full(P.obs.shape, np.nan, np.float32)
    cache = {}
    for n in np.where(P.mask.any((1, 2)))[0]:
        for j, ld in enumerate(P.leads):
            t = P.inits[n] + pd.Timedelta(hours=int(ld))
            if t not in cache:
                f = MN / f"{t:%Y}" / f"mn_{t:%Y%m%dT%H}.nc"
                cache[t] = None
                if f.exists():
                    with xr.open_dataset(f) as d:
                        cache[t] = np.stack([(d.u10.values[II, JJ] * Wt).sum(-1), (d.v10.values[II, JJ] * Wt).sum(-1)], -1)
            if cache[t] is not None:
                mn[n, j] = cache[t]
    return mn


def scores(P, m):
    rm = lambda f: P.rmse(f, m)
    out = {f"raw_{k}": rm(v) for k, v in P.raw.items()}
    out.update(mean_ifs_aifs=rm(P.base["ecmwf"]), mean_meps_arome=rm(P.base["lam"]), four_mean=rm(P.base["mix"]))
    seeds = {"mlp": P.seeds(P.a1, "point")}
    seeds.update({k: P.seeds(P.dual, tag) for k, tag in DUAL.items()})
    for k, fs in seeds.items():
        if fs:
            v = [rm(f) for f in fs.values()]
            out[k] = float(np.mean(v))
            out[f"{k}_seeds"] = v
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    res = {}
    for per in ("original", "extended"):
        P = D.Period(per)
        k6 = np.load(D.ROOT / "cache" / f"multimodel_{D.PERIODS[per]['tag']}_lam6.npz", allow_pickle=True)["keep"]
        P.mask &= k6[:, None, None]
        mn = metnordic_at_stations(P)
        m = P.mask & np.isfinite(mn).all(-1)          # every forecast on the samples with MET Nordic
        res[per] = dict(n_samples=int(m.sum()), n_runs=int(m.any((1, 2)).sum()),
                        n_stations=int(m.any((0, 1)).sum()), metnordic=P.rmse(mn, m), **scores(P, m))
        print(per, json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in res[per].items()}))
    json.dump(res, open(OUT / "stations.json", "w"), indent=1)
    print(OUT / "stations.json")


if __name__ == "__main__":
    main()
