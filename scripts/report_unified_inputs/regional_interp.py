"""Regional forecasts at the stations: nearest native 2.5 km cell (as in the samples) vs bilinear
interpolation, scored against the observations on the test runs of both periods.

    /cluster/projects/nn8106k/siyan/envs/windcast-x86/bin/python scripts/report_unified_inputs/regional_interp.py

For MEPS and AROME-Arctic the 10 m wind of every test run is read from the native files
(ROOT/<model>/YYYY/<model>_YYYYMMDDTHH.nc, earth-relative) and taken at the 99 stations both ways;
the nearest-cell values are checked against the sample file. Scored: each regional model, the
MEPS + AROME mean and the four-model mean (IFS / AIFS bilinear from 0.25 deg, unchanged).

OUT ROOT/results/report_unified_inputs/regional_interp.json
"""
import json
import sys
from pathlib import Path

import netCDF4
import numpy as np
from scipy.spatial import cKDTree

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "notebooks"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import duallib as D  # noqa: E402
from station_results import bilinear_weights  # noqa: E402
from windcast.geo import to_plane  # noqa: E402

MODELS = {"meps": "meps", "arome": "arome_arctic"}


def main():
    res = {}
    for per in ("original", "extended"):
        P = D.Period(per)
        e = np.load(D.ROOT / "cache" / f"multimodel_{D.PERIODS[per]['tag']}.npz", allow_pickle=True)
        st_xy = e["xy"].astype(float)
        test = np.where(P.mask.any((1, 2)))[0]
        near = {m: np.full(P.obs.shape, np.nan, np.float32) for m in MODELS}
        bil = {m: np.full(P.obs.shape, np.nan, np.float32) for m in MODELS}
        geo = {}
        for m, d in MODELS.items():
            for n in test:
                t = P.inits[n]
                f = D.ROOT / d / f"{t:%Y}" / f"{d}_{t:%Y%m%dT%H}.nc"
                if not f.exists():
                    continue
                with netCDF4.Dataset(f) as ds:
                    shape = ds["lon"].shape
                    if shape not in geo:
                        mxy = to_plane(ds["lon"][:].astype(float), ds["lat"][:].astype(float))
                        dist, k = cKDTree(mxy.reshape(-1, 2)).query(st_xy)
                        II, JJ, Wt = bilinear_weights(mxy, st_xy)
                        inside = (dist < 2.0) & (II.min(1) >= 0) & (JJ.min(1) >= 0) & \
                                 (II.max(1) < shape[0]) & (JJ.max(1) < shape[1])
                        geo[shape] = (np.unravel_index(k, shape), np.clip(II, 0, shape[0] - 1),
                                      np.clip(JJ, 0, shape[1] - 1), Wt, inside)
                    (ni, nj), II, JJ, Wt, inside = geo[shape]
                    fl = [int(x) for x in ds["lead"][:]]
                    jl = [fl.index(int(ld)) for ld in P.leads]
                    u = np.ma.filled(ds["u10"][jl].astype(np.float32), np.nan)
                    v = np.ma.filled(ds["v10"][jl].astype(np.float32), np.nan)
                near[m][n] = np.stack([u[:, ni, nj], v[:, ni, nj]], -1)
                b = np.stack([(u[:, II, JJ] * Wt).sum(-1), (v[:, II, JJ] * Wt).sum(-1)], -1)
                b[:, ~inside] = np.nan
                bil[m][n] = b
        # nearest values must reproduce the sample file
        chk = {m: float(np.nanmax(np.abs(near[m] - P.raw[m])[P.mask])) for m in MODELS}
        m_ = P.mask & np.isfinite(near["meps"]).all(-1) & np.isfinite(near["arome"]).all(-1) & \
             np.isfinite(bil["meps"]).all(-1) & np.isfinite(bil["arome"]).all(-1)
        r = lambda f: P.rmse(f, m_)
        gl = P.raw["ifs"] + P.raw["aifs"]
        out = dict(n_samples=int(m_.sum()), check_nearest_vs_samples=chk)
        for tag, src in (("nearest", near), ("bilinear", bil)):
            out[tag] = {"meps": r(src["meps"]), "arome": r(src["arome"]),
                        "mean_meps_arome": r((src["meps"] + src["arome"]) / 2),
                        "four_mean": r((gl + src["meps"] + src["arome"]) / 4)}
        diff = np.hypot(*np.moveaxis(bil["meps"] - near["meps"], -1, 0))[m_]
        out["meps_abs_diff_ms"] = {"mean": float(diff.mean()), "p95": float(np.percentile(diff, 95))}
        res[per] = out
        print(per, json.dumps(out, indent=1))
    out = D.ROOT / "results" / "report_unified_inputs" / "regional_interp.json"
    json.dump(res, open(out, "w"), indent=1)
    print(out)


if __name__ == "__main__":
    main()
