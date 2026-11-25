"""LaTeX rows of the station table of docs/report_unified_inputs/report.tex from stations.json
(station_results.py): RMSE (m/s) against the observations; trained models: mean over seeds of each
seed's RMSE, +- the seed standard deviation; '(1)' marks a single finished seed, TBD a missing run.

    /cluster/projects/nn8106k/siyan/envs/windcast-x86/bin/python scripts/report_unified_inputs/table_rows.py
"""
import json

import numpy as np

R = json.load(open("/cluster/work/projects/nn8106k/siyan/windcast/results/report_unified_inputs/stations.json"))
ROWS = [("IFS", "raw_ifs"), ("AIFS", "raw_aifs"), ("MEPS", "raw_meps"), ("AROME-Arctic", "raw_arome"),
        ("Mean IFS + AIFS", "mean_ifs_aifs"), ("Mean MEPS + AROME-Arctic", "mean_meps_arome"),
        ("Mean of all four", "four_mean"), ("MET Nordic analysis", "metnordic"), None,
        ("Point-wise MLP", "mlp"), None,
        ("Dual, 4 fields", "dual_4"), ("Dual, 6 fields", "dual_6"),
        ("Dual, 4 fields, $\\theta$ + mountain", "dual_4_phys"), ("Dual, 6 fields, $\\theta$ + mountain", "dual_6_phys")]


def cell(r, k):
    if k not in r:
        return "\\tbd"
    s = r.get(f"{k}_seeds")
    if not s or len(s) == 1:
        return f"{r[k]:.3f}" + (" (1)" if s else "")
    return f"{np.mean(s):.3f} $\\pm$ {np.std(s):.3f}"


for row in ROWS:
    if row is None:
        print("\\midrule")
        continue
    lab, k = row
    print(f"{lab} & {cell(R['original'], k)} & {cell(R['extended'], k)} \\\\")
print("% samples", R["original"]["n_samples"], R["extended"]["n_samples"], "runs", R["original"]["n_runs"],
      R["extended"]["n_runs"], "stations", R["original"]["n_stations"], R["extended"]["n_stations"])
