"""MET Nordic part of the dual-scale GNO report: table, figure and the LaTeX section, from
ROOT/results/metnordic/grid_eval.npz (scripts/eval_metnordic_grid.py).

    /cluster/projects/nn8106k/siyan/envs/windcast-x86/bin/python scripts/report_dual_gno/make_metnordic.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

ROOT = Path("/cluster/work/projects/nn8106k/siyan/windcast")
REPO = Path(__file__).resolve().parents[2]
FIGS = REPO / "docs" / "report_dual_gno" / "figs"          # the figures of report.tex
TABS = Path("/cluster/work/projects/nn8106k/siyan/windcast/results/report_dual_gno")  # numbers, LaTeX rows
FIGS.mkdir(parents=True, exist_ok=True)
TABS.mkdir(parents=True, exist_ok=True)
GRAY, INK, MUTED = "#8a8a85", "#1f1f1e", "#6b6b66"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
plt.rcParams.update({"font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "legend.frameon": False, "pdf.fonttype": 42,
                     "lines.linewidth": 1.6})

d = np.load(ROOT / "results" / "metnordic" / "grid_eval.npz", allow_pickle=True)
names = [str(x) for x in d["names"]]
cnt = d["cnt_pt"]
tot = cnt.sum()
rm = {k: float(np.sqrt(d[f"se_pt_{k}"].sum() / tot)) for k in names}
rm_pt = {k: np.sqrt(d[f"se_pt_{k}"] / np.maximum(cnt, 1)) for k in names}
seeds = {k: np.sqrt(d[f"se_seed_{k}"] / tot) for k in names if f"se_seed_{k}" in d}
dist = d["dist_station"]
far = dist > 10.0
rm_far = {k: float(np.sqrt(d[f"se_pt_{k}"][far].sum() / cnt[far].sum())) for k in names}
base, base_far = rm["four_mean"], rm_far["four_mean"]
LBL = {"ifs_aifs": "IFS + AIFS mean", "meps_arome": "MEPS + AROME-Arctic mean", "four_mean": "Four-model mean",
       "mlp": "Point-wise MLP", "sfno_mlp": "SFNO + MLP", "dual_gno": "Dual GNO, as specified",
       "dual_small_mlp": "Dual GNO, small + MLP head", "dual_small_mlp_phys": "\\quad + $\\theta$ + mountain"}

with open(TABS / "tab_metnordic_grid.tex", "w") as f:  # grid columns only
    for k in names:
        sd = f" $\\pm$ {np.std(100 * (seeds[k] / base - 1)):.1f}" if k in seeds else ""
        f.write(f"{LBL[k]} & {rm[k]:.3f} & {100 * (rm[k] / base - 1):+.1f}{sd} & "
                f"{100 * (rm_far[k] / base_far - 1):+.1f} \\\\\n")
        if k == "four_mean":
            f.write("\\midrule\n")
st = {k: float(np.sqrt(d[f"st_se_{k}"] / d["st_cnt"])) for k in ("metnordic", "ifs_aifs", "meps_arome", "four_mean")}

# ------------------------------------------------------------------ figure
cmap = LinearSegmentedColormap.from_list("div", [BLUE, "#e8e8e4", ORANGE])
xy, sxy, sin_ = d["xy"], d["st_xy"], d["st_in"]
fig = plt.figure(figsize=(7.0, 3.6))
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.1], wspace=0.12, left=0.01, right=0.99, top=0.92, bottom=0.36)
norm = TwoSlopeNorm(vcenter=0.0, vmin=-25, vmax=25)
for i, k in enumerate(["mlp", "dual_small_mlp_phys"]):
    ax = fig.add_subplot(gs[0, i])
    chg = 100 * (rm_pt[k] / rm_pt["four_mean"] - 1)
    sc = ax.scatter(xy[:, 0], xy[:, 1], c=chg, s=1.1, cmap=cmap, norm=norm, linewidths=0, rasterized=True)
    ax.plot(sxy[sin_, 0], sxy[sin_, 1], "^", ms=3.2, mfc="none", mec=INK, mew=0.6)
    ax.set_aspect("equal")
    ax.set_xlim(xy[:, 0].min() - 3, xy[:, 0].max() + 3)
    ax.set_ylim(xy[:, 1].min() - 3, xy[:, 1].max() + 3)
    ax.set_xticks([])
    ax.set_yticks([])
    for s_ in ax.spines.values():
        s_.set_visible(False)
    ax.set_title(["Point-wise MLP", "Dual GNO + $\\theta$ + mountain"][i], fontsize=8.2, color=INK, loc="left")
p0, p1 = fig.axes[0].get_position(), fig.axes[1].get_position()
cax = fig.add_axes([p0.x0 + 0.02, 0.22, p1.x1 - p0.x0 - 0.04, 0.03])
cb = fig.colorbar(sc, cax=cax, orientation="horizontal")
cb.set_label("RMSE change vs. four-model mean (%), triangles: stations", fontsize=7.5)
cb.outline.set_visible(False)
ax = fig.add_subplot(gs[0, 2])
pos = ax.get_position()
ax.set_position([pos.x0 + 0.07, pos.y0, pos.width - 0.07, pos.height])
bins = np.array([0, 5, 10, 20, 30, 50])
mid = 0.5 * (bins[1:] + bins[:-1])
curves = {}
for k, col, ls in [("meps_arome", GRAY, (0, (1, 1.5))), ("mlp", INK, "-"), ("sfno_mlp", GRAY, "-"),
                   ("dual_gno", ORANGE, "-"), ("dual_small_mlp", AQUA, "-"), ("dual_small_mlp_phys", BLUE, "-")]:
    v = []
    for a, b in zip(bins[:-1], bins[1:]):
        m = (dist >= a) & (dist < b)
        v.append(100 * (np.sqrt(d[f"se_pt_{k}"][m].sum() / cnt[m].sum()) /
                        np.sqrt(d["se_pt_four_mean"][m].sum() / cnt[m].sum()) - 1))
    curves[k] = v
    ax.plot(mid, v, color=col, ls=ls, marker="o", ms=3, label=LBL[k].replace("\\quad + ", "Dual small + MLP + "))
ax.axhline(0, color=MUTED, lw=0.7)
ax.set_xlabel("Distance to nearest station (km)")
ax.set_ylabel("Change vs. four-model mean (%)")
ax.grid(True, color="#e4e4e0", lw=0.6)
ax.legend(fontsize=6.2, loc="upper center", bbox_to_anchor=(0.35, -0.2), ncol=1)
ax.set_title("By distance to nearest station", fontsize=8.2, color=INK, loc="left")
fig.savefig(FIGS / "fig_metnordic.pdf")
plt.close(fig)

frac = [float(((dist >= a) & (dist < b)).mean()) for a, b in zip(bins[:-1], bins[1:])]
summary = dict(rmse=rm, rmse_far=rm_far, seeds={k: v.tolist() for k, v in seeds.items()}, station_check=st,
               n_points=int(len(dist)), n_runs=int(len(d["test_inits"])), n_station_in=int(sin_.sum()),
               dist_bins=bins.tolist(), dist_frac=frac, dist_curves=curves, case_init=str(d["case_init"]),
               frac_far=float(far.mean()))
json.dump(summary, open(TABS / "metnordic.json", "w"), indent=1)
print(json.dumps(summary, indent=1))

# ------------------------------------------------------------------ table with the station check
# tables/metnordic_stations.json: the same models and the MET Nordic analysis against the
# observations at the stations inside the grid, from the stitched station predictions
stj = TABS / "metnordic_stations.json"
if stj.exists():
    so = json.load(open(stj))
    with open(TABS / "tab_metnordic_full.tex", "w") as f:
        for k in names:
            sd = f" $\\pm$ {np.std(100 * (seeds[k] / base - 1)):.1f}" if k in seeds else ""
            f.write(f"{LBL[k]} & {rm[k]:.3f} & {100 * (rm[k] / base - 1):+.1f}{sd} & "
                    f"{100 * (rm_far[k] / base_far - 1):+.1f} & {so[k]:.3f} \\\\\n")
            if k == "four_mean":
                f.write("\\midrule\n")
        f.write(f"\\midrule\nMET Nordic analysis & -- & -- & -- & {so['metnordic']:.3f} \\\\\n")
