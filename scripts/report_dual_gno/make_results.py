"""Figures and LaTeX tables for the dual-scale GNO report (docs/report_dual_gno/report.tex).

    /cluster/projects/nn8106k/siyan/envs/windcast-x86/bin/python scripts/report_dual_gno/make_results.py

Everything is evaluated on the same test samples: test period (2025-04-01 .. 2025-09-12),
finite observation, and every input group (IFS+AIFS, MEPS+AROME, mix, old nwp) available.
Station predictions come from the held-out spatial group of each fold (5 folds stitched).
"""
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

R = Path("/cluster/work/projects/nn8106k/siyan/windcast")
C = R / "cache" / "multimodel_2024-03-01_2025-09-12"
REPO = Path(__file__).resolve().parents[2]
FIGS = REPO / "docs" / "report_dual_gno" / "figs"          # the figures of report.tex
TABS = Path("/cluster/work/projects/nn8106k/siyan/windcast/results/report_dual_gno")  # numbers, LaTeX rows
FIGS.mkdir(parents=True, exist_ok=True)
TABS.mkdir(parents=True, exist_ok=True)

GRAY, INK, MUTED = "#8a8a85", "#1f1f1e", "#6b6b66"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
plt.rcParams.update({"font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.grid": True, "grid.color": "#e4e4e0",
                     "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False,
                     "pdf.fonttype": 42, "lines.linewidth": 1.6})

# ---------------------------------------------------------------- common samples
G = {}
for g, s in [("ecmwf", ""), ("lam", "_lam"), ("mix", "_mix"), ("nwp", "_nwp")]:
    d = np.load(f"{C}{s}.npz", allow_pickle=True)
    G[g] = dict(base=d["st_fc"].mean(2), keep=d["keep"])
e = np.load(f"{C}.npz", allow_pickle=True)
inits = pd.DatetimeIndex(e["inits"])
leads = e["leads"]
ti = pd.Index(e["time"]).get_indexer(inits)
obs = np.stack([e["obs"][ti + ld] for ld in leads], 1)                       # (N, L, S, 2)
mask = (inits >= "2025-04-01")[:, None, None] & np.isfinite(obs).all(-1)
for g in G:
    mask &= G[g]["keep"][:, None, None]


def se(f):
    return np.sum((f - obs) ** 2, -1)


def rmse(f, m=mask):
    return float(np.sqrt(np.mean(se(f)[m])))


REF = rmse(G["ecmwf"]["base"])
pct = lambda x: 100 * (x / REF - 1)


def stitch(paths):
    out, base = None, None
    for p in paths:
        d = np.load(p, allow_pickle=True)
        t = d["test"]
        out = t if out is None else np.where(np.isfinite(t), t, out)
        if "base" in d:
            base = d["base"]
    return out, base


def a1(model, group, s, var=""):
    sfx = "" if group == "ecmwf" else "_" + group
    ps = [R / "runs_a1" / f"{model}_terrain_g{f}_s{s}{sfx}{var}_lr1e-05" / "pred.npz" for f in range(5)]
    if not all(p.exists() for p in ps):
        return None
    p, _ = stitch(ps)
    return G[group]["base"] + p


def dual(tag, s):
    ps = [R / "runs_dual" / f"dual_sfno-fno_g{f}_s{s}_{tag}_lr1e-05" / "pred.npz" for f in range(5)]
    if not all(p.exists() for p in ps):
        return None
    p, b = stitch(ps)
    assert np.nanmax(np.abs(b - G["mix"]["base"])[mask]) < 1e-4
    return b + p


def hist(path):
    return json.load(open(path / "history.json"))


groups = np.load(R / "runs_dual" / "dual_sfno-fno_g0_s0_rg60_rr10_lr1e-05" / "pred.npz", allow_pickle=True)["groups"]
numbers = {"n_samples": int(mask.sum()), "ref_rmse": REF, "n_stations": int(len(groups)),
           "stations_per_group": np.bincount(groups).tolist()}

# ---------------------------------------------------------------- table 1: input groups
rows1 = []
for g, lbl in [("ecmwf", "IFS + AIFS (0.25$^\\circ$)"), ("lam", "MEPS + AROME-Arctic (2.5 km)"),
               ("mix", "IFS + AIFS + MEPS + AROME-Arctic"), ("nwp", "IFS + GFS + MEPS + AROME-Arctic (old)")]:
    r = {"label": lbl, "unc": pct(rmse(G[g]["base"]))}
    for m in ["point", "sfno", "fno"]:
        v = [rmse(f) for s in range(3) if (f := a1(m, g, s)) is not None]
        r[m] = (pct(np.mean(v)), float(np.std([pct(x) for x in v]))) if v else None
    rows1.append(r)
numbers["groups"] = rows1
best = min(r["point"][0] for r in rows1)
with open(TABS / "tab_groups.tex", "w") as f:
    for r in rows1:
        cells = [f"{r['unc']:+.1f}"]
        for m in ["point", "sfno", "fno"]:
            mu, sd = r[m]
            c = f"{mu:+.1f} $\\pm$ {sd:.1f}"
            cells.append(f"\\textbf{{{c}}}" if abs(mu - best) < 1e-9 else c)
        f.write(f"{r['label']} & " + " & ".join(cells) + " \\\\\n")

# ---------------------------------------------------------------- table 2: dual-scale GNO
SMALL = "r16m8x16_g16m8x16_z0"
DUAL = [  # (tag, label, section)
    ("bilinear", "Dual NO + bilinear readout", "Original size (PDF design, linear head)"),
    ("rg60_rr10", "Dual NO + GNO", None),
    ("rg60_rr10_phys-all", "Dual NO + GNO + $\\theta$ + mountain", None),
    (f"bilinear_{SMALL}", "Dual NO + bilinear readout", "Small encoders, zero-initialised output"),
    (f"rg60_rr10_{SMALL}", "Dual NO + GNO", None),
    (f"bilinear_r16m8x16_g16m8x16_do0.3_wd0.01_z0", "  + dropout 0.3, weight decay $10^{-2}$ (bilinear)", None),
    (f"rg60_rr10_r16m8x16_g16m8x16_do0.3_wd0.01_z0", "  + dropout 0.3, weight decay $10^{-2}$ (GNO)", None),
    ("bilinear_z0_head-mlp", "Dual NO + bilinear readout", "MLP head instead of the linear projection: original size"),
    ("rg60_rr10_z0_head-mlp", "Dual NO + GNO", None),
    (f"bilinear_{SMALL}_head-mlp", "Dual NO + bilinear readout", "MLP head: small encoders"),
    (f"rg60_rr10_{SMALL}_head-mlp", "Dual NO + GNO", None),
    (f"rg60_rr10_phys-theta_{SMALL}_head-mlp", "Dual NO + GNO + $\\theta$", None),
    (f"rg60_rr10_phys-mountain_{SMALL}_head-mlp", "Dual NO + GNO + mountain", None),
    (f"rg60_rr10_phys-all_{SMALL}_head-mlp", "Dual NO + GNO + $\\theta$ + mountain", None),
]
rows2, PRED = [], {}
for tag, lbl, sec in DUAL:
    v, eps = [], []
    for s in range(3):
        f = dual(tag, s)
        if f is None:
            continue
        v.append(pct(rmse(f)))
        PRED.setdefault(tag, f)
        for k in range(5):
            h = hist(R / "runs_dual" / f"dual_sfno-fno_g{k}_s{s}_{tag}_lr1e-05")
            eps.append(int(np.argmin([x["val"] for x in h["history"]])))
    n_par = hist(R / "runs_dual" / f"dual_sfno-fno_g0_s0_{tag}_lr1e-05")["n_params"] if v else None
    rows2.append(dict(tag=tag, label=lbl, sec=sec, vals=v, n_par=n_par, best_epoch=float(np.mean(eps)) if eps else None))
numbers["dual"] = rows2
ref_rows = [("Uncorrected four-model mean", None, pct(rmse(G["mix"]["base"])), [], None),
            ("Point-wise MLP (mix)", 22530, None, [pct(rmse(a1("point", "mix", s))) for s in range(3)], "point"),
            ("SFNO + MLP (mix, 0.25$^\\circ$)", 607458, None, [pct(rmse(a1("sfno", "mix", s))) for s in range(3)], "sfno")]
eps_ref = {}
for m in ["point", "sfno"]:
    eps_ref[m] = np.mean([np.argmin([x["val"] for x in hist(R / "runs_a1" / f"{m}_terrain_g{k}_s{s}_mix_lr1e-05")["history"]])
                          for k in range(5) for s in range(3)])


def fmt_vals(v):
    if not v:
        return "--", "--"
    sd = f" $\\pm$ {np.std(v):.1f}" if len(v) > 1 else ""
    return f"{np.mean(v):+.1f}{sd}", str(len(v))


with open(TABS / "tab_dual.tex", "w") as f:
    f.write("\\multicolumn{5}{l}{\\emph{References (mix input group)}} \\\\\n")
    for lbl, npar, unc, v, m in ref_rows:
        mv, ns = (f"{unc:+.1f}", "--") if unc is not None else fmt_vals(v)
        ep = f"{eps_ref[m]:.0f}" if m else "--"
        f.write(f"\\quad {lbl} & {npar / 1e3:.0f} k & {ep} & {ns} & {mv} \\\\\n" if npar else
                f"\\quad {lbl} & -- & -- & -- & {mv} \\\\\n")
    for r in rows2:
        if r["sec"]:
            f.write(f"\\midrule\n\\multicolumn{{5}}{{l}}{{\\emph{{{r['sec']}}}}} \\\\\n")
        mv, ns = fmt_vals(r["vals"])
        npar = f"{r['n_par'] / 1e6:.2f} M" if r["n_par"] and r["n_par"] > 1e6 else (f"{r['n_par'] / 1e3:.0f} k" if r["n_par"] else "--")
        ep = f"{r['best_epoch']:.0f}" if r["best_epoch"] is not None else "--"
        lbl = r["label"].replace("  + ", "\\quad + ")
        f.write(f"\\quad {lbl} & {npar} & {ep} & {ns} & {mv} \\\\\n")

# ---------------------------------------------------------------- figure: training curves
fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.7), sharey=False)
curves = [(R / "runs_a1" / "point_terrain_g0_s0_mix_lr1e-05", "Point-wise MLP (22 k)", GRAY),
          (R / "runs_dual" / "dual_sfno-fno_g0_s0_rg60_rr10_lr1e-05", "Dual GNO, original (5.1 M)", ORANGE),
          (R / "runs_dual" / f"dual_sfno-fno_g0_s0_rg60_rr10_{SMALL}_head-mlp_lr1e-05", "Dual GNO, small + MLP head (147 k)", BLUE)]
for path, lbl, col in curves:
    H = hist(path)["history"]
    ep = [x["epoch"] + 1 for x in H]
    axs[0].plot(ep, [x["train"] for x in H], color=col, label=lbl)
    v = [x["val"] for x in H]
    axs[1].plot(ep, v, color=col, label=lbl)
    i = int(np.argmin(v))
    axs[1].plot(ep[i], v[i], "o", color=col, ms=5, mec="white", mew=1.2, zorder=5)
for ax, t in zip(axs, ["Training loss", "Validation loss (held-out stations, Jan–Mar 2025)"]):
    ax.set_title(t, fontsize=8.5, color=INK, loc="left")
    ax.set_xlabel("Epoch")
axs[0].set_ylabel("MSE of normalised residual")
h, l = axs[1].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=3, fontsize=7.5, bbox_to_anchor=(0.5, 0.0))
fig.tight_layout(rect=(0, 0.1, 1, 1))
fig.savefig(FIGS / "fig_curves.pdf")
plt.close(fig)

# ---------------------------------------------------------------- figure: summary bars
bars = [("Uncorrected four-model mean", [pct(rmse(G["mix"]["base"]))], GRAY),
        ("Dual GNO + $\\theta$ + mountain (original)", rows2[2]["vals"], ORANGE),
        ("Dual GNO (original)", rows2[1]["vals"], ORANGE),
        ("Dual bilinear (original)", rows2[0]["vals"], ORANGE),
        ("Dual GNO, small", rows2[4]["vals"], AQUA),
        ("Dual GNO, MLP head (original)", rows2[8]["vals"], BLUE),
        ("Dual GNO, small + MLP head", rows2[10]["vals"], BLUE),
        ("  + $\\theta$ + mountain", rows2[13]["vals"], BLUE),
        ("SFNO + MLP (mix)", ref_rows[2][3], GRAY),
        ("Point-wise MLP (mix)", ref_rows[1][3], INK)]
fig, ax = plt.subplots(figsize=(7.0, 2.9))
y = np.arange(len(bars))[::-1]
for yi, (lbl, v, col) in zip(y, bars):
    m = np.mean(v)
    ax.barh(yi, m, color=col, height=0.62)
    if len(v) > 1:
        ax.plot(v, [yi] * len(v), "o", color="white", ms=3.2, mec=INK, mew=0.6, zorder=5)
    ax.text(-0.12, yi, f"{m:+.1f} %", va="center", ha="right", fontsize=7.5, color="white", fontweight="bold")
ax.set_yticks(y)
ax.set_yticklabels([b[0] for b in bars], fontsize=7.8)
ax.axvline(ref_rows[1][3] and np.mean(ref_rows[1][3]), color=INK, lw=0.8, ls=(0, (3, 2)))
ax.set_xlabel("Test RMSE change vs. IFS + AIFS mean (%)   (bar: mean over seeds, dots: seeds)")
ax.set_xlim(-11.5, 0)
ax.grid(axis="y", visible=False)
fig.tight_layout()
fig.savefig(FIGS / "fig_bars.pdf")
plt.close(fig)

# ---------------------------------------------------------------- figure: lead time and station group
best_tag = f"rg60_rr10_phys-all_{SMALL}_head-mlp"
models = [("IFS + AIFS mean", G["ecmwf"]["base"], GRAY, (0, (1, 1.5))),
          ("Four-model mean", G["mix"]["base"], GRAY, "-"),
          ("Dual GNO (original)", PRED["rg60_rr10"], ORANGE, "-"),
          ("Point-wise MLP (mix)", a1("point", "mix", 0), INK, "-"),
          ("Dual GNO, small + MLP head + $\\theta$ + mtn", PRED[best_tag], BLUE, "-")]
fig, axs = plt.subplots(1, 2, figsize=(7.0, 3.3), gridspec_kw=dict(width_ratios=[1.25, 1]))
for lbl, f, col, ls in models:
    s2 = se(f)
    axs[0].plot(leads, [np.sqrt(np.mean(s2[:, l][mask[:, l]])) for l in range(len(leads))],
                color=col, ls=ls, label=lbl, marker="o" if col == BLUE else None, ms=3)
axs[0].set_xlabel("Lead time (h)")
axs[0].set_ylabel("Vector RMSE (m/s)")
axs[0].set_xticks(leads[::2])
axs[0].legend(fontsize=6.6, loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2)
axs[0].set_title("RMSE by lead time (seed 0)", fontsize=8.5, loc="left", color=INK)
gm = [("Point-wise MLP (mix)", a1("point", "mix", 0), INK), ("SFNO + MLP (mix)", a1("sfno", "mix", 0), GRAY),
      ("Dual GNO, small + MLP + $\\theta$ + mtn", PRED[best_tag], BLUE)]
ng = int(groups.max()) + 1
w = 0.26
per_group = {}
for i, (lbl, f, col) in enumerate(gm):
    vals = []
    for k in range(ng):
        m = mask & (groups == k)[None, None]
        vals.append(100 * (rmse(f, m) / rmse(G["mix"]["base"], m) - 1))
    per_group[lbl] = vals
    axs[1].bar(np.arange(ng) + (i - 1) * w, vals, width=w - 0.03, color=col, label=lbl)
axs[1].set_xticks(np.arange(ng))
axs[1].set_xticklabels([f"G{k}\n({c})" for k, c in enumerate(np.bincount(groups))], fontsize=7.2)
axs[1].set_ylabel("Change vs. four-model mean (%)")
axs[1].set_title("By held-out station group (n stations)", fontsize=8.5, loc="left", color=INK)
axs[1].legend(fontsize=6.6, loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=1)
axs[1].grid(axis="x", visible=False)
fig.tight_layout()
fig.savefig(FIGS / "fig_lead_group.pdf")
plt.close(fig)
numbers["per_group_vs_fourmean"] = per_group
numbers["lead_rmse"] = {lbl: [float(np.sqrt(np.mean(se(f)[:, l][mask[:, l]]))) for l in range(len(leads))]
                        for lbl, f, _, _ in models}
json.dump(numbers, open(TABS / "numbers.json", "w"), indent=1, default=float)
print(json.dumps({k: v for k, v in numbers.items() if k not in ("lead_rmse",)}, indent=1, default=float)[:4000])
