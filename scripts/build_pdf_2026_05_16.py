#!/usr/bin/env python
"""Build all figures for BENCHMARK_REPORT_2026-05-16.pdf.

Aggregates noised-TS sweep, reactant starting condition, threshold-sweep grid,
fmax-plateau, and existing pareto/lollipop/topo-recovery/rmsd-to-ts/d1-vs-d3.
All Sella names use the 3-axis convention:
  Sella {cartesian|internal} {Eckart}? {tuned|untuned} Hess.Freq.=N
"""
from __future__ import annotations

import os
import sys
import glob

import duckdb
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

SCRIPTS = "/lustre06/project/6033559/memoozd/GAD_plus/scripts"
sys.path.insert(0, SCRIPTS)
from plotting_style import apply_plot_style, palette  # noqa: E402

apply_plot_style()

ROOT = "/lustre06/project/6033559/memoozd/GAD_plus"
RUNS = "/lustre07/scratch/memoozd/gadplus/runs"
CSV  = f"{ROOT}/analysis_2026_04_29"
OUT  = f"{ROOT}/figures_2026_05_16"
os.makedirs(OUT, exist_ok=True)

con = duckdb.connect()
NOISES = [10, 30, 50, 100, 150, 200]

# ── master 4-axis table ────────────────────────────────────────────────────
# Prefer the refreshed 05_16 master if it exists (built by integrate_comprehensive_2026_05_16.py)
_master_v2 = f"{CSV}/master_2026_05_16.csv"
_master_v1 = f"{CSV}/master_2026_05_11.csv"
master = pd.read_csv(_master_v2 if os.path.exists(_master_v2) else _master_v1)
print(f"Reading master from: {'05_16' if os.path.exists(_master_v2) else '05_11'}")
# Threshold sweep table (built earlier)
tsweep = pd.read_csv(f"{CSV}/threshold_sweep_2026_05_16.csv")

# ── visual style ───────────────────────────────────────────────────────────
FAMILY_CMAP = {"plain GAD": palette()[1], "Sella": palette()[0], "hybrid": palette()[2]}
CONFIG_MARKER = {
    "GAD dt=0.003":                                  "o",
    "GAD dt=0.005":                                  "s",
    "GAD dt=0.007":                                  "D",
    "Sella cartesian Eckart untuned Hess.Freq.=1":   "o",
    "Sella cartesian tuned Hess.Freq.=1":            "s",
    "Sella internal tuned Hess.Freq.=1":             "v",
    "Sella cartesian Eckart untuned Hess.Freq.=3":   "X",
    "Hybrid damped Eckart eig tr=0.05":              "^",
    "Hybrid undamped Eckart eig tr=0.05":            "<",
}
SHORT = {
    "GAD dt=0.003": "G003", "GAD dt=0.005": "G005", "GAD dt=0.007": "G007",
    "Sella cartesian Eckart untuned Hess.Freq.=1":  "S-cart+eck-utuned-d1",
    "Sella cartesian tuned Hess.Freq.=1":           "S-cart-tuned-d1",
    "Sella internal tuned Hess.Freq.=1":            "S-int-tuned-d1",
    "Sella cartesian Eckart untuned Hess.Freq.=3":  "S-cart+eck-utuned-d3",
    "Hybrid damped Eckart eig tr=0.05":             "H-damp-Eck-tr0.05",
    "Hybrid undamped Eckart eig tr=0.05":           "H-undamp-Eck-tr0.05",
}


def per_config_color(config, family):
    base = FAMILY_CMAP[family]
    siblings = sorted({c for c, f in master[["config", "family"]].itertuples(index=False) if f == family})
    idx = siblings.index(config) if config in siblings else 0
    alphas = [1.0, 0.78, 0.58, 0.42]
    return base + tuple([alphas[idx % len(alphas)]])


# ────────────────────────────────────────────────────────────────────────
# F1.  Headline 4-axis (starting from NOISED TRUE TS)
# ────────────────────────────────────────────────────────────────────────
def render_4axis(out_path, subtitle, conv_field="conv_pct", conv_label=None, mdf=None):
    if mdf is None:
        mdf = master
    if conv_label is None:
        conv_label = r"TS conv % (Im. Freq. and $F_\mathrm{max}<0.01$)"
    fig, axes = plt.subplots(1, 4, figsize=(24, 6.2), sharex=True)
    panels = [
        (conv_field,     conv_label,                                          False),
        ("topo_pct",     "IRC TOPO-intended %",                               False),
        ("med_step",     "Median converged-step count",                       True),
        ("wall_per_conv","Wall-time per converged TS (s)",                    True),
    ]
    for ax, (col, ylab, logy) in zip(axes, panels):
        for (family, config), grp in mdf.groupby(["family", "config"], sort=False):
            grp = grp.sort_values("noise_pm")
            if col not in grp.columns or grp[col].isna().all():
                continue
            n_pts = grp[col].notna().sum()
            ls = "-" if n_pts >= 6 else "--"
            ax.plot(grp.loc[grp[col].notna(), "noise_pm"],
                    grp.loc[grp[col].notna(), col],
                    marker=CONFIG_MARKER[config], linestyle=ls, lw=2.2, ms=10,
                    color=per_config_color(config, family), label=config)
        ax.set_xlabel("TS noise (pm)", fontsize=16)
        ax.set_ylabel(ylab, fontsize=15)
        ax.tick_params(axis='both', labelsize=13)
        if logy:
            ax.set_yscale("log")
        else:
            ax.set_ylim(0, 100)
        ax.grid(alpha=0.3)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=14,
               bbox_to_anchor=(0.5, -0.20), frameon=False)
    fig.suptitle(subtitle, y=1.03, fontsize=18)
    fig.tight_layout()
    fig.savefig(out_path + ".pdf", bbox_inches="tight")
    fig.savefig(out_path + ".png", bbox_inches="tight", dpi=140)
    plt.close(fig)


render_4axis(
    f"{OUT}/fig_main_4axis",
    "Best-of-family across TS noise — starting from NOISED TRUE TS  ($n=287$ T1x test split, $F_\\mathrm{max}<0.01$)",
)
print("Wrote fig_main_4axis")


# ────────────────────────────────────────────────────────────────────────
# F1-threshold.  4-axis figures under different fmax thresholds (swappable)
# ────────────────────────────────────────────────────────────────────────
THRESH_FIELDS = [
    ("fmax_005",  0.05,   "ASE default",                  r"TS conv % (Im. Freq. and $F_\mathrm{max}<0.05$ — ASE default)"),
    ("fmax_023",  0.023,  "Gaussian standard",            r"TS conv % (Im. Freq. and $F_\mathrm{max}<0.023$ — Gaussian)"),
    ("fmax_010",  0.01,   "Project canonical",            r"TS conv % (Im. Freq. and $F_\mathrm{max}<0.01$ — project canonical)"),
    ("fmax_005t", 0.005,  "Tight",                        r"TS conv % (Im. Freq. and $F_\mathrm{max}<0.005$ — tight)"),
    ("fmax_001",  0.001,  "Sella README recommendation",  r"TS conv % (Im. Freq. and $F_\mathrm{max}<0.001$ — Sella README)"),
]

# For each threshold we need to construct a master-like df where conv_pct = the chosen threshold.
def master_for_threshold(field):
    """Replace conv_pct in master with the threshold-sweep conv column."""
    t = tsweep.rename(columns={"method": "config"})[["config", "noise_pm", field]].copy()
    t = t.rename(columns={field: "conv_pct_new"})
    m = master.merge(t, on=["config", "noise_pm"], how="left")
    m["conv_pct"] = m["conv_pct_new"].fillna(m["conv_pct"])  # fallback to canonical
    m = m.drop(columns=["conv_pct_new"])
    return m

for fld, val, descr, ylab in THRESH_FIELDS:
    sub = master_for_threshold(fld)
    nice = f"{val:g}"
    render_4axis(
        f"{OUT}/fig_main_4axis_fmax{str(val).replace('.','p')}",
        f"4-axis comparison at $F_\\mathrm{{max}}<{nice}$ — {descr}, starting from NOISED TRUE TS",
        conv_label=ylab,
        mdf=sub,
    )
    print(f"Wrote fig_main_4axis_fmax{str(val).replace('.','p')}")


# (Reactant @ 0pm bar chart removed 2026-05-18 per user request.
# To restore: git checkout 2f7c8a6~1 -- scripts/build_pdf_2026_05_16.py)


# ────────────────────────────────────────────────────────────────────────
# F3.  fmax-plateau:  GAD vs hybrid vs Sella, conv% vs threshold tightness
# ────────────────────────────────────────────────────────────────────────
# One panel per noise level (6 panels), x = log fmax threshold, y = conv%
thresh_x  = [0.05, 0.023, 0.01, 0.005, 0.001]
thresh_col = ["fmax_005", "fmax_023", "fmax_010", "fmax_005t", "fmax_001"]
plot_methods = [
    ("plain GAD", "GAD dt=0.005"),
    ("plain GAD", "GAD dt=0.007"),
    ("hybrid",    "Hybrid damped Eckart eig tr=0.05"),
    ("Sella",     "Sella cartesian Eckart untuned Hess.Freq.=1"),
    ("Sella",     "Sella cartesian Eckart untuned Hess.Freq.=3"),
]

fig, axes = plt.subplots(2, 3, figsize=(22, 11), sharey=True)
for ax, noise in zip(axes.flat, NOISES):
    for fam, cfg in plot_methods:
        sub = tsweep[(tsweep["method"] == cfg) & (tsweep["noise_pm"] == noise)]
        if len(sub) == 0:
            continue
        ys = [float(sub[c].iloc[0]) for c in thresh_col]
        ax.plot(thresh_x, ys, marker=CONFIG_MARKER[cfg], ms=11, lw=2.4,
                color=per_config_color(cfg, fam), label=cfg)
    ax.set_xscale("log")
    ax.invert_xaxis()  # tighter to the right
    ax.set_xlabel(r"$F_\mathrm{max}$ threshold (eV/Å, log; tighter $\to$ right)", fontsize=14)
    if ax in axes[:, 0]:
        ax.set_ylabel("TS conv % (Im. Freq. and threshold)", fontsize=14)
    ax.set_title(f"{noise} pm noise", fontsize=15)
    ax.set_ylim(-2, 100)
    ax.tick_params(axis='both', labelsize=12)
    ax.grid(alpha=0.3, which="both")
    for tv in [0.05, 0.023, 0.01, 0.005, 0.001]:
        ax.axvline(tv, color="lightgray", lw=0.5, zorder=0)
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=13,
           bbox_to_anchor=(0.5, -0.04), frameon=False)
fig.suptitle("The $F_\\mathrm{max}$-plateau:  conv % vs threshold tightness — starting from NOISED TRUE TS\n"
             "GAD plateaus near $F_\\mathrm{max}\\approx 0.01$ (no Newton landing → can't drive force below the plateau).  "
             "Hybrid and Sella reach $\\approx 0.005$ via Newton steps.",
             fontsize=16, y=0.99)
fig.tight_layout(rect=[0, 0.03, 1, 0.96])
fig.savefig(f"{OUT}/fig_fmax_plateau.pdf", bbox_inches="tight")
fig.savefig(f"{OUT}/fig_fmax_plateau.png", bbox_inches="tight", dpi=140)
plt.close(fig)
print("Wrote fig_fmax_plateau")


# ────────────────────────────────────────────────────────────────────────
# Carry over: pareto, lollipop, topo-recovery, rmsd-to-ts, d=1 vs d=3
# (Reuse identical code from build_pdf_2026_05_11.py but write to OUT)
# ────────────────────────────────────────────────────────────────────────

# Pareto plane per noise (6 panels)
fig, axes = plt.subplots(2, 3, figsize=(22, 12), sharey=True)
legend_handles = {}
for ax, noise in zip(axes.flat, NOISES):
    sub = master[master["noise_pm"] == noise].dropna(subset=["wall_per_conv", "topo_pct"]).copy()
    for (family, config), grp in sub.groupby(["family", "config"], sort=False):
        for _, r in grp.iterrows():
            sz = max(150, 15 * float(r["conv_pct"]))
            h = ax.scatter(r["wall_per_conv"], r["topo_pct"], s=sz,
                           marker=CONFIG_MARKER[config],
                           color=per_config_color(config, family),
                           edgecolor="black", linewidth=0.7, alpha=0.85)
            if config not in legend_handles:
                legend_handles[config] = h
    ax.set_xscale("log")
    ax.set_xlabel("Wall-time per converged TS (s, log)", fontsize=14)
    if ax in axes[:, 0]:
        ax.set_ylabel("IRC TOPO-intended %", fontsize=14)
    ax.tick_params(axis='both', labelsize=12)
    ax.set_title(f"{noise} pm noise", fontsize=15)
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.3, which="both")
legend_proxies, legend_labels = [], []
for cfg in legend_handles:
    family = master[master["config"] == cfg]["family"].iloc[0]
    legend_proxies.append(Line2D([0], [0], marker=CONFIG_MARKER[cfg],
                                  color="w", markerfacecolor=per_config_color(cfg, family),
                                  markeredgecolor="black", markeredgewidth=0.7,
                                  markersize=12, linestyle=""))
    legend_labels.append(cfg)
fig.legend(legend_proxies, legend_labels, loc="lower center", ncol=3, fontsize=13,
           bbox_to_anchor=(0.5, -0.04), frameon=False,
           handletextpad=0.8, columnspacing=1.6)
fig.suptitle("Pareto plane per noise — IRC TOPO % vs wall/conv\n"
             "Bubble size $\\propto$ TS conv %.  Upper-left = great; lower-right = bad.",
             y=1.01, fontsize=17)
fig.tight_layout(rect=[0, 0.03, 1, 0.97])
fig.savefig(f"{OUT}/fig_pareto_per_noise.pdf", bbox_inches="tight")
fig.savefig(f"{OUT}/fig_pareto_per_noise.png", bbox_inches="tight", dpi=140)
plt.close(fig)
print("Wrote fig_pareto_per_noise")


# Lollipop ranking — split low/high noise
cmap = plt.cm.RdYlGn
def render_lollipop_set(noise_set, save_name, title_suffix):
    fig, axes = plt.subplots(len(noise_set), 1, figsize=(18, 14))
    if len(noise_set) == 1:
        axes = [axes]
    for ax, noise in zip(axes, noise_set):
        sub = master[master["noise_pm"] == noise].dropna(subset=["wall_per_conv"]).sort_values("wall_per_conv").reset_index(drop=True)
        for i, r in sub.iterrows():
            topo = r["topo_pct"] if not np.isnan(r["topo_pct"]) else None
            color = cmap(min(max((topo or 0) / 100, 0), 1)) if topo is not None else "lightgray"
            ax.hlines(y=i, xmin=0, xmax=r["wall_per_conv"], color=color, lw=8, alpha=0.55)
            ax.scatter(r["wall_per_conv"], i, s=500, color=color,
                       edgecolor="black", linewidth=1.2, zorder=5)
            topo_str = f"{topo:.0f}%" if topo is not None else "n/a"
            anno = f"  TOPO {topo_str} / raw {r['conv_pct']:.0f}% / {r['wall_per_conv']:.1f}s"
            ax.text(r["wall_per_conv"], i, anno, va="center", ha="left", fontsize=11)
        ax.set_yticks(range(len(sub)))
        ax.set_yticklabels([SHORT.get(c, c) for c in sub["config"]], fontsize=13)
        ax.invert_yaxis()
        ax.set_xscale("log")
        xmin = sub["wall_per_conv"].min()
        xmax = sub["wall_per_conv"].max()
        ax.set_xlim(xmin / 2.5, xmax * 50)
        ax.tick_params(axis='x', labelsize=12)
        ax.set_xlabel("Wall-time per converged TS (s, log)" if noise == noise_set[-1] else "", fontsize=14)
        ax.set_title(f"{noise} pm noise (top row = fastest)", fontsize=16, loc="left", pad=10)
        ax.grid(alpha=0.3, which="both", axis="x")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=100))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=list(axes), fraction=0.015, pad=0.02, orientation="vertical")
    cbar.set_label("IRC TOPO % (gray = not measured)", fontsize=13)
    fig.suptitle(f"Method rankings by wall-time per converged TS — {title_suffix}\nhead color = IRC TOPO",
                 y=0.995, fontsize=17)
    fig.tight_layout(rect=[0, 0, 0.94, 0.97])
    fig.savefig(f"{OUT}/{save_name}.pdf", bbox_inches="tight")
    fig.savefig(f"{OUT}/{save_name}.png", bbox_inches="tight", dpi=140)
    plt.close(fig)

render_lollipop_set([10, 30, 50],  "fig_ranking_lollipop_low",  "low noise (10/30/50 pm)")
render_lollipop_set([100, 150, 200], "fig_ranking_lollipop_high", "high noise (100/150/200 pm)")
print("Wrote fig_ranking_lollipop_low / _high")


# TOPO recovery bar chart
recovery = master[["family", "config", "noise_pm", "recovery_pp"]].dropna().copy()
reps = {
    "plain GAD": "GAD dt=0.005",
    "Sella":     "Sella cartesian Eckart untuned Hess.Freq.=1",
    "hybrid":    "Hybrid damped Eckart eig tr=0.05",
}
rec_plot = recovery[recovery["config"].isin(reps.values())].copy()

fig, ax = plt.subplots(figsize=(15, 6.2))
families = list(reps)
w = 0.27
for i, fam in enumerate(families):
    sub = rec_plot[rec_plot["family"] == fam].sort_values("noise_pm")
    xs = np.array([NOISES.index(n) for n in sub["noise_pm"]]) + (i - 1) * w
    ys = sub["recovery_pp"].values
    ax.bar(xs, ys, width=w, label=reps[fam],
           color=FAMILY_CMAP[fam], edgecolor="black", linewidth=0.5)
    for x_, y_ in zip(xs, ys):
        ax.text(x_, y_ + (0.4 if y_ > 0 else -0.4), f"{y_:+.1f}",
                ha="center", va="bottom" if y_ > 0 else "top", fontsize=11)
ax.axhline(0, color="black", lw=0.8)
ax.set_xticks(range(6)); ax.set_xticklabels([f"{n} pm" for n in NOISES], fontsize=14)
ax.set_ylabel("IRC TOPO $-$ TS conv (percentage points)", fontsize=14)
ax.set_title("Who gains from IRC chemistry validation?\n(Positive = IRC saves trajectories; negative = IRC catches wrong-saddle 'wins')",
             fontsize=15)
ax.grid(alpha=0.3, axis="y")
ax.legend(loc="upper left", fontsize=13, framealpha=0.95)
ax.tick_params(axis='y', labelsize=12)
ax.set_ylim(-6, 12)
fig.tight_layout()
fig.savefig(f"{OUT}/fig_topo_recovery.pdf", bbox_inches="tight")
fig.savefig(f"{OUT}/fig_topo_recovery.png", bbox_inches="tight", dpi=140)
plt.close(fig)
print("Wrote fig_topo_recovery")


# ────────────────────────────────────────────────────────────────────────
# RMSD-to-TS: read the existing CSV summary
# ────────────────────────────────────────────────────────────────────────
rmsd_csv = f"{CSV}/rmsd_distributions_2026_05_11.csv"
if os.path.exists(rmsd_csv):
    rmsd_dist = pd.read_csv(rmsd_csv)
    nlist = sorted(rmsd_dist["noise_pm"].unique())
    fig, axes = plt.subplots(1, len(nlist), figsize=(6.5 * len(nlist), 5.5), sharey=True)
    if len(nlist) == 1:
        axes = [axes]
    cfg_colors = {
        "GAD dt=0.005":                                  FAMILY_CMAP["plain GAD"],
        "Sella cartesian Eckart untuned Hess.Freq.=1":   FAMILY_CMAP["Sella"],
        "Hybrid damped Eckart eig tr=0.05":              FAMILY_CMAP["hybrid"],
    }
    for ax, n in zip(axes, nlist):
        sub = rmsd_dist[(rmsd_dist["noise_pm"] == n) & (rmsd_dist["converged"])]
        for config, grp in sub.groupby("config"):
            if config not in cfg_colors:
                continue
            vals = grp.sort_values("rmsd")["rmsd"].values
            if len(vals) < 2:
                continue
            cdf = np.arange(1, len(vals)+1) / len(vals)
            ax.plot(vals, cdf, lw=3, color=cfg_colors[config], label=config)
        ax.axvline(0.3, color="gray", lw=1.5, linestyle="--", alpha=0.7, label="0.3 Å (RMSD-intended cutoff)")
        ax.set_xlabel("RMSD vs labelled T1x TS (Å)", fontsize=14)
        ax.set_ylabel("Cumulative fraction" if ax is axes[0] else "", fontsize=14)
        ax.set_title(f"{n} pm noise (converged only)", fontsize=15)
        ax.set_xscale("log")
        ax.tick_params(axis='both', labelsize=12)
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=11, loc="lower right")
    fig.suptitle("RMSD of converged TS vs labelled T1x reference — starting from NOISED TRUE TS",
                 fontsize=16, y=1.01)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_rmsd_to_ts.pdf", bbox_inches="tight")
    fig.savefig(f"{OUT}/fig_rmsd_to_ts.png", bbox_inches="tight", dpi=140)
    plt.close(fig)
    print("Wrote fig_rmsd_to_ts")


# ────────────────────────────────────────────────────────────────────────
# d=1 vs d=3 dual-panel
# ────────────────────────────────────────────────────────────────────────
d1 = master[master["config"] == "Sella cartesian Eckart untuned Hess.Freq.=1"].copy()
d3 = master[master["config"] == "Sella cartesian Eckart untuned Hess.Freq.=3"].copy()

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
ax = axes[0]
ax.plot(d1["noise_pm"], d1["conv_pct"], marker="o", lw=3, ms=10, label="d=1 (project canonical)", color=FAMILY_CMAP["Sella"])
ax.plot(d3["noise_pm"], d3["conv_pct"], marker="X", lw=3, ms=10, label="d=3 (Sella library default)", color=FAMILY_CMAP["Sella"] + (0.6,))
ax.set_xlabel("TS noise (pm)", fontsize=14)
ax.set_ylabel(r"TS conv % (Im. Freq. and $F_\mathrm{max}<0.01$)", fontsize=14)
ax.set_ylim(0, 100); ax.grid(alpha=0.3)
ax.set_title("TS conv: d=3 wins +3–4 pp at low/mid noise", fontsize=14)
ax.legend(fontsize=12); ax.tick_params(axis='both', labelsize=12)

ax = axes[1]
ax.plot(d1["noise_pm"], d1["topo_pct"], marker="o", lw=3, ms=10, label="d=1 (project canonical)", color=FAMILY_CMAP["Sella"])
ax.plot(d3["noise_pm"], d3["topo_pct"], marker="X", lw=3, ms=10, label="d=3 (Sella library default)", color=FAMILY_CMAP["Sella"] + (0.6,))
ax.set_xlabel("TS noise (pm)", fontsize=14)
ax.set_ylabel("IRC TOPO-intended %", fontsize=14)
ax.set_ylim(0, 100); ax.grid(alpha=0.3)
ax.set_title("IRC TOPO: d=1 wins at every noise level", fontsize=14)
ax.legend(fontsize=12); ax.tick_params(axis='both', labelsize=12)

fig.suptitle("Sella cartesian Eckart untuned: Hess.Freq.=1 (project) vs Hess.Freq.=3 (library default)",
             fontsize=16, y=1.02)
fig.tight_layout()
fig.savefig(f"{OUT}/fig_d1_vs_d3.pdf", bbox_inches="tight")
fig.savefig(f"{OUT}/fig_d1_vs_d3.png", bbox_inches="tight", dpi=140)
plt.close(fig)
print("Wrote fig_d1_vs_d3")


# ────────────────────────────────────────────────────────────────────────
# F-longbudget: R4 result — 10k steps @ 50pm, conv vs fmax-tightness
# Sources: (1) integrate script's longbudget_2026_05_16.csv (log-parsed live),
# falling back to (2) summary parquets when SLURM jobs complete.
# ────────────────────────────────────────────────────────────────────────
lb_csv = f"{CSV}/longbudget_2026_05_16.csv"
lb_rows = []
if os.path.exists(lb_csv):
    raw = pd.read_csv(lb_csv)
    fam_map = {"GAD dt=0.005 ×10k": "plain GAD", "Sella libdef ×10k": "Sella", "Hybrid damped ×10k": "hybrid"}
    for _, row in raw.iterrows():
        fam = fam_map.get(row["config"], "?")
        for thr, key in [(0.05,"fmax_005"),(0.023,"fmax_023"),(0.01,"fmax_010"),(0.005,"fmax_005t"),(0.001,"fmax_001")]:
            lb_rows.append({"label": f"{row['config']} (n={row['n']})", "family": fam,
                            "threshold": thr, "conv_pct": row[key]})

if lb_rows:
    lb = pd.DataFrame(lb_rows)
    fig, ax = plt.subplots(figsize=(13, 6.5))
    fam_colors = {"plain GAD": FAMILY_CMAP["plain GAD"], "Sella": FAMILY_CMAP["Sella"], "hybrid": FAMILY_CMAP["hybrid"]}
    for (label, fam), grp in lb.groupby(["label", "family"]):
        grp = grp.sort_values("threshold", ascending=False)
        ax.plot(grp["threshold"], grp["conv_pct"], marker="o", lw=3, ms=11,
                color=fam_colors[fam], label=label)
    ax.set_xscale("log"); ax.invert_xaxis()
    ax.set_xlabel(r"$F_\mathrm{max}$ threshold (eV/Å, log; tighter $\to$ right)", fontsize=14)
    ax.set_ylabel("TS conv % (Im. Freq. and threshold)", fontsize=14)
    ax.set_title("Long-budget probe @ 50 pm noise: does 10000 steps reach $F_\\mathrm{max}<0.001$?\n"
                 "(starting from noised true TS)", fontsize=15)
    ax.set_ylim(-2, 100); ax.grid(alpha=0.3, which="both")
    ax.tick_params(axis='both', labelsize=12)
    ax.legend(fontsize=13, loc="lower left")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_longbudget.pdf", bbox_inches="tight")
    fig.savefig(f"{OUT}/fig_longbudget.png", bbox_inches="tight", dpi=140)
    plt.close(fig)
    print("Wrote fig_longbudget")

print("\nAll figures done.")
