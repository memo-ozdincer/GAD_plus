#!/usr/bin/env python
"""Pull data and build all figures for the 2026-05-11 PDF.

Outputs:
  figures_2026_05_11/fig_main_4axis.pdf       — headline 4-panel, multi-variant
  figures_2026_05_11/fig_pareto_per_noise.pdf — wall/conv vs IRC TOPO scatter, per noise
  figures_2026_05_11/fig_ranking_lollipop.pdf — wall/conv lollipop ranking, per noise
  figures_2026_05_11/fig_rmsd_to_ts.pdf       — RMSD-to-known-TS CDFs
  figures_2026_05_11/fig_topo_recovery.pdf    — IRC TOPO recovery bar chart
  figures_2026_05_11/master_table.csv         — unified data
"""
from __future__ import annotations

import os
import sys

import duckdb
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPTS = "/lustre06/project/6033559/memoozd/GAD_plus/scripts"
sys.path.insert(0, SCRIPTS)
from plotting_style import apply_plot_style, palette, palette_map  # noqa: E402

apply_plot_style()

ROOT = "/lustre06/project/6033559/memoozd/GAD_plus"
RUNS = "/lustre07/scratch/memoozd/gadplus/runs"
OUT  = f"{ROOT}/figures_2026_05_11"
CSV  = f"{ROOT}/analysis_2026_04_29"
os.makedirs(OUT, exist_ok=True)


# ── Data assembly ───────────────────────────────────────────────────────

def grab_summary(glob_path, label_family, label_config):
    """Pull raw conv + wall + med_steps per noise from a hybrid-style summary glob."""
    df = duckdb.execute(f"""
        WITH src AS (
            SELECT *, CAST(regexp_extract(filename, '_(\\d+)pm', 1) AS INTEGER) AS np
            FROM read_parquet('{glob_path}', filename=true)
        )
        SELECT np AS noise_pm, COUNT(*) AS n,
               SUM(CASE WHEN converged THEN 1 ELSE 0 END) AS nc,
               PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY total_steps)
                   FILTER (WHERE converged) AS med_step,
               SUM(wall_time_s) AS sw
        FROM src GROUP BY np ORDER BY np
    """).df()
    df["conv_pct"]      = 100 * df["nc"] / df["n"]
    df["wall_per_conv"] = df["sw"] / df["nc"].replace(0, np.nan)
    df["family"]        = label_family
    df["config"]        = label_config
    return df[["family", "config", "noise_pm", "conv_pct", "med_step", "wall_per_conv"]]


def grab_irc(glob_path, label_family, label_config):
    """Pull IRC TOPO + RMSD-intended per noise."""
    df = duckdb.execute(f"""
        WITH src AS (
            SELECT *, CAST(regexp_extract(filename, '_(\\d+)pm', 1) AS INTEGER) AS np
            FROM read_parquet('{glob_path}', filename=true)
        )
        SELECT np AS noise_pm, COUNT(*) AS n,
               SUM(CASE WHEN topology_intended THEN 1 ELSE 0 END)*100.0/COUNT(*) AS topo_pct,
               SUM(CASE WHEN intended THEN 1 ELSE 0 END)*100.0/COUNT(*) AS rmsd_pct
        FROM src GROUP BY np ORDER BY np
    """).df()
    df["family"] = label_family
    df["config"] = label_config
    return df[["family", "config", "noise_pm", "topo_pct", "rmsd_pct"]]


def sella_from_csv(method_label, config_label):
    """Pull Sella raw conv + IRC from the canonical test_summary_full.csv + test_irc/*"""
    sdf = pd.read_csv(f"{CSV}/test_summary_full.csv")
    sdf["conv"] = sdf["is_saddle"] & sdf["fmax_loose"]
    sub = sdf[sdf["method"] == method_label].copy()
    raw = sub.groupby("noise_pm").agg(
        n=("sample_id", "count"),
        nc=("conv", "sum"),
        sw=("wall_time_s", "sum"),
    ).reset_index()
    med = sub[sub["conv"]].groupby("noise_pm")["total_steps"].median().rename("med_step").reset_index()
    raw = raw.merge(med, on="noise_pm", how="left")
    raw["conv_pct"]      = 100 * raw["nc"] / raw["n"]
    raw["wall_per_conv"] = raw["sw"] / raw["nc"].replace(0, np.nan)
    raw["family"]        = "Sella"
    raw["config"]        = config_label
    return raw[["family", "config", "noise_pm", "conv_pct", "med_step", "wall_per_conv"]]


# ────────────────────────────────────────────────────────────────────────
# 1) Plain GAD: dt=0.003, 0.005, 0.007 — full noise sweep
# ────────────────────────────────────────────────────────────────────────
gad_raw, gad_irc = [], []
for dt_tag, label in [("dt003", "GAD dt=0.003"), ("dt005", "GAD dt=0.005"), ("dt007", "GAD dt=0.007")]:
    gad_raw.append(grab_summary(
        f"{RUNS}/test_dtgrid/gad_{dt_tag}_fmax/summary_*.parquet",
        "plain GAD", label))
    gad_irc.append(grab_irc(
        f"{RUNS}/test_irc/gad_{dt_tag}_fmax/irc_validation_*.parquet",
        "plain GAD", label))


# ────────────────────────────────────────────────────────────────────────
# 2) Sella variants — canonical libdef/default/internal from test_summary_full + IRC dirs
# ────────────────────────────────────────────────────────────────────────
sella_raw = [
    sella_from_csv("Sella libdef",   "Sella cartesian Eckart untuned Hess.Freq.=1"),
    sella_from_csv("Sella default",  "Sella cartesian tuned Hess.Freq.=1"),
    sella_from_csv("Sella internal", "Sella internal tuned Hess.Freq.=1"),
]
sella_irc = [
    grab_irc(f"{RUNS}/test_irc/sella_carteck_libdef/irc_validation_*.parquet",
             "Sella", "Sella cartesian Eckart untuned Hess.Freq.=1"),
    grab_irc(f"{RUNS}/test_irc/sella_carteck_default/irc_validation_*.parquet",
             "Sella", "Sella cartesian tuned Hess.Freq.=1"),
    grab_irc(f"{RUNS}/test_irc/sella_internal_default/irc_validation_*.parquet",
             "Sella", "Sella internal tuned Hess.Freq.=1"),
]

# Sella libdef with d=3 (Hessian every 3 steps) — raw conv only, no IRC
sella_d3 = duckdb.execute(f"""
    WITH src AS (
        SELECT *, CAST(regexp_extract(filename, '_(\\d+)pm', 1) AS INTEGER) AS np
        FROM read_parquet('{RUNS}/test_hessfreq/sella_carteck_libdef_d3/summary_*.parquet', filename=true)
    )
    SELECT np AS noise_pm, COUNT(*) AS n,
           SUM(CASE WHEN converged THEN 1 ELSE 0 END) AS nc,
           PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY total_steps)
               FILTER (WHERE converged) AS med_step,
           SUM(wall_time_s) AS sw
    FROM src GROUP BY np ORDER BY np
""").df()
sella_d3["conv_pct"]      = 100 * sella_d3["nc"] / sella_d3["n"]
sella_d3["wall_per_conv"] = sella_d3["sw"] / sella_d3["nc"].replace(0, np.nan)
sella_d3["family"]        = "Sella"
sella_d3["config"]        = "Sella cartesian Eckart untuned Hess.Freq.=3"
sella_raw.append(sella_d3[["family", "config", "noise_pm", "conv_pct", "med_step", "wall_per_conv"]])


# ────────────────────────────────────────────────────────────────────────
# 3) Hybrid: damped + undamped Eckart eig-switch tr=0.05 — full sweep
# ────────────────────────────────────────────────────────────────────────
hyb_raw, hyb_irc = [], []

# Damped — all 6 noises live in runs/hybrid_for_irc/...
hyb_raw.append(grab_summary(
    f"{RUNS}/hybrid_for_irc/hybrid_damped_eckart_swtrue_dt5e-3_tr0.05_*pm/summary_*.parquet",
    "hybrid", "Hybrid damped Eckart eig tr=0.05"))
hyb_irc.append(grab_irc(
    f"{RUNS}/irc_hybrid/hybrid_damped_eckart_swtrue_dt5e-3_tr0.05_*pm/irc_validation_*.parquet",
    "hybrid", "Hybrid damped Eckart eig tr=0.05"))

# Undamped — stitch deeper (10, 100pm) + extension (30, 50, 150, 200pm)
deep_und = grab_summary(
    f"{RUNS}/hybrid_deeper/hybrid_eckart_swtrue_dt5e-3_tr0.05_sf1e-2_*pm/summary_*.parquet",
    "hybrid", "Hybrid undamped Eckart eig tr=0.05")
ext_und  = grab_summary(
    f"{RUNS}/hybrid_extension/hybrid_eckart_swtrue_dt5e-3_tr0.05_sf1e-2_*pm/summary_*.parquet",
    "hybrid", "Hybrid undamped Eckart eig tr=0.05")
hyb_raw.append(pd.concat([deep_und, ext_und]).sort_values("noise_pm").reset_index(drop=True))
deep_und_irc = grab_irc(
    f"{RUNS}/irc_hybrid_deeper/hybrid_eckart_swtrue_dt5e-3_tr0.05_sf1e-2_*pm/irc_validation_*.parquet",
    "hybrid", "Hybrid undamped Eckart eig tr=0.05")
ext_und_irc  = grab_irc(
    f"{RUNS}/irc_hybrid_extension/hybrid_eckart_swtrue_dt5e-3_tr0.05_sf1e-2_*pm/irc_validation_*.parquet",
    "hybrid", "Hybrid undamped Eckart eig tr=0.05")
hyb_irc.append(pd.concat([deep_und_irc, ext_und_irc]).sort_values("noise_pm").reset_index(drop=True))


# Combine all
raw_all = pd.concat(gad_raw + sella_raw + hyb_raw, ignore_index=True)
irc_all = pd.concat(gad_irc + sella_irc + hyb_irc, ignore_index=True)
master  = raw_all.merge(irc_all, on=["family", "config", "noise_pm"], how="left")

# Add chemistry-recovery column
master["recovery_pp"] = master["topo_pct"] - master["conv_pct"]

# Save the master table
master = master.sort_values(["family", "config", "noise_pm"]).reset_index(drop=True)
master.to_csv(f"{CSV}/master_2026_05_11.csv", index=False)
print(f"Wrote master table: {len(master)} rows")
print(master.round(2).to_string(index=False))


# ────────────────────────────────────────────────────────────────────────
# Visual style
# ────────────────────────────────────────────────────────────────────────
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


def per_config_color(config, family):
    """Slightly different shades of the family color per config."""
    base = FAMILY_CMAP[family]
    # Use seaborn's lighter/darker shades via mixing with white
    siblings = sorted({c for c, f in master[["config", "family"]].itertuples(index=False) if f == family})
    idx = siblings.index(config)
    # Mix alpha for variation
    alphas = [1.0, 0.75, 0.55, 0.4]
    return base + tuple([alphas[idx % len(alphas)]])


# ────────────────────────────────────────────────────────────────────────
# 4-panel figure: TS conv / IRC TOPO / med steps / wall vs noise
# Larger fonts so figure is readable at PDF screen size on a laptop.
# ────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(24, 6), sharex=True)
panels = [
    ("conv_pct",      r"TS conv % (Im. Freq. and $F_\mathrm{max}<0.01$)", False),
    ("topo_pct",      "IRC TOPO-intended %",                              False),
    ("med_step",      "Median converged-step count",                       True),
    ("wall_per_conv", "Wall-time per converged TS (s)",                    True),
]
for ax, (col, ylab, logy) in zip(axes, panels):
    for (family, config), grp in master.groupby(["family", "config"], sort=False):
        grp = grp.sort_values("noise_pm")
        if col == "topo_pct" and grp[col].isna().all():
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
    if logy: ax.set_yscale("log")
    else: ax.set_ylim(0, 100)
    ax.grid(alpha=0.3)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=14,
           bbox_to_anchor=(0.5, -0.20), frameon=False)
fig.suptitle("Best-of-family comparison across TS noise ($n=287$ T1x test split)",
             y=1.03, fontsize=18)
fig.tight_layout()
fig.savefig(f"{OUT}/fig_main_4axis.pdf", bbox_inches="tight")
fig.savefig(f"{OUT}/fig_main_4axis.png", bbox_inches="tight", dpi=140)
print("Wrote fig_main_4axis")


# ────────────────────────────────────────────────────────────────────────
# Pareto scatter: wall/conv vs IRC TOPO per noise (6 panels)
# - No per-point text annotations (collide); use a shared legend instead.
# - Methods get short codes; family color encodes group; marker = config.
# ────────────────────────────────────────────────────────────────────────
SHORT = {
    "GAD dt=0.003": "G003", "GAD dt=0.005": "G005", "GAD dt=0.007": "G007",
    "Sella cartesian Eckart untuned Hess.Freq.=1":  "S-cart+eck-utuned-d1",
    "Sella cartesian tuned Hess.Freq.=1":           "S-cart-tuned-d1",
    "Sella internal tuned Hess.Freq.=1":            "S-int-tuned-d1",
    "Sella cartesian Eckart untuned Hess.Freq.=3":  "S-cart+eck-utuned-d3",
    "Hybrid damped Eckart eig tr=0.05":             "H-damp-Eck-tr0.05",
    "Hybrid undamped Eckart eig tr=0.05":           "H-undamp-Eck-tr0.05",
}

fig, axes = plt.subplots(2, 3, figsize=(22, 12), sharey=True)
noises = [10, 30, 50, 100, 150, 200]
legend_handles = {}
for ax, noise in zip(axes.flat, noises):
    sub = master[master["noise_pm"] == noise].copy()
    sub = sub.dropna(subset=["wall_per_conv", "topo_pct"])
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
# Build custom legend handles with fixed small marker size, so the legend
# doesn't inherit the bubble-size scaling from the scatter.
from matplotlib.lines import Line2D
legend_proxies = []
legend_labels = []
for cfg, h in legend_handles.items():
    family = master[master["config"] == cfg]["family"].iloc[0]
    legend_proxies.append(Line2D([0], [0], marker=CONFIG_MARKER[cfg],
                                  color="w", markerfacecolor=per_config_color(cfg, family),
                                  markeredgecolor="black", markeredgewidth=0.7,
                                  markersize=12, label=cfg, linestyle=""))
    legend_labels.append(cfg)
fig.legend(legend_proxies, legend_labels,
           loc="lower center", ncol=3, fontsize=13,
           bbox_to_anchor=(0.5, -0.04), frameon=False,
           handletextpad=0.8, columnspacing=1.6)
fig.suptitle("Pareto plane per noise: IRC TOPO % vs wall/conv  (upper-left = great; lower-right = bad)\n"
             "Bubble size $\\propto$ TS conv %",
             y=1.01, fontsize=17)
fig.tight_layout(rect=[0, 0.03, 1, 0.97])
fig.savefig(f"{OUT}/fig_pareto_per_noise.pdf", bbox_inches="tight")
fig.savefig(f"{OUT}/fig_pareto_per_noise.png", bbox_inches="tight", dpi=140)
plt.close(fig)
print("Wrote fig_pareto_per_noise")


# ────────────────────────────────────────────────────────────────────────
# Lollipop ranking per noise: 6 panels stacked vertically — much more width
# - One TALL figure: 6 rows × 1 col, each row is wide and tall enough for labels
# - Annotations to the right with consistent x-offset (no log-multiplier)
# ────────────────────────────────────────────────────────────────────────
# Split into 2 figures (3 panels each) so each fits on one PDF page.
cmap = plt.cm.RdYlGn

def render_lollipop_set(noise_set, save_name, title_suffix):
    fig, axes = plt.subplots(len(noise_set), 1, figsize=(18, 14))
    if len(noise_set) == 1:
        axes = [axes]
    for ax, noise in zip(axes, noise_set):
        sub = master[master["noise_pm"] == noise].copy()
        sub = sub.dropna(subset=["wall_per_conv"]).sort_values("wall_per_conv").reset_index(drop=True)
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
    fig.suptitle(f"Method rankings by wall-time per converged TS — {title_suffix}\n"
                 "head color = IRC TOPO",
                 y=0.995, fontsize=17)
    fig.tight_layout(rect=[0, 0, 0.94, 0.97])
    fig.savefig(f"{OUT}/{save_name}.pdf", bbox_inches="tight")
    fig.savefig(f"{OUT}/{save_name}.png", bbox_inches="tight", dpi=140)
    plt.close(fig)

render_lollipop_set([10, 30, 50],  "fig_ranking_lollipop_low",  "low noise (10/30/50 pm)")
render_lollipop_set([100, 150, 200], "fig_ranking_lollipop_high", "high noise (100/150/200 pm)")
print("Wrote fig_ranking_lollipop_low / _high")


# ────────────────────────────────────────────────────────────────────────
# RMSD-to-known-TS distributions — SKIPPED (slow on login; rmsd_to_known_ts_compare.csv covers median/p95)
# ────────────────────────────────────────────────────────────────────────
print("Skipping per-sample RMSD distribution figure (use existing rmsd_to_known_ts_compare.csv summary stats)")


# ────────────────────────────────────────────────────────────────────────
# TOPO-recovery bar chart: IRC TOPO − Raw conv, per (family, noise)
# ────────────────────────────────────────────────────────────────────────
recovery = master[["family", "config", "noise_pm", "recovery_pp"]].copy()
recovery = recovery.dropna()
# Pick one representative config per family for the recovery chart
reps = {
    "plain GAD": "GAD dt=0.005",
    "Sella":     "Sella cartesian Eckart untuned Hess.Freq.=1",
    "hybrid":    "Hybrid damped Eckart eig tr=0.05",
}
rec_plot = recovery[recovery["config"].isin(reps.values())].copy()

fig, ax = plt.subplots(figsize=(15, 6.2))
families = list(reps)
families_x = np.arange(6)
w = 0.27
for i, fam in enumerate(families):
    sub = rec_plot[rec_plot["family"] == fam].sort_values("noise_pm")
    xs = np.array([noises.index(n) for n in sub["noise_pm"]]) + (i - 1) * w
    ys = sub["recovery_pp"].values
    ax.bar(xs, ys, width=w, label=reps[fam],
           color=FAMILY_CMAP[fam], edgecolor="black", linewidth=0.5)
    for x_, y_ in zip(xs, ys):
        ax.text(x_, y_ + (0.4 if y_ > 0 else -0.4), f"{y_:+.1f}",
                ha="center", va="bottom" if y_ > 0 else "top", fontsize=11)
ax.axhline(0, color="black", lw=0.8)
ax.set_xticks(range(6)); ax.set_xticklabels([f"{n} pm" for n in noises], fontsize=14)
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

print("\nAll figures done")
