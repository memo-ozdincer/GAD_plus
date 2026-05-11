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
    sella_from_csv("Sella libdef",   "Sella libdef (cart+Eckart)"),
    sella_from_csv("Sella default",  "Sella default (cart no-Eckart)"),
    sella_from_csv("Sella internal", "Sella internal (lib default)"),
]
sella_irc = [
    grab_irc(f"{RUNS}/test_irc/sella_carteck_libdef/irc_validation_*.parquet",
             "Sella", "Sella libdef (cart+Eckart)"),
    grab_irc(f"{RUNS}/test_irc/sella_carteck_default/irc_validation_*.parquet",
             "Sella", "Sella default (cart no-Eckart)"),
    grab_irc(f"{RUNS}/test_irc/sella_internal_default/irc_validation_*.parquet",
             "Sella", "Sella internal (lib default)"),
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
sella_d3["config"]        = "Sella libdef Hess-freq d=3"
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
    "GAD dt=0.003":                       "o",
    "GAD dt=0.005":                       "s",
    "GAD dt=0.007":                       "D",
    "Sella libdef (cart+Eckart)":         "o",
    "Sella default (cart no-Eckart)":     "s",
    "Sella internal (lib default)":       "v",
    "Sella libdef Hess-freq d=3":         "X",
    "Hybrid damped Eckart eig tr=0.05":   "^",
    "Hybrid undamped Eckart eig tr=0.05": "<",
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
# 4-panel figure: raw conv / IRC TOPO / med steps / wall vs noise
# ────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(22, 5), sharex=True)
panels = [
    ("conv_pct",      r"Raw conv % ($n_{neg}=1$ and $f_{max}<0.01$)", False),
    ("topo_pct",      "IRC TOPO-intended %",                                                 False),
    ("med_step",      "Median converged-step count",                                          True),
    ("wall_per_conv", "Wall-time per converged TS (s)",                                       True),
]
for ax, (col, ylab, logy) in zip(axes, panels):
    for (family, config), grp in master.groupby(["family", "config"], sort=False):
        grp = grp.sort_values("noise_pm")
        if col == "topo_pct" and grp[col].isna().all():
            continue
        # plain line for full sweep, dashed for partial
        n_pts = grp[col].notna().sum()
        ls = "-" if n_pts >= 6 else "--"
        ax.plot(grp.loc[grp[col].notna(), "noise_pm"],
                grp.loc[grp[col].notna(), col],
                marker=CONFIG_MARKER[config], linestyle=ls, lw=1.8, ms=8,
                color=per_config_color(config, family), label=config)
    ax.set_xlabel("TS noise (pm)"); ax.set_ylabel(ylab)
    if logy: ax.set_yscale("log")
    else: ax.set_ylim(0, 100)
    ax.grid(alpha=0.3)
# Legend below
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=9,
           bbox_to_anchor=(0.5, -0.18), frameon=False)
fig.suptitle("Best-of-family comparison across TS noise (n=287 T1x test split)", y=1.02, fontsize=14)
fig.tight_layout()
fig.savefig(f"{OUT}/fig_main_4axis.pdf", bbox_inches="tight")
fig.savefig(f"{OUT}/fig_main_4axis.png", bbox_inches="tight", dpi=140)
print("Wrote fig_main_4axis")


# ────────────────────────────────────────────────────────────────────────
# Pareto scatter: wall/conv vs IRC TOPO per noise (6 panels)
# ────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharey=True)
noises = [10, 30, 50, 100, 150, 200]
for ax, noise in zip(axes.flat, noises):
    sub = master[master["noise_pm"] == noise].copy()
    sub = sub.dropna(subset=["wall_per_conv", "topo_pct"])
    for (family, config), grp in sub.groupby(["family", "config"], sort=False):
        for _, r in grp.iterrows():
            sz = max(80, 8 * float(r["conv_pct"]))   # raw conv → bubble size
            ax.scatter(r["wall_per_conv"], r["topo_pct"], s=sz,
                       marker=CONFIG_MARKER[config],
                       color=per_config_color(config, family),
                       edgecolor="black", linewidth=0.6, alpha=0.85,
                       label=config if noise == 10 else None)
            ax.annotate(config.replace("Hybrid ", "H ")
                        .replace("Sella ", "S ").replace("GAD ", "G "),
                        xy=(r["wall_per_conv"], r["topo_pct"]),
                        xytext=(5, 5), textcoords="offset points",
                        fontsize=7, alpha=0.75)
    ax.set_xscale("log")
    ax.set_xlabel("Wall-time per converged TS (s, log)")
    if ax in axes[:, 0]:
        ax.set_ylabel("IRC TOPO-intended %")
    ax.set_title(f"{noise} pm noise")
    ax.set_ylim(0, 100); ax.grid(alpha=0.3, which="both")
fig.suptitle("Pareto plane per noise — IRC TOPO % vs wall/conv (lower-right = bad; upper-left = great)\n"
             "Bubble size $\\propto$ raw conv %", y=1.02, fontsize=14)
fig.tight_layout()
fig.savefig(f"{OUT}/fig_pareto_per_noise.pdf", bbox_inches="tight")
fig.savefig(f"{OUT}/fig_pareto_per_noise.png", bbox_inches="tight", dpi=140)
print("Wrote fig_pareto_per_noise")


# ────────────────────────────────────────────────────────────────────────
# Lollipop ranking per noise (6 panels): wall/conv ascending, head color = IRC TOPO
# ────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
for ax, noise in zip(axes.flat, noises):
    sub = master[master["noise_pm"] == noise].copy()
    sub = sub.dropna(subset=["wall_per_conv"]).sort_values("wall_per_conv")
    sub = sub.reset_index(drop=True)
    cmap = plt.cm.RdYlGn  # red=bad TOPO, green=good TOPO
    for i, r in sub.iterrows():
        topo = r["topo_pct"] if not np.isnan(r["topo_pct"]) else None
        color = cmap(min(max((topo or 0) / 100, 0), 1)) if topo is not None else "lightgray"
        ax.hlines(y=i, xmin=0, xmax=r["wall_per_conv"], color=color, lw=4, alpha=0.6)
        ax.scatter(r["wall_per_conv"], i, s=300, color=color,
                   edgecolor="black", linewidth=1, zorder=5)
        # Annotate with TOPO/raw conv
        anno = f"TOPO {topo:.0f}%" if topo is not None else "TOPO --"
        anno += f"  raw {r['conv_pct']:.0f}%"
        ax.text(r["wall_per_conv"] * 1.08, i, anno, va="center", fontsize=8)
    ax.set_yticks(range(len(sub)))
    ax.set_yticklabels([c.replace("Hybrid ", "H ").replace("Sella ", "S ").replace("GAD ", "G ")
                         for c in sub["config"]], fontsize=8)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlabel("Wall-time per converged TS (s, log)")
    ax.set_title(f"{noise} pm noise — methods ranked by wall (top = fastest)")
    ax.grid(alpha=0.3, which="both", axis="x")
# Add a colorbar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=100))
sm.set_array([])
cbar = fig.colorbar(sm, ax=list(axes.flat), fraction=0.02, pad=0.04, orientation="vertical")
cbar.set_label("IRC TOPO %", fontsize=9)
fig.suptitle("Ranking by wall-time per converged TS  •  head color = IRC TOPO\n"
             "Fast + green = best; slow + red = worst", y=1.00, fontsize=14)
fig.savefig(f"{OUT}/fig_ranking_lollipop.pdf", bbox_inches="tight")
fig.savefig(f"{OUT}/fig_ranking_lollipop.png", bbox_inches="tight", dpi=140)
print("Wrote fig_ranking_lollipop")


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
    "Sella":     "Sella libdef (cart+Eckart)",
    "hybrid":    "Hybrid damped Eckart eig tr=0.05",
}
rec_plot = recovery[recovery["config"].isin(reps.values())].copy()

fig, ax = plt.subplots(figsize=(11, 4.6))
families = list(reps)
families_x = np.arange(6)   # 6 noise levels
w = 0.27
for i, fam in enumerate(families):
    sub = rec_plot[rec_plot["family"] == fam].sort_values("noise_pm")
    xs = np.array([noises.index(n) for n in sub["noise_pm"]]) + (i - 1) * w
    ys = sub["recovery_pp"].values
    colors = [palette()[1] if v > 0 else palette()[3] for v in ys]  # green if positive, red if negative
    ax.bar(xs, ys, width=w, label=reps[fam],
           color=FAMILY_CMAP[fam], edgecolor="black", linewidth=0.5)
    for x_, y_ in zip(xs, ys):
        ax.text(x_, y_ + (1.5 if y_ > 0 else -1.5), f"{y_:+.1f}",
                ha="center", va="bottom" if y_ > 0 else "top", fontsize=8)
ax.axhline(0, color="black", lw=0.8)
ax.set_xticks(range(6)); ax.set_xticklabels([f"{n} pm" for n in noises])
ax.set_ylabel("IRC TOPO − Raw conv (percentage points)")
ax.set_title("Who gains from IRC chemistry validation?  Positive = IRC saves trajectories; negative = IRC catches wrong-saddle 'wins'")
ax.grid(alpha=0.3, axis="y")
ax.legend(loc="upper right", fontsize=9)
fig.tight_layout()
fig.savefig(f"{OUT}/fig_topo_recovery.pdf", bbox_inches="tight")
fig.savefig(f"{OUT}/fig_topo_recovery.png", bbox_inches="tight", dpi=140)
print("Wrote fig_topo_recovery")

print("\nAll figures done")
