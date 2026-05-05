#!/usr/bin/env python
"""Comprehensive analysis for IRC_COMPREHENSIVE_2026-04-20.

Covers 5 TS-finding methods x 6 noise levels, plus available IRC data.
Writes all numbers to /tmp/analysis_2026_04_20.txt and generates figures.
"""
from __future__ import annotations

import os
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plotting_style import apply_plot_style, palette_color

apply_plot_style()

NOISES = [10, 30, 50, 100, 150, 200]
OUT = Path("/lustre06/project/6033559/memoozd/GAD_plus/figures")
OUT.mkdir(exist_ok=True)

# Method -> list of (parquet_path, convergence_column, label_short)
METHODS = {
    "GAD Eckart":       ([f"/lustre07/scratch/memoozd/gadplus/runs/round2/summary_gad_dt003_{n}pm.parquet" for n in [10,30,50]]
                       + [f"/lustre07/scratch/memoozd/gadplus/runs/round3/summary_gad_dt003_{n}pm.parquet" for n in [100,150,200]],
                         "converged", palette_color(0), "o"),
    "GAD no-Eckart":    ([f"/lustre07/scratch/memoozd/gadplus/runs/gad_no_eckart/summary_gad_dt003_no_eckart_{n}pm.parquet" for n in NOISES],
                         "converged", palette_color(9), "D"),
    "Sella cart+Eckart":([f"/lustre07/scratch/memoozd/gadplus/runs/sella_2000/summary_sella_cartesian_eckart_fmax0p01_{n}pm.parquet" for n in NOISES],
                         "conv_nneg1_fmax001", palette_color(3), "s"),
    "Sella cart no-Eckart":([f"/lustre07/scratch/memoozd/gadplus/runs/sella_2000/summary_sella_cartesian_fmax0p01_{n}pm.parquet" for n in NOISES],
                         "conv_nneg1_fmax001", palette_color(1), "^"),
    "Sella internal":   ([f"/lustre07/scratch/memoozd/gadplus/runs/sella_2000/summary_sella_internal_fmax0p01_{n}pm.parquet" for n in NOISES],
                         "conv_nneg1_fmax001", palette_color(4), "v"),
}

LINES = []
def log(s=""):
    print(s); LINES.append(s)


def load_method(method):
    paths, conv_col, _, _ = METHODS[method]
    rows = []
    for i, p in enumerate(paths):
        noise = NOISES[i]
        if not os.path.exists(p):
            rows.append({"noise": noise, "tot": 0, "conv": 0, "wall_mean": np.nan, "wall_med": np.nan, "steps_mean": np.nan, "steps_med": np.nan, "available": False})
            continue
        r = duckdb.execute(f"""
            SELECT COUNT(*),
                   SUM(CASE WHEN {conv_col} THEN 1 ELSE 0 END),
                   AVG(wall_time_s), MEDIAN(wall_time_s),
                   AVG(total_steps), MEDIAN(total_steps)
            FROM '{p}'
        """).fetchone()
        rows.append({"noise": noise, "tot": r[0], "conv": r[1], "wall_mean": r[2], "wall_med": r[3], "steps_mean": r[4], "steps_med": r[5], "available": True})
    return pd.DataFrame(rows)


# ======================================================================
log("="*80)
log("PER-METHOD CONVERGENCE RATES (n_neg==1 + native force criterion)")
log("="*80)
log()
all_tables = {}
for m in METHODS:
    df = load_method(m)
    all_tables[m] = df
    conv_col = METHODS[m][1]
    log(f"[{m}]  (uses {conv_col})")
    for _, row in df.iterrows():
        if not row["available"]:
            log(f"  {row['noise']:>3d}pm:  {'(pending)':>40s}")
            continue
        rate = 100 * row["conv"] / 300
        log(f"  {row['noise']:>3d}pm:  conv={row['conv']:>3d}/300 ({rate:5.1f}%)   "
            f"wall: mean={row['wall_mean']:5.1f}s med={row['wall_med']:5.1f}s   "
            f"steps: mean={row['steps_mean']:6.0f} med={row['steps_med']:6.0f}")
    log()

# Convergence matrix
log("CONVERGENCE-RATE MATRIX (% of 300)")
log("="*80)
mat = pd.DataFrame({m: [100*all_tables[m].iloc[i]["conv"]/300 if all_tables[m].iloc[i]["available"] else np.nan for i in range(6)] for m in METHODS}, index=NOISES)
log(mat.round(1).to_string())
log()

# Wall-time matrix
log("MEDIAN WALL TIME (s) PER SAMPLE")
log("="*80)
walltable = pd.DataFrame({m: [all_tables[m].iloc[i]["wall_med"] if all_tables[m].iloc[i]["available"] else np.nan for i in range(6)] for m in METHODS}, index=NOISES)
log(walltable.round(1).to_string())
log()

# Steps matrix
log("MEDIAN TOTAL_STEPS PER SAMPLE")
log("="*80)
steptable = pd.DataFrame({m: [all_tables[m].iloc[i]["steps_med"] if all_tables[m].iloc[i]["available"] else np.nan for i in range(6)] for m in METHODS}, index=NOISES)
log(steptable.round(0).to_string())
log()

# TSs per hour
log("TSs DELIVERED PER GPU-HOUR (conv / (N_tot * wall_mean_sec / 3600))")
log("="*80)
tsphr = pd.DataFrame({}, index=NOISES)
for m in METHODS:
    vals = []
    for i in range(6):
        r = all_tables[m].iloc[i]
        if not r["available"]:
            vals.append(np.nan); continue
        total_h = (r["tot"] * r["wall_mean"]) / 3600.0
        vals.append(r["conv"] / total_h if total_h > 0 else np.nan)
    tsphr[m] = vals
log(tsphr.round(1).to_string())
log()

# ======================================================================
# IRC data where available
log("="*80)
log("IRC DATA (sella_hip on existing TS sets — old 1000-step Sella for Sella runs)")
log("="*80)
IRC_SETS = {
    "GAD Eckart (all endpoints)":   "/lustre07/scratch/memoozd/gadplus/runs/irc_sellahip_allendpoints/*.parquet",
    "GAD Eckart (converged only)":  "/lustre07/scratch/memoozd/gadplus/runs/irc_sellahip_full/*.parquet",
    "Sella cart+Eckart 1000 (all)": "/lustre07/scratch/memoozd/gadplus/runs/irc_sellahip_on_sella_allep/*.parquet",
    "Sella cart+Eckart 1000 (conv)":"/lustre07/scratch/memoozd/gadplus/runs/irc_sellahip_on_sella/*.parquet",
    "Rigorous on GAD (conv)":       "/lustre07/scratch/memoozd/gadplus/runs/irc_rigorous_full/*.parquet",
}
for label, glob in IRC_SETS.items():
    try:
        r = duckdb.execute(f"""
            SELECT noise_pm, COUNT(*), SUM(CASE WHEN topology_intended THEN 1 ELSE 0 END), SUM(CASE WHEN intended THEN 1 ELSE 0 END)
            FROM '{glob}' GROUP BY noise_pm ORDER BY noise_pm
        """).df()
        log(f"\n[{label}]")
        for _, row in r.iterrows():
            topo = 100 * row[2] / row[1]
            rmsd = 100 * row[3] / row[1]
            log(f"  {int(row['noise_pm']):>3d}pm:  n={int(row[1]):>4d}  TOPO={topo:5.1f}%  RMSD={rmsd:5.1f}%")
    except Exception as e:
        log(f"  [{label}] ERROR: {e}")

# ======================================================================
# FIGURES
# ======================================================================
log("\n\nGENERATING FIGURES...")

plt.rcParams.update({"font.size": 10, "text.usetex": False})
apply_plot_style()


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight", dpi=150)
    plt.close(fig)
    log(f"  wrote {name}.(pdf|png)")


# Fig 1: Convergence rate vs noise, all 5 methods
fig, ax = plt.subplots(figsize=(8.5, 5.5))
for m in METHODS:
    _, _, color, marker = METHODS[m]
    df = all_tables[m]
    xs, ys = [], []
    for i, noise in enumerate(NOISES):
        r = df.iloc[i]
        if not r["available"]: continue
        xs.append(noise); ys.append(100 * r["conv"] / 300)
    ax.plot(xs, ys, color=color, marker=marker, linewidth=2, markersize=8,
            markerfacecolor="white", markeredgewidth=2, label=m)
ax.set_xlabel("TS noise (pm)", fontsize=11)
ax.set_ylabel("convergence rate (% of 300)", fontsize=11)
ax.set_xticks(NOISES)
ax.set_ylim(0, 105)
ax.grid(alpha=0.3)
ax.legend(loc="lower left", fontsize=9, framealpha=0.95)
save(fig, "fig_all_methods_conv")

# Fig 2: Wall time (median) vs noise
fig, ax = plt.subplots(figsize=(8.5, 5))
for m in METHODS:
    _, _, color, marker = METHODS[m]
    df = all_tables[m]
    xs, ys = [], []
    for i, noise in enumerate(NOISES):
        r = df.iloc[i]
        if not r["available"]: continue
        xs.append(noise); ys.append(r["wall_med"])
    ax.plot(xs, ys, color=color, marker=marker, linewidth=2, markersize=8,
            markerfacecolor="white", markeredgewidth=2, label=m)
ax.set_xlabel("TS noise (pm)", fontsize=11)
ax.set_ylabel("median wall time / sample (s)", fontsize=11)
ax.set_xticks(NOISES)
ax.set_yscale("log")
ax.grid(alpha=0.3, which="both")
ax.legend(loc="upper left", fontsize=9, framealpha=0.95)
save(fig, "fig_all_methods_wall")

# Fig 3: TSs per GPU-hour
fig, ax = plt.subplots(figsize=(8.5, 5))
for m in METHODS:
    _, _, color, marker = METHODS[m]
    df = all_tables[m]
    xs, ys = [], []
    for i, noise in enumerate(NOISES):
        r = df.iloc[i]
        if not r["available"]: continue
        total_h = (r["tot"] * r["wall_mean"]) / 3600.0
        xs.append(noise); ys.append(r["conv"] / total_h if total_h > 0 else np.nan)
    ax.plot(xs, ys, color=color, marker=marker, linewidth=2, markersize=8,
            markerfacecolor="white", markeredgewidth=2, label=m)
ax.set_xlabel("TS noise (pm)", fontsize=11)
ax.set_ylabel("valid TSs delivered per GPU-hour", fontsize=11)
ax.set_xticks(NOISES)
ax.set_yscale("log")
ax.grid(alpha=0.3, which="both")
ax.legend(loc="upper right", fontsize=9, framealpha=0.95)
save(fig, "fig_all_methods_tsph")

# Fig 4: Eckart delta (helps GAD, neutral for Sella cartesian)
fig, ax = plt.subplots(figsize=(8, 5))
gad_delta = []
sella_delta = []
for i, noise in enumerate(NOISES):
    g_e = all_tables["GAD Eckart"].iloc[i]
    g_n = all_tables["GAD no-Eckart"].iloc[i]
    s_e = all_tables["Sella cart+Eckart"].iloc[i]
    s_n = all_tables["Sella cart no-Eckart"].iloc[i]
    gad_delta.append(100 * (g_e["conv"] - g_n["conv"]) / 300 if g_e["available"] and g_n["available"] else np.nan)
    sella_delta.append(100 * (s_e["conv"] - s_n["conv"]) / 300 if s_e["available"] and s_n["available"] else np.nan)

x = np.arange(len(NOISES))
w = 0.35
ax.bar(x - w/2, gad_delta, w, color=palette_color(0), label="GAD (Eckart − no Eckart)")
ax.bar(x + w/2, sella_delta, w, color=palette_color(3), label="Sella cart (Eckart − no Eckart)")
ax.axhline(0, color=palette_color(7), lw=0.8)
for i, (g, s) in enumerate(zip(gad_delta, sella_delta)):
    if not np.isnan(g):
        ax.text(i - w/2, g + (0.2 if g >= 0 else -0.6), f"{g:+.1f}", ha="center", fontsize=9, color=palette_color(0))
    if not np.isnan(s):
        ax.text(i + w/2, s + (0.2 if s >= 0 else -0.6), f"{s:+.1f}", ha="center", fontsize=9, color=palette_color(3))
ax.set_xticks(x); ax.set_xticklabels([f"{n} pm" for n in NOISES])
ax.set_xlabel("TS noise (pm)", fontsize=11)
ax.set_ylabel("Δ conv. rate from Eckart projection (pp)", fontsize=11)
ax.grid(alpha=0.3, axis="y")
ax.legend(loc="upper left", fontsize=10)
save(fig, "fig_eckart_delta")

# Fig 5: Sella coord system (cart vs int at equal treatment — both no-Eckart)
fig, ax = plt.subplots(figsize=(8, 5))
for m, color, marker in [("Sella cart no-Eckart", palette_color(1), "^"),
                          ("Sella internal", palette_color(4), "v")]:
    df = all_tables[m]
    xs, ys = [], []
    for i, noise in enumerate(NOISES):
        r = df.iloc[i]
        if not r["available"]: continue
        xs.append(noise); ys.append(100 * r["conv"] / 300)
    ax.plot(xs, ys, color=color, marker=marker, linewidth=2, markersize=9,
            markerfacecolor="white", markeredgewidth=2, label=m)
ax.set_xlabel("TS noise (pm)", fontsize=11)
ax.set_ylabel("convergence rate (%)", fontsize=11)
ax.set_xticks(NOISES)
ax.set_ylim(0, 105)
ax.grid(alpha=0.3)
ax.legend(loc="lower left", fontsize=10)
save(fig, "fig_sella_coord_comparison")

# Save text report
with open("/tmp/analysis_2026_04_20.txt", "w") as f:
    f.write("\n".join(LINES))
print("Report: /tmp/analysis_2026_04_20.txt")
