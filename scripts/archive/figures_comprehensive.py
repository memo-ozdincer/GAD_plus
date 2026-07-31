#!/usr/bin/env python
"""Figures for IRC_COMPREHENSIVE_2026-04-17.tex.

Main-narrative figures (4 panels):
  fig_main_gad_conv      — GAD convergence rate vs noise, line chart
  fig_main_gad_irc       — GAD IRC TOPO bar chart (all endpoints)  [existing-ish, we regenerate for consistency]
  fig_main_sella_conv    — Sella convergence rate vs noise, line chart
  fig_main_sella_irc     — Sella IRC TOPO bar chart (all endpoints) [existing-ish]

Supplement figures:
  fig_gad_sella_conv_overlay       — both conv rates on one chart
  fig_gad_sella_irc_overlay        — both IRC TOPO rates on one chart
  fig_gate_vs_irc_overlap_gad      — per-sample 4-quadrant stacked chart
  fig_gate_vs_irc_overlap_sella    — same for Sella
  fig_walltime_by_method           — TS-finding wall time comparison
  fig_steps_by_method              — average steps per sample
  fig_gad_all_steps_distribution   — distribution of total_steps for GAD
  fig_sella_all_steps_distribution — distribution of total_steps for Sella
"""
from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plotting_style import apply_plot_style, palette_color

apply_plot_style()

OUT = Path("/lustre06/project/6033559/memoozd/GAD_plus/figures")
OUT.mkdir(exist_ok=True)
NOISES = [10, 30, 50, 100, 150, 200]

# Palette
C_INTENDED = palette_color(2)
C_HALF     = palette_color(8)
C_UNINT    = palette_color(3)
C_GAD      = palette_color(0)
C_SELLA    = palette_color(3)


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"wrote {OUT}/{name}.(pdf|png)")


# ------------------- Data loaders -------------------

def load_gad_summary():
    rows = []
    for noise, sub in [(10,"round2"),(30,"round2"),(50,"round2"),(100,"round3"),(150,"round3"),(200,"round3")]:
        df = duckdb.execute(f"""
            SELECT *, {noise} AS noise_pm_fixed FROM '/lustre07/scratch/memoozd/gadplus/runs/{sub}/summary_gad_dt003_{noise}pm.parquet'
        """).df()
        df["noise_pm"] = noise
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


def load_sella_summary():
    rows = []
    for noise in NOISES:
        df = duckdb.execute(f"""
            SELECT * FROM '/lustre07/scratch/memoozd/gadplus/runs/sella_1000_coords/summary_sella_cartesian_eckart_fmax0p01_{noise}pm.parquet'
        """).df()
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


def load_irc(dir_path):
    return duckdb.execute(f"SELECT * FROM '{dir_path}/*.parquet'").df()


# ------------------- Figure builders -------------------

def fig_conv_line(summary, conv_col, title_short, color, name):
    """Line chart: convergence rate vs noise. summary has noise_pm + conv_col (bool)."""
    rates = []
    ns = []
    for n in NOISES:
        g = summary[summary["noise_pm"] == n]
        total = 300  # universe is always 300
        conv = int(g[conv_col].fillna(False).astype(bool).sum())
        rates.append(100 * conv / total)
        ns.append(conv)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(NOISES, rates, color=color, marker="o", linewidth=2.2, markersize=8,
            markerfacecolor="white", markeredgewidth=2)
    for x, y, n in zip(NOISES, rates, ns):
        ax.annotate(f"{n}/300\n{y:.1f}%", (x, y), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=8.5)
    ax.set_xlabel("TS noise (pm)", fontsize=11)
    ax.set_ylabel("convergence rate (%)", fontsize=11)
    ax.set_ylim(0, 108)
    ax.set_xticks(NOISES)
    ax.grid(alpha=0.3)
    save(fig, name)


def fig_irc_bars(irc, int_col, half_col, color_int, color_half, color_un, name):
    ns, pi, ph, pu = [], [], [], []
    for n_ in NOISES:
        g = irc[irc["noise_pm"] == n_]
        total = len(g)
        if total == 0:
            continue
        ni = int(g[int_col].sum())
        nh = int((g[half_col] & ~g[int_col]).sum())
        nu = total - ni - nh
        ns.append(total)
        pi.append(100*ni/total); ph.append(100*nh/total); pu.append(100*nu/total)

    x = np.arange(len(ns))
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.bar(x, pi, 0.65, label="intended", color=color_int, edgecolor="white", linewidth=0.6)
    ax.bar(x, ph, 0.65, bottom=pi, label="half-intended", color=color_half, edgecolor="white", linewidth=0.6)
    ax.bar(x, pu, 0.65, bottom=np.array(pi)+np.array(ph), label="unintended", color=color_un, edgecolor="white", linewidth=0.6)

    for i, (a, b, c) in enumerate(zip(pi, ph, pu)):
        if a > 6: ax.text(x[i], a/2, f"{a:.1f}%", ha="center", va="center", fontsize=9, color="white", fontweight="bold")
        if b > 6: ax.text(x[i], a + b/2, f"{b:.1f}%", ha="center", va="center", fontsize=9, color=palette_color(7))
        if c > 6: ax.text(x[i], a + b + c/2, f"{c:.1f}%", ha="center", va="center", fontsize=9, color="white", fontweight="bold")
    for i, n in enumerate(ns):
        ax.text(x[i], 101.5, f"n={n}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{n_} pm" for n_ in NOISES[:len(ns)]])
    ax.set_xlabel("TS noise (pm)", fontsize=11)
    ax.set_ylabel("fraction of IRC runs (%)", fontsize=11)
    ax.set_ylim(0, 108)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)
    save(fig, name)


def fig_conv_overlay(gad_summary, sella_summary):
    """GAD vs Sella convergence rate on same chart."""
    gad_rates = []
    sella_rates = []
    for n in NOISES:
        g_g = gad_summary[gad_summary["noise_pm"] == n]
        g_s = sella_summary[sella_summary["noise_pm"] == n]
        gad_rates.append(100 * g_g["converged"].fillna(False).astype(bool).sum() / 300)
        sella_rates.append(100 * g_s["conv_nneg1_fmax001"].fillna(False).astype(bool).sum() / 300)

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(NOISES, gad_rates, "o-", color=C_GAD, linewidth=2.2, markersize=9,
            label=r"GAD (gad_dt003): $n_{neg}=1 \wedge \|\mathbf{F}\|/N < 0.01$",
            markerfacecolor="white", markeredgewidth=2)
    ax.plot(NOISES, sella_rates, "s-", color=C_SELLA, linewidth=2.2, markersize=9,
            label=r"Sella (cart+eckart): $n_{neg}=1 \wedge \max_i |f_i| < 0.01$",
            markerfacecolor="white", markeredgewidth=2)
    for x, y1, y2 in zip(NOISES, gad_rates, sella_rates):
        ax.annotate(f"{y1:.1f}", (x, y1), xytext=(8, -2), textcoords="offset points", fontsize=9, color=C_GAD)
        ax.annotate(f"{y2:.1f}", (x, y2), xytext=(8, -12), textcoords="offset points", fontsize=9, color=C_SELLA)
    ax.set_xlabel("TS noise (pm)", fontsize=11)
    ax.set_ylabel("convergence rate (% of 300)", fontsize=11)
    ax.set_ylim(0, 108)
    ax.set_xticks(NOISES)
    ax.legend(loc="lower left", fontsize=10, framealpha=0.95)
    ax.grid(alpha=0.3)
    save(fig, "fig_conv_overlay")


def fig_irc_overlay(gad_irc, sella_irc):
    """GAD vs Sella IRC TOPO-int% on same chart (all endpoints)."""
    gad_t, sella_t = [], []
    for n in NOISES:
        g_g = gad_irc[gad_irc["noise_pm"] == n]
        g_s = sella_irc[sella_irc["noise_pm"] == n]
        gad_t.append(100 * g_g["topology_intended"].mean() if len(g_g) else np.nan)
        sella_t.append(100 * g_s["topology_intended"].mean() if len(g_s) else np.nan)

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(NOISES, gad_t, "o-", color=C_GAD, linewidth=2.2, markersize=9,
            label="GAD endpoints → sella_hip IRC (TOPO-int)",
            markerfacecolor="white", markeredgewidth=2)
    ax.plot(NOISES, sella_t, "s-", color=C_SELLA, linewidth=2.2, markersize=9,
            label="Sella endpoints → sella_hip IRC (TOPO-int)",
            markerfacecolor="white", markeredgewidth=2)
    for x, y1, y2 in zip(NOISES, gad_t, sella_t):
        if not np.isnan(y1):
            ax.annotate(f"{y1:.1f}", (x, y1), xytext=(8, 4), textcoords="offset points", fontsize=9, color=C_GAD)
        if not np.isnan(y2):
            ax.annotate(f"{y2:.1f}", (x, y2), xytext=(8, -14), textcoords="offset points", fontsize=9, color=C_SELLA)
    ax.set_xlabel("TS noise (pm)", fontsize=11)
    ax.set_ylabel("IRC TOPO-intended rate (%)", fontsize=11)
    ax.set_ylim(0, 108)
    ax.set_xticks(NOISES)
    ax.legend(loc="lower left", fontsize=10, framealpha=0.95)
    ax.grid(alpha=0.3)
    save(fig, "fig_irc_overlay")


def fig_criterion_overlap(gad_summary, gad_irc, sella_summary, sella_irc):
    """Per-sample 4-quadrant breakdown: converged vs IRC-TOPO-pass."""
    cats = ["both", "criterion only", "IRC only", "neither"]
    colors = [palette_color(2), palette_color(8), palette_color(4), palette_color(3)]

    def compute_pcts(summary, irc_df, conv_col):
        mat = np.zeros((4, len(NOISES)))
        ns = []
        for i, noise in enumerate(NOISES):
            s = summary[summary["noise_pm"] == noise][["sample_id", conv_col]].rename(columns={conv_col: "conv"})
            r = irc_df[irc_df["noise_pm"] == noise][["sample_id", "topology_intended"]]
            m = s.merge(r, on="sample_id", how="outer")
            conv = m["conv"].fillna(False).astype(bool)
            topo = m["topology_intended"].fillna(False).astype(bool)
            n = len(m)
            both = int((conv & topo).sum())
            g_only = int((conv & ~topo).sum())
            i_only = int((~conv & topo).sum())
            nei = int((~conv & ~topo).sum())
            ns.append(n)
            mat[:, i] = [100*both/n, 100*g_only/n, 100*i_only/n, 100*nei/n]
        return mat, ns

    for summary, irc_df, conv_col, label, name in [
        (gad_summary, gad_irc, "converged", "GAD (force_norm<0.01)", "fig_gate_vs_irc_gad"),
        (sella_summary, sella_irc, "conv_nneg1_fmax001", "Sella (fmax<0.01)", "fig_gate_vs_irc_sella"),
    ]:
        mat, ns = compute_pcts(summary, irc_df, conv_col)
        x = np.arange(len(NOISES))
        fig, ax = plt.subplots(figsize=(8, 4.8))
        bottom = np.zeros(len(NOISES))
        for i, (c, cat) in enumerate(zip(colors, cats)):
            ax.bar(x, mat[i], 0.65, bottom=bottom, label=cat, color=c, edgecolor="white", linewidth=0.6)
            for xi, v in enumerate(mat[i]):
                if v > 5:
                    ax.text(xi, bottom[xi] + v/2, f"{v:.0f}%", ha="center", va="center",
                            fontsize=8.5, color="white", fontweight="bold")
            bottom += mat[i]
        for xi, n in enumerate(ns):
            ax.text(x[xi], 101, f"n={n}", ha="center", va="bottom", fontsize=8.5)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{n_} pm" for n_ in NOISES])
        ax.set_xlabel("TS noise (pm)", fontsize=11)
        ax.set_ylabel("fraction of samples (%)", fontsize=11)
        ax.set_ylim(0, 108)
        ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=9)
        save(fig, name)


def fig_walltime_steps(gad_summary, sella_summary):
    """Wall time and steps comparison."""
    gad_w = [gad_summary[gad_summary["noise_pm"]==n]["wall_time_s"].median() for n in NOISES]
    sella_w = [sella_summary[sella_summary["noise_pm"]==n]["wall_time_s"].median() for n in NOISES]
    gad_s = [gad_summary[gad_summary["noise_pm"]==n]["total_steps"].median() for n in NOISES]
    sella_s = [sella_summary[sella_summary["noise_pm"]==n]["total_steps"].median() for n in NOISES]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5))
    a1.plot(NOISES, gad_w, "o-", color=C_GAD, linewidth=2, markersize=8, label="GAD", markerfacecolor="white", markeredgewidth=2)
    a1.plot(NOISES, sella_w, "s-", color=C_SELLA, linewidth=2, markersize=8, label="Sella", markerfacecolor="white", markeredgewidth=2)
    a1.set_yscale("log")
    a1.set_xlabel("TS noise (pm)"); a1.set_ylabel("median wall time / sample (s, log scale)")
    a1.set_xticks(NOISES); a1.grid(alpha=0.3, which="both"); a1.legend(fontsize=10)

    a2.plot(NOISES, gad_s, "o-", color=C_GAD, linewidth=2, markersize=8, label="GAD", markerfacecolor="white", markeredgewidth=2)
    a2.plot(NOISES, sella_s, "s-", color=C_SELLA, linewidth=2, markersize=8, label="Sella", markerfacecolor="white", markeredgewidth=2)
    a2.set_yscale("log")
    a2.set_xlabel("TS noise (pm)"); a2.set_ylabel("median total_steps / sample (log scale)")
    a2.set_xticks(NOISES); a2.grid(alpha=0.3, which="both"); a2.legend(fontsize=10)
    save(fig, "fig_walltime_steps")


def fig_step_distributions(gad_summary, sella_summary):
    """Step count distributions per noise (violin-ish via histogram grid)."""
    fig, axes = plt.subplots(2, 6, figsize=(16, 6), sharey=True)
    for j, noise in enumerate(NOISES):
        g_g = gad_summary[gad_summary["noise_pm"] == noise]["total_steps"]
        g_s = sella_summary[sella_summary["noise_pm"] == noise]["total_steps"]
        bins = np.logspace(0, np.log10(max(g_g.max(), g_s.max())+1), 40)
        axes[0][j].hist(g_g, bins=bins, color=C_GAD, alpha=0.8, edgecolor="white")
        axes[0][j].set_xscale("log"); axes[0][j].set_title(f"GAD {noise}pm", fontsize=10)
        axes[1][j].hist(g_s, bins=bins, color=C_SELLA, alpha=0.8, edgecolor="white")
        axes[1][j].set_xscale("log"); axes[1][j].set_title(f"Sella {noise}pm", fontsize=10)
        axes[1][j].set_xlabel("total_steps (log)", fontsize=9)
    axes[0][0].set_ylabel("# samples", fontsize=10)
    axes[1][0].set_ylabel("# samples", fontsize=10)
    fig.tight_layout()
    save(fig, "fig_step_distributions")


# ------------------- Main -------------------

def main():
    plt.rcParams.update({"font.size": 10, "text.usetex": False})
    apply_plot_style()

    print("Loading data...")
    gad_summary = load_gad_summary()
    sella_summary = load_sella_summary()
    gad_irc_allep = load_irc("/lustre07/scratch/memoozd/gadplus/runs/irc_sellahip_allendpoints")
    sella_irc_allep = load_irc("/lustre07/scratch/memoozd/gadplus/runs/irc_sellahip_on_sella_allep")
    gad_irc_conv = load_irc("/lustre07/scratch/memoozd/gadplus/runs/irc_sellahip_full")
    sella_irc_conv = load_irc("/lustre07/scratch/memoozd/gadplus/runs/irc_sellahip_on_sella")
    print(f"GAD summary: {len(gad_summary)}, Sella summary: {len(sella_summary)}")
    print(f"GAD IRC all: {len(gad_irc_allep)}, Sella IRC all: {len(sella_irc_allep)}")
    print(f"GAD IRC conv: {len(gad_irc_conv)}, Sella IRC conv: {len(sella_irc_conv)}")

    # Main panels (each separate)
    fig_conv_line(gad_summary, "converged", "GAD", C_GAD, "fig_main_gad_conv")
    fig_irc_bars(gad_irc_allep, "topology_intended", "topology_half_intended",
                 C_INTENDED, C_HALF, C_UNINT, "fig_main_gad_irc")
    fig_conv_line(sella_summary, "conv_nneg1_fmax001", "Sella", C_SELLA, "fig_main_sella_conv")
    fig_irc_bars(sella_irc_allep, "topology_intended", "topology_half_intended",
                 C_INTENDED, C_HALF, C_UNINT, "fig_main_sella_irc")

    # Overlays
    fig_conv_overlay(gad_summary, sella_summary)
    fig_irc_overlay(gad_irc_allep, sella_irc_allep)

    # Criterion overlap
    fig_criterion_overlap(gad_summary, gad_irc_allep, sella_summary, sella_irc_allep)

    # Wall time / steps
    fig_walltime_steps(gad_summary, sella_summary)
    fig_step_distributions(gad_summary, sella_summary)


if __name__ == "__main__":
    main()
