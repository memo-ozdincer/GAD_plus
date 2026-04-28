#!/usr/bin/env python
"""Figure set for IRC_COMPREHENSIVE_2026-04-28 (3-method focus + step-size).

Builds on figures_master_2026_04_20.py but only emits a focused 3-method set:
GAD Eckart, Sella cart+Eckart, Sella internal. Drops no-Eckart variants.

Outputs (all written to figures/, suffix _3m so they don't overwrite the
5-method originals):
  fig_cmp_conv_3m.{pdf,png}            - 3-method conv rate line ("TS converged" axis)
  fig_cmp_irc_topo_3m.{pdf,png}        - 3-method IRC TOPO line (Sella int 200pm filled in)
  fig_irc_intended_grouped_3m.{pdf,png} - grouped bar chart, IRC TOPO, 3 methods x 6 noise
  fig_gad_stepsize_vs_step.{pdf,png}   - GAD median step displacement vs step, faceted by noise
"""
from __future__ import annotations

import os
from pathlib import Path

import duckdb
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("/lustre06/project/6033559/memoozd/GAD_plus/figures")
OUT.mkdir(exist_ok=True)
NOISES = [10, 30, 50, 100, 150, 200]

# Consistent 3-color palette
C_GAD       = "#1f77b4"
C_SELLA_CE  = "#d62728"
C_SELLA_INT = "#9467bd"

METHODS_3M = {
    "gad_eckart": (
        [f"/lustre07/scratch/memoozd/gadplus/runs/round2/summary_gad_dt003_{n}pm.parquet" for n in [10,30,50]]
      + [f"/lustre07/scratch/memoozd/gadplus/runs/round3/summary_gad_dt003_{n}pm.parquet" for n in [100,150,200]],
        "converged",
        "/lustre07/scratch/memoozd/gadplus/runs/irc_sellahip_allendpoints",
        C_GAD, "o", "GAD Eckart",
    ),
    "sella_carte_2k": (
        [f"/lustre07/scratch/memoozd/gadplus/runs/sella_2000/summary_sella_cartesian_eckart_fmax0p01_{n}pm.parquet" for n in NOISES],
        "conv_nneg1_fmax001",
        "/lustre07/scratch/memoozd/gadplus/runs/irc_sella_carte_2000",
        C_SELLA_CE, "s", "Sella cart+Eckart",
    ),
    "sella_int_2k": (
        [f"/lustre07/scratch/memoozd/gadplus/runs/sella_2000/summary_sella_internal_fmax0p01_{n}pm.parquet" for n in NOISES],
        "conv_nneg1_fmax001",
        "/lustre07/scratch/memoozd/gadplus/runs/irc_sella_int_2000",
        C_SELLA_INT, "v", "Sella internal",
    ),
}


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"wrote {name}")


def load_summary(paths, conv_col):
    d = {}
    for p, noise in zip(paths, NOISES):
        if not os.path.exists(p):
            d[noise] = (0, 0); continue
        r = duckdb.execute(
            f"SELECT COUNT(*), SUM(CASE WHEN {conv_col} THEN 1 ELSE 0 END) FROM '{p}'"
        ).fetchone()
        d[noise] = (r[0] or 0, r[1] or 0)
    return d


def load_irc(irc_dir):
    if not os.path.exists(irc_dir):
        return None
    return duckdb.execute(f"SELECT * FROM '{irc_dir}/*.parquet'").df()


def fig_cmp_conv_3m():
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for k, (paths, conv_col, _, color, marker, label) in METHODS_3M.items():
        summary = load_summary(paths, conv_col)
        xs, rates = [], []
        for n in NOISES:
            tot, c = summary[n]
            if tot > 0:
                rates.append(100 * c / 300); xs.append(n)
        ax.plot(xs, rates, "-", color=color, marker=marker, linewidth=2.2, markersize=9,
                markerfacecolor="white", markeredgewidth=2, label=label)
    ax.set_xlabel("TS noise (pm)", fontsize=11)
    ax.set_ylabel("TS converged rate (%)", fontsize=11)
    ax.set_xticks(NOISES)
    ax.set_ylim(0, 105)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left", fontsize=10, framealpha=0.95)
    save(fig, "fig_cmp_conv_3m")


def fig_cmp_irc_topo_3m():
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for k, (_, _, irc_dir, color, marker, label) in METHODS_3M.items():
        irc = load_irc(irc_dir)
        if irc is None:
            continue
        xs, ys = [], []
        for n in NOISES:
            g = irc[irc["noise_pm"] == n]
            if len(g):
                xs.append(n); ys.append(100 * g["topology_intended"].mean())
        ax.plot(xs, ys, "-", color=color, marker=marker, linewidth=2.2, markersize=9,
                markerfacecolor="white", markeredgewidth=2, label=label)
    ax.set_xlabel("TS noise (pm)", fontsize=11)
    ax.set_ylabel("IRC TOPO-intended (%)", fontsize=11)
    ax.set_xticks(NOISES)
    ax.set_ylim(0, 105)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left", fontsize=10, framealpha=0.95)
    save(fig, "fig_cmp_irc_topo_3m")


def fig_irc_intended_grouped_3m():
    """Grouped bar chart: IRC TOPO-intended, 3 methods x 6 noise levels."""
    method_rates = {}
    for k, (_, _, irc_dir, color, marker, label) in METHODS_3M.items():
        irc = load_irc(irc_dir)
        rates = []
        for n in NOISES:
            if irc is None:
                rates.append(np.nan); continue
            g = irc[irc["noise_pm"] == n]
            rates.append(100 * g["topology_intended"].mean() if len(g) else np.nan)
        method_rates[k] = (rates, color, label)

    fig, ax = plt.subplots(figsize=(10, 5.0))
    n_methods = len(METHODS_3M)
    width = 0.27
    x = np.arange(len(NOISES))
    offsets = np.linspace(-(n_methods-1)/2, (n_methods-1)/2, n_methods) * width
    for i, (k, (rates, color, label)) in enumerate(method_rates.items()):
        bars = ax.bar(x + offsets[i], rates, width, color=color, edgecolor="white",
                      linewidth=0.8, label=label)
        for xi, r in zip(x + offsets[i], rates):
            if not np.isnan(r):
                ax.text(xi, r + 1.2, f"{r:.1f}", ha="center", va="bottom", fontsize=8)
    ax.set_xlabel("TS noise (pm)", fontsize=11)
    ax.set_ylabel("IRC TOPO-intended (%)", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in NOISES])
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper right", fontsize=10, framealpha=0.95)
    save(fig, "fig_irc_intended_grouped_3m")


def fig_gad_stepsize_vs_step():
    """GAD median per-step displacement vs step, faceted by noise.

    Reads disp_from_last from the gad_eckart_fmax/ trajectories (canonical
    fmax-gated GAD pool, Round 6). 6 panels, one per noise level.
    """
    base = "/lustre07/scratch/memoozd/gadplus/runs/gad_eckart_fmax"
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.8), sharey=True, sharex=False)
    axes = axes.flatten()
    for ax, noise in zip(axes, NOISES):
        glob = f"{base}/traj_gad_dt003_fmax_{noise}pm_*.parquet"
        try:
            df = duckdb.execute(f"""
                SELECT step,
                       quantile_cont(disp_from_last, 0.5) AS med,
                       quantile_cont(disp_from_last, 0.25) AS q25,
                       quantile_cont(disp_from_last, 0.75) AS q75,
                       COUNT(*) AS n
                FROM '{glob}'
                WHERE step > 0
                GROUP BY step
                ORDER BY step
            """).df()
        except Exception as e:
            ax.text(0.5, 0.5, f"no data\n{e}", ha="center", va="center",
                    transform=ax.transAxes, fontsize=8)
            ax.set_title(f"{noise} pm")
            continue
        ax.fill_between(df["step"], df["q25"], df["q75"], color=C_GAD, alpha=0.22,
                        label="IQR")
        ax.plot(df["step"], df["med"], color=C_GAD, linewidth=1.2, label="median")
        ax.set_title(f"{noise} pm  (n={df['n'].iloc[0] if len(df) else 0})", fontsize=10)
        ax.set_xlabel("step")
        ax.set_yscale("log")
        ax.grid(alpha=0.3, which="both")
    axes[0].set_ylabel("per-step displacement |Δx| (Å)")
    axes[3].set_ylabel("per-step displacement |Δx| (Å)")
    axes[0].legend(loc="upper right", fontsize=8)
    fig.suptitle("GAD Eckart (dt=0.003) per-step displacement, by noise",
                 fontsize=12, y=1.00)
    fig.tight_layout()
    save(fig, "fig_gad_stepsize_vs_step")


def main():
    fig_cmp_conv_3m()
    fig_cmp_irc_topo_3m()
    fig_irc_intended_grouped_3m()
    fig_gad_stepsize_vs_step()


if __name__ == "__main__":
    main()
