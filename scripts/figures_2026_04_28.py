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

from plotting_style import apply_plot_style, palette_color

apply_plot_style()

OUT = Path("/lustre06/project/6033559/memoozd/GAD_plus/figures")
OUT.mkdir(exist_ok=True)
NOISES = [10, 30, 50, 100, 150, 200]

# Consistent 3-color palette
C_GAD       = palette_color(0)
C_SELLA_CE  = palette_color(3)
C_SELLA_INT = palette_color(4)

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
    """Grouped + stacked bar chart: 3 methods per noise, each bar partitioned
    into intended / half-intended / unintended (proportional, sums to 100%)."""
    C_INT, C_HALF, C_UN = palette_color(2), palette_color(8), palette_color(3)

    method_data = {}
    for k, (_, _, irc_dir, color, marker, label) in METHODS_3M.items():
        irc = load_irc(irc_dir)
        ints, halfs, uns = [], [], []
        for n in NOISES:
            if irc is None:
                ints.append(np.nan); halfs.append(np.nan); uns.append(np.nan); continue
            g = irc[irc["noise_pm"] == n]
            if not len(g):
                ints.append(np.nan); halfs.append(np.nan); uns.append(np.nan); continue
            tot = len(g)
            ni = int(g["topology_intended"].sum())
            nh = int((g["topology_half_intended"] & ~g["topology_intended"]).sum())
            nu = tot - ni - nh
            ints.append(100*ni/tot); halfs.append(100*nh/tot); uns.append(100*nu/tot)
        method_data[k] = (ints, halfs, uns, color, label)

    fig, ax = plt.subplots(figsize=(11, 5.4))
    n_methods = len(METHODS_3M)
    width = 0.27
    x = np.arange(len(NOISES))
    offsets = np.linspace(-(n_methods-1)/2, (n_methods-1)/2, n_methods) * width
    for i, (k, (ints, halfs, uns, color, label)) in enumerate(method_data.items()):
        xpos = x + offsets[i]
        ax.bar(xpos, ints, width, color=C_INT, edgecolor=color, linewidth=1.4,
               label="intended" if i == 0 else None)
        ax.bar(xpos, halfs, width, bottom=ints, color=C_HALF, edgecolor=color,
               linewidth=1.4, label="half-intended" if i == 0 else None)
        ax.bar(xpos, uns, width, bottom=np.array(ints)+np.array(halfs),
               color=C_UN, edgecolor=color, linewidth=1.4,
               label="unintended" if i == 0 else None)
        # Method label above each bar
        for xi, ni in zip(xpos, ints):
            if not np.isnan(ni):
                ax.text(xi, ni/2, f"{ni:.0f}", ha="center", va="center",
                        fontsize=8, fontweight="bold", color="white")
        # Add a thin colored stripe above each group for method identity
        for xi in xpos:
            ax.plot([xi - width/2, xi + width/2], [101.5, 101.5],
                    color=color, linewidth=2.4, solid_capstyle="butt")
    ax.set_xlabel("TS noise (pm)", fontsize=11)
    ax.set_ylabel("IRC TOPO outcome (%)", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in NOISES])
    ax.set_ylim(0, 108)
    ax.grid(axis="y", alpha=0.3)
    # Two legends: outcome class (color), method (edge color)
    method_handles = [plt.Rectangle((0,0),1,1, facecolor="white",
                                    edgecolor=v[3], linewidth=1.8, label=v[4])
                      for v in method_data.values()]
    leg1 = ax.legend(loc="upper right", fontsize=9, framealpha=0.95, title="outcome")
    ax.add_artist(leg1)
    ax.legend(handles=method_handles, loc="upper left", fontsize=9,
              framealpha=0.95, title="method (edge)")
    save(fig, "fig_irc_intended_grouped_3m")


def fig_gad_stepsize_vs_step():
    """GAD median per-step displacement vs step, faceted by noise.

    Reads disp_from_last from the gad_eckart_fmax/ trajectories (canonical
    fmax-criterion GAD pool, Round 6). 6 panels, one per noise level.
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
