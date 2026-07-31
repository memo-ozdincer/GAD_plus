#!/usr/bin/env python
"""One stacked bar chart per criterion. Each chart: x=noise, stacks=[intended, half, unintended].

Denominator at each noise level = number of gad_dt003-converged TSs (i.e. number of IRC runs).
Bars sum to 100%.
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

IRC_DIR = "/lustre07/scratch/memoozd/gadplus/runs/irc_sellahip_full"
OUT = Path("/lustre06/project/6033559/memoozd/GAD_plus/figures")
OUT.mkdir(exist_ok=True)
NOISES = [10, 30, 50, 100, 150, 200]

# Consistent palette across all criteria charts
C_INTENDED = palette_color(2)   # green
C_HALF     = palette_color(8)   # amber
C_UNINT    = palette_color(3)   # red


def save(fig, name: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"wrote {OUT}/{name}.(pdf|png)")


def stacked_bars(
    irc: pd.DataFrame,
    int_col: str,
    half_col: str,
    title: str,
    name: str,
    subtitle: str = "",
) -> None:
    """Produce one chart: x=noise, y=%, stacked [intended, half, unintended]."""
    ns, p_int, p_half, p_un = [], [], [], []
    for noise in NOISES:
        g = irc[irc["noise_pm"] == noise]
        n = len(g)
        ni = int(g[int_col].sum())
        # Half excludes full intended (they're mutually exclusive by construction)
        nh = int((g[half_col] & ~g[int_col]).sum())
        nu = n - ni - nh
        ns.append(n)
        p_int.append(100 * ni / n)
        p_half.append(100 * nh / n)
        p_un.append(100 * nu / n)

    x = np.arange(len(NOISES))
    width = 0.65

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    b_int = ax.bar(x, p_int, width, label="intended",
                   color=C_INTENDED, edgecolor="white", linewidth=0.6)
    b_half = ax.bar(x, p_half, width, bottom=p_int, label="half-intended",
                    color=C_HALF, edgecolor="white", linewidth=0.6)
    b_un = ax.bar(x, p_un, width, bottom=np.array(p_int) + np.array(p_half),
                  label="unintended", color=C_UNINT, edgecolor="white", linewidth=0.6)

    # Annotate each segment with count if segment is big enough
    for noise_i, (ni, nh, nu) in enumerate(zip(p_int, p_half, p_un)):
        if ni > 6:
            ax.text(x[noise_i], ni / 2, f"{ni:.1f}%",
                    ha="center", va="center", fontsize=9, color="white", fontweight="bold")
        if nh > 6:
            ax.text(x[noise_i], ni + nh / 2, f"{nh:.1f}%",
                    ha="center", va="center", fontsize=9, color=palette_color(7))
        if nu > 6:
            ax.text(x[noise_i], ni + nh + nu / 2, f"{nu:.1f}%",
                    ha="center", va="center", fontsize=9, color="white", fontweight="bold")

    # n= above each bar
    for noise_i, n in enumerate(ns):
        ax.text(x[noise_i], 101.5, f"n={n}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{n} pm" for n in NOISES])
    ax.set_xlabel("starting TS noise")
    ax.set_ylabel("fraction of IRC runs (%)")
    ax.set_ylim(0, 108)
    ax.legend(loc="lower right", fontsize=10, framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)
    save(fig, name)


def endpoint_quality_bars(irc: pd.DataFrame) -> None:
    """Bar chart for endpoint vibrational quality: both / one / neither at minimum."""
    ns, p_both, p_one, p_neither = [], [], [], []
    for noise in NOISES:
        g = irc[irc["noise_pm"] == noise]
        n = len(g)
        fwd_min = (g["forward_n_neg_vib"] == 0).fillna(False)
        rev_min = (g["reverse_n_neg_vib"] == 0).fillna(False)
        both = int((fwd_min & rev_min).sum())
        either = int((fwd_min ^ rev_min).sum())
        neither = n - both - either
        ns.append(n)
        p_both.append(100 * both / n)
        p_one.append(100 * either / n)
        p_neither.append(100 * neither / n)

    x = np.arange(len(NOISES))
    width = 0.65

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.bar(x, p_both, width, label="both endpoints at minimum",
           color=C_INTENDED, edgecolor="white", linewidth=0.6)
    ax.bar(x, p_one, width, bottom=p_both, label="only one endpoint at minimum",
           color=C_HALF, edgecolor="white", linewidth=0.6)
    ax.bar(x, p_neither, width, bottom=np.array(p_both) + np.array(p_one),
           label="neither at minimum (ridge-stall)",
           color=C_UNINT, edgecolor="white", linewidth=0.6)

    for noise_i, (pb, po, pn) in enumerate(zip(p_both, p_one, p_neither)):
        if pb > 6:
            ax.text(x[noise_i], pb / 2, f"{pb:.1f}%",
                    ha="center", va="center", fontsize=9, color="white", fontweight="bold")
        if po > 3:
            ax.text(x[noise_i], pb + po / 2, f"{po:.1f}%",
                    ha="center", va="center", fontsize=8.5, color=palette_color(7))
        if pn > 3:
            ax.text(x[noise_i], pb + po + pn / 2, f"{pn:.1f}%",
                    ha="center", va="center", fontsize=8.5, color="white", fontweight="bold")

    for noise_i, n in enumerate(ns):
        ax.text(x[noise_i], 101.5, f"n={n}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{n} pm" for n in NOISES])
    ax.set_xlabel("starting TS noise")
    ax.set_ylabel("fraction of IRC runs (%)")
    ax.set_ylim(0, 108)
    ax.legend(loc="lower right", fontsize=10, framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)
    save(fig, "fig_sella_bars_endpoint")


def main() -> None:
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 11,
        "text.usetex": False,
    })
    apply_plot_style()
    irc = duckdb.execute(f"SELECT * FROM '{IRC_DIR}/*.parquet'").df()
    print(f"loaded {len(irc)} rows")

    # Criterion 1: bond-graph topology
    stacked_bars(
        irc,
        int_col="topology_intended",
        half_col="topology_half_intended",
        title="sella_hip IRC by starting TS noise — bond-graph topology criterion",
        subtitle="intended = both directions match labeled reactant/product by bond-graph isomorphism "
                 "(denominator = gad_dt003 converged TSs)",
        name="fig_sella_bars_topo",
    )

    # Criterion 2: strict RMSD (<0.3 A, direction-agnostic)
    stacked_bars(
        irc,
        int_col="intended",
        half_col="half_intended",
        title="sella_hip IRC by starting TS noise — strict RMSD criterion (<0.3 Å)",
        subtitle="intended = both endpoints match labeled reactant/product by Kabsch+Hungarian RMSD "
                 "(denominator = gad_dt003 converged TSs)",
        name="fig_sella_bars_rmsd",
    )

    # Criterion 3: endpoint vibrational quality
    endpoint_quality_bars(irc)


if __name__ == "__main__":
    main()
