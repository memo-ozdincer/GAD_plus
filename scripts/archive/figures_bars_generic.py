#!/usr/bin/env python
"""Produce 3-way bar charts (intended/half/unintended) per criterion for a given IRC results dir.

Usage: python scripts/figures_bars_generic.py <input_dir> <output_prefix>

Writes fig_{prefix}_bars_{topo,rmsd,endpoint}.{pdf,png} into figures/.
"""
from __future__ import annotations

import sys
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

C_INTENDED = palette_color(2)
C_HALF     = palette_color(8)
C_UNINT    = palette_color(3)


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"wrote {OUT}/{name}.(pdf|png)")


def stacked_bars(irc, int_col, half_col, title, subtitle, name, noises):
    ns, pi, ph, pu = [], [], [], []
    for n_ in noises:
        g = irc[irc["noise_pm"] == n_]
        n = len(g)
        if n == 0:
            continue
        ni = int(g[int_col].sum())
        nh = int((g[half_col] & ~g[int_col]).sum())
        nu = n - ni - nh
        ns.append(n)
        pi.append(100 * ni / n)
        ph.append(100 * nh / n)
        pu.append(100 * nu / n)

    x = np.arange(len(ns))
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.bar(x, pi, 0.65, label="intended", color=C_INTENDED, edgecolor="white", linewidth=0.6)
    ax.bar(x, ph, 0.65, bottom=pi, label="half-intended", color=C_HALF, edgecolor="white", linewidth=0.6)
    ax.bar(x, pu, 0.65, bottom=np.array(pi) + np.array(ph), label="unintended",
           color=C_UNINT, edgecolor="white", linewidth=0.6)

    for i, (a, b, c) in enumerate(zip(pi, ph, pu)):
        if a > 6:
            ax.text(x[i], a / 2, f"{a:.1f}%", ha="center", va="center", fontsize=9, color="white", fontweight="bold")
        if b > 6:
            ax.text(x[i], a + b / 2, f"{b:.1f}%", ha="center", va="center", fontsize=9, color=palette_color(7))
        if c > 6:
            ax.text(x[i], a + b + c / 2, f"{c:.1f}%", ha="center", va="center", fontsize=9, color="white", fontweight="bold")
    for i, n in enumerate(ns):
        ax.text(x[i], 101.5, f"n={n}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{n_} pm" for n_ in noises[:len(ns)]])
    ax.set_xlabel("starting TS noise")
    ax.set_ylabel("fraction of IRC runs (%)")
    ax.set_ylim(0, 108)
    ax.legend(loc="lower right", fontsize=10, framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)
    save(fig, name)


def endpoint_bars(irc, title, subtitle, name, noises):
    ns, pb, po, pn = [], [], [], []
    for n_ in noises:
        g = irc[irc["noise_pm"] == n_]
        n = len(g)
        if n == 0:
            continue
        fm = (g["forward_n_neg_vib"] == 0).fillna(False)
        rm = (g["reverse_n_neg_vib"] == 0).fillna(False)
        both = int((fm & rm).sum())
        one  = int((fm ^ rm).sum())
        nei  = n - both - one
        ns.append(n); pb.append(100*both/n); po.append(100*one/n); pn.append(100*nei/n)

    x = np.arange(len(ns))
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.bar(x, pb, 0.65, label="both endpoints at minimum", color=C_INTENDED, edgecolor="white", linewidth=0.6)
    ax.bar(x, po, 0.65, bottom=pb, label="only one endpoint at minimum", color=C_HALF, edgecolor="white", linewidth=0.6)
    ax.bar(x, pn, 0.65, bottom=np.array(pb) + np.array(po), label="neither at minimum (ridge-stall)",
           color=C_UNINT, edgecolor="white", linewidth=0.6)

    for i, (a, b, c) in enumerate(zip(pb, po, pn)):
        if a > 6: ax.text(x[i], a/2, f"{a:.1f}%", ha="center", va="center", fontsize=9, color="white", fontweight="bold")
        if b > 3: ax.text(x[i], a + b/2, f"{b:.1f}%", ha="center", va="center", fontsize=8.5, color=palette_color(7))
        if c > 3: ax.text(x[i], a + b + c/2, f"{c:.1f}%", ha="center", va="center", fontsize=8.5, color="white", fontweight="bold")
    for i, n in enumerate(ns):
        ax.text(x[i], 101.5, f"n={n}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{n_} pm" for n_ in noises[:len(ns)]])
    ax.set_xlabel("starting TS noise")
    ax.set_ylabel("fraction of IRC runs (%)")
    ax.set_ylim(0, 108)
    ax.legend(loc="lower right", fontsize=10, framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)
    save(fig, name)


def main():
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "text.usetex": False})
    apply_plot_style()
    input_dir = sys.argv[1]
    prefix = sys.argv[2]
    label_detail = sys.argv[3] if len(sys.argv) > 3 else "IRC validation"
    noises = [10, 30, 50, 100, 150, 200]
    irc = duckdb.execute(f"SELECT * FROM '{input_dir}/*.parquet'").df()
    print(f"loaded {len(irc)} rows from {input_dir}")

    stacked_bars(irc, "topology_intended", "topology_half_intended",
                 f"{label_detail} — bond-graph topology criterion",
                 "intended = forward+reverse match labeled (R,P) pair by bond-graph isomorphism",
                 f"fig_{prefix}_bars_topo", noises)
    stacked_bars(irc, "intended", "half_intended",
                 f"{label_detail} — strict RMSD criterion (<0.3 Å)",
                 "intended = both endpoints match labeled (R,P) by Kabsch+Hungarian RMSD",
                 f"fig_{prefix}_bars_rmsd", noises)
    endpoint_bars(irc,
                  f"{label_detail} — endpoint vibrational quality",
                  "both / one / neither IRC endpoint at a true minimum (n_neg,vib = 0)",
                  f"fig_{prefix}_bars_endpoint", noises)


if __name__ == "__main__":
    main()
