#!/usr/bin/env python
"""IRC outcome bar charts (partitioned/stacked by intended / half / unintended).

For each (method, noise) cell, classify samples into:
  - intended (both endpoints match R/P)
  - half-intended (one endpoint matches)
  - unintended (neither)

Produces:
  figures/fig_irc_bars_topo.pdf    # TOPO version (chemistry ground truth)
  figures/fig_irc_bars_rmsd.pdf    # RMSD<0.3Å version (strict)
  figures/fig_irc_bars_combined.pdf  # 2-panel side-by-side TOPO + RMSD
"""
from __future__ import annotations

import os
from pathlib import Path

import duckdb
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

OUT = Path("/lustre06/project/6033559/memoozd/GAD_plus/figures")
OUT.mkdir(exist_ok=True, parents=True)
RUNS = Path("/lustre07/scratch/memoozd/gadplus/runs/test_irc")

NOISES = [10, 30, 50, 100, 150, 200]

# Methods (display label, source dir, color hue for the stack)
# Order matters: shows in the legend & x-axis grouping.
METHODS = [
    ("GAD dt=0.003 (5k)",   "gad_dt003_fmax",         "#08306b"),
    ("GAD dt=0.005 (5k)",   "gad_dt005_fmax",         "#2171b5"),
    ("GAD dt=0.007 (5k)",   "gad_dt007_fmax",         "#ff7f0e"),
    ("Sella libdef",        "sella_carteck_libdef",   "#d62728"),
    ("Sella default",       "sella_carteck_default",  "#9467bd"),
    ("Sella internal",      "sella_internal_default", "#8c564b"),
]

# Stack colors — same hues for both panels:
C_INT  = "#2ca02c"   # green: both endpoints match
C_HALF = "#ffbb78"   # peach: one endpoint matches
C_UNI  = "#bdbdbd"   # gray: neither


def get_irc_counts(method_dir: str, noise: int, kind: str = "topo") -> dict | None:
    """Return n_total, n_intended, n_half, n_unintended for the given cell.
    `kind` ∈ {'topo', 'rmsd'}.
    """
    p = RUNS / method_dir / f"irc_validation_sella_hip_allendpoints_{noise}pm.parquet"
    if not p.exists(): return None
    n_with_coords = duckdb.execute(f"SELECT COUNT(*) FROM '{p}' WHERE forward_coords_flat IS NOT NULL").fetchone()[0]
    if n_with_coords < 50:  # likely the broken pre-fix file
        return None
    if kind == "topo":
        col_full, col_half = "topology_intended", "topology_half_intended"
    else:
        col_full, col_half = "intended", "half_intended"
    df = duckdb.execute(f"""
      SELECT
        COUNT(*) AS n_total,
        SUM(CAST({col_full} AS INT)) AS n_int,
        SUM(CAST({col_half} AS INT)) AS n_half
      FROM '{p}'
    """).df()
    r = df.iloc[0]
    return {
        "n_total": int(r["n_total"]),
        "n_int": int(r["n_int"]),
        "n_half": int(r["n_half"]),
        "n_uni": int(r["n_total"]) - int(r["n_int"]) - int(r["n_half"]),
    }


def draw_panel(ax, kind: str, title: str, ylabel: bool = True):
    """One panel = grouped bars (group=noise level, bar=method, stacked
    by intended/half/unintended)."""
    n_methods = len(METHODS)
    n_noises = len(NOISES)
    bar_w = 0.13
    group_w = bar_w * n_methods
    method_offsets = np.linspace(-group_w/2 + bar_w/2, group_w/2 - bar_w/2, n_methods)

    for j, (label, mdir, hue) in enumerate(METHODS):
        for i, noise in enumerate(NOISES):
            r = get_irc_counts(mdir, noise, kind=kind)
            x = i + method_offsets[j]
            if r is None or r["n_total"] == 0:
                # Missing cell: draw an outlined empty bar
                ax.bar(x, 0, bar_w, color="white", edgecolor="lightgray", linewidth=0.5)
                ax.text(x, 1, "—", ha="center", va="bottom", fontsize=6, color="gray")
                continue
            n = r["n_total"]
            p_int  = 100 * r["n_int"]  / n
            p_half = 100 * r["n_half"] / n
            p_uni  = 100 * r["n_uni"]  / n
            # Stacked: bottom = intended (green), middle = half (peach), top = unintended (gray)
            ax.bar(x, p_int,  bar_w, color=C_INT,  edgecolor=hue, linewidth=1.0)
            ax.bar(x, p_half, bar_w, color=C_HALF, edgecolor=hue, linewidth=1.0,
                   bottom=p_int)
            ax.bar(x, p_uni,  bar_w, color=C_UNI,  edgecolor=hue, linewidth=1.0,
                   bottom=p_int + p_half, alpha=0.55)
            # Label intended pct on top of green segment if it's tall enough
            if p_int >= 8:
                ax.text(x, p_int / 2, f"{p_int:.0f}", ha="center", va="center",
                        fontsize=6, color="white", fontweight="bold")

    ax.set_xticks(range(n_noises))
    ax.set_xticklabels([f"{n}pm" for n in NOISES])
    ax.set_xlabel("TS-noise (pm)", fontsize=10)
    if ylabel:
        ax.set_ylabel("% of n=287 samples", fontsize=10)
    ax.set_ylim(0, 105)
    ax.grid(alpha=0.25, axis="y")
    ax.set_title(title, fontsize=11, fontweight="bold")

    # Method legend (one per color) — horizontally below the title
    method_handles = [Patch(facecolor="white", edgecolor=hue, linewidth=1.5, label=label)
                      for label, _, hue in METHODS]
    ax.legend(handles=method_handles,
              title="bar edge color = method",
              loc="upper right", fontsize=7, title_fontsize=8,
              ncol=1, framealpha=0.95)

    # Stack legend (intended/half/unintended) inside the panel
    stack_handles = [
        Patch(facecolor=C_INT,  edgecolor="black", linewidth=0.5, label="intended (both R+P)"),
        Patch(facecolor=C_HALF, edgecolor="black", linewidth=0.5, label="half (one of R/P)"),
        Patch(facecolor=C_UNI,  edgecolor="black", linewidth=0.5, alpha=0.55, label="unintended"),
    ]
    leg2 = ax.legend(handles=stack_handles, loc="lower right", fontsize=8,
                     title="stack fill = outcome", title_fontsize=8, framealpha=0.95)
    ax.add_artist(leg2)
    # restore the method legend (matplotlib only shows the last legend)
    ax.legend(handles=method_handles,
              title="bar edge color = method",
              loc="upper right", fontsize=7, title_fontsize=8,
              ncol=1, framealpha=0.95)
    ax.add_artist(leg2)


def fig_topo_only():
    fig, ax = plt.subplots(figsize=(13, 6))
    draw_panel(ax, kind="topo",
               title="IRC TOPO-intended outcomes — chemistry ground truth (test, n=287)")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_irc_bars_topo.{ext}", bbox_inches="tight", dpi=140)
    plt.close(fig)
    print("wrote fig_irc_bars_topo")


def fig_rmsd_only():
    fig, ax = plt.subplots(figsize=(13, 6))
    draw_panel(ax, kind="rmsd",
               title="IRC RMSD-intended outcomes (Kabsch+Hungarian, threshold 0.3Å)")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_irc_bars_rmsd.{ext}", bbox_inches="tight", dpi=140)
    plt.close(fig)
    print("wrote fig_irc_bars_rmsd")


def fig_combined():
    fig, axes = plt.subplots(2, 1, figsize=(14, 11), sharex=True)
    draw_panel(axes[0], kind="topo",
               title="(A) IRC TOPO-intended (element-aware bond-graph isomorphism — chemistry ground truth)",
               ylabel=True)
    draw_panel(axes[1], kind="rmsd",
               title="(B) IRC RMSD-intended (Kabsch+Hungarian RMSD < 0.3Å — strict geometry)",
               ylabel=True)
    fig.suptitle("IRC validation outcomes by (method, noise) — test split, n=287",
                 fontsize=13, fontweight="bold", y=1.005)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_irc_bars_combined.{ext}", bbox_inches="tight", dpi=140)
    plt.close(fig)
    print("wrote fig_irc_bars_combined")


def main():
    fig_topo_only()
    fig_rmsd_only()
    fig_combined()


if __name__ == "__main__":
    main()
