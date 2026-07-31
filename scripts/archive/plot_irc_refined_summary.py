#!/usr/bin/env python
"""Plot summary figures for the refined IRC rerun.

Reads the saved IRC validation parquet outputs and writes:
1. A stacked-bar summary of what happened at each noise level.
2. A threshold-sweep plot showing how many refined index-1 candidates
   would pass under looser post-refinement fmax criteria.
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plotting_style import apply_plot_style, palette_color

apply_plot_style()

RUNS = "/lustre07/scratch/memoozd/gadplus/runs/irc_validation_300"
OUT = "/lustre06/project/6033559/memoozd/GAD_plus/figures"
os.makedirs(OUT, exist_ok=True)

NOISE_LEVELS = [0, 10, 50]
THRESHOLDS = [0.005, 0.0055, 0.006, 0.0065, 0.007, 0.008, 0.01]

COLORS = {
    "criterion_failed": palette_color(7),
    "unintended": palette_color(1),
    "topology_half_only": palette_color(8),
    "half_intended": palette_color(9),
    "intended": palette_color(2),
    "sweep": palette_color(0),
}


def _load(noise_pm: int) -> pd.DataFrame:
    return pd.read_parquet(f"{RUNS}/irc_validation_{noise_pm}pm.parquet")


def panel_outcomes(ax: plt.Axes) -> None:
    x = np.arange(len(NOISE_LEVELS))
    width = 0.72

    criterion_failed = []
    unintended = []
    topology_half_only = []
    half_intended = []
    intended = []

    for noise in NOISE_LEVELS:
        df = _load(noise)
        criterion_failed.append(int((df["error"] == "ts_quality_gate_failed").sum()))
        ok = df[df["error"].isna()].copy()
        intended.append(int(ok["intended"].fillna(False).sum()))
        half_mask = ok["half_intended"].fillna(False)
        topo_half_mask = ok["topology_half_intended"].fillna(False)
        half_intended.append(int(half_mask.sum()))
        topology_half_only.append(int((~half_mask & topo_half_mask).sum()))
        unintended.append(int((~half_mask & ~topo_half_mask & ~ok["intended"].fillna(False)).sum()))

    bottom = np.zeros(len(NOISE_LEVELS))
    for label, values, color in [
        ("TS criterion failed", criterion_failed, COLORS["criterion_failed"]),
        ("Unintended", unintended, COLORS["unintended"]),
        ("Topology-half only", topology_half_only, COLORS["topology_half_only"]),
        ("Half-intended", half_intended, COLORS["half_intended"]),
        ("Intended", intended, COLORS["intended"]),
    ]:
        ax.bar(x, values, width=width, bottom=bottom, color=color, edgecolor="white", linewidth=1.0, label=label)
        bottom += np.asarray(values)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{n} pm" for n in NOISE_LEVELS])
    ax.set_ylabel("Count (30 samples)")
    ax.set_title("Refined IRC Outcomes")
    ax.set_ylim(0, 30)
    ax.grid(True, axis="y", alpha=0.2)
    ax.legend(frameon=False, fontsize=9, loc="upper right")


def panel_threshold_sweep(ax: plt.Axes) -> None:
    for noise in NOISE_LEVELS:
        df = _load(noise)
        cand = df[df["refined_force_max"].notna()].copy()
        counts = []
        for thr in THRESHOLDS:
            passes_criterion = (cand["refined_force_max"] < thr) & cand["refined_n_neg"].eq(1)
            counts.append(int(passes_criterion.sum()))
        ax.plot(THRESHOLDS, counts, marker="o", linewidth=2, label=f"{noise} pm")

    ax.axvline(0.005, color=palette_color(7), linestyle="--", alpha=0.6)
    ax.axvline(0.006, color=COLORS["sweep"], linestyle=":", alpha=0.8)
    ax.text(0.00502, 2.0, "old refine criterion", rotation=90, va="bottom", ha="left", fontsize=9, color=palette_color(7))
    ax.text(0.00602, 2.0, "new refine criterion", rotation=90, va="bottom", ha="left", fontsize=9, color=COLORS["sweep"])
    ax.set_xlabel("Post-refinement fmax criterion")
    ax.set_ylabel("Refined index-1 TS admitted")
    ax.set_title("Threshold Sensitivity of Saved Results")
    ax.set_xlim(min(THRESHOLDS) - 0.0001, max(THRESHOLDS) + 0.0002)
    ax.set_ylim(0, 31)
    ax.grid(True, alpha=0.2)
    ax.legend(frameon=False)


def main() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    panel_outcomes(axes[0])
    panel_threshold_sweep(axes[1])
    fig.suptitle("IRC Validation After GAD TS Refinement", fontsize=15, fontweight="bold")
    fig.tight_layout()

    out_png = f"{OUT}/fig_irc_refined_summary.png"
    out_pdf = f"{OUT}/fig_irc_refined_summary.pdf"
    fig.savefig(out_png)
    fig.savefig(out_pdf)
    plt.close(fig)
    print(f"Saved {out_png}")
    print(f"Saved {out_pdf}")


if __name__ == "__main__":
    main()
