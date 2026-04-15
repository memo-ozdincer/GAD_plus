#!/usr/bin/env python
"""Plot refined fmax distributions for attempted vs gate-failed IRC rows."""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RUNS = "/lustre07/scratch/memoozd/gadplus/runs/irc_validation_300"
OUT = "/lustre06/project/6033559/memoozd/GAD_plus/figures"
os.makedirs(OUT, exist_ok=True)

NOISE_LEVELS = [0, 10, 50]
COLORS = {
    "attempted": "#0072B2",
    "failed": "#D55E00",
}


def main() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.5), sharey=True)

    for ax, noise in zip(axes, NOISE_LEVELS):
        df = pd.read_parquet(f"{RUNS}/irc_validation_{noise}pm.parquet")
        attempted = df.loc[df["error"].isna(), "refined_force_max"].dropna().to_numpy()
        failed = df.loc[df["error"] == "ts_quality_gate_failed", "refined_force_max"].dropna().to_numpy()

        parts = ax.violinplot(
            [attempted, failed],
            positions=[1, 2],
            showmeans=False,
            showextrema=False,
            showmedians=True,
            widths=0.8,
        )
        for i, body in enumerate(parts["bodies"]):
            body.set_facecolor(COLORS["attempted"] if i == 0 else COLORS["failed"])
            body.set_edgecolor("white")
            body.set_alpha(0.65)
        parts["cmedians"].set_color("black")

        rng = np.random.default_rng(1234 + noise)
        if len(attempted):
            ax.scatter(1 + rng.normal(0, 0.04, len(attempted)), attempted, s=18, c=COLORS["attempted"], alpha=0.85)
        if len(failed):
            ax.scatter(2 + rng.normal(0, 0.04, len(failed)), failed, s=18, c=COLORS["failed"], alpha=0.85)

        ax.axhline(0.005, color="gray", linestyle="--", alpha=0.7)
        ax.axhline(0.006, color="#009E73", linestyle=":", alpha=0.85)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["IRC ran", "Gate failed"])
        ax.set_title(f"{noise} pm")
        ax.grid(True, axis="y", alpha=0.2)

    axes[0].set_ylabel("Refined TS fmax")
    fig.suptitle("Refined TS Force Levels for Attempted vs Skipped IRC Rows", fontsize=15, fontweight="bold")
    fig.tight_layout()

    out_png = f"{OUT}/fig_irc_refined_fmax.png"
    out_pdf = f"{OUT}/fig_irc_refined_fmax.pdf"
    fig.savefig(out_png)
    fig.savefig(out_pdf)
    plt.close(fig)
    print(f"Saved {out_png}")
    print(f"Saved {out_pdf}")


if __name__ == "__main__":
    main()
