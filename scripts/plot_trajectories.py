#!/usr/bin/env python
"""Phase 4: Trajectory visualization from Parquet data.

Picks 3 representative trajectories (fast convergence, slow convergence, failure)
and plots 2x2 grids: energy, force_norm, n_neg, eigenvalues vs step.

No GPU needed — reads Parquet files only.

Usage:
  python scripts/plot_trajectories.py --traj-dir /path/to/runs/noise_survey
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_data(traj_dir: str):
    """Load summary and trajectory data via DuckDB."""
    import duckdb

    # Find summary files
    summary_pattern = os.path.join(traj_dir, "summary_*.parquet")
    traj_pattern = os.path.join(traj_dir, "traj_*.parquet")

    summary_df = duckdb.execute(f"""
        SELECT * FROM '{summary_pattern}'
    """).df()

    return summary_df, traj_pattern


def pick_representatives(summary_df):
    """Pick 3 representative runs: fast converge, slow converge, failure."""
    converged = summary_df[summary_df["converged"] == True].copy()
    failed = summary_df[summary_df["converged"] == False].copy()

    picks = {}

    if len(converged) > 0:
        # Fast: lowest converged_step
        fast_idx = converged["converged_step"].idxmin()
        picks["fast_convergence"] = converged.loc[fast_idx]

        # Slow: highest converged_step
        if len(converged) > 1:
            slow_idx = converged["converged_step"].idxmax()
            picks["slow_convergence"] = converged.loc[slow_idx]

    if len(failed) > 0:
        # Failure: pick one with highest total_steps (ran the longest)
        fail_idx = failed["total_steps"].idxmax()
        picks["failure"] = failed.loc[fail_idx]

    return picks


def plot_trajectory(traj_df, title: str, output_path: str):
    """Plot 2x2 grid: energy, force_norm, n_neg, eigenvalues vs step."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    steps = traj_df["step"].values

    # Energy vs step
    ax = axes[0, 0]
    ax.plot(steps, traj_df["energy"].values, "b-", linewidth=1)
    ax.set_xlabel("Step")
    ax.set_ylabel("Energy (eV)")
    ax.set_title("Energy")
    ax.grid(True, alpha=0.3)

    # Force norm vs step
    ax = axes[0, 1]
    ax.semilogy(steps, traj_df["force_norm"].values, "r-", linewidth=1)
    ax.axhline(y=0.01, color="g", linestyle="--", alpha=0.7, label="threshold (0.01)")
    ax.set_xlabel("Step")
    ax.set_ylabel("Force norm (eV/A)")
    ax.set_title("Force convergence")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # n_neg vs step
    ax = axes[1, 0]
    ax.plot(steps, traj_df["n_neg"].values, "k-", linewidth=1, drawstyle="steps-post")
    ax.axhline(y=1, color="g", linestyle="--", alpha=0.7, label="target (n_neg=1)")
    ax.set_xlabel("Step")
    ax.set_ylabel("n_neg")
    ax.set_title("Negative eigenvalue count")
    ax.set_ylim(-0.5, max(traj_df["n_neg"].max() + 1, 5))
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Eigenvalues vs step
    ax = axes[1, 1]
    ax.plot(steps, traj_df["eig0"].values, "b-", linewidth=1, label="eig0 (lowest)")
    ax.plot(steps, traj_df["eig1"].values, "r-", linewidth=1, label="eig1")
    ax.axhline(y=0, color="k", linestyle="-", alpha=0.3)
    ax.set_xlabel("Step")
    ax.set_ylabel("Eigenvalue")
    ax.set_title("Lowest eigenvalues")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--traj-dir", type=str, required=True,
                        help="Directory with summary_*.parquet and traj_*.parquet")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(args.traj_dir, "plots")
    os.makedirs(output_dir, exist_ok=True)

    import duckdb

    summary_df, traj_pattern = load_data(args.traj_dir)
    print(f"Loaded {len(summary_df)} summary records")
    print(f"  Converged: {summary_df['converged'].sum()}/{len(summary_df)}")

    picks = pick_representatives(summary_df)
    if not picks:
        print("No trajectories to plot.")
        return

    for label, row in picks.items():
        run_id = row["run_id"]
        sample_id = int(row["sample_id"])
        formula = row.get("formula", "unknown")
        conv_step = row.get("converged_step", None)
        start_method = row.get("start_method", "unknown")

        print(f"\n--- {label}: sample={sample_id}, formula={formula}, "
              f"start={start_method}, conv_step={conv_step} ---")

        # Load trajectory for this run+sample
        traj_df = duckdb.execute(f"""
            SELECT step, energy, force_norm, n_neg, eig0, eig1,
                   mode_overlap, disp_from_start, dist_to_known_ts
            FROM '{traj_pattern}'
            WHERE run_id = '{run_id}' AND sample_id = {sample_id}
            ORDER BY step
        """).df()

        if len(traj_df) == 0:
            print(f"  No trajectory data found for run_id={run_id}, sample_id={sample_id}")
            continue

        title = f"{label.replace('_', ' ').title()}: {formula} ({start_method})"
        if conv_step is not None and not np.isnan(conv_step):
            title += f" — converged at step {int(conv_step)}"
        else:
            title += f" — failed ({row.get('failure_type', 'unknown')})"

        out_path = os.path.join(output_dir, f"{label}.png")
        plot_trajectory(traj_df, title, out_path)

    print(f"\nAll plots saved to {output_dir}")


if __name__ == "__main__":
    main()
