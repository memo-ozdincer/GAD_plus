#!/usr/bin/env python
"""GAD final-state RMSD-to-TS distributions on test, from traj parquets.

Reads test_set/test_dtgrid traj parquets, takes the LAST recorded step
per (run_id, sample_id), and uses the existing 'dist_to_known_ts' column
(already RMSD-aligned) for the histogram. Combines with Sella RMSD
distributions from analyze_rmsd_bimodal.py output to make the
'unimodal vs bimodal' comparison figure.

Output: figures/fig_rmsd_distrib_combined.pdf
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path("/lustre07/scratch/memoozd/gadplus/runs")
OUT = Path("/lustre06/project/6033559/memoozd/GAD_plus/analysis_2026_04_29")
FIG = Path("/lustre06/project/6033559/memoozd/GAD_plus/figures")
OUT.mkdir(exist_ok=True, parents=True)
NOISES = [10, 30, 50, 100, 150, 200]


def gad_final_rmsd(method_dir, noise_pm):
    """Last-step dist_to_known_ts per sample, from traj parquets."""
    if not method_dir.exists():
        return None
    glob = f"{method_dir}/traj_*_{noise_pm}pm_*.parquet"
    try:
        df = duckdb.execute(f"""
            SELECT sample_id, dist_to_known_ts AS rmsd_to_ts, n_neg, force_max, total_steps
            FROM (
                SELECT sample_id, dist_to_known_ts, n_neg, force_max,
                       step AS total_steps,
                       ROW_NUMBER() OVER (PARTITION BY sample_id ORDER BY step DESC) AS rn
                FROM '{glob}'
            )
            WHERE rn = 1
        """).df()
        return df
    except Exception as e:
        print(f"  err on {method_dir}/{noise_pm}pm: {e}")
        return None


def main():
    # GAD methods (keys -> dir)
    gad_methods = {
        "GAD dt=0.003 (2k)":  BASE / "test_set/gad_dt003_fmax",
        "GAD dt=0.005 (2k)":  BASE / "test_set/gad_dt005_fmax",
        "GAD dt=0.003 (5k)":  BASE / "test_dtgrid/gad_dt003_fmax",
        "GAD dt=0.007 (5k)":  BASE / "test_dtgrid/gad_dt007_fmax",
    }

    rows = []
    for label, mdir in gad_methods.items():
        for n in NOISES:
            df = gad_final_rmsd(mdir, n)
            if df is None or not len(df): continue
            for _, r in df.iterrows():
                rows.append({"method": label, "noise_pm": n,
                             "rmsd_to_ts": float(r["rmsd_to_ts"]),
                             "n_neg": int(r["n_neg"]),
                             "force_max": float(r["force_max"]),
                             "total_steps": int(r["total_steps"])})
    gad_df = pd.DataFrame(rows)
    gad_df.to_csv(OUT / "gad_test_rmsd.csv", index=False)
    print(f"GAD: {len(gad_df)} rows")

    # Load Sella from earlier
    sella_csv = OUT / "test_summary_full.csv"
    if sella_csv.exists():
        sella_df = pd.read_csv(sella_csv)
        sella_df = sella_df[["method", "noise_pm", "rmsd_to_ts"]].copy()
    else:
        sella_df = pd.DataFrame()

    # Merge & figure
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.5))
    axes = axes.flatten()
    palette = {
        "GAD dt=0.003 (5k)":  "#1f77b4",
        "GAD dt=0.007 (5k)":  "#17becf",
        "Sella libdef":       "#d62728",
        "Sella default":      "#ff7f0e",
        "Sella internal":     "#9467bd",
    }
    for ax, noise in zip(axes, NOISES):
        for label, color in palette.items():
            if label.startswith("GAD"):
                g = gad_df[(gad_df["method"] == label) & (gad_df["noise_pm"] == noise)]
            else:
                g = sella_df[(sella_df["method"] == label) & (sella_df["noise_pm"] == noise)]
            if not len(g): continue
            r = g["rmsd_to_ts"].dropna().values
            r = r[(r >= 0) & (r < 5)]
            if len(r):
                ax.hist(r, bins=50, alpha=0.45, color=color, label=label, density=True)
        ax.set_title(f"{noise} pm")
        ax.set_xlabel("RMSD to true TS (Å)")
        ax.set_ylabel("density")
        ax.set_xlim(0, 1.5)
        ax.legend(fontsize=7)
    fig.suptitle("Final-geometry RMSD-to-TS: GAD (5k steps) vs Sella (test, n=287)")
    fig.tight_layout()
    fig.savefig(FIG / "fig_rmsd_distrib_combined.pdf", bbox_inches="tight")
    fig.savefig(FIG / "fig_rmsd_distrib_combined.png", bbox_inches="tight", dpi=150)
    print(f"wrote {FIG / 'fig_rmsd_distrib_combined.pdf'}")

    # Also print bimodality analysis
    print("\nBimodality test: fraction in 'close' (<0.05Å) and 'far' (>0.5Å) bins")
    print("=" * 80)
    print(f"{'method':<22} {'noise':>5}  {'<0.05Å':>8} {'0.05-0.5Å':>10} {'>0.5Å':>8}  {'n':>4}")
    for label in palette:
        if label.startswith("GAD"):
            src = gad_df
        else:
            src = sella_df
        for n in NOISES:
            g = src[(src["method"] == label) & (src["noise_pm"] == n)]
            if not len(g): continue
            r = g["rmsd_to_ts"].dropna().values
            tot = len(r)
            close = (r < 0.05).sum()
            mid = ((r >= 0.05) & (r < 0.5)).sum()
            far = (r >= 0.5).sum()
            print(f"{label:<22} {n:>5}  {100*close/tot:>7.1f}% {100*mid/tot:>9.1f}%  "
                  f"{100*far/tot:>7.1f}%  {tot:>4}")


if __name__ == "__main__":
    main()
