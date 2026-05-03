#!/usr/bin/env python
"""Comprehensive figure script for the test-set comparison (2026-04-29).

Outputs to figures/:
  fig_test_conv_3thresh.pdf      - 3-method conv vs noise, 3 fmax thresholds
  fig_test_dtgrid.pdf            - GAD dt grid at fmax<0.01
  fig_test_threshold_curves.pdf  - cumulative conv vs threshold (per method, per noise)
  fig_test_rmsd_distrib.pdf      - GAD vs Sella RMSD distrib (already from analyze_rmsd_gad)
  fig_test_steps_hist.pdf        - histogram of step counts (converged-only)
"""
from __future__ import annotations

import os
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path("/lustre07/scratch/memoozd/gadplus/runs")
OUT  = Path("/lustre06/project/6033559/memoozd/GAD_plus/figures")
ANL  = Path("/lustre06/project/6033559/memoozd/GAD_plus/analysis_2026_04_29")
OUT.mkdir(exist_ok=True, parents=True)
NOISES = [10, 30, 50, 100, 150, 200]

C_GAD003 = "#1f77b4"
C_GAD007 = "#17becf"
C_SELLA_LIB = "#d62728"
C_SELLA_DEF = "#ff7f0e"
C_SELLA_INT = "#9467bd"


def _save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"wrote {name}")


def find_summary(mdir, noise_pm):
    if not mdir.exists(): return None
    cands = [f for f in os.listdir(mdir)
             if f.startswith("summary") and f.endswith(".parquet") and f"_{noise_pm}pm" in f]
    return mdir / cands[0] if cands else None


def conv_at(path, threshold, kind="fmax"):
    if not path or not path.exists(): return None
    cols = set(duckdb.execute(f"DESCRIBE SELECT * FROM '{path}'").df()["column_name"])
    fmax_col = "final_fmax" if "final_fmax" in cols else "final_force_max"
    if kind == "fmax":
        col = fmax_col
    else:
        col = "final_force_norm"
    r = duckdb.execute(
        f"SELECT 100.0*AVG(CASE WHEN final_n_neg=1 AND {col} < {threshold} THEN 1.0 ELSE 0.0 END) "
        f"FROM '{path}'"
    ).fetchone()
    return r[0]


METHODS_3M = {
    "GAD dt=0.007 (5k)":  (BASE / "test_dtgrid/gad_dt007_fmax", C_GAD007, "o"),
    "Sella libdef":       (BASE / "test_set/sella_carteck_libdef", C_SELLA_LIB, "s"),
    "Sella internal":     (BASE / "test_set/sella_internal_default", C_SELLA_INT, "v"),
}


def fig_conv_3thresh():
    """3 methods × 3 thresholds (rows) × noise (x). 1 fig with 3 subplots."""
    thresholds = [0.05, 0.01, 0.005]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), sharey=True)
    for ax, thresh in zip(axes, thresholds):
        for label, (mdir, color, marker) in METHODS_3M.items():
            xs, ys = [], []
            for n in NOISES:
                r = conv_at(find_summary(mdir, n), thresh, "fmax")
                if r is None: continue
                xs.append(n); ys.append(r)
            if xs:
                ax.plot(xs, ys, "-", color=color, marker=marker, linewidth=2.2,
                        markersize=9, markerfacecolor="white", markeredgewidth=2,
                        label=label)
        ax.set_title(f"fmax < {thresh}")
        ax.set_xlabel("TS noise (pm)")
        ax.set_xticks(NOISES)
        ax.set_ylim(0, 105)
        ax.grid(alpha=0.3)
        ax.legend(loc="lower left", fontsize=9)
    axes[0].set_ylabel("TS-converged rate (%)")
    fig.suptitle("3-method comparison vs convergence threshold (test, n=287)", y=1.02)
    fig.tight_layout()
    _save(fig, "fig_test_conv_3thresh")


def fig_dtgrid():
    """GAD dt grid at fmax<0.01 — one line per dt."""
    fig, ax = plt.subplots(figsize=(8.5, 5))
    dts = [("0.003", "#1f77b4"), ("0.004", "#17becf"), ("0.005", "#2ca02c"),
           ("0.006", "#bcbd22"), ("0.007", "#ff7f0e"), ("0.008", "#d62728")]
    for dt, color in dts:
        mdir = BASE / f"test_dtgrid/gad_dt{dt.replace('.','').replace('00','0')[1:]}_fmax"
        # safer: hardcode mapping
    # Use hardcoded dirs
    methods = [
        ("dt=0.003", BASE / "test_dtgrid/gad_dt003_fmax", "#1f77b4", "o"),
        ("dt=0.004", BASE / "test_dtgrid/gad_dt004_fmax", "#17becf", "D"),
        ("dt=0.005", BASE / "test_dtgrid/gad_dt005_fmax", "#2ca02c", "s"),
        ("dt=0.006", BASE / "test_dtgrid/gad_dt006_fmax", "#bcbd22", "p"),
        ("dt=0.007", BASE / "test_dtgrid/gad_dt007_fmax", "#ff7f0e", "^"),
        ("dt=0.008", BASE / "test_dtgrid/gad_dt008_fmax", "#d62728", "v"),
    ]
    for label, mdir, color, marker in methods:
        xs, ys = [], []
        for n in NOISES:
            r = conv_at(find_summary(mdir, n), 0.01, "fmax")
            if r is None: continue
            xs.append(n); ys.append(r)
        if xs:
            ax.plot(xs, ys, "-", color=color, marker=marker, linewidth=2,
                    markersize=8, markerfacecolor="white", markeredgewidth=1.8,
                    label=label)
    ax.set_xlabel("TS noise (pm)")
    ax.set_ylabel("TS-converged rate (%, fmax<0.01)")
    ax.set_title("GAD dt grid on test (5000 steps, n=287)")
    ax.set_xticks(NOISES)
    ax.set_ylim(0, 105)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left", fontsize=10)
    _save(fig, "fig_test_dtgrid")


def fig_threshold_curves():
    """Convergence rate as a function of threshold, one panel per noise level."""
    thresh_grid = np.geomspace(1e-4, 0.1, 30)
    methods = [
        ("GAD dt=0.007", BASE / "test_dtgrid/gad_dt007_fmax", C_GAD007),
        ("Sella libdef", BASE / "test_set/sella_carteck_libdef", C_SELLA_LIB),
        ("Sella internal", BASE / "test_set/sella_internal_default", C_SELLA_INT),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.5), sharey=True, sharex=True)
    axes = axes.flatten()
    for ax, n in zip(axes, NOISES):
        for label, mdir, color in methods:
            p = find_summary(mdir, n)
            if not p: continue
            cols = set(duckdb.execute(f"DESCRIBE SELECT * FROM '{p}'").df()["column_name"])
            fmax_col = "final_fmax" if "final_fmax" in cols else "final_force_max"
            df = duckdb.execute(
                f"SELECT final_n_neg AS nn, {fmax_col} AS fmax FROM '{p}'"
            ).df()
            rates = []
            for t in thresh_grid:
                rates.append(100 * ((df["nn"] == 1) & (df["fmax"] < t)).sum() / len(df))
            ax.semilogx(thresh_grid, rates, color=color, linewidth=2, label=label)
        ax.axvline(0.05, color="gray", linestyle=":", alpha=0.5, label="Sella default")
        ax.axvline(0.01, color="gray", linestyle="--", alpha=0.5, label="Our canonical")
        ax.set_title(f"{n} pm")
        ax.set_xlabel("fmax threshold")
        ax.set_ylim(0, 105)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("TS-converged rate (%)")
    axes[3].set_ylabel("TS-converged rate (%)")
    axes[2].legend(loc="lower right", fontsize=8)
    fig.suptitle("Convergence rate vs fmax threshold (n=287, n_neg=1 ∧ fmax<thresh)", y=1.01)
    fig.tight_layout()
    _save(fig, "fig_test_threshold_curves")


def fig_steps_hist():
    """Histogram of total_steps for converged samples, per method."""
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.5))
    axes = axes.flatten()
    methods = [
        ("GAD dt=0.007", BASE / "test_dtgrid/gad_dt007_fmax", C_GAD007),
        ("Sella libdef", BASE / "test_set/sella_carteck_libdef", C_SELLA_LIB),
    ]
    for ax, n in zip(axes, NOISES):
        for label, mdir, color in methods:
            p = find_summary(mdir, n)
            if not p: continue
            cols = set(duckdb.execute(f"DESCRIBE SELECT * FROM '{p}'").df()["column_name"])
            fmax_col = "final_fmax" if "final_fmax" in cols else "final_force_max"
            df = duckdb.execute(
                f"SELECT total_steps, final_n_neg AS nn, {fmax_col} AS fmax "
                f"FROM '{p}' WHERE final_n_neg=1 AND {fmax_col} < 0.01"
            ).df()
            if not len(df): continue
            steps = df["total_steps"].values
            ax.hist(steps, bins=30, alpha=0.55, color=color, label=f"{label} (n={len(steps)})",
                    density=True)
        ax.set_title(f"{n} pm")
        ax.set_xlabel("steps to TS")
        ax.set_xlim(0, max(2000, ax.get_xlim()[1]))
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("density")
    axes[3].set_ylabel("density")
    fig.suptitle("Steps to convergence (test, n=287, fmax<0.01 ∧ n_neg=1)", y=1.01)
    fig.tight_layout()
    _save(fig, "fig_test_steps_hist")


def main():
    fig_conv_3thresh()
    fig_dtgrid()
    fig_threshold_curves()
    fig_steps_hist()


if __name__ == "__main__":
    main()
