#!/usr/bin/env python
"""Trajectory dynamics: fmax(t) and force_norm(t) curves per method/noise.

For each (method, noise) cell, samples up to N trajectory parquets and
computes:
  - median fmax(step) and force_norm(step) curves
  - 25/75 quantile bands
  - first-crossing step for thresholds {0.05, 0.01, 0.005, 0.001}
  - plateau detection: step at which dlog(fmax)/dstep < 1e-3 over a 100-step window

Output:
  analysis_2026_04_29/dynamics_curves.csv     (one row per method × noise × step)
  analysis_2026_04_29/dynamics_crossings.csv  (one row per method × noise × sample × threshold)
  figures/fig_dynamics_fmax.pdf               (panels per noise level, lines per method)
  figures/fig_dynamics_fnorm.pdf

Methods covered (where trajectory parquets exist):
  GAD dt=0.003, dt=0.005, dt=0.007 (test)
  GAD adaptive_dt (test)
  GAD low-dt: dt=1e-3, 5e-4, 1e-4 (test, partial)
  Sella libdef (train, 100pm only)
"""
from __future__ import annotations

import os
from pathlib import Path
from glob import glob

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
FIG.mkdir(exist_ok=True, parents=True)

NOISES = [10, 30, 50, 100, 150, 200]
THRESHOLDS = [0.05, 0.01, 0.005, 0.001]
SAMPLES_PER_CELL = 30          # cap to keep this fast
STEP_CAP = 5000                # max step to retain for visualization

# methods: label -> (run_dir, color, linestyle, n_samples_to_use)
# Method labels deliberately have NO commas so the CSV is duckdb-queryable
# without quote-detection edge cases.
METHODS = {
    "gad_dt003_5k":  (BASE / "test_dtgrid/gad_dt003_fmax",  "#1f77b4", "-",  SAMPLES_PER_CELL),
    "gad_dt005_5k":  (BASE / "test_dtgrid/gad_dt005_fmax",  "#2ca02c", "-",  SAMPLES_PER_CELL),
    "gad_dt007_5k":  (BASE / "test_dtgrid/gad_dt007_fmax",  "#ff7f0e", "-",  SAMPLES_PER_CELL),
    "gad_adaptive_dt": (BASE / "test_set/gad_adaptive_dt",  "#d62728", "--", SAMPLES_PER_CELL),
    "gad_dt001_20k": (BASE / "test_lowdt/gad_dt001_fmax",   "#9467bd", ":",  SAMPLES_PER_CELL),
}

# Display names (used only in legend, not in CSV).
DISPLAY = {
    "gad_dt003_5k": "GAD dt=0.003 (5k)",
    "gad_dt005_5k": "GAD dt=0.005 (5k)",
    "gad_dt007_5k": "GAD dt=0.007 (5k)",
    "gad_adaptive_dt": "GAD adaptive dt",
    "gad_dt001_20k": "GAD dt=1e-3 (20k)",
    "sella_libdef_train_trajlog": "Sella libdef (train trajlog)",
    "sella_libdef_test_trajlog":  "Sella libdef (test trajlog)",
}

# Sella trajectory data: train (1 noise) + test (6 noise once 60148863 lands).
SELLA_TRAJLOG = {
    "sella_libdef_train_trajlog":  (BASE / "sella_trajlog/carteck_libdef",                "#17becf", "-",  SAMPLES_PER_CELL),
    "sella_libdef_test_trajlog":   (BASE / "test_sella_trajlog/carteck_libdef",           "#0080A0", "-",  SAMPLES_PER_CELL),
}


def load_traj(traj_path: Path, max_step: int = STEP_CAP) -> pd.DataFrame:
    """Returns df with cols [step, force_max, force_norm]."""
    cols = set(duckdb.execute(f"DESCRIBE SELECT * FROM '{traj_path}'").df()["column_name"])
    if "force_max" not in cols and "fmax" in cols:
        fmax_col = "fmax"
    else:
        fmax_col = "force_max"
    fnorm_col = "force_norm"
    df = duckdb.execute(
        f"SELECT step, {fmax_col} AS force_max, {fnorm_col} AS force_norm FROM '{traj_path}' "
        f"WHERE step < {max_step} ORDER BY step"
    ).df()
    return df


def collect_curves(method_dir: Path, noise_pm: int, n_samples: int):
    """Return per-sample arrays (rows = samples, cols = step)."""
    pattern = str(method_dir / f"traj_*_{noise_pm}pm_*.parquet")
    files = sorted(glob(pattern))[:n_samples]
    if not files:
        return None
    fmax_all, fnorm_all = [], []
    max_step = 0
    for f in files:
        df = load_traj(Path(f))
        if len(df) == 0: continue
        fmax_all.append((df["step"].values, df["force_max"].values))
        fnorm_all.append((df["step"].values, df["force_norm"].values))
        max_step = max(max_step, df["step"].max())
    if not fmax_all: return None

    # Pad onto common step grid (forward-fill last value)
    step_grid = np.arange(0, max_step + 1)
    fmax_mat = np.full((len(fmax_all), len(step_grid)), np.nan)
    fnorm_mat = np.full((len(fnorm_all), len(step_grid)), np.nan)
    for i, ((s, fm), (_, fn)) in enumerate(zip(fmax_all, fnorm_all)):
        # forward-fill: the trajectory ends at last step; values stay constant after
        last_step = s.max()
        fmax_mat[i, :last_step + 1] = np.interp(step_grid[:last_step + 1], s, fm)
        fmax_mat[i, last_step + 1:] = fm[-1]
        fnorm_mat[i, :last_step + 1] = np.interp(step_grid[:last_step + 1], s, fn)
        fnorm_mat[i, last_step + 1:] = fn[-1]

    return step_grid, fmax_mat, fnorm_mat


def crossing_step(fmax: np.ndarray, threshold: float) -> int | None:
    """First step where fmax < threshold."""
    below = np.where(fmax < threshold)[0]
    return int(below[0]) if len(below) else None


def detect_plateau(fmax: np.ndarray, window: int = 100, slope_thresh: float = 1e-3) -> int | None:
    """First step where the rolling log-slope drops below slope_thresh."""
    if len(fmax) < window + 10: return None
    log_f = np.log10(np.clip(fmax, 1e-8, None))
    # rolling slope = (log_f[t+w] - log_f[t]) / w
    slopes = np.abs(log_f[window:] - log_f[:-window]) / window
    plateau_starts = np.where(slopes < slope_thresh)[0]
    return int(plateau_starts[0]) if len(plateau_starts) else None


def main():
    rows_curve = []
    rows_cross = []

    all_methods = {**METHODS, **SELLA_TRAJLOG}

    for label, (mdir, color, ls, n_s) in all_methods.items():
        for n in NOISES:
            res = collect_curves(mdir, n, n_s)
            if res is None:
                print(f"skip {label} {n}pm (no traj)")
                continue
            step_grid, fmax_mat, fnorm_mat = res
            print(f"{label} {n}pm: {fmax_mat.shape[0]} samples × {fmax_mat.shape[1]} steps")

            # Per-sample crossings + plateau detection
            for i in range(fmax_mat.shape[0]):
                fm = fmax_mat[i]
                pl = detect_plateau(fm)
                row = {"method": label, "noise_pm": n, "sample_idx": i,
                       "plateau_step": pl, "final_fmax": float(fm[~np.isnan(fm)][-1])}
                for t in THRESHOLDS:
                    row[f"cross_fmax_{t:.4g}"] = crossing_step(fm, t)
                    row[f"cross_fnorm_{t:.4g}"] = crossing_step(fnorm_mat[i], t)
                rows_cross.append(row)

            # Median + quantile curves on common step grid
            f50 = np.nanmedian(fmax_mat, axis=0)
            f25 = np.nanpercentile(fmax_mat, 25, axis=0)
            f75 = np.nanpercentile(fmax_mat, 75, axis=0)
            n50 = np.nanmedian(fnorm_mat, axis=0)
            for j, s in enumerate(step_grid):
                rows_curve.append({"method": label, "noise_pm": n, "step": int(s),
                                   "fmax_p25": f25[j], "fmax_p50": f50[j], "fmax_p75": f75[j],
                                   "fnorm_p50": n50[j]})

    if not rows_curve:
        print("No data — abort.")
        return

    df_curve = pd.DataFrame(rows_curve)
    df_cross = pd.DataFrame(rows_cross)
    df_curve.to_csv(OUT / "dynamics_curves.csv", index=False)
    df_cross.to_csv(OUT / "dynamics_crossings.csv", index=False)
    print(f"wrote {OUT/'dynamics_curves.csv'} ({len(df_curve)} rows)")
    print(f"wrote {OUT/'dynamics_crossings.csv'} ({len(df_cross)} rows)")

    # =========== Figure: fmax(step) per noise level ===========
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=True)
    axes = axes.flatten()
    for ax, n in zip(axes, NOISES):
        for label, (_, color, ls, _) in all_methods.items():
            sub = df_curve[(df_curve["method"] == label) & (df_curve["noise_pm"] == n)]
            if not len(sub): continue
            ax.plot(sub["step"], sub["fmax_p50"], color=color, linestyle=ls,
                    linewidth=1.8, label=DISPLAY.get(label, label), alpha=0.9)
            ax.fill_between(sub["step"], sub["fmax_p25"], sub["fmax_p75"],
                            color=color, alpha=0.12)
        for t in THRESHOLDS:
            ax.axhline(t, color="gray", linestyle=":", alpha=0.4, linewidth=0.8)
            ax.text(ax.get_xlim()[1] * 0.99, t * 1.05, f"{t}", color="gray",
                    fontsize=7, ha="right", va="bottom")
        ax.set_yscale("log")
        ax.set_xlabel("step")
        ax.set_title(f"{n} pm")
        ax.set_ylim(1e-4, 20)
        ax.grid(alpha=0.25, which="both")
    axes[0].set_ylabel("$f_{\\max}$ (eV/Å)")
    axes[3].set_ylabel("$f_{\\max}$ (eV/Å)")
    axes[2].legend(loc="upper right", fontsize=7, framealpha=0.85)
    fig.suptitle("Trajectory dynamics: $f_{\\max}$ vs step "
                 "(median + IQR band, n=30 samples per cell)", y=1.005)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"fig_dynamics_fmax.{ext}", bbox_inches="tight", dpi=140)
    plt.close(fig)
    print(f"wrote fig_dynamics_fmax.{{pdf,png}}")

    # =========== Figure: force_norm(step) ===========
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=True)
    axes = axes.flatten()
    for ax, n in zip(axes, NOISES):
        for label, (_, color, ls, _) in all_methods.items():
            sub = df_curve[(df_curve["method"] == label) & (df_curve["noise_pm"] == n)]
            if not len(sub): continue
            ax.plot(sub["step"], sub["fnorm_p50"], color=color, linestyle=ls,
                    linewidth=1.8, label=DISPLAY.get(label, label), alpha=0.9)
        for t in THRESHOLDS:
            ax.axhline(t, color="gray", linestyle=":", alpha=0.4, linewidth=0.8)
        ax.set_yscale("log")
        ax.set_xlabel("step")
        ax.set_title(f"{n} pm")
        ax.set_ylim(1e-4, 20)
        ax.grid(alpha=0.25, which="both")
    axes[0].set_ylabel("$\\|F\\|_{\\text{mean}}$")
    axes[3].set_ylabel("$\\|F\\|_{\\text{mean}}$")
    axes[2].legend(loc="upper right", fontsize=7, framealpha=0.85)
    fig.suptitle("Trajectory dynamics: $\\|F\\|_{\\text{mean}}$ vs step "
                 "(median, n=30 samples per cell)", y=1.005)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"fig_dynamics_fnorm.{ext}", bbox_inches="tight", dpi=140)
    plt.close(fig)
    print(f"wrote fig_dynamics_fnorm.{{pdf,png}}")

    # =========== Crossing-step summary table ===========
    print("\n=== Median first-crossing step (fmax) ===")
    for t in THRESHOLDS:
        col = f"cross_fmax_{t:.4g}"
        print(f"\n  threshold {t}:")
        print(f"  {'method':<28} {'10':>5} {'30':>5} {'50':>5} {'100':>5} {'150':>5} {'200':>5}")
        for m in all_methods:
            row = [f"  {m:<28}"]
            for n in NOISES:
                sub = df_cross[(df_cross["method"] == m) & (df_cross["noise_pm"] == n)]
                if not len(sub):
                    row.append("   --")
                    continue
                vals = sub[col].dropna()
                if not len(vals):
                    row.append("   --")
                else:
                    med = int(vals.median())
                    n_cross = len(vals)
                    n_total = len(sub)
                    row.append(f"{med:>4}/{n_cross}/{n_total}")
            print(" ".join(row))

    # Plateau-step distribution
    print("\n=== Median plateau-step (where dlog(fmax)/dstep < 1e-3) ===")
    print(f"  {'method':<28} {'10':>7} {'30':>7} {'50':>7} {'100':>7} {'150':>7} {'200':>7}")
    for m in all_methods:
        row = [f"  {m:<28}"]
        for n in NOISES:
            sub = df_cross[(df_cross["method"] == m) & (df_cross["noise_pm"] == n)]
            if not len(sub):
                row.append("     --")
                continue
            vals = sub["plateau_step"].dropna()
            if not len(vals):
                row.append("   none")
            else:
                med = int(vals.median())
                row.append(f"{med:>7}")
        print(" ".join(row))


if __name__ == "__main__":
    main()
