#!/usr/bin/env python
"""Comprehensive compute analysis: GAD vs Sella step/walltime/Hessian budget.

Investigates the question: GAD takes much more steps per converged TS than Sella.
What does it really cost in wall time, Hessian calls, etc., and what would
equal-compute look like?

Outputs:
  analysis_2026_04_29/compute_summary.csv     # method × noise → cost stats
  analysis_2026_04_29/gad_truncation.csv      # GAD: first-crossing step distribution
  analysis_2026_04_29/sella_truncation.csv    # Sella: first-crossing step distribution
  analysis_2026_04_29/dynamics_walltime.csv   # fmax(t) aligned by walltime
  figures/fig_compute_step_dist.pdf           # converged-step distribution histograms
  figures/fig_compute_wall_per_conv.pdf       # wall-time per converged TS
  figures/fig_compute_dynamics_walltime.pdf   # fmax vs cumulative walltime
  figures/fig_compute_truncation_cdf.pdf      # GAD CDF: at step N, what % converged?
  figures/fig_compute_pareto.pdf              # decision plot: budget vs n_TS
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

OUT_CSV = Path("/lustre06/project/6033559/memoozd/GAD_plus/analysis_2026_04_29")
OUT_FIG = Path("/lustre06/project/6033559/memoozd/GAD_plus/figures")
OUT_CSV.mkdir(exist_ok=True, parents=True)
OUT_FIG.mkdir(exist_ok=True, parents=True)
RUNS = Path("/lustre07/scratch/memoozd/gadplus/runs")

NOISES = [10, 30, 50, 100, 150, 200]

# (label, summary_glob_pattern, traj_glob_pattern_or_None, color, marker)
METHODS = [
    ("GAD dt=0.003 (5k)",
     str(RUNS / "test_dtgrid/gad_dt003_fmax/summary_*.parquet"),
     str(RUNS / "test_dtgrid/gad_dt003_fmax/traj_*{noise}pm*.parquet"),
     palette_color(0), "o"),
    ("GAD dt=0.005 (5k)",
     str(RUNS / "test_dtgrid/gad_dt005_fmax/summary_*.parquet"),
     str(RUNS / "test_dtgrid/gad_dt005_fmax/traj_*{noise}pm*.parquet"),
     palette_color(9), "s"),
    ("GAD dt=0.007 (5k)",
     str(RUNS / "test_dtgrid/gad_dt007_fmax/summary_*.parquet"),
     str(RUNS / "test_dtgrid/gad_dt007_fmax/traj_*{noise}pm*.parquet"),
     palette_color(1), "^"),
    ("Sella cart+Eckart, δ0=0.10 γ=0.40 H/step (2k)",
     str(RUNS / "test_set/sella_carteck_libdef/summary_*.parquet"),
     str(RUNS / "test_sella_trajlog/carteck_libdef/traj_*{noise}pm*.parquet"),
     palette_color(3), "D"),
    ("Sella cart+Eckart, δ0=0.10 γ=0.40 H/step (5k)",
     str(RUNS / "test_sella_extended/carteck_libdef_5k/summary_*.parquet"),
     None,
     palette_color(4), "v"),
    ("Sella cart+Eckart, δ0=0.10 γ=0.40 H/step (10k)",
     str(RUNS / "test_sella_extended/carteck_libdef_10k/summary_*.parquet"),
     None,
     palette_color(5), "P"),
    ("Sella cart+Eckart, δ0=0.048 γ=0 H/step (2k)",
     str(RUNS / "test_set/sella_carteck_default/summary_*.parquet"),
     None,
     palette_color(6), "X"),
    ("Sella internal, δ0=0.048 γ=0 H/step (2k)",
     str(RUNS / "test_set/sella_internal_default/summary_*.parquet"),
     None,
     palette_color(7), "*"),
]


# =================================================================
# 1. Per-cell summary: step counts, wall time, cost-per-converged
# =================================================================
def build_summary() -> pd.DataFrame:
    rows = []
    for label, glob_summary, _, _, _ in METHODS:
        for noise in NOISES:
            try:
                df = duckdb.execute(rf"""
                SELECT
                  COUNT(*) AS n,
                  SUM(CAST(converged AS INT)) AS n_conv,
                  AVG(total_steps) AS avg_total_steps,
                  MEDIAN(total_steps) AS med_total_steps,
                  AVG(CASE WHEN converged THEN converged_step END) AS avg_step_conv,
                  MEDIAN(CASE WHEN converged THEN converged_step END) AS med_step_conv,
                  QUANTILE_CONT(CASE WHEN converged THEN converged_step END, 0.25) AS p25_step_conv,
                  QUANTILE_CONT(CASE WHEN converged THEN converged_step END, 0.75) AS p75_step_conv,
                  QUANTILE_CONT(CASE WHEN converged THEN converged_step END, 0.95) AS p95_step_conv,
                  AVG(wall_time_s) AS avg_wall_s,
                  MEDIAN(wall_time_s) AS med_wall_s,
                  SUM(wall_time_s) AS total_wall_s,
                  MEDIAN(wall_time_s/total_steps)*1000.0 AS med_ms_per_step
                FROM read_parquet('{glob_summary}', filename=true)
                WHERE regexp_extract(filename, '_(\d+)pm', 1)='{noise}' AND total_steps>0
                """).df()
                if len(df) == 0 or df["n"].iloc[0] == 0:
                    continue
                r = df.iloc[0]
                rows.append({
                    "method": label,
                    "noise_pm": noise,
                    "n": int(r["n"]),
                    "n_conv": int(r["n_conv"]),
                    "conv_pct": 100.0 * r["n_conv"] / r["n"],
                    "avg_total_steps": r["avg_total_steps"],
                    "med_total_steps": r["med_total_steps"],
                    "avg_step_conv": r["avg_step_conv"],
                    "med_step_conv": r["med_step_conv"],
                    "p25_step_conv": r["p25_step_conv"],
                    "p75_step_conv": r["p75_step_conv"],
                    "p95_step_conv": r["p95_step_conv"],
                    "avg_wall_s": r["avg_wall_s"],
                    "med_wall_s": r["med_wall_s"],
                    "total_wall_s": r["total_wall_s"],
                    "wall_per_conv_TS_s": (r["total_wall_s"] / r["n_conv"]) if r["n_conv"] > 0 else np.nan,
                    "med_ms_per_step": r["med_ms_per_step"],
                })
            except Exception as e:
                print(f"err {label} {noise}pm: {e}")
                continue
    return pd.DataFrame(rows)


# =================================================================
# 2. GAD per-trajectory truncation: first step where conv was achieved
# =================================================================
def gad_first_conv_step(traj_glob: str, noise: int) -> pd.DataFrame:
    """For each GAD sample at this noise, find smallest step where
    (n_neg=1 AND force_max<0.01) first true. NaN if never."""
    glob_full = traj_glob.replace("{noise}", str(noise))
    df = duckdb.execute(rf"""
    WITH per_step AS (
      SELECT regexp_extract(filename, '_(\d+)\.parquet$', 1) AS sample_idx,
             step, force_max, n_neg
      FROM read_parquet('{glob_full}', filename=true)
    ),
    first_cross AS (
      SELECT sample_idx, MIN(step) AS first_step,
             MIN(CASE WHEN force_max<0.005 THEN step END) AS first_005,
             MIN(CASE WHEN force_max<0.001 THEN step END) AS first_001
      FROM per_step
      WHERE n_neg = 1 AND force_max < 0.01
      GROUP BY sample_idx
    ),
    all_samples AS (SELECT DISTINCT sample_idx FROM per_step)
    SELECT a.sample_idx,
           f.first_step AS step_to_fmax01,
           f.first_005 AS step_to_fmax005,
           f.first_001 AS step_to_fmax001
    FROM all_samples a LEFT JOIN first_cross f USING(sample_idx)
    """).df()
    return df


def build_gad_truncation() -> pd.DataFrame:
    """For each GAD method × noise, what fraction of converged samples
    had first achieved (n_neg=1 ∧ fmax<0.01) by step N? Run for various N."""
    grid_steps = [10, 30, 50, 75, 100, 150, 200, 300, 400, 500, 700, 1000, 1500,
                  2000, 3000, 5000]
    rows = []
    for label, _, traj_glob, _, _ in METHODS:
        if traj_glob is None or "gad" not in label.lower(): continue
        for noise in NOISES:
            try:
                df = gad_first_conv_step(traj_glob, noise)
            except Exception as e:
                print(f"err {label} {noise}pm: {e}"); continue
            n_total = len(df)
            n_ever_conv = df["step_to_fmax01"].notna().sum()
            for s in grid_steps:
                n_by = (df["step_to_fmax01"] <= s).sum()
                rows.append({
                    "method": label, "noise_pm": noise, "step_budget": s,
                    "n_total": n_total, "n_ever_conv": int(n_ever_conv),
                    "n_conv_by_step": int(n_by),
                    "frac_of_ever_conv": (n_by / n_ever_conv) if n_ever_conv > 0 else 0.0,
                    "frac_of_total": n_by / n_total if n_total > 0 else 0.0,
                })
    return pd.DataFrame(rows)


# =================================================================
# 3. Sella per-trajectory truncation: same idea, but based on force_max only
#     (Sella traj parquets don't record n_neg per step). Use force_max<0.01
#     as a proxy — most of these samples have n_neg=1 by then.
# =================================================================
def sella_first_conv_step(traj_glob: str, noise: int) -> pd.DataFrame:
    glob_full = traj_glob.replace("{noise}", str(noise))
    df = duckdb.execute(rf"""
    WITH per_step AS (
      SELECT sample_id, step, force_max
      FROM read_parquet('{glob_full}')
    ),
    first_cross AS (
      SELECT sample_id, MIN(step) AS first_step,
             MIN(CASE WHEN force_max<0.005 THEN step END) AS first_005,
             MIN(CASE WHEN force_max<0.001 THEN step END) AS first_001
      FROM per_step WHERE force_max < 0.01
      GROUP BY sample_id
    ),
    all_samples AS (SELECT DISTINCT sample_id FROM per_step)
    SELECT a.sample_id,
           f.first_step AS step_to_fmax01,
           f.first_005 AS step_to_fmax005,
           f.first_001 AS step_to_fmax001
    FROM all_samples a LEFT JOIN first_cross f ON a.sample_id=f.sample_id
    """).df()
    return df


def build_sella_truncation() -> pd.DataFrame:
    grid_steps = [1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 75, 100, 200, 500, 1000, 2000]
    rows = []
    for label, _, traj_glob, _, _ in METHODS:
        if traj_glob is None or "sella" not in label.lower(): continue
        for noise in NOISES:
            try:
                df = sella_first_conv_step(traj_glob, noise)
            except Exception as e:
                print(f"err {label} {noise}pm: {e}"); continue
            n_total = len(df)
            n_ever = df["step_to_fmax01"].notna().sum()
            for s in grid_steps:
                n_by = (df["step_to_fmax01"] <= s).sum()
                rows.append({
                    "method": label, "noise_pm": noise, "step_budget": s,
                    "n_total": n_total, "n_ever_conv": int(n_ever),
                    "n_conv_by_step": int(n_by),
                    "frac_of_ever_conv": (n_by / n_ever) if n_ever > 0 else 0.0,
                    "frac_of_total": n_by / n_total if n_total > 0 else 0.0,
                })
    return pd.DataFrame(rows)


# =================================================================
# 4. Dynamics aligned by wall-time (median + IQR over samples, by walltime bin)
# =================================================================
def dynamics_by_walltime(method_label: str, traj_glob: str, noise: int,
                         wall_per_step_s: float, n_bins: int = 60) -> pd.DataFrame:
    """For each method × noise: sample fmax along the wall-time axis at log-spaced
    time bins. Estimates wall_time = step * wall_per_step_s for Sella (where it's
    not stored in the traj). Uses traj's wall_time_s for GAD."""
    glob_full = traj_glob.replace("{noise}", str(noise))
    is_gad = "gad" in method_label.lower()

    # For GAD: wall_time_s is actual cumulative wall-clock (from per-step traj).
    # For Sella: estimate from step * wall_per_step.
    if is_gad:
        time_expr = "wall_time_s"
    else:
        time_expr = f"CAST(step AS DOUBLE)*{wall_per_step_s}"

    # Bin walltime in log space, take median fmax per bin per sample then median.
    # First compute per-sample fmax interpolated at each bin time.
    bin_edges_s = np.geomspace(0.1, 400, n_bins + 1)

    sample_id_expr = "sample_idx" if is_gad else "sample_id"
    sample_id_select = "regexp_extract(filename, '_(\\d+)\\.parquet$', 1) AS sample_idx" if is_gad else "sample_id"

    df = duckdb.execute(rf"""
    SELECT {sample_id_select}, step, force_max, {time_expr} AS wall_s
    FROM read_parquet('{glob_full}'{', filename=true' if is_gad else ''})
    """).df()
    if df.empty: return pd.DataFrame()
    df = df.dropna(subset=["force_max", "wall_s"])
    df = df[df["wall_s"] > 0].copy()
    df["log_wall"] = np.log10(df["wall_s"])

    # For each sample: we use force_max trajectory; sample at each bin center.
    # Implementation: assign each step to its log-wall bin; take median fmax in bin.
    log_edges = np.log10(bin_edges_s)
    centers = 0.5 * (log_edges[:-1] + log_edges[1:])
    df["bin"] = np.digitize(df["log_wall"], log_edges) - 1
    df = df[(df["bin"] >= 0) & (df["bin"] < n_bins)].copy()

    # For each sample × bin → minimum fmax up to bin (since fmax should monotone-ish)
    g = df.groupby(["sample_id" if not is_gad else "sample_idx", "bin"])["force_max"].min().reset_index()
    by_bin = g.groupby("bin")["force_max"]
    out = pd.DataFrame({
        "wall_s_center": 10**centers[g["bin"].unique().min():g["bin"].unique().max()+1] if False else 10**centers,
        "bin": np.arange(n_bins),
    })
    out["fmax_med"] = out["bin"].map(by_bin.median())
    out["fmax_p25"] = out["bin"].map(by_bin.quantile(0.25))
    out["fmax_p75"] = out["bin"].map(by_bin.quantile(0.75))
    out["n_samples_in_bin"] = out["bin"].map(by_bin.count())
    out["method"] = method_label
    out["noise_pm"] = noise
    return out


def build_dynamics_walltime(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Run dynamics_by_walltime for all (method,noise) where we have traj data.
    Use summary_df for ms_per_step lookup (Sella sims wall from step counts)."""
    rows = []
    for label, _, traj_glob, _, _ in METHODS:
        if traj_glob is None: continue
        for noise in NOISES:
            sub = summary_df[(summary_df["method"] == label) & (summary_df["noise_pm"] == noise)]
            if sub.empty: continue
            ms_per_step = sub["med_ms_per_step"].iloc[0]
            try:
                d = dynamics_by_walltime(label, traj_glob, noise, ms_per_step / 1000.0)
                if not d.empty:
                    rows.append(d)
            except Exception as e:
                print(f"err walltime {label} {noise}pm: {e}")
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


# =================================================================
# Plots
# =================================================================
def plot_step_dist(summary_df: pd.DataFrame):
    """Box-style: P25/median/P75/P95 of converged-step per method × noise."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharey=True)
    for ax, noise in zip(axes.flatten(), NOISES):
        sub = summary_df[summary_df["noise_pm"] == noise].copy()
        sub = sub[sub["n_conv"] > 0]
        ys = []
        labels = []
        colors = []
        for label, _, _, color, _ in METHODS:
            row = sub[sub["method"] == label]
            if row.empty: continue
            r = row.iloc[0]
            ys.append([r["p25_step_conv"], r["med_step_conv"], r["p75_step_conv"], r["p95_step_conv"]])
            labels.append(label)
            colors.append(color)
        x = np.arange(len(ys))
        for xi, (yarr, c) in enumerate(zip(ys, colors)):
            ax.plot([xi, xi], [yarr[0], yarr[3]], color=c, lw=2.0, alpha=0.4)
            ax.plot([xi, xi], [yarr[0], yarr[2]], color=c, lw=4.0, alpha=0.7)
            ax.plot(xi, yarr[1], "o", color=c, markersize=10, markeredgecolor=palette_color(7), markeredgewidth=0.7)
            ax.text(xi, yarr[3]*1.1 if yarr[3]>0 else 1, f"{int(yarr[1])}",
                    ha="center", fontsize=7)
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels([l.replace(" ", "\n", 1).replace("(", "\n(") for l in labels],
                           fontsize=7, rotation=0)
        ax.set_title(f"{noise}pm noise", fontsize=10)
        ax.grid(alpha=0.3, axis="y")
        if noise in (10, 100):
            ax.set_ylabel("steps to convergence (log)")
    fig.suptitle("Steps-to-convergence distribution by method × noise (P25/median/P75 thick, +P95 thin tail)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT_FIG / f"fig_compute_step_dist.{ext}", bbox_inches="tight", dpi=140)
    plt.close(fig)
    print("wrote fig_compute_step_dist")


def plot_wall_per_conv(summary_df: pd.DataFrame):
    """Per noise: wall-time-per-converged-TS bar chart, log scale."""
    fig, ax = plt.subplots(figsize=(13, 6))
    n_methods = len(METHODS)
    bar_w = 0.13
    method_offsets = np.linspace(-bar_w*n_methods/2 + bar_w/2,
                                  bar_w*n_methods/2 - bar_w/2, n_methods)
    for j, (label, _, _, color, _) in enumerate(METHODS):
        for i, noise in enumerate(NOISES):
            row = summary_df[(summary_df["method"] == label) &
                              (summary_df["noise_pm"] == noise)]
            if row.empty: continue
            v = row["wall_per_conv_TS_s"].iloc[0]
            if not np.isfinite(v) or v <= 0: continue
            ax.bar(i + method_offsets[j], v, bar_w,
                   color=color, label=label if i == 0 else None,
                   edgecolor=palette_color(7), linewidth=0.4)
    ax.set_xticks(range(len(NOISES)))
    ax.set_xticklabels([f"{n}pm" for n in NOISES])
    ax.set_yscale("log")
    ax.set_ylabel("wall-time per converged TS (sec, log)")
    ax.set_xlabel("TS-noise (pm)")
    ax.set_title("Cost per converged TS:  total wall / n_converged   (n=287 attempts per cell)",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, ncol=2, loc="upper left", framealpha=0.95)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT_FIG / f"fig_compute_wall_per_conv.{ext}", bbox_inches="tight", dpi=140)
    plt.close(fig)
    print("wrote fig_compute_wall_per_conv")


def plot_dynamics_walltime(dyn_df: pd.DataFrame):
    """fmax(t) median + IQR aligned by wall-time. Two panels: 30pm + 200pm."""
    if dyn_df.empty: return
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    chosen_noises = [30, 100, 200]
    for ax, noise in zip(axes, chosen_noises):
        for label, _, _, color, _ in METHODS:
            sub = dyn_df[(dyn_df["method"] == label) & (dyn_df["noise_pm"] == noise)]
            if sub.empty: continue
            sub = sub.sort_values("wall_s_center")
            sub = sub[sub["fmax_med"].notna()]
            if sub.empty: continue
            ax.plot(sub["wall_s_center"], sub["fmax_med"], color=color, label=label, lw=1.7)
            ax.fill_between(sub["wall_s_center"],
                            sub["fmax_p25"], sub["fmax_p75"],
                            color=color, alpha=0.15)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.axhline(0.05, color=palette_color(7), ls=":", alpha=0.5)
        ax.axhline(0.01, color=palette_color(7), ls="--", alpha=0.7)
        ax.axhline(0.005, color=palette_color(7), ls=":", alpha=0.5)
        ax.axhline(0.001, color=palette_color(7), ls=":", alpha=0.5)
        ax.text(0.11, 0.011, "fmax=0.01 (default criterion)", fontsize=7)
        ax.set_xlabel("cumulative wall-time (s, log)")
        ax.set_ylabel(r"$f_{\max}$ median (eV/Å, log)")
        ax.set_title(f"{noise}pm noise", fontsize=11)
        ax.grid(alpha=0.3)
        if noise == 30:
            ax.legend(fontsize=8, loc="upper right", framealpha=0.95, ncol=2)
    fig.suptitle("$f_{\\max}$ trajectory aligned by wall-time (median across samples; shaded = IQR)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT_FIG / f"fig_compute_dynamics_walltime.{ext}", bbox_inches="tight", dpi=140)
    plt.close(fig)
    print("wrote fig_compute_dynamics_walltime")


def plot_truncation_cdf(gad_trunc: pd.DataFrame, sella_trunc: pd.DataFrame):
    """Per method × noise: CDF of 'fraction of ever-converged samples that
    had reached convergence by step N' as a function of N."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True, sharey=True)
    combined = pd.concat([gad_trunc, sella_trunc], ignore_index=True)
    for ax, noise in zip(axes.flatten(), NOISES):
        for label, _, _, color, _ in METHODS:
            sub = combined[(combined["method"] == label) & (combined["noise_pm"] == noise)]
            if sub.empty: continue
            sub = sub.sort_values("step_budget")
            ax.plot(sub["step_budget"], 100*sub["frac_of_total"],
                    color=color, label=label, lw=1.8, marker="o", markersize=4)
        ax.set_xscale("log")
        ax.set_xlabel("step budget cap")
        ax.set_ylabel("% of n=287 converged by step N")
        ax.set_title(f"{noise}pm noise", fontsize=10)
        ax.grid(alpha=0.3)
        ax.set_ylim(0, 105)
        if noise == 10:
            ax.legend(fontsize=7, loc="lower right", framealpha=0.95, ncol=2)
    fig.suptitle("Step-budget vs convergence fraction\n"
                 "(fraction of all n=287 samples that achieved $n_{neg}=1$ ∧ $f_{\\max}<0.01$ by step N)",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT_FIG / f"fig_compute_truncation_cdf.{ext}", bbox_inches="tight", dpi=140)
    plt.close(fig)
    print("wrote fig_compute_truncation_cdf")


def plot_pareto(summary_df: pd.DataFrame):
    """Per noise: scatter of (total_wall_s spent, n_converged_TS).
    Each method = one point per noise. Pareto frontier."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharey=False)
    for ax, noise in zip(axes.flatten(), NOISES):
        sub = summary_df[summary_df["noise_pm"] == noise]
        for label, _, _, color, marker in METHODS:
            row = sub[sub["method"] == label]
            if row.empty: continue
            x = row["total_wall_s"].iloc[0]
            y = row["n_conv"].iloc[0]
            ax.scatter(x, y, color=color, label=label, s=80, marker=marker,
                       edgecolor=palette_color(7), linewidth=0.6)
            ax.annotate(label, (x, y), fontsize=6, alpha=0.7,
                        xytext=(3, 3), textcoords="offset points")
        ax.set_xscale("log")
        ax.set_xlabel("total wall-time on n=287 attempts (s, log)")
        ax.set_ylabel("n converged ($n_{neg}=1$ ∧ $f_{\\max}<0.01$)")
        ax.set_title(f"{noise}pm noise", fontsize=10)
        ax.grid(alpha=0.3)
    fig.suptitle("Compute–accuracy frontier:  cost vs n_converged",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT_FIG / f"fig_compute_pareto.{ext}", bbox_inches="tight", dpi=140)
    plt.close(fig)
    print("wrote fig_compute_pareto")


# =================================================================
# Main
# =================================================================
def main():
    print("=== Building summary table ===")
    summary = build_summary()
    summary.to_csv(OUT_CSV / "compute_summary.csv", index=False)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    print("\n=== Building GAD trajectory truncation table ===")
    gad_trunc = build_gad_truncation()
    gad_trunc.to_csv(OUT_CSV / "gad_truncation.csv", index=False)
    print(gad_trunc.head(20).to_string(index=False))

    print("\n=== Building Sella trajectory truncation table ===")
    sella_trunc = build_sella_truncation()
    sella_trunc.to_csv(OUT_CSV / "sella_truncation.csv", index=False)
    print(sella_trunc.head(20).to_string(index=False))

    print("\n=== Building wall-time-aligned dynamics ===")
    dyn = build_dynamics_walltime(summary)
    dyn.to_csv(OUT_CSV / "dynamics_walltime.csv", index=False)
    print(f"dynamics_walltime: {len(dyn)} rows")

    print("\n=== Plotting ===")
    plot_step_dist(summary)
    plot_wall_per_conv(summary)
    plot_dynamics_walltime(dyn)
    plot_truncation_cdf(gad_trunc, sella_trunc)
    plot_pareto(summary)

    print("\n=== DONE ===")


if __name__ == "__main__":
    main()
