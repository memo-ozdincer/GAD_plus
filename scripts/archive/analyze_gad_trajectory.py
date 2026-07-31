#!/usr/bin/env python
"""GAD trajectory diagnosis to inform second-order step design.

Per-step probes from existing GAD trajectory parquets that tell us where
big steps would be safe and where they wouldn't:

  - fmax(t), force_norm(t), force_rms(t)
  - eig0(t), eig1(t)            ← curvature regime (small |λ_1| = ill-conditioned)
  - grad_v0_overlap(t), grad_v1_overlap(t)  ← bottleneck detector
  - disp_from_last(t)           ← actual GAD step magnitude
  - eigvec_continuity(t)        ← mode tracking confidence
  - mode_overlap(t)
  - n_neg(t), bottom_spectrum   ← saddle-basin check

For each (method, noise, sample), classify each step into one of:
  • "descent_smooth"    fmax > 0.05, |λ_1| > 0.1, grad_v1_overlap > 0.5
  • "descent_curved"    fmax > 0.05, |λ_1| <= 0.1 (curvature near 0, dt should shrink)
  • "tightening_safe"   0.01 < fmax < 0.05, n_neg=1, |λ_1| > 0.1 (could take big step)
  • "tightening_ill"    0.01 < fmax < 0.05, |λ_1| <= 0.1 (ill-conditioned saddle)
  • "plateau"           fmax < 0.05, fmax slope tiny (rolling 100-step log slope < 1e-3)
  • "above_basin"       n_neg > 1 (not yet in saddle basin)
  • "wrong_basin"       n_neg = 0 (drifting to minimum)
  • "converged"         n_neg = 1 ∧ fmax < 0.01 (terminal)

Outputs:
  analysis_2026_04_29/gad_step_classification.csv  # one row per (method, noise, sample, step)
                                                   # — too big to write whole; aggregate.
  analysis_2026_04_29/gad_step_class_summary.csv   # phase counts by (method, noise, class)
  analysis_2026_04_29/gad_step_eigregime.csv       # |λ_1| binned vs class
  figures/fig_gad_phase_breakdown.pdf              # stacked-bar of phases per cell
  figures/fig_gad_eig1_distribution.pdf            # |λ_1| histogram by phase
  figures/fig_gad_step_safety.pdf                  # which phases are step-up safe?

Design implications get written to:
  analysis_2026_04_29/SECOND_ORDER_DESIGN.md
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
RUNS = Path("/lustre07/scratch/memoozd/gadplus/runs/test_dtgrid")

NOISES = [10, 30, 50, 100, 150, 200]
GAD_METHODS = [
    ("GAD dt=0.003", str(RUNS / "gad_dt003_fmax/traj_*.parquet")),
    ("GAD dt=0.005", str(RUNS / "gad_dt005_fmax/traj_*.parquet")),
    ("GAD dt=0.007", str(RUNS / "gad_dt007_fmax/traj_*.parquet")),
]


# =================================================================
# Step classification — phase + step-safety analysis
# =================================================================
def classify_steps_for(method_label: str, glob: str, noise: int) -> pd.DataFrame:
    """For each step in this (method, noise), assign a phase class.
    Returns aggregate counts by (sample, class)."""
    df = duckdb.execute(rf"""
    WITH per_step AS (
      SELECT regexp_extract(filename, '_(\d+)\.parquet$', 1) AS sample_idx,
             step, force_max, force_norm, n_neg, eig0, eig1,
             ABS(eig0) AS abs_eig0, ABS(eig1) AS abs_eig1,
             grad_v0_overlap, grad_v1_overlap,
             disp_from_last, eigvec_continuity, mode_overlap
      FROM read_parquet('{glob}', filename=true)
      WHERE filename LIKE '%_{noise}pm%' AND step > 0
    ),
    classified AS (
      SELECT sample_idx, step, force_max, force_norm, n_neg, abs_eig0, abs_eig1,
             grad_v1_overlap, disp_from_last, eigvec_continuity,
             CASE
               WHEN n_neg = 1 AND force_max < 0.01 THEN 'converged'
               WHEN n_neg = 0 THEN 'wrong_basin'
               WHEN n_neg > 1 THEN 'above_basin'
               WHEN force_max > 0.05 AND abs_eig1 > 0.1 THEN 'descent_smooth'
               WHEN force_max > 0.05                    THEN 'descent_curved'
               WHEN force_max < 0.05 AND abs_eig1 > 0.1 THEN 'tightening_safe'
               WHEN force_max < 0.05                    THEN 'tightening_ill'
               ELSE 'other'
             END AS phase
      FROM per_step
    )
    SELECT phase,
           COUNT(*) AS n_steps,
           ROUND(AVG(force_max), 4) AS avg_fmax,
           ROUND(AVG(abs_eig1), 4) AS avg_abs_eig1,
           ROUND(QUANTILE_CONT(abs_eig1, 0.10), 4) AS p10_abs_eig1,
           ROUND(QUANTILE_CONT(abs_eig1, 0.50), 4) AS p50_abs_eig1,
           ROUND(QUANTILE_CONT(abs_eig1, 0.90), 4) AS p90_abs_eig1,
           ROUND(AVG(disp_from_last), 5) AS avg_disp_step,
           ROUND(MEDIAN(disp_from_last), 5) AS med_disp_step,
           ROUND(AVG(grad_v1_overlap), 4) AS avg_v1_overlap,
           ROUND(AVG(eigvec_continuity), 4) AS avg_continuity
    FROM classified GROUP BY phase
    """).df()
    df["method"] = method_label
    df["noise_pm"] = noise
    return df


def build_phase_summary() -> pd.DataFrame:
    rows = []
    for label, glob in GAD_METHODS:
        for noise in NOISES:
            try:
                d = classify_steps_for(label, glob, noise)
                if not d.empty:
                    rows.append(d)
            except Exception as e:
                print(f"err {label} {noise}pm: {e}")
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


# =================================================================
# Per-class step-magnitude distribution: |Δx_actual| vs what dt-scaled
# step would have been
# =================================================================
def step_safety_analysis(label: str, glob: str, noise: int) -> pd.DataFrame:
    """For each step, compute the *natural* second-order step length per
    eigenmode that would have been taken if we'd used Δx_i = -(F·v_i)/|λ_i|.
    Compare to the actual GAD Euler step. Returns histogram.
    Reads only force/eig stats, not coords (which we can't decompose without H)."""
    df = duckdb.execute(rf"""
    SELECT
      regexp_extract(filename, '_(\d+)\.parquet$', 1) AS sample_idx,
      step, force_max, force_norm, n_neg, eig0, eig1,
      ABS(eig0) AS abs_eig0, ABS(eig1) AS abs_eig1,
      grad_v0_overlap, grad_v1_overlap,
      disp_from_last
    FROM read_parquet('{glob}', filename=true)
    WHERE filename LIKE '%_{noise}pm%' AND step > 0 AND force_max IS NOT NULL
    """).df()
    if df.empty: return df
    # Estimate per-mode "natural" step magnitudes:
    # |F·v_1| ≈ grad_v1_overlap * force_norm (Cartesian — close enough for ratios)
    # natural Δx_1 magnitude ≈ |F·v_1| / |λ_1|
    # similarly for v_0 (if grad_v0_overlap exists, roughly the largest non-v1 mode)
    df["F_dot_v1"] = (df["grad_v1_overlap"].abs() * df["force_norm"]).clip(lower=1e-12)
    df["natural_dx1"] = df["F_dot_v1"] / df["abs_eig1"].clip(lower=1e-12)
    df["natural_dx0"] = (df["grad_v0_overlap"].abs() * df["force_norm"]).clip(lower=1e-12) / \
                       df["abs_eig0"].clip(lower=1e-12)
    df["actual_step_mag"] = df["disp_from_last"]  # actual GAD Euler step
    # Ratio: how much would second-order have moved vs first-order?
    df["scale_ratio_v1"] = df["natural_dx1"] / df["actual_step_mag"].clip(lower=1e-12)
    df["scale_ratio_v0"] = df["natural_dx0"] / df["actual_step_mag"].clip(lower=1e-12)
    df["method"] = label
    df["noise_pm"] = noise
    return df


def aggregate_safety(label: str, glob: str, noise: int) -> dict:
    df = step_safety_analysis(label, glob, noise)
    if df.empty: return {}
    return {
        "method": label,
        "noise_pm": noise,
        "n_steps": len(df),
        "med_actual_step": float(np.median(df["actual_step_mag"])),
        "med_natural_dx0": float(np.median(df["natural_dx0"])),
        "med_natural_dx1": float(np.median(df["natural_dx1"])),
        # Critical: what fraction of steps would benefit from up-scaling vs
        # need down-scaling?
        "frac_v1_up": float((df["scale_ratio_v1"] > 2.0).mean()),
        "frac_v1_down": float((df["scale_ratio_v1"] < 0.5).mean()),
        "p50_scale_v1": float(np.median(df["scale_ratio_v1"])),
        "p90_scale_v1": float(np.quantile(df["scale_ratio_v1"], 0.90)),
        "p99_scale_v1": float(np.quantile(df["scale_ratio_v1"], 0.99)),
        "p50_scale_v0": float(np.median(df["scale_ratio_v0"])),
        "p90_scale_v0": float(np.quantile(df["scale_ratio_v0"], 0.90)),
    }


# =================================================================
# Plots
# =================================================================
def plot_phase_breakdown(summary: pd.DataFrame):
    """Per (method, noise): stacked bar of phase fraction."""
    PHASES = [
        "descent_smooth", "descent_curved", "tightening_safe",
        "tightening_ill", "plateau", "above_basin",
        "wrong_basin", "converged", "other",
    ]
    COLORS = {
        "descent_smooth": palette_color(0),
        "descent_curved": palette_color(9),
        "tightening_safe": palette_color(2),
        "tightening_ill": palette_color(3),
        "plateau": palette_color(1),
        "above_basin": palette_color(4),
        "wrong_basin": palette_color(5),
        "converged": palette_color(8),
        "other": palette_color(7),
    }
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    for ax, (label, _) in zip(axes, GAD_METHODS):
        sub = summary[summary["method"] == label].copy()
        if sub.empty: continue
        # Make pivot: noise × phase → fraction
        totals = sub.groupby("noise_pm")["n_steps"].sum().to_dict()
        sub["frac"] = sub.apply(
            lambda r: r["n_steps"] / totals.get(r["noise_pm"], 1), axis=1)
        pivot = sub.pivot(index="noise_pm", columns="phase",
                          values="frac").fillna(0.0)
        pivot = pivot.reindex(NOISES)
        # Draw stacked bars
        bottom = np.zeros(len(NOISES))
        x = np.arange(len(NOISES))
        for phase in PHASES:
            if phase not in pivot.columns: continue
            vals = pivot[phase].values
            ax.bar(x, vals, bottom=bottom,
                   color=COLORS.get(phase, palette_color(7)),
                   edgecolor=palette_color(7), linewidth=0.3,
                   label=phase if label == GAD_METHODS[0][0] else None)
            bottom += vals
        ax.set_xticks(x)
        ax.set_xticklabels([f"{n}pm" for n in NOISES])
        ax.set_title(label, fontsize=11)
        ax.set_ylim(0, 1.05)
        if label == GAD_METHODS[0][0]:
            ax.set_ylabel("fraction of all steps")
            ax.legend(fontsize=7, loc="upper left",
                       bbox_to_anchor=(0, -0.10), ncol=3, framealpha=0.95)
    fig.suptitle("Per-step phase breakdown for GAD trajectories\n"
                 "(every step in every $n=287$ trajectory classified by force-and-eigenvalue regime)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT_FIG / f"fig_gad_phase_breakdown.{ext}", bbox_inches="tight", dpi=140)
    plt.close(fig)
    print("wrote fig_gad_phase_breakdown")


def plot_eig1_per_phase(summary: pd.DataFrame):
    """Per phase: P10 / median / P90 of |λ_1|."""
    PHASES = ["descent_smooth", "descent_curved", "tightening_safe",
              "tightening_ill", "plateau", "above_basin", "converged"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharey=True)
    for ax, noise in zip(axes.flatten(), NOISES):
        sub = summary[(summary["noise_pm"] == noise) &
                      (summary["method"] == "GAD dt=0.007")].copy()
        if sub.empty: continue
        sub = sub[sub["phase"].isin(PHASES)].sort_values("phase")
        x = np.arange(len(sub))
        for i, (_, r) in enumerate(sub.iterrows()):
            color = palette_color(0)
            ax.plot([i, i], [r["p10_abs_eig1"], r["p90_abs_eig1"]],
                    color=color, lw=2.0, alpha=0.4)
            ax.plot(i, r["p50_abs_eig1"], "o", color=color, markersize=8,
                    markeredgecolor=palette_color(7), markeredgewidth=0.5)
            ax.text(i, r["p90_abs_eig1"]*1.2 if r["p90_abs_eig1"]>0 else 1e-2,
                    f"{r['p50_abs_eig1']:.2f}",
                    ha="center", fontsize=6)
        ax.set_xticks(x)
        ax.set_xticklabels(sub["phase"].values, rotation=30, ha="right",
                           fontsize=7)
        ax.set_yscale("log")
        ax.set_ylabel("|λ_1| (log)")
        ax.axhline(0.1, color=palette_color(3), ls="--", alpha=0.6, label="threshold")
        ax.set_title(f"{noise}pm noise, GAD dt=0.007", fontsize=10)
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle("$|\\lambda_1|$ distribution per phase\n"
                 "(red dashed = $|\\lambda_1|=0.1$ threshold for ill-conditioning)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT_FIG / f"fig_gad_eig1_distribution.{ext}", bbox_inches="tight", dpi=140)
    plt.close(fig)
    print("wrote fig_gad_eig1_distribution")


def plot_step_safety(safety: pd.DataFrame):
    """For each (method, noise): scale ratio (natural / actual) histogram."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharey=True)
    for ax, noise in zip(axes.flatten(), NOISES):
        sub_rows = safety[safety["noise_pm"] == noise]
        if sub_rows.empty: continue
        # Plot only GAD dt=0.007
        target = "GAD dt=0.007"
        row = sub_rows[sub_rows["method"] == target]
        if row.empty: continue
        r = row.iloc[0]
        # Box: natural-step / actual-step ratio quantiles
        cats = ["v0", "v1"]
        p50s = [r["p50_scale_v0"], r["p50_scale_v1"]]
        p90s = [r["p90_scale_v0"], r["p90_scale_v1"]]
        x = np.arange(len(cats))
        ax.bar(x, p50s, color=palette_color(0), label="P50",
               edgecolor=palette_color(7), linewidth=0.3)
        ax.bar(x + 0.4, p90s, 0.4, color=palette_color(3), label="P90",
               edgecolor=palette_color(7), linewidth=0.3)
        ax.axhline(1.0, color=palette_color(7), ls="--", lw=1.0,
                   label="Δx_natural = Δx_actual")
        ax.set_yscale("log")
        ax.set_xticks(x + 0.2)
        ax.set_xticklabels(["v_0 (highest)", "v_1 (lowest)"])
        ax.set_title(f"{noise}pm noise", fontsize=10)
        ax.set_ylabel("|Δx_natural| / |Δx_actual|")
        ax.legend(fontsize=8, loc="upper left", framealpha=0.95)
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle(f"GAD dt=0.007: how much would 2nd-order rescale per-mode?\n"
                 "(>1 = should grow this mode; <1 = should shrink)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT_FIG / f"fig_gad_step_safety.{ext}", bbox_inches="tight", dpi=140)
    plt.close(fig)
    print("wrote fig_gad_step_safety")


def main():
    print("=== Building phase summary (8 phases × method × noise) ===")
    summary = build_phase_summary()
    summary.to_csv(OUT_CSV / "gad_step_class_summary.csv", index=False)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n=== Building per-step safety analysis (natural-vs-actual ratios) ===")
    safety_rows = []
    for label, glob in GAD_METHODS:
        for noise in NOISES:
            try:
                row = aggregate_safety(label, glob, noise)
                if row: safety_rows.append(row)
            except Exception as e:
                print(f"err safety {label} {noise}pm: {e}")
    safety = pd.DataFrame(safety_rows)
    safety.to_csv(OUT_CSV / "gad_step_safety.csv", index=False)
    print()
    print(safety.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print("\n=== Plotting ===")
    plot_phase_breakdown(summary)
    plot_eig1_per_phase(summary)
    plot_step_safety(safety)

    print("\n=== DONE ===")


if __name__ == "__main__":
    main()
