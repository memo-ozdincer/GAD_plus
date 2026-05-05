#!/usr/bin/env python
"""Narrative + full-sweep figures for IRC_TEST_2026-04-29.

Produces:
  figures/fig_narrative_irc.pdf      # Two-framing IRC headline (the simple story)
  figures/fig_narrative_conv.pdf     # Two-framing raw conv (companion)
  figures/fig_sella_variants.pdf     # All Sella variants grouped (algorithm-space)
  figures/fig_full_sweep_grid.pdf    # All methods × all noise — the "everything" view
  figures/fig_gad_lineage.pdf        # GAD dt grid + low-dt + adaptive

Structure:
  TOP (narrative): one simple comparison + one "Sella variants are different optimizers" panel
  BOTTOM (full sweep): the comprehensive grid for the appendix
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

from plotting_style import apply_plot_style, palette_color

apply_plot_style()

OUT = Path("/lustre06/project/6033559/memoozd/GAD_plus/figures")
OUT.mkdir(exist_ok=True, parents=True)
RUNS = Path("/lustre07/scratch/memoozd/gadplus/runs")
ANL = Path("/lustre06/project/6033559/memoozd/GAD_plus/analysis_2026_04_29")

NOISES = [10, 30, 50, 100, 150, 200]

# ---- Color palette: GAD = warm, Sella variants = cool family ----
C_GAD_LINE   = palette_color(0)   # canonical GAD
C_GAD_TUNED  = palette_color(1)   # tuned GAD (dt=0.007)
C_SELLA_LIB  = palette_color(3)   # cartesian+Eckart, delta0=0.10, gamma=0.40
C_SELLA_DEF  = palette_color(4)   # cartesian+Eckart, delta0=0.048, gamma=0
C_SELLA_INT  = palette_color(5)   # internal coords
C_SELLA_D3   = palette_color(6)   # cadence d=3
C_SELLA_NOH  = palette_color(7)   # no Hessian
C_GAD_DT3    = palette_color(0)
C_GAD_DT5    = palette_color(2)
C_GAD_DT7    = palette_color(1)
C_GAD_AD     = palette_color(3)
C_GAD_LOW    = palette_color(4)

SELLA_CART_LS_H1 = r"Sella cart+Eckart ($\delta_0$=0.10, $\gamma$=0.40, H/step)"
SELLA_CART_NOLS_H1 = r"Sella cart+Eckart ($\delta_0$=0.048, $\gamma$=0, H/step)"
SELLA_INT_NOLS_H1 = r"Sella internal ($\delta_0$=0.048, $\gamma$=0, H/step)"
SELLA_CART_LS_H3 = r"Sella cart+Eckart ($\delta_0$=0.10, $\gamma$=0.40, H/3)"
SELLA_CART_LS_H10 = r"Sella cart+Eckart ($\delta_0$=0.10, $\gamma$=0.40, H/10)"
SELLA_CART_LS_H25 = r"Sella cart+Eckart ($\delta_0$=0.10, $\gamma$=0.40, H/25)"
SELLA_CART_LS_NOH = r"Sella cart+Eckart ($\delta_0$=0.10, $\gamma$=0.40, no HIP H)"
SELLA_CART_LS_5K = r"Sella cart+Eckart ($\delta_0$=0.10, $\gamma$=0.40, 5k steps)"


def topo_pct(method_dir: str, noise: int) -> float | None:
    p = RUNS / "test_irc" / method_dir / f"irc_validation_sella_hip_allendpoints_{noise}pm.parquet"
    if not p.exists(): return None
    try:
        n_real = duckdb.execute(f"SELECT COUNT(*) FROM '{p}' WHERE forward_coords_flat IS NOT NULL").fetchone()[0]
        if n_real < 50: return None
        v = duckdb.execute(f"SELECT 100.0*AVG(CAST(topology_intended AS DOUBLE)) FROM '{p}'").fetchone()[0]
        return v
    except Exception: return None


def conv_pct(summary_path, criterion: str = "n_neg=1 ∧ fmax<0.01") -> float | None:
    if summary_path is None or not os.path.exists(summary_path): return None
    cols = set(duckdb.execute(f"DESCRIBE SELECT * FROM '{summary_path}'").df()["column_name"])
    fmax_col = "final_fmax" if "final_fmax" in cols else "final_force_max"
    if "fmax<0.05" in criterion:
        cond = f"final_n_neg=1 AND {fmax_col}<0.05"
    elif "fmax<0.01" in criterion:
        cond = f"final_n_neg=1 AND {fmax_col}<0.01"
    else: return None
    try:
        v = duckdb.execute(f"SELECT 100.0*AVG(CAST({cond} AS DOUBLE)) FROM '{summary_path}'").fetchone()[0]
        return v
    except Exception: return None


def find_summary(mdir: Path, noise: int) -> str | None:
    if not mdir.exists(): return None
    cands = [f for f in os.listdir(mdir) if f.startswith("summary") and f"_{noise}pm" in f and f.endswith(".parquet")]
    return str(mdir / cands[0]) if cands else None


# =============================================================================
# 1. fig_narrative_irc — the simple two-framing IRC story
# =============================================================================
def fig_narrative_irc():
    """One figure, two panels:
       (A) Most-faithful framing: best-of-each, both tuned, both with HIP H every step
       (B) Sella-as-multiple-optimizers: same method family with different coordinates / step controls
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

    # --- Panel A: most-faithful (1 GAD vs 1 Sella) ---
    ax = axes[0]
    methods = [
        ("GAD dt=0.007 (5k steps, our best)", "gad_dt007_fmax", C_GAD_TUNED, "o", "-"),
        (SELLA_CART_LS_H1, "sella_carteck_libdef", C_SELLA_LIB, "s", "-"),
    ]
    for label, mdir, color, marker, ls in methods:
        ys = [topo_pct(mdir, n) for n in NOISES]
        ax.plot(NOISES, ys, ls, marker=marker, color=color, linewidth=2.4, markersize=10,
                markerfacecolor="white", markeredgewidth=2.5, label=label)
    ax.set_title("(A) Most-faithful: best-of-each\n(both tuned, HIP H every step)",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("TS-noise (pm)", fontsize=11)
    ax.set_ylabel("IRC TOPO-intended rate (%)", fontsize=11)
    ax.set_xticks(NOISES)
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left", fontsize=9, framealpha=0.95)
    # Annotation arrow at 200pm
    a_y_g = topo_pct("gad_dt007_fmax", 200) or 0
    a_y_s = topo_pct("sella_carteck_libdef", 200) or 0
    ax.annotate(f"+{a_y_g - a_y_s:.1f}pp", xy=(200, (a_y_g+a_y_s)/2),
                xytext=(170, 30), fontsize=11, fontweight="bold", color=palette_color(7),
                arrowprops=dict(arrowstyle='->', color=palette_color(7), linewidth=1.2))

    # --- Panel B: Sella variants are different optimizers ---
    ax = axes[1]
    methods = [
        ("GAD dt=0.007 (best, reference)",     "gad_dt007_fmax", C_GAD_TUNED, "o", "-"),
        (SELLA_CART_LS_H1, "sella_carteck_libdef", C_SELLA_LIB, "s", "-"),
        (SELLA_CART_NOLS_H1, "sella_carteck_default", C_SELLA_DEF, "v", "--"),
        (SELLA_INT_NOLS_H1, "sella_internal_default", C_SELLA_INT, "^", ":"),
    ]
    for label, mdir, color, marker, ls in methods:
        ys = [topo_pct(mdir, n) for n in NOISES]
        ax.plot(NOISES, ys, ls, marker=marker, color=color, linewidth=2.0, markersize=8,
                markerfacecolor="white", markeredgewidth=1.8, label=label)
    ax.set_title("(B) Sella variants — different algorithms\n(coord system / hparams matter)",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("TS-noise (pm)", fontsize=11)
    ax.set_xticks(NOISES)
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left", fontsize=8, framealpha=0.95)

    fig.suptitle("IRC TOPO-intended rate: GAD vs Sella (test split, n=287)",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_narrative_irc.{ext}", bbox_inches="tight", dpi=140)
    plt.close(fig)
    print("wrote fig_narrative_irc")


# =============================================================================
# 2. fig_narrative_conv — companion: raw conv at two thresholds
# =============================================================================
def fig_narrative_conv():
    """2 panels: fmax<0.05 and fmax<0.01,
       both ∧ n_neg=1.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    methods = [
        ("GAD dt=0.007 (5k)", lambda n: find_summary(RUNS / "test_dtgrid/gad_dt007_fmax", n), C_GAD_TUNED, "o", "-"),
        (SELLA_CART_LS_H1,
         lambda n: find_summary(RUNS / "test_set/sella_carteck_libdef", n), C_SELLA_LIB, "s", "-"),
        (SELLA_CART_NOLS_H1,
         lambda n: find_summary(RUNS / "test_set/sella_carteck_default", n), C_SELLA_DEF, "v", "--"),
        (SELLA_INT_NOLS_H1,
         lambda n: find_summary(RUNS / "test_set/sella_internal_default", n), C_SELLA_INT, "^", ":"),
        (SELLA_CART_LS_H3,
         lambda n: find_summary(RUNS / "test_hessfreq/sella_carteck_libdef_d3", n), C_SELLA_D3, "D", "-."),
    ]

    for ax, criterion, title in zip(axes,
        ["fmax<0.05", "fmax<0.01"],
        ["(A) Loose criterion $f_\\max<0.05$",
         "(B) Strict criterion $f_\\max<0.01$ (our canonical)"]):
        for label, getter, color, marker, ls in methods:
            ys = []
            for n in NOISES:
                p = getter(n)
                ys.append(conv_pct(p, criterion=f"n_neg=1 ∧ {criterion}") if p else None)
            ax.plot(NOISES, ys, ls, marker=marker, color=color, linewidth=2.0,
                    markersize=8, markerfacecolor="white", markeredgewidth=1.8, label=label)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("TS-noise (pm)", fontsize=11)
        ax.set_xticks(NOISES)
        ax.set_ylim(0, 100)
        ax.grid(alpha=0.3)
        if ax is axes[0]:
            ax.set_ylabel("Conv rate ($n_{\\rm neg}{=}1\\ \\wedge\\ f_\\max{<}T$, %)", fontsize=11)
        ax.legend(loc="lower left", fontsize=8, framealpha=0.95)

    fig.suptitle("Raw convergence: GAD vs Sella variants at two convergence criteria (test, n=287)",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_narrative_conv.{ext}", bbox_inches="tight", dpi=140)
    plt.close(fig)
    print("wrote fig_narrative_conv")


# =============================================================================
# 3. fig_sella_variants — Sella across all hyperparam axes
# =============================================================================
def fig_sella_variants():
    """Sella variants grouped by step control, coordinates,
       Hessian cadence (every 1/3/5/10/25/never)."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)

    # Panel A: step-control parameters
    ax = axes[0]
    methods = [
        (r"$\delta_0$=0.10, $\gamma$=0.40",
         lambda n: find_summary(RUNS / "test_set/sella_carteck_libdef", n), C_SELLA_LIB, "s"),
        (r"$\delta_0$=0.048, $\gamma$=0",
         lambda n: find_summary(RUNS / "test_set/sella_carteck_default", n), C_SELLA_DEF, "v"),
    ]
    for label, getter, color, marker in methods:
        ys = [conv_pct(getter(n), "n_neg=1 ∧ fmax<0.01") for n in NOISES]
        ax.plot(NOISES, ys, "-", marker=marker, color=color, linewidth=2.0,
                markersize=8, markerfacecolor="white", markeredgewidth=1.8, label=label)
    ax.set_title("Sella, Cartesian + Eckart\nstep-control parameters differ", fontsize=10, fontweight="bold")
    ax.set_xlabel("TS-noise (pm)")
    ax.set_ylabel("Conv ($f_\\max{<}0.01\\,\\wedge\\,n_{\\rm neg}{=}1$, %)")
    ax.set_xticks(NOISES); ax.set_ylim(0, 100); ax.grid(alpha=0.3)
    ax.legend(loc="lower left", fontsize=9)

    # Panel B: coord system
    ax = axes[1]
    methods = [
        (r"Cartesian+Eckart ($\delta_0$=0.10, $\gamma$=0.40)",
         lambda n: find_summary(RUNS / "test_set/sella_carteck_libdef", n), C_SELLA_LIB, "s"),
        (r"Internal coords ($\delta_0$=0.048, $\gamma$=0)",
         lambda n: find_summary(RUNS / "test_set/sella_internal_default", n), C_SELLA_INT, "^"),
    ]
    for label, getter, color, marker in methods:
        ys = [conv_pct(getter(n), "n_neg=1 ∧ fmax<0.01") for n in NOISES]
        ax.plot(NOISES, ys, "-", marker=marker, color=color, linewidth=2.0,
                markersize=8, markerfacecolor="white", markeredgewidth=1.8, label=label)
    ax.set_title("Sella coordinate system\nand step control differ",
                 fontsize=10, fontweight="bold")
    ax.set_xlabel("TS-noise (pm)")
    ax.set_xticks(NOISES); ax.set_ylim(0, 100); ax.grid(alpha=0.3)
    ax.legend(loc="lower left", fontsize=9)

    # Panel C: Hessian cadence
    ax = axes[2]
    cadences = [
        (1,  "H/step", C_SELLA_LIB,  "s",
         lambda n: find_summary(RUNS / "test_set/sella_carteck_libdef", n)),
        (3,  "H/3", C_SELLA_D3, "D",
         lambda n: find_summary(RUNS / "test_hessfreq/sella_carteck_libdef_d3", n)),
        (10, "H/10", palette_color(8), "P",
         lambda n: find_summary(RUNS / "test_hessfreq/sella_carteck_libdef_d10", n)),
        (25, "H/25", palette_color(9), "X",
         lambda n: find_summary(RUNS / "test_hessfreq/sella_carteck_libdef_d25", n)),
        (None, "no HIP H (BFGS only)", C_SELLA_NOH, "*", None),
    ]
    for d, label, color, marker, getter in cadences:
        if getter is None:
            # Sella nohess from parsed CSV
            import pandas as pd
            df = pd.read_csv(ANL / "sella_nohess_partial.csv")
            df = df[df["method"]=="carteck_nohess"].sort_values("noise_pm")
            xs = df["noise_pm"].tolist()
            ys = df["ours_TS_pct_partial"].tolist()
            ax.plot(xs, ys, "-", marker=marker, color=color, linewidth=2.0,
                    markersize=10, markerfacecolor="white", markeredgewidth=1.8,
                    label=label + " (partial coverage)")
            continue
        ys = [conv_pct(getter(n), "n_neg=1 ∧ fmax<0.01") for n in NOISES]
        ax.plot(NOISES, ys, "-", marker=marker, color=color, linewidth=2.0,
                markersize=8, markerfacecolor="white", markeredgewidth=1.8, label=label)
    ax.set_title("Sella, Cartesian+Eckart, $\\delta_0$=0.10, $\\gamma$=0.40\nHIP Hessian injection cadence",
                 fontsize=10, fontweight="bold")
    ax.set_xlabel("TS-noise (pm)")
    ax.set_xticks(NOISES); ax.set_ylim(0, 100); ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)

    fig.suptitle("Sella's hyperparameter axes: same criterion ($n_{\\rm neg}{=}1\\,\\wedge\\,f_\\max{<}0.01$), test n=287",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_sella_variants.{ext}", bbox_inches="tight", dpi=140)
    plt.close(fig)
    print("wrote fig_sella_variants")


# =============================================================================
# 4. fig_gad_lineage — GAD across dt and step-budget axes
# =============================================================================
def fig_gad_lineage():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

    # Panel A: dt grid (5k steps)
    ax = axes[0]
    dts = [
        ("dt=0.003", "test_dtgrid/gad_dt003_fmax", palette_color(0), "o"),
        ("dt=0.004", "test_dtgrid/gad_dt004_fmax", palette_color(9), "D"),
        ("dt=0.005", "test_dtgrid/gad_dt005_fmax", palette_color(2), "s"),
        ("dt=0.006", "test_dtgrid/gad_dt006_fmax", palette_color(9), "p"),
        ("dt=0.007", "test_dtgrid/gad_dt007_fmax", C_GAD_TUNED, "^"),
        ("dt=0.008 (unstable)", "test_dtgrid/gad_dt008_fmax", palette_color(3), "v"),
    ]
    for label, mdir, color, marker in dts:
        ys = [conv_pct(find_summary(RUNS / mdir, n), "n_neg=1 ∧ fmax<0.01") for n in NOISES]
        ax.plot(NOISES, ys, "-", marker=marker, color=color, linewidth=1.8,
                markersize=7, markerfacecolor="white", markeredgewidth=1.5, label=label)
    ax.set_title("GAD: dt grid (5k step budget)\nsweet spot at dt=0.007", fontsize=11, fontweight="bold")
    ax.set_xlabel("TS-noise (pm)")
    ax.set_ylabel("Conv ($f_\\max{<}0.01\\,\\wedge\\,n_{\\rm neg}{=}1$, %)")
    ax.set_xticks(NOISES); ax.set_ylim(0, 100); ax.grid(alpha=0.3)
    ax.legend(loc="lower left", fontsize=8, ncol=2)

    # Panel B: dt vs step-budget tradeoff (low-dt diagnostic)
    ax = axes[1]
    methods = [
        ("dt=0.007, 5000 steps (canonical)", "test_dtgrid/gad_dt007_fmax", C_GAD_TUNED, "o", "-"),
        ("dt=0.003, 5000 steps", "test_dtgrid/gad_dt003_fmax", palette_color(0), "s", "--"),
        ("adaptive_dt (clamped, 2000)", "test_set/gad_adaptive_dt", C_GAD_AD, "v", ":"),
    ]
    for label, mdir, color, marker, ls in methods:
        ys = [conv_pct(find_summary(RUNS / mdir, n), "n_neg=1 ∧ fmax<0.01") for n in NOISES]
        ax.plot(NOISES, ys, ls, marker=marker, color=color, linewidth=2.0,
                markersize=9, markerfacecolor="white", markeredgewidth=2.0, label=label)
    # Low-dt overlay (partial coverage)
    import re, glob
    for label, mdir, color, marker in [
        ("dt=$10^{-3}$, 20k steps (partial)", "test_lowdt/gad_dt001_fmax", palette_color(4), "P"),
        ("dt=$5{\\times}10^{-4}$, 40k steps (partial)", "test_lowdt/gad_dt0005_fmax", palette_color(6), "X"),
    ]:
        ys = []
        for n in NOISES:
            f = find_summary(RUNS / mdir, n)
            if f:
                ys.append(conv_pct(f, "n_neg=1 ∧ fmax<0.01"))
            else:
                ys.append(None)
        ax.plot(NOISES, ys, "-", marker=marker, color=color, linewidth=1.5,
                markersize=8, markerfacecolor="none", markeredgewidth=1.5,
                label=label, alpha=0.7)
    ax.set_title("GAD: dt vs step-budget\nstructural plateau at dt$\\to 0$",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("TS-noise (pm)")
    ax.set_xticks(NOISES); ax.set_ylim(0, 100); ax.grid(alpha=0.3)
    ax.legend(loc="lower left", fontsize=8)

    fig.suptitle("GAD: hyperparameter axes (test, n=287; low-dt rows partial)",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_gad_lineage.{ext}", bbox_inches="tight", dpi=140)
    plt.close(fig)
    print("wrote fig_gad_lineage")


# =============================================================================
# 5. fig_full_sweep_grid — every method on one canvas
# =============================================================================
def fig_full_sweep_grid():
    """All methods × all noise levels, color-coded by family.
       The 'expert acrobatic' panel: every comparison in one place.
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)

    # GAD on the left, Sella on the right
    gad_methods = [
        ("GAD dt=0.003 (5k, canonical)", "test_dtgrid/gad_dt003_fmax", palette_color(0), "o", "-"),
        ("GAD dt=0.005 (5k)", "test_dtgrid/gad_dt005_fmax", palette_color(9), "D", "-"),
        ("GAD dt=0.007 (5k, our best)", "test_dtgrid/gad_dt007_fmax", palette_color(1), "^", "-"),
        ("GAD dt=0.008 (5k, unstable)", "test_dtgrid/gad_dt008_fmax", palette_color(3), "v", "--"),
        ("GAD adaptive_dt (clamp, broken)", "test_set/gad_adaptive_dt", palette_color(7), "x", ":"),
        ("GAD dt=$10^{-3}$ (20k, partial)", "test_lowdt/gad_dt001_fmax", palette_color(4), "P", "-."),
    ]
    sella_methods = [
        (SELLA_CART_LS_H1, "test_set/sella_carteck_libdef", C_SELLA_LIB, "s", "-"),
        (SELLA_CART_NOLS_H1, "test_set/sella_carteck_default", C_SELLA_DEF, "v", "--"),
        (SELLA_INT_NOLS_H1, "test_set/sella_internal_default", C_SELLA_INT, "^", ":"),
        (SELLA_CART_LS_H3, "test_hessfreq/sella_carteck_libdef_d3", C_SELLA_D3, "D", "-."),
        (SELLA_CART_LS_H10, "test_hessfreq/sella_carteck_libdef_d10", palette_color(8), "P", "-."),
        (SELLA_CART_LS_H25, "test_hessfreq/sella_carteck_libdef_d25", palette_color(9), "X", "-."),
        (SELLA_CART_LS_5K, "test_sella_extended/carteck_libdef_5k", palette_color(9), "*", "-"),
    ]

    for ax, methods, title in zip(axes, [gad_methods, sella_methods],
                                   ["GAD family (test n=287, fmax<0.01 ∧ n_neg=1)",
                                    "Sella family (same criterion, test n=287)"]):
        for label, mdir, color, marker, ls in methods:
            ys = [conv_pct(find_summary(RUNS / mdir, n), "n_neg=1 ∧ fmax<0.01") for n in NOISES]
            ax.plot(NOISES, ys, ls, marker=marker, color=color, linewidth=1.8,
                    markersize=7, markerfacecolor="white", markeredgewidth=1.5, label=label)
        # Sella nohess line
        if "Sella" in title:
            import pandas as pd
            try:
                df = pd.read_csv(ANL / "sella_nohess_partial.csv")
                df = df[df["method"]=="carteck_nohess"].sort_values("noise_pm")
                ax.plot(df["noise_pm"], df["ours_TS_pct_partial"], ":", marker="*",
                        color=palette_color(7), linewidth=1.5, markersize=10,
                        markerfacecolor="white", markeredgewidth=1.5,
                        label=SELLA_CART_LS_NOH + " (partial)", alpha=0.7)
            except Exception:
                pass
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("TS-noise (pm)")
        if ax is axes[0]:
            ax.set_ylabel("Conv ($f_\\max{<}0.01\\,\\wedge\\,n_{\\rm neg}{=}1$, %)")
        ax.set_xticks(NOISES); ax.set_ylim(0, 100); ax.grid(alpha=0.3)
        ax.legend(loc="lower left", fontsize=7.5, ncol=1, framealpha=0.95)

    fig.suptitle("Full sweep — every variant on one canvas (test, n=287)",
                 fontsize=13, fontweight="bold", y=1.005)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_full_sweep_grid.{ext}", bbox_inches="tight", dpi=140)
    plt.close(fig)
    print("wrote fig_full_sweep_grid")


def main():
    fig_narrative_irc()
    fig_narrative_conv()
    fig_sella_variants()
    fig_gad_lineage()
    fig_full_sweep_grid()


if __name__ == "__main__":
    main()
