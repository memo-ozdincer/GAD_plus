#!/usr/bin/env python
"""Figure set for IRC_COMPREHENSIVE_2026-04-28v2.

Differences from v1 (figures_2026_04_28.py):
  * "Sella cart+Eckart" line uses the delta0=0.10, gamma=0.40 setting
    (n=100 — tuning grid winner from the sella_tune sweep). The
    delta0=0.048, gamma=0.0 setting (n=300) goes to the appendix.
  * Grouped bar chart is now stacked: each bar shows
    intended / half-intended / unintended partitions (0-100% of cell).
  * Two appendix figures: Sella tuning comparison (3 configs x 6 noise)
    and GAD bigger-dt comparison (whatever is available).

Output names use _v2 suffix where they collide with v1.
"""
from __future__ import annotations

import os
from pathlib import Path

import duckdb
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plotting_style import apply_plot_style, palette_color

apply_plot_style()

OUT = Path("/lustre06/project/6033559/memoozd/GAD_plus/figures")
OUT.mkdir(exist_ok=True)
NOISES = [10, 30, 50, 100, 150, 200]

C_GAD       = palette_color(0)
C_SELLA_CE  = palette_color(3)
C_SELLA_INT = palette_color(4)

C_INT, C_HALF, C_UN = palette_color(2), palette_color(8), palette_color(3)

# --- Canonical 3-method config (Sella cart+Eckart, delta0=0.10, gamma=0.40, n=100) ---
METHODS_3M = {
    "gad_eckart": dict(
        summaries=[f"/lustre07/scratch/memoozd/gadplus/runs/round2/summary_gad_dt003_{n}pm.parquet"
                   for n in [10,30,50]]
                + [f"/lustre07/scratch/memoozd/gadplus/runs/round3/summary_gad_dt003_{n}pm.parquet"
                   for n in [100,150,200]],
        conv_col="converged",
        irc_dir="/lustre07/scratch/memoozd/gadplus/runs/irc_sellahip_allendpoints",
        denom=300,
        color=C_GAD, marker="o", label="GAD Eckart",
    ),
    "sella_tuned": dict(
        summaries=[f"/lustre07/scratch/memoozd/gadplus/runs/sella_tune/libdef/"
                   f"summary_sella_cartesian_eckart_fmax0p01_libdef_{n}pm.parquet"
                   for n in NOISES],
        conv_col="conv_nneg1_fmax001",
        irc_dir="/lustre07/scratch/memoozd/gadplus/runs/irc_sella_libdef",
        denom=100,
        color=C_SELLA_CE, marker="s", label="Sella cart+Eckart (δ₀=0.10, γ=0.40)",
    ),
    "sella_int": dict(
        summaries=[f"/lustre07/scratch/memoozd/gadplus/runs/sella_2000/"
                   f"summary_sella_internal_fmax0p01_{n}pm.parquet" for n in NOISES],
        conv_col="conv_nneg1_fmax001",
        irc_dir="/lustre07/scratch/memoozd/gadplus/runs/irc_sella_int_2000",
        denom=300,
        color=C_SELLA_INT, marker="v", label="Sella internal",
    ),
}

# Sella tuning configs for appendix
SELLA_TUNE = {
    "default": dict(
        path_tmpl="/lustre07/scratch/memoozd/gadplus/runs/sella_tune/default/"
                  "summary_sella_cartesian_eckart_fmax0p01_default_{n}pm.parquet",
        color=palette_color(3), marker="s", label="δ₀=0.048, γ=0"),
    "libdef": dict(
        path_tmpl="/lustre07/scratch/memoozd/gadplus/runs/sella_tune/libdef/"
                  "summary_sella_cartesian_eckart_fmax0p01_libdef_{n}pm.parquet",
        color=palette_color(0), marker="o", label="δ₀=0.10, γ=0.40"),
    "lson": dict(
        path_tmpl="/lustre07/scratch/memoozd/gadplus/runs/sella_tune/lson/"
                  "summary_sella_cartesian_eckart_fmax0p01_lson_{n}pm.parquet",
        color=palette_color(4), marker="^", label="lson (δ₀=0.048, γ=0.4)"),
}

GAD_BIGDT = {
    "gad_dt003_fmax (canonical)": dict(
        path_tmpl="/lustre07/scratch/memoozd/gadplus/runs/gad_eckart_fmax/"
                  "summary_gad_dt003_fmax_{n}pm.parquet",
        color=palette_color(0), marker="o"),
    "gad_dt005_fmax": dict(
        path_tmpl="/lustre07/scratch/memoozd/gadplus/runs/gad_bigger_dt/gad_dt005_fmax/"
                  "summary_gad_dt005_fmax_{n}pm.parquet",
        color=palette_color(2), marker="s"),
    "gad_dt010_fmax": dict(
        path_tmpl="/lustre07/scratch/memoozd/gadplus/runs/gad_bigger_dt/gad_dt010_fmax/"
                  "summary_gad_dt010_fmax_{n}pm.parquet",
        color=palette_color(1), marker="^"),
    "gad_dt020_fmax": dict(
        path_tmpl="/lustre07/scratch/memoozd/gadplus/runs/gad_bigger_dt/gad_dt020_fmax/"
                  "summary_gad_dt020_fmax_{n}pm.parquet",
        color=palette_color(3), marker="v"),
}


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"wrote {name}")


def conv_rate(path, conv_col, denom):
    if not os.path.exists(path):
        return None
    r = duckdb.execute(
        f"SELECT COUNT(*), SUM(CASE WHEN {conv_col} THEN 1 ELSE 0 END) FROM '{path}'"
    ).fetchone()
    n, c = r[0] or 0, r[1] or 0
    if n == 0: return None
    return 100.0 * c / max(n, denom if denom else n), n


def load_irc(irc_dir):
    if not os.path.exists(irc_dir):
        return None
    files = [f for f in os.listdir(irc_dir) if f.endswith(".parquet")]
    if not files: return None
    return duckdb.execute(f"SELECT * FROM '{irc_dir}/*.parquet'").df()


def fig_cmp_conv_v2():
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for k, m in METHODS_3M.items():
        xs, ys = [], []
        for path, n in zip(m["summaries"], NOISES):
            r = conv_rate(path, m["conv_col"], m["denom"])
            if r is None: continue
            xs.append(n); ys.append(r[0])
        ax.plot(xs, ys, "-", color=m["color"], marker=m["marker"], linewidth=2.2,
                markersize=9, markerfacecolor="white", markeredgewidth=2,
                label=m["label"])
    ax.set_xlabel("TS noise (pm)", fontsize=11)
    ax.set_ylabel("TS converged rate (%)", fontsize=11)
    ax.set_xticks(NOISES); ax.set_ylim(0, 105)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left", fontsize=9.5, framealpha=0.95)
    save(fig, "fig_cmp_conv_3m_v2")


def fig_cmp_irc_topo_v2():
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for k, m in METHODS_3M.items():
        irc = load_irc(m["irc_dir"])
        if irc is None:
            # Mark missing IRC (libdef IRC pending)
            continue
        xs, ys = [], []
        for n in NOISES:
            g = irc[irc["noise_pm"] == n]
            if len(g):
                xs.append(n); ys.append(100 * g["topology_intended"].mean())
        ax.plot(xs, ys, "-", color=m["color"], marker=m["marker"], linewidth=2.2,
                markersize=9, markerfacecolor="white", markeredgewidth=2,
                label=m["label"])
    ax.set_xlabel("TS noise (pm)", fontsize=11)
    ax.set_ylabel("IRC TOPO-intended (%)", fontsize=11)
    ax.set_xticks(NOISES); ax.set_ylim(0, 105)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left", fontsize=9.5, framealpha=0.95)
    save(fig, "fig_cmp_irc_topo_3m_v2")


def fig_irc_grouped_stacked_v2():
    """Stacked grouped bar: per (noise, method), stack of intended/half/unintended.
    Bars edge-colored by method (so method identity is visible) and fill-colored
    by outcome class."""
    method_data = {}
    for k, m in METHODS_3M.items():
        irc = load_irc(m["irc_dir"])
        ints, halfs, uns = [], [], []
        for n in NOISES:
            if irc is None:
                ints.append(np.nan); halfs.append(np.nan); uns.append(np.nan); continue
            g = irc[irc["noise_pm"] == n]
            if not len(g):
                ints.append(np.nan); halfs.append(np.nan); uns.append(np.nan); continue
            tot = len(g)
            ni = int(g["topology_intended"].sum())
            nh = int((g["topology_half_intended"] & ~g["topology_intended"]).sum())
            nu = tot - ni - nh
            ints.append(100*ni/tot); halfs.append(100*nh/tot); uns.append(100*nu/tot)
        method_data[k] = (ints, halfs, uns, m["color"], m["label"])

    fig, ax = plt.subplots(figsize=(11.5, 5.6))
    n_methods = len(METHODS_3M)
    width = 0.27
    x = np.arange(len(NOISES))
    offsets = np.linspace(-(n_methods-1)/2, (n_methods-1)/2, n_methods) * width

    plotted_outcome = False
    for i, (k, (ints, halfs, uns, color, label)) in enumerate(method_data.items()):
        xpos = x + offsets[i]
        ax.bar(xpos, ints, width, color=C_INT, edgecolor=color, linewidth=1.6,
               label="intended" if not plotted_outcome else None)
        ax.bar(xpos, halfs, width, bottom=ints, color=C_HALF, edgecolor=color,
               linewidth=1.6, label="half-intended" if not plotted_outcome else None)
        ax.bar(xpos, uns, width,
               bottom=np.array(ints, dtype=float) + np.array(halfs, dtype=float),
               color=C_UN, edgecolor=color, linewidth=1.6,
               label="unintended" if not plotted_outcome else None)
        plotted_outcome = True
        for xi, ni in zip(xpos, ints):
            if not np.isnan(ni) and ni > 8:
                ax.text(xi, ni/2, f"{ni:.0f}", ha="center", va="center",
                        fontsize=8, fontweight="bold", color="white")
    ax.set_xlabel("TS noise (pm)", fontsize=11)
    ax.set_ylabel("IRC TOPO outcome (%)", fontsize=11)
    ax.set_xticks(x); ax.set_xticklabels([str(n) for n in NOISES])
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)

    # Two legends
    method_handles = [plt.Rectangle((0,0),1,1, facecolor="white",
                                    edgecolor=v[3], linewidth=1.8, label=v[4])
                      for v in method_data.values()]
    leg1 = ax.legend(loc="upper right", fontsize=9, framealpha=0.95, title="outcome")
    ax.add_artist(leg1)
    ax.legend(handles=method_handles, loc="upper left", fontsize=8.5,
              framealpha=0.95, title="method (edge color)")
    save(fig, "fig_irc_intended_grouped_3m_v2")


def fig_sella_tune_grid():
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for k, c in SELLA_TUNE.items():
        xs, ys, ns = [], [], []
        for n in NOISES:
            p = c["path_tmpl"].format(n=n)
            if not os.path.exists(p): continue
            r = duckdb.execute(
                f"SELECT 100.0*AVG(CASE WHEN conv_nneg1_fmax001 THEN 1.0 ELSE 0.0 END), COUNT(*) FROM '{p}'"
            ).fetchone()
            xs.append(n); ys.append(r[0]); ns.append(r[1])
        ax.plot(xs, ys, "-", color=c["color"], marker=c["marker"], linewidth=2.0,
                markersize=8, markerfacecolor="white", markeredgewidth=1.8,
                label=c["label"])
    ax.set_xlabel("TS noise (pm)", fontsize=11)
    ax.set_ylabel("TS converged rate (%, fmax<0.01 ∧ n_neg=1)", fontsize=11)
    ax.set_xticks(NOISES); ax.set_ylim(0, 105)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left", fontsize=10, framealpha=0.95)
    save(fig, "fig_sella_tune_grid")


def fig_gad_bigger_dt():
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for k, c in GAD_BIGDT.items():
        xs, ys, ns = [], [], []
        for n in NOISES:
            p = c["path_tmpl"].format(n=n)
            if not os.path.exists(p): continue
            r = duckdb.execute(
                f"SELECT 100.0*AVG(CASE WHEN converged THEN 1.0 ELSE 0.0 END), COUNT(*) FROM '{p}'"
            ).fetchone()
            xs.append(n); ys.append(r[0]); ns.append(r[1])
        if not xs: continue
        ax.plot(xs, ys, "-", color=c["color"], marker=c["marker"], linewidth=2.0,
                markersize=8, markerfacecolor="white", markeredgewidth=1.8,
                label=k)
    ax.set_xlabel("TS noise (pm)", fontsize=11)
    ax.set_ylabel("TS converged rate (%, fmax<0.01 ∧ n_neg=1)", fontsize=11)
    ax.set_xticks(NOISES); ax.set_ylim(0, 105)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left", fontsize=9.5, framealpha=0.95)
    save(fig, "fig_gad_bigger_dt")


def fig_gad_stepsize_vs_step():
    """Same as v1: GAD median per-step displacement vs step, faceted by noise."""
    base = "/lustre07/scratch/memoozd/gadplus/runs/gad_eckart_fmax"
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.8), sharey=True, sharex=False)
    axes = axes.flatten()
    for ax, noise in zip(axes, NOISES):
        glob = f"{base}/traj_gad_dt003_fmax_{noise}pm_*.parquet"
        try:
            df = duckdb.execute(f"""
                SELECT step,
                       quantile_cont(disp_from_last, 0.5) AS med,
                       quantile_cont(disp_from_last, 0.25) AS q25,
                       quantile_cont(disp_from_last, 0.75) AS q75
                FROM '{glob}' WHERE step > 0 GROUP BY step ORDER BY step
            """).df()
        except Exception:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes); ax.set_title(f"{noise} pm")
            continue
        ax.fill_between(df["step"], df["q25"], df["q75"], color=C_GAD, alpha=0.22,
                        label="IQR")
        ax.plot(df["step"], df["med"], color=C_GAD, linewidth=1.2, label="median")
        ax.set_title(f"{noise} pm")
        ax.set_xlabel("step"); ax.set_yscale("log")
        ax.grid(alpha=0.3, which="both")
    axes[0].set_ylabel("|Δx| per step (Å)"); axes[3].set_ylabel("|Δx| per step (Å)")
    axes[0].legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    save(fig, "fig_gad_stepsize_vs_step_v2")


def main():
    fig_cmp_conv_v2()
    fig_cmp_irc_topo_v2()
    fig_irc_grouped_stacked_v2()
    fig_sella_tune_grid()
    fig_gad_bigger_dt()
    fig_gad_stepsize_vs_step()


if __name__ == "__main__":
    main()
