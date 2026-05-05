#!/usr/bin/env python
"""Analysis pipeline for the hybrid_gad_newton's hybrid_gad sweep (60398168).

Reads all summary parquets from runs/hybrid_gad_newton/<tag>/, builds:
  analysis_2026_04_29/hybrid_gad_newton_summary.csv  # per (cell, noise, trust_radius)
  analysis_2026_04_29/hybrid_gad_newton_pivot.md     # readable tables
  figures/fig_hybrid_conv_vs_tr.pdf                  # conv_pct vs trust_radius (per cell, per noise)
  figures/fig_hybrid_steps_vs_tr.pdf                 # median steps vs trust_radius
  figures/fig_hybrid_switch_compare.pdf              # switch=True vs False side-by-side
  figures/fig_hybrid_method_compare.pdf              # 5-cell × 2-noise heatmap
  figures/fig_hybrid_step_phases.pdf                 # last_method (gad / newton) breakdown
"""
from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_CSV = Path("/lustre06/project/6033559/memoozd/GAD_plus/analysis_2026_04_29")
OUT_FIG = Path("/lustre06/project/6033559/memoozd/GAD_plus/figures")
OUT_CSV.mkdir(exist_ok=True, parents=True); OUT_FIG.mkdir(exist_ok=True, parents=True)
RUNS = Path("/lustre07/scratch/memoozd/gadplus/runs/hybrid_gad_newton")

NOISES = [10, 100]
TRS = [0.005, 0.01, 0.02, 0.05, 0.10]
CELL_LABELS = {
    "hybrid_swfalse":               ("hybrid (no Eckart, force-switch)",        "#1f77b4", "o"),
    "hybrid_eckart_swfalse":        ("hybrid_eckart, switch=False (force-switch)", "#2ca02c", "s"),
    "hybrid_eckart_swtrue":         ("hybrid_eckart, switch=True (eig-clear)",     "#d62728", "^"),
    "hybrid_damped_eckart_swfalse": ("hybrid_damped_eckart, switch=False",         "#9467bd", "D"),
    "hybrid_damped_eckart_swtrue":  ("hybrid_damped_eckart, switch=True",          "#ff7f0e", "v"),
}


def parse_dir_name(dirname: str) -> dict | None:
    """e.g. 'hybrid_eckart_swtrue_dt5e-3_tr0.01_100pm' → {...}"""
    parts = dirname.split("_")
    if "dt5e-3" not in parts: return None
    # method is everything before 'dt5e-3'; trust = tr0.XX; noise = NNpm
    try:
        i_dt = parts.index("dt5e-3")
        method_parts = parts[:i_dt]
        method = "_".join(method_parts).lower()
        # find tr*
        tr_part = next(p for p in parts if p.startswith("tr"))
        trust = float(tr_part[2:])
        noise_part = next(p for p in parts if p.endswith("pm"))
        noise_pm = int(noise_part.replace("pm", ""))
        return dict(method=method, trust_radius=trust, noise_pm=noise_pm)
    except Exception:
        return None


def build_summary() -> pd.DataFrame:
    rows = []
    if not RUNS.exists():
        print(f"WARNING: {RUNS} doesn't exist yet"); return pd.DataFrame()
    for d in sorted(RUNS.iterdir()):
        if not d.is_dir(): continue
        meta = parse_dir_name(d.name)
        if meta is None: continue
        files = list(d.glob("summary_*.parquet"))
        if not files: continue
        try:
            agg = duckdb.execute(f"""
              SELECT COUNT(*) AS n, SUM(CAST(converged AS INT)) AS n_conv,
                AVG(total_steps) AS avg_total_steps,
                MEDIAN(total_steps) AS med_total_steps,
                AVG(CASE WHEN converged THEN converged_step END) AS avg_step_conv,
                MEDIAN(CASE WHEN converged THEN converged_step END) AS med_step_conv,
                QUANTILE_CONT(CASE WHEN converged THEN converged_step END, 0.95) AS p95_step_conv,
                AVG(wall_time_s) AS avg_wall_s, MEDIAN(wall_time_s) AS med_wall_s,
                SUM(wall_time_s) AS total_wall_s, AVG(final_force_max) AS avg_fmax,
                MIN(final_force_max) AS min_fmax,
                AVG(CAST(final_n_neg AS DOUBLE)) AS avg_n_neg,
                COUNT(CASE WHEN last_step_method LIKE '%newton%' THEN 1 END) AS n_used_newton
              FROM '{files[0]}'
            """).df()
            r = agg.iloc[0]
            rows.append({**meta, "n": int(r["n"]), "n_conv": int(r["n_conv"]),
                "conv_pct": 100.0 * r["n_conv"] / r["n"] if r["n"] else 0,
                "avg_total_steps": r["avg_total_steps"],
                "med_total_steps": r["med_total_steps"],
                "med_step_conv": r["med_step_conv"],
                "p95_step_conv": r["p95_step_conv"],
                "avg_wall_s": r["avg_wall_s"], "med_wall_s": r["med_wall_s"],
                "total_wall_s": r["total_wall_s"],
                "wall_per_conv_s": (r["total_wall_s"] / r["n_conv"]) if r["n_conv"] > 0 else np.nan,
                "avg_fmax": r["avg_fmax"], "min_fmax": r["min_fmax"],
                "avg_n_neg": r["avg_n_neg"],
                "n_used_newton": int(r["n_used_newton"]),
                "frac_used_newton": int(r["n_used_newton"]) / r["n"] if r["n"] else 0.0,
            })
        except Exception as e:
            print(f"err {d.name}: {e}")
    if not rows:
        print("WARNING: no rows accumulated")
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["method", "noise_pm", "trust_radius"])


def plot_conv_vs_tr(df: pd.DataFrame):
    """One row × 2 cols (10pm, 100pm); each line = method."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
    for ax, noise in zip(axes, NOISES):
        sub = df[df["noise_pm"] == noise]
        if sub.empty: continue
        for key, (label, color, marker) in CELL_LABELS.items():
            cell = sub[sub["method"] == key].sort_values("trust_radius")
            if cell.empty: continue
            ax.plot(cell["trust_radius"], cell["conv_pct"],
                    marker=marker, color=color, label=label, lw=1.7, markersize=8)
            for _, r in cell.iterrows():
                ax.annotate(f"{r['conv_pct']:.0f}", (r["trust_radius"], r["conv_pct"]),
                            xytext=(2, 4), textcoords="offset points", fontsize=6)
        ax.set_xscale("log")
        ax.set_xlabel("trust_radius (Å)")
        ax.set_ylabel("conv % ($n_{neg}{=}1 \\wedge f_{\\max}{<}0.01$)" if noise == NOISES[0] else "")
        ax.set_title(f"{noise} pm noise, n=287")
        ax.set_ylim(0, 105)
        ax.grid(alpha=0.3)
        if noise == NOISES[0]:
            ax.legend(fontsize=8, loc="lower right", framealpha=0.95)
    fig.suptitle("Hybrid GAD-Newton sweep — convergence vs trust_radius",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT_FIG / f"fig_hybrid_conv_vs_tr.{ext}", bbox_inches="tight", dpi=140)
    plt.close(fig); print("wrote fig_hybrid_conv_vs_tr")


def plot_steps_vs_tr(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
    for ax, noise in zip(axes, NOISES):
        sub = df[df["noise_pm"] == noise]
        if sub.empty: continue
        for key, (label, color, marker) in CELL_LABELS.items():
            cell = sub[sub["method"] == key].sort_values("trust_radius")
            if cell.empty: continue
            ax.plot(cell["trust_radius"], cell["med_step_conv"],
                    marker=marker, color=color, label=label, lw=1.7, markersize=8)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("trust_radius (Å)")
        ax.set_ylabel("median steps to converge (log)" if noise == NOISES[0] else "")
        ax.set_title(f"{noise} pm noise")
        ax.grid(alpha=0.3)
        if noise == NOISES[0]:
            ax.legend(fontsize=8, loc="upper right", framealpha=0.95)
    fig.suptitle("Hybrid GAD–Newton — median steps to converge vs trust_radius",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT_FIG / f"fig_hybrid_steps_vs_tr.{ext}", bbox_inches="tight", dpi=140)
    plt.close(fig); print("wrote fig_hybrid_steps_vs_tr")


def plot_switch_compare(df: pd.DataFrame):
    """For each method-family (eckart, damped_eckart): switch=True vs False curves."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharey=True)
    families = [
        ("hybrid_eckart", "Hybrid Eckart"),
        ("hybrid_damped_eckart", "Hybrid Damped Eckart"),
    ]
    for j, noise in enumerate(NOISES):
        for i, (family, fam_label) in enumerate(families):
            ax = axes[i, j]
            for sw, color in [("False", "#1f77b4"), ("True", "#d62728")]:
                key = f"{family}_sw{sw.lower()}"
                cell = df[(df["method"] == key) & (df["noise_pm"] == noise)].sort_values("trust_radius")
                if cell.empty: continue
                ax.plot(cell["trust_radius"], cell["conv_pct"],
                        marker="o", color=color, lw=2.0, markersize=8,
                        label=f"switch={sw}")
            ax.set_xscale("log")
            ax.set_xlabel("trust_radius (Å)")
            if j == 0:
                ax.set_ylabel("conv %")
            ax.set_title(f"{fam_label} — {noise}pm")
            ax.grid(alpha=0.3); ax.set_ylim(0, 105)
            ax.legend(fontsize=9, loc="lower right")
    fig.suptitle("switch_based_on_hessian_eigval — does using eigenvalue-clear-index1 vs $\\|F\\|<10^{-3}$ trigger help?",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT_FIG / f"fig_hybrid_switch_compare.{ext}", bbox_inches="tight", dpi=140)
    plt.close(fig); print("wrote fig_hybrid_switch_compare")


def plot_method_heatmap(df: pd.DataFrame):
    """5 method × 5 trust_radius heatmap of conv_pct, two panels (10pm, 100pm)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, noise in zip(axes, NOISES):
        sub = df[df["noise_pm"] == noise]
        if sub.empty: continue
        # Pivot: rows=method, cols=trust_radius, values=conv_pct
        pv = sub.pivot(index="method", columns="trust_radius", values="conv_pct").reindex(
            list(CELL_LABELS.keys()))
        if pv.empty: continue
        im = ax.imshow(pv.values, aspect="auto", cmap="RdYlGn", vmin=0, vmax=100,
                       interpolation="nearest")
        ax.set_xticks(range(len(pv.columns)))
        ax.set_xticklabels([f"{tr:g}" for tr in pv.columns])
        ax.set_yticks(range(len(pv.index)))
        ax.set_yticklabels([CELL_LABELS.get(m, (m,))[0] for m in pv.index],
                           fontsize=8)
        ax.set_xlabel("trust_radius (Å)")
        ax.set_title(f"{noise} pm noise")
        for i in range(len(pv.index)):
            for j in range(len(pv.columns)):
                v = pv.values[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                            color="white" if v < 50 else "black",
                            fontsize=9, fontweight="bold")
        plt.colorbar(im, ax=ax, label="conv %")
    fig.suptitle("Hybrid GAD-Newton sweep: full grid (5 methods × 5 trust radii)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT_FIG / f"fig_hybrid_method_compare.{ext}", bbox_inches="tight", dpi=140)
    plt.close(fig); print("wrote fig_hybrid_method_compare")


def plot_phase_breakdown(df: pd.DataFrame):
    """frac_used_newton: shows which trust radii actually triggered the Newton phase."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
    for ax, noise in zip(axes, NOISES):
        sub = df[df["noise_pm"] == noise]
        if sub.empty: continue
        for key, (label, color, marker) in CELL_LABELS.items():
            cell = sub[sub["method"] == key].sort_values("trust_radius")
            if cell.empty: continue
            ax.plot(cell["trust_radius"], cell["frac_used_newton"]*100,
                    marker=marker, color=color, label=label, lw=1.7, markersize=8)
        ax.set_xscale("log")
        ax.set_xlabel("trust_radius (Å)")
        ax.set_ylabel("% of samples using Newton on the LAST step" if noise == NOISES[0] else "")
        ax.set_title(f"{noise} pm noise")
        ax.set_ylim(-5, 105); ax.grid(alpha=0.3)
        if noise == NOISES[0]:
            ax.legend(fontsize=8, loc="upper right", framealpha=0.95)
    fig.suptitle("Did the hybrid switch fire? (% samples whose terminating step used Newton)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT_FIG / f"fig_hybrid_step_phases.{ext}", bbox_inches="tight", dpi=140)
    plt.close(fig); print("wrote fig_hybrid_step_phases")


def write_pivot_md(df: pd.DataFrame):
    lines = ["# Hybrid GAD-Newton sweep — pivot tables", ""]
    lines.append("Conv % by (method × trust_radius), per noise level:\n")
    for noise in NOISES:
        sub = df[df["noise_pm"] == noise]
        if sub.empty: continue
        lines.append(f"\n## {noise} pm noise\n")
        pv = sub.pivot(index="method", columns="trust_radius", values="conv_pct")
        lines.append(pv.to_string(float_format=lambda x: f"{x:.1f}"))
        lines.append("\n")
        # Median steps
        lines.append(f"\n### Median steps to converge — {noise}pm:\n")
        pv = sub.pivot(index="method", columns="trust_radius", values="med_step_conv")
        lines.append(pv.to_string(float_format=lambda x: f"{x:.0f}"))
        lines.append("\n")
        # Newton-fired fraction
        lines.append(f"\n### Fraction of last-steps that used Newton — {noise}pm:\n")
        pv = sub.pivot(index="method", columns="trust_radius", values="frac_used_newton")
        lines.append(pv.to_string(float_format=lambda x: f"{x:.2f}"))
        lines.append("\n")
    (OUT_CSV / "hybrid_gad_newton_pivot.md").write_text("\n".join(lines))
    print("wrote hybrid_gad_newton_pivot.md")


def main():
    print("=== Building summary ===")
    df = build_summary()
    if df.empty:
        print("No cells found yet"); return
    df.to_csv(OUT_CSV / "hybrid_gad_newton_summary.csv", index=False)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print()
    plot_conv_vs_tr(df)
    plot_steps_vs_tr(df)
    plot_switch_compare(df)
    plot_method_heatmap(df)
    plot_phase_breakdown(df)
    write_pivot_md(df)
    print("=== DONE ===")


if __name__ == "__main__":
    main()
