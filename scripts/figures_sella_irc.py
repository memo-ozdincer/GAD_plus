#!/usr/bin/env python
"""Generate figures for IRC_RESULTS_2026-04-16 — sella_hip on gad_dt003 TSs."""
from __future__ import annotations

import os
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plotting_style import apply_plot_style, palette_color

apply_plot_style()

IRC_DIR = "/lustre07/scratch/memoozd/gadplus/runs/irc_sellahip_full"
NOISE_TO_SURVEY = {10:"round2",30:"round2",50:"round2",100:"round3",150:"round3",200:"round3"}
SURVEY_ROOT = "/lustre07/scratch/memoozd/gadplus/runs"
OUT = Path("/lustre06/project/6033559/memoozd/GAD_plus/figures")
OUT.mkdir(exist_ok=True)

NOISES = [10, 30, 50, 100, 150, 200]


def load() -> pd.DataFrame:
    irc = duckdb.execute(f"SELECT * FROM '{IRC_DIR}/*.parquet'").df()
    ts = []
    for noise, sub in NOISE_TO_SURVEY.items():
        p = f"{SURVEY_ROOT}/{sub}/summary_gad_dt003_{noise}pm.parquet"
        ts.append(duckdb.execute(f"SELECT * FROM '{p}' WHERE converged").df())
    ts = pd.concat(ts, ignore_index=True)
    ts_renamed = ts[["sample_id","noise_pm","converged_step","final_force_norm",
                     "final_n_neg","final_energy","final_eig0","total_steps"]].rename(
        columns={"converged_step": "ts_converged_step",
                 "final_force_norm": "ts_final_force_norm",
                 "final_n_neg": "ts_final_n_neg",
                 "final_energy": "ts_final_energy",
                 "final_eig0": "ts_final_eig0",
                 "total_steps": "ts_total_steps"})
    return irc.merge(ts_renamed, on=["sample_id","noise_pm"], how="left")


def save(fig, name: str) -> None:
    for ext in ("pdf", "png"):
        path = OUT / f"{name}.{ext}"
        fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"wrote {OUT}/{name}.(pdf|png)")


def fig_rates(m: pd.DataFrame) -> None:
    rates = {crit: [] for crit in ["TOPO-int","TOPO-half","RMSD-int","RMSD-half"]}
    for noise in NOISES:
        g = m[m["noise_pm"] == noise]
        rates["TOPO-int"].append(100 * g["topology_intended"].mean())
        rates["TOPO-half"].append(100 * g["topology_half_intended"].mean())
        rates["RMSD-int"].append(100 * g["intended"].mean())
        rates["RMSD-half"].append(100 * g["half_intended"].mean())

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    styles = {
        "TOPO-int":  dict(color=palette_color(0), marker="o", lw=2.2, label="TOPO-intended (bond graph)"),
        "TOPO-half": dict(color=palette_color(0), marker="o", lw=1.2, ls="--", label="TOPO-half only"),
        "RMSD-int":  dict(color=palette_color(3), marker="s", lw=2.2, label="RMSD-intended ($<$0.3\\,\\AA)"),
        "RMSD-half": dict(color=palette_color(3), marker="s", lw=1.2, ls="--", label="RMSD-half only"),
    }
    for k, r in rates.items():
        ax.plot(NOISES, r, **styles[k])
    ax.set_xlabel("TS noise (pm)")
    ax.set_ylabel("rate (%)")
    ax.set_title("IRC validation rates vs starting TS noise (sella\\_hip, $n=1462$)")
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 100)
    ax.legend(loc="center right", fontsize=9)
    save(fig, "fig_sella_rates_vs_noise")


def fig_endpoint_quality(m: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    cats = ["both_min","only_fwd","only_rev","neither","missing"]
    colors = [palette_color(2),palette_color(2),palette_color(8),palette_color(3),palette_color(7)]
    labels = ["both endpoints at minimum", "only forward", "only reverse",
              "neither at minimum", "missing data"]

    mat = np.zeros((len(cats), len(NOISES)))
    for i, noise in enumerate(NOISES):
        g = m[m["noise_pm"] == noise]
        fwd_min = (g["forward_n_neg_vib"] == 0)
        rev_min = (g["reverse_n_neg_vib"] == 0)
        fwd_miss = g["forward_n_neg_vib"].isna()
        rev_miss = g["reverse_n_neg_vib"].isna()
        miss = fwd_miss | rev_miss
        fwd_min = fwd_min & ~fwd_miss
        rev_min = rev_min & ~rev_miss

        both   = (fwd_min & rev_min & ~miss).sum()
        f_only = (fwd_min & ~rev_min & ~miss).sum()
        r_only = (~fwd_min & rev_min & ~miss).sum()
        nei    = (~fwd_min & ~rev_min & ~miss).sum()
        m_count = miss.sum()
        total = len(g)
        mat[:, i] = [100 * x / total for x in (both, f_only, r_only, nei, m_count)]

    bottom = np.zeros(len(NOISES))
    for i, (lab, col) in enumerate(zip(labels, colors)):
        ax.bar(range(len(NOISES)), mat[i], bottom=bottom, color=col, label=lab,
               edgecolor="white", linewidth=0.5)
        bottom += mat[i]
    ax.set_xticks(range(len(NOISES)))
    ax.set_xticklabels([f"{n}pm" for n in NOISES])
    ax.set_ylabel("fraction of runs (%)")
    ax.set_title("Endpoint vibrational quality — is IRC endpoint a true minimum? "
                 "($n_{\\text{neg}}=0$ on Eckart vib Hessian)")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=9)
    ax.set_ylim(0, 100)
    save(fig, "fig_sella_endpoint_quality")


def fig_rmsd_distributions(m: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    ok = m[m["topology_intended"]]
    bad = m[~m["topology_intended"]]
    for ax, col, title in zip(
        axes,
        ["rmsd_reactant","rmsd_product"],
        ["RMSD to labeled reactant (min over directions)",
         "RMSD to labeled product (min over directions)"],
    ):
        ok_vals = ok[col].dropna()
        bad_vals = bad[col].dropna()
        bins = np.linspace(0, max(ok_vals.max(), bad_vals.max(), 1.5), 60)
        ax.hist(ok_vals, bins=bins, color=palette_color(0), alpha=0.75,
                label=f"TOPO-intended (n={len(ok_vals)})", edgecolor="white")
        ax.hist(bad_vals, bins=bins, color=palette_color(3), alpha=0.75,
                label=f"TOPO-failed (n={len(bad_vals)})", edgecolor="white")
        ax.axvline(0.3, color=palette_color(7), ls="--", lw=1, label="RMSD threshold 0.3\\,\\AA")
        ax.set_xlabel("RMSD (\\AA)")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("count")
    fig.suptitle("Endpoint RMSD distributions — IRC outcome correlates with geometric proximity to label",
                 fontsize=11)
    save(fig, "fig_sella_rmsd_distributions")


def fig_ts_quality_vs_outcome(m: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ok = m[m["topology_intended"]]
    bad = m[~m["topology_intended"]]
    # Log-scale converged_step because range is 50-2000
    ax.scatter(np.abs(ok["ts_final_eig0"]), ok["ts_converged_step"], s=8, alpha=0.35,
               color=palette_color(0), label=f"TOPO-intended (n={len(ok)})")
    ax.scatter(np.abs(bad["ts_final_eig0"]), bad["ts_converged_step"], s=15, alpha=0.75,
               color=palette_color(3), label=f"TOPO-failed (n={len(bad)})", marker="x")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("$|\\lambda_0|$ at TS (eV/\\AA$^2$) — saddle sharpness")
    ax.set_ylabel("GAD converged\\_step")
    ax.set_title("TS quality (from GAD) vs IRC outcome")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=9)
    save(fig, "fig_sella_ts_quality_vs_outcome")


def fig_systematic_failures(m: pd.DataFrame) -> None:
    # Pivot TOPO-intended across (sid, noise); find sids failing at >=5 noise levels
    pivot = m.pivot_table(
        index="sample_id", columns="noise_pm",
        values="topology_intended", aggfunc="first",
    )
    present = pivot.notna()
    failing = ((~pivot.fillna(False).astype(bool)) & present)
    fcount = failing.sum(axis=1)
    worst = fcount[fcount >= 5].sort_values(ascending=False)
    if len(worst) == 0:
        return

    sids = worst.index.tolist()
    heat = pivot.loc[sids]
    # Map: green = TOPO-int, red = fail, gray = absent
    colors = np.full((len(sids), len(NOISES)), 0.5)  # gray
    for i, sid in enumerate(sids):
        for j, noise in enumerate(NOISES):
            v = heat.loc[sid, noise] if noise in heat.columns else np.nan
            if pd.isna(v):
                colors[i, j] = 0.5
            elif v:
                colors[i, j] = 1.0
            else:
                colors[i, j] = 0.0

    fig, ax = plt.subplots(figsize=(6, 0.3 * len(sids) + 1.5))
    cmap = matplotlib.colors.ListedColormap([palette_color(3), palette_color(7), palette_color(2)])
    bounds = [-0.5, 0.25, 0.75, 1.5]
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
    ax.imshow(colors, aspect="auto", cmap=cmap, norm=norm)

    formulas = [m[m["sample_id"] == sid]["formula"].iloc[0] for sid in sids]
    ax.set_yticks(range(len(sids)))
    ax.set_yticklabels([f"sid {sid}  {f}" for sid, f in zip(sids, formulas)], fontsize=8)
    ax.set_xticks(range(len(NOISES)))
    ax.set_xticklabels([f"{n}pm" for n in NOISES], fontsize=9)
    ax.set_title(f"Systematic TOPO failures ({len(sids)} samples fail at $\\geq$ 5 noise levels)",
                 fontsize=10)

    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color=palette_color(2), label="TOPO-intended"),
        Patch(color=palette_color(3), label="TOPO-failed"),
        Patch(color=palette_color(7), label="not in sample"),
    ], bbox_to_anchor=(1.01, 0.5), loc="center left", fontsize=8)
    save(fig, "fig_sella_systematic_failures")


def fig_wall_time(m: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    data = [m[m["noise_pm"] == n]["wall_time_s"].values for n in NOISES]
    bp = ax.boxplot(data, positions=range(len(NOISES)), widths=0.6,
                    patch_artist=True, showfliers=True)
    for patch in bp["boxes"]:
        patch.set_facecolor(palette_color(0))
        patch.set_alpha(0.65)
    ax.set_xticks(range(len(NOISES)))
    ax.set_xticklabels([f"{n}pm" for n in NOISES])
    ax.set_ylabel("wall time / sample (s, fwd+rev IRC)")
    ax.set_title("IRC wall-time per sample — sella\\_hip")
    ax.grid(alpha=0.3, axis="y")
    save(fig, "fig_sella_wall_time")


def main() -> None:
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 11,
        "text.usetex": False,
    })
    apply_plot_style()
    m = load()
    print(f"loaded {len(m)} rows")
    fig_rates(m)
    fig_endpoint_quality(m)
    fig_rmsd_distributions(m)
    fig_ts_quality_vs_outcome(m)
    fig_systematic_failures(m)
    fig_wall_time(m)


if __name__ == "__main__":
    main()
