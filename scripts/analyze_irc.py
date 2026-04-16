#!/usr/bin/env python
"""Summarize IRC validation parquets. Produces:

  1. Per-(method, noise) summary table (stdout + optional CSV).
  2. One stacked bar chart per noise level: intended / half / unintended
     proportions by method. Saved as PNG into the output directory.
  3. Text notes on unintended runs: how many landed at a valid minimum
     (n_neg_vib == 0) but not the intended one, and the closest-miss
     RMSDs — focused on the smallest min(forward/reverse, reactant/product)
     RMSDs so you can see how far off the "wrong minimum" really is.

Works on partial results. Safe to run mid-experiment.

Usage:
  python scripts/analyze_irc.py <dir1> [dir2 ...]
  python scripts/analyze_irc.py /lustre07/scratch/memoozd/gadplus/runs/irc_sellahip_rigorous
"""
from __future__ import annotations

import argparse
import os
import sys

import duckdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


_COLORS = {
    "intended": "#2ca02c",      # green
    "half": "#ff7f0e",          # orange
    "unintended": "#d62728",    # red
}


def _bar_chart_per_noise(df, out_dir):
    """One PNG per noise level. X-axis = method, stacked bars."""
    noise_levels = sorted(df["noise_pm"].unique())
    saved = []
    for noise in noise_levels:
        sub = df[df["noise_pm"] == noise].sort_values("method").reset_index(drop=True)
        methods = sub["method"].tolist()
        n = sub["n"].astype(float).values
        pct_int = 100.0 * sub["n_intended"].astype(float).values / np.maximum(n, 1)
        pct_half = 100.0 * sub["n_half"].astype(float).values / np.maximum(n, 1)
        pct_err = 100.0 * sub["n_error"].astype(float).values / np.maximum(n, 1)
        pct_unint = np.clip(100.0 - pct_int - pct_half - pct_err, 0, 100)

        fig, ax = plt.subplots(figsize=(max(4.0, 1.6 * len(methods) + 2), 4.5))
        x = np.arange(len(methods))
        width = 0.6
        ax.bar(x, pct_int, width, color=_COLORS["intended"], label="intended")
        ax.bar(x, pct_half, width, bottom=pct_int, color=_COLORS["half"], label="half")
        ax.bar(x, pct_unint, width, bottom=pct_int + pct_half,
               color=_COLORS["unintended"], label="unintended")
        if pct_err.max() > 0:
            ax.bar(x, pct_err, width, bottom=pct_int + pct_half + pct_unint,
                   color="#7f7f7f", label="error")

        for i, (pi, ph, pu, nn) in enumerate(zip(
                pct_int, pct_half, pct_unint, sub["n"])):
            total_n = int(nn)
            ax.text(i, 101, f"N={total_n}", ha="center", va="bottom", fontsize=9)
            if pi > 4:
                ax.text(i, pi / 2, f"{pi:.0f}%", ha="center", va="center",
                        fontsize=9, color="white")
            if ph > 4:
                ax.text(i, pi + ph / 2, f"{ph:.0f}%", ha="center", va="center",
                        fontsize=9, color="white")
            if pu > 4:
                ax.text(i, pi + ph + pu / 2, f"{pu:.0f}%", ha="center", va="center",
                        fontsize=9, color="white")

        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=0)
        ax.set_ylabel("% of samples")
        ax.set_ylim(0, 110)
        ax.set_title(f"IRC endpoint matching — noise={noise}pm")
        ax.legend(loc="upper right", frameon=False, fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()

        out_path = os.path.join(out_dir, f"irc_bars_{noise}pm.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        saved.append(out_path)
    return saved


def _unintended_text_notes(source_sql, con):
    """For each (method, noise), count unintended runs where both endpoints
    landed at valid minima (n_neg_vib == 0 on both) and list the closest
    RMSD misses.
    """
    query = f"""
        WITH u AS (
            SELECT
                method,
                noise_pm,
                sample_id,
                formula,
                forward_n_neg_vib,
                reverse_n_neg_vib,
                forward_min_vib_eig,
                reverse_min_vib_eig,
                LEAST(
                    COALESCE(forward_rmsd_reactant, 1e9),
                    COALESCE(forward_rmsd_product,  1e9),
                    COALESCE(reverse_rmsd_reactant, 1e9),
                    COALESCE(reverse_rmsd_product,  1e9)
                ) AS min_rmsd_any,
                forward_rmsd_reactant, forward_rmsd_product,
                reverse_rmsd_reactant, reverse_rmsd_product
            FROM {source_sql}
            WHERE intended = FALSE AND half_intended = FALSE AND error IS NULL
        )
        SELECT * FROM u ORDER BY method, noise_pm, min_rmsd_any
    """
    return con.execute(query).df()


def _print_unintended_notes(unint_df):
    print()
    print("=" * 78)
    print("Unintended runs — endpoint geometry diagnostics")
    print("=" * 78)
    if len(unint_df) == 0:
        print("(no unintended runs in the dataset)")
        return

    for (method, noise), group in unint_df.groupby(["method", "noise_pm"]):
        n_total = len(group)
        f_nneg = group["forward_n_neg_vib"]
        r_nneg = group["reverse_n_neg_vib"]

        both_min = int(((f_nneg == 0) & (r_nneg == 0)).sum())
        one_min = int(((f_nneg == 0) | (r_nneg == 0)).sum()) - both_min
        neither_min = n_total - both_min - one_min
        missing = int((f_nneg.isna() | r_nneg.isna()).sum())

        print()
        print(f"[{method} @ {noise}pm]  {n_total} unintended")
        print(f"  both endpoints at valid minimum (n_neg=0 on both): {both_min}")
        print(f"  one endpoint at valid minimum:                       {one_min}")
        print(f"  neither endpoint at valid minimum:                   {neither_min}")
        if missing:
            print(f"  (missing spectral data on one+ endpoint: {missing})")

        k = min(5, n_total)
        print(f"  Closest-miss top {k} samples (lowest any-to-any RMSD):")
        top = group.nsmallest(k, "min_rmsd_any")
        for _, row in top.iterrows():
            sid = int(row["sample_id"])
            f_nn = row["forward_n_neg_vib"]
            r_nn = row["reverse_n_neg_vib"]
            f_min = row["forward_min_vib_eig"]
            r_min = row["reverse_min_vib_eig"]
            fr = row["forward_rmsd_reactant"]
            fp = row["forward_rmsd_product"]
            rr = row["reverse_rmsd_reactant"]
            rp = row["reverse_rmsd_product"]

            def _fmt_rmsd(v):
                return f"{v:.3f}" if v is not None and not np.isnan(v) else "  -  "

            def _fmt_eig(v):
                return f"{v:+.4f}" if v is not None and not np.isnan(v) else "   -   "

            print(
                f"    sid={sid:>4}  formula={row['formula']:>12}  "
                f"n_neg=[f={f_nn if not np.isnan(f_nn) else '-'}, "
                f"r={r_nn if not np.isnan(r_nn) else '-'}]  "
                f"min_eig=[f={_fmt_eig(f_min)}, r={_fmt_eig(r_min)}]  "
                f"RMSDs (fR/fP/rR/rP)={_fmt_rmsd(fr)}/{_fmt_rmsd(fp)}"
                f"/{_fmt_rmsd(rr)}/{_fmt_rmsd(rp)}"
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dirs", nargs="+",
                        help="One or more directories containing irc_validation_*.parquet")
    parser.add_argument("--csv", type=str, default=None,
                        help="Optional CSV output for the summary table")
    parser.add_argument("--plots-dir", type=str, default=None,
                        help="Where to write bar-chart PNGs (default: first input dir)")
    args = parser.parse_args()

    globs = []
    for d in args.dirs:
        if not os.path.isdir(d):
            print(f"WARNING: not a directory: {d}", file=sys.stderr)
            continue
        globs.append(os.path.join(d, "irc_validation_*.parquet"))

    if not globs:
        sys.exit("No valid directories provided.")

    read_args = ", ".join(f"'{g}'" for g in globs)
    source_sql = f"read_parquet([{read_args}], union_by_name=true)"

    con = duckdb.connect(":memory:")

    total = con.execute(f"SELECT COUNT(*) FROM {source_sql}").fetchone()[0]
    if total == 0:
        sys.exit("No rows found in the specified directories.")
    print(f"Total rows: {total}")

    summary_sql = f"""
        SELECT
            method,
            noise_pm,
            COUNT(*) AS n,
            SUM(CASE WHEN intended           THEN 1 ELSE 0 END) AS n_intended,
            SUM(CASE WHEN half_intended      THEN 1 ELSE 0 END) AS n_half,
            SUM(CASE WHEN topology_intended  THEN 1 ELSE 0 END) AS n_topo_int,
            SUM(CASE WHEN topology_half_intended THEN 1 ELSE 0 END) AS n_topo_half,
            SUM(CASE WHEN error IS NOT NULL  THEN 1 ELSE 0 END) AS n_error,
            ROUND(100.0 * SUM(CASE WHEN intended          THEN 1 ELSE 0 END) / COUNT(*), 1) AS intended_pct,
            ROUND(100.0 * SUM(CASE WHEN topology_intended THEN 1 ELSE 0 END) / COUNT(*), 1) AS topo_int_pct,
            ROUND(100.0 * SUM(CASE WHEN error IS NOT NULL THEN 1 ELSE 0 END) / COUNT(*), 1) AS error_pct,
            ROUND(AVG(COALESCE(wall_time_s, 0)), 1) AS avg_wall_s
        FROM {source_sql}
        GROUP BY method, noise_pm
        ORDER BY method, noise_pm
    """
    summary_df = con.execute(summary_sql).df()

    print()
    print("Per-(method, noise) summary:")
    print(summary_df.to_string(index=False))

    if args.csv:
        summary_df.to_csv(args.csv, index=False)
        print(f"\nWrote summary CSV: {args.csv}")

    plots_dir = args.plots_dir or args.dirs[0]
    os.makedirs(plots_dir, exist_ok=True)
    saved = _bar_chart_per_noise(summary_df, plots_dir)
    if saved:
        print()
        print(f"Wrote {len(saved)} bar chart(s):")
        for p in saved:
            print(f"  {p}")

    unint_df = _unintended_text_notes(source_sql, con)
    _print_unintended_notes(unint_df)

    print()
    print("Errors observed (first 20, if any):")
    err_df = con.execute(f"""
        SELECT method, noise_pm, sample_id, error
        FROM {source_sql}
        WHERE error IS NOT NULL
        LIMIT 20
    """).df()
    if len(err_df) == 0:
        print("  (none)")
    else:
        print(err_df.to_string(index=False))


if __name__ == "__main__":
    main()
