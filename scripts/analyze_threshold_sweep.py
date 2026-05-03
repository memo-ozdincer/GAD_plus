#!/usr/bin/env python
"""Comprehensive convergence-criterion sweep on test data.

For every method × noise level, recomputes the convergence rate under
multiple criteria using the final-state columns:
  - fmax<{1e-2, 5e-3, 1e-3, 1e-4}  ∧ n_neg=1
  - force_norm<{1e-2, 5e-3, 1e-3, 1e-4} ∧ n_neg=1

Sella canonical = fmax<0.01 (Sella repo default). Other thresholds for
sensitivity analysis.

Output:
  analysis_2026_04_29/threshold_sweep_table.md
  analysis_2026_04_29/threshold_sweep.csv
"""
from __future__ import annotations

import os
from pathlib import Path

import duckdb
import pandas as pd

BASE = Path("/lustre07/scratch/memoozd/gadplus/runs")
OUT = Path("/lustre06/project/6033559/memoozd/GAD_plus/analysis_2026_04_29")
OUT.mkdir(exist_ok=True, parents=True)

NOISES = [10, 30, 50, 100, 150, 200]
# Wider grid: 0.05 = Sella library default; 0.01 = our canonical;
# tighter values for paper-grade strictness.
THRESHOLDS = [0.05, 0.01, 0.005, 0.001, 1e-4]


def find_summary(mdir, noise_pm):
    if not mdir.exists(): return None
    cands = [f for f in os.listdir(mdir)
             if f.startswith("summary") and f.endswith(".parquet") and f"_{noise_pm}pm" in f]
    return mdir / cands[0] if cands else None


def get_metrics(path):
    """Return DataFrame with normalized columns: sample_id, n_neg, fmax, force_norm,
    total_steps, wall_time_s, n_calls."""
    if not path: return None
    cols = set(duckdb.execute(f"DESCRIBE SELECT * FROM '{path}'").df()["column_name"])
    fmax_col = "final_fmax" if "final_fmax" in cols else "final_force_max"
    nfev_expr = "n_func_evals" if "n_func_evals" in cols else "total_steps"
    return duckdb.execute(f"""
        SELECT sample_id, final_n_neg AS n_neg,
               {fmax_col} AS fmax,
               final_force_norm AS force_norm,
               total_steps, wall_time_s,
               {nfev_expr} AS n_calls
        FROM '{path}'
    """).df()


def main():
    all_rows = []

    method_specs = {
        "GAD dt=0.003 (2k)":  BASE / "test_set/gad_dt003_fmax",
        "GAD dt=0.005 (2k)":  BASE / "test_set/gad_dt005_fmax",
        "GAD dt=0.003 (5k)":  BASE / "test_dtgrid/gad_dt003_fmax",
        "GAD dt=0.004 (5k)":  BASE / "test_dtgrid/gad_dt004_fmax",
        "GAD dt=0.005 (5k)":  BASE / "test_dtgrid/gad_dt005_fmax",
        "GAD dt=0.006 (5k)":  BASE / "test_dtgrid/gad_dt006_fmax",
        "GAD dt=0.007 (5k)":  BASE / "test_dtgrid/gad_dt007_fmax",
        "GAD dt=0.008 (5k)":  BASE / "test_dtgrid/gad_dt008_fmax",
        "GAD adaptive_dt":    BASE / "test_set/gad_adaptive_dt",
        "Sella default":      BASE / "test_set/sella_carteck_default",
        "Sella libdef":       BASE / "test_set/sella_carteck_libdef",
        "Sella internal":     BASE / "test_set/sella_internal_default",
        "Sella nohess cart":  BASE / "test_set/sella_carteck_nohess",
        "Sella nohess int":   BASE / "test_set/sella_internal_nohess",
    }

    for label, mdir in method_specs.items():
        for n in NOISES:
            p = find_summary(mdir, n)
            if not p: continue
            try:
                df = get_metrics(p)
            except Exception as e:
                print(f"err {label} {n}pm: {e}"); continue
            if df is None or not len(df): continue

            n_total = len(df)
            for thresh in THRESHOLDS:
                # With saddle requirement
                fmax_pass = ((df["n_neg"] == 1) & (df["fmax"] < thresh)).sum()
                fnorm_pass = ((df["n_neg"] == 1) & (df["force_norm"] < thresh)).sum()
                # WITHOUT saddle requirement (Sella's own default convergence: just fmax)
                fmax_pass_no_sad = (df["fmax"] < thresh).sum()
                fnorm_pass_no_sad = (df["force_norm"] < thresh).sum()
                all_rows.append({
                    "method": label, "noise_pm": n, "n_total": n_total,
                    "threshold": thresh,
                    "conv_fmax_pct": 100 * fmax_pass / n_total,
                    "conv_fnorm_pct": 100 * fnorm_pass / n_total,
                    "conv_fmax_nosad_pct": 100 * fmax_pass_no_sad / n_total,
                    "conv_fnorm_nosad_pct": 100 * fnorm_pass_no_sad / n_total,
                    "median_steps": float(df["total_steps"].median()),
                    "median_wall_s": float(df["wall_time_s"].median()),
                    "median_calls": float(df["n_calls"].median()),
                })

    out = pd.DataFrame(all_rows)
    out.to_csv(OUT / "threshold_sweep.csv", index=False)
    print(f"wrote {OUT / 'threshold_sweep.csv'}")

    # Summary tables — one per threshold, fmax then fnorm, with and without saddle req
    md_lines = ["# Convergence-criterion sensitivity sweep on test\n"]
    gate_options = [
        ("conv_fmax_pct", "fmax", "n_neg=1 ∧ "),
        ("conv_fnorm_pct", "force_norm", "n_neg=1 ∧ "),
        ("conv_fmax_nosad_pct", "fmax", ""),  # Sella library default style
        ("conv_fnorm_nosad_pct", "force_norm", ""),
    ]
    for gate, force_label, sad_prefix in gate_options:
        for thresh in THRESHOLDS:
            md_lines.append(f"\n## {sad_prefix}{force_label} < {thresh}\n")
            md_lines.append("| method | 10 | 30 | 50 | 100 | 150 | 200 |")
            md_lines.append("|---" * 7 + "|")
            for label in method_specs:
                row = [f"| {label} "]
                for n in NOISES:
                    sub = out[(out["method"] == label) & (out["noise_pm"] == n) & (out["threshold"] == thresh)]
                    if len(sub):
                        row.append(f"| {sub.iloc[0][gate]:.1f} ")
                    else:
                        row.append("| — ")
                row.append("|")
                md_lines.append("".join(row))

    # Compute / efficiency table
    md_lines.append("\n\n## Compute (median per-sample, all attempts incl. unconverged)\n")
    md_lines.append("| method | 10 | 30 | 50 | 100 | 150 | 200 |")
    md_lines.append("|---" * 7 + "|")
    for label in method_specs:
        row = [f"| {label} "]
        for n in NOISES:
            sub = out[(out["method"] == label) & (out["noise_pm"] == n) & (out["threshold"] == 0.01)]
            if len(sub):
                row.append(f"| {sub.iloc[0]['median_calls']:.0f} calls / {sub.iloc[0]['median_wall_s']:.1f}s ")
            else:
                row.append("| — ")
        row.append("|")
        md_lines.append("".join(row))

    (OUT / "threshold_sweep_table.md").write_text("\n".join(md_lines))
    print(f"wrote {OUT / 'threshold_sweep_table.md'}")

    # Print headline tables to stdout
    for thresh in [0.01, 0.001]:
        for gate in ["conv_fmax_pct", "conv_fnorm_pct"]:
            label = "fmax" if "fmax" in gate else "force_norm"
            print(f"\n=== n_neg=1 ∧ {label} < {thresh} ===")
            print(f"{'method':<26}  {'10':>5} {'30':>5} {'50':>5} {'100':>5} {'150':>5} {'200':>5}")
            for m in method_specs:
                rates = []
                for n in NOISES:
                    sub = out[(out["method"] == m) & (out["noise_pm"] == n) & (out["threshold"] == thresh)]
                    rates.append(f"{sub.iloc[0][gate]:5.1f}" if len(sub) else "  --")
                print(f"{m:<26}  {' '.join(rates)}")


if __name__ == "__main__":
    main()
