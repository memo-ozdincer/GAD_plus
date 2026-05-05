#!/usr/bin/env python
"""Auto-discovering analyzer for the 2026-04-29 test-set comprehensive sweep.

Walks runs/test_set/ and runs/test_dtgrid/ and runs/test_reactant/ and
runs/test_irc/, prints a method-x-noise table for every threshold of
interest. Robust to missing cells.

Usage:
  python scripts/analyze_test.py
  python scripts/analyze_test.py --show-irc
  python scripts/analyze_test.py --threshold 1e-4   # strict (paper) criterion

Threshold options: 1e-4, 1e-3, 1e-2 (default), all (prints all 3).
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import duckdb

BASE = Path("/lustre07/scratch/memoozd/gadplus/runs")
NOISES = [0, 1, 5, 10, 30, 50, 100, 150, 200]


def detect_n_samples(path):
    if not os.path.exists(path):
        return None
    return duckdb.execute(f"SELECT COUNT(*) FROM '{path}'").fetchone()[0]


def detect_conv_col(path):
    if not os.path.exists(path):
        return None
    cols = set(duckdb.execute(f"DESCRIBE SELECT * FROM '{path}'").df()["column_name"])
    if "converged" in cols and "conv_nneg1_fmax001" not in cols:
        return "converged"  # GAD style
    return None


def rate(path, conv_col):
    if path is None or not os.path.exists(path):
        return None, 0
    r = duckdb.execute(
        f"SELECT 100.0*AVG(CAST({conv_col} AS DOUBLE)), COUNT(*) FROM '{path}'"
    ).fetchone()
    return r[0], r[1]


def sella_rate(path, criterion="fmax001"):
    """For Sella summaries, conv_nneg1_<criterion> columns exist."""
    if path is None or not os.path.exists(path):
        return None, 0
    col = f"conv_nneg1_{criterion}"
    r = duckdb.execute(
        f"SELECT 100.0*AVG(CAST({col} AS DOUBLE)), COUNT(*) FROM '{path}'"
    ).fetchone()
    return r[0], r[1]


def find_summary(method_dir, noise_pm):
    """Find summary parquet for a (method_dir, noise) cell."""
    if not os.path.exists(method_dir):
        return None
    candidates = [f for f in os.listdir(method_dir)
                  if f.startswith("summary") and f.endswith(".parquet") and f"_{noise_pm}pm" in f]
    if candidates:
        return os.path.join(method_dir, candidates[0])
    return None


def print_table(title, method_paths, conv_fn, noises=None):
    if noises is None:
        noises = NOISES
    print(f"\n{title}")
    print("=" * 78)
    print(f"{'method':<32}  " + "  ".join(f"{n:>5}" for n in noises))
    print("-" * (32 + 7 * len(noises)))
    for label, dirpath in method_paths.items():
        rates = []
        for n in noises:
            p = find_summary(dirpath, n)
            r, _ = conv_fn(p)
            rates.append(f"{r:5.1f}" if r is not None else "  --")
        print(f"{label:<32}  " + "  ".join(rates))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--show-irc", action="store_true")
    ap.add_argument("--threshold", choices=["fmax001", "fmax003", "force001", "force003", "all"],
                    default="fmax001")
    args = ap.parse_args()

    # ============ TS-converged tables ============
    gad_methods = {}
    for d in sorted(os.listdir(BASE / "test_set")):
        if d.startswith("gad_"):
            gad_methods[d] = str(BASE / "test_set" / d)
    for d in sorted(os.listdir(BASE / "test_dtgrid")) if (BASE / "test_dtgrid").exists() else []:
        gad_methods[d + " (5000 steps)"] = str(BASE / "test_dtgrid" / d)

    sella_methods = {}
    for d in sorted(os.listdir(BASE / "test_set")):
        if d.startswith("sella_"):
            sella_methods[d] = str(BASE / "test_set" / d)

    react_methods = {}
    if (BASE / "test_reactant").exists():
        for d in sorted(os.listdir(BASE / "test_reactant")):
            react_methods[d] = str(BASE / "test_reactant" / d)

    print_table("GAD on test (n=287, fmax<0.01 ∧ n_neg=1)", gad_methods,
                lambda p: rate(p, "converged"))
    print_table("Sella on test (n=287, fmax<0.01 ∧ n_neg=1)", sella_methods,
                lambda p: sella_rate(p, "fmax001"))
    if react_methods:
        print_table("FROM REACTANTS on test (single-ended)", react_methods,
                    lambda p: rate(p, "converged") if p and "gad_" in p else sella_rate(p, "fmax001"),
                    noises=[0])  # only one noise level, but pos_reactant is the start

    if args.threshold == "all":
        print_table("Sella on test, fmax<0.001 strict", sella_methods,
                    lambda p: sella_rate(p, "fmax001"))  # closest column

    # ============ Steps used (when converged) ============
    print("\nGAD median steps when converged (test_set/test_dtgrid)")
    print("=" * 78)
    print(f"{'method':<32}  " + "  ".join(f"{n:>5}" for n in [10, 30, 50, 100, 150, 200]))
    for label, dirpath in gad_methods.items():
        meds = []
        for n in [10, 30, 50, 100, 150, 200]:
            p = find_summary(dirpath, n)
            if not p: meds.append("  --"); continue
            try:
                r = duckdb.execute(f"SELECT median(total_steps) FROM '{p}' WHERE converged").fetchone()
                meds.append(f"{int(r[0]):5d}" if r[0] else "  --")
            except Exception:
                meds.append("  --")
        print(f"{label:<32}  " + "  ".join(meds))

    # ============ IRC validation if requested ============
    if args.show_irc and (BASE / "test_irc").exists():
        print("\n\nIRC TOPO-intended on test")
        print("=" * 78)
        for d in sorted(os.listdir(BASE / "test_irc")):
            irc_dir = str(BASE / "test_irc" / d)
            if not os.path.exists(irc_dir): continue
            files = [f for f in os.listdir(irc_dir) if f.endswith(".parquet")]
            if not files: continue
            try:
                df = duckdb.execute(f"SELECT noise_pm, AVG(CAST(topology_intended AS DOUBLE)) AS r FROM '{irc_dir}/*.parquet' GROUP BY noise_pm ORDER BY noise_pm").df()
                rates_str = "  ".join(f"{r['r']*100:5.1f}" for _, r in df.iterrows())
                print(f"{d:<32}  {rates_str}")
            except Exception as e:
                print(f"{d:<32}  err: {e}")


if __name__ == "__main__":
    main()
