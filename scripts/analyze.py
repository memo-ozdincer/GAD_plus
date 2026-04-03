#!/usr/bin/env python
"""DuckDB analysis of GADplus experiment results.

Reads all Parquet files from experiment output directories and computes
aggregate statistics: convergence rates, failure distributions, timing.

Usage:
    python scripts/analyze.py /scratch/memoozd/gadplus/runs/20250403_*
    python scripts/analyze.py --results-dir /scratch/memoozd/gadplus/runs
"""
from __future__ import annotations

import argparse
import sys

import duckdb


def success_by_method_and_noise(results_glob: str):
    """Convergence rate broken down by search method and noise level."""
    query = f"""
    SELECT
        search_method,
        start_method,
        COUNT(*) as total,
        SUM(CASE WHEN converged THEN 1 ELSE 0 END) as converged,
        ROUND(100.0 * SUM(CASE WHEN converged THEN 1 ELSE 0 END) / COUNT(*), 1) as rate_pct,
        ROUND(AVG(CASE WHEN converged THEN total_steps END), 0) as avg_steps,
        ROUND(AVG(CASE WHEN converged THEN wall_time_s END), 1) as avg_time_s
    FROM '{results_glob}'
    GROUP BY search_method, start_method
    ORDER BY search_method, start_method
    """
    return duckdb.execute(query).df()


def failure_autopsy(results_glob: str):
    """Distribution of failure types by search method."""
    query = f"""
    SELECT
        search_method,
        failure_type,
        COUNT(*) as count,
        ROUND(AVG(final_force_norm), 4) as avg_force,
        ROUND(AVG(final_n_neg), 1) as avg_n_neg
    FROM '{results_glob}'
    WHERE NOT converged AND failure_type IS NOT NULL
    GROUP BY search_method, failure_type
    ORDER BY search_method, count DESC
    """
    return duckdb.execute(query).df()


def hardest_samples(results_glob: str, top_k: int = 20):
    """Samples with lowest convergence rate across all methods."""
    query = f"""
    SELECT
        formula,
        sample_id,
        COUNT(*) as total_runs,
        SUM(CASE WHEN converged THEN 1 ELSE 0 END) as n_converged,
        ROUND(100.0 * SUM(CASE WHEN converged THEN 1 ELSE 0 END) / COUNT(*), 1) as rate_pct
    FROM '{results_glob}'
    GROUP BY formula, sample_id
    ORDER BY rate_pct ASC, total_runs DESC
    LIMIT {top_k}
    """
    return duckdb.execute(query).df()


def eigenvalue_evolution(traj_glob: str, run_id: str):
    """Eigenvalue trajectory for a specific run."""
    query = f"""
    SELECT step, n_neg, eig0, eig1, force_norm, mode_overlap, phase
    FROM '{traj_glob}'
    WHERE run_id = '{run_id}'
    ORDER BY step
    """
    return duckdb.execute(query).df()


def main():
    parser = argparse.ArgumentParser(description="Analyze GADplus experiment results")
    parser.add_argument("results_dir", nargs="?", default="/scratch/memoozd/gadplus/runs")
    parser.add_argument("--top-k", type=int, default=20, help="Top K hardest samples")
    args = parser.parse_args()

    summary_glob = f"{args.results_dir}/*/summary_*.parquet"
    traj_glob = f"{args.results_dir}/*/traj_*.parquet"

    print("=" * 70)
    print("GADplus Experiment Analysis")
    print("=" * 70)

    print("\n--- Convergence by Method & Noise ---")
    try:
        df = success_by_method_and_noise(summary_glob)
        print(df.to_string(index=False))
    except Exception as e:
        print(f"  No summary data found: {e}")

    print("\n--- Failure Autopsy ---")
    try:
        df = failure_autopsy(summary_glob)
        print(df.to_string(index=False))
    except Exception as e:
        print(f"  No failure data found: {e}")

    print(f"\n--- Top {args.top_k} Hardest Samples ---")
    try:
        df = hardest_samples(summary_glob, args.top_k)
        print(df.to_string(index=False))
    except Exception as e:
        print(f"  No data found: {e}")


if __name__ == "__main__":
    main()
