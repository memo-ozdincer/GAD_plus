#!/usr/bin/env python
"""Aggregate a hybrid HPO grid into one clean table.

Reads every ``summary_*.parquet`` under ``runs/<grid_name>/`` and writes:

* ``aggregate.parquet`` / ``aggregate.csv`` — one row per cell, with all
                                              hyperparameters + stats.
* ``missing_cells.csv`` — task_ids + parameter sets without a summary yet.

Robust to partial completion: ``union_by_name=true`` makes missing cells
silently absent. Idempotent.

Usage:
    python scripts/aggregate_hybrid_grid.py                       # fmax01 grid
    python scripts/aggregate_hybrid_grid.py --grid big
    python scripts/aggregate_hybrid_grid.py --threshold-fmax 0.02 # post-hoc gate
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import duckdb
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_hybrid_grid_index import GRIDS  # noqa: E402


def aggregate(grid_root: Path, fmax_threshold: float) -> pd.DataFrame:
    glob = f"{grid_root}/**/summary_*.parquet"
    q = f"""
        SELECT
            method, noise_pm, trust_radius, gad_dt, switch_by_eig,
            COUNT(*)                                                   AS n,
            SUM(converged::INT)                                        AS n_conv,
            ROUND(100.0 * SUM(converged::INT) / COUNT(*), 2)           AS pct_conv,
            SUM(CASE WHEN final_n_neg = 1
                          AND final_force_max < {fmax_threshold}
                     THEN 1 ELSE 0 END)                                AS n_conv_at_thr,
            ROUND(100.0 * SUM(CASE WHEN final_n_neg = 1
                                        AND final_force_max < {fmax_threshold}
                                   THEN 1 ELSE 0 END) / COUNT(*), 2)   AS pct_conv_at_thr,
            ROUND(AVG(CASE WHEN converged THEN converged_step END), 1) AS avg_conv_step,
            ROUND(AVG(total_steps), 1)                                 AS avg_total_steps,
            ROUND(AVG(wall_time_s), 2)                                 AS avg_wall_s,
            ROUND(AVG(final_force_max), 4)                             AS avg_final_fmax,
            ROUND(AVG(final_force_norm), 4)                            AS avg_final_fnorm,
            ROUND(AVG(final_eig0), 3)                                  AS avg_final_eig0,
            SUM(CASE WHEN final_n_neg = 1 THEN 1 ELSE 0 END)           AS n_nneg1
        FROM read_parquet('{glob}', union_by_name=true)
        GROUP BY method, noise_pm, trust_radius, gad_dt, switch_by_eig
        ORDER BY noise_pm, pct_conv DESC
    """
    return duckdb.sql(q).df()


def find_missing(grid_root: Path) -> pd.DataFrame:
    idx_path = grid_root / "grid_index.csv"
    if not idx_path.exists():
        print(f"WARN: grid_index.csv not at {idx_path}. "
              "Run scripts/build_hybrid_grid_index.py.", file=sys.stderr)
        return pd.DataFrame()
    idx = pd.read_csv(idx_path)
    idx["has_summary"] = idx["summary_path"].apply(lambda p: Path(p).exists())
    return idx.loc[~idx["has_summary"]].copy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", default="fmax01", choices=sorted(GRIDS.keys()))
    parser.add_argument("--threshold-fmax", type=float, default=None,
                        help="Post-hoc fmax threshold for the _at_thr columns. "
                             "Default = the grid's own convergence threshold.")
    args = parser.parse_args()

    spec = GRIDS[args.grid]
    if not spec.root.exists():
        sys.exit(f"Grid dir not found: {spec.root}")

    thr = args.threshold_fmax if args.threshold_fmax is not None else spec.force_threshold
    print(f"Scanning {spec.root} (post-hoc fmax threshold: {thr})")

    agg = aggregate(spec.root, thr)
    print(f"Aggregated {len(agg)} cells")

    out_parquet = spec.root / "aggregate.parquet"
    out_csv     = spec.root / "aggregate.csv"
    agg.to_parquet(out_parquet)
    agg.to_csv(out_csv, index=False)
    print(f"  wrote {out_parquet}")
    print(f"  wrote {out_csv}")

    missing = find_missing(spec.root)
    miss_csv = spec.root / "missing_cells.csv"
    if len(missing):
        missing.to_csv(miss_csv, index=False)
        print(f"  wrote {miss_csv} ({len(missing)}/{spec.array_size} cells pending)")
    else:
        print("  no missing cells")

    if len(agg):
        print()
        for noise in sorted(agg["noise_pm"].unique()):
            sub = agg[agg["noise_pm"] == noise].head(10)
            print(f"=== Top 10 by pct_conv @ {noise}pm ===")
            cols = ["method", "switch_by_eig", "gad_dt", "trust_radius",
                    "pct_conv", "avg_conv_step", "avg_wall_s"]
            print(sub[cols].to_string(index=False))
            print()


if __name__ == "__main__":
    main()
