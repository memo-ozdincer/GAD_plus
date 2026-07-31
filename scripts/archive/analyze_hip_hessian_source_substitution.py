#!/usr/bin/env python
"""Combine HIP Hessian-substitution smoke shards into an auditable summary."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dirs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    for run_dir in args.run_dirs:
        path = run_dir / "summary.parquet"
        if not path.is_file():
            raise FileNotFoundError(path)
        rows.extend(pq.read_table(path).to_pylist())
    rows.sort(key=lambda row: (row["source"], row["method"]))
    energy_errors = [row["initial_energy_abs_difference_eV"] for row in rows]
    force_errors = [row["initial_force_abs_difference_eV_per_A"] for row in rows]
    summary = {
        "n_rows": len(rows),
        "max_initial_energy_abs_difference_eV": max(energy_errors),
        "max_initial_force_abs_difference_eV_per_A": max(force_errors),
        "outcomes": [
            {
                key: row[key]
                for key in (
                    "source", "method", "reported_converged", "strict_converged",
                    "steps", "final_fmax_eV_per_A", "final_n_neg", "wall_time_s", "error",
                )
            }
            for row in rows
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
