#!/usr/bin/env python3
"""Summarize the predeclared pointwise-GAD failure diagnostics.

The script does not rerun a calculator or infer an endpoint.  It reads the
immutable task records and (for the long-budget panel) the recorded local
trajectory bundles.  Thus the 1k/3k/10k checkpoints are observations of the
same pointwise trajectory, not separate restarts.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


def _rows(root: Path) -> dict[int, dict[str, Any]]:
    return {
        int(row["sample_id"]): row
        for path in sorted((root / "tasks").glob("task_*.json"))
        if (row := json.loads(path.read_text()))
    }


def _counts(rows: Iterable[dict[str, Any]]) -> dict[str, int]:
    records = list(rows)
    return {
        "tasks": len(records),
        "calculator_valid": sum(not bool(row.get("error")) for row in records),
        "terminal_index0": sum(row.get("final_n_neg") == 0 for row in records),
        "terminal_index1": sum(row.get("final_n_neg") == 1 for row in records),
        "search_gate": sum(bool(row.get("search_gate")) for row in records),
        "strict_gate": sum(bool(row.get("strict_gate")) for row in records),
        "calculator_errors": sum(bool(row.get("error")) for row in records),
    }


def _checkpoint(row: dict[str, Any], limit: int) -> dict[str, Any] | None:
    bundle = row.get("trajectory_bundle")
    if not bundle:
        return None
    path = Path(bundle) / "trajectory.parquet"
    if not path.is_file():
        return None
    records = pq.read_table(path).to_pylist()
    observed = [record for record in records if int(record["iteration"]) <= limit]
    return observed[-1] if observed else None


def _first_gate_iteration(row: dict[str, Any], force_limit: float) -> int | None:
    bundle = row.get("trajectory_bundle")
    if not bundle:
        return None
    path = Path(bundle) / "trajectory.parquet"
    if not path.is_file():
        return None
    for record in pq.read_table(path, columns=["iteration", "n_neg", "force_max"]).to_pylist():
        if int(record["n_neg"]) == 1 and float(record["force_max"]) < force_limit:
            return int(record["iteration"])
    return None


def _extension_summary(rows: dict[int, dict[str, Any]], ids: list[int]) -> dict[str, Any]:
    subset = [rows[sample_id] for sample_id in ids]
    summary: dict[str, Any] = {"terminal": _counts(subset), "checkpoints": {}}
    for limit in (1000, 3000, 10000):
        checkpoints = [value for row in subset if (value := _checkpoint(row, limit)) is not None]
        summary["checkpoints"][str(limit)] = {
            "observed": len(checkpoints),
            "index1_and_fmax_lt_0p03": sum(
                int(value["n_neg"]) == 1 and float(value["force_max"]) < 0.03
                for value in checkpoints
            ),
            "index1_and_fmax_lt_0p01": sum(
                int(value["n_neg"]) == 1 and float(value["force_max"]) < 0.01
                for value in checkpoints
            ),
            "index0": sum(int(value["n_neg"]) == 0 for value in checkpoints),
            "median_fmax": (
                float(
                    sorted(float(value["force_max"]) for value in checkpoints)[
                        len(checkpoints) // 2
                    ]
                )
                if checkpoints
                else None
            ),
        }
    for force_limit, label in ((0.03, "first_local_iteration"), (0.01, "first_strict_iteration")):
        hits = [
            value
            for row in subset
            if (value := _first_gate_iteration(row, force_limit)) is not None
        ]
        summary[label] = {
            "count": len(hits),
            "median": sorted(hits)[len(hits) // 2] if hits else None,
            "values": {
                str(sample_id): _first_gate_iteration(rows[sample_id], force_limit)
                for sample_id in ids
            },
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--minimum-root", type=Path, required=True)
    parser.add_argument("--extension-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    baseline = _rows(args.minimum_root / "competitive")
    subspace = _rows(args.minimum_root / "competitive_subspace")
    pairs = []
    for sample_id in sorted(set(baseline) | set(subspace)):
        before, after = baseline.get(sample_id, {}), subspace.get(sample_id, {})
        pairs.append(
            {
                "sample_id": sample_id,
                "baseline_n_neg": before.get("final_n_neg"),
                "subspace_n_neg": after.get("final_n_neg"),
                "baseline_fmax": before.get("final_fmax"),
                "subspace_fmax": after.get("final_fmax"),
                "baseline_search_gate": bool(before.get("search_gate")),
                "subspace_search_gate": bool(after.get("search_gate")),
                "baseline_error": before.get("error", ""),
                "subspace_error": after.get("error", ""),
            }
        )
    prevented = sum(pair["baseline_n_neg"] == 0 and pair["subspace_n_neg"] != 0 for pair in pairs)

    extension_rows = _rows(args.extension_root)
    high_force_ids = [
        59,
        64,
        80,
        85,
        116,
        120,
        123,
        127,
        131,
        160,
        166,
        197,
        208,
        213,
        218,
        221,
        234,
        246,
        273,
        279,
        281,
        282,
    ]
    control_ids = [19, 79]
    report = {
        "minimum_prevention": {
            "baseline": _counts(baseline.values()),
            "competitive_subspace": _counts(subspace.values()),
            "index0_prevented": prevented,
            "paired_rows": pairs,
        },
        "force_extension": {
            "original_high_force_22": _extension_summary(extension_rows, high_force_ids),
            "near_threshold_controls_2": _extension_summary(extension_rows, control_ids),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
