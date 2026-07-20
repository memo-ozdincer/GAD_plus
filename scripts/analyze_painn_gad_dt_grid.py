#!/usr/bin/env python
"""Report the prespecified pure-GAD PaiNN timestep grid without cherry-picking."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows_by_dt: dict[float, list[dict]] = {}
    for manifest_path in sorted(args.grid_dir.glob("dt-*/manifest.json")):
        with manifest_path.open() as handle:
            manifest = json.load(handle)
        dt = float(manifest["gad"]["dt"])
        summary = manifest_path.with_name("summary.parquet")
        if not summary.is_file():
            raise FileNotFoundError(f"missing summary for dt={dt}: {summary}")
        rows = pq.read_table(summary).to_pylist()
        if not rows or any(row.get("method") != "gad" for row in rows):
            raise ValueError(f"dt={dt} does not contain a nonempty GAD-only screen")
        key = {(row["candidate_file"], row["noise_pm"], row["seed"]) for row in rows}
        if len(key) != len(rows):
            raise ValueError(f"dt={dt} contains duplicate GAD starts")
        rows_by_dt[dt] = rows

    if not rows_by_dt:
        raise FileNotFoundError(f"no completed dt manifests under {args.grid_dir}")
    expected_keys: set[tuple] | None = None
    report: dict[str, object] = {"dts": {}, "selection": "not performed"}
    for dt, rows in sorted(rows_by_dt.items()):
        keys = {(row["candidate_file"], row["noise_pm"], row["seed"]) for row in rows}
        if expected_keys is None:
            expected_keys = keys
        elif keys != expected_keys:
            raise ValueError("dt screens do not cover identical paired starts")
        by_noise: dict[str, dict[str, float | int]] = {}
        for noise in sorted({float(row["noise_pm"]) for row in rows}):
            group = [row for row in rows if float(row["noise_pm"]) == noise]
            success = sum(bool(row["strict_converged"]) for row in group)
            by_noise[str(noise)] = {
                "n_starts": len(group),
                "strict_successes": success,
                "strict_rate": success / len(group),
            }
        success = sum(bool(row["strict_converged"]) for row in rows)
        report["dts"][str(dt)] = {
            "n_starts": len(rows),
            "strict_successes": success,
            "strict_rate": success / len(rows),
            "by_noise_pm": by_noise,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
