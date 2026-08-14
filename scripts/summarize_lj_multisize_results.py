#!/usr/bin/env python3
"""Produce reproducible pooled and failure-taxonomy tables for an LJ run."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--expected-tasks", type=int, default=1344)
    return parser.parse_args()


def _median(values: list[float]) -> float | None:
    return float(statistics.median(values)) if values else None


def _quantile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _effective_count(labels: list[Any]) -> float | None:
    if not labels:
        return None
    counts = Counter(labels)
    total = len(labels)
    entropy = -sum((count / total) * math.log(count / total) for count in counts.values())
    return math.exp(entropy)


def _saddle_family_count(rows: list[dict[str, Any]]) -> tuple[int, float | None]:
    """Cluster strict saddles by permutation-invariant distances and energy."""
    if not rows:
        return 0, None
    parent = list(range(len(rows)))

    def find(item: int) -> int:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(first: int, second: int) -> None:
        first, second = find(first), find(second)
        if first != second:
            parent[second] = first

    for first in range(len(rows)):
        a = rows[first]
        for second in range(first + 1, len(rows)):
            b = rows[second]
            if abs(float(a["final_energy"]) - float(b["final_energy"])) > 1.0e-4:
                continue
            fa, fb = a["fingerprint"], b["fingerprint"]
            rms = math.sqrt(sum((x - y) ** 2 for x, y in zip(fa, fb, strict=True)) / len(fa))
            if rms <= 1.0e-3:
                union(first, second)
    labels = [find(index) for index in range(len(rows))]
    return len(set(labels)), _effective_count(labels)


def _failure_class(row: dict[str, Any]) -> str:
    if not row.get("calculator_valid"):
        return "calculator_error"
    if row.get("strict_ts"):
        return "strict_index1"
    n_neg = int(row["final_n_neg"])
    if n_neg == 0:
        return "index0"
    if n_neg == 1:
        return "index1_force_limited"
    return "multi_negative"


def _summarize(group: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in group if row.get("calculator_valid")]
    strict = [row for row in valid if row.get("strict_ts")]
    endpoints = [row for row in valid if row.get("endpoint_minima")]
    connected = [row for row in valid if row.get("connected_cutoff_1p5")]
    near_flat = [
        row for row in valid
        if abs(float(row.get("final_lambda2_scaled", math.inf))) < 0.01
    ]
    evaluations = [float(row["evaluations"]) for row in valid]
    strict_families, effective_strict_families = _saddle_family_count(strict)
    endpoint_pairs = [
        tuple(sorted(round(float(value) / 1.0e-4) for value in row["endpoint_energies"]))
        for row in endpoints if len(row.get("endpoint_energies", [])) == 2
    ]
    taxonomy = Counter(_failure_class(row) for row in group)
    return {
        "planned": len(group),
        "valid": len(valid),
        "strict": len(strict),
        "strict_rate_planned": len(strict) / len(group),
        "strict_rate_valid": len(strict) / len(valid) if valid else math.nan,
        "endpoint_minima": len(endpoints),
        "endpoint_rate_planned": len(endpoints) / len(group),
        "connected_final": len(connected),
        "fragmented_final": len(valid) - len(connected),
        "near_flat_abs_lambda2_scaled_lt_0p01": len(near_flat),
        "median_evaluations_valid": _median(evaluations),
        "p95_evaluations_valid": _quantile(evaluations, 0.95),
        "max_evaluations_valid": max(evaluations) if evaluations else None,
        "median_strict_evaluations": _median([float(row["evaluations"]) for row in strict]),
        "median_strict_wall_s": _median([float(row["wall_time_s"]) for row in strict]),
        "strict_saddle_families": strict_families,
        "effective_strict_saddle_families": effective_strict_families,
        "endpoint_energy_pairs": len(set(endpoint_pairs)),
        "effective_endpoint_energy_pairs": _effective_count(endpoint_pairs),
        "failure_taxonomy": dict(sorted(taxonomy.items())),
    }


def _group(rows: list[dict[str, Any]], fields: tuple[str, ...]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[field] for field in fields)].append(row)
    output = []
    for key, group in sorted(groups.items()):
        output.append({**dict(zip(fields, key, strict=True)), **_summarize(group)})
    return output


def _matched(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_start: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        key = (
            row["size"], row["basin"], row["start_family"],
            row["level_sigma"], row["sample_id"],
        )
        by_start[key][row["method"]] = row
    methods = sorted({row["method"] for row in rows})
    output = []
    for size in sorted({int(row["size"]) for row in rows}):
        starts = [value for key, value in by_start.items() if int(key[0]) == size]
        for left_index, left in enumerate(methods):
            for right in methods[left_index + 1:]:
                pairs = [pair for pair in starts if left in pair and right in pair]
                both_valid = [
                    pair for pair in pairs
                    if pair[left].get("calculator_valid") and pair[right].get("calculator_valid")
                ]
                left_only = sum(
                    bool(pair[left].get("strict_ts")) and not bool(pair[right].get("strict_ts"))
                    for pair in both_valid
                )
                right_only = sum(
                    bool(pair[right].get("strict_ts")) and not bool(pair[left].get("strict_ts"))
                    for pair in both_valid
                )
                output.append(
                    {
                        "size": size,
                        "left": left,
                        "right": right,
                        "matched_valid": len(both_valid),
                        "left_only_strict": left_only,
                        "right_only_strict": right_only,
                        "net_left_wins": left_only - right_only,
                    }
                )
    return output


def _mcnemar_exact(left_only: int, right_only: int) -> float | None:
    discordant = left_only + right_only
    if discordant == 0:
        return None
    tail = min(left_only, right_only)
    probability = sum(math.comb(discordant, value) for value in range(tail + 1)) / 2**discordant
    return min(1.0, 2.0 * probability)


def _matched_cs2_intrinsic(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    fields = ("size", "basin", "start_family", "level_sigma")
    by_cell: dict[tuple[Any, ...], dict[int, dict[str, dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for row in rows:
        if row["method"] in {"cs2", "intrinsic_lambda2"}:
            key = tuple(row[field] for field in fields)
            by_cell[key][int(row["sample_id"])][row["method"]] = row
    output = []
    mismatches = 0
    for key, starts in sorted(by_cell.items()):
        pairs = [
            pair for pair in starts.values()
            if {"cs2", "intrinsic_lambda2"}.issubset(pair)
        ]
        valid = [
            pair for pair in pairs
            if pair["cs2"].get("calculator_valid")
            and pair["intrinsic_lambda2"].get("calculator_valid")
        ]
        for pair in valid:
            left, right = pair["cs2"], pair["intrinsic_lambda2"]
            if (
                left["seed"] != right["seed"]
                or left["initial_n_neg"] != right["initial_n_neg"]
                or left["initial_energy"] != right["initial_energy"]
                or left["initial_fmax"] != right["initial_fmax"]
            ):
                mismatches += 1
        cs2_only = sum(
            bool(pair["cs2"].get("strict_ts"))
            and not bool(pair["intrinsic_lambda2"].get("strict_ts"))
            for pair in valid
        )
        intrinsic_only = sum(
            bool(pair["intrinsic_lambda2"].get("strict_ts"))
            and not bool(pair["cs2"].get("strict_ts"))
            for pair in valid
        )
        output.append(
            {
                **dict(zip(fields, key, strict=True)),
                "matched_valid": len(valid),
                "both_strict": sum(
                    bool(pair["cs2"].get("strict_ts"))
                    and bool(pair["intrinsic_lambda2"].get("strict_ts"))
                    for pair in valid
                ),
                "cs2_only_strict": cs2_only,
                "intrinsic_only_strict": intrinsic_only,
                "neither_strict": sum(
                    not bool(pair["cs2"].get("strict_ts"))
                    and not bool(pair["intrinsic_lambda2"].get("strict_ts"))
                    for pair in valid
                ),
                "net_cs2_wins": cs2_only - intrinsic_only,
                "mcnemar_exact_p_two_sided": _mcnemar_exact(cs2_only, intrinsic_only),
            }
        )
    return output, mismatches


def main() -> None:
    args = parse_args()
    result_path = args.run_root / "all_results.json"
    if not result_path.exists():
        raise FileNotFoundError(f"aggregate output is missing: {result_path}")
    rows = json.loads(result_path.read_text())
    if len(rows) != args.expected_tasks:
        raise RuntimeError(f"expected {args.expected_tasks} rows, found {len(rows)}")
    task_ids = [int(row["task_id"]) for row in rows]
    if sorted(task_ids) != list(range(args.expected_tasks)):
        raise RuntimeError("task IDs do not cover the exact expected range")

    matched_cs2_intrinsic, start_mismatches = _matched_cs2_intrinsic(rows)
    if start_mismatches:
        raise RuntimeError(f"found {start_mismatches} mismatched CS2/intrinsic starts")
    analysis = {
        "run_root": str(args.run_root.resolve()),
        "expected_tasks": args.expected_tasks,
        "method_size": _group(rows, ("method", "size")),
        "method_size_basin": _group(rows, ("method", "size", "basin")),
        "method_size_start_level": _group(
            rows, ("method", "size", "start_family", "level_sigma")
        ),
        "matched_pairwise": _matched(rows),
        "matched_cs2_vs_intrinsic_by_cell": matched_cs2_intrinsic,
        "matched_cs2_intrinsic_start_mismatches": start_mismatches,
    }
    (args.run_root / "analysis.json").write_text(json.dumps(analysis, indent=2) + "\n")

    lines = [
        "# Multi-size LJ pooled analysis",
        "",
        "## Pooled by size",
        "",
        "| Method | LJ | Valid | Strict | Two minima | Fragmented | Near-flat | Median/p95 evals | Families/pairs |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in analysis["method_size"]:
        median = row["median_evaluations_valid"]
        lines.append(
            f"| {row['method']} | {row['size']} | {row['valid']}/{row['planned']} | "
            f"{row['strict']}/{row['planned']} ({100 * row['strict_rate_planned']:.1f}%) | "
            f"{row['endpoint_minima']}/{row['planned']} | {row['fragmented_final']} | "
            f"{row['near_flat_abs_lambda2_scaled_lt_0p01']} | "
            f"{row['median_evaluations_valid']:.1f}/{row['p95_evaluations_valid']:.1f} | "
            f"{row['strict_saddle_families']}/{row['endpoint_energy_pairs']} |" if median is not None else
            f"| {row['method']} | {row['size']} | {row['valid']}/{row['planned']} | "
            f"{row['strict']}/{row['planned']} ({100 * row['strict_rate_planned']:.1f}%) | "
            f"{row['endpoint_minima']}/{row['planned']} | {row['fragmented_final']} | "
            f"{row['near_flat_abs_lambda2_scaled_lt_0p01']} | n/a | "
            f"{row['strict_saddle_families']}/{row['endpoint_energy_pairs']} |"
        )
    lines.extend(
        [
            "",
            "`Two minima` means two tightly relaxed local minimum endpoints; it does not by itself "
            "establish distinct topology or a particular database transition.",
            "`Families/pairs` are strict-saddle fingerprint families and rounded downhill endpoint-energy pairs.",
            "",
            "## Terminal-index taxonomy by size",
            "",
            "| Method | LJ | Strict index 1 | Index 0 | Index 1 force-limited | Multi-negative | Calculator error |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in analysis["method_size"]:
        taxonomy = row["failure_taxonomy"]
        lines.append(
            f"| {row['method']} | {row['size']} | {taxonomy.get('strict_index1', 0)} | "
            f"{taxonomy.get('index0', 0)} | {taxonomy.get('index1_force_limited', 0)} | "
            f"{taxonomy.get('multi_negative', 0)} | {taxonomy.get('calculator_error', 0)} |"
        )
    lines.extend(
        [
            "",
            "## Matched strict-success discordance",
            "",
            "| LJ | Left | Right | Matched valid | Left only | Right only | Net left |",
            "|---:|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in analysis["matched_pairwise"]:
        lines.append(
            f"| {row['size']} | {row['left']} | {row['right']} | "
            f"{row['matched_valid']} | {row['left_only_strict']} | "
            f"{row['right_only_strict']} | {row['net_left_wins']} |"
        )
    lines.extend(
        [
            "",
            "## CS2 versus intrinsic lambda2 by matched cell",
            "",
            "All rows below passed an exact equality check on seed and initial E/index/fmax.",
            "",
            "| LJ | Basin | Start | Level | Matched | Both | CS2 only | Intrinsic only | Neither | Net CS2 | McNemar p |",
            "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in analysis["matched_cs2_vs_intrinsic_by_cell"]:
        p_value = row["mcnemar_exact_p_two_sided"]
        lines.append(
            f"| {row['size']} | {row['basin']} | {row['start_family']} | "
            f"{row['level_sigma']:.2f} | {row['matched_valid']} | {row['both_strict']} | "
            f"{row['cs2_only_strict']} | {row['intrinsic_only_strict']} | "
            f"{row['neither_strict']} | {row['net_cs2_wins']} | "
            f"{p_value:.4g} |" if p_value is not None else
            f"| {row['size']} | {row['basin']} | {row['start_family']} | "
            f"{row['level_sigma']:.2f} | {row['matched_valid']} | {row['both_strict']} | "
            f"{row['cs2_only_strict']} | {row['intrinsic_only_strict']} | "
            f"{row['neither_strict']} | {row['net_cs2_wins']} | n/a |"
        )
    (args.run_root / "ANALYSIS.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {args.run_root / 'analysis.json'} and ANALYSIS.md", flush=True)


if __name__ == "__main__":
    main()
