#!/usr/bin/env python3
"""Build a checked CS²-GAD HIP report against completed historical references."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any


NOISES = (0.10, 0.15, 0.20)
PLANNED = 287
HISTORICAL = {
    0.10: {"plain_strict": 209, "sella_strict": 209, "plain_irc_topo": 225, "sella_irc_topo": 208},
    0.15: {"plain_strict": 167, "sella_strict": 155, "plain_irc_topo": 177, "sella_irc_topo": 143},
    0.20: {"plain_strict": 128, "sella_strict": 78, "plain_irc_topo": 128, "sella_irc_topo": 67},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--allow-missing-irc", action="store_true")
    return parser.parse_args()


def _wilson(successes: int, total: int) -> list[float]:
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [center - half, center + half]


def _quantile(values: list[int], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _cell(run_root: Path, noise: float, require_irc: bool) -> dict[str, Any]:
    root = run_root / f"noise_{noise:.2f}A"
    summary = json.loads((root / "summary.json").read_text())
    paths = sorted((root / "samples").glob("sample_*.json"))
    rows = [json.loads(path.read_text()) for path in paths]
    ids = [int(row["sample_id"]) for row in rows]
    if len(rows) != PLANNED or sorted(ids) != list(range(PLANNED)):
        raise RuntimeError(f"noise {noise:.2f}: search coverage is not exactly 0..{PLANNED - 1}")
    if summary["planned"] != PLANNED or summary["rows"] != PLANNED:
        raise RuntimeError(f"noise {noise:.2f}: aggregate denominator mismatch")
    if any(row.get("protocol_sha256") != summary["protocol_sha256"] for row in rows):
        raise RuntimeError(f"noise {noise:.2f}: mixed search protocols")
    evaluations = [int(row["n_evaluations"]) for row in rows if row["calculator_valid"]]
    strict = int(summary["strict_ts"])
    history = HISTORICAL[noise]
    output: dict[str, Any] = {
        "noise_angstrom": noise,
        "planned": PLANNED,
        "valid": int(summary["calculator_valid"]),
        "strict": strict,
        "strict_rate": strict / PLANNED,
        "strict_ci95_wilson": _wilson(strict, PLANNED),
        "terminal_class_counts": summary["terminal_class_counts"],
        "median_evaluations_valid": statistics.median(evaluations),
        "p90_evaluations_valid": _quantile(evaluations, 0.90),
        "p95_evaluations_valid": _quantile(evaluations, 0.95),
        "max_evaluations_valid": max(evaluations),
        "median_wall_time_s_valid": summary["median_wall_time_s_valid"],
        "plain_gad_strict": history["plain_strict"],
        "sella_strict": history["sella_strict"],
        "strict_delta_vs_plain_pp": 100 * (strict - history["plain_strict"]) / PLANNED,
        "strict_delta_vs_sella_pp": 100 * (strict - history["sella_strict"]) / PLANNED,
        "search_protocol_sha256": summary["protocol_sha256"],
    }
    irc_path = run_root / "irc_topo" / f"noise_{noise:.2f}A" / "summary.json"
    if not irc_path.exists():
        if require_irc:
            raise FileNotFoundError(f"missing matched IRC aggregate: {irc_path}")
        output["irc_topo"] = None
        return output
    irc = json.loads(irc_path.read_text())
    if irc["planned"] != PLANNED or irc["rows"] != PLANNED:
        raise RuntimeError(f"noise {noise:.2f}: IRC denominator mismatch")
    intended = int(irc["topology_intended"])
    output["irc_topo"] = {
        "valid": int(irc["irc_valid"]),
        "intended": intended,
        "intended_rate": intended / PLANNED,
        "intended_ci95_wilson": _wilson(intended, PLANNED),
        "strict_and_intended": int(irc["strict_and_topology_intended"]),
        "errors": int(irc["irc_errors"]),
        "plain_gad_intended": history["plain_irc_topo"],
        "sella_intended": history["sella_irc_topo"],
        "delta_vs_plain_pp": 100 * (intended - history["plain_irc_topo"]) / PLANNED,
        "delta_vs_sella_pp": 100 * (intended - history["sella_irc_topo"]) / PLANNED,
        "protocol_sha256": irc["protocol_sha256"],
    }
    return output


def main() -> None:
    args = parse_args()
    cells = [_cell(args.run_root, noise, not args.allow_missing_irc) for noise in NOISES]
    analysis = {
        "run_root": str(args.run_root.resolve()),
        "method": "CS2-GAD",
        "historical_denominator": PLANNED,
        "historical_reference_counts": HISTORICAL,
        "cells": cells,
    }
    (args.run_root / "analysis.json").write_text(json.dumps(analysis, indent=2) + "\n")
    lines = [
        "# HIP CS2-GAD checked analysis",
        "",
        "## Local strict recovery",
        "",
        "| Noise (A) | Valid | CS2 strict | Plain GAD | Sella | Delta vs plain | Median/p95 evals |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for cell in cells:
        lines.append(
            f"| {cell['noise_angstrom']:.2f} | {cell['valid']}/{PLANNED} | "
            f"{cell['strict']}/{PLANNED} ({100 * cell['strict_rate']:.1f}%) | "
            f"{cell['plain_gad_strict']}/{PLANNED} | {cell['sella_strict']}/{PLANNED} | "
            f"{cell['strict_delta_vs_plain_pp']:+.1f} pp | "
            f"{cell['median_evaluations_valid']:.1f}/{cell['p95_evaluations_valid']:.1f} |"
        )
    lines.extend(
        [
            "",
            "## Terminal-index taxonomy",
            "",
            "| Noise (A) | Strict TS | Index 0 | Index 1 force-limited | Multi-negative | Calculator error |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for cell in cells:
        taxonomy = cell["terminal_class_counts"]
        lines.append(
            f"| {cell['noise_angstrom']:.2f} | {taxonomy.get('strict_ts', 0)} | "
            f"{taxonomy.get('index_zero', 0)} | {taxonomy.get('index_one_force_limited', 0)} | "
            f"{taxonomy.get('multi_negative', 0)} | {taxonomy.get('calculator_error', 0)} |"
        )
    if all(cell["irc_topo"] is not None for cell in cells):
        lines.extend(
            [
                "",
                "## All-endpoint intended IRC_TOPO",
                "",
                "| Noise (A) | IRC valid | CS2 intended | Plain GAD | Sella | Delta vs plain | Strict and intended |",
                "|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for cell in cells:
            irc = cell["irc_topo"]
            lines.append(
                f"| {cell['noise_angstrom']:.2f} | {irc['valid']}/{PLANNED} | "
                f"{irc['intended']}/{PLANNED} ({100 * irc['intended_rate']:.1f}%) | "
                f"{irc['plain_gad_intended']}/{PLANNED} | {irc['sella_intended']}/{PLANNED} | "
                f"{irc['delta_vs_plain_pp']:+.1f} pp | {irc['strict_and_intended']} |"
            )
    else:
        lines.extend(["", "Matched all-endpoint IRC_TOPO is still pending."])
    lines.extend(
        [
            "",
            "Rates use all 287 planned starts. Wilson intervals and exact protocol digests are in `analysis.json`.",
        ]
    )
    (args.run_root / "ANALYSIS.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {args.run_root / 'analysis.json'} and ANALYSIS.md", flush=True)


if __name__ == "__main__":
    main()
