#!/usr/bin/env python3
"""Build one denominator-explicit table for matched g-xTB noise campaigns.

The manifest names raw-search roots and optional endpoint-score roots.  It
intentionally aggregates terminal/search data separately from endpoint
topology: an optimizer may find a local index-one candidate without recovering
the intended two-basin event.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


def _intrinsic_rows(root: Path) -> list[dict[str, Any]]:
    return [json.loads(path.read_text()) for path in sorted((root / "tasks").glob("task_*.json"))]


def _legacy_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.glob("task_*/summary_*.parquet")):
        rows.extend(pq.read_table(path).to_pylist())
    return rows


def _topology_rows(root: Path | None) -> dict[int, dict[str, Any]]:
    if root is None or not root.exists():
        return {}
    return {
        int(row["sample_id"]): row
        for path in sorted(root.glob("task_*.json"))
        if (row := json.loads(path.read_text())) and "sample_id" in row
    }


def _raw_rows(root: Path, family: str) -> list[dict[str, Any]]:
    if family in {"competitive", "competitive_subspace"}:
        return _intrinsic_rows(root)
    if family in {"regular_gad", "sella"}:
        return _legacy_rows(root)
    raise ValueError(f"unknown family {family!r}")


def _as_bool(row: dict[str, Any], key: str) -> bool:
    return bool(row.get(key, False))


def _calculator_valid(row: dict[str, Any], family: str) -> bool:
    """Distinguish calculator failure from an optimizer's nonconvergence."""

    if family in {"competitive", "competitive_subspace"}:
        return not bool(row.get("error"))
    if family == "sella":
        return not bool(row.get("failure_type")) and not bool(row.get("final_eval_error"))
    fmax = float(row.get("final_force_max", math.nan))
    return int(row.get("final_n_neg", -1)) >= 0 and math.isfinite(fmax)


def _final_fmax(row: dict[str, Any], family: str) -> float:
    key = "final_fmax" if family in {"competitive", "competitive_subspace"} else "final_force_max"
    return float(row.get(key, math.inf))


def _summary(spec: dict[str, Any]) -> dict[str, Any]:
    family = spec["family"]
    raw = _raw_rows(Path(spec["raw_root"]), family)
    topology = _topology_rows(Path(spec["topology_root"]) if spec.get("topology_root") else None)
    valid = [row for row in raw if _calculator_valid(row, family)]
    local = [
        row
        for row in valid
        if int(row.get("final_n_neg", -1)) == 1 and _final_fmax(row, family) < 0.03
    ]
    strict = [
        row
        for row in valid
        if int(row.get("final_n_neg", -1)) == 1 and _final_fmax(row, family) < 0.01
    ]
    endpoint_minima = [
        row for row in local if _as_bool(topology.get(int(row["sample_id"]), {}), "endpoint_minima")
    ]
    intended = [
        row
        for row in local
        if _as_bool(topology.get(int(row["sample_id"]), {}), "native_endpoint_topology")
    ]
    return {
        "method": family,
        "noise_angstrom": float(spec["noise_angstrom"]),
        "budget_updates": int(spec["budget_updates"]),
        "starts": len(raw),
        "calculator_valid": len(valid),
        "calculator_error": len(raw) - len(valid),
        "terminal_index0": sum(int(row.get("final_n_neg", -1)) == 0 for row in valid),
        "terminal_index1_high_force": sum(
            int(row.get("final_n_neg", -1)) == 1 and _final_fmax(row, family) >= 0.03
            for row in valid
        ),
        "terminal_index_gt1": sum(int(row.get("final_n_neg", -1)) > 1 for row in valid),
        "local_index1": len(local),
        "strict_index1": len(strict),
        "endpoint_scored": len(topology),
        "endpoint_minima": len(endpoint_minima),
        "native_topology": len(intended),
        "native_topology_per_start": len(intended) / len(raw) if raw else None,
        "native_topology_per_local": len(intended) / len(local) if local else None,
    }


def _markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Matched Transition1x / g-xTB noise grid",
        "",
        (
            "`native_topology` is scored only after an index-1 local candidate passes "
            "`fmax < 0.03 eV Å^-1`; its denominator is shown explicitly."
        ),
        "",
        "| method | noise (Å) | updates | starts | valid | index 0 | index 1 / high force | index >1 | local index 1 | strict index 1 | endpoint scored | native topology | topology / starts | topology / local |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        fraction = lambda value: "—" if value is None else f"{100 * value:.1f}%"
        lines.append(
            "| {method} | {noise_angstrom:.2f} | {budget_updates} | {starts} | "
            "{calculator_valid} | {terminal_index0} | {terminal_index1_high_force} | "
            "{terminal_index_gt1} | {local_index1} | {strict_index1} | {endpoint_scored} | "
            "{native_topology} | {per_start} | {per_local} |".format(
                **row,
                per_start=fraction(row["native_topology_per_start"]),
                per_local=fraction(row["native_topology_per_local"]),
            )
        )
    return "\n".join(lines) + "\n"


def _wilson(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    """Two-sided Wilson interval, robust even for zero observed successes."""

    if trials <= 0:
        return math.nan, math.nan
    p = successes / trials
    denom = 1.0 + z * z / trials
    centre = (p + z * z / (2.0 * trials)) / denom
    half = z * math.sqrt(p * (1.0 - p) / trials + z * z / (4.0 * trials * trials)) / denom
    return centre - half, centre + half


def _analysis(rows: list[dict[str, Any]]) -> str:
    """Write a compact interpretation without hiding rates or denominators."""

    lines = [
        "# Matched Transition1x / g-xTB grid interpretation",
        "",
        "Every cell has 287 paired Cartesian-noise starts. `local index 1` means "
        "projected index one and $f_{\\max}<0.03$ eV Å$^{-1}$; `native topology` "
        "additionally requires the two independently relaxed downhill branches to "
        "match the labelled Transition1x endpoint pair. Thus topology/start is the "
        "primary robustness measure, while topology/local is candidate selectivity.",
        "",
        "Wilson 95% intervals below quantify only finite-start sampling uncertainty; "
        "they do not include calculator, data-set, or hyperparameter uncertainty.",
        "",
    ]
    for noise in sorted({row["noise_angstrom"] for row in rows}):
        panel = sorted(
            (row for row in rows if row["noise_angstrom"] == noise),
            key=lambda row: (-float(row["native_topology_per_start"] or 0.0), row["method"]),
        )
        lines.extend([
            f"## Noise $\\sigma={noise:.2f}$ Å",
            "",
            "| method | local / starts | native topology / starts (95% Wilson) | native topology / local | calculator errors | terminal index 0 | terminal index >1 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ])
        for row in panel:
            lo, hi = _wilson(row["native_topology"], row["starts"])
            selectivity = (
                f"{row['native_topology']}/{row['local_index1']} "
                f"({100 * row['native_topology_per_local']:.1f}%)"
                if row["local_index1"] else "0/0 (—)"
            )
            lines.append(
                f"| {row['method']} | {row['local_index1']}/{row['starts']} "
                f"({100 * row['local_index1'] / row['starts']:.1f}%) | "
                f"{row['native_topology']}/{row['starts']} "
                f"({100 * row['native_topology_per_start']:.1f}%; "
                f"{100 * lo:.1f}–{100 * hi:.1f}%) | "
                f"{selectivity} | "
                f"{row['calculator_error']} | {row['terminal_index0']} | "
                f"{row['terminal_index_gt1']} |"
            )
        winner = panel[0]
        lines.extend([
            "",
            f"The highest observed native-topology recovery in this panel is "
            f"`{winner['method']}`: {winner['native_topology']}/{winner['starts']} "
            f"({100 * winner['native_topology_per_start']:.1f}%).",
            "",
        ])
    lines.extend([
        "## Reading the failure columns",
        "",
        "`terminal index 0` denotes capture in a minimum-like basin at the budget; "
        "`terminal index >1` denotes remaining higher-order curvature; and the omitted "
        "valid starts are index-one but force-limited. These categories are terminal "
        "outcomes, not post hoc exclusions. Compare them against `local / starts` before "
        "attributing a topology difference to endpoint selectivity.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="JSON list of campaign specifications")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    specs = json.loads(args.manifest.read_text())
    if not isinstance(specs, list):
        raise TypeError("manifest must be a JSON list")
    rows = sorted(
        (_summary(spec) for spec in specs), key=lambda row: (row["noise_angstrom"], row["method"])
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
    (args.output_dir / "SUMMARY.md").write_text(_markdown(rows))
    (args.output_dir / "ANALYSIS.md").write_text(_analysis(rows))
    print(_markdown(rows), end="")


if __name__ == "__main__":
    main()
