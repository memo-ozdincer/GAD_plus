#!/usr/bin/env python3
"""Physical-validity and diversity analysis for an intrinsic-GAD LJ7 sweep."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

from gadplus.calculator.lennard_jones import (
    lj_atomic_nums,
    make_lj_predict_fn,
    pair_distances,
)
from gadplus.projection import atomic_nums_to_symbols, vib_eig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    worker = subparsers.add_parser("worker")
    worker.add_argument("--run-root", type=Path, required=True)
    worker.add_argument("--shard-id", type=int, required=True)
    worker.add_argument("--n-shards", type=int, required=True)
    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--run-root", type=Path, required=True)
    aggregate.add_argument("--expected-shards", type=int, required=True)
    return parser.parse_args()


def _is_connected(coords: torch.Tensor, cutoff: float = 1.5) -> bool:
    distances = torch.cdist(coords, coords)
    neighbours = distances < cutoff
    visited = {0}
    frontier = [0]
    while frontier:
        atom = frontier.pop()
        for candidate in torch.nonzero(neighbours[atom], as_tuple=False).reshape(-1).tolist():
            if candidate not in visited:
                visited.add(candidate)
                frontier.append(candidate)
    return len(visited) == coords.shape[0]


def worker(args: argparse.Namespace) -> None:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    with (args.run_root / "all_results.csv").open(newline="") as handle:
        source_rows = list(csv.DictReader(handle))
    predictor = make_lj_predict_fn()
    atomic_nums = lj_atomic_nums(7)
    symbols = atomic_nums_to_symbols(atomic_nums)
    output_rows = []
    for source in source_rows[args.shard_id :: args.n_shards]:
        if source["converged"].lower() != "true" or not source["final_coords"]:
            continue
        coords = torch.tensor(json.loads(source["final_coords"]), dtype=torch.float64)
        out = predictor(coords, atomic_nums, do_hessian=True)
        evals, _, _ = vib_eig(out["hessian"], coords, symbols)
        spectral_scale = float(torch.sqrt(torch.mean(evals.square())).item())
        distances = pair_distances(coords)
        centered = coords - coords.mean(dim=0, keepdim=True)
        endpoint_energies = sorted(
            [float(source["endpoint_energy_1"]), float(source["endpoint_energy_2"])]
        )
        endpoints_finite = all(math.isfinite(value) for value in endpoint_energies)
        output_rows.append(
            {
                "panel": source["panel"],
                "noise": float(source["noise"]),
                "sample_id": int(source["sample_id"]),
                "energy": float(source["final_energy"]),
                "n_neg": int(source["final_n_neg"]),
                "force_max": float(source["final_force_max"]),
                "downhill_valid": source["downhill_valid"].lower() == "true",
                "same_reference_event": source["correct_event"].lower() == "true",
                "eig0": float(evals[0].item()),
                "eig1": float(evals[1].item()),
                "eig0_scaled": float(evals[0].item()) / spectral_scale,
                "eig1_scaled": float(evals[1].item()) / spectral_scale,
                "spectral_scale": spectral_scale,
                "near_flat": float(evals[1].item()) / spectral_scale < 0.01,
                "min_pair_distance": float(distances.min().item()),
                "max_pair_distance": float(distances.max().item()),
                "cluster_radius": float(torch.linalg.vector_norm(centered, dim=1).max().item()),
                "connected_cutoff_1p5": _is_connected(coords),
                "pair_fingerprint": torch.sort(distances).values.tolist(),
                "endpoint_energy_low": endpoint_energies[0],
                "endpoint_energy_high": endpoint_energies[1],
                "endpoint_energy_distinct": endpoints_finite
                and abs(endpoint_energies[1] - endpoint_energies[0]) > 1.0e-4,
                "barrier_from_lower_endpoint": (
                    float(source["final_energy"]) - endpoint_energies[0]
                    if endpoints_finite
                    else math.nan
                ),
                "barrier_from_higher_endpoint": (
                    float(source["final_energy"]) - endpoint_energies[1]
                    if endpoints_finite
                    else math.nan
                ),
            }
        )
    output_dir = args.run_root / "diversity_shards"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"shard_{args.shard_id:03d}.json").write_text(
        json.dumps(output_rows, separators=(",", ":")) + "\n"
    )


class _UnionFind:
    def __init__(self, size: int):
        self.parent = list(range(size))

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, first: int, second: int) -> None:
        first_root = self.find(first)
        second_root = self.find(second)
        if first_root != second_root:
            self.parent[second_root] = first_root


def _assign_saddle_families(rows: list[dict]) -> None:
    fingerprint_tolerance = 1.0e-3
    energy_tolerance = 1.0e-4
    fingerprints = np.asarray([row["pair_fingerprint"] for row in rows])
    energies = np.asarray([row["energy"] for row in rows])
    dimension = fingerprints.shape[1]
    features = np.column_stack(
        (
            fingerprints / (fingerprint_tolerance * math.sqrt(dimension)),
            energies / energy_tolerance,
        )
    )
    tree = cKDTree(features)
    union_find = _UnionFind(len(rows))
    for first, second in tree.query_pairs(r=math.sqrt(2.0)):
        fingerprint_rms = float(np.sqrt(np.mean((fingerprints[first] - fingerprints[second]) ** 2)))
        if (
            fingerprint_rms <= fingerprint_tolerance
            and abs(energies[first] - energies[second]) <= energy_tolerance
        ):
            union_find.union(first, second)
    roots = {}
    for index, row in enumerate(rows):
        root = union_find.find(index)
        roots.setdefault(root, len(roots))
        row["saddle_family"] = roots[root]
        row["endpoint_energy_pair"] = tuple(
            sorted(
                (
                    round(row["endpoint_energy_low"] / energy_tolerance),
                    round(row["endpoint_energy_high"] / energy_tolerance),
                )
            )
        )


def _effective_count(labels: list) -> float:
    counts = Counter(labels)
    total = sum(counts.values())
    entropy = -sum((count / total) * math.log(count / total) for count in counts.values())
    return math.exp(entropy)


def _quantile(values: list[float], probability: float) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return float(np.quantile(finite, probability)) if finite else math.nan


def aggregate(args: argparse.Namespace) -> None:
    paths = sorted((args.run_root / "diversity_shards").glob("shard_*.json"))
    if len(paths) != args.expected_shards:
        raise RuntimeError(f"expected {args.expected_shards} shards, found {len(paths)}")
    rows = []
    for path in paths:
        rows.extend(json.loads(path.read_text()))
    _assign_saddle_families(rows)

    groups = defaultdict(list)
    for row in rows:
        groups[(row["panel"], row["noise"])].append(row)
    summary = []
    for (panel, noise), group in sorted(groups.items()):
        valid = [row for row in group if row["downhill_valid"]]
        family_labels = [row["saddle_family"] for row in valid]
        event_labels = [row["endpoint_energy_pair"] for row in valid]
        family_counts = Counter(family_labels)
        summary.append(
            {
                "panel": panel,
                "noise": noise,
                "n_strict": len(group),
                "n_downhill_valid": len(valid),
                "n_saddle_families": len(set(family_labels)),
                "effective_saddle_families": _effective_count(family_labels)
                if family_labels
                else 0.0,
                "largest_family_fraction": max(family_counts.values()) / len(valid)
                if valid
                else math.nan,
                "n_endpoint_energy_pairs_lower_bound": len(set(event_labels)),
                "distinct_endpoint_energy_rate": sum(
                    row["endpoint_energy_distinct"] for row in valid
                )
                / len(valid)
                if valid
                else math.nan,
                "near_flat_rate": sum(row["near_flat"] for row in valid) / len(valid)
                if valid
                else math.nan,
                "fragmented_rate_cutoff_1p5": sum(not row["connected_cutoff_1p5"] for row in valid)
                / len(valid)
                if valid
                else math.nan,
                "energy_median": _quantile([row["energy"] for row in valid], 0.5),
                "energy_q95": _quantile([row["energy"] for row in valid], 0.95),
                "barrier_from_higher_median": _quantile(
                    [row["barrier_from_higher_endpoint"] for row in valid], 0.5
                ),
                "barrier_from_higher_q95": _quantile(
                    [row["barrier_from_higher_endpoint"] for row in valid], 0.95
                ),
                "eig1_scaled_q05": _quantile([row["eig1_scaled"] for row in valid], 0.05),
                "max_pair_q95": _quantile([row["max_pair_distance"] for row in valid], 0.95),
            }
        )

    (args.run_root / "diversity_all.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n"
    )
    (args.run_root / "diversity_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    lines = [
        "# LJ7 physical-validity and diversity summary",
        "",
        (
            "Endpoint-energy-pair counts are lower bounds on event diversity; equal-energy "
            "isomers can be merged by this diagnostic."
        ),
        "",
        (
            "| Panel | Noise | Valid | TS families | Effective TS families | Largest family | "
            "Endpoint-pair lower bound | Distinct endpoints | Near-flat | Fragmented | Barrier q95 |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['panel']} | {row['noise']:.3f} | {row['n_downhill_valid']} | "
            f"{row['n_saddle_families']} | {row['effective_saddle_families']:.1f} | "
            f"{100 * row['largest_family_fraction']:.1f}% | "
            f"{row['n_endpoint_energy_pairs_lower_bound']} | "
            f"{100 * row['distinct_endpoint_energy_rate']:.1f}% | "
            f"{100 * row['near_flat_rate']:.1f}% | "
            f"{100 * row['fragmented_rate_cutoff_1p5']:.1f}% | "
            f"{row['barrier_from_higher_q95']:.3f} |"
        )
    (args.run_root / "DIVERSITY_SUMMARY.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    if args.command == "worker":
        worker(args)
    else:
        aggregate(args)


if __name__ == "__main__":
    main()
