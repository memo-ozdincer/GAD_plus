#!/usr/bin/env python3
"""Paired multi-size Lennard-Jones benchmark for intrinsic GAD, CS2-GAD, and Sella.

The benchmark uses published Cambridge Energy Landscape Database minima for
LJ13, LJ31, LJ38, LJ55, and LJ75.  LJ38 and LJ75 additionally include their
lowest icosahedral competitor, so the two funnels are represented explicitly.
Every method receives byte-identical starts.  This is a local saddle-search
benchmark; it does not claim to solve the global-minimum problem.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from gadplus.calculator.lennard_jones import lj_atomic_nums, make_lj_predict_fn, pair_distances
from gadplus.core.convergence import force_max
from gadplus.projection import atomic_nums_to_symbols, get_mass_weights, vib_eig
from gadplus.search.intrinsic_gad import IntrinsicGADConfig, run_intrinsic_gad
from lj_intrinsic_noise_sweep import (
    _downhill_endpoints,
    _fingerprint,
    _relax_endpoint,
    _wilson_interval,
)
from lj_method_progression_sweep import _connected, _run_sella


METHODS = ("intrinsic_lambda2", "cs2", "sella")
SIZES = (13, 31, 38, 55, 75)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--output-root", type=Path, required=True)
    common.add_argument(
        "--asset-root",
        type=Path,
        default=Path("references/lj_global_minima"),
    )
    common.add_argument("--sizes", default=",".join(map(str, SIZES)))
    common.add_argument("--methods", default=",".join(METHODS))
    common.add_argument("--start-families", default="mode_push,cartesian_noise")
    common.add_argument("--levels", default="0.10,0.20")
    common.add_argument("--n-samples", type=int, default=16)
    common.add_argument("--seed", type=int, default=20260809)
    common.add_argument("--max-steps", type=int, default=500)
    common.add_argument("--endpoint-fmax", type=float, default=1.0e-5)

    sub.add_parser("prepare", parents=[common])
    worker = sub.add_parser("worker", parents=[common])
    worker.add_argument("--task-id", type=int, required=True)
    aggregate = sub.add_parser("aggregate", parents=[common])
    aggregate.add_argument("--expected-tasks", type=int, required=True)
    return parser.parse_args()


def _items(text: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in text.split(",") if item.strip())


def _floats(text: str) -> tuple[float, ...]:
    return tuple(float(item) for item in _items(text))


def _ints(text: str) -> tuple[int, ...]:
    return tuple(int(item) for item in _items(text))


def _basins(asset_root: Path, sizes: tuple[int, ...]) -> list[tuple[int, str, Path]]:
    basins: list[tuple[int, str, Path]] = []
    for size in sizes:
        basins.append((size, "global", asset_root / f"lj{size}_global.points"))
        competitor = asset_root / f"lj{size}_icosahedral.points"
        if competitor.exists():
            basins.append((size, "icosahedral", competitor))
    return basins


def _tasks(args: argparse.Namespace) -> list[tuple[str, int, str, str, float, int]]:
    methods = _items(args.methods)
    if not methods or any(method not in METHODS for method in methods):
        raise ValueError(f"methods must be a nonempty subset of {METHODS}")
    families = _items(args.start_families)
    if not families or any(family not in {"mode_push", "cartesian_noise"} for family in families):
        raise ValueError("start families must be mode_push and/or cartesian_noise")
    return [
        (method, size, basin, family, level, sample_id)
        for method in methods
        for size, basin, _ in _basins(args.asset_root, _ints(args.sizes))
        for family in families
        for level in _floats(args.levels)
        for sample_id in range(args.n_samples)
    ]


def _load_points(path: Path, size: int) -> torch.Tensor:
    values = [float(value) for value in path.read_text().split()]
    if len(values) != 3 * size:
        raise ValueError(f"{path} contains {len(values)} values; expected {3 * size}")
    coords = torch.tensor(values, dtype=torch.float64).reshape(size, 3)
    return coords - coords.mean(dim=0, keepdim=True)


def _spectrum(predictor, coords: torch.Tensor, atomic_nums: torch.Tensor):
    output = predictor(coords, atomic_nums, do_hessian=True)
    eigenvalues, modes, _ = vib_eig(
        output["hessian"], coords, atomic_nums_to_symbols(atomic_nums)
    )
    return output, eigenvalues, modes


def _make_start(
    minimum: torch.Tensor,
    modes_mw: torch.Tensor,
    atomic_nums: torch.Tensor,
    family: str,
    level: float,
    sample_id: int,
    seed: int,
) -> tuple[torch.Tensor, int]:
    generator = torch.Generator().manual_seed(seed)
    if family == "cartesian_noise":
        start = minimum + level * torch.randn(
            minimum.shape, generator=generator, dtype=minimum.dtype
        )
        return start - start.mean(dim=0, keepdim=True), -1

    n_modes = modes_mw.shape[1]
    mode_index = sample_id % min(8, n_modes)
    _, _, _, inv_sqrt_mass = get_mass_weights(atomic_nums_to_symbols(atomic_nums))
    direction = (inv_sqrt_mass * modes_mw[:, mode_index]).reshape_as(minimum)
    direction = direction / torch.sqrt(torch.mean(torch.sum(direction.square(), dim=1)))
    sign = -1.0 if (sample_id // min(8, n_modes)) % 2 else 1.0
    jitter = 0.01 * torch.randn(minimum.shape, generator=generator, dtype=minimum.dtype)
    start = minimum + sign * level * direction + jitter
    return start - start.mean(dim=0, keepdim=True), mode_index


def prepare(args: argparse.Namespace) -> None:
    predictor = make_lj_predict_fn()
    rows = []
    for size, basin, path in _basins(args.asset_root, _ints(args.sizes)):
        source_coords = _load_points(path, size)
        atomic_nums = lj_atomic_nums(size)
        source_output, source_eigenvalues, _ = _spectrum(
            predictor, source_coords, atomic_nums
        )
        relaxed = _relax_endpoint(predictor, atomic_nums, source_coords)
        relaxation_attempts = 1
        while relaxed["force_max"] >= 1.0e-5 and relaxation_attempts < 3:
            relaxed = _relax_endpoint(predictor, atomic_nums, relaxed["coords"])
            relaxation_attempts += 1
        if relaxed["n_neg"] != 0 or relaxed["force_max"] >= 1.0e-5:
            raise RuntimeError(
                f"tight reference relaxation failed for LJ{size} {basin}: "
                f"n_neg={relaxed['n_neg']} fmax={relaxed['force_max']:.3e}"
            )
        rows.append(
            {
                "size": size,
                "basin": basin,
                "source": str(path.resolve()),
                "source_energy": float(source_output["energy"].item()),
                "source_force_max": force_max(source_output["forces"]),
                "source_n_neg": int((source_eigenvalues < -1.0e-4).sum().item()),
                "energy": relaxed["energy"],
                "force_max": relaxed["force_max"],
                "n_neg": relaxed["n_neg"],
                "relaxation_attempts": relaxation_attempts,
                "relaxation_iterations": relaxed["n_iterations"],
                "coords": relaxed["coords"].tolist(),
            }
        )
    args.output_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "surface": "reduced Lennard-Jones clusters",
        "epsilon": 1.0,
        "sigma": 1.0,
        "mass": 1.008,
        "methods": list(_items(args.methods)),
        "start_families": list(_items(args.start_families)),
        "levels_sigma": list(_floats(args.levels)),
        "n_samples_per_cell": args.n_samples,
        "seed": args.seed,
        "max_steps": args.max_steps,
        "strict_gate": {"n_neg": 1, "fmax": 0.01, "index_threshold": 1.0e-4},
        "profiles": {
            "intrinsic_lambda2": {"gate_variant": "lambda2", "tau_s": 0.01, "eta": 0.05},
            "cs2": {"gate_variant": "competitive_subspace", "tau_s": 0.01, "eta": 0.01},
            "sella": "Cartesian Eckart RS-P-RFO with current analytic Hessian",
        },
        "basins": rows,
    }
    protocol_path = args.output_root / "protocol.json"
    if protocol_path.exists():
        if json.loads(protocol_path.read_text()) != payload:
            raise RuntimeError(
                f"protocol mismatch in {protocol_path}; use a new output root"
            )
    else:
        protocol_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        json.dumps(
            [{key: value for key, value in row.items() if key != "coords"} for row in rows],
            indent=2,
        ),
        flush=True,
    )


def _run_intrinsic(method: str, predictor, coords, atomic_nums, max_steps: int):
    config = IntrinsicGADConfig(
        max_steps=max_steps,
        spectral_temperature=0.01,
        step_fraction=0.05 if method == "intrinsic_lambda2" else 0.01,
        gate_variant="lambda2" if method == "intrinsic_lambda2" else "competitive_subspace",
        force_threshold=0.01,
        force_criterion="fmax",
        record_history=False,
    )
    return run_intrinsic_gad(predictor, coords, atomic_nums, config)


def worker(args: argparse.Namespace) -> None:
    torch.set_num_threads(max(1, int(__import__("os").environ.get("OMP_NUM_THREADS", "1"))))
    tasks = _tasks(args)
    if not 0 <= args.task_id < len(tasks):
        raise ValueError(f"task-id must be in [0, {len(tasks)})")
    method, size, basin, family, level, sample_id = tasks[args.task_id]
    seed = args.seed + 100000 * size + 1000 * sample_id + int(round(100 * level))
    row: dict[str, Any] = {
        "task_id": args.task_id,
        "method": method,
        "size": size,
        "basin": basin,
        "start_family": family,
        "level_sigma": level,
        "sample_id": sample_id,
        "seed": seed,
        "calculator_valid": False,
        "strict_ts": False,
        "endpoint_minima": False,
        "error": "",
    }
    try:
        protocol = json.loads((args.output_root / "protocol.json").read_text())
        reference = next(
            item for item in protocol["basins"]
            if item["size"] == size and item["basin"] == basin
        )
        minimum = torch.tensor(reference["coords"], dtype=torch.float64)
        atomic_nums = lj_atomic_nums(size)
        predictor = make_lj_predict_fn()
        _, _, minimum_modes = _spectrum(predictor, minimum, atomic_nums)
        start, mode_index = _make_start(
            minimum, minimum_modes, atomic_nums, family, level, sample_id, seed
        )
        initial_output, initial_eigenvalues, _ = _spectrum(predictor, start, atomic_nums)

        if method == "sella":
            sella_args = argparse.Namespace(euler_max_steps=args.max_steps)
            result = _run_sella(predictor, atomic_nums, start, sella_args)
            final_coords = result["coords"]
            steps = int(result["total_steps"])
            evaluations = int(result["n_evaluations"])
            wall_time = float(result["wall_time_s"])
        else:
            result_raw = _run_intrinsic(method, predictor, start, atomic_nums, args.max_steps)
            final_coords = result_raw.final_coords.to(torch.float64)
            steps = result_raw.total_steps
            evaluations = result_raw.n_evaluations
            wall_time = result_raw.wall_time_s

        final_output, final_eigenvalues, _ = _spectrum(predictor, final_coords, atomic_nums)
        final_fmax = force_max(final_output["forces"])
        final_n_neg = int((final_eigenvalues < -1.0e-4).sum().item())
        strict = final_n_neg == 1 and final_fmax < 0.01
        endpoints: list[dict[str, Any]] = []
        if strict:
            endpoints = _downhill_endpoints(
                predictor, atomic_nums, final_coords, displacement=0.03
            )
        endpoint_minima = len(endpoints) == 2 and all(
            endpoint["n_neg"] == 0 and endpoint["force_max"] < args.endpoint_fmax
            for endpoint in endpoints
        )
        spectral_scale = float(torch.sqrt(torch.mean(final_eigenvalues.square())).item())
        row.update(
            {
                "calculator_valid": True,
                "mode_index": mode_index,
                "initial_energy": float(initial_output["energy"].item()),
                "initial_n_neg": int((initial_eigenvalues < -1.0e-4).sum().item()),
                "initial_fmax": force_max(initial_output["forces"]),
                "strict_ts": strict,
                "endpoint_minima": endpoint_minima,
                "steps": steps,
                "evaluations": evaluations,
                "wall_time_s": wall_time,
                "final_energy": float(final_output["energy"].item()),
                "final_n_neg": final_n_neg,
                "final_fmax": final_fmax,
                "final_lambda1": float(final_eigenvalues[0].item()),
                "final_lambda2": float(final_eigenvalues[1].item()),
                "final_lambda2_scaled": float(final_eigenvalues[1].item()) / spectral_scale,
                "connected_cutoff_1p5": _connected(final_coords),
                "max_pair_distance": float(pair_distances(final_coords).max().item()),
                "fingerprint": _fingerprint(final_coords).tolist(),
                "endpoint_energies": [float(endpoint["energy"]) for endpoint in endpoints],
            }
        )
    except Exception as exc:  # noqa: BLE001 - exceptions are recorded outcomes.
        row["error"] = f"{type(exc).__name__}: {exc}"

    task_root = args.output_root / "tasks"
    task_root.mkdir(parents=True, exist_ok=True)
    (task_root / f"task_{args.task_id:05d}.json").write_text(
        json.dumps(row, indent=2, sort_keys=True) + "\n"
    )


def _median(values: list[float]) -> float:
    return float(statistics.median(values)) if values else math.nan


def aggregate(args: argparse.Namespace) -> None:
    paths = sorted((args.output_root / "tasks").glob("task_*.json"))
    if len(paths) != args.expected_tasks:
        raise RuntimeError(f"expected {args.expected_tasks} tasks, found {len(paths)}")
    rows = [json.loads(path.read_text()) for path in paths]
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            row["method"], row["size"], row["basin"],
            row["start_family"], row["level_sigma"],
        )
        groups[key].append(row)
    summary = []
    for key, group in sorted(groups.items()):
        valid = [row for row in group if row["calculator_valid"]]
        strict = [row for row in valid if row["strict_ts"]]
        endpoint = [row for row in valid if row["endpoint_minima"]]
        lo, hi = _wilson_interval(len(strict), len(valid)) if valid else (math.nan, math.nan)
        summary.append(
            {
                "method": key[0], "size": key[1], "basin": key[2],
                "start_family": key[3], "level_sigma": key[4],
                "planned": len(group), "valid": len(valid),
                "strict": len(strict), "strict_rate": len(strict) / len(valid) if valid else math.nan,
                "strict_ci95": [lo, hi],
                "endpoint_minima": len(endpoint),
                "endpoint_rate": len(endpoint) / len(valid) if valid else math.nan,
                "median_strict_evaluations": _median([row["evaluations"] for row in strict]),
                "median_strict_wall_s": _median([row["wall_time_s"] for row in strict]),
                "errors": len(group) - len(valid),
            }
        )
    (args.output_root / "all_results.json").write_text(json.dumps(rows, indent=2) + "\n")
    (args.output_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (args.output_root / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)
    lines = [
        "# Multi-size LJ CS2 benchmark", "",
        "| Method | LJ | Basin | Start | Level | Valid | Strict | Two minima | Median evals | Errors |",
        "|---|---:|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['method']} | {row['size']} | {row['basin']} | {row['start_family']} | "
            f"{row['level_sigma']:.2f} | {row['valid']}/{row['planned']} | "
            f"{row['strict']}/{row['valid']} | {row['endpoint_minima']}/{row['valid']} | "
            f"{row['median_strict_evaluations']:.1f} | {row['errors']} |"
        )
    (args.output_root / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    print(f"aggregated {len(rows)} tasks into {len(summary)} cells", flush=True)


def main() -> None:
    args = parse_args()
    if args.command == "prepare":
        prepare(args)
    elif args.command == "worker":
        worker(args)
    else:
        aggregate(args)


if __name__ == "__main__":
    main()
