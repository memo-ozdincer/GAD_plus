#!/usr/bin/env python3
"""Reproducible LJ7 capture-basin sweep for pointwise intrinsic GAD.

The experiment has two panels:

``saddle``
    Add Cartesian Gaussian noise directly to a reference LJ7 saddle.  This
    measures the optimizer's local capture basin.

``pushed``
    Add the same noise realization to the mode-pushed minimum used to target
    that saddle.  This measures robustness of the complete targeting protocol.

The primary outcome requires the maintained strict TS gate, valid relaxation
of both unstable-mode branches to projected minima, and agreement of the
unordered endpoint-basin pair with the reference event. TS energy and sorted
pair-distance fingerprints are retained as secondary saddle-family metrics.
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
from scipy.optimize import minimize

from gadplus.calculator.lennard_jones import (
    lj_atomic_nums,
    make_lj_predict_fn,
    pair_distances,
    pentagonal_bipyramid_geometry,
)
from gadplus.projection import atomic_nums_to_symbols, get_mass_weights, vib_eig
from gadplus.search.intrinsic_gad import IntrinsicGADConfig, run_intrinsic_gad

DEFAULT_NOISES = "0,0.005,0.01,0.02,0.03,0.05,0.075,0.10,0.125,0.15,0.20,0.25,0.30,0.40,0.50"
FIELDNAMES = (
    "panel",
    "noise",
    "sample_id",
    "seed",
    "converged",
    "downhill_valid",
    "correct_event",
    "correct_reference",
    "energy_match",
    "geometry_match",
    "final_energy",
    "reference_energy_delta",
    "pair_fingerprint_rms",
    "final_n_neg",
    "final_force_max",
    "final_gate_weight",
    "initial_n_neg",
    "initial_gate_weight",
    "total_steps",
    "n_evaluations",
    "failure_type",
    "error",
    "endpoint_energy_1",
    "endpoint_energy_2",
    "endpoint_n_neg_1",
    "endpoint_n_neg_2",
    "endpoint_force_max_1",
    "endpoint_force_max_2",
    "endpoint_pair_error",
    "final_coords",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="Construct the deterministic reference TS.")
    prepare.add_argument("--output-root", type=Path, required=True)
    prepare.add_argument("--push-amplitude", type=float, default=0.26)

    worker = subparsers.add_parser("worker", help="Run one deterministic task shard.")
    worker.add_argument("--output-root", type=Path, required=True)
    worker.add_argument("--shard-id", type=int, required=True)
    worker.add_argument("--n-shards", type=int, required=True)
    worker.add_argument("--n-samples", type=int, default=96)
    worker.add_argument("--noises", default=DEFAULT_NOISES)
    worker.add_argument("--seed", type=int, default=20260726)
    worker.add_argument("--max-steps", type=int, default=200)
    worker.add_argument("--energy-tolerance", type=float, default=1.0e-4)
    worker.add_argument("--fingerprint-tolerance", type=float, default=1.0e-3)
    worker.add_argument("--endpoint-displacement", type=float, default=0.03)
    worker.add_argument("--endpoint-force-threshold", type=float, default=1.0e-5)

    aggregate = subparsers.add_parser("aggregate", help="Aggregate completed shards.")
    aggregate.add_argument("--output-root", type=Path, required=True)
    aggregate.add_argument("--expected-shards", type=int, required=True)
    return parser.parse_args()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _fingerprint(coords: torch.Tensor) -> torch.Tensor:
    return torch.sort(pair_distances(coords.to(torch.float64))).values


def _relax_endpoint(
    predictor,
    atomic_nums: torch.Tensor,
    start: torch.Tensor,
) -> dict[str, Any]:
    """Relax one downhill branch and return a projected-minimum fingerprint."""

    shape = start.shape

    def energy_gradient(flat_coords):
        coords = torch.as_tensor(flat_coords, dtype=torch.float64).reshape(shape)
        out = predictor(coords, atomic_nums, do_hessian=False)
        return (
            float(out["energy"].item()),
            (-out["forces"]).detach().cpu().numpy().reshape(-1),
        )

    optimized = minimize(
        energy_gradient,
        start.detach().cpu().numpy().reshape(-1),
        jac=True,
        method="L-BFGS-B",
        options={"gtol": 1.0e-10, "ftol": 1.0e-14, "maxiter": 1000, "maxls": 50},
    )
    coords = torch.as_tensor(optimized.x, dtype=torch.float64).reshape(shape)
    out = predictor(coords, atomic_nums, do_hessian=True)
    evals, _, _ = vib_eig(
        out["hessian"],
        coords,
        atomic_nums_to_symbols(atomic_nums),
    )
    return {
        "optimizer_success": bool(optimized.success),
        "energy": float(out["energy"].item()),
        "force_max": float(out["forces"].abs().amax().item()),
        "n_neg": int((evals < -1.0e-4).sum().item()),
        "coords": coords,
        "fingerprint": _fingerprint(coords),
        "n_iterations": int(optimized.nit),
        "n_evaluations": int(optimized.nfev),
    }


def _downhill_endpoints(
    predictor,
    atomic_nums: torch.Tensor,
    saddle: torch.Tensor,
    displacement: float,
) -> list[dict[str, Any]]:
    out = predictor(saddle, atomic_nums, do_hessian=True)
    _, modes_mw, _ = vib_eig(
        out["hessian"],
        saddle,
        atomic_nums_to_symbols(atomic_nums),
    )
    _, _, _, inv_sqrt_mass = get_mass_weights(atomic_nums_to_symbols(atomic_nums))
    direction = (inv_sqrt_mass * modes_mw[:, 0]).reshape_as(saddle)
    direction = direction / torch.linalg.vector_norm(direction)
    return [
        _relax_endpoint(
            predictor,
            atomic_nums,
            saddle + sign * displacement * direction,
        )
        for sign in (-1.0, 1.0)
    ]


def _endpoint_payload(endpoint: dict[str, Any]) -> dict[str, Any]:
    return {
        "optimizer_success": endpoint["optimizer_success"],
        "energy": endpoint["energy"],
        "force_max": endpoint["force_max"],
        "n_neg": endpoint["n_neg"],
        "coords": endpoint["coords"].tolist(),
        "fingerprint": endpoint["fingerprint"].tolist(),
        "n_iterations": endpoint["n_iterations"],
        "n_evaluations": endpoint["n_evaluations"],
    }


def _match_endpoint_pair(
    endpoints: list[dict[str, Any]],
    reference_endpoints: list[dict[str, Any]],
    energy_tolerance: float,
    fingerprint_tolerance: float,
) -> tuple[bool, float]:
    """Match unordered endpoint pairs by energy and invariant geometry."""

    best_valid = False
    best_error = math.inf
    for assignment in ((0, 1), (1, 0)):
        valid = True
        total_error = 0.0
        for endpoint_index, reference_index in enumerate(assignment):
            endpoint = endpoints[endpoint_index]
            reference = reference_endpoints[reference_index]
            energy_error = abs(endpoint["energy"] - float(reference["energy"]))
            reference_fingerprint = torch.tensor(
                reference["fingerprint"],
                dtype=torch.float64,
            )
            fingerprint_error = float(
                torch.sqrt(
                    torch.mean((endpoint["fingerprint"] - reference_fingerprint) ** 2)
                ).item()
            )
            valid &= energy_error <= energy_tolerance
            valid &= fingerprint_error <= fingerprint_tolerance
            total_error += energy_error / energy_tolerance
            total_error += fingerprint_error / fingerprint_tolerance
        if total_error < best_error:
            best_error = total_error
            best_valid = valid
    return best_valid, best_error


def prepare_reference(args: argparse.Namespace) -> None:
    torch.set_num_threads(1)
    predictor = make_lj_predict_fn()
    atomic_nums = lj_atomic_nums(7)
    minimum = pentagonal_bipyramid_geometry()
    symbols = atomic_nums_to_symbols(atomic_nums)
    minimum_out = predictor(minimum, atomic_nums, do_hessian=True)
    evals, modes_mw, _ = vib_eig(minimum_out["hessian"], minimum, symbols)
    _, _, _, inv_sqrt_mass = get_mass_weights(symbols)
    pushed = minimum + args.push_amplitude * (inv_sqrt_mass * modes_mw[:, 0]).reshape_as(minimum)
    result = run_intrinsic_gad(
        predictor,
        pushed,
        atomic_nums,
        IntrinsicGADConfig(max_steps=100, record_history=True),
    )
    if not result.converged:
        raise RuntimeError(f"reference construction failed: {result.failure_type}")
    reference_endpoints = _downhill_endpoints(
        predictor,
        atomic_nums,
        result.final_coords.to(torch.float64),
        displacement=0.03,
    )
    if any(endpoint["n_neg"] != 0 for endpoint in reference_endpoints):
        raise RuntimeError("a reference downhill branch did not reach a projected minimum")

    payload = {
        "surface": "reduced_lj7",
        "epsilon": 1.0,
        "sigma": 1.0,
        "mass": 1.008,
        "push_mode": 0,
        "push_amplitude_mw": args.push_amplitude,
        "minimum_eigenvalues_first_six": [float(x) for x in evals[:6]],
        "minimum_coords": minimum.tolist(),
        "pushed_coords": pushed.tolist(),
        "reference_ts_coords": result.final_coords.tolist(),
        "reference_energy": result.final_energy,
        "reference_n_neg": result.final_n_neg,
        "reference_force_max": result.final_force_max,
        "reference_pair_fingerprint": _fingerprint(result.final_coords).tolist(),
        "reference_steps": result.total_steps,
        "downhill_validation": {
            "kind": "unstable-mode displacement followed by L-BFGS energy minimization",
            "displacement_sigma": 0.03,
            "endpoints": [_endpoint_payload(endpoint) for endpoint in reference_endpoints],
        },
        "optimizer": {
            "name": "pointwise_intrinsic_gad",
            "spectral_temperature": 0.01,
            "step_fraction": 0.05,
            "force_threshold": 0.01,
            "index_threshold": 1.0e-4,
        },
    }
    _write_json(args.output_root / "reference.json", payload)
    print(
        f"reference energy={result.final_energy:.12f} fmax={result.final_force_max:.3e} "
        f"steps={result.total_steps}",
        flush=True,
    )


def _load_reference(output_root: Path) -> dict[str, Any]:
    return json.loads((output_root / "reference.json").read_text())


def _parse_noises(text: str) -> list[float]:
    noises = [float(value) for value in text.split(",") if value.strip()]
    if not noises or min(noises) < 0:
        raise ValueError("noises must contain nonnegative values")
    return noises


def _task_grid(noises: list[float], n_samples: int) -> list[tuple[str, float, int]]:
    return [
        (panel, noise, sample_id)
        for panel in ("saddle", "pushed")
        for noise in noises
        for sample_id in range(n_samples)
    ]


def run_worker(args: argparse.Namespace) -> None:
    if not 0 <= args.shard_id < args.n_shards:
        raise ValueError("shard-id must satisfy 0 <= shard-id < n-shards")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    reference = _load_reference(args.output_root)
    reference_energy = float(reference["reference_energy"])
    reference_fingerprint = torch.tensor(
        reference["reference_pair_fingerprint"],
        dtype=torch.float64,
    )
    reference_endpoints = reference["downhill_validation"]["endpoints"]
    bases = {
        "saddle": torch.tensor(reference["reference_ts_coords"], dtype=torch.float64),
        "pushed": torch.tensor(reference["pushed_coords"], dtype=torch.float64),
    }
    predictor = make_lj_predict_fn()
    atomic_nums = lj_atomic_nums(7)
    config = IntrinsicGADConfig(max_steps=args.max_steps, record_history=True)
    tasks = _task_grid(_parse_noises(args.noises), args.n_samples)
    assigned = tasks[args.shard_id :: args.n_shards]
    rows: list[dict[str, Any]] = []

    for panel, noise, sample_id in assigned:
        seed = args.seed + sample_id
        generator = torch.Generator().manual_seed(seed)
        perturbation = torch.randn(
            bases[panel].shape,
            generator=generator,
            dtype=torch.float64,
        )
        start = bases[panel] + noise * perturbation
        start = start - start.mean(dim=0, keepdim=True)
        row: dict[str, Any] = {
            "panel": panel,
            "noise": noise,
            "sample_id": sample_id,
            "seed": seed,
            "converged": False,
            "downhill_valid": False,
            "correct_event": False,
            "correct_reference": False,
            "energy_match": False,
            "geometry_match": False,
            "final_energy": math.nan,
            "reference_energy_delta": math.nan,
            "pair_fingerprint_rms": math.nan,
            "final_n_neg": -1,
            "final_force_max": math.inf,
            "final_gate_weight": math.nan,
            "initial_n_neg": -1,
            "initial_gate_weight": math.nan,
            "total_steps": 0,
            "n_evaluations": 0,
            "failure_type": "",
            "error": "",
            "endpoint_energy_1": math.nan,
            "endpoint_energy_2": math.nan,
            "endpoint_n_neg_1": -1,
            "endpoint_n_neg_2": -1,
            "endpoint_force_max_1": math.inf,
            "endpoint_force_max_2": math.inf,
            "endpoint_pair_error": math.nan,
            "final_coords": "",
        }
        try:
            result = run_intrinsic_gad(predictor, start, atomic_nums, config)
            final_fingerprint = _fingerprint(result.final_coords)
            fingerprint_rms = float(
                torch.sqrt(torch.mean((final_fingerprint - reference_fingerprint) ** 2)).item()
            )
            energy_delta = result.final_energy - reference_energy
            energy_match = abs(energy_delta) <= args.energy_tolerance
            geometry_match = fingerprint_rms <= args.fingerprint_tolerance
            correct_reference = result.converged and energy_match and geometry_match
            initial = result.history[0] if result.history else None
            endpoints: list[dict[str, Any]] = []
            downhill_valid = False
            correct_event = False
            endpoint_pair_error = math.nan
            if result.converged:
                endpoints = _downhill_endpoints(
                    predictor,
                    atomic_nums,
                    result.final_coords.to(torch.float64),
                    displacement=args.endpoint_displacement,
                )
                downhill_valid = all(
                    endpoint["n_neg"] == 0 and endpoint["force_max"] < args.endpoint_force_threshold
                    for endpoint in endpoints
                )
                endpoint_match, endpoint_pair_error = _match_endpoint_pair(
                    endpoints,
                    reference_endpoints,
                    energy_tolerance=args.energy_tolerance,
                    fingerprint_tolerance=args.fingerprint_tolerance,
                )
                correct_event = downhill_valid and endpoint_match
            row.update(
                {
                    "converged": result.converged,
                    "downhill_valid": downhill_valid,
                    "correct_event": correct_event,
                    "correct_reference": correct_reference,
                    "energy_match": energy_match,
                    "geometry_match": geometry_match,
                    "final_energy": result.final_energy,
                    "reference_energy_delta": energy_delta,
                    "pair_fingerprint_rms": fingerprint_rms,
                    "final_n_neg": result.final_n_neg,
                    "final_force_max": result.final_force_max,
                    "final_gate_weight": result.final_gate_weight,
                    "initial_n_neg": initial.n_neg if initial else result.final_n_neg,
                    "initial_gate_weight": initial.gate_weight
                    if initial
                    else result.final_gate_weight,
                    "total_steps": result.total_steps,
                    "n_evaluations": result.n_evaluations,
                    "failure_type": result.failure_type or "",
                    "endpoint_energy_1": endpoints[0]["energy"] if endpoints else math.nan,
                    "endpoint_energy_2": endpoints[1]["energy"] if endpoints else math.nan,
                    "endpoint_n_neg_1": endpoints[0]["n_neg"] if endpoints else -1,
                    "endpoint_n_neg_2": endpoints[1]["n_neg"] if endpoints else -1,
                    "endpoint_force_max_1": endpoints[0]["force_max"] if endpoints else math.inf,
                    "endpoint_force_max_2": endpoints[1]["force_max"] if endpoints else math.inf,
                    "endpoint_pair_error": endpoint_pair_error,
                    "final_coords": json.dumps(result.final_coords.tolist(), separators=(",", ":")),
                }
            )
        except Exception as exc:  # noqa: BLE001 - failures are experimental outcomes.
            row["failure_type"] = "exception"
            row["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)

    shard_dir = args.output_root / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    output = shard_dir / f"shard_{args.shard_id:03d}.csv"
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    print(f"shard={args.shard_id} tasks={len(rows)} output={output}", flush=True)


def _as_bool(value: str) -> bool:
    return value.lower() == "true"


def _wilson_interval(successes: int, total: int) -> tuple[float, float]:
    if total == 0:
        return math.nan, math.nan
    z = 1.959963984540054
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    half = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total))
    half /= denominator
    return center - half, center + half


def aggregate(args: argparse.Namespace) -> None:
    shard_paths = sorted((args.output_root / "shards").glob("shard_*.csv"))
    if len(shard_paths) != args.expected_shards:
        raise RuntimeError(f"expected {args.expected_shards} shards, found {len(shard_paths)}")
    rows: list[dict[str, str]] = []
    for path in shard_paths:
        with path.open(newline="") as handle:
            rows.extend(csv.DictReader(handle))

    groups: dict[tuple[str, float], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[(row["panel"], float(row["noise"]))].append(row)

    summary_fields = (
        "panel",
        "noise_sigma",
        "n",
        "strict_converged",
        "strict_rate",
        "downhill_valid",
        "downhill_valid_rate",
        "correct_event",
        "correct_event_rate",
        "correct_event_ci95_low",
        "correct_event_ci95_high",
        "correct_reference",
        "correct_rate",
        "wrong_saddle",
        "exceptions",
        "median_steps_correct",
        "median_initial_n_neg",
    )
    summary_rows: list[dict[str, Any]] = []
    for (panel, noise), group in sorted(groups.items()):
        strict = sum(_as_bool(row["converged"]) for row in group)
        downhill_valid = sum(_as_bool(row["downhill_valid"]) for row in group)
        correct_event = sum(_as_bool(row["correct_event"]) for row in group)
        correct_reference = sum(_as_bool(row["correct_reference"]) for row in group)
        wrong = sum(
            _as_bool(row["converged"]) and not _as_bool(row["correct_event"]) for row in group
        )
        exceptions = sum(row["failure_type"] == "exception" for row in group)
        correct_steps = [int(row["total_steps"]) for row in group if _as_bool(row["correct_event"])]
        initial_indices = [
            int(row["initial_n_neg"]) for row in group if int(row["initial_n_neg"]) >= 0
        ]
        ci_low, ci_high = _wilson_interval(correct_event, len(group))
        summary_rows.append(
            {
                "panel": panel,
                "noise_sigma": noise,
                "n": len(group),
                "strict_converged": strict,
                "strict_rate": strict / len(group),
                "downhill_valid": downhill_valid,
                "downhill_valid_rate": downhill_valid / len(group),
                "correct_event": correct_event,
                "correct_event_rate": correct_event / len(group),
                "correct_event_ci95_low": ci_low,
                "correct_event_ci95_high": ci_high,
                "correct_reference": correct_reference,
                "correct_rate": correct_reference / len(group),
                "wrong_saddle": wrong,
                "exceptions": exceptions,
                "median_steps_correct": statistics.median(correct_steps)
                if correct_steps
                else math.nan,
                "median_initial_n_neg": statistics.median(initial_indices)
                if initial_indices
                else math.nan,
            }
        )

    with (args.output_root / "all_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    with (args.output_root / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    lines = [
        "# Pointwise intrinsic GAD LJ7 noise sweep",
        "",
        (
            "The primary correct-event outcome requires strict TS convergence, two valid "
            "downhill projected minima, and the same unordered endpoint-basin pair as the reference."
        ),
        "",
        (
            "| Panel | Noise / sigma | n | Strict | Valid downhill | Correct event | "
            "Same TS geometry | Other event | Exceptions | Median correct steps | Median initial index |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['panel']} | {row['noise_sigma']:.3f} | {row['n']} | "
            f"{100 * row['strict_rate']:.1f}% | {100 * row['downhill_valid_rate']:.1f}% | "
            f"{100 * row['correct_event_rate']:.1f}% | {100 * row['correct_rate']:.1f}% | "
            f"{row['wrong_saddle']} | {row['exceptions']} | "
            f"{row['median_steps_correct']} | {row['median_initial_n_neg']} |"
        )
    (args.output_root / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    print(f"aggregated rows={len(rows)} groups={len(summary_rows)}", flush=True)


def main() -> None:
    args = parse_args()
    if args.command == "prepare":
        prepare_reference(args)
    elif args.command == "worker":
        run_worker(args)
    else:
        aggregate(args)


if __name__ == "__main__":
    main()
