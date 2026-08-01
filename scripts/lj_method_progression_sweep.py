#!/usr/bin/env python3
"""Paired LJ7 sweep of the four GAD formulations in the method derivation.

Every method receives exactly the same deterministic noised starts and the
same strict terminal test: projected Morse index one, ``fmax < 0.01``, then
two downhill branches relaxed to projected minima.  The three Euler fields
are deliberately implemented here with the *instantaneous* lowest mode:

``ordinary_gad``
    Full one-mode GAD at every point.
``hard_gate``
    Descent for ``lambda_2 < 0`` and GAD for ``lambda_2 >= 0``.
``historical_lambda2``
    ``sigmoid(50 lambda_2)`` blend from the historical LJ runner.
``intrinsic``
    The current closed-form, scale-covariant intrinsic method.

This is a controlled analytic-LJ comparison, not a molecular benchmark and
not a claim that one method is universally superior on every PES.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from ase import Atoms
from sella import Sella

from gadplus.calculator.lennard_jones import (
    lj_atomic_nums,
    make_lj_predict_fn,
    pair_distances,
)
from gadplus.calculator.sella import (
    FullHessianASECalculator,
    full_hessian_function,
    refresh_hessian_after_kicks,
)
from gadplus.core.adaptive_dt import cap_displacement
from gadplus.core.convergence import force_max, is_ts_converged
from gadplus.projection import atomic_nums_to_symbols, gad_dynamics_projected, vib_eig
from gadplus.search.intrinsic_gad import IntrinsicGADConfig, run_intrinsic_gad
from lj_intrinsic_noise_sweep import (
    _downhill_endpoints,
    _fingerprint,
    _match_endpoint_pair,
    _parse_noises,
    _wilson_interval,
    prepare_reference,
)


METHODS = ("ordinary_gad", "hard_gate", "historical_lambda2", "intrinsic", "sella")
EULER_METHODS = frozenset(METHODS[:3])
DEFAULT_NOISES = "0.10,0.20,0.40"
FIELDNAMES = (
    "method",
    "panel",
    "noise",
    "sample_id",
    "seed",
    "initial_n_neg",
    "initial_lambda2",
    "converged",
    "downhill_valid",
    "correct_event",
    "correct_reference",
    "total_steps",
    "n_evaluations",
    "wall_time_s",
    "final_energy",
    "final_force_max",
    "final_n_neg",
    "final_lambda1",
    "final_lambda2",
    "final_lambda2_scaled",
    "final_connected_cutoff_1p5",
    "final_max_pair_distance",
    "failure_type",
    "error",
    "endpoint_energy_1",
    "endpoint_energy_2",
    "endpoint_n_neg_1",
    "endpoint_n_neg_2",
    "endpoint_force_max_1",
    "endpoint_force_max_2",
    "final_coords",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="Construct the common reference event.")
    prepare.add_argument("--output-root", type=Path, required=True)
    prepare.add_argument("--push-amplitude", type=float, default=0.26)

    worker = subparsers.add_parser("worker", help="Run one paired method/start task.")
    worker.add_argument("--output-root", type=Path, required=True)
    worker.add_argument("--task-id", type=int, required=True)
    worker.add_argument("--n-samples", type=int, default=48)
    worker.add_argument("--noises", default=DEFAULT_NOISES)
    worker.add_argument("--methods", default=",".join(METHODS))
    worker.add_argument("--seed", type=int, default=20260727)
    worker.add_argument("--euler-max-steps", type=int, default=8000)
    worker.add_argument("--intrinsic-max-steps", type=int, default=200)
    worker.add_argument("--dt", type=float, default=0.005)
    worker.add_argument("--max-atom-disp", type=float, default=0.005)

    aggregate = subparsers.add_parser("aggregate", help="Aggregate all paired tasks.")
    aggregate.add_argument("--output-root", type=Path, required=True)
    aggregate.add_argument("--expected-tasks", type=int, required=True)
    return parser.parse_args()


def _parse_methods(text: str) -> tuple[str, ...]:
    methods = tuple(item.strip() for item in text.split(",") if item.strip())
    if not methods or any(method not in METHODS for method in methods):
        raise ValueError(f"methods must be a nonempty subset of {METHODS}")
    if len(set(methods)) != len(methods):
        raise ValueError("methods must not repeat a method")
    return methods


def _task_grid(
    methods: tuple[str, ...], noises: list[float], n_samples: int
) -> list[tuple[str, str, float, int]]:
    return [
        (method, panel, noise, sample_id)
        for method in methods
        for panel in ("saddle", "pushed")
        for noise in noises
        for sample_id in range(n_samples)
    ]


def _connected(coords: torch.Tensor, cutoff: float = 1.5) -> bool:
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


def _evaluate_spectrum(predictor, atomic_nums: torch.Tensor, coords: torch.Tensor) -> tuple[Any, torch.Tensor]:
    out = predictor(coords, atomic_nums, do_hessian=True)
    evals, modes, _ = vib_eig(
        out["hessian"], coords, atomic_nums_to_symbols(atomic_nums)
    )
    return out, evals, modes


def _run_euler_field(
    method: str,
    predictor,
    atomic_nums: torch.Tensor,
    start: torch.Tensor,
    *,
    max_steps: int,
    dt: float,
    max_atom_disp: float,
) -> dict[str, Any]:
    """Run one documented instantaneous Euler field in float64 arithmetic."""

    if method not in EULER_METHODS:
        raise ValueError(f"not an Euler method: {method}")
    coords = start.detach().clone().to(torch.float64).reshape(-1, 3)
    symbols = atomic_nums_to_symbols(atomic_nums)
    began = time.monotonic()
    last: dict[str, Any] | None = None

    for step in range(max_steps):
        out = predictor(coords, atomic_nums, do_hessian=True)
        forces = out["forces"].reshape_as(coords).to(torch.float64)
        evals, modes, _ = vib_eig(out["hessian"], coords, symbols)
        n_neg = int((evals < -1.0e-4).sum().item())
        fmax = force_max(forces)
        last = {
            "coords": coords,
            "energy": float(out["energy"].item()),
            "n_neg": n_neg,
            "force_max": fmax,
            "lambda1": float(evals[0].item()),
            "lambda2": float(evals[1].item()),
            "evals": evals,
        }
        if is_ts_converged(n_neg, fmax, 0.01, criterion="fmax"):
            return {
                **last,
                "converged": True,
                "total_steps": step + 1,
                "n_evaluations": step + 1,
                "wall_time_s": time.monotonic() - began,
            }

        if method == "ordinary_gad":
            gate = 1.0
        elif method == "hard_gate":
            gate = 1.0 if float(evals[1].item()) >= 0.0 else 0.0
        else:
            gate = float(torch.sigmoid(50.0 * evals[1]).item())
        direction, _, _ = gad_dynamics_projected(
            coords, forces, modes[:, 0], symbols, gad_blend_weight=gate
        )
        coords = (coords + cap_displacement(dt * direction, max_atom_disp)).detach()

    assert last is not None
    return {
        **last,
        "converged": False,
        "total_steps": max_steps,
        "n_evaluations": max_steps,
        "wall_time_s": time.monotonic() - began,
    }


def _run_method(
    method: str,
    predictor,
    atomic_nums: torch.Tensor,
    start: torch.Tensor,
    args: argparse.Namespace,
) -> dict[str, Any]:
    if method in EULER_METHODS:
        return _run_euler_field(
            method,
            predictor,
            atomic_nums,
            start,
            max_steps=args.euler_max_steps,
            dt=args.dt,
            max_atom_disp=args.max_atom_disp,
        )

    if method == "sella":
        return _run_sella(predictor, atomic_nums, start, args)

    result = run_intrinsic_gad(
        predictor,
        start,
        atomic_nums,
        IntrinsicGADConfig(max_steps=args.intrinsic_max_steps, record_history=False),
    )
    out, evals, _ = _evaluate_spectrum(predictor, atomic_nums, result.final_coords)
    return {
        "coords": result.final_coords.to(torch.float64),
        "energy": float(out["energy"].item()),
        "n_neg": int((evals < -1.0e-4).sum().item()),
        "force_max": force_max(out["forces"]),
        "lambda1": float(evals[0].item()),
        "lambda2": float(evals[1].item()),
        "evals": evals,
        "converged": result.converged,
        "total_steps": result.total_steps,
        "n_evaluations": result.n_evaluations,
        "wall_time_s": result.wall_time_s,
    }


def _run_sella(
    predictor,
    atomic_nums: torch.Tensor,
    start: torch.Tensor,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Cartesian Sella with an exact, Eckart-cleaned Hessian after each kick."""
    coords = start.detach().clone().to(torch.float64).reshape(-1, 3)
    atoms = Atoms(numbers=atomic_nums.detach().cpu().numpy(), positions=coords.cpu().numpy())
    calculator = FullHessianASECalculator(predictor, atomic_nums, "cpu")
    atoms.calc = calculator
    began = time.monotonic()
    try:
        optimizer = Sella(
            atoms=atoms, order=1, internal=False, logfile=None,
            delta0=0.1, gamma=0.4, diag_every_n=1,
            hessian_function=full_hessian_function(calculator, eckart_project=True),
            rho_inc=1.035, rho_dec=5.0, sigma_inc=1.15, sigma_dec=0.65,
        )
        refresh_hessian_after_kicks(optimizer.pes)
        optimizer.run(fmax=0.01, steps=args.euler_max_steps)
        steps = int(optimizer.nsteps)
    except Exception:
        steps = args.euler_max_steps
    final_coords = torch.as_tensor(atoms.positions, dtype=torch.float64)
    out, evals, _ = _evaluate_spectrum(predictor, atomic_nums, final_coords)
    forces = out["forces"].reshape_as(final_coords)
    n_neg = int((evals < -1.0e-4).sum().item())
    fmax = force_max(forces)
    return {
        "coords": final_coords, "energy": float(out["energy"].item()),
        "n_neg": n_neg, "force_max": fmax,
        "lambda1": float(evals[0].item()), "lambda2": float(evals[1].item()),
        "evals": evals, "converged": is_ts_converged(n_neg, fmax, 0.01, criterion="fmax"),
        "total_steps": steps, "n_evaluations": calculator.n_evaluations + 1,
        "wall_time_s": time.monotonic() - began,
    }


def _empty_row(method: str, panel: str, noise: float, sample_id: int, seed: int) -> dict[str, Any]:
    return {
        "method": method,
        "panel": panel,
        "noise": noise,
        "sample_id": sample_id,
        "seed": seed,
        "initial_n_neg": -1,
        "initial_lambda2": math.nan,
        "converged": False,
        "downhill_valid": False,
        "correct_event": False,
        "correct_reference": False,
        "total_steps": 0,
        "n_evaluations": 0,
        "wall_time_s": math.nan,
        "final_energy": math.nan,
        "final_force_max": math.inf,
        "final_n_neg": -1,
        "final_lambda1": math.nan,
        "final_lambda2": math.nan,
        "final_lambda2_scaled": math.nan,
        "final_connected_cutoff_1p5": False,
        "final_max_pair_distance": math.nan,
        "failure_type": "",
        "error": "",
        "endpoint_energy_1": math.nan,
        "endpoint_energy_2": math.nan,
        "endpoint_n_neg_1": -1,
        "endpoint_n_neg_2": -1,
        "endpoint_force_max_1": math.inf,
        "endpoint_force_max_2": math.inf,
        "final_coords": "",
    }


def run_worker(args: argparse.Namespace) -> None:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    methods = _parse_methods(args.methods)
    tasks = _task_grid(methods, _parse_noises(args.noises), args.n_samples)
    if not 0 <= args.task_id < len(tasks):
        raise ValueError(f"task-id must be in [0, {len(tasks)})")
    method, panel, noise, sample_id = tasks[args.task_id]
    seed = args.seed + sample_id
    row = _empty_row(method, panel, noise, sample_id, seed)
    try:
        reference = json.loads((args.output_root / "reference.json").read_text())
        reference_energy = float(reference["reference_energy"])
        reference_fingerprint = torch.tensor(
            reference["reference_pair_fingerprint"], dtype=torch.float64
        )
        reference_endpoints = reference["downhill_validation"]["endpoints"]
        base_key = "reference_ts_coords" if panel == "saddle" else "pushed_coords"
        base = torch.tensor(reference[base_key], dtype=torch.float64)
        generator = torch.Generator().manual_seed(seed)
        start = base + noise * torch.randn(base.shape, generator=generator, dtype=base.dtype)
        start = start - start.mean(dim=0, keepdim=True)

        predictor = make_lj_predict_fn()
        atomic_nums = lj_atomic_nums(7)
        _, initial_evals, _ = _evaluate_spectrum(predictor, atomic_nums, start)
        row["initial_n_neg"] = int((initial_evals < -1.0e-4).sum().item())
        row["initial_lambda2"] = float(initial_evals[1].item())
        result = _run_method(method, predictor, atomic_nums, start, args)
        final_coords = result["coords"]
        final_fingerprint = _fingerprint(final_coords)
        spectral_scale = float(torch.sqrt(torch.mean(result["evals"].square())).item())
        endpoints: list[dict[str, Any]] = []
        downhill_valid = False
        correct_event = False
        if result["converged"]:
            endpoints = _downhill_endpoints(predictor, atomic_nums, final_coords, displacement=0.03)
            downhill_valid = all(
                endpoint["n_neg"] == 0 and endpoint["force_max"] < 1.0e-5
                for endpoint in endpoints
            )
            endpoint_match, _ = _match_endpoint_pair(
                endpoints,
                reference_endpoints,
                energy_tolerance=1.0e-4,
                fingerprint_tolerance=1.0e-3,
            )
            correct_event = downhill_valid and endpoint_match
        row.update(
            {
                "converged": result["converged"],
                "downhill_valid": downhill_valid,
                "correct_event": correct_event,
                "correct_reference": (
                    result["converged"]
                    and abs(result["energy"] - reference_energy) <= 1.0e-4
                    and float(torch.sqrt(torch.mean((final_fingerprint - reference_fingerprint) ** 2)).item())
                    <= 1.0e-3
                ),
                "total_steps": result["total_steps"],
                "n_evaluations": result["n_evaluations"],
                "wall_time_s": result["wall_time_s"],
                "final_energy": result["energy"],
                "final_force_max": result["force_max"],
                "final_n_neg": result["n_neg"],
                "final_lambda1": result["lambda1"],
                "final_lambda2": result["lambda2"],
                "final_lambda2_scaled": result["lambda2"] / spectral_scale,
                "final_connected_cutoff_1p5": _connected(final_coords),
                "final_max_pair_distance": float(pair_distances(final_coords).max().item()),
                "endpoint_energy_1": endpoints[0]["energy"] if endpoints else math.nan,
                "endpoint_energy_2": endpoints[1]["energy"] if endpoints else math.nan,
                "endpoint_n_neg_1": endpoints[0]["n_neg"] if endpoints else -1,
                "endpoint_n_neg_2": endpoints[1]["n_neg"] if endpoints else -1,
                "endpoint_force_max_1": endpoints[0]["force_max"] if endpoints else math.inf,
                "endpoint_force_max_2": endpoints[1]["force_max"] if endpoints else math.inf,
                "final_coords": json.dumps(final_coords.tolist(), separators=(",", ":")),
            }
        )
    except Exception as exc:  # noqa: BLE001 - errors are experimental outcomes.
        row["failure_type"] = "exception"
        row["error"] = f"{type(exc).__name__}: {exc}"

    task_dir = args.output_root / "tasks"
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / f"task_{args.task_id:04d}.json").write_text(
        json.dumps(row, indent=2, sort_keys=True) + "\n"
    )


def _median(values: list[float]) -> float:
    return float(statistics.median(values)) if values else math.nan


def aggregate(args: argparse.Namespace) -> None:
    paths = sorted((args.output_root / "tasks").glob("task_*.json"))
    if len(paths) != args.expected_tasks:
        raise RuntimeError(f"expected {args.expected_tasks} tasks, found {len(paths)}")
    rows = [json.loads(path.read_text()) for path in paths]
    groups: dict[tuple[str, str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["method"], row["panel"], float(row["noise"]))].append(row)

    summary = []
    for (method, panel, noise), group in sorted(groups.items()):
        strict = [row for row in group if row["converged"]]
        valid = [row for row in group if row["downhill_valid"]]
        correct = [row for row in group if row["correct_event"]]
        summary.append(
            {
                "method": method,
                "panel": panel,
                "noise_sigma": noise,
                "n": len(group),
                "strict": len(strict),
                "strict_rate": len(strict) / len(group),
                "downhill_valid": len(valid),
                "downhill_valid_rate": len(valid) / len(group),
                "correct_event": len(correct),
                "correct_event_rate": len(correct) / len(group),
                "correct_event_ci95": _wilson_interval(len(correct), len(group)),
                "near_flat_valid": sum(row["final_lambda2_scaled"] < 0.01 for row in valid),
                "fragmented_valid": sum(not row["final_connected_cutoff_1p5"] for row in valid),
                "exceptions": sum(row["failure_type"] == "exception" for row in group),
                "median_strict_evaluations": _median([row["n_evaluations"] for row in strict]),
                "median_valid_evaluations": _median([row["n_evaluations"] for row in valid]),
                "median_strict_wall_s": _median([row["wall_time_s"] for row in strict]),
            }
        )

    (args.output_root / "all_results.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n"
    )
    (args.output_root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    with (args.output_root / "all_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Paired LJ7 progression sweep",
        "",
        "All methods use identical starts and the same strict TS and two-branch endpoint tests.",
        "The three predecessor fields use `dt=0.005`, a per-atom cap of `0.005`,",
        "and up to 8000 Hessian evaluations; intrinsic GAD uses its documented 200-step bound.",
        "",
        "| Method | Panel | Noise | n | Strict | Valid endpoints | Same event | Median strict evaluations | Near-flat valid | Fragmented valid | Exceptions |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['method']} | {row['panel']} | {row['noise_sigma']:.2f} | {row['n']} | "
            f"{100 * row['strict_rate']:.1f}% | {100 * row['downhill_valid_rate']:.1f}% | "
            f"{100 * row['correct_event_rate']:.1f}% | {row['median_strict_evaluations']:.1f} | "
            f"{row['near_flat_valid']} | {row['fragmented_valid']} | {row['exceptions']} |"
        )
    (args.output_root / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    print(f"aggregated rows={len(rows)} groups={len(summary)}", flush=True)


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
