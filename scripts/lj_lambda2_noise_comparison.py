#!/usr/bin/env python3
"""Two-point comparison with the historical fixed-Euler lambda2 gate."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import torch
from lj_intrinsic_noise_sweep import (
    _downhill_endpoints,
    _fingerprint,
    _match_endpoint_pair,
    _wilson_interval,
)

from gadplus.calculator.lennard_jones import lj_atomic_nums, make_lj_predict_fn
from gadplus.projection import atomic_nums_to_symbols, vib_eig
from gadplus.search.gad_search import GADSearchConfig, run_gad_search


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    worker = subparsers.add_parser("worker")
    worker.add_argument("--output-root", type=Path, required=True)
    worker.add_argument("--task-id", type=int, required=True)
    worker.add_argument("--n-samples", type=int, default=48)
    worker.add_argument("--noises", default="0.10,0.20")
    worker.add_argument("--seed", type=int, default=20260726)
    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--output-root", type=Path, required=True)
    aggregate.add_argument("--expected-tasks", type=int, required=True)
    return parser.parse_args()


def _tasks(noises: str, n_samples: int) -> list[tuple[str, float, int]]:
    return [
        (panel, float(noise), sample_id)
        for panel in ("saddle", "pushed")
        for noise in noises.split(",")
        for sample_id in range(n_samples)
    ]


def run_worker(args: argparse.Namespace) -> None:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    tasks = _tasks(args.noises, args.n_samples)
    if not 0 <= args.task_id < len(tasks):
        raise ValueError(f"task-id must be in [0, {len(tasks)})")
    panel, noise, sample_id = tasks[args.task_id]
    reference = json.loads((args.output_root / "reference.json").read_text())
    reference_endpoints = reference["downhill_validation"]["endpoints"]
    reference_energy = float(reference["reference_energy"])
    reference_fingerprint = torch.tensor(
        reference["reference_pair_fingerprint"], dtype=torch.float64
    )
    base = torch.tensor(
        reference["reference_ts_coords"] if panel == "saddle" else reference["pushed_coords"],
        dtype=torch.float64,
    )
    seed = args.seed + sample_id
    generator = torch.Generator().manual_seed(seed)
    start = base + noise * torch.randn(base.shape, generator=generator, dtype=base.dtype)
    start = start - start.mean(dim=0, keepdim=True)

    predictor = make_lj_predict_fn()
    atomic_nums = lj_atomic_nums(7)
    initial_out = predictor(start, atomic_nums, do_hessian=True)
    initial_evals, _, _ = vib_eig(
        initial_out["hessian"], start, atomic_nums_to_symbols(atomic_nums)
    )
    config = GADSearchConfig(
        n_steps=8000,
        dt=0.005,
        k_track=0,
        use_projection=True,
        use_adaptive_dt=False,
        max_atom_disp=0.005,
        min_interatomic_dist=0.0,
        force_threshold=0.01,
        force_criterion="fmax",
        blend_sharpness=50.0,
    )
    row = {
        "method": "historical_lambda2_k50_euler",
        "panel": panel,
        "noise": noise,
        "sample_id": sample_id,
        "seed": seed,
        "initial_n_neg": int((initial_evals < -1.0e-4).sum().item()),
        "initial_gate_weight": float(torch.sigmoid(50.0 * initial_evals[1]).item()),
        "converged": False,
        "downhill_valid": False,
        "correct_event": False,
        "same_ts_geometry": False,
        "total_steps": 0,
        "final_energy": math.nan,
        "final_force_max": math.inf,
        "final_n_neg": -1,
        "failure_type": "",
        "error": "",
    }
    try:
        result = run_gad_search(predictor, start, atomic_nums, config)
        final_fingerprint = _fingerprint(result.final_coords)
        fingerprint_error = float(
            torch.sqrt(torch.mean((final_fingerprint - reference_fingerprint) ** 2)).item()
        )
        same_ts = (
            result.converged
            and abs(result.final_energy - reference_energy) <= 1.0e-4
            and fingerprint_error <= 1.0e-3
        )
        endpoints = []
        downhill_valid = False
        correct_event = False
        if result.converged:
            endpoints = _downhill_endpoints(
                predictor,
                atomic_nums,
                result.final_coords.to(torch.float64),
                displacement=0.03,
            )
            downhill_valid = all(
                endpoint["n_neg"] == 0 and endpoint["force_max"] < 1.0e-5 for endpoint in endpoints
            )
            event_match, _ = _match_endpoint_pair(
                endpoints,
                reference_endpoints,
                energy_tolerance=1.0e-4,
                fingerprint_tolerance=1.0e-3,
            )
            correct_event = downhill_valid and event_match
        row.update(
            {
                "converged": result.converged,
                "downhill_valid": downhill_valid,
                "correct_event": correct_event,
                "same_ts_geometry": same_ts,
                "total_steps": result.total_steps,
                "final_energy": result.final_energy,
                "final_force_max": result.final_force_max,
                "final_n_neg": result.final_n_neg,
                "failure_type": result.failure_type or "",
                "endpoint_energies": [endpoint["energy"] for endpoint in endpoints],
            }
        )
    except Exception as exc:  # noqa: BLE001 - failures are experiment outcomes.
        row["failure_type"] = "exception"
        row["error"] = f"{type(exc).__name__}: {exc}"

    output_dir = args.output_root / "lambda2_tasks"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"task_{args.task_id:03d}.json").write_text(
        json.dumps(row, indent=2, sort_keys=True) + "\n"
    )


def aggregate(args: argparse.Namespace) -> None:
    paths = sorted((args.output_root / "lambda2_tasks").glob("task_*.json"))
    if len(paths) != args.expected_tasks:
        raise RuntimeError(f"expected {args.expected_tasks} tasks, found {len(paths)}")
    rows = [json.loads(path.read_text()) for path in paths]
    groups = defaultdict(list)
    for row in rows:
        groups[(row["panel"], row["noise"])].append(row)
    lines = [
        "# Historical lambda2-gated Euler comparison",
        "",
        "Configuration: `k=50`, `dt=0.005`, per-atom cap `0.005`, 8000-step budget.",
        "",
        "| Panel | Noise / sigma | n | Strict | Valid downhill | Correct event | Same TS | Median steps | Exceptions |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    summary = []
    for (panel, noise), group in sorted(groups.items()):
        n = len(group)
        strict = sum(row["converged"] for row in group)
        valid = sum(row["downhill_valid"] for row in group)
        correct = sum(row["correct_event"] for row in group)
        same_ts = sum(row["same_ts_geometry"] for row in group)
        exceptions = sum(row["failure_type"] == "exception" for row in group)
        steps = [row["total_steps"] for row in group if row["correct_event"]]
        ci_low, ci_high = _wilson_interval(correct, n)
        record = {
            "panel": panel,
            "noise": noise,
            "n": n,
            "strict_rate": strict / n,
            "downhill_valid_rate": valid / n,
            "correct_event_rate": correct / n,
            "correct_event_ci95": [ci_low, ci_high],
            "same_ts_rate": same_ts / n,
            "median_correct_steps": statistics.median(steps) if steps else math.nan,
            "exceptions": exceptions,
        }
        summary.append(record)
        lines.append(
            f"| {panel} | {noise:.3f} | {n} | {100 * strict / n:.1f}% | "
            f"{100 * valid / n:.1f}% | {100 * correct / n:.1f}% | {100 * same_ts / n:.1f}% | "
            f"{record['median_correct_steps']} | {exceptions} |"
        )
    (args.output_root / "lambda2_all_results.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n"
    )
    (args.output_root / "lambda2_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    (args.output_root / "LAMBDA2_SUMMARY.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    if args.command == "worker":
        run_worker(args)
    else:
        aggregate(args)


if __name__ == "__main__":
    main()
