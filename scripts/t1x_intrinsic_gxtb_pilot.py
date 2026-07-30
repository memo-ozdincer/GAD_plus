#!/usr/bin/env python3
"""Packed Transition1x competitive-GAD worker on the coherent g-xTB surface.

Compute workers write exact local trajectory bundles only. W&B export is a
separate post-run operation and cannot affect the pointwise optimizer.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch


def _parse_ids(text: str) -> list[int]:
    values = [int(item) for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("at least one sample id is required")
    return values


def _starts(text: str) -> tuple[tuple[str, float], ...]:
    values = tuple(float(item) for item in text.split(",") if item.strip())
    if not values or any(value < 0 for value in values):
        raise ValueError("noise levels must be a nonempty list of nonnegative values")
    return tuple(
        ("labelled_ts", 0.0) if value == 0.0 else (f"noise_{value:.2f}A".replace(".", "p"), value)
        for value in values
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--h5", default=os.environ.get("GADPLUS_T1X_H5", "data/transition1x.h5"))
    common.add_argument("--split", default="test")
    common.add_argument("--sample-ids", default="2,17,43")
    common.add_argument("--noise-levels", default="0,0.05,0.10,0.20")
    common.add_argument("--output-root", type=Path, required=True)
    common.add_argument("--gxtb-executable", default=os.environ.get("GADPLUS_GXTB_EXE"))
    common.add_argument(
        "--parallel",
        type=int,
        default=int(os.environ.get("GADPLUS_GXTB_PARALLEL", "1")),
    )
    common.add_argument("--search-fmax", type=float, default=0.03)
    common.add_argument("--strict-fmax", type=float, default=0.01)
    common.add_argument("--max-steps", type=int, default=100)
    common.add_argument("--step-fraction", type=float, default=0.01)
    common.add_argument("--spectral-temperature", type=float, default=0.01)
    common.add_argument(
        "--gate-variant",
        choices=(
            "lambda2",
            "alignment",
            "competitive",
            "competitive_subspace",
            "guard",
            "gad",
        ),
        default="competitive",
    )
    common.add_argument("--seed", type=int, default=20260727)
    common.add_argument("--skip-endpoints", action="store_true")
    common.add_argument("--record-trajectories", action="store_true")
    common.add_argument("--campaign", default="t1x-gxtb-evaluation")
    common.add_argument(
        "--selection-stage",
        choices=("calibration", "evaluation", "instrumentation-smoke"),
        default="evaluation",
    )

    prepare = subparsers.add_parser("prepare-native", parents=[common])
    prepare.add_argument("--task-id", type=int, required=True)
    worker = subparsers.add_parser("worker", parents=[common])
    worker.add_argument("--task-id", type=int, required=True)
    aggregate = subparsers.add_parser("aggregate", parents=[common])
    aggregate.add_argument("--expected-tasks", type=int, required=True)
    return parser.parse_args()


def _predict_fn(args: argparse.Namespace):
    from gadplus.calculator.gxtb import load_gxtb_calculator, make_gxtb_predict_fn

    if not args.gxtb_executable:
        raise ValueError("set --gxtb-executable or GADPLUS_GXTB_EXE")
    return make_gxtb_predict_fn(
        load_gxtb_calculator(
            executable=args.gxtb_executable,
            n_threads=1,
            parallel=args.parallel,
        )
    )


def _records(args: argparse.Namespace):
    from gadplus.data.direct_t1x import load_t1x_records_direct

    return load_t1x_records_direct(args.h5, args.split, _parse_ids(args.sample_ids))


def _spectrum(predict_fn, coords: torch.Tensor, atomic_numbers: torch.Tensor):
    from gadplus.projection import atomic_nums_to_symbols, vib_eig

    output = predict_fn(coords, atomic_numbers, do_hessian=True, require_grad=False)
    eigenvalues, modes, _ = vib_eig(
        output["hessian"],
        coords,
        atomic_nums_to_symbols(atomic_numbers),
    )
    return output, eigenvalues, modes


def _endpoint_rows(predict_fn, saddle: torch.Tensor, atomic_numbers: torch.Tensor) -> list[dict]:
    from gadplus.projection import atomic_nums_to_symbols, get_mass_weights
    from gadplus.search.native_endpoints import relax_to_minimum

    _, _, modes = _spectrum(predict_fn, saddle, atomic_numbers)
    _, _, _, inv_sqrt_mass = get_mass_weights(atomic_nums_to_symbols(atomic_numbers))
    direction = (inv_sqrt_mass * modes[:, 0]).reshape_as(saddle)
    direction = direction / torch.linalg.vector_norm(direction).clamp_min(1.0e-12)
    rows = []
    for sign in (-1.0, 1.0):
        relaxed = relax_to_minimum(
            saddle + sign * 0.05 * direction,
            atomic_numbers,
            predict_fn,
            fmax=0.001,
            max_steps=500,
        )
        rows.append(
            {
                "sign": sign,
                "converged": relaxed.converged,
                "force_max": relaxed.force_max,
                "energy": relaxed.energy,
                "coords": relaxed.coords.tolist(),
                "error": relaxed.error or "",
            }
        )
    return rows


def prepare_native(args: argparse.Namespace) -> None:
    from gadplus.search.native_endpoints import relax_to_minimum

    torch.set_num_threads(1)
    predict_fn = _predict_fn(args)
    records = _records(args)
    tasks = [
        (sample_id, role)
        for sample_id in _parse_ids(args.sample_ids)
        for role in ("reactant", "product")
    ]
    if not 0 <= args.task_id < len(tasks):
        raise ValueError(f"task-id must be in [0, {len(tasks)})")
    sample_id, role = tasks[args.task_id]
    record = records[sample_id]
    coords = record.reactant if role == "reactant" else record.product
    if coords is None:
        raise ValueError(f"sample {sample_id} has no compatible labelled {role}")
    result = relax_to_minimum(
        coords,
        torch.as_tensor(record.atomic_nums, dtype=torch.long),
        predict_fn,
        fmax=0.001,
        max_steps=500,
    )
    label_dir = args.output_root / "native_labels"
    label_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "sample_id": sample_id,
        "role": role,
        "converged": result.converged,
        "force_max": result.force_max,
        "energy": result.energy,
        "coords": result.coords.tolist(),
        "error": result.error or "",
    }
    (label_dir / f"sample_{sample_id}_{role}.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )


def _native_label(output_root: Path, sample_id: int, role: str) -> dict:
    path = output_root / "native_labels" / f"sample_{sample_id}_{role}.json"
    if not path.exists():
        raise FileNotFoundError(f"missing prepared native endpoint: {path}")
    return json.loads(path.read_text())


def run_worker(args: argparse.Namespace) -> None:
    from gadplus.logging.pointwise import IntrinsicTrajectoryRecorder
    from gadplus.search.intrinsic_gad import IntrinsicGADConfig, run_intrinsic_gad
    from gadplus.search.irc_validate import score_endpoints

    torch.set_num_threads(1)
    tasks = [
        (sample_id, label, noise)
        for sample_id in _parse_ids(args.sample_ids)
        for label, noise in _starts(args.noise_levels)
    ]
    if not 0 <= args.task_id < len(tasks):
        raise ValueError(f"task-id must be in [0, {len(tasks)})")
    sample_id, start_label, noise = tasks[args.task_id]
    actual_seed = args.seed + 1000 * sample_id + args.task_id
    row = {
        "sample_id": sample_id,
        "start": start_label,
        "noise_angstrom": noise,
        "seed": actual_seed,
        "step_fraction": args.step_fraction,
        "spectral_temperature": args.spectral_temperature,
        "gate_variant": args.gate_variant,
        "search_gate": False,
        "strict_gate": False,
        "endpoint_minima": False,
        "native_endpoint_topology": False,
        "labelled_endpoint_topology": False,
        "error": "",
    }
    recorder = None
    try:
        record = _records(args)[sample_id]
        atomic_numbers = torch.as_tensor(record.atomic_nums, dtype=torch.long)
        reference = torch.as_tensor(record.transition_state, dtype=torch.float64)
        generator = torch.Generator().manual_seed(actual_seed)
        start = reference + noise * torch.randn(
            reference.shape,
            generator=generator,
            dtype=reference.dtype,
        )
        start = start - start.mean(dim=0, keepdim=True)
        predict_fn = _predict_fn(args)
        initial_out, initial_eigenvalues, _ = _spectrum(predict_fn, start, atomic_numbers)
        config = IntrinsicGADConfig(
            max_steps=args.max_steps,
            step_fraction=args.step_fraction,
            spectral_temperature=args.spectral_temperature,
            gate_variant=args.gate_variant,
            force_threshold=args.search_fmax,
            force_criterion="fmax",
            record_history=False,
        )
        if args.record_trajectories:
            run_name = (
                f"t1x-{args.split}-{sample_id:03d}-{noise:.2f}A-"
                f"{args.gate_variant}-seed{actual_seed}"
            )
            recorder = IntrinsicTrajectoryRecorder(
                args.output_root / "trajectories",
                run_name,
                atomic_numbers,
                config={
                    **asdict(config),
                    "campaign": args.campaign,
                    "selection_stage": args.selection_stage,
                    "dataset": "Transition1x",
                    "split": args.split,
                    "sample_id": sample_id,
                    "rxn": record.rxn,
                    "formula": record.formula,
                    "noise_angstrom": noise,
                    "seed": actual_seed,
                    "calculator": "g-xTB",
                    "instrumentation_level": "full-competitive",
                },
                reference_coordinates={"labelled_ts": reference},
            )
        result = run_intrinsic_gad(
            predict_fn,
            start,
            atomic_numbers,
            config,
            observer=recorder,
        )
        final_coords = result.final_coords.to(torch.float64)
        final_out, final_eigenvalues, _ = _spectrum(predict_fn, final_coords, atomic_numbers)
        fmax = float(final_out["forces"].abs().amax().item())
        n_neg = int((final_eigenvalues < -1.0e-4).sum().item())
        row.update(
            {
                "formula": record.formula,
                "rxn": record.rxn,
                "n_atoms": len(atomic_numbers),
                "initial_energy_eV": float(initial_out["energy"].item()),
                "initial_n_neg": int((initial_eigenvalues < -1.0e-4).sum().item()),
                "initial_lambda1": float(initial_eigenvalues[0].item()),
                "initial_lambda2": float(initial_eigenvalues[1].item()),
                "final_energy_eV": float(final_out["energy"].item()),
                "final_n_neg": n_neg,
                "final_lambda1": float(final_eigenvalues[0].item()),
                "final_lambda2": float(final_eigenvalues[1].item()),
                "final_fmax": fmax,
                "total_steps": result.total_steps,
                "n_evaluations": result.n_evaluations,
                "wall_time_s": result.wall_time_s,
                "search_gate": bool(n_neg == 1 and fmax < args.search_fmax),
                "strict_gate": bool(n_neg == 1 and fmax < args.strict_fmax),
                "final_coords": final_coords.tolist(),
            }
        )
        if row["search_gate"] and not args.skip_endpoints:
            endpoints = _endpoint_rows(predict_fn, final_coords, atomic_numbers)
            row["endpoints"] = endpoints
            row["endpoint_minima"] = bool(
                all(
                    endpoint["converged"] and endpoint["force_max"] < 0.001
                    for endpoint in endpoints
                )
            )
            native_reactant = _native_label(args.output_root, sample_id, "reactant")
            native_product = _native_label(args.output_root, sample_id, "product")
            forward = np.asarray(endpoints[0]["coords"], dtype=np.float64)
            reverse = np.asarray(endpoints[1]["coords"], dtype=np.float64)
            native_score = score_endpoints(
                forward,
                reverse,
                atomic_numbers,
                torch.as_tensor(native_reactant["coords"]),
                torch.as_tensor(native_product["coords"]),
                rmsd_threshold=0.3,
                predict_fn=predict_fn,
            )
            labelled_score = score_endpoints(
                forward,
                reverse,
                atomic_numbers,
                torch.as_tensor(record.reactant),
                torch.as_tensor(record.product) if record.product is not None else None,
                rmsd_threshold=0.3,
                predict_fn=predict_fn,
            )
            row["native_endpoint_topology"] = bool(native_score.topology_intended)
            row["labelled_endpoint_topology"] = bool(labelled_score.topology_intended)
            row["native_endpoint_error"] = native_score.error or native_score.topology_error or ""
            row["labelled_endpoint_error"] = (
                labelled_score.error or labelled_score.topology_error or ""
            )
        if recorder is not None:
            row["trajectory_bundle"] = str(
                recorder.flush(
                    result,
                    summary={
                        "calculator_valid": True,
                        "local_ts": row["search_gate"],
                        "strict_ts": row["strict_gate"],
                        "endpoint_minima": row["endpoint_minima"],
                        "native_topology": row["native_endpoint_topology"],
                        "labelled_topology": row["labelled_endpoint_topology"],
                        "instrumentation_level": "full-competitive",
                    },
                )
            )
    except Exception as error:  # noqa: BLE001 - calculator failure is a result.
        row["error"] = f"{type(error).__name__}: {error}"

    task_dir = args.output_root / "tasks"
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / f"task_{args.task_id:03d}.json").write_text(
        json.dumps(row, indent=2, sort_keys=True) + "\n"
    )


def aggregate(args: argparse.Namespace) -> None:
    paths = sorted((args.output_root / "tasks").glob("task_*.json"))
    if len(paths) != args.expected_tasks:
        raise RuntimeError(f"expected {args.expected_tasks} tasks, found {len(paths)}")
    rows = [json.loads(path.read_text()) for path in paths]
    summary = {
        "n_tasks": len(rows),
        "search_gate": sum(bool(row["search_gate"]) for row in rows),
        "strict_gate": sum(bool(row["strict_gate"]) for row in rows),
        "endpoint_minima": sum(bool(row["endpoint_minima"]) for row in rows),
        "native_endpoint_topology": sum(bool(row["native_endpoint_topology"]) for row in rows),
        "labelled_endpoint_topology": sum(bool(row["labelled_endpoint_topology"]) for row in rows),
        "errors": sum(bool(row["error"]) for row in rows),
        "trajectory_bundles": sum(bool(row.get("trajectory_bundle")) for row in rows),
        "rows": rows,
    }
    (args.output_root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    lines = [
        "# Intrinsic GAD g-xTB Transition1x pilot",
        "",
        "| sample | start | initial index | final index | fmax | search gate | strict gate | endpoint minima | native topology | error |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['sample_id']} | {row['start']} | {row.get('initial_n_neg', -1)} | "
            f"{row.get('final_n_neg', -1)} | {row.get('final_fmax', math.nan):.3e} | "
            f"{row['search_gate']} | {row['strict_gate']} | {row['endpoint_minima']} | "
            f"{row['native_endpoint_topology']} | {row['error']} |"
        )
    (args.output_root / "SUMMARY.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    if args.command == "prepare-native":
        prepare_native(args)
    elif args.command == "worker":
        run_worker(args)
    else:
        aggregate(args)


if __name__ == "__main__":
    main()
