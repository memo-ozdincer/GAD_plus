#!/usr/bin/env python
"""Test finite-step compatibility of energy, force, and Hessian products.

HIP and SCINE reuse directions from the stored paired HIP/Sella trajectories.
The analytic LJ mode generates a deterministic control panel. This diagnostic
never differentiates or substitutes HIP's direct-force Jacobian.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

OUTCOME_ORDER = ("both", "gad_only", "sella_only", "neither")
PHASE_ORDER = ("initial", "first_shrink", "floor_or_terminal")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("hip", "scine", "lj"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--probe-deltas",
        type=float,
        nargs="+",
        default=[0.08, 0.04, 0.02, 0.01, 0.005],
        help="Maximum per-atom displacement in Angstrom.",
    )
    parser.add_argument("--n-per-outcome", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument(
        "--paired-table",
        type=Path,
        default=Path(
            "/lustre07/scratch/memoozd/gadplus/runs/"
            "hip-paired-outcomes-65828024/per_sample.parquet"
        ),
    )
    parser.add_argument(
        "--sella-dir",
        type=Path,
        default=Path(
            "/lustre07/scratch/memoozd/gadplus/runs/"
            "test_sella_trajlog/carteck_libdef"
        ),
    )
    parser.add_argument("--noise-pm", type=int, default=150)
    parser.add_argument(
        "--h5",
        default="/lustre06/project/6033559/memoozd/data/transition1x.h5",
    )
    parser.add_argument(
        "--checkpoint",
        default="/lustre06/project/6033559/memoozd/models/hip_v2.ckpt",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--scine-functional", default="DFTB0")
    parser.add_argument("--lj-noise", type=float, default=0.05)
    parser.add_argument(
        "--hessian-treatment",
        choices=("raw", "eckart", "both"),
        default="both",
        help="Curvature used in the Taylor model. LJ always uses raw.",
    )
    return parser.parse_args()


def select_panel(
    paired_table: Path, n_per_outcome: int, seed: int
) -> tuple[pd.DataFrame, dict[str, list[int]]]:
    frame = pd.read_parquet(paired_table)
    rng = np.random.default_rng(seed)
    selected: dict[str, list[int]] = {}
    chunks = []
    for outcome in OUTCOME_ORDER:
        candidates = np.sort(
            frame.loc[frame.outcome == outcome, "sample_id"].astype(int).unique()
        )
        if len(candidates) < n_per_outcome:
            raise ValueError(
                f"Outcome {outcome} has {len(candidates)} candidates, "
                f"needs {n_per_outcome}"
            )
        ids = sorted(
            int(value)
            for value in rng.choice(candidates, n_per_outcome, replace=False)
        )
        selected[outcome] = ids
        chunks.append(frame.loc[frame.sample_id.isin(ids)].copy())
    return pd.concat(chunks, ignore_index=True), selected


def trajectory_path(directory: Path, noise_pm: int, sample_id: int) -> Path:
    paths = sorted(directory.glob(f"traj_*_{noise_pm}pm_*_{sample_id}.parquet"))
    if len(paths) != 1:
        raise RuntimeError(
            f"Expected one {noise_pm} pm trajectory for sample {sample_id}: {paths}"
        )
    return paths[0]


def max_atom_displacement(displacement: np.ndarray) -> float:
    return float(np.linalg.norm(displacement.reshape(-1, 3), axis=1).max())


def trajectory_directions(trajectory: pd.DataFrame) -> list[dict]:
    coords = [
        np.asarray(value, dtype=np.float64).reshape(-1, 3)
        for value in trajectory.coords_flat
    ]
    valid = [
        index
        for index in range(len(coords) - 1)
        if max_atom_displacement(coords[index + 1] - coords[index]) > 1.0e-8
    ]
    if not valid:
        raise ValueError("Trajectory has no nonzero steps")

    delta = trajectory.delta_trust.to_numpy(dtype=np.float64)
    initial_delta = float(delta[valid[0]])
    shrink_candidates = [
        index
        for index in valid
        if np.isfinite(delta[index]) and delta[index] < 0.75 * initial_delta
    ]
    shrink_index = (
        shrink_candidates[0] if shrink_candidates else valid[len(valid) // 2]
    )

    finite_delta = delta[np.isfinite(delta)]
    minimum_delta = float(finite_delta.min()) if len(finite_delta) else math.nan
    floor_candidates = [
        index
        for index in valid
        if np.isfinite(minimum_delta)
        and np.isfinite(delta[index])
        and delta[index] <= minimum_delta * 1.01 + 1.0e-12
    ]
    floor_index = floor_candidates[0] if floor_candidates else valid[-1]

    chosen = (
        ("initial", valid[0], "first_nonzero_step"),
        (
            "first_shrink",
            shrink_index,
            "first_delta_below_75pct"
            if shrink_candidates
            else "midpoint_fallback",
        ),
        (
            "floor_or_terminal",
            floor_index,
            "first_minimum_delta" if floor_candidates else "terminal_fallback",
        ),
    )
    rows = []
    for phase, index, selection_reason in chosen:
        observed = coords[index + 1] - coords[index]
        observed_max = max_atom_displacement(observed)
        rows.append(
            {
                "phase": phase,
                "phase_index": int(index),
                "phase_selection": selection_reason,
                "coords": coords[index],
                "direction": observed / observed_max,
                "observed_step": observed,
                "observed_max_atom_A": observed_max,
                "trust_radius_A": float(delta[index]),
            }
        )
    return rows


def build_predict_fn(args: argparse.Namespace) -> tuple[Callable, str, torch.dtype]:
    if args.backend == "hip":
        from gadplus.calculator.hip import load_hip_calculator, make_hip_predict_fn

        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        calculator = load_hip_calculator(args.checkpoint, device=device)
        return make_hip_predict_fn(calculator), device, torch.float32
    if args.backend == "scine":
        from gadplus.calculator.scine import (
            load_scine_calculator,
            make_scine_predict_fn,
        )

        calculator = load_scine_calculator(args.scine_functional)
        return make_scine_predict_fn(calculator), "cpu", torch.float64

    from gadplus.calculator.lennard_jones import make_lj_predict_fn

    return make_lj_predict_fn(), "cpu", torch.float64


def scalar(value) -> torch.Tensor:
    return torch.as_tensor(value).reshape(())


def hessian_variants(
    hessian: torch.Tensor,
    coords: torch.Tensor,
    atomic_nums: torch.Tensor,
    treatment: str,
) -> dict[str, torch.Tensor]:
    raw = 0.5 * (hessian.to(torch.float64) + hessian.to(torch.float64).T)
    variants = {}
    if treatment in {"raw", "both"}:
        variants["raw"] = raw
    if treatment in {"eckart", "both"}:
        from gadplus.projection.projection import (
            _eckart_projector,
            atomic_nums_to_symbols,
            get_mass_weights,
        )

        symbols = atomic_nums_to_symbols(atomic_nums)
        masses, _m3, sqrt_m, sqrt_m_inv = get_mass_weights(
            symbols, device=raw.device, dtype=torch.float64
        )
        projector = _eckart_projector(coords.to(torch.float64), masses)
        hessian_mw = (
            torch.diag(sqrt_m_inv) @ raw @ torch.diag(sqrt_m_inv)
        )
        hessian_mw = projector @ hessian_mw @ projector
        variants["eckart_applied"] = (
            torch.diag(sqrt_m)
            @ (0.5 * (hessian_mw + hessian_mw.T))
            @ torch.diag(sqrt_m)
        )
    return variants


def evaluate_direction(
    *,
    predict_fn: Callable,
    coords_np: np.ndarray,
    direction_np: np.ndarray,
    observed_step_np: np.ndarray,
    atomic_nums_np: np.ndarray,
    probe_deltas: list[float],
    device: str,
    dtype: torch.dtype,
    hessian_treatment: str,
    metadata: dict,
) -> tuple[list[dict], list[dict]]:
    coords = torch.as_tensor(coords_np, dtype=dtype, device=device)
    direction = torch.as_tensor(direction_np, dtype=dtype, device=device)
    atomic_nums = torch.as_tensor(atomic_nums_np, dtype=torch.long, device=device)

    base = predict_fn(
        coords, atomic_nums, do_hessian=True, require_grad=False
    )
    energy0 = scalar(base["energy"]).to(torch.float64)
    force0 = torch.as_tensor(base["forces"], device=device).reshape_as(coords)
    hessian = torch.as_tensor(base["hessian"], device=device).reshape(
        coords.numel(), coords.numel()
    )
    force0_64 = force0.to(torch.float64).reshape(-1)
    gradient64 = -force0_64
    variants = hessian_variants(
        hessian, coords, atomic_nums, hessian_treatment
    )

    probe_rows = []
    for delta in probe_deltas:
        step = (float(delta) * direction).to(dtype)
        endpoint = predict_fn(
            coords + step,
            atomic_nums,
            do_hessian=False,
            require_grad=False,
        )
        delta_energy = float(
            (scalar(endpoint["energy"]).to(torch.float64) - energy0).item()
        )
        endpoint_force = torch.as_tensor(
            endpoint["forces"], device=device
        ).reshape_as(coords)
        step64 = step.to(torch.float64).reshape(-1)
        linear = float(torch.dot(gradient64, step64).item())
        delta_force = endpoint_force.to(torch.float64).reshape(-1) - force0_64
        for variant_name, hessian64 in variants.items():
            hessian_step = hessian64 @ step64
            quadratic = float(0.5 * torch.dot(step64, hessian_step).item())
            predicted_second = linear + quadratic
            energy_first_residual = abs(delta_energy - linear)
            energy_second_residual = abs(delta_energy - predicted_second)
            force_second_residual = float(
                (delta_force + hessian_step).norm().item()
            )
            probe_rows.append(
                {
                    **metadata,
                    "hessian_variant": variant_name,
                    "probe_delta_A": float(delta),
                    "delta_energy_eV": delta_energy,
                    "predicted_linear_eV": linear,
                    "predicted_quadratic_eV": quadratic,
                    "predicted_second_eV": predicted_second,
                    "rho_energy": (
                        delta_energy / predicted_second
                        if abs(predicted_second) > 1.0e-14
                        else math.nan
                    ),
                    "energy_first_residual_eV": energy_first_residual,
                    "energy_second_residual_eV": energy_second_residual,
                    "energy_second_relative": energy_second_residual
                    / (abs(delta_energy) + abs(predicted_second) + 1.0e-12),
                    "force_second_residual_eV_per_A": force_second_residual,
                    "force_second_relative": force_second_residual
                    / (
                        float(delta_force.norm().item())
                        + float(hessian_step.norm().item())
                        + 1.0e-12
                    ),
                }
            )

    observed_step = torch.as_tensor(
        observed_step_np, dtype=dtype, device=device
    )
    observed_endpoint = predict_fn(
        coords + observed_step,
        atomic_nums,
        do_hessian=False,
        require_grad=False,
    )
    observed_step64 = observed_step.to(torch.float64).reshape(-1)
    observed_delta_energy = float(
        (scalar(observed_endpoint["energy"]).to(torch.float64) - energy0).item()
    )
    observed_linear = float(torch.dot(gradient64, observed_step64).item())
    direction_rows = []
    for variant_name, hessian64 in variants.items():
        observed_quadratic = float(
            0.5
            * torch.dot(
                observed_step64, hessian64 @ observed_step64
            ).item()
        )
        observed_predicted = observed_linear + observed_quadratic
        direction_rows.append(
            {
                **metadata,
                "hessian_variant": variant_name,
                "observed_delta_energy_eV": observed_delta_energy,
                "observed_predicted_energy_eV": observed_predicted,
                "observed_rho_energy": (
                    observed_delta_energy / observed_predicted
                    if abs(observed_predicted) > 1.0e-14
                    else math.nan
                ),
                "observed_energy_model_relative": abs(
                    observed_delta_energy - observed_predicted
                )
                / (
                    abs(observed_delta_energy)
                    + abs(observed_predicted)
                    + 1.0e-12
                ),
            }
        )
    return probe_rows, direction_rows


def fitted_order(frame: pd.DataFrame, column: str, largest_n: int | None) -> float:
    ordered = frame.sort_values("probe_delta_A", ascending=False)
    if largest_n is not None:
        ordered = ordered.head(largest_n)
    x = np.log(ordered.probe_delta_A.to_numpy(dtype=np.float64))
    y = np.log(
        np.maximum(ordered[column].to_numpy(dtype=np.float64), 1.0e-30)
    )
    if len(x) < 2 or not np.isfinite(y).all():
        return math.nan
    return float(np.polyfit(x, y, deg=1)[0])


def add_orders(probes: pd.DataFrame, directions: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "backend",
        "outcome",
        "sample_id",
        "phase",
        "phase_index",
        "hessian_variant",
    ]
    rows = []
    for values, group in probes.groupby(keys, dropna=False):
        row = dict(zip(keys, values))
        for metric, label in (
            ("energy_first_residual_eV", "energy_first"),
            ("energy_second_residual_eV", "energy_second"),
            ("force_second_residual_eV_per_A", "force_second"),
        ):
            row[f"{label}_order_all"] = fitted_order(group, metric, None)
            row[f"{label}_order_large4"] = fitted_order(group, metric, 4)
        rows.append(row)
    orders = pd.DataFrame(rows)
    return directions.merge(orders, on=keys, how="left", validate="one_to_one")


def paired_tasks(args: argparse.Namespace, selected_frame: pd.DataFrame) -> list[dict]:
    from gadplus.data.direct_t1x import load_t1x_records_direct

    ids = sorted(int(value) for value in selected_frame.sample_id.unique())
    records = load_t1x_records_direct(args.h5, "test", ids)
    outcomes = selected_frame.set_index("sample_id").outcome.to_dict()
    tasks = []
    for sample_id in ids:
        trajectory = pd.read_parquet(
            trajectory_path(args.sella_dir, args.noise_pm, sample_id),
            columns=["step", "delta_trust", "coords_flat"],
        ).sort_values("step")
        for direction in trajectory_directions(trajectory):
            tasks.append(
                {
                    **direction,
                    "outcome": str(outcomes[sample_id]),
                    "sample_id": sample_id,
                    "formula": records[sample_id].formula,
                    "atomic_nums": records[sample_id].atomic_nums,
                }
            )
    return tasks


def lj_tasks(args: argparse.Namespace) -> list[dict]:
    from gadplus.calculator.lennard_jones import (
        center_geometry,
        lj_atomic_nums,
        pentagonal_bipyramid_geometry,
    )

    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    base = pentagonal_bipyramid_geometry()
    tasks = []
    for sample_id in range(4 * args.n_per_outcome):
        coords = center_geometry(
            base
            + args.lj_noise
            * torch.randn(base.shape, generator=generator, dtype=torch.float64)
        )
        for phase in PHASE_ORDER:
            direction = torch.randn(
                base.shape, generator=generator, dtype=torch.float64
            )
            direction = direction / direction.norm(dim=1).max()
            observed_step = 0.05 * direction
            tasks.append(
                {
                    "phase": phase,
                    "phase_index": PHASE_ORDER.index(phase),
                    "phase_selection": "deterministic_lj_control",
                    "coords": coords.numpy(),
                    "direction": direction.numpy(),
                    "observed_step": observed_step.numpy(),
                    "observed_max_atom_A": 0.05,
                    "trust_radius_A": math.nan,
                    "outcome": "exact_control",
                    "sample_id": sample_id,
                    "formula": "LJ7",
                    "atomic_nums": lj_atomic_nums(7).numpy(),
                }
            )
    return tasks


def sanitize_json(value):
    if isinstance(value, dict):
        return {str(key): sanitize_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize_json(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    return value


def main() -> None:
    args = parse_args()
    if any(delta <= 0 for delta in args.probe_deltas):
        raise ValueError("Probe deltas must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.backend == "lj":
        args.hessian_treatment = "raw"
        selected: dict[str, list[int]] = {"exact_control": list(
            range(4 * args.n_per_outcome)
        )}
        tasks = lj_tasks(args)
    else:
        selected_frame, selected = select_panel(
            args.paired_table, args.n_per_outcome, args.seed
        )
        tasks = paired_tasks(args, selected_frame)

    predict_fn, device, dtype = build_predict_fn(args)
    probes = []
    direction_rows = []
    for task_id, task in enumerate(tasks):
        metadata = {
            "backend": args.backend,
            "outcome": task["outcome"],
            "sample_id": int(task["sample_id"]),
            "formula": task["formula"],
            "phase": task["phase"],
            "phase_index": int(task["phase_index"]),
            "phase_selection": task["phase_selection"],
            "observed_max_atom_A": float(task["observed_max_atom_A"]),
            "trust_radius_A": float(task["trust_radius_A"]),
        }
        probe_rows, task_direction_rows = evaluate_direction(
            predict_fn=predict_fn,
            coords_np=task["coords"],
            direction_np=task["direction"],
            observed_step_np=task["observed_step"],
            atomic_nums_np=task["atomic_nums"],
            probe_deltas=[float(value) for value in args.probe_deltas],
            device=device,
            dtype=dtype,
            hessian_treatment=args.hessian_treatment,
            metadata=metadata,
        )
        probes.extend(probe_rows)
        direction_rows.extend(task_direction_rows)
        print(
            json.dumps(
                {
                    "task": task_id + 1,
                    "n_tasks": len(tasks),
                    "backend": args.backend,
                    "outcome": task["outcome"],
                    "sample_id": int(task["sample_id"]),
                    "phase": task["phase"],
                },
                sort_keys=True,
            ),
            flush=True,
        )

    probe_frame = pd.DataFrame(probes)
    direction_frame = add_orders(probe_frame, pd.DataFrame(direction_rows))
    probe_frame.to_parquet(args.output_dir / "probes.parquet", index=False)
    direction_frame.to_parquet(
        args.output_dir / "directions.parquet", index=False
    )

    order_columns = [
        column for column in direction_frame if column.endswith("_order_large4")
    ]
    diagnostic_columns = [
        "observed_rho_energy",
        "observed_energy_model_relative",
        *order_columns,
    ]
    grouped = (
        direction_frame.groupby(
            ["hessian_variant", "outcome", "phase"], dropna=False
        )[
            diagnostic_columns
        ]
        .median()
        .reset_index()
    )
    grouped.to_csv(args.output_dir / "group_medians.csv", index=False)
    summary = {
        "backend": args.backend,
        "seed": args.seed,
        "selected_sample_ids": selected,
        "probe_deltas_A": [float(value) for value in args.probe_deltas],
        "n_directions": len(direction_frame),
        "group_medians": grouped.to_dict(orient="records"),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(sanitize_json(summary), indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({"summary": sanitize_json(summary)}, sort_keys=True))


if __name__ == "__main__":
    main()
