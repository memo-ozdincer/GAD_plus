#!/usr/bin/env python
"""Identify the recurrence used by stored HIP GAD trajectories.

This re-evaluates the first stored geometry and compares the observed first
step with several projected-GAD coordinate conventions. The still-weighted
candidate is included only to exclude the failed code path; no benchmark data
are attributed to it. This is an artifact-provenance check, not an optimizer
benchmark.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from gadplus.calculator.hip import load_hip_calculator, make_hip_predict_fn
from gadplus.data.transition1x import Transition1xDataset, UsePos
from gadplus.projection.projection import (
    _eckart_projector,
    atomic_nums_to_symbols,
    gad_dynamics_projected,
    get_mass_weights,
    vib_eig,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--sample-ids", type=int, nargs="+", default=[0, 2, 3, 5])
    parser.add_argument("--noise-label", type=int, default=150)
    parser.add_argument("--dt", type=float, default=0.007)
    return parser.parse_args()


def candidate_metrics(observed: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    observed = observed.reshape(-1).double()
    candidate = candidate.reshape(-1).double()
    residual = candidate - observed
    return {
        "relative_error": float(residual.norm() / observed.norm()),
        "cosine": float(torch.dot(observed, candidate) / (observed.norm() * candidate.norm())),
        "scale": float(torch.dot(observed, candidate) / torch.dot(candidate, candidate)),
        "max_abs_error": float(residual.abs().max()),
    }


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    calculator = load_hip_calculator(str(args.checkpoint), device=device)
    predict_fn = make_hip_predict_fn(calculator)
    dataset = Transition1xDataset(
        str(args.dataset),
        split="test",
        max_samples=max(args.sample_ids) + 1,
        transform=UsePos("pos_transition"),
    )

    rows: list[dict[str, float | int | str]] = []
    for sample_id in args.sample_ids:
        paths = sorted(
            args.trajectory_dir.glob(
                f"traj_*_{args.noise_label}pm_*_{sample_id}.parquet"
            )
        )
        if len(paths) != 1:
            raise RuntimeError(f"Expected one trajectory for sample {sample_id}: {paths}")
        trajectory = pd.read_parquet(paths[0], columns=["step", "coords_flat"])
        trajectory = trajectory.sort_values("step")
        x0 = torch.tensor(
            np.asarray(trajectory.iloc[0].coords_flat).reshape(-1, 3),
            dtype=torch.float32,
            device=device,
        )
        x1 = torch.tensor(
            np.asarray(trajectory.iloc[1].coords_flat).reshape(-1, 3),
            dtype=torch.float32,
            device=device,
        )
        observed = (x1 - x0) / args.dt
        z = dataset[sample_id].z.to(device)
        symbols = atomic_nums_to_symbols(z)

        output = predict_fn(x0, z, do_hessian=True, require_grad=False)
        forces = output["forces"].reshape(-1, 3)
        hessian = output["hessian"].reshape(x0.numel(), x0.numel())
        _, modes_mw, _ = vib_eig(hessian, x0, symbols)
        v_mw = modes_mw[:, 0].double()

        masses, _, sqrt_m, sqrt_m_inv = get_mass_weights(
            symbols, device=device, dtype=torch.float64
        )
        projector = _eckart_projector(x0.double(), masses)
        force_flat = forces.reshape(-1).double()
        grad_mw = projector @ (-sqrt_m_inv * force_flat)
        v_mw = projector @ v_mw
        v_mw = v_mw / v_mw.norm()
        dq = projector @ (
            -grad_mw + 2.0 * torch.dot(v_mw, grad_mw) * v_mw
        )

        hybrid, _, _ = gad_dynamics_projected(
            coords=x0,
            forces=forces,
            v=v_mw,
            atomsymbols=symbols,
        )

        hessian = 0.5 * (hessian.double() + hessian.double().T)
        _, raw_modes = torch.linalg.eigh(hessian)
        v_raw = raw_modes[:, 0]
        raw_cartesian = force_flat - 2.0 * torch.dot(force_flat, v_raw) * v_raw

        candidates = {
            "mass_coordinate_unweighted": (sqrt_m_inv * dq).reshape_as(forces),
            "failed_still_weighted_return": (sqrt_m * dq).reshape_as(forces),
            "cartesian_reflection_mw_mode": hybrid,
            "raw_cartesian": raw_cartesian.reshape_as(forces),
        }
        for name, candidate in candidates.items():
            rows.append(
                {
                    "sample_id": sample_id,
                    "candidate": name,
                    **candidate_metrics(observed, candidate),
                }
            )

    result = pd.DataFrame(rows)
    print(result.to_string(index=False))
    print("\nMedian by candidate:")
    print(
        result.groupby("candidate")[
            ["relative_error", "cosine", "scale", "max_abs_error"]
        ].median().sort_values("relative_error").to_string()
    )


if __name__ == "__main__":
    main()
