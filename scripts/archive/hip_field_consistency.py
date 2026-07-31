#!/usr/bin/env python
"""Measure whether HIP's direct energy, force, and Hessian agree locally.

For each held-out geometry and random Cartesian direction ``v`` this records
three independent directional identities of a conservative potential:

    F.v = -dE/ds,   H.v = -dF/ds,   v.H.v = d2E/ds2.

HIP predicts the three fields with separate heads, so these are diagnostics,
not identities that can be assumed from the model definition.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def _scalar_energy(out: dict) -> torch.Tensor:
    return out["energy"].reshape(())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=5)
    parser.add_argument("--directions", type=int, default=3)
    parser.add_argument("--epsilon", type=float, default=0.01)
    parser.add_argument("--split", default="test")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--h5-path", default="/lustre06/project/6033559/memoozd/data/transition1x.h5")
    parser.add_argument("--checkpoint", default="/lustre06/project/6033559/memoozd/models/hip_v2.ckpt")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    if args.epsilon <= 0:
        raise ValueError("--epsilon must be positive")

    from gadplus.calculator.hip import load_hip_calculator, make_hip_predict_fn
    from gadplus.data.transition1x import Transition1xDataset, UsePos

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(20260716)
    predict_fn = make_hip_predict_fn(load_hip_calculator(args.checkpoint, device=args.device))
    dataset = Transition1xDataset(
        args.h5_path, split=args.split, max_samples=args.n_samples, transform=UsePos("pos_transition")
    )
    rows: list[dict] = []
    for sample_id, sample in enumerate(dataset):
        coords = sample.pos.to(args.device)
        z = sample.z.to(args.device)
        base = predict_fn(coords, z, do_hessian=True, require_grad=False)
        energy = _scalar_energy(base)
        forces = base["forces"].reshape_as(coords)
        hessian = base["hessian"].reshape(coords.numel(), coords.numel())
        for direction_id in range(args.directions):
            direction = torch.randn_like(coords)
            direction = direction / direction.norm()
            plus = predict_fn(coords + args.epsilon * direction, z, do_hessian=False, require_grad=False)
            minus = predict_fn(coords - args.epsilon * direction, z, do_hessian=False, require_grad=False)
            e_plus = _scalar_energy(plus)
            e_minus = _scalar_energy(minus)
            f_plus = plus["forces"].reshape_as(coords)
            f_minus = minus["forces"].reshape_as(coords)

            d_energy = (e_plus - e_minus) / (2.0 * args.epsilon)
            d_force = (f_plus - f_minus).reshape(-1) / (2.0 * args.epsilon)
            hessian_v = hessian @ direction.reshape(-1)
            energy_curvature = (e_plus - 2.0 * energy + e_minus) / (args.epsilon**2)
            direct_force_projection = (forces * direction).sum()
            direct_hessian_curvature = direction.reshape(-1) @ hessian_v
            force_jacobian_hessian = -d_force

            row = {
                "sample_id": sample_id,
                "formula": str(sample.formula),
                "direction_id": direction_id,
                "epsilon_A": args.epsilon,
                "energy_force_abs_error_eV_per_A": float((direct_force_projection + d_energy).abs()),
                "hessian_force_jac_rel_error": float(
                    (hessian_v - force_jacobian_hessian).norm()
                    / force_jacobian_hessian.norm().clamp_min(1.0e-8)
                ),
                "hessian_energy_curvature_abs_error_eV_per_A2": float(
                    (direct_hessian_curvature - energy_curvature).abs()
                ),
                "hessian_relative_antisymmetry": float(
                    (hessian - hessian.T).norm() / hessian.norm().clamp_min(1.0e-8)
                ),
                "direct_force_norm_eV_per_A": float(forces.norm()),
                "direct_hessian_v_norm_eV_per_A2": float(hessian_v.norm()),
            }
            rows.append(row)
            print(json.dumps(row, sort_keys=True))

    pq.write_table(pa.Table.from_pylist(rows), output_dir / "directions.parquet")
    summary = {
        "n_rows": len(rows),
        "median_energy_force_abs_error_eV_per_A": float(
            torch.tensor([row["energy_force_abs_error_eV_per_A"] for row in rows]).median()
        ),
        "median_hessian_force_jac_rel_error": float(
            torch.tensor([row["hessian_force_jac_rel_error"] for row in rows]).median()
        ),
        "median_hessian_energy_curvature_abs_error_eV_per_A2": float(
            torch.tensor([row["hessian_energy_curvature_abs_error_eV_per_A2"] for row in rows]).median()
        ),
    }
    with (output_dir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print(json.dumps({"summary": summary}, sort_keys=True))


if __name__ == "__main__":
    main()
