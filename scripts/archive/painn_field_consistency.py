#!/usr/bin/env python
"""Measure conservative PaiNN E/F/H directional identities on held-out T1x."""
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


def _energy(out: dict) -> torch.Tensor:
    return out["energy"].reshape(())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-samples", type=int, default=5)
    parser.add_argument("--directions", type=int, default=3)
    parser.add_argument("--epsilon", type=float, default=0.01)
    parser.add_argument("--split", default="test")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--h5-path", default="/lustre06/project/6033559/memoozd/data/transition1x.h5")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    if args.epsilon <= 0:
        raise ValueError("--epsilon must be positive")

    from gadplus.calculator.neuralneb import (
        NEURALNEB_MODELS_DIR,
        NeuralNebPaiNNCalculator,
        make_neuralneb_predict_fn,
    )
    from gadplus.data.transition1x import Transition1xDataset, UsePos

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(args.checkpoint) if args.checkpoint else NEURALNEB_MODELS_DIR / "painn0.sd"
    predict_fn = make_neuralneb_predict_fn(
        NeuralNebPaiNNCalculator(checkpoint=checkpoint, device=args.device)
    )
    dataset = Transition1xDataset(
        args.h5_path, split=args.split, max_samples=args.n_samples, transform=UsePos("pos_transition")
    )
    torch.manual_seed(20260716)
    rows: list[dict] = []
    for sample_id, sample in enumerate(dataset):
        coords = sample.pos.to(args.device)
        atomic_nums = sample.z.to(args.device)
        base = predict_fn(coords, atomic_nums, do_hessian=True, require_grad=False)
        energy = _energy(base)
        forces = base["forces"].reshape_as(coords)
        hessian = base["hessian"].reshape(coords.numel(), coords.numel())
        for direction_id in range(args.directions):
            direction = torch.randn_like(coords)
            direction = direction / direction.norm()
            plus = predict_fn(coords + args.epsilon * direction, atomic_nums, do_hessian=False, require_grad=False)
            minus = predict_fn(coords - args.epsilon * direction, atomic_nums, do_hessian=False, require_grad=False)
            d_energy = (_energy(plus) - _energy(minus)) / (2.0 * args.epsilon)
            d_force = (plus["forces"].reshape_as(coords) - minus["forces"].reshape_as(coords)).reshape(-1) / (2.0 * args.epsilon)
            hessian_v = hessian @ direction.reshape(-1)
            energy_curvature = (_energy(plus) - 2.0 * energy + _energy(minus)) / args.epsilon**2
            force_jacobian_hessian = -d_force
            direct_force_projection = (forces * direction).sum()
            direct_hessian_curvature = direction.reshape(-1) @ hessian_v
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
            print(json.dumps(row, sort_keys=True), flush=True)

    pq.write_table(pa.Table.from_pylist(rows), output_dir / "directions.parquet")
    summary = {
        "n_rows": len(rows),
        "median_energy_force_abs_error_eV_per_A": float(torch.tensor([r["energy_force_abs_error_eV_per_A"] for r in rows]).median()),
        "median_hessian_force_jac_rel_error": float(torch.tensor([r["hessian_force_jac_rel_error"] for r in rows]).median()),
        "median_hessian_energy_curvature_abs_error_eV_per_A2": float(torch.tensor([r["hessian_energy_curvature_abs_error_eV_per_A2"] for r in rows]).median()),
    }
    with (output_dir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print(json.dumps({"summary": summary}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
