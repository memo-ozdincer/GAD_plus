#!/usr/bin/env python
"""Compare HIP's predicted Hessian with force- and energy-derived curvature."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def _rows_of_negative_jacobian(values: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        [
            torch.autograd.grad(-component, coords, retain_graph=True)[0].reshape(-1)
            for component in values.reshape(-1)
        ]
    )


def _relative_error(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left - right).norm() / right.norm().clamp_min(1.0e-8))


def _mode_overlap(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left.reshape(-1).double()
    right = right.reshape(-1).double()
    return float((left @ right).abs() / (left.norm() * right.norm()).clamp_min(1.0e-12))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-id", type=int, default=2)
    parser.add_argument("--h5", default="/lustre06/project/6033559/memoozd/data/transition1x.h5")
    parser.add_argument("--checkpoint", default="/lustre06/project/6033559/memoozd/models/hip_v2.ckpt")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    from gadplus.calculator.hip import coords_to_pyg_batch, load_hip_calculator
    from gadplus.data.transition1x import Transition1xDataset, UsePos
    from gadplus.projection import atomic_nums_to_symbols, vib_eig

    dataset = Transition1xDataset(args.h5, split="test", transform=UsePos("pos_transition"))
    sample = dataset[args.sample_id]
    coords = sample.pos.to("cuda", torch.float32)
    atomic_nums = sample.z.to("cuda")
    calculator = load_hip_calculator(args.checkpoint, device="cuda", hessian_method="predict")
    model = calculator.potential

    batch = coords_to_pyg_batch(coords, atomic_nums, device=torch.device("cuda"))
    batch.pos = batch.pos.detach().clone().requires_grad_(True)
    energy, direct_forces, outputs = model.forward(batch, otf_graph=True)
    h_pred = outputs["hessian"].reshape(coords.numel(), coords.numel())
    h_force = _rows_of_negative_jacobian(direct_forces, batch.pos)
    energy_forces = -torch.autograd.grad(energy.sum(), batch.pos, create_graph=True)[0]
    h_energy = _rows_of_negative_jacobian(energy_forces, batch.pos)

    sources = {
        "predicted": h_pred,
        "force_jacobian": h_force,
        "energy_hessian": h_energy,
    }
    eig = {}
    symbols = atomic_nums_to_symbols(atomic_nums)
    for name, matrix in sources.items():
        symmetric = 0.5 * (matrix + matrix.T)
        values, vectors, _basis = vib_eig(symmetric, coords, symbols)
        eig[name] = (values, vectors)

    summary = {
        "sample_id": args.sample_id,
        "formula": str(sample.formula),
        "energy_eV": float(energy.reshape(())),
        "direct_force_fmax_eV_per_A": float(direct_forces.abs().max()),
        "energy_force_fmax_eV_per_A": float(energy_forces.abs().max()),
        "predicted_vs_force_jacobian_relative": _relative_error(h_pred, h_force),
        "predicted_vs_energy_hessian_relative": _relative_error(h_pred, h_energy),
        "force_jacobian_vs_energy_hessian_relative": _relative_error(h_force, h_energy),
        "force_jacobian_relative_antisymmetry": float(
            (h_force - h_force.T).norm() / h_force.norm().clamp_min(1.0e-8)
        ),
        "energy_hessian_relative_antisymmetry": float(
            (h_energy - h_energy.T).norm() / h_energy.norm().clamp_min(1.0e-8)
        ),
    }
    for name, (values, vectors) in eig.items():
        summary[f"{name}_n_neg"] = int((values < -1.0e-4).sum())
        summary[f"{name}_eig0_eV_per_A2"] = float(values[0])
    for left, right in (("predicted", "force_jacobian"), ("predicted", "energy_hessian")):
        summary[f"{left}_vs_{right}_v1_overlap"] = _mode_overlap(eig[left][1][:, 0], eig[right][1][:, 0])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
