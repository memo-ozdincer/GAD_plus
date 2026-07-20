#!/usr/bin/env python
"""Measure HIP curvature-source disagreement over a fixed T1x test panel."""
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
from hip_hessian_source_smoke import _mode_overlap, _relative_error, _rows_of_negative_jacobian


def evaluate(sample_id, dataset, model):
    from gadplus.calculator.hip import coords_to_pyg_batch
    from gadplus.projection import atomic_nums_to_symbols, vib_eig

    sample = dataset[sample_id]
    coords = torch.as_tensor(sample.transition_state, device="cuda", dtype=torch.float32)
    atomic_nums = torch.as_tensor(sample.atomic_nums, device="cuda", dtype=torch.long)
    batch = coords_to_pyg_batch(coords, atomic_nums, device=torch.device("cuda"))
    batch.pos = batch.pos.detach().clone().requires_grad_(True)
    energy, direct_forces, outputs = model.forward(batch, otf_graph=True)
    h_pred = outputs["hessian"].reshape(coords.numel(), coords.numel())
    h_force = _rows_of_negative_jacobian(direct_forces, batch.pos)
    energy_forces = -torch.autograd.grad(energy.sum(), batch.pos, create_graph=True)[0]
    h_energy = _rows_of_negative_jacobian(energy_forces, batch.pos)
    sources = {"predicted": h_pred, "force_jacobian": h_force, "energy_hessian": h_energy}
    eig = {}
    symbols = atomic_nums_to_symbols(atomic_nums)
    for name, matrix in sources.items():
        eig[name] = vib_eig(0.5 * (matrix + matrix.T), coords, symbols)[:2]
    row = {
        "sample_id": sample_id,
        "formula": sample.formula,
        "direct_force_fmax_eV_per_A": float(direct_forces.abs().max()),
        "energy_force_fmax_eV_per_A": float(energy_forces.abs().max()),
        "predicted_vs_force_jacobian_relative": _relative_error(h_pred, h_force),
        "predicted_vs_energy_hessian_relative": _relative_error(h_pred, h_energy),
        "force_jacobian_vs_energy_hessian_relative": _relative_error(h_force, h_energy),
        "force_jacobian_relative_antisymmetry": float((h_force - h_force.T).norm() / h_force.norm().clamp_min(1.0e-8)),
    }
    for name, (values, vectors) in eig.items():
        row[f"{name}_n_neg"] = int((values < -1.0e-4).sum())
        row[f"{name}_eig0_eV_per_A2"] = float(values[0])
    for right in ("force_jacobian", "energy_hessian"):
        row[f"predicted_vs_{right}_v1_overlap"] = _mode_overlap(
            eig["predicted"][1][:, 0], eig[right][1][:, 0]
        )
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-ids", type=int, nargs="+", default=list(range(12)))
    parser.add_argument("--h5", default="/lustre06/project/6033559/memoozd/data/transition1x.h5")
    parser.add_argument("--checkpoint", default="/lustre06/project/6033559/memoozd/models/hip_v2.ckpt")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    from gadplus.calculator.hip import load_hip_calculator
    from gadplus.data.direct_t1x import load_t1x_records_direct

    dataset = load_t1x_records_direct(args.h5, "test", args.sample_ids)
    model = load_hip_calculator(args.checkpoint, device="cuda").potential
    rows = []
    for sample_id in args.sample_ids:
        row = evaluate(sample_id, dataset, model)
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), args.output_dir / "sources.parquet")
    keys = [
        "predicted_vs_force_jacobian_relative",
        "predicted_vs_energy_hessian_relative",
        "force_jacobian_vs_energy_hessian_relative",
        "force_jacobian_relative_antisymmetry",
        "predicted_vs_force_jacobian_v1_overlap",
        "predicted_vs_energy_hessian_v1_overlap",
    ]
    summary = {"n_samples": len(rows)}
    for key in keys:
        summary[f"median_{key}"] = float(torch.tensor([row[key] for row in rows]).median())
    summary["predicted_index_one_count"] = sum(row["predicted_n_neg"] == 1 for row in rows)
    summary["force_jacobian_index_one_count"] = sum(row["force_jacobian_n_neg"] == 1 for row in rows)
    summary["energy_hessian_index_one_count"] = sum(row["energy_hessian_n_neg"] == 1 for row in rows)
    with (args.output_dir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print(json.dumps({"summary": summary}, sort_keys=True))


if __name__ == "__main__":
    main()
