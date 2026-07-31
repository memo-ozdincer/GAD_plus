#!/usr/bin/env python
"""Paired HIP GAD/Sella smoke with fixed direct E/F and substituted Hessians."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from ase import Atoms

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def final_metrics(predict_fn, coords: torch.Tensor, atomic_nums: torch.Tensor) -> dict:
    from gadplus.projection import atomic_nums_to_symbols, vib_eig

    out = predict_fn(coords, atomic_nums, do_hessian=True, require_grad=False)
    forces = out["forces"].reshape(-1, 3)
    eigenvalues, _vectors, _basis = vib_eig(
        out["hessian"], coords, atomic_nums_to_symbols(atomic_nums),
    )
    fmax = float(forces.abs().max())
    n_neg = int((eigenvalues < -1.0e-4).sum())
    return {
        "final_energy_eV": float(out["energy"].reshape(())),
        "final_fmax_eV_per_A": fmax,
        "final_n_neg": n_neg,
        "strict_converged": bool(n_neg == 1 and fmax < 0.01),
    }


def direct_head_agreement(reference: dict, substituted: dict) -> dict:
    energy_error = (reference["energy"].reshape(-1) - substituted["energy"].reshape(-1)).abs().max()
    force_error = (reference["forces"].reshape(-1) - substituted["forces"].reshape(-1)).abs().max()
    return {
        "initial_energy_abs_difference_eV": float(energy_error),
        "initial_force_abs_difference_eV_per_A": float(force_error),
    }


def run_sella(predict_fn, start, atomic_nums, args) -> tuple[torch.Tensor, int, bool, str | None]:
    from gadplus.calculator.sella import (
        FullHessianASECalculator,
        full_hessian_function,
        refresh_hessian_after_kicks,
    )
    from sella import Sella

    atoms = Atoms(
        numbers=atomic_nums.detach().cpu().numpy(),
        positions=start.detach().cpu().numpy(),
    )
    calculator = FullHessianASECalculator(predict_fn, atomic_nums, device=args.device)
    atoms.calc = calculator
    try:
        optimizer = Sella(
            atoms=atoms,
            order=1,
            internal=False,
            trajectory=None,
            logfile=None,
            delta0=args.sella_delta0,
            diag_every_n=1,
            gamma=0.0,
            hessian_function=full_hessian_function(calculator, eckart_project=True),
        )
        refresh_hessian_after_kicks(optimizer.pes)
        reported = bool(optimizer.run(fmax=0.01, steps=args.n_steps))
        return torch.as_tensor(atoms.positions, dtype=torch.float32, device=args.device), int(optimizer.nsteps), reported, None
    except Exception as exc:
        return start.clone(), 0, False, repr(exc)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-ids", type=int, nargs="+", default=[0, 2])
    parser.add_argument("--noise-a", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument(
        "--dataset-order", choices=("direct_hdf5", "historical"), default="direct_hdf5",
        help="Use historical Transition1xDataset indexing when reproducing existing runs.",
    )
    parser.add_argument(
        "--noise-mode", choices=("independent", "historical_sequential"), default="independent",
        help="historical_sequential reproduces method_single.py's CPU random draw order.",
    )
    parser.add_argument("--n-steps", type=int, default=100)
    parser.add_argument("--gad-dt", type=float, default=0.007)
    parser.add_argument("--gad-k-track", type=int, default=8)
    parser.add_argument("--gad-max-atom-disp", type=float, default=float("inf"))
    parser.add_argument("--gad-min-interatomic-dist", type=float, default=0.0)
    parser.add_argument("--sella-delta0", type=float, default=0.048)
    parser.add_argument(
        "--sources", nargs="+", default=["predicted", "force_jacobian", "energy_hessian"],
        choices=["predicted", "force_jacobian", "energy_hessian"],
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--h5", default="/lustre06/project/6033559/memoozd/data/transition1x.h5")
    parser.add_argument("--checkpoint", default="/lustre06/project/6033559/memoozd/models/hip_v2.ckpt")
    args = parser.parse_args()

    from gadplus.calculator.hip import (
        load_hip_calculator,
        make_hip_predict_fn,
        make_hip_curvature_source_predict_fn,
    )
    from gadplus.search.gad_search import GADSearchConfig, run_gad_search

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.dataset_order == "historical":
        from gadplus.data.transition1x import Transition1xDataset, UsePos

        dataset = Transition1xDataset(
            args.h5, split="test", max_samples=max(args.sample_ids) + 1,
            transform=UsePos("pos_transition"),
        )
        records = {
            sample_id: {
                "coords": dataset[sample_id].pos_transition,
                "atomic_nums": dataset[sample_id].z,
                "formula": str(dataset[sample_id].formula),
            }
            for sample_id in args.sample_ids
        }
        if args.noise_mode == "historical_sequential":
            generator = torch.Generator(device="cpu").manual_seed(args.seed)
            historical_noise = {}
            for sample_id in range(max(args.sample_ids) + 1):
                coords = dataset[sample_id].pos_transition
                historical_noise[sample_id] = torch.randn(
                    coords.shape, generator=generator, dtype=coords.dtype,
                ) * args.noise_a
        else:
            historical_noise = {}
    else:
        from gadplus.data.direct_t1x import load_t1x_records_direct

        direct_ids = (
            range(max(args.sample_ids) + 1)
            if args.noise_mode == "historical_sequential" else args.sample_ids
        )
        direct_records = load_t1x_records_direct(args.h5, "test", direct_ids)
        records = {
            sample_id: {
                "coords": record.transition_state,
                "atomic_nums": record.atomic_nums,
                "formula": record.formula,
            }
            for sample_id, record in direct_records.items()
        }
        if args.noise_mode == "historical_sequential":
            generator = torch.Generator(device="cpu").manual_seed(args.seed)
            historical_noise = {
                sample_id: torch.randn(
                    records[sample_id]["coords"].shape,
                    generator=generator,
                    dtype=torch.float32,
                ) * args.noise_a
                for sample_id in range(max(args.sample_ids) + 1)
            }
        else:
            historical_noise = {}
    calculator = load_hip_calculator(args.checkpoint, device=args.device)
    reference_predict_fn = make_hip_predict_fn(calculator)
    gad_config = GADSearchConfig(
        n_steps=args.n_steps,
        dt=args.gad_dt,
        k_track=args.gad_k_track,
        use_projection=True,
        force_threshold=0.01,
        force_criterion="fmax",
        max_atom_disp=args.gad_max_atom_disp,
        min_interatomic_dist=args.gad_min_interatomic_dist,
    )
    rows: list[dict] = []
    for sample_id in args.sample_ids:
        record = records[sample_id]
        reference = torch.as_tensor(record["coords"], dtype=torch.float32, device=args.device)
        atomic_nums = torch.as_tensor(record["atomic_nums"], dtype=torch.long, device=args.device)
        if args.noise_mode == "historical_sequential":
            noise = historical_noise[sample_id]
        else:
            generator = torch.Generator(device="cpu").manual_seed(args.seed + sample_id)
            noise = torch.randn(reference.shape, generator=generator) * args.noise_a
        start = reference + noise.to(args.device)
        reference_start = reference_predict_fn(start, atomic_nums, do_hessian=True, require_grad=False)
        for source in args.sources:
            predict_fn = make_hip_curvature_source_predict_fn(calculator, source)
            source_start = predict_fn(start, atomic_nums, do_hessian=True, require_grad=False)
            agreement = direct_head_agreement(reference_start, source_start)
            for method in ("gad", "sella"):
                started = time.perf_counter()
                if method == "gad":
                    result = run_gad_search(predict_fn, start, atomic_nums, gad_config)
                    final = result.final_coords.to(args.device)
                    optimizer = {
                        "reported_converged": result.converged,
                        "steps": result.total_steps,
                        "error": None,
                    }
                else:
                    final, steps, reported, error = run_sella(predict_fn, start, atomic_nums, args)
                    optimizer = {"reported_converged": reported, "steps": steps, "error": error}
                row = {
                    "sample_id": sample_id,
                    "formula": record["formula"],
                    "source": source,
                    "method": method,
                    "noise_A": args.noise_a,
                    "wall_time_s": time.perf_counter() - started,
                    **agreement,
                    **optimizer,
                    **final_metrics(predict_fn, final, atomic_nums),
                }
                rows.append(row)
                pq.write_table(pa.Table.from_pylist(rows), args.output_dir / "summary.parquet")
                print(json.dumps(row, sort_keys=True), flush=True)
    with (args.output_dir / "manifest.json").open("w") as handle:
        json.dump(vars(args), handle, indent=2, sort_keys=True, default=str)


if __name__ == "__main__":
    main()
