#!/usr/bin/env python
"""Paired GAD/Sella pilot on an already accepted PaiNN-native saddle set."""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from ase import Atoms

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def final_metrics(predict_fn, coords: torch.Tensor, z: torch.Tensor) -> dict:
    from gadplus.projection import atomic_nums_to_symbols, vib_eig

    out = predict_fn(coords, z, do_hessian=True, require_grad=False)
    forces = out["forces"].reshape(-1, 3)
    evals, _vectors, _basis = vib_eig(out["hessian"], coords, atomic_nums_to_symbols(z))
    fmax = float(forces.abs().max())
    n_neg = int((evals < -1.0e-4).sum().item())
    return {
        "energy_eV": float(out["energy"].detach().reshape(())),
        "fmax_eV_per_A": fmax,
        "force_mean_eV_per_A": float(torch.linalg.vector_norm(forces, dim=1).mean()),
        "n_neg": n_neg,
        "eig0_eV_per_A2": float(evals[0]) if evals.numel() else float("nan"),
        "strict_converged": bool(n_neg == 1 and fmax < 0.01),
    }


def run_sella(
    predict_fn,
    start: torch.Tensor,
    z: torch.Tensor,
    *,
    max_steps: int,
    fmax: float,
    delta0: float,
    device: str,
) -> tuple[torch.Tensor, int, bool, int, str | None]:
    from gadplus.calculator.sella import (
        FullHessianASECalculator,
        full_hessian_function,
        refresh_hessian_after_kicks,
    )
    from sella import Sella

    atoms = Atoms(numbers=z.detach().cpu().numpy(), positions=start.detach().cpu().numpy())
    calculator = FullHessianASECalculator(predict_fn, z, device=device)
    atoms.calc = calculator
    try:
        optimizer = Sella(
            atoms=atoms,
            order=1,
            internal=False,
            trajectory=None,
            logfile=None,
            delta0=delta0,
            diag_every_n=1,
            gamma=0.0,
            rho_inc=1.035,
            rho_dec=5.0,
            sigma_inc=1.15,
            sigma_dec=0.65,
            hessian_function=full_hessian_function(calculator, eckart_project=True),
        )
        refresh_hessian_after_kicks(optimizer.pes)
        reported = bool(optimizer.run(fmax=fmax, steps=max_steps))
        return (
            torch.as_tensor(atoms.positions, dtype=torch.float32, device=device),
            int(optimizer.nsteps),
            reported,
            calculator.n_evaluations,
            None,
        )
    except Exception as exc:
        return start.detach().clone(), 0, False, calculator.n_evaluations, repr(exc)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-dir", required=True)
    parser.add_argument(
        "--validation-summary",
        required=True,
        help="summary.json from validate_painn_native_saddles.py; only accepted candidates run",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--noise-pm", type=float, nargs="+", default=[50.0, 100.0])
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=("gad", "sella"),
        default=("gad", "sella"),
        help="optimizer methods to run; use a single method for tuning screens",
    )
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--gad-dt", type=float, default=0.005)
    parser.add_argument("--sella-fmax", type=float, default=0.01)
    parser.add_argument("--sella-delta0", type=float, default=0.048)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    from gadplus.calculator.neuralneb import (
        NEURALNEB_MODELS_DIR,
        NeuralNebPaiNNCalculator,
        make_neuralneb_predict_fn,
    )
    from gadplus.search.gad_search import GADSearchConfig, run_gad_search

    with Path(args.validation_summary).open() as handle:
        validation_rows = json.load(handle)
    accepted_names = {
        str(row["candidate"])
        for row in validation_rows
        if bool(row.get("accepted", False))
    }
    candidate_paths = [
        path for path in sorted(glob.glob(os.path.join(args.candidate_dir, "candidate_*.npz")))
        if os.path.basename(path) in accepted_names
    ]
    if not candidate_paths:
        raise ValueError("Topology validation accepted no candidates; paired pilot is forbidden")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(args.checkpoint) if args.checkpoint else NEURALNEB_MODELS_DIR / "painn0.sd"
    predict_fn = make_neuralneb_predict_fn(NeuralNebPaiNNCalculator(checkpoint, args.device))
    gad_cfg = GADSearchConfig(
        n_steps=args.max_steps,
        dt=args.gad_dt,
        k_track=8,
        use_projection=True,
        use_adaptive_dt=False,
        force_threshold=0.01,
        force_criterion="fmax",
        # The pilot is deliberately pure GAD: no descent gate, blending,
        # multi-mode extension, displacement cap, or collision guard.
        max_atom_disp=float("inf"),
        min_interatomic_dist=0.0,
        descent_until_nneg=0,
        blend_sharpness=0.0,
        multimode="",
    )
    rows: list[dict] = []
    for candidate_path in candidate_paths:
        with np.load(candidate_path, allow_pickle=False) as data:
            sample_id = int(data["sample_id"].item())
            z = torch.as_tensor(data["atomic_numbers"], dtype=torch.long, device=args.device)
            reference = torch.as_tensor(data["coords"], dtype=torch.float32, device=args.device)
        for noise_pm in args.noise_pm:
            for seed in range(args.seeds):
                generator = torch.Generator(device="cpu").manual_seed(20260716 + sample_id * 1000 + int(noise_pm) * 10 + seed)
                noise = torch.randn(reference.shape, generator=generator, dtype=reference.dtype) * (noise_pm / 100.0)
                start = reference + noise.to(args.device)
                for method in args.methods:
                    t0 = time.perf_counter()
                    if method == "gad":
                        result = run_gad_search(predict_fn, start, z, gad_cfg)
                        final = result.final_coords.to(args.device)
                        optimizer_info = {
                            "reported_converged": result.converged,
                            "steps": result.total_steps,
                            "backend_evaluations": result.total_steps,
                            "optimizer_error": None,
                        }
                    else:
                        final, steps, reported, calls, error = run_sella(
                            predict_fn,
                            start,
                            z,
                            max_steps=args.max_steps,
                            fmax=args.sella_fmax,
                            delta0=args.sella_delta0,
                            device=args.device,
                        )
                        optimizer_info = {
                            "reported_converged": reported,
                            "steps": steps,
                            "backend_evaluations": calls,
                            "optimizer_error": error,
                        }
                    metrics = final_metrics(predict_fn, final, z)
                    row = {
                        "sample_id": sample_id,
                        "candidate_file": os.path.basename(candidate_path),
                        "method": method,
                        "noise_pm": noise_pm,
                        "seed": seed,
                        "start_rmsd_A": float(torch.sqrt((noise.square()).mean())),
                        "wall_time_s": time.perf_counter() - t0,
                        **optimizer_info,
                        **metrics,
                        "coords_flat": final.detach().cpu().reshape(-1).tolist(),
                        "atomic_numbers": z.detach().cpu().tolist(),
                    }
                    rows.append(row)
                    pq.write_table(pa.Table.from_pylist(rows), output_dir / "summary.parquet")
                    print(json.dumps({key: value for key, value in row.items() if key not in {"coords_flat", "atomic_numbers"}}, sort_keys=True))
    with (output_dir / "manifest.json").open("w") as handle:
        json.dump(
            {
                "candidate_files": candidate_paths,
                "validation_summary": str(Path(args.validation_summary).resolve()),
                "noise_pm": args.noise_pm,
                "seeds": args.seeds,
                "methods": args.methods,
                "max_steps": args.max_steps,
                "gad": {
                    "dt": args.gad_dt,
                    "projected": True,
                    "pure": True,
                    "max_atom_disp": "infinity",
                    "min_interatomic_dist": 0.0,
                },
                "sella": {"internal": False, "exact_hessian_every_step": True, "eckart_projected": True},
            },
            handle,
            indent=2,
            sort_keys=True,
        )


if __name__ == "__main__":
    main()
