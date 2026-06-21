#!/usr/bin/env python
"""Screen Transition1x transition states with HIP and save index-1 samples."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from gadplus.calculator.hip import load_hip_calculator, make_hip_predict_fn
from gadplus.core.convergence import (
    NEG_EIGVAL_THRESHOLD,
    count_negative_eigenvalues,
    force_max,
)
from gadplus.data.transition1x import Transition1xDataset
from gadplus.paths import hip_checkpoint_path, scratch_dir, transition1x_h5_path
from gadplus.projection.projection import atomic_nums_to_symbols, get_mass_weights, vib_eig


def _as_single_structure(tensor: torch.Tensor, n_atoms: int | None = None) -> torch.Tensor:
    """Remove a leading singleton batch dimension when HIP returns one."""
    if tensor.dim() >= 1 and tensor.shape[0] == 1:
        if n_atoms is None or tensor.numel() != n_atoms * 3:
            return tensor.squeeze(0)
    return tensor


def _stable_mode_sign(mode: torch.Tensor) -> torch.Tensor:
    """Fix the arbitrary eigenvector sign for reproducible saved files."""
    pivot = torch.argmax(mode.abs())
    if mode[pivot] < 0:
        return -mode
    return mode


def _tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test")
    parser.add_argument("--target-count", type=int, default=10)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of valid Transition1x samples to load before screening.",
    )
    parser.add_argument("--force-threshold", type=float, default=0.01)
    parser.add_argument("--negative-threshold", type=float, default=NEG_EIGVAL_THRESHOLD)
    parser.add_argument("--purify-hessian", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    if args.target_count <= 0:
        raise ValueError("--target-count must be positive")
    if args.negative_threshold <= 0:
        raise ValueError("--negative-threshold must be positive")

    output_dir = args.output_dir or scratch_dir() / "runs" / "t1x_ts_hip_index1_samples"
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = hip_checkpoint_path()
    h5_path = transition1x_h5_path()
    print(f"HIP checkpoint: {ckpt_path}", flush=True)
    print(f"Transition1x H5: {h5_path}", flush=True)
    print(f"Output dir: {output_dir}", flush=True)
    print(
        f"Screening split={args.split} for n_neg==1 and fmax<{args.force_threshold:g}",
        flush=True,
    )

    calculator = load_hip_calculator(str(ckpt_path), device=args.device)
    predict_fn = make_hip_predict_fn(calculator)
    dataset = Transition1xDataset(str(h5_path), split=args.split, max_samples=args.max_samples)
    print(f"Loaded {len(dataset)} samples", flush=True)

    manifest_rows: list[dict[str, object]] = []
    found = 0
    manifest_path = output_dir / "manifest.csv"

    for sample_idx, sample in enumerate(dataset):
        coords = sample.pos_transition.to(args.device)
        atomic_nums = sample.z.to(args.device)
        n_atoms = int(atomic_nums.numel())
        formula = getattr(sample, "formula", f"sample_{sample_idx}")

        out = predict_fn(coords, atomic_nums, do_hessian=True, require_grad=False)
        forces = _as_single_structure(out["forces"], n_atoms).reshape(n_atoms, 3).double()
        hessian = out["hessian"].reshape(3 * n_atoms, 3 * n_atoms).double()

        atomsymbols = atomic_nums_to_symbols(atomic_nums.detach().cpu())
        evals_vib, evecs_vib_3n, _ = vib_eig(
            hessian,
            coords,
            atomsymbols,
            purify=args.purify_hessian,
        )
        n_neg = int((evals_vib < -args.negative_threshold).sum().item())
        fmax = force_max(forces)

        eig0 = float(evals_vib[0].item())
        eig1 = float(evals_vib[1].item()) if evals_vib.numel() > 1 else float("nan")
        print(
            f"[{sample_idx:04d}] {formula:>12s} n_neg={n_neg} "
            f"fmax={fmax:.6g} eig0={eig0:.6g} eig1={eig1:.6g}",
            flush=True,
        )

        if n_neg != 1 or fmax >= args.force_threshold:
            continue

        mode_mw = _stable_mode_sign(evecs_vib_3n[:, 0].to(torch.float64))
        _, _, _, sqrt_m_inv = get_mass_weights(atomsymbols, device=mode_mw.device)
        mode_cart = (sqrt_m_inv * mode_mw).reshape(n_atoms, 3)
        mode_cart = mode_cart / (mode_cart.norm() + 1e-12)

        energy = out.get("energy")
        if isinstance(energy, torch.Tensor):
            energy_value = float(energy.detach().cpu().reshape(-1)[0].item())
        else:
            energy_value = float("nan") if energy is None else float(energy)

        record = {
            "dataset_index": sample_idx,
            "split": args.split,
            "formula": formula,
            "rxn": getattr(sample, "rxn", ""),
            "atomic_numbers": _tensor_to_numpy(atomic_nums),
            "coordinates": _tensor_to_numpy(coords),
            "forces": _tensor_to_numpy(forces),
            "energy": energy_value,
            "fmax": fmax,
            "n_negative_eckart": n_neg,
            "negative_threshold": args.negative_threshold,
            "eckart_eigenvalues": _tensor_to_numpy(evals_vib),
            "reaction_coordinate_mw": _tensor_to_numpy(mode_mw),
            "reaction_coordinate_cartesian": _tensor_to_numpy(mode_cart),
            "hip_hessian_cartesian": _tensor_to_numpy(hessian),
            "purify_hessian_for_eigendecomposition": bool(args.purify_hessian),
        }

        found += 1
        out_path = output_dir / f"sample_{found:02d}_idx{sample_idx:04d}_{formula}.npy"
        np.save(out_path, record, allow_pickle=True)

        manifest_rows.append(
            {
                "rank": found,
                "dataset_index": sample_idx,
                "formula": formula,
                "rxn": getattr(sample, "rxn", ""),
                "path": str(out_path),
                "fmax": fmax,
                "n_negative_eckart": n_neg,
                "eig0": eig0,
                "eig1": eig1,
                "energy": energy_value,
            }
        )
        print(f"  saved {out_path}", flush=True)

        if found >= args.target_count:
            break

    if manifest_rows:
        with manifest_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0].keys()))
            writer.writeheader()
            writer.writerows(manifest_rows)
        print(f"Wrote manifest: {manifest_path}", flush=True)

    print(f"Found {found}/{args.target_count} requested samples", flush=True)
    if found < args.target_count:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
