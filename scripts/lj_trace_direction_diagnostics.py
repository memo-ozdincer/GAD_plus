#!/usr/bin/env python
"""Replay selected LJ7 GAD samples with direction diagnostics.

The main batched sweep stores final summaries. This script replays a small set
of sample ids and records per-step geometry/direction features that help answer
whether high-noise failures leave the low-noise basin early.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

import lj_batched_gad_runner as lj_batch
from gadplus.calculator.lennard_jones import pentagonal_bipyramid_geometry


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sample-ids", required=True, help="Comma-separated sample ids.")
    p.add_argument("--noise", type=float, required=True)
    p.add_argument("--safe-noise", type=float, default=0.03)
    p.add_argument("--dt", type=float, required=True)
    p.add_argument("--max-atom-disp", type=float, default=0.05)
    p.add_argument("--mass", type=float, default=1.008)
    p.add_argument("--n-steps", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto")
    return p.parse_args()


def rmsd_to_ref(coords: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    diff = coords.reshape(coords.shape[0], -1) - ref.reshape(ref.shape[0], -1)
    return torch.linalg.vector_norm(diff, dim=1) / (coords.shape[1] ** 0.5)


def cosine_to(target_step: torch.Tensor, actual_step: torch.Tensor) -> torch.Tensor:
    numerator = (target_step * actual_step).sum(dim=1)
    denom = (
        torch.linalg.vector_norm(target_step, dim=1)
        * torch.linalg.vector_norm(actual_step, dim=1)
    ).clamp_min(1.0e-30)
    return numerator / denom


def make_subset_starts(args: argparse.Namespace, sample_ids: list[int], noise: float, device: torch.device):
    nsamples = max(sample_ids) + 1
    start_args = argparse.Namespace(
        n_atoms=7,
        n_samples=nsamples,
        noise=noise,
        seed=args.seed,
        sigma=1.0,
        start_from="minimum_noised",
    )
    return lj_batch.make_starts(start_args, device)[sample_ids]


def main() -> None:
    args = parse_args()
    sample_ids = [int(s) for s in args.sample_ids.split(",") if s.strip()]
    device = lj_batch.resolve_device(args.device)
    n = len(sample_ids)
    n_atoms = 7

    coords = make_subset_starts(args, sample_ids, args.noise, device)
    safe = make_subset_starts(args, sample_ids, args.safe_noise, device)
    minimum = pentagonal_bipyramid_geometry().reshape(1, n_atoms, 3).to(device)
    minimum = minimum.expand(n, -1, -1).reshape(n, -1)

    energy_fn, grad_fn, hess_fn = lj_batch.make_lj_fns(n_atoms, 1.0, 1.0, device)
    rows: list[dict] = []

    for step in range(args.n_steps):
        coords_3d = coords.reshape(n, n_atoms, 3)
        energies = energy_fn(coords)
        grads = grad_fn(coords)
        forces = -grads
        hess = hess_fn(coords)
        hess = 0.5 * (hess + hess.transpose(-1, -2))

        u_int = lj_batch.batched_internal_basis(coords_3d, args.mass)
        h_i = u_int.transpose(1, 2) @ ((hess / args.mass) @ u_int)
        evals, evecs_i = torch.linalg.eigh(h_i)
        n_neg = (evals < -1.0e-4).sum(dim=1)
        fmax = lj_batch.force_max(forces)

        v_i = evecs_i[:, :, 0]
        f_q = forces / args.mass**0.5
        f_i = (u_int.transpose(1, 2) @ f_q.unsqueeze(-1)).squeeze(-1)
        direction_i = f_i - 2.0 * (f_i * v_i).sum(dim=1, keepdim=True) * v_i
        direction_q = (u_int @ direction_i.unsqueeze(-1)).squeeze(-1)
        direction_x = direction_q / args.mass**0.5
        raw_step = args.dt * direction_x
        raw_atom = torch.linalg.vector_norm(raw_step.reshape(n, n_atoms, 3), dim=2).amax(dim=1)
        scale = torch.minimum(
            torch.ones_like(raw_atom),
            torch.as_tensor(args.max_atom_disp, dtype=coords.dtype, device=device)
            / raw_atom.clamp_min(1.0e-12),
        )
        applied_step = scale[:, None] * raw_step
        applied_atom = torch.linalg.vector_norm(
            applied_step.reshape(n, n_atoms, 3), dim=2
        ).amax(dim=1)

        to_safe = safe - coords
        to_min = minimum - coords
        safe_dist = rmsd_to_ref(coords.reshape(n, n_atoms, 3), safe.reshape(n, n_atoms, 3))
        min_dist = rmsd_to_ref(coords.reshape(n, n_atoms, 3), minimum.reshape(n, n_atoms, 3))
        safe_after = rmsd_to_ref(
            (coords + applied_step).reshape(n, n_atoms, 3),
            safe.reshape(n, n_atoms, 3),
        )
        min_after = rmsd_to_ref(
            (coords + applied_step).reshape(n, n_atoms, 3),
            minimum.reshape(n, n_atoms, 3),
        )
        cos_safe = cosine_to(to_safe, applied_step)
        cos_min = cosine_to(to_min, applied_step)

        for j, sid in enumerate(sample_ids):
            rows.append(
                {
                    "sample_id": sid,
                    "step": step,
                    "noise": args.noise,
                    "safe_noise": args.safe_noise,
                    "dt": args.dt,
                    "max_atom_disp": args.max_atom_disp,
                    "mass": args.mass,
                    "energy": float(energies[j].detach().cpu()),
                    "n_neg": int(n_neg[j].detach().cpu()),
                    "eig0": float(evals[j, 0].detach().cpu()),
                    "eig1": float(evals[j, 1].detach().cpu()),
                    "fmax": float(fmax[j].detach().cpu()),
                    "raw_atom_step": float(raw_atom[j].detach().cpu()),
                    "applied_atom_step": float(applied_atom[j].detach().cpu()),
                    "step_scale": float(scale[j].detach().cpu()),
                    "rmsd_to_safe_before": float(safe_dist[j].detach().cpu()),
                    "rmsd_to_safe_after": float(safe_after[j].detach().cpu()),
                    "delta_rmsd_to_safe": float((safe_after[j] - safe_dist[j]).detach().cpu()),
                    "rmsd_to_min_before": float(min_dist[j].detach().cpu()),
                    "rmsd_to_min_after": float(min_after[j].detach().cpu()),
                    "delta_rmsd_to_min": float((min_after[j] - min_dist[j]).detach().cpu()),
                    "cos_step_to_safe": float(cos_safe[j].detach().cpu()),
                    "cos_step_to_min": float(cos_min[j].detach().cpu()),
                }
            )

        coords = coords + applied_step
        coords = coords.reshape(n, n_atoms, 3)
        coords = coords - coords.mean(dim=1, keepdim=True)
        coords = coords.reshape(n, -1).detach()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_parquet(args.output, index=False)
    df.to_csv(args.output.with_suffix(".csv"), index=False)
    print(f"Wrote {args.output} ({len(df)} rows)")


if __name__ == "__main__":
    main()
