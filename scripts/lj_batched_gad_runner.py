#!/usr/bin/env python
"""Batched full-GPU GAD sweeps for LJ7.

This runner is intentionally narrower than ``lj_paper_runner.py``: it only
handles GAD, but it vectorizes all LJ7 samples in one process using
``torch.func.vmap`` over analytic LJ gradients/Hessians and a batched Eckart
projection. It is the fast path for GAD step-size/noise retuning.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from gadplus.calculator.lennard_jones import (
    pair_indices,
    pentagonal_bipyramid_geometry,
    random_cluster_geometry,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-atoms", type=int, default=7)
    parser.add_argument("--n-samples", type=int, default=287)
    parser.add_argument(
        "--sample-ids",
        type=str,
        default=None,
        help=(
            "Optional comma-separated original sample ids to run. If omitted, "
            "runs sample ids range(n_samples)."
        ),
    )
    parser.add_argument("--noise", type=float, required=True)
    parser.add_argument("--dt", type=float, required=True)
    parser.add_argument("--n-steps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--mass", type=float, default=1.008, help="Identical LJ atom mass.")
    parser.add_argument("--force-threshold", type=float, default=0.01)
    parser.add_argument("--max-atom-disp", type=float, default=0.05)
    parser.add_argument(
        "--max-atom-disp-start",
        type=float,
        default=None,
        help=(
            "Optional initial trust radius. If set below --max-atom-disp, "
            "the allowed per-atom displacement ramps geometrically by "
            "--max-atom-disp-growth each step until it reaches --max-atom-disp."
        ),
    )
    parser.add_argument("--max-atom-disp-growth", type=float, default=1.0)
    parser.add_argument(
        "--descent-until-nneg",
        type=int,
        default=-1,
        help=(
            "If >=0, use projected force descent until each sample has "
            "n_neg <= this value, then lock into single-mode GAD."
        ),
    )
    parser.add_argument(
        "--blend-sharpness",
        type=float,
        default=0.0,
        help=(
            "If positive, smoothly gate the v1 GAD inversion with "
            "sigmoid(blend_sharpness * lambda2). This is force descent "
            "while the second vibrational mode is negative and ordinary "
            "single-mode GAD once lambda2 is positive."
        ),
    )
    parser.add_argument("--start-from", choices=["minimum_noised", "random"], default="minimum_noised")
    parser.add_argument("--device", choices=["cuda", "cpu", "auto"], default="auto")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/lj_batched_gad"))
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def parse_sample_ids(args: argparse.Namespace) -> list[int]:
    if args.sample_ids is None:
        return list(range(args.n_samples))
    out = [int(s) for s in args.sample_ids.split(",") if s.strip()]
    if not out:
        raise ValueError("--sample-ids was provided but no ids were parsed.")
    if min(out) < 0:
        raise ValueError("--sample-ids must be nonnegative.")
    return out


def make_starts(args: argparse.Namespace, device: torch.device) -> torch.Tensor:
    starts = []
    for sample_id in parse_sample_ids(args):
        gen = torch.Generator().manual_seed(args.seed + sample_id)
        if args.start_from == "minimum_noised" and args.n_atoms == 7:
            coords = pentagonal_bipyramid_geometry(args.sigma)
            if args.noise > 0:
                coords = coords + args.noise * torch.randn(coords.shape, generator=gen, dtype=coords.dtype)
        else:
            coords = random_cluster_geometry(args.n_atoms, sigma=args.sigma, generator=gen)
        coords = coords - coords.mean(dim=0, keepdim=True)
        starts.append(coords.reshape(-1))
    return torch.stack(starts).to(device=device, dtype=torch.float64)


def make_lj_fns(n_atoms: int, sigma: float, epsilon: float, device: torch.device):
    pairs = torch.tensor(pair_indices(n_atoms), dtype=torch.long, device=device)
    i_idx = pairs[:, 0]
    j_idx = pairs[:, 1]

    def energy_one(flat: torch.Tensor) -> torch.Tensor:
        xyz = flat.reshape(n_atoms, 3)
        rij = xyz[i_idx] - xyz[j_idx]
        dist = torch.linalg.vector_norm(rij, dim=1).clamp_min(1.0e-12)
        sr6 = (sigma / dist) ** 6
        return 4.0 * epsilon * torch.sum(sr6 * (sr6 - 1.0))

    grad_one = torch.func.grad(energy_one)
    hess_one = torch.func.hessian(energy_one)
    return torch.func.vmap(energy_one), torch.func.vmap(grad_one), torch.func.vmap(hess_one)


def batched_internal_basis(coords: torch.Tensor, mass: float) -> torch.Tensor:
    """Return batched Eckart internal bases in mass-weighted space.

    Args:
        coords: [B, N, 3]
        mass: identical atom mass

    Returns:
        U_int: [B, 3N, 3N - 6]
    """
    bsz, n_atoms, _ = coords.shape
    dtype = coords.dtype
    device = coords.device
    sqrt_m = torch.as_tensor(mass, dtype=dtype, device=device).sqrt()
    centered = coords - coords.mean(dim=1, keepdim=True)

    cols = []
    for axis in range(3):
        col = torch.zeros((bsz, n_atoms, 3), dtype=dtype, device=device)
        col[:, :, axis] = sqrt_m
        cols.append(col.reshape(bsz, -1))

    eye = torch.eye(3, dtype=dtype, device=device)
    for axis in range(3):
        omega = eye[axis].reshape(1, 1, 3).expand_as(centered)
        rot = torch.cross(omega, centered, dim=2) * sqrt_m
        cols.append(rot.reshape(bsz, -1))

    ext = torch.stack(cols, dim=2)
    q, _ = torch.linalg.qr(ext, mode="complete")
    return q[:, :, 6:]


def force_max(forces_flat: torch.Tensor) -> torch.Tensor:
    return forces_flat.abs().amax(dim=1)


def run(args: argparse.Namespace) -> pd.DataFrame:
    device = resolve_device(args.device)
    if args.n_atoms != 7:
        raise SystemExit("Batched runner is currently intended for nonlinear LJ7 clusters.")
    if args.blend_sharpness < 0:
        raise ValueError("--blend-sharpness must be nonnegative.")

    sample_ids = parse_sample_ids(args)
    coords = make_starts(args, device)
    coords0 = coords.clone()
    energy_fn, grad_fn, hess_fn = make_lj_fns(args.n_atoms, args.sigma, args.epsilon, device)

    n = len(sample_ids)
    dim = 3 * args.n_atoms
    converged = torch.zeros(n, dtype=torch.bool, device=device)
    gad_active = torch.full(
        (n,),
        args.descent_until_nneg < 0,
        dtype=torch.bool,
        device=device,
    )
    gad_switch_step = torch.full((n,), -1, dtype=torch.long, device=device)
    converged_step = torch.full((n,), -1, dtype=torch.long, device=device)
    final_n_neg = torch.zeros(n, dtype=torch.long, device=device)
    final_eig0 = torch.zeros(n, dtype=torch.float64, device=device)
    final_eig1 = torch.zeros(n, dtype=torch.float64, device=device)
    final_fmax = torch.full((n,), torch.inf, dtype=torch.float64, device=device)
    final_energy = torch.zeros(n, dtype=torch.float64, device=device)
    cap_hits = torch.zeros(n, dtype=torch.long, device=device)
    first_cap_step = torch.full((n,), -1, dtype=torch.long, device=device)
    max_raw_atom_step = torch.zeros(n, dtype=torch.float64, device=device)
    max_applied_atom_step = torch.zeros(n, dtype=torch.float64, device=device)
    min_applied_atom_step_after_cap = torch.full((n,), torch.inf, dtype=torch.float64, device=device)
    n_neg_initial = torch.full((n,), -1, dtype=torch.long, device=device)
    n_neg_step10 = torch.full((n,), -1, dtype=torch.long, device=device)
    n_neg_step100 = torch.full((n,), -1, dtype=torch.long, device=device)
    fmax_initial = torch.full((n,), torch.inf, dtype=torch.float64, device=device)
    fmax_step10 = torch.full((n,), torch.inf, dtype=torch.float64, device=device)
    fmax_step100 = torch.full((n,), torch.inf, dtype=torch.float64, device=device)

    t0 = time.time()
    for step in range(args.n_steps):
        coords_3d = coords.reshape(n, args.n_atoms, 3)
        energies = energy_fn(coords)
        grads = grad_fn(coords)
        forces = -grads
        hess = hess_fn(coords)
        hess = 0.5 * (hess + hess.transpose(-1, -2))

        u_int = batched_internal_basis(coords_3d, args.mass)
        h_mw = hess / args.mass
        h_i = torch.matmul(u_int.transpose(1, 2), torch.matmul(h_mw, u_int))
        evals, evecs_i = torch.linalg.eigh(h_i)
        n_neg = (evals < -1.0e-4).sum(dim=1)
        fmax = force_max(forces)
        if step == 0:
            n_neg_initial = n_neg
            fmax_initial = fmax
        elif step == 10:
            n_neg_step10 = n_neg
            fmax_step10 = fmax
        elif step == 100:
            n_neg_step100 = n_neg
            fmax_step100 = fmax
        now_converged = (n_neg == 1) & (fmax < args.force_threshold) & (~converged)

        if now_converged.any():
            converged_step[now_converged] = step
        converged |= now_converged

        if args.descent_until_nneg >= 0:
            switch_now = (~gad_active) & (n_neg <= args.descent_until_nneg) & (~converged)
            gad_active |= switch_now
            gad_switch_step = torch.where(
                switch_now & (gad_switch_step < 0),
                torch.full_like(gad_switch_step, step),
                gad_switch_step,
            )

        final_n_neg = n_neg
        final_eig0 = evals[:, 0]
        final_eig1 = evals[:, 1]
        final_fmax = fmax
        final_energy = energies
        if bool(converged.all().detach().cpu().item()):
            break

        v_i = evecs_i[:, :, 0]
        f_q = forces / args.mass**0.5
        f_i = torch.matmul(u_int.transpose(1, 2), f_q.unsqueeze(-1)).squeeze(-1)
        if args.blend_sharpness > 0:
            blend_weight = torch.sigmoid(args.blend_sharpness * evals[:, 1])
        else:
            blend_weight = torch.ones(n, dtype=coords.dtype, device=device)
        gad_direction_i = f_i - 2.0 * blend_weight[:, None] * (
            f_i * v_i
        ).sum(dim=1, keepdim=True) * v_i
        direction_i = torch.where(gad_active[:, None], gad_direction_i, f_i)
        direction_q = torch.matmul(u_int, direction_i.unsqueeze(-1)).squeeze(-1)
        direction_x = direction_q / args.mass**0.5
        step_x = args.dt * direction_x
        step_3d = step_x.reshape(n, args.n_atoms, 3)
        atom_norm = torch.linalg.vector_norm(step_3d, dim=2).amax(dim=1).clamp_min(1.0e-12)
        if args.max_atom_disp_start is not None:
            if args.max_atom_disp_start <= 0:
                raise ValueError("--max-atom-disp-start must be positive.")
            if args.max_atom_disp_growth < 1.0:
                raise ValueError("--max-atom-disp-growth must be >= 1.0.")
            step_cap = min(
                args.max_atom_disp,
                args.max_atom_disp_start * (args.max_atom_disp_growth**step),
            )
        else:
            step_cap = args.max_atom_disp
        scale = torch.minimum(
            torch.ones_like(atom_norm),
            torch.as_tensor(step_cap, dtype=coords.dtype, device=device) / atom_norm,
        )
        applied_step_x = scale[:, None] * step_x
        applied_atom_norm = torch.linalg.vector_norm(
            applied_step_x.reshape(n, args.n_atoms, 3), dim=2
        ).amax(dim=1)
        active = ~converged
        cap_now = (scale < 0.999999) & active
        cap_hits += cap_now.to(torch.long)
        first_cap_step = torch.where(
            cap_now & (first_cap_step < 0),
            torch.full_like(first_cap_step, step),
            first_cap_step,
        )
        max_raw_atom_step = torch.where(
            active, torch.maximum(max_raw_atom_step, atom_norm), max_raw_atom_step
        )
        max_applied_atom_step = torch.where(
            active,
            torch.maximum(max_applied_atom_step, applied_atom_norm),
            max_applied_atom_step,
        )
        min_applied_atom_step_after_cap = torch.where(
            active & (cap_hits > 0),
            torch.minimum(min_applied_atom_step_after_cap, applied_atom_norm),
            min_applied_atom_step_after_cap,
        )
        coords = torch.where(active[:, None], coords + applied_step_x, coords)
        coords = coords.reshape(n, args.n_atoms, 3)
        coords = coords - coords.mean(dim=1, keepdim=True)
        coords = coords.reshape(n, dim).detach()

        if step % 100 == 0:
            c = int(converged.sum().detach().cpu().item())
            print(
                f"step={step:5d} converged={c}/{n} "
                f"median_fmax={float(fmax.median().detach().cpu()):.3e}",
                flush=True,
            )

    wall = time.time() - t0
    coords_cpu = coords.detach().cpu()
    coords0_cpu = coords0.detach().cpu()
    rows = []
    for i in range(n):
        rows.append(
            {
                "sample_id": i,
                "original_sample_id": sample_ids[i],
                "surface": "lennard_jones",
                "method": f"lj7_batched_gad_dt{args.dt:g}_Z1",
                "start_from": args.start_from,
                "n_atoms": args.n_atoms,
                "atomic_number": 1,
                "mass": args.mass,
                "epsilon": args.epsilon,
                "sigma": args.sigma,
                "noise": args.noise,
                "dt": args.dt,
                "max_atom_disp": args.max_atom_disp,
                "max_atom_disp_start": args.max_atom_disp_start,
                "max_atom_disp_growth": args.max_atom_disp_growth,
                "descent_until_nneg": args.descent_until_nneg,
                "blend_sharpness": args.blend_sharpness,
                "force_threshold": args.force_threshold,
                "converged": bool(converged[i].detach().cpu().item()),
                "converged_step": int(converged_step[i].detach().cpu().item())
                if converged_step[i] >= 0
                else None,
                "gad_switch_step": int(gad_switch_step[i].detach().cpu().item())
                if gad_switch_step[i] >= 0
                else (0 if args.descent_until_nneg < 0 else None),
                "total_steps": int(converged_step[i].detach().cpu().item()) + 1
                if converged_step[i] >= 0
                else args.n_steps,
                "final_n_neg": int(final_n_neg[i].detach().cpu().item()),
                "final_eig0": float(final_eig0[i].detach().cpu().item()),
                "final_eig1": float(final_eig1[i].detach().cpu().item()),
                "final_force_max": float(final_fmax[i].detach().cpu().item()),
                "final_energy": float(final_energy[i].detach().cpu().item()),
                "cap_hits": int(cap_hits[i].detach().cpu().item()),
                "first_cap_step": int(first_cap_step[i].detach().cpu().item())
                if first_cap_step[i] >= 0
                else None,
                "max_raw_atom_step": float(max_raw_atom_step[i].detach().cpu().item()),
                "max_applied_atom_step": float(max_applied_atom_step[i].detach().cpu().item()),
                "min_applied_atom_step_after_cap": (
                    float(min_applied_atom_step_after_cap[i].detach().cpu().item())
                    if torch.isfinite(min_applied_atom_step_after_cap[i])
                    else None
                ),
                "n_neg_initial": int(n_neg_initial[i].detach().cpu().item()),
                "n_neg_step10": int(n_neg_step10[i].detach().cpu().item()),
                "n_neg_step100": int(n_neg_step100[i].detach().cpu().item()),
                "fmax_initial": float(fmax_initial[i].detach().cpu().item()),
                "fmax_step10": float(fmax_step10[i].detach().cpu().item()),
                "fmax_step100": float(fmax_step100[i].detach().cpu().item()),
                "wall_time_s": wall / n,
                "coords_flat": coords_cpu[i].tolist(),
                "start_coords_flat": coords0_cpu[i].tolist(),
                "conv_nneg1_fmax001": bool(
                    final_n_neg[i].detach().cpu().item() == 1
                    and final_fmax[i].detach().cpu().item() < 0.01
                ),
                "conv_nneg1_fmax003": bool(
                    final_n_neg[i].detach().cpu().item() == 1
                    and final_fmax[i].detach().cpu().item() < 0.03
                ),
                "conv_nneg1_fmax005": bool(
                    final_n_neg[i].detach().cpu().item() == 1
                    and final_fmax[i].detach().cpu().item() < 0.05
                ),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    n_requested = len(parse_sample_ids(args))
    print(
        f"batched LJ GAD | n={n_requested} noise={args.noise:g} dt={args.dt:g} "
        f"steps={args.n_steps} device={resolve_device(args.device)}",
        flush=True,
    )
    df = run(args)
    noise_tag = int(round(args.noise * 1000))
    out = args.output_dir / f"summary_lj7_batched_gad_dt{args.dt:g}_{noise_tag}milli.parquet"
    df.to_parquet(out)
    n = len(df)
    c = int(df["conv_nneg1_fmax001"].sum())
    print(f"Wrote {out} ({n} rows); n_neg=1 & fmax<0.01: {c}/{n} ({100*c/n:.1f}%)")


if __name__ == "__main__":
    main()
