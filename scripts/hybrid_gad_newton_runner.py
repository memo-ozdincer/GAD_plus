#!/usr/bin/env python
"""Standalone runner for the hybrid_gad_newton's three hybrid_gad step functions.

Methods:
  hybrid             — src/gadplus/search/hybrid_gad_eigfollownewton.py
                       (no Eckart; force-norm switch only)
  hybrid_eckart      — src/gadplus/search/hybrid_gad_eigfollownewton_eckart.py
                       (Eckart-projected; switch_based_on_hessian_eigval={False,True})
  hybrid_damped_eckart — src/gadplus/search/hybrid_gad_damped_eigfollownewton_eckart.py
                       (damped variant; same switch toggle)

Each step calls predict_fn for energy/forces/Hessian, then dispatches to the
hybrid_gad_newton's step function. Uses Eckart-projected eigenvalue counting for
n_neg (consistent with the rest of the project).

Usage:
  python scripts/hybrid_hybrid_gad_newton_runner.py \
      --method hybrid_eckart --switch-by-eig false \
      --gad-dt 5e-3 --trust-radius 0.01 \
      --noise 0.01 --n-samples 287 --n-steps 1000 \
      --output-dir /lustre07/scratch/memoozd/gadplus/runs/hybrid_hybrid_gad_newton/<cell>
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import uuid
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from gadplus.calculator.hip import load_hip_calculator, make_hip_predict_fn
from gadplus.data.transition1x import Transition1xDataset, UsePos
from gadplus.projection import vib_eig, atomic_nums_to_symbols

# ── hybrid_gad_newton step functions ──────────────────────────────────────────
from gadplus.search.hybrid_gad_eigfollownewton import (
    hybrid_gad_newton_step_from_force,
)
from gadplus.search.hybrid_gad_eigfollownewton_eckart import (
    projected_hybrid_gad_newton_step as proj_step_plain,
    masses_from_z,
)
from gadplus.search.hybrid_gad_damped_eigfollownewton_eckart import (
    projected_hybrid_gad_newton_step as proj_step_damped,
)


def fmax(forces: torch.Tensor) -> float:
    f = forces.reshape(-1, 3)
    return float(torch.linalg.vector_norm(f, dim=1).max().item())


def fnorm(forces: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(forces.reshape(-1)).item())


def info_scalar(info: dict, key: str, default=None) -> float | None:
    value = info.get(key, default)
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return None
        return float(value.detach().reshape(-1)[0].cpu().item())
    return float(value)


def n_neg_eckart(hessian: torch.Tensor, coords: torch.Tensor,
                 atomic_nums: torch.Tensor) -> tuple[int, float, float]:
    """Eckart-projected n_neg + eig0 + eig1 (vibrational only)."""
    syms = atomic_nums_to_symbols(atomic_nums)
    evals, _, _ = vib_eig(hessian, coords, syms, purify=False)
    evals_sorted = torch.sort(evals).values
    n_neg = int((evals_sorted < 0).sum().item())
    eig0 = float(evals_sorted[0].item()) if evals_sorted.numel() > 0 else 0.0
    eig1 = float(evals_sorted[1].item()) if evals_sorted.numel() > 1 else 0.0
    return n_neg, eig0, eig1


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--method", required=True,
                   choices=["hybrid", "hybrid_eckart", "hybrid_damped_eckart"])
    p.add_argument("--switch-by-eig", default="false",
                   choices=["true", "false"],
                   help="(only for *_eckart methods) switch_based_on_hessian_eigval")
    p.add_argument("--gad-dt", type=float, default=5e-3)
    p.add_argument("--trust-radius", type=float, default=0.01)
    p.add_argument("--switch-force", type=float, default=1.0e-3)
    # min_curvature defaults match each hybrid_gad_newton file's own default:
    #   hybrid_gad_eigfollownewton.py:                 1.0e-6
    #   hybrid_gad_eigfollownewton_eckart.py:          1.0e-6
    #   hybrid_gad_damped_eigfollownewton_eckart.py:   1.0e-8
    p.add_argument("--min-curvature", type=float, default=None,
                   help="Override min_curvature; if None, use each function's natural default")
    p.add_argument("--noise", type=float, required=True,
                   help="Gaussian noise stddev in Å (e.g. 0.01 = 10pm)")
    p.add_argument("--n-samples", type=int, default=287)
    p.add_argument("--n-steps", type=int, default=1000)
    p.add_argument("--split", default="test")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--force-threshold", type=float, default=0.01,
                   help="fmax convergence gate (with n_neg=1)")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    noise_pm = int(round(args.noise * 1000))

    device = args.device if torch.cuda.is_available() else "cpu"

    # Locate HIP + dataset
    for ckpt in ["/lustre06/project/6033559/memoozd/models/hip_v2.ckpt",
                 "/project/rrg-aspuru/memoozd/models/hip_v2.ckpt"]:
        if os.path.exists(ckpt): break
    else: sys.exit("hip_v2.ckpt not found")

    for h5 in ["/lustre06/project/6033559/memoozd/data/transition1x.h5",
               "/project/rrg-aspuru/memoozd/data/transition1x.h5"]:
        if os.path.exists(h5): break
    else: sys.exit("transition1x.h5 not found")

    calculator = load_hip_calculator(ckpt, device=device)
    predict_fn = make_hip_predict_fn(calculator)
    print(f"HIP loaded on {device}")

    dataset = Transition1xDataset(
        h5, split=args.split, max_samples=args.n_samples,
        transform=UsePos("pos_transition"),
    )
    print(f"Loaded {len(dataset)} samples (split={args.split})")

    switch_by_eig = (args.switch_by_eig.lower() == "true")
    method_tag = args.method
    if args.method != "hybrid":
        method_tag = f"{args.method}_swEIG" if switch_by_eig else f"{args.method}_swFORCE"
    method_tag = f"{method_tag}_dt{args.gad_dt:g}_tr{args.trust_radius:g}"
    run_id = f"{method_tag}_{noise_pm}pm_{uuid.uuid4().hex[:8]}"
    summary_path = out_dir / f"summary_{method_tag}_{noise_pm}pm.parquet"

    # Pre-generate noise
    torch.manual_seed(args.seed)
    noise_vecs = {}
    for i in range(len(dataset)):
        s = dataset[i]
        noise_vecs[i] = torch.randn_like(s.pos) * args.noise

    # ── Per-sample loop ──────────────────────────────────────────────
    rows = []
    t_total = time.time()
    for i in range(len(dataset)):
        sample = dataset[i]
        coords_ts = sample.pos.to(device)
        z = sample.z.to(device)
        formula = getattr(sample, "formula", f"sample_{i}")
        coords = (coords_ts + noise_vecs[i].to(device)).double()
        atomic_nums = z

        # Get masses for eckart variants
        if args.method != "hybrid":
            masses = masses_from_z(atomic_nums, device=coords.device,
                                   dtype=coords.dtype)
        else:
            masses = None

        # Per-step trajectory accumulator (light, sparse)
        traj_rows = []

        t0 = time.time()
        converged = False
        converged_step = None
        final_force_max = float("nan")
        final_force_norm = float("nan")
        final_n_neg = -1
        final_eig0 = 0.0
        final_eig1 = 0.0
        final_energy = float("nan")
        final_method_used = ""
        final_step_norm_cart = float("nan")
        final_force_norm_internal = float("nan")
        final_target_eigval = float("nan")
        n_steps_actual = 0
        for step_idx in range(args.n_steps):
            out = predict_fn(coords, atomic_nums, do_hessian=True,
                             require_grad=False)
            E = out["energy"]; F = out["forces"]; H = out["hessian"]
            F = F.reshape(-1, 3).double()
            H = H.reshape(F.numel(), F.numel()).double()

            fmax_v = fmax(F)
            fnorm_v = fnorm(F)
            E_v = float(E.item()) if hasattr(E, "item") else float(E)

            # n_neg from Eckart-projected vibrational Hessian
            n_neg, eig0, eig1 = n_neg_eckart(H, coords, atomic_nums)

            traj_rows.append({
                "sample_id": i, "step": step_idx, "energy": E_v,
                "force_max": fmax_v, "force_norm": fnorm_v,
                "n_neg": n_neg, "eig0": eig0, "eig1": eig1,
                "step_method": None,
                "step_norm_cart": None,
                "force_norm_internal": None,
                "target_eigval": None,
            })

            # Convergence check
            if n_neg == 1 and fmax_v < args.force_threshold:
                converged = True
                converged_step = step_idx
                final_force_max = fmax_v; final_force_norm = fnorm_v
                final_n_neg = n_neg; final_eig0 = eig0; final_eig1 = eig1
                final_energy = E_v
                n_steps_actual = step_idx + 1
                break

            # Compute step from hybrid_gad_newton.
            # Pass min_curvature only if the user overrode it; otherwise let
            # each function use its own default (matches hybrid_gad_newton's __main__).
            mc_kw = {} if args.min_curvature is None else {"min_curvature": args.min_curvature}
            if args.method == "hybrid":
                step, info = hybrid_gad_newton_step_from_force(
                    F.reshape(-1), H, target_mode=0, gad_dt=args.gad_dt,
                    switch_force=args.switch_force,
                    trust_radius=args.trust_radius,
                    **mc_kw,
                )
                used = info.get("method", "?")
            elif args.method == "hybrid_eckart":
                step, info = proj_step_plain(
                    force_cart=F, hessian_cart=H, coords=coords.double(),
                    masses=masses, target_mode=0, gad_dt=args.gad_dt,
                    switch_based_on_hessian_eigval=switch_by_eig,
                    switch_force=args.switch_force,
                    trust_radius=args.trust_radius,
                    **mc_kw,
                )
                used = info.get("method", "?")
            elif args.method == "hybrid_damped_eckart":
                step, info = proj_step_damped(
                    force_cart=F, hessian_cart=H, coords=coords.double(),
                    masses=masses, target_mode=0, gad_dt=args.gad_dt,
                    switch_based_on_hessian_eigval=switch_by_eig,
                    switch_force=args.switch_force,
                    trust_radius=args.trust_radius,
                    **mc_kw,
                )
                used = info.get("method", "?")
            final_method_used = used
            step_norm_cart = info_scalar(
                info,
                "step_norm_cart",
                default=torch.linalg.vector_norm(step),
            )
            force_norm_internal = info_scalar(
                info,
                "force_norm_internal",
                default=info.get("force_norm"),
            )
            target_eigval = info_scalar(info, "target_eigval")
            traj_rows[-1].update({
                "step_method": used,
                "step_norm_cart": step_norm_cart,
                "force_norm_internal": force_norm_internal,
                "target_eigval": target_eigval,
            })

            # Apply step. Defensive on shape.
            step = step.reshape_as(coords)
            coords = (coords + step).detach()

            final_force_max = fmax_v; final_force_norm = fnorm_v
            final_n_neg = n_neg; final_eig0 = eig0; final_eig1 = eig1
            final_energy = E_v
            final_step_norm_cart = step_norm_cart
            final_force_norm_internal = force_norm_internal
            final_target_eigval = target_eigval
            n_steps_actual = step_idx + 1

        wall = time.time() - t0
        # Write trajectory parquet
        traj_path = out_dir / f"traj_{method_tag}_{noise_pm}pm_{run_id[-8:]}_{i}.parquet"
        if traj_rows:
            pd.DataFrame(traj_rows).to_parquet(traj_path)

        rows.append({
            "sample_id": i, "formula": formula, "method": method_tag,
            "noise_pm": noise_pm, "n_steps_setting": args.n_steps,
            "converged": converged, "converged_step": converged_step,
            "total_steps": n_steps_actual,
            "final_force_max": final_force_max,
            "final_force_norm": final_force_norm,
            "final_step_norm_cart": final_step_norm_cart,
            "final_force_norm_internal": final_force_norm_internal,
            "final_target_eigval": final_target_eigval,
            "final_n_neg": final_n_neg,
            "final_eig0": final_eig0, "final_eig1": final_eig1,
            "final_energy": final_energy,
            "wall_time_s": wall, "last_step_method": final_method_used,
            "trust_radius": args.trust_radius,
            "gad_dt": args.gad_dt,
            "switch_by_eig": switch_by_eig,
        })

        status = "CONV" if converged else "FAIL"
        print(f"  [{i:3d}] {formula:>12s} | {status} | n_neg={final_n_neg} "
              f"fmax={final_force_max:.4f} steps={n_steps_actual} wall={wall:.1f}s "
              f"last_method={final_method_used}", flush=True)

    pd.DataFrame(rows).to_parquet(summary_path)
    print(f"\nWrote {summary_path} ({len(rows)} rows)")
    print(f"Total wall: {time.time()-t_total:.0f}s")


if __name__ == "__main__":
    main()
