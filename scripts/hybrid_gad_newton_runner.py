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
from gadplus.core.convergence import count_negative_eigenvalues
from gadplus.data.transition1x import Transition1xDataset, UsePos
from gadplus.paths import hip_checkpoint_path, transition1x_h5_path
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
    return float(forces.reshape(-1).abs().max().item())


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
    n_neg = count_negative_eigenvalues(evals_sorted)
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
    p.add_argument(
        "--final-trust-radius",
        type=float,
        default=None,
        help=(
            "Optional second-stage trust radius. Once a trajectory reaches "
            "n_neg=1 and fmax <= --final-trust-force, subsequent steps use "
            "this smaller radius for polishing."
        ),
    )
    p.add_argument(
        "--final-trust-force",
        type=float,
        default=0.05,
        help="fmax threshold that activates --final-trust-radius polishing.",
    )
    p.add_argument("--switch-force", type=float, default=1.0e-3)
    p.add_argument(
        "--target-mode-strategy",
        default="fixed",
        choices=["fixed", "neg_force_coupling"],
        help=(
            "For hybrid_damped_eckart only: use fixed target_mode=0, or select "
            "the negative internal mode with the largest absolute force "
            "projection at each Markovian step."
        ),
    )
    p.add_argument(
        "--high-index-descent",
        default="gad",
        choices=["gad", "gradient", "index_controlled", "newton"],
        help=(
            "For hybrid_damped_eckart only: when n_neg > 1, use the normal GAD "
            "branch, projected gradient descent, index-controlled Newton, or "
            "projected Newton descent."
        ),
    )
    # min_curvature defaults match each hybrid_gad_newton file's own default:
    #   hybrid_gad_eigfollownewton.py:                 1.0e-6
    #   hybrid_gad_eigfollownewton_eckart.py:          1.0e-6
    #   hybrid_gad_damped_eigfollownewton_eckart.py:   1.0e-8
    p.add_argument("--min-curvature", type=float, default=None,
                   help="Override min_curvature; if None, use each function's natural default")
    p.add_argument("--noise", type=float, default=0.0,
                   help="Gaussian noise stddev in Å (e.g. 0.01 = 10pm)")
    p.add_argument("--n-samples", type=int, default=287)
    p.add_argument("--n-steps", type=int, default=400)
    p.add_argument("--split", default="test")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--save-traj", "-save-traj", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="Write per-sample trajectory parquet files.")
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--start-from",
        default="geodesic_mid",
        choices=["ts_noised", "reactant", "product", "midpoint", "geodesic_mid"],
        help=(
            "Initial geometry: noised TS (default), reactant, product, "
            "linear midpoint, or reactant-product geodesic midpoint."
        ),
    )
    p.add_argument("--force-threshold", type=float, default=0.01,
                   help="fmax convergence criterion (with n_neg=1)")
    return p.parse_args()


def starting_coords(sample, start_from: str, device: str, noise: torch.Tensor) -> tuple[torch.Tensor, str] | None:
    coords_ts = sample.pos.to(device)
    if start_from == "ts_noised":
        return coords_ts + noise.to(device), "ts_noised"

    if start_from == "reactant":
        if not hasattr(sample, "pos_reactant"):
            return None
        return sample.pos_reactant.to(device), "reactant"

    if start_from == "product":
        if not hasattr(sample, "pos_product"):
            return None
        pos_p = sample.pos_product.to(device)
        if pos_p.abs().sum() < 1e-6:
            return None
        return pos_p, "product"

    if start_from in {"midpoint", "geodesic_mid"}:
        if not hasattr(sample, "pos_reactant") or not hasattr(sample, "pos_product"):
            return None
        pos_r = sample.pos_reactant.to(device)
        pos_p = sample.pos_product.to(device)
        if pos_p.abs().sum() < 1e-6:
            return None
        if start_from == "midpoint":
            return 0.5 * (pos_r + pos_p), "midpoint"

        from gadplus.geometry.interpolation import geodesic_interpolation

        return (
            geodesic_interpolation(
                pos_r,
                pos_p,
                n_images=3,
                atoms=atomic_nums_to_symbols(sample.z),
            )[1],
            "geodesic_mid",
        )

    raise ValueError(f"Unknown start_from: {start_from}")


def main():
    args = parse_args()
    if args.method != "hybrid_damped_eckart" and args.high_index_descent != "gad":
        sys.exit("--high-index-descent is only implemented for --method hybrid_damped_eckart")
    if args.method != "hybrid_damped_eckart" and args.target_mode_strategy != "fixed":
        sys.exit("--target-mode-strategy is only implemented for --method hybrid_damped_eckart")
    if args.final_trust_radius is not None and args.final_trust_radius <= 0:
        sys.exit("--final-trust-radius must be positive when set")
    if args.final_trust_force <= 0:
        sys.exit("--final-trust-force must be positive")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    noise_pm = int(round(args.noise * 1000))

    device = args.device if torch.cuda.is_available() else "cpu"

    # Locate HIP + dataset
    try:
        ckpt = str(hip_checkpoint_path())
        h5 = str(transition1x_h5_path())
    except FileNotFoundError as exc:
        sys.exit(str(exc))

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
    if args.high_index_descent != "gad":
        method_tag = f"{method_tag}_hi{args.high_index_descent}"
    if args.target_mode_strategy != "fixed":
        method_tag = f"{method_tag}_tm{args.target_mode_strategy}"
    if args.final_trust_radius is not None:
        method_tag = f"{method_tag}_polishtr{args.final_trust_radius:g}"
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
        z = sample.z.to(device)
        formula = getattr(sample, "formula", f"sample_{i}")
        start = starting_coords(sample, args.start_from, device, noise_vecs[i])
        if start is None:
            print(f"  [{i:3d}] {formula:>12s} | SKIP: unavailable start={args.start_from}", flush=True)
            continue
        coords, start_method_str = start
        coords = coords.double()
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
        final_target_mode = -1
        final_target_force_coupling = float("nan")
        final_active_trust_radius = args.trust_radius
        polish_active = False
        polish_activated_step = None
        n_steps_actual = 0
        for step_idx in range(args.n_steps):
            active_trust_radius = (
                args.final_trust_radius
                if polish_active and args.final_trust_radius is not None
                else args.trust_radius
            )
            out = predict_fn(coords, atomic_nums, do_hessian=True,
                             require_grad=False)
            E = out["energy"]; F = out["forces"]; H = out["hessian"]
            F = F.reshape(-1, 3).double()
            H = H.reshape(F.numel(), F.numel()).double()

            fmax_v = fmax(F)
            fnorm_v = fnorm(F)
            E_v = float(E.item()) if hasattr(E, "item") else float(E)

            n_neg, eig0, eig1 = n_neg_eckart(H, coords, atomic_nums)

            if args.save_traj:
                traj_rows.append({
                    "sample_id": i, "step": step_idx, "energy": E_v,
                    "force_max": fmax_v, "force_norm": fnorm_v,
                    "n_neg": n_neg, "eig0": eig0, "eig1": eig1,
                    "step_method": None,
                    "step_norm_cart": None,
                    "force_norm_internal": None,
                    "target_eigval": None,
                    "target_mode": None,
                    "target_force_coupling": None,
                    "active_trust_radius": active_trust_radius,
                    "polish_active": polish_active,
                })

            # Convergence check before computing or applying the next step, so
            # an already-converged starting geometry exits at step 0.
            if n_neg == 1 and fmax_v < args.force_threshold:
                converged = True
                converged_step = step_idx
                final_force_max = fmax_v; final_force_norm = fnorm_v
                final_n_neg = n_neg; final_eig0 = eig0; final_eig1 = eig1
                final_energy = E_v
                final_active_trust_radius = active_trust_radius
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
                    trust_radius=active_trust_radius,
                    **mc_kw,
                )
                used = info.get("method", "?")
            elif args.method == "hybrid_eckart":
                step, info = proj_step_plain(
                    force_cart=F, hessian_cart=H, coords=coords.double(),
                    masses=masses, target_mode=0, gad_dt=args.gad_dt,
                    switch_based_on_hessian_eigval=switch_by_eig,
                    switch_force=args.switch_force,
                    trust_radius=active_trust_radius,
                    **mc_kw,
                )
                used = info.get("method", "?")
            elif args.method == "hybrid_damped_eckart":
                step, info = proj_step_damped(
                    force_cart=F, hessian_cart=H, coords=coords.double(),
                    masses=masses, target_mode=0, gad_dt=args.gad_dt,
                    switch_based_on_hessian_eigval=switch_by_eig,
                    switch_force=args.switch_force,
                    trust_radius=active_trust_radius,
                    target_mode_strategy=args.target_mode_strategy,
                    high_index_descent=args.high_index_descent,
                    **mc_kw,
                )
                used = info.get("method", "?")
            target_eigval = info_scalar(info, "target_eigval")
            target_mode_v = int(info.get("target_mode", 0))
            target_force_coupling = info_scalar(info, "target_force_coupling")
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
            if args.save_traj:
                traj_rows[-1].update({
                    "step_method": used,
                    "step_norm_cart": step_norm_cart,
                    "force_norm_internal": force_norm_internal,
                    "target_eigval": target_eigval,
                    "target_mode": target_mode_v,
                    "target_force_coupling": target_force_coupling,
                })

            if (
                args.final_trust_radius is not None
                and not polish_active
                and n_neg == 1
                and fmax_v <= args.final_trust_force
            ):
                polish_active = True
                polish_activated_step = step_idx

            # Apply step. Defensive on shape.
            step = step.reshape_as(coords)
            coords = (coords + step).detach()

            final_force_max = fmax_v; final_force_norm = fnorm_v
            final_n_neg = n_neg; final_eig0 = eig0; final_eig1 = eig1
            final_energy = E_v
            final_step_norm_cart = step_norm_cart
            final_force_norm_internal = force_norm_internal
            final_target_eigval = target_eigval
            final_target_mode = target_mode_v
            final_target_force_coupling = target_force_coupling
            final_active_trust_radius = active_trust_radius
            n_steps_actual = step_idx + 1

        wall = time.time() - t0
        # Write trajectory parquet
        traj_path = out_dir / f"traj_{method_tag}_{noise_pm}pm_{run_id[-8:]}_{i}.parquet"
        if args.save_traj and traj_rows:
            pd.DataFrame(traj_rows).to_parquet(traj_path)

        rows.append({
            "sample_id": i, "formula": formula, "method": method_tag,
            "noise_pm": noise_pm, "n_steps_setting": args.n_steps,
            "start_method": start_method_str,
            "converged": converged, "converged_step": converged_step,
            "total_steps": n_steps_actual,
            "final_force_max": final_force_max,
            "final_force_norm": final_force_norm,
            "final_step_norm_cart": final_step_norm_cart,
            "final_force_norm_internal": final_force_norm_internal,
            "final_target_eigval": final_target_eigval,
            "final_target_mode": final_target_mode,
            "final_target_force_coupling": final_target_force_coupling,
            "final_n_neg": final_n_neg,
            "final_eig0": final_eig0, "final_eig1": final_eig1,
            "final_energy": final_energy,
            "wall_time_s": wall, "last_step_method": final_method_used,
            "trust_radius": args.trust_radius,
            "final_trust_radius": args.final_trust_radius,
            "final_trust_force": args.final_trust_force,
            "final_active_trust_radius": final_active_trust_radius,
            "polish_activated_step": polish_activated_step,
            "gad_dt": args.gad_dt,
            "switch_by_eig": switch_by_eig,
            "high_index_descent": args.high_index_descent,
            "target_mode_strategy": args.target_mode_strategy,
            "coords_flat": coords.detach().reshape(-1).cpu().numpy().astype(float).tolist(),
            "atomic_nums": atomic_nums.detach().cpu().numpy().astype(int).tolist(),
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
