#!/usr/bin/env python
"""Run a SINGLE method at a SINGLE noise level. Designed for max parallelism.

Usage:
  python scripts/method_single.py --method gad_projected --noise 0.05 --n-samples 300 --n-steps 1000
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import uuid

import pandas as pd
import torch
from ase import Atoms
from ase.io import write as ase_write

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


METHOD_CONFIGS = {
    # === Round 1 methods ===
    "gad_projected": dict(runner="gad", dt=0.01, k_track=0, adaptive=False, max_disp=0.35),
    "gad_small_dt": dict(runner="gad", dt=0.005, k_track=0, adaptive=False, max_disp=0.35),
    "gad_adaptive_dt": dict(runner="gad", dt=0.01, k_track=0, adaptive=True, max_disp=0.35),
    "gad_tight_clamp": dict(runner="gad", dt=0.01, k_track=0, adaptive=False, max_disp=0.1),
    "gad_adaptive_tight": dict(runner="gad", dt=0.01, k_track=0, adaptive=True, max_disp=0.1),
    "nr_gad_pingpong": dict(
        runner="pingpong",
        dt=0.01,
        k_track=0,
        adaptive=False,
        max_disp=0.35,
        nr_damping=1.0,
        nr_max_step_norm=0.3,
    ),
    "nr_gad_pp_adaptive": dict(
        runner="pingpong",
        dt=0.01,
        k_track=0,
        adaptive=True,
        max_disp=0.35,
        nr_damping=1.0,
        nr_max_step_norm=0.3,
    ),
    # Damped NR-GAD variants
    "nr_gad_damped_02": dict(
        runner="pingpong",
        dt=0.005,
        k_track=0,
        adaptive=False,
        max_disp=0.35,
        nr_damping=0.2,
        nr_max_step_norm=0.1,
    ),
    "nr_gad_damped_01": dict(
        runner="pingpong",
        dt=0.005,
        k_track=0,
        adaptive=False,
        max_disp=0.35,
        nr_damping=0.1,
        nr_max_step_norm=0.05,
    ),
    "nr_gad_damped_03": dict(
        runner="pingpong",
        dt=0.005,
        k_track=0,
        adaptive=False,
        max_disp=0.35,
        nr_damping=0.3,
        nr_max_step_norm=0.15,
    ),
    # Preconditioned GAD
    "precond_gad_001": dict(
        runner="gad",
        dt=0.005,
        k_track=0,
        adaptive=False,
        max_disp=0.35,
        preconditioned=True,
        eig_floor=0.01,
    ),
    "precond_gad_005": dict(
        runner="gad",
        dt=0.005,
        k_track=0,
        adaptive=False,
        max_disp=0.35,
        preconditioned=True,
        eig_floor=0.05,
    ),
    "precond_gad_01": dict(
        runner="gad",
        dt=0.005,
        k_track=0,
        adaptive=False,
        max_disp=0.35,
        preconditioned=True,
        eig_floor=0.1,
    ),
    "precond_gad_dt01": dict(
        runner="gad",
        dt=0.01,
        k_track=0,
        adaptive=False,
        max_disp=0.35,
        preconditioned=True,
        eig_floor=0.01,
    ),
    # === Round 2: A — Pure GAD improvements ===
    # A1: Corrected adaptive dt (Multi-Mode GAD parameters)
    "adaptive_mm": dict(
        runner="gad", dt=0.002, k_track=0, adaptive=True, max_disp=0.35, dt_min=1e-5, dt_max=0.08
    ),
    "adaptive_mm2": dict(
        runner="gad", dt=0.001, k_track=0, adaptive=True, max_disp=0.35, dt_min=1e-5, dt_max=0.05
    ),
    # A2: Smaller fixed dt
    "gad_dt003": dict(runner="gad", dt=0.003, k_track=0, adaptive=False, max_disp=0.35),
    # New 2026-04-17: gad_dt003 without Eckart projection (use raw Hessian eigvec for GAD dynamics).
    # Convergence uses fmax (matches Sella's criterion for fair comparison).
    "gad_dt003_no_eckart": dict(
        runner="gad", dt=0.003, k_track=0, adaptive=False, max_disp=0.35,
        use_projection=False,
        force_criterion="fmax", force_threshold=0.01,
    ),
    # gad_dt003 with the new fmax-based criterion (same dynamics as gad_dt003, different criterion)
    "gad_dt003_fmax": dict(
        runner="gad", dt=0.003, k_track=0, adaptive=False, max_disp=0.35,
        force_criterion="fmax", force_threshold=0.01,
    ),
    # A3: Clamping extremes
    "gad_no_clamp": dict(runner="gad", dt=0.005, k_track=0, adaptive=False, max_disp=999.0),
    # A4: Adaptive dt with floor fix
    "adaptive_floor": dict(
        runner="gad", dt=0.005, k_track=0, adaptive=True, max_disp=0.35, dt_min=1e-3, dt_max=0.05
    ),
    # === Round 2: B — Preconditioned descent diagnostic (hard switch, NOT diffusion-compat) ===
    # Preconditioned descent when n_neg>=2, preconditioned GAD when n_neg<2
    # Tests: Is GAD's v₁ ascent helpful at n_neg>=2?
    "precond_descent": dict(
        runner="pingpong",
        dt=0.005,
        k_track=0,
        adaptive=False,
        max_disp=0.35,
        nr_damping=1.0,
        nr_max_step_norm=0.3,
        descent_mode="preconditioned",
        nr_eig_floor=0.01,
    ),
    # === Round 2: C — Blended preconditioned GAD (diffusion-compatible) ===
    # F_blend = F + 2·sigmoid(k·λ₂)·(F·v₁)v₁, then Δx = dt · |H|⁻¹ · F_blend
    # One scalar w controls ascend-v₁-or-not. Everything else is preconditioned descent.
    "blend_k50": dict(
        runner="gad",
        dt=0.005,
        k_track=0,
        adaptive=False,
        max_disp=0.35,
        preconditioned=True,
        eig_floor=0.01,
        blend_sharpness=50.0,
    ),
    "blend_k100": dict(
        runner="gad",
        dt=0.005,
        k_track=0,
        adaptive=False,
        max_disp=0.35,
        preconditioned=True,
        eig_floor=0.01,
        blend_sharpness=100.0,
    ),
    # === Round 3: Blend WITHOUT preconditioning (plain Euler) ===
    # F_blend = F + 2·sigmoid(k·λ₂)·(F·v₁)v₁, Δx = dt · F_blend
    # Tests whether smooth λ₂-blend helps on top of the already-good gad_small_dt base.
    "blend_plain_k50": dict(
        runner="gad", dt=0.005, k_track=0, adaptive=False, max_disp=0.35, blend_sharpness=50.0
    ),
    "blend_plain_k100": dict(
        runner="gad", dt=0.005, k_track=0, adaptive=False, max_disp=0.35, blend_sharpness=100.0
    ),
    "blend_plain_k10": dict(
        runner="gad", dt=0.005, k_track=0, adaptive=False, max_disp=0.35, blend_sharpness=10.0
    ),
    # === Round 3: Even smaller fixed dt ===
    "gad_dt002": dict(runner="gad", dt=0.002, k_track=0, adaptive=False, max_disp=0.35),
    # === Round 4: One-way descent→GAD ===
    # Plain gradient descent (blend_weight=0) until n_neg <= threshold, then GAD permanently
    "descent_then_gad_2": dict(
        runner="gad", dt=0.005, k_track=0, adaptive=False, max_disp=0.35, descent_until_nneg=2
    ),
    "descent_then_gad_3": dict(
        runner="gad", dt=0.005, k_track=0, adaptive=False, max_disp=0.35, descent_until_nneg=3
    ),
    # === Round 4: NR→GAD one-way (Newton descent until n_neg≤2, then GAD permanently) ===
    # Uses nr_gad_pingpong framework but with one-way switch logic
    # Small NR steps via tight displacement cap to prevent overshoot
    "nr_then_gad_cap01": dict(runner="pingpong", dt=0.005, k_track=0, adaptive=False,
                              max_disp=0.01, nr_damping=0.3, nr_max_step_norm=0.01,
                              descent_mode="newton", one_way=True, one_way_threshold=2),
    "nr_then_gad_cap005": dict(runner="pingpong", dt=0.005, k_track=0, adaptive=False,
                               max_disp=0.005, nr_damping=0.3, nr_max_step_norm=0.005,
                               descent_mode="newton", one_way=True, one_way_threshold=2),
    # === Round 4: Multi-mode GAD ===
    # Ascend along ALL negative-eigenvalue modes, not just v₁
    "multimode_all_neg": dict(runner="gad", dt=0.005, k_track=0, adaptive=False, max_disp=0.35,
                              multimode="all_neg"),
    # Smooth differentiable version: sigmoid(-λᵢ·k) weight per mode
    "multimode_smooth": dict(runner="gad", dt=0.005, k_track=0, adaptive=False, max_disp=0.35,
                             multimode="smooth", multimode_sharpness=50.0),
    # Top-2: always flip v₁ and v₂
    "multimode_top2": dict(runner="gad", dt=0.005, k_track=0, adaptive=False, max_disp=0.35,
                           multimode="top2"),
    # === NR-polish on top of GAD: addresses the GAD plateau at fmax≈0.01 ===
    # Switches to spectral-partitioned Newton-Raphson when n_neg=1 (refine
    # phase), GAD otherwise (navigate phase). Driven to fmax<1e-4 (paper-strict).
    "nr_gad_polish_dt007_strict": dict(
        runner="pingpong",
        dt=0.007,
        k_track=0,
        adaptive=False,
        max_disp=0.35,
        nr_damping=1.0,
        nr_max_step_norm=0.3,
        force_criterion="fmax",
        force_threshold=1e-4,
    ),
    "nr_gad_polish_dt007_loose": dict(
        # Same dynamics, looser criterion (matches our other comparisons).
        runner="pingpong",
        dt=0.007,
        k_track=0,
        adaptive=False,
        max_disp=0.35,
        nr_damping=1.0,
        nr_max_step_norm=0.3,
        force_criterion="fmax",
        force_threshold=0.01,
    ),
    # === Tight convergence presets (for high-precision TS refinement) ===
    "gad_projected_fmax1e4": dict(
        runner="gad",
        dt=0.01,
        k_track=0,
        adaptive=False,
        max_disp=0.35,
        force_criterion="fmax",
        force_threshold=1e-4,
    ),
    "gad_small_dt_fmax1e4": dict(
        runner="gad",
        dt=0.005,
        k_track=0,
        adaptive=False,
        max_disp=0.35,
        force_criterion="fmax",
        force_threshold=1e-4,
    ),
    "gad_dt003_fmax1e4": dict(
        runner="gad",
        dt=0.003,
        k_track=0,
        adaptive=False,
        max_disp=0.35,
        force_criterion="fmax",
        force_threshold=1e-4,
    ),
    # === 2026-04-28: bigger GAD dt sweep, matched fmax<0.01 criterion ===
    # Matches Round 6 canonical setup but tests larger step sizes for
    # fairness against Sella's quasi-Newton ~10^-2 Å step.
    "gad_dt005_fmax": dict(
        runner="gad", dt=0.005, k_track=0, adaptive=False, max_disp=0.35,
        force_criterion="fmax", force_threshold=0.01,
    ),
    "gad_dt010_fmax": dict(
        runner="gad", dt=0.010, k_track=0, adaptive=False, max_disp=0.35,
        force_criterion="fmax", force_threshold=0.01,
    ),
    "gad_dt020_fmax": dict(
        runner="gad", dt=0.020, k_track=0, adaptive=False, max_disp=0.35,
        force_criterion="fmax", force_threshold=0.01,
    ),
    # === 2026-04-29: dt grid filling between dt=0.005 (works) and dt=0.010 (collapses) ===
    "gad_dt004_fmax": dict(
        runner="gad", dt=0.004, k_track=0, adaptive=False, max_disp=0.35,
        force_criterion="fmax", force_threshold=0.01,
    ),
    "gad_dt006_fmax": dict(
        runner="gad", dt=0.006, k_track=0, adaptive=False, max_disp=0.35,
        force_criterion="fmax", force_threshold=0.01,
    ),
    "gad_dt007_fmax": dict(
        runner="gad", dt=0.007, k_track=0, adaptive=False, max_disp=0.35,
        force_criterion="fmax", force_threshold=0.01,
    ),
    "gad_dt008_fmax": dict(
        runner="gad", dt=0.008, k_track=0, adaptive=False, max_disp=0.35,
        force_criterion="fmax", force_threshold=0.01,
    ),
    # Very small dt — for diagnostic / dynamics-fidelity studies. With dt this
    # small, fixed step budgets won't always converge; useful for studying
    # *whether* a sample is GAD-amenable in the limit of fine-grained Euler.
    "gad_dt001_fmax": dict(
        runner="gad", dt=0.001, k_track=0, adaptive=False, max_disp=0.35,
        force_criterion="fmax", force_threshold=0.01,
    ),
    "gad_dt0005_fmax": dict(
        runner="gad", dt=0.0005, k_track=0, adaptive=False, max_disp=0.35,
        force_criterion="fmax", force_threshold=0.01,
    ),
    "gad_dt0001_fmax": dict(
        runner="gad", dt=0.0001, k_track=0, adaptive=False, max_disp=0.35,
        force_criterion="fmax", force_threshold=0.01,
    ),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, required=True, choices=list(METHOD_CONFIGS.keys()))
    parser.add_argument("--noise", type=float, required=True, help="Gaussian noise std (Angstrom)")
    parser.add_argument("--n-samples", type=int, default=300)
    parser.add_argument("--n-steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument(
        "--random-offset",
        type=int,
        default=0,
        help="Skip first N samples (for randomized sampling from full dataset)",
    )
    parser.add_argument(
        "--force-threshold",
        type=float,
        default=0.01,
        help="Convergence threshold for selected force criterion",
    )
    parser.add_argument(
        "--force-criterion",
        type=str,
        default="fmax",
        choices=["fmax", "force_norm"],
        help="Force criterion for convergence gating",
    )
    parser.add_argument(
        "--save-ts-xyz",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write converged TS geometries to a multi-frame XYZ file",
    )
    parser.add_argument("--start-from", type=str, default="ts_noised",
                        choices=["ts_noised", "reactant", "product", "midpoint"],
                        help="Initial geometry: noised TS (default), reactant, product, or linear midpoint.")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    noise_pm = int(round(args.noise * 1000))
    mcfg = METHOD_CONFIGS[args.method]
    force_threshold = mcfg.get("force_threshold", args.force_threshold)
    force_criterion = mcfg.get("force_criterion", args.force_criterion)
    print(
        f"Device: {device} | method={args.method} | noise={noise_pm}pm | "
        f"samples={args.n_samples} | steps={args.n_steps} | dt={mcfg['dt']} "
        f"| conv={force_criterion}<{force_threshold}"
    )

    # ---- Paths ----
    for ckpt_path in [
        "/lustre06/project/6033559/memoozd/models/hip_v2.ckpt",
        "/project/rrg-aspuru/memoozd/models/hip_v2.ckpt",
    ]:
        if os.path.exists(ckpt_path):
            break
    else:
        sys.exit("hip_v2.ckpt not found")

    for h5_path in [
        "/lustre06/project/6033559/memoozd/data/transition1x.h5",
        "/project/rrg-aspuru/memoozd/data/transition1x.h5",
    ]:
        if os.path.exists(h5_path):
            break
    else:
        sys.exit("transition1x.h5 not found")

    output_dir = args.output_dir or "/lustre07/scratch/memoozd/gadplus/runs/method_cmp_300"
    os.makedirs(output_dir, exist_ok=True)

    # ---- Load HIP ----
    from gadplus.calculator.hip import load_hip_calculator, make_hip_predict_fn

    calculator = load_hip_calculator(ckpt_path, device=device)
    predict_fn = make_hip_predict_fn(calculator)
    print("HIP loaded")

    # ---- Load dataset ----
    from gadplus.data.transition1x import Transition1xDataset, UsePos

    dataset = Transition1xDataset(
        h5_path,
        split=args.split,
        max_samples=args.n_samples + args.random_offset,
        transform=UsePos("pos_transition"),
    )
    print(f"Loaded {len(dataset)} samples (split={args.split})")

    # ---- Build config ----
    from gadplus.logging.trajectory import TrajectoryLogger

    runner = mcfg["runner"]

    # Lazy imports per runner to avoid import errors from unused modules
    if runner == "gad":
        from gadplus.search.gad_search import GADSearchConfig, run_gad_search
    elif runner == "pingpong":
        from gadplus.search.nr_gad_pingpong import NRGADPingPongConfig, run_nr_gad_pingpong
    elif runner == "blended":
        from gadplus.search.blended_gad import BlendedGADConfig, run_blended_gad
    elif runner == "rfo_gad":
        from gadplus.search.rfo_gad import RFOGADConfig, run_rfo_gad

    if runner == "gad":
        cfg = GADSearchConfig(
            n_steps=args.n_steps,
            dt=mcfg["dt"],
            k_track=mcfg["k_track"],
            use_projection=mcfg.get("use_projection", True),
            use_adaptive_dt=mcfg.get("adaptive", False),
            dt_min=mcfg.get("dt_min", 1e-4),
            dt_max=mcfg.get("dt_max", 0.05),
            dt_adaptation="eigenvalue_clamped",
            max_atom_disp=mcfg["max_disp"],
            force_threshold=force_threshold,
            force_criterion=force_criterion,
            use_preconditioning=mcfg.get("preconditioned", False),
            eig_floor=mcfg.get("eig_floor", 0.01),
            blend_sharpness=mcfg.get("blend_sharpness", 0.0),
            descent_until_nneg=mcfg.get("descent_until_nneg", 0),
            multimode=mcfg.get("multimode", ""),
            multimode_sharpness=mcfg.get("multimode_sharpness", 50.0),
        )
    elif runner == "pingpong":
        cfg = NRGADPingPongConfig(
            max_steps=args.n_steps,
            gad_dt=mcfg["dt"],
            k_track=mcfg["k_track"],
            use_adaptive_dt=mcfg.get("adaptive", False),
            dt_min=mcfg.get("dt_min", 1e-4),
            dt_max=mcfg.get("dt_max", 0.05),
            nr_max_step=0.3,
            nr_eig_floor=mcfg.get("nr_eig_floor", 1e-6),
            nr_damping=mcfg.get("nr_damping", 0.2),
            nr_max_step_norm=mcfg.get("nr_max_step_norm", 0.1),
            max_atom_disp=mcfg["max_disp"],
            force_threshold=force_threshold,
            force_criterion=force_criterion,
            descent_mode=mcfg.get("descent_mode", "newton"),
            one_way=mcfg.get("one_way", False),
            one_way_threshold=mcfg.get("one_way_threshold", 2),
        )
    elif runner == "blended":
        cfg = BlendedGADConfig(
            n_steps=args.n_steps,
            dt=mcfg["dt"],
            k_track=mcfg["k_track"],
            blend_sharpness=mcfg.get("blend_sharpness", 50.0),
            max_atom_disp=mcfg["max_disp"],
            force_threshold=force_threshold,
            force_criterion=force_criterion,
        )
    elif runner == "rfo_gad":
        cfg = RFOGADConfig(
            n_steps=args.n_steps,
            dt=mcfg["dt"],
            k_track=mcfg["k_track"],
            max_atom_disp=mcfg["max_disp"],
            force_threshold=force_threshold,
            force_criterion=force_criterion,
        )
    else:
        sys.exit(f"Unknown runner: {runner}")

    # ---- Sample range (supports random offset into full dataset) ----
    offset = args.random_offset
    sample_indices = list(range(offset, len(dataset)))
    print(f"Sample range: [{offset}, {len(dataset)}) = {len(sample_indices)} samples")

    # ---- Pre-generate noise ----
    torch.manual_seed(args.seed)
    noise_vecs = {}
    for i in sample_indices:
        sample = dataset[i]
        noise_vecs[i] = torch.randn_like(sample.pos) * args.noise

    # ---- Run ----
    run_id = f"{args.method}_{noise_pm}pm_{uuid.uuid4().hex[:8]}"
    results = []
    ts_atoms: list[Atoms] = []
    ts_index_rows: list[dict] = []
    t_total = time.time()
    ts_xyz_path = os.path.join(output_dir, f"ts_all_{args.method}_{noise_pm}pm.xyz")
    ts_index_path = os.path.join(output_dir, f"ts_index_{args.method}_{noise_pm}pm.parquet")

    for i in sample_indices:
        sample = dataset[i]
        coords_ts = sample.pos.to(device)
        z = sample.z.to(device)
        formula = getattr(sample, "formula", f"sample_{i}")

        if args.start_from == "ts_noised":
            coords_start = coords_ts + noise_vecs[i].to(device)
            start_method_str = f"noised_ts_{noise_pm}pm"
        elif args.start_from == "reactant":
            if not hasattr(sample, "pos_reactant"):
                print(f"  [{i:3d}] {formula:>12s} | SKIP: no pos_reactant on sample")
                continue
            coords_start = sample.pos_reactant.to(device)
            start_method_str = "reactant"
        elif args.start_from == "product":
            if not hasattr(sample, "pos_product"):
                print(f"  [{i:3d}] {formula:>12s} | SKIP: no pos_product on sample")
                continue
            pos_p = sample.pos_product.to(device)
            if pos_p.abs().sum() < 1e-6:
                print(f"  [{i:3d}] {formula:>12s} | SKIP: pos_product is all zeros")
                continue
            coords_start = pos_p
            start_method_str = "product"
        elif args.start_from == "midpoint":
            if not hasattr(sample, "pos_reactant") or not hasattr(sample, "pos_product"):
                print(f"  [{i:3d}] {formula:>12s} | SKIP: midpoint needs reactant+product")
                continue
            pos_r = sample.pos_reactant.to(device)
            pos_p = sample.pos_product.to(device)
            if pos_p.abs().sum() < 1e-6:
                print(f"  [{i:3d}] {formula:>12s} | SKIP: pos_product missing")
                continue
            coords_start = 0.5 * (pos_r + pos_p)
            start_method_str = "midpoint"

        logger = TrajectoryLogger(
            output_dir=output_dir,
            run_id=run_id,
            sample_id=i,
            start_method=start_method_str,
            search_method=args.method,
            formula=formula,
        )

        t0 = time.time()
        if runner == "gad":
            result = run_gad_search(
                predict_fn, coords_start, z, cfg, logger=logger, known_ts_coords=coords_ts
            )
        elif runner == "pingpong":
            result = run_nr_gad_pingpong(
                predict_fn, coords_start, z, cfg, logger=logger, known_ts_coords=coords_ts
            )
        elif runner == "blended":
            result = run_blended_gad(
                predict_fn, coords_start, z, cfg, logger=logger, known_ts_coords=coords_ts
            )
        elif runner == "rfo_gad":
            result = run_rfo_gad(
                predict_fn, coords_start, z, cfg, logger=logger, known_ts_coords=coords_ts
            )
        wall = time.time() - t0
        logger.flush()

        status = "CONV" if result.converged else "FAIL"
        print(
            f"  [{i:3d}] {formula:>12s} | {status} | n_neg={result.final_n_neg} "
            f"| force_norm={result.final_force_norm:.4f} "
            f"| force_max={result.final_force_max:.4f} "
            f"| steps={result.total_steps:3d} | {wall:.1f}s"
        )

        final_coords_flat = result.final_coords.reshape(-1).float().tolist()
        ts_xyz_frame = None
        if result.converged and args.save_ts_xyz:
            frame_idx = len(ts_atoms)
            ts_xyz_frame = frame_idx
            atoms = Atoms(
                numbers=z.detach().cpu().numpy(),
                positions=result.final_coords.detach().cpu().numpy(),
            )
            atoms.info["sample_id"] = int(i)
            atoms.info["formula"] = str(formula)
            atoms.info["method"] = args.method
            atoms.info["noise_pm"] = int(noise_pm)
            ts_atoms.append(atoms)
            ts_index_rows.append(
                {
                    "run_id": run_id,
                    "method": args.method,
                    "noise_pm": noise_pm,
                    "sample_id": i,
                    "formula": formula,
                    "frame_index": frame_idx,
                    "converged_step": result.converged_step,
                    "final_force_norm": result.final_force_norm,
                    "final_force_max": result.final_force_max,
                }
            )

        results.append(
            {
                "run_id": run_id,
                "method": args.method,
                "noise_pm": noise_pm,
                "sample_id": i,
                "formula": formula,
                "converged": result.converged,
                "converged_step": result.converged_step,
                "total_steps": result.total_steps,
                "final_n_neg": result.final_n_neg,
                "final_force_norm": result.final_force_norm,
                "final_force_max": result.final_force_max,
                "final_energy": result.final_energy,
                "final_eig0": result.final_eig0,
                "final_coords_flat": final_coords_flat,
                "ts_xyz_path": ts_xyz_path if result.converged and args.save_ts_xyz else None,
                "ts_xyz_frame": ts_xyz_frame,
                "wall_time_s": wall,
            }
        )

    total_wall = time.time() - t_total

    # ---- Save ----
    df = pd.DataFrame(results)
    out_path = os.path.join(output_dir, f"summary_{args.method}_{noise_pm}pm.parquet")
    df.to_parquet(out_path)

    if args.save_ts_xyz and ts_atoms:
        ase_write(ts_xyz_path, ts_atoms)
        pd.DataFrame(ts_index_rows).to_parquet(ts_index_path, index=False)
        print(f"Saved TS XYZ: {ts_xyz_path} ({len(ts_atoms)} frames)")
        print(f"Saved TS index: {ts_index_path}")
    elif args.save_ts_xyz:
        print("No converged structures; TS XYZ not written.")

    n_conv = df["converged"].sum()
    rate = 100 * n_conv / len(df)
    avg_steps = df.loc[df["converged"], "converged_step"].mean()
    print(f"\n{'=' * 60}")
    print(
        f"{args.method} @ {noise_pm}pm: {n_conv}/{len(df)} ({rate:.1f}%), "
        f"avg steps={avg_steps:.0f}, wall={total_wall:.0f}s ({total_wall / 60:.1f}min)"
    )
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
