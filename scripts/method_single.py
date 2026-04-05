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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


METHOD_CONFIGS = {
    "gad_projected": dict(runner="gad", dt=0.01, k_track=0, adaptive=False, max_disp=0.35),
    "gad_small_dt": dict(runner="gad", dt=0.005, k_track=0, adaptive=False, max_disp=0.35),
    "gad_adaptive_dt": dict(runner="gad", dt=0.01, k_track=0, adaptive=True, max_disp=0.35),
    "gad_tight_clamp": dict(runner="gad", dt=0.01, k_track=0, adaptive=False, max_disp=0.1),
    "gad_adaptive_tight": dict(runner="gad", dt=0.01, k_track=0, adaptive=True, max_disp=0.1),
    "nr_gad_pingpong": dict(runner="pingpong", dt=0.01, k_track=0, adaptive=False, max_disp=0.35,
                            nr_damping=1.0, nr_max_step_norm=0.3),  # original (undamped)
    "nr_gad_pp_adaptive": dict(runner="pingpong", dt=0.01, k_track=0, adaptive=True, max_disp=0.35,
                               nr_damping=1.0, nr_max_step_norm=0.3),
    # Damped NR-GAD variants
    "nr_gad_damped_02": dict(runner="pingpong", dt=0.005, k_track=0, adaptive=False, max_disp=0.35,
                             nr_damping=0.2, nr_max_step_norm=0.1),
    "nr_gad_damped_01": dict(runner="pingpong", dt=0.005, k_track=0, adaptive=False, max_disp=0.35,
                             nr_damping=0.1, nr_max_step_norm=0.05),
    "nr_gad_damped_03": dict(runner="pingpong", dt=0.005, k_track=0, adaptive=False, max_disp=0.35,
                             nr_damping=0.3, nr_max_step_norm=0.15),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, required=True, choices=list(METHOD_CONFIGS.keys()))
    parser.add_argument("--noise", type=float, required=True, help="Gaussian noise std (Angstrom)")
    parser.add_argument("--n-samples", type=int, default=300)
    parser.add_argument("--n-steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--random-offset", type=int, default=0,
                        help="Skip first N samples (for randomized sampling from full dataset)")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    noise_pm = int(round(args.noise * 1000))
    mcfg = METHOD_CONFIGS[args.method]
    print(f"Device: {device} | method={args.method} | noise={noise_pm}pm | "
          f"samples={args.n_samples} | steps={args.n_steps} | dt={mcfg['dt']}")

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
        h5_path, split=args.split,
        max_samples=args.n_samples + args.random_offset,
        transform=UsePos("pos_transition"),
    )
    print(f"Loaded {len(dataset)} samples (split={args.split})")

    # ---- Build config ----
    from gadplus.search.gad_search import GADSearchConfig, run_gad_search
    from gadplus.search.nr_gad_pingpong import NRGADPingPongConfig, run_nr_gad_pingpong
    from gadplus.logging.trajectory import TrajectoryLogger

    if mcfg["runner"] == "gad":
        cfg = GADSearchConfig(
            n_steps=args.n_steps, dt=mcfg["dt"], k_track=mcfg["k_track"],
            use_projection=True,
            use_adaptive_dt=mcfg["adaptive"],
            dt_min=1e-4, dt_max=0.05, dt_adaptation="eigenvalue_clamped",
            max_atom_disp=mcfg["max_disp"],
            force_threshold=0.01,
        )
    else:
        cfg = NRGADPingPongConfig(
            max_steps=args.n_steps, gad_dt=mcfg["dt"], k_track=mcfg["k_track"],
            use_adaptive_dt=mcfg["adaptive"],
            dt_min=1e-4, dt_max=0.05,
            nr_max_step=0.3, nr_eig_floor=1e-6,
            nr_damping=mcfg.get("nr_damping", 0.2),
            nr_max_step_norm=mcfg.get("nr_max_step_norm", 0.1),
            max_atom_disp=mcfg["max_disp"],
            force_threshold=0.01,
        )

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
    t_total = time.time()

    for i in sample_indices:
        sample = dataset[i]
        coords_ts = sample.pos.to(device)
        z = sample.z.to(device)
        formula = getattr(sample, "formula", f"sample_{i}")

        coords_start = coords_ts + noise_vecs[i].to(device)

        logger = TrajectoryLogger(
            output_dir=output_dir, run_id=run_id, sample_id=i,
            start_method=f"noised_ts_{noise_pm}pm",
            search_method=args.method, formula=formula,
        )

        t0 = time.time()
        if mcfg["runner"] == "gad":
            result = run_gad_search(predict_fn, coords_start, z, cfg,
                                    logger=logger, known_ts_coords=coords_ts)
        else:
            result = run_nr_gad_pingpong(predict_fn, coords_start, z, cfg,
                                         logger=logger, known_ts_coords=coords_ts)
        wall = time.time() - t0
        logger.flush()

        status = "CONV" if result.converged else "FAIL"
        print(f"  [{i:3d}] {formula:>12s} | {status} | n_neg={result.final_n_neg} "
              f"| force={result.final_force_norm:.4f} | steps={result.total_steps:3d} | {wall:.1f}s")

        results.append({
            "method": args.method,
            "noise_pm": noise_pm,
            "sample_id": i,
            "formula": formula,
            "converged": result.converged,
            "converged_step": result.converged_step,
            "total_steps": result.total_steps,
            "final_n_neg": result.final_n_neg,
            "final_force_norm": result.final_force_norm,
            "final_energy": result.final_energy,
            "final_eig0": result.final_eig0,
            "wall_time_s": wall,
        })

    total_wall = time.time() - t_total

    # ---- Save ----
    df = pd.DataFrame(results)
    out_path = os.path.join(output_dir, f"summary_{args.method}_{noise_pm}pm.parquet")
    df.to_parquet(out_path)

    n_conv = df["converged"].sum()
    rate = 100 * n_conv / len(df)
    avg_steps = df.loc[df["converged"], "converged_step"].mean()
    print(f"\n{'='*60}")
    print(f"{args.method} @ {noise_pm}pm: {n_conv}/{len(df)} ({rate:.1f}%), "
          f"avg steps={avg_steps:.0f}, wall={total_wall:.0f}s ({total_wall/60:.1f}min)")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
