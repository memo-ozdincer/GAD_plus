#!/usr/bin/env python
"""Method comparison: GAD vs NR-GAD ping-pong vs adaptive-dt GAD.

Compares multiple search strategies across noise levels on the same samples.
Each method runs on the same starting geometries (same seed + noise).

Methods:
  1. gad_projected:  Level 2 baseline (Eckart projection, fixed dt=0.01)
  2. gad_adaptive:   Level 3 (Eckart + eigenvalue-clamped adaptive dt)
  3. nr_gad_pingpong: NR when n_neg>=2 (pure descent), GAD when n_neg<2
  4. nr_gad_pp_adaptive: Ping-pong + adaptive dt in GAD phase

Usage:
  python scripts/method_comparison.py --noise 0.1 --n-samples 50
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--noise", type=float, required=True, help="Gaussian noise std (Angstrom)")
    parser.add_argument("--n-samples", type=int, default=50)
    parser.add_argument("--n-steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    noise_pm = int(round(args.noise * 1000))
    print(f"Device: {device}, noise={args.noise}A ({noise_pm}pm), "
          f"samples={args.n_samples}, steps={args.n_steps}")

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

    output_dir = args.output_dir or "/lustre07/scratch/memoozd/gadplus/runs/method_comparison"
    os.makedirs(output_dir, exist_ok=True)

    # ---- Load HIP ----
    from gadplus.calculator.hip import load_hip_calculator, make_hip_predict_fn
    calculator = load_hip_calculator(ckpt_path, device=device)
    predict_fn = make_hip_predict_fn(calculator)
    print("HIP loaded")

    # ---- Load dataset ----
    from gadplus.data.transition1x import Transition1xDataset, UsePos
    dataset = Transition1xDataset(
        h5_path, split=args.split, max_samples=args.n_samples,
        transform=UsePos("pos_transition"),
    )
    print(f"Loaded {len(dataset)} samples (split={args.split})")

    # ---- Pre-generate noise ----
    torch.manual_seed(args.seed)
    noise_per_sample = []
    for i in range(len(dataset)):
        sample = dataset[i]
        noise_per_sample.append(torch.randn_like(sample.pos) * args.noise)

    # ---- Define methods ----
    from gadplus.search.gad_search import GADSearchConfig, run_gad_search
    from gadplus.search.nr_gad_pingpong import NRGADPingPongConfig, run_nr_gad_pingpong
    from gadplus.logging.trajectory import TrajectoryLogger

    methods = {
        # --- Baseline ---
        "gad_projected": {
            "runner": "gad",
            "config": GADSearchConfig(
                n_steps=args.n_steps, dt=0.01, k_track=0,
                use_projection=True, use_adaptive_dt=False,
                force_threshold=0.01,
            ),
        },
        # --- Adaptive dt (eigenvalue-clamped): dt ~ 1/clamp(|eig0|) ---
        "gad_adaptive_dt": {
            "runner": "gad",
            "config": GADSearchConfig(
                n_steps=args.n_steps, dt=0.01, k_track=0,
                use_projection=True, use_adaptive_dt=True,
                dt_min=1e-4, dt_max=0.05, dt_adaptation="eigenvalue_clamped",
                force_threshold=0.01,
            ),
        },
        # --- Tight displacement clamp: 0.1A max per atom (default 0.35) ---
        "gad_tight_clamp": {
            "runner": "gad",
            "config": GADSearchConfig(
                n_steps=args.n_steps, dt=0.01, k_track=0,
                use_projection=True, use_adaptive_dt=False,
                max_atom_disp=0.1,
                force_threshold=0.01,
            ),
        },
        # --- Adaptive dt + tight clamp combo ---
        "gad_adaptive_tight": {
            "runner": "gad",
            "config": GADSearchConfig(
                n_steps=args.n_steps, dt=0.01, k_track=0,
                use_projection=True, use_adaptive_dt=True,
                dt_min=1e-4, dt_max=0.05, dt_adaptation="eigenvalue_clamped",
                max_atom_disp=0.1,
                force_threshold=0.01,
            ),
        },
        # --- Small dt (conservative): dt=0.005 for stability ---
        "gad_small_dt": {
            "runner": "gad",
            "config": GADSearchConfig(
                n_steps=args.n_steps, dt=0.005, k_track=0,
                use_projection=True, use_adaptive_dt=False,
                force_threshold=0.01,
            ),
        },
        # --- NR-GAD ping-pong: NR when n_neg>=2, GAD when n_neg<2 ---
        "nr_gad_pingpong": {
            "runner": "pingpong",
            "config": NRGADPingPongConfig(
                max_steps=args.n_steps, gad_dt=0.01, k_track=0,
                use_adaptive_dt=False,
                nr_max_step=0.3, nr_eig_floor=1e-6,
                force_threshold=0.01,
            ),
        },
        # --- NR-GAD ping-pong + adaptive dt in GAD phase ---
        "nr_gad_pp_adaptive": {
            "runner": "pingpong",
            "config": NRGADPingPongConfig(
                max_steps=args.n_steps, gad_dt=0.01, k_track=0,
                use_adaptive_dt=True, dt_min=1e-4, dt_max=0.05,
                nr_max_step=0.3, nr_eig_floor=1e-6,
                force_threshold=0.01,
            ),
        },
    }

    # ---- Run all methods ----
    all_results = []
    run_tag = f"cmp_{noise_pm}pm_{uuid.uuid4().hex[:8]}"

    for method_name, method_spec in methods.items():
        print(f"\n{'='*60}")
        print(f"Method: {method_name}")
        print(f"{'='*60}")

        n_conv = 0
        steps_if_conv = []

        for i in range(len(dataset)):
            sample = dataset[i]
            coords_ts = sample.pos.to(device)
            z = sample.z.to(device)
            formula = getattr(sample, "formula", f"sample_{i}")

            coords_start = coords_ts + noise_per_sample[i].to(device)

            logger = TrajectoryLogger(
                output_dir=output_dir,
                run_id=f"{run_tag}_{method_name}",
                sample_id=i,
                start_method=f"noised_ts_{noise_pm}pm",
                search_method=method_name,
                formula=formula,
            )

            t0 = time.time()
            if method_spec["runner"] == "gad":
                result = run_gad_search(
                    predict_fn, coords_start, z, method_spec["config"],
                    logger=logger, known_ts_coords=coords_ts,
                )
            else:
                result = run_nr_gad_pingpong(
                    predict_fn, coords_start, z, method_spec["config"],
                    logger=logger, known_ts_coords=coords_ts,
                )
            wall = time.time() - t0
            logger.flush()

            status = "CONV" if result.converged else "FAIL"
            if result.converged:
                n_conv += 1
                steps_if_conv.append(result.converged_step or result.total_steps)

            print(f"  [{i:3d}] {formula:>12s} | {status} | n_neg={result.final_n_neg} "
                  f"| force={result.final_force_norm:.4f} | steps={result.total_steps:3d} | {wall:.1f}s")

            all_results.append({
                "method": method_name,
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

        rate = 100 * n_conv / len(dataset)
        avg_steps = sum(steps_if_conv) / len(steps_if_conv) if steps_if_conv else float("nan")
        print(f"\n  => {method_name}: {n_conv}/{len(dataset)} ({rate:.0f}%), avg steps={avg_steps:.0f}")

    # ---- Save and summarize ----
    df = pd.DataFrame(all_results)
    out_path = os.path.join(output_dir, f"comparison_{noise_pm}pm.parquet")
    df.to_parquet(out_path)

    print(f"\n{'='*70}")
    print(f"METHOD COMPARISON at noise={noise_pm}pm")
    print(f"{'='*70}")
    print(f"{'Method':>25s} {'Conv':>8s} {'Rate':>6s} {'Avg Steps':>10s} {'Avg Time':>10s}")
    print("-" * 62)

    for method_name in methods:
        m = df[df["method"] == method_name]
        n_conv = m["converged"].sum()
        rate = 100 * n_conv / len(m)
        avg_steps = m.loc[m["converged"], "converged_step"].mean()
        avg_time = m["wall_time_s"].mean()
        print(f"{method_name:>25s} {n_conv:3d}/{len(m):3d} {rate:5.1f}% {avg_steps:10.0f} {avg_time:9.1f}s")

    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
