#!/usr/bin/env python
"""Phase 6: Basin mapping — at what noise level do we find DIFFERENT transition states?

Start from known TS (noise=0), confirm it stays converged (sanity).
Then increase noise and check if the converged TS matches the original (RMSD < 0.1A).

Usage:
  python scripts/basin_map.py --n-samples 20
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import uuid

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=20)
    parser.add_argument("--n-steps", type=int, default=300)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--rmsd-threshold", type=float, default=0.1,
                        help="RMSD threshold for 'same TS' (Angstrom)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}, samples={args.n_samples}")

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

    output_dir = args.output_dir or "/lustre07/scratch/memoozd/gadplus/runs/basin_map"
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

    # ---- Run ----
    from gadplus.search.gad_search import GADSearchConfig, run_gad_search

    noise_levels_angstrom = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]

    cfg = GADSearchConfig(
        n_steps=args.n_steps,
        dt=args.dt,
        k_track=0,
        use_projection=True,
        use_adaptive_dt=False,
        force_threshold=0.01,
    )

    results = []
    torch.manual_seed(args.seed)

    for i in range(len(dataset)):
        sample = dataset[i]
        coords_ts = sample.pos.to(device)
        z = sample.z.to(device)
        formula = getattr(sample, "formula", f"sample_{i}")

        print(f"\n--- Sample {i}: {formula} ({coords_ts.shape[0]} atoms) ---")

        for noise_a in noise_levels_angstrom:
            noise_pm = int(round(noise_a * 1000))

            if noise_a > 0:
                noise_vec = torch.randn_like(coords_ts) * noise_a
                coords_start = coords_ts + noise_vec
            else:
                coords_start = coords_ts.clone()

            t0 = time.time()
            result = run_gad_search(
                predict_fn, coords_start, z, cfg,
                known_ts_coords=coords_ts,
            )
            wall = time.time() - t0

            # Compute RMSD to original TS
            final_coords = result.final_coords.to(device)
            rmsd = float(torch.sqrt(torch.mean((final_coords - coords_ts) ** 2)).item())
            same_ts = rmsd < args.rmsd_threshold

            status = "CONV" if result.converged else "FAIL"
            same_str = "SAME" if (result.converged and same_ts) else ("DIFF" if result.converged else "N/A")

            print(f"  noise={noise_pm:4d}pm | {status} | {same_str} | "
                  f"RMSD={rmsd:.4f}A | n_neg={result.final_n_neg} | {wall:.1f}s")

            results.append({
                "sample_id": i,
                "formula": formula,
                "noise_angstrom": noise_a,
                "noise_pm": noise_pm,
                "converged": result.converged,
                "converged_step": result.converged_step,
                "rmsd_to_original_ts": rmsd,
                "same_ts": same_ts if result.converged else None,
                "final_n_neg": result.final_n_neg,
                "final_force_norm": result.final_force_norm,
                "final_energy": result.final_energy,
                "wall_time_s": wall,
            })

    # ---- Save and summarize ----
    df = pd.DataFrame(results)
    out_path = os.path.join(output_dir, "basin_map_results.parquet")
    df.to_parquet(out_path)

    print(f"\n{'='*70}")
    print("BASIN MAPPING SUMMARY")
    print("=" * 70)
    print(f"{'Noise (pm)':>10s} {'Conv':>6s} {'Same TS':>8s} {'Diff TS':>8s} {'Avg RMSD':>10s}")
    print("-" * 45)

    for noise_a in noise_levels_angstrom:
        noise_pm = int(round(noise_a * 1000))
        subset = df[df["noise_pm"] == noise_pm]
        n_conv = subset["converged"].sum()
        conv_subset = subset[subset["converged"]]
        n_same = conv_subset["same_ts"].sum() if len(conv_subset) > 0 else 0
        n_diff = len(conv_subset) - n_same
        avg_rmsd = conv_subset["rmsd_to_original_ts"].mean() if len(conv_subset) > 0 else float("nan")

        print(f"{noise_pm:10d} {n_conv:3d}/{len(subset):2d} {int(n_same):8d} {int(n_diff):8d} {avg_rmsd:10.4f}")

    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
