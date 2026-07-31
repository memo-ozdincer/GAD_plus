#!/usr/bin/env python
"""Phase 1: Parameter sweep for pure GAD (gad_projected, Eckart on).

Grid search over dt and k_track on 10 test-split samples at noise=0.05A.
100 steps each. Outputs a Parquet summary + prints best params.

Usage:
  srun python scripts/sweep_dt.py
  # or: sbatch scripts/run_sweep_dt.slurm
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import uuid
from itertools import product as grid

import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=10)
    parser.add_argument("--n-steps", type=int, default=100)
    parser.add_argument("--noise", type=float, default=0.05, help="Gaussian noise std (Angstrom)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

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

    output_dir = args.output_dir or "/lustre07/scratch/memoozd/gadplus/runs/sweep_dt"
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

    # ---- Sweep grid ----
    dt_values = [0.001, 0.003, 0.005, 0.01, 0.02]
    k_track_values = [0, 4, 8]

    from gadplus.search.gad_search import GADSearchConfig, run_gad_search

    torch.manual_seed(args.seed)
    # Pre-generate noise for reproducibility across configs
    noise_per_sample = []
    for i in range(len(dataset)):
        sample = dataset[i]
        noise = torch.randn_like(sample.pos) * args.noise
        noise_per_sample.append(noise)

    results = []
    total_configs = len(dt_values) * len(k_track_values)
    config_idx = 0

    for dt, k_track in grid(dt_values, k_track_values):
        config_idx += 1
        print(f"\n--- Config {config_idx}/{total_configs}: dt={dt}, k_track={k_track} ---")

        n_conv = 0
        steps_if_conv = []
        times = []

        for i in range(len(dataset)):
            sample = dataset[i]
            coords = sample.pos.to(device) + noise_per_sample[i].to(device)
            z = sample.z.to(device)
            formula = getattr(sample, "formula", f"sample_{i}")

            cfg = GADSearchConfig(
                n_steps=args.n_steps,
                dt=dt,
                k_track=k_track,
                use_projection=True,
                use_adaptive_dt=False,
                force_threshold=0.01,
            )

            t0 = time.time()
            result = run_gad_search(predict_fn, coords, z, cfg)
            wall = time.time() - t0
            times.append(wall)

            status = "CONV" if result.converged else "FAIL"
            if result.converged:
                n_conv += 1
                steps_if_conv.append(result.converged_step or result.total_steps)

            print(f"  [{i}] {formula:>12s} | {status} | n_neg={result.final_n_neg} "
                  f"| force={result.final_force_norm:.4f} | steps={result.total_steps} | {wall:.1f}s")

            results.append({
                "dt": dt,
                "k_track": k_track,
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
        avg_time = sum(times) / len(times)
        print(f"  => {n_conv}/{len(dataset)} converged ({rate:.0f}%), "
              f"avg conv steps={avg_steps:.0f}, avg time={avg_time:.1f}s")

    # ---- Save results ----
    df = pd.DataFrame(results)
    out_path = os.path.join(output_dir, "sweep_dt_results.parquet")
    df.to_parquet(out_path)
    print(f"\nResults saved to {out_path}")

    # ---- Summary table ----
    print("\n" + "=" * 70)
    print("SWEEP SUMMARY")
    print("=" * 70)
    print(f"{'dt':>8s} {'k_track':>7s} {'conv':>6s} {'rate':>6s} {'avg_steps':>10s} {'avg_time':>10s}")
    print("-" * 50)

    summary = df.groupby(["dt", "k_track"]).agg(
        n_conv=("converged", "sum"),
        total=("converged", "count"),
        avg_steps=("converged_step", "mean"),
        avg_time=("wall_time_s", "mean"),
    ).reset_index()
    summary["rate"] = 100 * summary["n_conv"] / summary["total"]

    # Sort by rate desc, then avg_steps asc
    summary = summary.sort_values(["rate", "avg_steps"], ascending=[False, True])

    for _, row in summary.iterrows():
        print(f"{row['dt']:8.3f} {int(row['k_track']):7d} "
              f"{int(row['n_conv']):3d}/{int(row['total']):2d} {row['rate']:5.1f}% "
              f"{row['avg_steps']:10.1f} {row['avg_time']:9.1f}s")

    best = summary.iloc[0]
    print(f"\nBEST: dt={best['dt']}, k_track={int(best['k_track'])}")
    print(f"  Conv rate: {best['rate']:.0f}%, Avg steps: {best['avg_steps']:.0f}")


if __name__ == "__main__":
    main()
