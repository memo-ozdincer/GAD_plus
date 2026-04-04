#!/usr/bin/env python
"""Quick parameter ranging for pure GAD: single (dt, k_track) config.

Launched in parallel by run_param_range.slurm — one job per config.
Writes results to a per-config Parquet file.

Usage:
    python scripts/param_range.py --dt 0.005 --k-track 0
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import torch
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from gadplus.calculator.hip import load_hip_calculator, make_hip_predict_fn
from gadplus.data.transition1x import Transition1xDataset, UsePos
from gadplus.geometry.noise import add_gaussian_noise
from gadplus.search.gad_search import GADSearchConfig, run_gad_search


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dt", type=float, required=True)
    parser.add_argument("--k-track", type=int, required=True)
    parser.add_argument("--noise-pm", type=int, default=50)
    parser.add_argument("--n-samples", type=int, default=10)
    parser.add_argument("--n-steps", type=int, default=300)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Config: dt={args.dt}, k_track={args.k_track}, noise={args.noise_pm}pm")

    ckpt = "/lustre06/project/6033559/memoozd/models/hip_v2.ckpt"
    h5 = "/lustre06/project/6033559/memoozd/data/transition1x.h5"
    out_dir = "/lustre07/scratch/memoozd/gadplus/runs/param_range"
    os.makedirs(out_dir, exist_ok=True)

    print("Loading HIP...")
    calculator = load_hip_calculator(ckpt, device=device)
    predict_fn = make_hip_predict_fn(calculator)

    print("Loading dataset...")
    dataset = Transition1xDataset(h5, split="train", max_samples=args.n_samples,
                                  transform=UsePos("pos_transition"))
    print(f"  {len(dataset)} samples")

    cfg = GADSearchConfig(
        n_steps=args.n_steps,
        dt=args.dt,
        k_track=args.k_track,
        use_projection=False,
        use_adaptive_dt=False,
        force_threshold=0.01,
    )

    noise_ang = args.noise_pm / 100.0
    results = []
    n_conv = 0
    total_steps_conv = 0
    t0 = time.time()

    for i in range(len(dataset)):
        sample = dataset[i]
        coords = sample.pos.to(device)
        z = sample.z.to(device)
        coords_noised = add_gaussian_noise(coords, rms_angstrom=noise_ang, seed=42 + i)

        result = run_gad_search(predict_fn, coords_noised, z, cfg)

        if result.converged:
            n_conv += 1
            total_steps_conv += result.total_steps

        results.append({
            "dt": args.dt,
            "k_track": args.k_track,
            "sample_id": i,
            "converged": result.converged,
            "total_steps": result.total_steps,
            "final_n_neg": result.final_n_neg,
            "final_force_norm": result.final_force_norm,
            "final_eig0": result.final_eig0,
            "wall_time_s": result.wall_time_s,
        })

        status = "CONV" if result.converged else "FAIL"
        print(f"  [{i}] {status} steps={result.total_steps} n_neg={result.final_n_neg} "
              f"force={result.final_force_norm:.4f}")

    elapsed = time.time() - t0
    avg_steps = total_steps_conv / max(n_conv, 1)

    out_path = os.path.join(out_dir, f"param_dt{args.dt}_k{args.k_track}.parquet")
    table = pa.Table.from_pylist(results)
    pq.write_table(table, out_path)

    print(f"\ndt={args.dt} k_track={args.k_track} | conv={n_conv}/{len(dataset)} | "
          f"avg_steps={avg_steps:.0f} | time={elapsed:.1f}s")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
