#!/usr/bin/env python
"""Phase 2: Noise robustness survey with gad_projected (Level 2).

Runs gad_projected on N train-split samples at varying noise levels.
Outputs per-sample summary Parquet + per-step trajectory Parquet.

Usage:
  python scripts/noise_survey.py --noise 0.1 --n-samples 50 --n-steps 300
  # Submit one per noise level via scripts/run_noise_survey.slurm
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
    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument("--k-track", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}, noise={args.noise}A, samples={args.n_samples}, "
          f"steps={args.n_steps}, dt={args.dt}, k_track={args.k_track}")

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

    noise_pm = int(round(args.noise * 1000))  # Angstrom to picometers
    output_dir = args.output_dir or f"/lustre07/scratch/memoozd/gadplus/runs/noise_survey"
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
    from gadplus.logging.trajectory import TrajectoryLogger

    cfg = GADSearchConfig(
        n_steps=args.n_steps,
        dt=args.dt,
        k_track=args.k_track,
        use_projection=True,
        use_adaptive_dt=False,
        force_threshold=0.01,
    )

    run_id = f"noise_{noise_pm}pm_{uuid.uuid4().hex[:8]}"
    torch.manual_seed(args.seed)

    summaries = []
    t_total = time.time()

    for i in range(len(dataset)):
        sample = dataset[i]
        coords_ts = sample.pos.to(device)
        z = sample.z.to(device)
        formula = getattr(sample, "formula", f"sample_{i}")
        rxn = getattr(sample, "rxn", "")

        # Add noise
        noise = torch.randn_like(coords_ts) * args.noise
        coords_start = coords_ts + noise

        logger = TrajectoryLogger(
            output_dir=output_dir,
            run_id=run_id,
            sample_id=i,
            start_method=f"noised_ts_{noise_pm}pm",
            search_method="gad_projected",
            formula=formula,
            rxn=rxn,
        )

        t0 = time.time()
        result = run_gad_search(
            predict_fn, coords_start, z, cfg,
            logger=logger,
            known_ts_coords=coords_ts,
        )
        wall = time.time() - t0

        # Flush trajectory
        logger.flush()

        status = "CONV" if result.converged else "FAIL"
        print(f"  [{i:3d}] {formula:>12s} | {status} | n_neg={result.final_n_neg} "
              f"| force={result.final_force_norm:.4f} | steps={result.total_steps:3d} | {wall:.1f}s")

        summaries.append({
            "run_id": run_id,
            "sample_id": i,
            "formula": formula,
            "rxn": rxn,
            "noise_angstrom": args.noise,
            "noise_pm": noise_pm,
            "start_method": f"noised_ts_{noise_pm}pm",
            "search_method": "gad_projected",
            "dt": args.dt,
            "k_track": args.k_track,
            "n_steps": args.n_steps,
            "converged": result.converged,
            "converged_step": result.converged_step,
            "total_steps": result.total_steps,
            "final_n_neg": result.final_n_neg,
            "final_force_norm": result.final_force_norm,
            "final_energy": result.final_energy,
            "final_eig0": result.final_eig0,
            "wall_time_s": wall,
            "failure_type": result.failure_type,
        })

    total_wall = time.time() - t_total

    # ---- Save summary ----
    df = pd.DataFrame(summaries)
    summary_path = os.path.join(output_dir, f"summary_{run_id}.parquet")
    df.to_parquet(summary_path)

    n_conv = df["converged"].sum()
    rate = 100 * n_conv / len(df)
    avg_steps = df.loc[df["converged"], "converged_step"].mean()
    avg_time = df["wall_time_s"].mean()

    print(f"\n{'='*60}")
    print(f"NOISE={args.noise}A ({noise_pm}pm): {n_conv}/{len(df)} converged ({rate:.1f}%)")
    print(f"Avg conv steps: {avg_steps:.0f}, Avg time/sample: {avg_time:.1f}s")
    print(f"Total wall time: {total_wall:.0f}s ({total_wall/60:.1f}min)")
    print(f"Summary saved: {summary_path}")

    if not df["converged"].all():
        print(f"\nFailure types:")
        print(df.loc[~df["converged"], "failure_type"].value_counts().to_string())


if __name__ == "__main__":
    main()
