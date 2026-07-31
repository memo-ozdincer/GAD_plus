#!/usr/bin/env python
"""Pure GAD sweep: all noise levels × all starting geometries.

Designed to be run as individual SLURM jobs (one per config) for parallelism.
Each job handles one starting geometry at one noise level across 50 samples.

Usage:
    # Single config:
    python scripts/pure_gad_sweep.py --start noised_ts --noise-pm 100 --dt 0.005 --k-track 0

    # Launched by run_pure_gad_sweep.sh via sbatch
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import uuid

import torch
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from gadplus.calculator.hip import load_hip_calculator, make_hip_predict_fn
from gadplus.data.transition1x import Transition1xDataset, UsePos
from gadplus.geometry.noise import add_gaussian_noise
from gadplus.geometry.starting import make_starting_coords
from gadplus.logging.trajectory import TrajectoryLogger
from gadplus.logging.autopsy import classify_failure
from gadplus.search.gad_search import GADSearchConfig, run_gad_search


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", required=True, choices=["noised_ts", "reactant", "product", "midpoint_rt"])
    parser.add_argument("--noise-pm", type=int, default=0, help="Noise in picometers (only for noised_ts)")
    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument("--k-track", type=int, default=0)
    parser.add_argument("--n-steps", type=int, default=300)
    parser.add_argument("--n-samples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str,
                        default="/lustre07/scratch/memoozd/gadplus/runs/pure_gad_sweep")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    print(f"Device: {device}")
    print(f"Config: start={args.start}, noise={args.noise_pm}pm, dt={args.dt}, k_track={args.k_track}")

    # Paths
    ckpt = "/lustre06/project/6033559/memoozd/models/hip_v2.ckpt"
    h5 = "/lustre06/project/6033559/memoozd/data/transition1x.h5"
    os.makedirs(args.output_dir, exist_ok=True)

    # Load
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
    run_id = str(uuid.uuid4())[:8]
    start_label = f"{args.start}_noise{args.noise_pm}pm"
    results = []

    t_total = time.time()
    for i in range(len(dataset)):
        sample = dataset[i]
        coords = make_starting_coords(sample, args.start,
                                      noise_rms=noise_ang, seed=args.seed + i)
        coords = coords.to(device)
        z = sample.z.to(device)
        known_ts = sample.pos_transition.to(device)
        formula = sample.formula if hasattr(sample, "formula") else f"sample_{i}"
        rxn = sample.rxn if hasattr(sample, "rxn") else ""

        logger = TrajectoryLogger(
            output_dir=args.output_dir,
            run_id=run_id,
            sample_id=i,
            start_method=start_label,
            search_method="pure_gad",
            rxn=rxn,
            formula=formula,
        )

        result = run_gad_search(predict_fn, coords, z, cfg,
                                logger=logger, known_ts_coords=known_ts)

        if not result.converged and logger.rows:
            failure_type = classify_failure(logger.rows)
            result.failure_type = failure_type.value

        results.append({
            "sample_id": i,
            "formula": formula,
            "rxn": rxn,
            "start_method": start_label,
            "search_method": "pure_gad",
            "dt": args.dt,
            "k_track": args.k_track,
            "converged": result.converged,
            "converged_step": result.converged_step,
            "total_steps": result.total_steps,
            "final_n_neg": result.final_n_neg,
            "final_force_norm": result.final_force_norm,
            "final_energy": result.final_energy,
            "final_eig0": result.final_eig0,
            "wall_time_s": result.wall_time_s,
            "failure_type": result.failure_type,
        })

        status = "CONV" if result.converged else "FAIL"
        print(f"  [{i:3d}] {formula:>12s} | {status} | steps={result.total_steps:3d} | "
              f"n_neg={result.final_n_neg} | force={result.final_force_norm:.4f}")

    # Save summary
    summary_path = os.path.join(args.output_dir, f"summary_{start_label}_{run_id}.parquet")
    table = pa.Table.from_pylist(results)
    pq.write_table(table, summary_path)

    elapsed = time.time() - t_total
    n_conv = sum(1 for r in results if r["converged"])
    print(f"\n{'='*60}")
    print(f"DONE: {n_conv}/{len(results)} converged ({100*n_conv/max(len(results),1):.1f}%)")
    print(f"Time: {elapsed:.1f}s ({elapsed/60:.1f}min)")
    print(f"Summary: {summary_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
