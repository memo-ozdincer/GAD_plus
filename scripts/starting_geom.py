#!/usr/bin/env python
"""Phase 3: Starting geometry comparison with gad_projected (Level 2).

Compares GAD convergence from different starting geometries:
  - Noised TS (0.01A = 10pm)
  - Reactant
  - Product
  - Midpoint (linear interpolation reactant→product)

Usage:
  python scripts/starting_geom.py --start noised_ts --n-samples 50
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


def get_starting_coords(sample, start_type: str, device, noise: float = 0.01):
    """Get starting coordinates based on start_type."""
    coords_ts = sample.pos.to(device)

    if start_type == "noised_ts":
        noise_vec = torch.randn_like(coords_ts) * noise
        return coords_ts + noise_vec

    elif start_type == "reactant":
        if hasattr(sample, "pos_reactant"):
            return sample.pos_reactant.to(device)
        return None

    elif start_type == "product":
        if hasattr(sample, "pos_product"):
            pos_prod = sample.pos_product.to(device)
            # Check if product is all zeros (missing)
            if pos_prod.abs().sum() < 1e-6:
                return None
            return pos_prod
        return None

    elif start_type == "midpoint":
        if hasattr(sample, "pos_reactant") and hasattr(sample, "pos_product"):
            pos_r = sample.pos_reactant.to(device)
            pos_p = sample.pos_product.to(device)
            if pos_p.abs().sum() < 1e-6:
                return None
            return 0.5 * (pos_r + pos_p)
        return None

    elif start_type == "geodesic_mid":
        if hasattr(sample, "pos_reactant") and hasattr(sample, "pos_product"):
            pos_r = sample.pos_reactant.to(device)
            pos_p = sample.pos_product.to(device)
            if pos_p.abs().sum() < 1e-6:
                return None
            from gadplus.geometry.interpolation import geodesic_interpolation
            # 3 images: reactant, midpoint, product
            images = geodesic_interpolation(pos_r, pos_p, n_images=3)
            return images[1]  # midpoint
        return None

    else:
        raise ValueError(f"Unknown start_type: {start_type}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=str, required=True,
                        choices=["noised_ts", "reactant", "product", "midpoint", "geodesic_mid"])
    parser.add_argument("--n-samples", type=int, default=50)
    parser.add_argument("--n-steps", type=int, default=300)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--k-track", type=int, default=0)
    parser.add_argument("--noise", type=float, default=0.01, help="Noise for noised_ts (Angstrom)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}, start={args.start}, samples={args.n_samples}, "
          f"steps={args.n_steps}, dt={args.dt}")

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

    output_dir = args.output_dir or "/lustre07/scratch/memoozd/gadplus/runs/starting_geom"
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

    run_id = f"start_{args.start}_{uuid.uuid4().hex[:8]}"
    torch.manual_seed(args.seed)

    summaries = []
    skipped = 0
    t_total = time.time()

    for i in range(len(dataset)):
        sample = dataset[i]
        z = sample.z.to(device)
        coords_ts = sample.pos.to(device)
        formula = getattr(sample, "formula", f"sample_{i}")
        rxn = getattr(sample, "rxn", "")

        coords_start = get_starting_coords(sample, args.start, device, args.noise)
        if coords_start is None:
            skipped += 1
            continue

        logger = TrajectoryLogger(
            output_dir=output_dir,
            run_id=run_id,
            sample_id=i,
            start_method=args.start,
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
        logger.flush()

        status = "CONV" if result.converged else "FAIL"
        print(f"  [{i:3d}] {formula:>12s} | {status} | n_neg={result.final_n_neg} "
              f"| force={result.final_force_norm:.4f} | steps={result.total_steps:3d} | {wall:.1f}s")

        summaries.append({
            "run_id": run_id,
            "sample_id": i,
            "formula": formula,
            "rxn": rxn,
            "start_method": args.start,
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
    n_run = len(df)
    rate = 100 * n_conv / n_run if n_run > 0 else 0
    avg_steps = df.loc[df["converged"], "converged_step"].mean()

    print(f"\n{'='*60}")
    print(f"START={args.start}: {n_conv}/{n_run} converged ({rate:.1f}%)")
    if skipped:
        print(f"  Skipped {skipped} samples (missing geometry)")
    print(f"Avg conv steps: {avg_steps:.0f}, Total wall: {total_wall:.0f}s ({total_wall/60:.1f}min)")
    print(f"Summary saved: {summary_path}")

    if not df["converged"].all() and n_run > 0:
        print(f"\nFailure types:")
        print(df.loc[~df["converged"], "failure_type"].value_counts().to_string())


if __name__ == "__main__":
    main()
