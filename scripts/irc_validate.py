#!/usr/bin/env python
"""Phase 5: IRC validation on converged TS from Phase 2 noise survey.

Takes converged TS geometries from the noise survey and runs IRC
forward/backward to check if the TS connects the intended reactant/product.

Usage:
  python scripts/irc_validate.py --noise-pm 10 --max-validate 10
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--noise-pm", type=int, default=10, help="Noise level (pm) to validate")
    parser.add_argument("--max-validate", type=int, default=10, help="Max converged TS to validate")
    parser.add_argument("--irc-steps", type=int, default=100, help="Max IRC steps per direction")
    parser.add_argument("--rmsd-threshold", type=float, default=0.3, help="RMSD threshold for matching (A)")
    parser.add_argument("--survey-dir", type=str, default=None,
                        help="Directory with noise survey results")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--n-dataset-samples", type=int, default=300,
                        help="Number of dataset samples to load (must cover sample_ids in survey)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}, noise_pm={args.noise_pm}, max_validate={args.max_validate}")

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

    survey_dir = args.survey_dir or "/lustre07/scratch/memoozd/gadplus/runs/noise_survey_300"
    output_dir = args.output_dir or "/lustre07/scratch/memoozd/gadplus/runs/irc_validation"
    os.makedirs(output_dir, exist_ok=True)

    # ---- Load HIP ----
    from gadplus.calculator.hip import load_hip_calculator, make_hip_predict_fn
    calculator = load_hip_calculator(ckpt_path, device=device)
    predict_fn = make_hip_predict_fn(calculator)
    print("HIP loaded")

    # ---- Find converged TS from noise survey ----
    import duckdb

    # Get converged samples at the specified noise level
    converged_df = duckdb.execute(f"""
        SELECT sample_id, run_id, converged_step, final_force_norm, final_n_neg, formula
        FROM '{survey_dir}/summary_*.parquet'
        WHERE converged = true AND noise_pm = {args.noise_pm}
        ORDER BY converged_step ASC
        LIMIT {args.max_validate}
    """).df()

    print(f"Found {len(converged_df)} converged TS at noise={args.noise_pm}pm")
    if len(converged_df) == 0:
        print("No converged TS to validate.")
        return

    # ---- Load dataset to get reference geometries ----
    from gadplus.data.transition1x import Transition1xDataset, UsePos
    dataset = Transition1xDataset(
        h5_path, split="train", max_samples=args.n_dataset_samples,
        transform=UsePos("pos_transition"),
    )

    # ---- Load converged TS coords from trajectory parquet ----
    from gadplus.search.irc_validate import run_irc_validation

    results = []
    for _, row in converged_df.iterrows():
        sample_id = int(row["sample_id"])
        run_id = row["run_id"]
        formula = row["formula"]
        conv_step = int(row["converged_step"])

        print(f"\n--- Validating sample {sample_id} ({formula}), converged at step {conv_step} ---")

        # Get final coords from trajectory
        try:
            traj_df = duckdb.execute(f"""
                SELECT coords_flat
                FROM '{survey_dir}/traj_*.parquet'
                WHERE run_id = '{run_id}' AND sample_id = {sample_id}
                ORDER BY step DESC
                LIMIT 1
            """).df()
        except Exception as e:
            print(f"  Error reading trajectory: {e}")
            results.append({
                "sample_id": sample_id, "formula": formula,
                "intended": False, "half_intended": False,
                "rmsd_reactant": None, "rmsd_product": None,
                "error": str(e),
            })
            continue

        if len(traj_df) == 0:
            print(f"  No trajectory data found")
            results.append({
                "sample_id": sample_id, "formula": formula,
                "intended": False, "half_intended": False,
                "rmsd_reactant": None, "rmsd_product": None,
                "error": "no trajectory",
            })
            continue

        coords_flat = traj_df["coords_flat"].iloc[0]
        n_atoms = len(coords_flat) // 3
        ts_coords = torch.tensor(coords_flat, dtype=torch.float32, device=device).reshape(n_atoms, 3)

        # Get reference geometries
        sample = dataset[sample_id]
        z = sample.z.to(device)
        reactant_coords = sample.pos_reactant.to(device) if hasattr(sample, "pos_reactant") else None
        product_coords = None
        if hasattr(sample, "pos_product"):
            pp = sample.pos_product.to(device)
            if pp.abs().sum() > 1e-6:
                product_coords = pp

        t0 = time.time()
        irc_result = run_irc_validation(
            ts_coords=ts_coords,
            atomic_nums=z,
            predict_fn=predict_fn,
            reactant_coords=reactant_coords,
            product_coords=product_coords,
            rmsd_threshold=args.rmsd_threshold,
            max_steps=args.irc_steps,
        )
        wall = time.time() - t0

        status = "INTENDED" if irc_result.intended else (
            "HALF" if irc_result.half_intended else "UNINTENDED"
        )
        if irc_result.error:
            status = f"ERROR: {irc_result.error}"

        rmsd_r = f"{irc_result.rmsd_to_reactant:.3f}" if irc_result.rmsd_to_reactant is not None else "N/A"
        rmsd_p = f"{irc_result.rmsd_to_product:.3f}" if irc_result.rmsd_to_product is not None else "N/A"
        print(f"  {status} | RMSD->R={rmsd_r} RMSD->P={rmsd_p} | {wall:.1f}s")

        results.append({
            "sample_id": sample_id,
            "formula": formula,
            "noise_pm": args.noise_pm,
            "intended": irc_result.intended,
            "half_intended": irc_result.half_intended,
            "rmsd_reactant": irc_result.rmsd_to_reactant,
            "rmsd_product": irc_result.rmsd_to_product,
            "error": irc_result.error,
            "wall_time_s": wall,
        })

    # ---- Summary ----
    df = pd.DataFrame(results)
    out_path = os.path.join(output_dir, f"irc_validation_{args.noise_pm}pm.parquet")
    df.to_parquet(out_path)

    n_intended = df["intended"].sum()
    n_half = df["half_intended"].sum()
    n_unintended = len(df) - n_intended - n_half - df["error"].notna().sum()
    n_error = df["error"].notna().sum()

    print(f"\n{'='*60}")
    print(f"IRC VALIDATION at noise={args.noise_pm}pm ({len(df)} samples)")
    print(f"  Intended:     {n_intended}")
    print(f"  Half:         {n_half}")
    print(f"  Unintended:   {n_unintended}")
    print(f"  Error:        {n_error}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
