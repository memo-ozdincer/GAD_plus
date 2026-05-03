#!/usr/bin/env python
"""Phase 5: IRC validation on converged TS from Phase 2 noise survey.

Trust the noise-survey's converged label. For each converged sample, take
the geometry at the exact `converged_step` recorded in the summary parquet
and run Sella IRC forward/backward to check whether it connects the
intended reactant/product. No recomputation, no refinement, no gating.

Usage:
  python scripts/irc_validate.py --noise-pm 10 --max-validate 10
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from ase import Atoms

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from visualize_3d import _write_viewer_bundle


def _coords_flat(coords: torch.Tensor | np.ndarray | None) -> list[float] | None:
    if coords is None:
        return None
    if isinstance(coords, torch.Tensor):
        arr = coords.detach().cpu().numpy()
    else:
        arr = np.asarray(coords, dtype=float)
    return arr.reshape(-1).astype(float).tolist()


def _write_irc_viewer_bundle(
    base_dir: str,
    run_id: str,
    sample_id: int,
    formula: str,
    atomic_nums: torch.Tensor,
    reactant_coords: torch.Tensor | None,
    ts_coords: torch.Tensor,
    reverse_coords: np.ndarray | None,
    forward_coords: np.ndarray | None,
    product_coords: torch.Tensor | None,
) -> tuple[str, str, str] | tuple[None, None, None]:
    numbers = atomic_nums.detach().cpu().numpy().astype(int)
    frames = []

    def add_frame(label: str, coords: torch.Tensor | np.ndarray | None) -> None:
        if coords is None:
            return
        if isinstance(coords, torch.Tensor):
            arr = coords.detach().cpu().numpy().reshape(-1, 3)
        else:
            arr = np.asarray(coords, dtype=float).reshape(-1, 3)
        atoms = Atoms(numbers=numbers.tolist(), positions=arr)
        atoms.info["comment"] = label
        frames.append(atoms)

    add_frame("reactant_ref", reactant_coords)
    add_frame("irc_reverse_endpoint", reverse_coords)
    add_frame("ts_input", ts_coords)
    add_frame("irc_forward_endpoint", forward_coords)
    add_frame("product_ref", product_coords)

    if not frames:
        return None, None, None
    return _write_viewer_bundle(base_dir, run_id, sample_id, formula, frames)


def _load_ts_at_converged_step(
    survey_dir: str,
    method: str,
    noise_pm: int,
    sample_id: int,
    converged_step: int,
) -> pd.DataFrame:
    """Fetch the trajectory row at the exact converged_step. Narrow filename glob."""
    import duckdb

    # traj files are named traj_<method>_<noise>pm_<hash>_<sample_id>.parquet
    pattern = f"{survey_dir}/traj_{method}_{noise_pm}pm_*_{sample_id}.parquet"
    query = f"""
        SELECT step, coords_flat
        FROM '{pattern}'
        WHERE step = {converged_step}
        LIMIT 1
    """
    return duckdb.execute(query).df()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--noise-pm", type=int, default=10, help="Noise level (pm) to validate")
    parser.add_argument("--max-validate", type=int, default=10, help="Max converged TS to validate")
    parser.add_argument("--irc-steps", type=int, default=500, help="Max IRC steps per direction")
    parser.add_argument("--rmsd-threshold", type=float, default=0.3, help="RMSD threshold for matching (A)")
    parser.add_argument("--survey-dir", type=str, default=None,
                        help="Directory with noise survey results")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--n-dataset-samples", type=int, default=300,
                        help="Number of dataset samples to load (must cover sample_ids in survey)")
    parser.add_argument("--split", type=str, default="train", choices=["train", "test", "val"],
                        help="Transition1x split for reactant/product references. "
                             "MUST match the split that produced the survey runs.")
    parser.add_argument(
        "--method",
        type=str,
        default="sella_baseline",
        choices=["sella_baseline", "sella_hip", "rigorous"],
        help="IRC integrator: sella_baseline (vanilla Sella + BFGS Hessian), "
             "sella_hip (Sella IRC + HIP MW+Eckart Hessian every step), "
             "rigorous (HIP predictor-corrector with K-step hold).",
    )
    parser.add_argument(
        "--source-method",
        type=str,
        default="gad_dt003",
        help="GAD method in the summary parquets to pull converged TS from.",
    )
    parser.add_argument(
        "--all-endpoints",
        action="store_true",
        default=False,
        help="Ignore the converged filter; run IRC on every sample's final "
             "trajectory coords (last step). IRC becomes the convergence criterion.",
    )
    parser.add_argument(
        "--coords-source",
        type=str,
        default="traj",
        choices=["traj", "summary"],
        help="Where to read TS coords from. 'traj' (default) = narrow glob on "
             "traj_<method>_<noise>pm_*_<sid>.parquet at step=converged_step. "
             "'summary' = read coords_flat column directly from the summary "
             "parquet (used for Sella-found TSs that have coords logged).",
    )
    parser.add_argument(
        "--write-viewer-bundles",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write viewer bundles for IRC endpoint inspection",
    )
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
    viewer_dir = os.path.join(output_dir, f"viewer_noise_{args.noise_pm}pm")
    if args.write_viewer_bundles:
        os.makedirs(viewer_dir, exist_ok=True)

    # ---- Load HIP ----
    from gadplus.calculator.hip import load_hip_calculator, make_hip_predict_fn
    calculator = load_hip_calculator(ckpt_path, device=device)
    predict_fn = make_hip_predict_fn(calculator)
    print("HIP loaded")

    # ---- Find converged TS from noise survey ----
    import duckdb

    # Narrow single-file read avoids O(N_files) Lustre metadata stalls on glob.
    summary_path = f"{survey_dir}/summary_{args.source_method}_{args.noise_pm}pm.parquet"
    extra_cols = ", coords_flat" if args.coords_source == "summary" else ""
    if args.all_endpoints:
        # IRC on every sample's final trajectory coords. No converged filter;
        # fetch at step = total_steps - 1 for each sample.
        converged_df = duckdb.execute(f"""
            SELECT sample_id, method,
                   CAST(total_steps - 1 AS DOUBLE) AS converged_step,
                   final_force_norm, final_n_neg, formula, converged {extra_cols}
            FROM '{summary_path}'
            ORDER BY sample_id ASC
            LIMIT {args.max_validate}
        """).df()
    else:
        converged_df = duckdb.execute(f"""
            SELECT sample_id, method, converged_step, final_force_norm,
                   final_n_neg, formula, true AS converged {extra_cols}
            FROM '{summary_path}'
            WHERE converged = true
            ORDER BY converged_step ASC
            LIMIT {args.max_validate}
        """).df()

    print(f"Found {len(converged_df)} converged TS at noise={args.noise_pm}pm (source_method={args.source_method})")
    if len(converged_df) == 0:
        print("No converged TS to validate.")
        return

    # ---- Load dataset to get reference geometries ----
    from gadplus.data.transition1x import Transition1xDataset, UsePos
    dataset = Transition1xDataset(
        h5_path, split=args.split, max_samples=args.n_dataset_samples,
        transform=UsePos("pos_transition"),
    )
    print(f"Dataset: split={args.split}, loaded {len(dataset)} samples")

    # ---- Dispatch IRC integrator based on --method ----
    from gadplus.search.irc_validate import run_irc_validation
    from gadplus.search.irc_sella_hip import run_irc_sella_hip
    from gadplus.search.irc_rigorous import run_irc_rigorous

    def _run_irc(**kwargs):
        if args.method == "sella_baseline":
            return run_irc_validation(**kwargs)
        if args.method == "sella_hip":
            return run_irc_sella_hip(**kwargs)
        if args.method == "rigorous":
            return run_irc_rigorous(**kwargs)
        raise ValueError(f"unknown --method: {args.method}")

    print(f"Method: {args.method} | max_steps={args.irc_steps}")

    results = []
    for _, row in converged_df.iterrows():
        sample_id = int(row["sample_id"])
        source_method = row["method"]
        run_id = f"{source_method}_{args.noise_pm}pm_s{sample_id}"  # synthetic id
        formula = row["formula"]
        conv_step = int(row["converged_step"])

        print(f"\n--- Validating sample {sample_id} ({formula}), converged_step={conv_step} ---")

        try:
            if args.coords_source == "summary":
                # Coords are embedded in the summary row itself.
                traj_df = pd.DataFrame([{"step": conv_step, "coords_flat": row["coords_flat"]}])
            else:
                traj_df = _load_ts_at_converged_step(
                    survey_dir=survey_dir,
                    method=source_method,
                    noise_pm=args.noise_pm,
                    sample_id=sample_id,
                    converged_step=conv_step,
                )
        except Exception as e:
            print(f"  Error reading trajectory: {e}")
            results.append({
                "method": args.method,
                "run_id": run_id,
                "sample_id": sample_id, "formula": formula,
                "noise_pm": args.noise_pm,
                "converged_step": conv_step,
                "intended": False, "half_intended": False,
                "topology_intended": False, "topology_half_intended": False,
                "rmsd_reactant": None, "rmsd_product": None,
                "forward_graph_matches_reactant": False,
                "forward_graph_matches_product": False,
                "reverse_graph_matches_reactant": False,
                "reverse_graph_matches_product": False,
                "error": str(e),
                "topology_error": str(e),
            })
            continue

        if len(traj_df) == 0:
            print(f"  No trajectory row at step {conv_step}")
            results.append({
                "method": args.method,
                "run_id": run_id,
                "sample_id": sample_id, "formula": formula,
                "noise_pm": args.noise_pm,
                "converged_step": conv_step,
                "intended": False, "half_intended": False,
                "topology_intended": False, "topology_half_intended": False,
                "rmsd_reactant": None, "rmsd_product": None,
                "forward_graph_matches_reactant": False,
                "forward_graph_matches_product": False,
                "reverse_graph_matches_reactant": False,
                "reverse_graph_matches_product": False,
                "error": "no trajectory row at converged_step",
                "topology_error": "no trajectory row at converged_step",
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
        irc_result = _run_irc(
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
        topology_status = "INTENDED" if irc_result.topology_intended else (
            "HALF" if irc_result.topology_half_intended else "UNINTENDED"
        )
        if irc_result.error:
            status = f"ERROR: {irc_result.error}"

        rmsd_r = f"{irc_result.rmsd_to_reactant:.3f}" if irc_result.rmsd_to_reactant is not None else "N/A"
        rmsd_p = f"{irc_result.rmsd_to_product:.3f}" if irc_result.rmsd_to_product is not None else "N/A"
        print(
            f"  RMSD={status} | TOPO={topology_status} "
            f"| RMSD->R={rmsd_r} RMSD->P={rmsd_p} | {wall:.1f}s"
        )
        if irc_result.topology_error:
            print(f"  Topology warning: {irc_result.topology_error}")

        bundle_dir = None
        multi_xyz = None
        sequence_dir = None
        if args.write_viewer_bundles:
            bundle_dir, multi_xyz, sequence_dir = _write_irc_viewer_bundle(
                base_dir=viewer_dir,
                run_id=run_id,
                sample_id=sample_id,
                formula=formula,
                atomic_nums=z,
                reactant_coords=reactant_coords,
                ts_coords=ts_coords,
                reverse_coords=irc_result.reverse_coords,
                forward_coords=irc_result.forward_coords,
                product_coords=product_coords,
            )

        results.append({
            "method": args.method,
            "run_id": run_id,
            "sample_id": sample_id,
            "formula": formula,
            "noise_pm": args.noise_pm,
            "source_gad_converged": bool(row.get("converged", True)),
            "atomic_nums": z.detach().cpu().numpy().astype(int).tolist(),
            "converged_step": conv_step,
            "intended": irc_result.intended,
            "half_intended": irc_result.half_intended,
            "topology_intended": irc_result.topology_intended,
            "topology_half_intended": irc_result.topology_half_intended,
            "rmsd_reactant": irc_result.rmsd_to_reactant,
            "rmsd_product": irc_result.rmsd_to_product,
            "forward_rmsd_reactant": irc_result.forward_rmsd_to_reactant,
            "forward_rmsd_product": irc_result.forward_rmsd_to_product,
            "reverse_rmsd_reactant": irc_result.reverse_rmsd_to_reactant,
            "reverse_rmsd_product": irc_result.reverse_rmsd_to_product,
            "forward_graph_matches_reactant": irc_result.forward_graph_matches_reactant,
            "forward_graph_matches_product": irc_result.forward_graph_matches_product,
            "reverse_graph_matches_reactant": irc_result.reverse_graph_matches_reactant,
            "reverse_graph_matches_product": irc_result.reverse_graph_matches_product,
            "forward_n_neg_vib": irc_result.forward_n_neg_vib,
            "reverse_n_neg_vib": irc_result.reverse_n_neg_vib,
            "forward_min_vib_eig": irc_result.forward_min_vib_eig,
            "reverse_min_vib_eig": irc_result.reverse_min_vib_eig,
            "error": irc_result.error,
            "topology_error": irc_result.topology_error,
            "ts_coords_flat": _coords_flat(ts_coords),
            "reactant_coords_flat": _coords_flat(reactant_coords),
            "product_coords_flat": _coords_flat(product_coords),
            "forward_coords_flat": _coords_flat(irc_result.forward_coords),
            "reverse_coords_flat": _coords_flat(irc_result.reverse_coords),
            "viewer_bundle_dir": bundle_dir,
            "viewer_multi_xyz": multi_xyz,
            "viewer_sequence_dir": sequence_dir,
            "wall_time_s": wall,
        })

    # ---- Summary ----
    df = pd.DataFrame(results)
    suffix = "_allendpoints" if args.all_endpoints else ""
    out_path = os.path.join(output_dir, f"irc_validation_{args.method}{suffix}_{args.noise_pm}pm.parquet")
    df.to_parquet(out_path)

    n_intended = df["intended"].sum()
    n_half = df["half_intended"].sum()
    n_topology_intended = df["topology_intended"].sum()
    n_topology_half = df["topology_half_intended"].sum()
    n_error = df["error"].notna().sum()
    n_unintended = len(df) - n_intended - n_half - n_error

    print(f"\n{'='*60}")
    print(f"IRC VALIDATION at noise={args.noise_pm}pm ({len(df)} samples)")
    print(f"  Intended:     {n_intended}")
    print(f"  Half:         {n_half}")
    print(f"  Topo intended:{n_topology_intended}")
    print(f"  Topo half:    {n_topology_half}")
    print(f"  Unintended:   {n_unintended}")
    print(f"  Error:        {n_error}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
