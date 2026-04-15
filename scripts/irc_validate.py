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


def _ts_metric_column(criterion: str) -> str:
    if criterion == "fmax":
        return "force_max"
    return "force_norm"


def _load_ts_candidate(
    survey_dir: str,
    run_id: str,
    sample_id: int,
    pick_mode: str,
    criterion: str,
) -> tuple[pd.DataFrame, str]:
    import duckdb

    metric_col = _ts_metric_column(criterion)
    available_cols = duckdb.execute(
        f"DESCRIBE SELECT * FROM '{survey_dir}/traj_*.parquet'"
    ).df()["column_name"].tolist()
    select_cols = ["step", "coords_flat", "n_neg"]
    if "force_norm" in available_cols:
        select_cols.append("force_norm")
    if "force_max" in available_cols:
        select_cols.append("force_max")
    if metric_col not in available_cols:
        metric_col = "force_norm"
        criterion = "force_norm"
    select_sql = ", ".join(select_cols)

    if pick_mode == "final":
        query = f"""
            SELECT {select_sql}
            FROM '{survey_dir}/traj_*.parquet'
            WHERE run_id = '{run_id}' AND sample_id = {sample_id}
            ORDER BY step DESC
            LIMIT 1
        """
        return duckdb.execute(query).df(), criterion

    if pick_mode == "best_nneg1":
        query = f"""
            SELECT {select_sql}
            FROM '{survey_dir}/traj_*.parquet'
            WHERE run_id = '{run_id}' AND sample_id = {sample_id} AND n_neg = 1
            ORDER BY {metric_col} ASC, step ASC
            LIMIT 1
        """
        df = duckdb.execute(query).df()
        if len(df) > 0:
            return df, criterion

    query = f"""
        SELECT {select_sql}
        FROM '{survey_dir}/traj_*.parquet'
        WHERE run_id = '{run_id}' AND sample_id = {sample_id}
        ORDER BY {metric_col} ASC, step ASC
        LIMIT 1
    """
    return duckdb.execute(query).df(), criterion


def _recompute_ts_metrics(
    predict_fn,
    coords: torch.Tensor,
    atomic_nums: torch.Tensor,
    atomsymbols: list[str],
) -> tuple[float, float, int]:
    from gadplus.core.convergence import force_max, force_mean
    from gadplus.projection import vib_eig

    out = predict_fn(coords, atomic_nums, do_hessian=True, require_grad=False)
    forces = out["forces"]
    hessian = out["hessian"]
    if forces.dim() == 3 and forces.shape[0] == 1:
        forces = forces[0]
    forces = forces.reshape(-1, 3)
    evals_vib, _evecs_vib, _Q = vib_eig(hessian, coords, atomsymbols, purify=False)
    return force_mean(forces), force_max(forces), int((evals_vib < 0).sum().item())


def _refine_ts_candidate(
    predict_fn,
    coords: torch.Tensor,
    atomic_nums: torch.Tensor,
    *,
    n_steps: int,
    dt: float,
    force_threshold: float,
    force_criterion: str,
):
    from gadplus.search.gad_search import GADSearchConfig, run_gad_search

    cfg = GADSearchConfig(
        n_steps=n_steps,
        dt=dt,
        k_track=0,
        use_projection=True,
        use_adaptive_dt=False,
        max_atom_disp=0.35,
        force_threshold=force_threshold,
        force_criterion=force_criterion,
        use_preconditioning=False,
        descent_until_nneg=0,
    )
    return run_gad_search(
        predict_fn=predict_fn,
        coords0=coords,
        atomic_nums=atomic_nums,
        cfg=cfg,
        logger=None,
        known_ts_coords=None,
    )


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
    parser.add_argument(
        "--ts-pick",
        type=str,
        default="best_nneg1",
        choices=["best_nneg1", "final", "best_force"],
        help="How to choose the TS geometry from the saved trajectory before IRC",
    )
    parser.add_argument(
        "--ts-force-criterion",
        type=str,
        default="fmax",
        choices=["force_norm", "fmax"],
        help="Force metric for TS screening and candidate selection",
    )
    parser.add_argument(
        "--ts-force-threshold",
        type=float,
        default=0.01,
        help="TS quality threshold applied before IRC",
    )
    parser.add_argument(
        "--skip-if-ts-poor",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip IRC when the selected TS candidate fails the pre-IRC quality gate",
    )
    parser.add_argument(
        "--refine-ts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run a short projected GAD refinement from the selected TS candidate before IRC",
    )
    parser.add_argument("--refine-steps", type=int, default=300)
    parser.add_argument("--refine-dt", type=float, default=0.003)
    parser.add_argument(
        "--refine-force-criterion",
        type=str,
        default="fmax",
        choices=["force_norm", "fmax"],
    )
    parser.add_argument("--refine-force-threshold", type=float, default=0.005)
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
    from gadplus.projection import atomic_nums_to_symbols

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

        # Get TS candidate coords from trajectory
        try:
            traj_df, criterion_used = _load_ts_candidate(
                survey_dir=survey_dir,
                run_id=run_id,
                sample_id=sample_id,
                pick_mode=args.ts_pick,
                criterion=args.ts_force_criterion,
            )
        except Exception as e:
            print(f"  Error reading trajectory: {e}")
            results.append({
                "run_id": run_id,
                "sample_id": sample_id, "formula": formula,
                "noise_pm": args.noise_pm,
                "intended": False, "half_intended": False,
                "topology_intended": False, "topology_half_intended": False,
                "rmsd_reactant": None, "rmsd_product": None,
                "forward_graph_matches_reactant": False,
                "forward_graph_matches_product": False,
                "reverse_graph_matches_reactant": False,
                "reverse_graph_matches_product": False,
                "ts_pick_mode": args.ts_pick,
                "ts_force_criterion": args.ts_force_criterion,
                "error": str(e),
                "topology_error": str(e),
            })
            continue

        if len(traj_df) == 0:
            print(f"  No trajectory data found")
            results.append({
                "run_id": run_id,
                "sample_id": sample_id, "formula": formula,
                "noise_pm": args.noise_pm,
                "intended": False, "half_intended": False,
                "topology_intended": False, "topology_half_intended": False,
                "rmsd_reactant": None, "rmsd_product": None,
                "error": "no trajectory",
                "forward_graph_matches_reactant": False,
                "forward_graph_matches_product": False,
                "reverse_graph_matches_reactant": False,
                "reverse_graph_matches_product": False,
                "ts_pick_mode": args.ts_pick,
                "ts_force_criterion": args.ts_force_criterion,
                "topology_error": "no trajectory",
            })
            continue

        coords_flat = traj_df["coords_flat"].iloc[0]
        candidate_step = int(traj_df["step"].iloc[0])
        candidate_nneg_logged = int(traj_df["n_neg"].iloc[0]) if "n_neg" in traj_df.columns else None
        n_atoms = len(coords_flat) // 3
        ts_coords = torch.tensor(coords_flat, dtype=torch.float32, device=device).reshape(n_atoms, 3)

        # Get reference geometries
        sample = dataset[sample_id]
        z = sample.z.to(device)
        atomsymbols = atomic_nums_to_symbols(z)
        reactant_coords = sample.pos_reactant.to(device) if hasattr(sample, "pos_reactant") else None
        product_coords = None
        if hasattr(sample, "pos_product"):
            pp = sample.pos_product.to(device)
            if pp.abs().sum() > 1e-6:
                product_coords = pp

        ts_force_norm_recomputed, ts_force_max_recomputed, ts_n_neg_recomputed = _recompute_ts_metrics(
            predict_fn=predict_fn,
            coords=ts_coords,
            atomic_nums=z,
            atomsymbols=atomsymbols,
        )
        ts_force_value = (
            ts_force_max_recomputed if args.ts_force_criterion == "fmax"
            else ts_force_norm_recomputed
        )
        ts_quality_ok = (ts_n_neg_recomputed == 1 and ts_force_value < args.ts_force_threshold)
        print(
            f"  TS candidate: step={candidate_step} pick={args.ts_pick} "
            f"| n_neg(logged)={candidate_nneg_logged} n_neg(recomputed)={ts_n_neg_recomputed} "
            f"| force_norm={ts_force_norm_recomputed:.5f} "
            f"| fmax={ts_force_max_recomputed:.5f} "
            f"| gate[{args.ts_force_criterion}<{args.ts_force_threshold:.4g}]={ts_quality_ok}"
        )

        refined_ts_coords = ts_coords
        refine_result = None
        refined_force_norm = ts_force_norm_recomputed
        refined_force_max = ts_force_max_recomputed
        refined_n_neg = ts_n_neg_recomputed
        refined_quality_ok = ts_quality_ok

        if args.refine_ts and not ts_quality_ok:
            refine_t0 = time.time()
            refine_result = _refine_ts_candidate(
                predict_fn=predict_fn,
                coords=ts_coords,
                atomic_nums=z,
                n_steps=args.refine_steps,
                dt=args.refine_dt,
                force_threshold=args.refine_force_threshold,
                force_criterion=args.refine_force_criterion,
            )
            refined_ts_coords = refine_result.final_coords.to(device)
            refined_force_norm, refined_force_max, refined_n_neg = _recompute_ts_metrics(
                predict_fn=predict_fn,
                coords=refined_ts_coords,
                atomic_nums=z,
                atomsymbols=atomsymbols,
            )
            refined_force_value = (
                refined_force_max if args.refine_force_criterion == "fmax"
                else refined_force_norm
            )
            refined_quality_ok = (
                refined_n_neg == 1 and refined_force_value < args.refine_force_threshold
            )
            print(
                f"  Refinement: converged={refine_result.converged} "
                f"| steps={refine_result.total_steps} "
                f"| n_neg={refined_n_neg} "
                f"| force_norm={refined_force_norm:.5f} "
                f"| fmax={refined_force_max:.5f} "
                f"| gate[{args.refine_force_criterion}<{args.refine_force_threshold:.4g}]={refined_quality_ok} "
                f"| {time.time() - refine_t0:.1f}s"
            )

        if args.skip_if_ts_poor and not refined_quality_ok:
            results.append({
                "run_id": run_id,
                "sample_id": sample_id,
                "formula": formula,
                "noise_pm": args.noise_pm,
                "atomic_nums": z.detach().cpu().numpy().astype(int).tolist(),
                "ts_pick_mode": args.ts_pick,
                "ts_force_criterion": criterion_used,
                "ts_force_threshold": args.ts_force_threshold,
                "candidate_step": candidate_step,
                "candidate_n_neg_logged": candidate_nneg_logged,
                "ts_force_norm_recomputed": ts_force_norm_recomputed,
                "ts_force_max_recomputed": ts_force_max_recomputed,
                "ts_n_neg_recomputed": ts_n_neg_recomputed,
                "ts_quality_ok": ts_quality_ok,
                "refine_ts": args.refine_ts,
                "refine_steps": args.refine_steps,
                "refine_dt": args.refine_dt,
                "refine_force_criterion": args.refine_force_criterion,
                "refine_force_threshold": args.refine_force_threshold,
                "refine_converged": None if refine_result is None else refine_result.converged,
                "refine_total_steps": None if refine_result is None else refine_result.total_steps,
                "refined_force_norm": refined_force_norm,
                "refined_force_max": refined_force_max,
                "refined_n_neg": refined_n_neg,
                "refined_quality_ok": refined_quality_ok,
                "intended": False,
                "half_intended": False,
                "topology_intended": False,
                "topology_half_intended": False,
                "rmsd_reactant": None,
                "rmsd_product": None,
                "forward_graph_matches_reactant": False,
                "forward_graph_matches_product": False,
                "reverse_graph_matches_reactant": False,
                "reverse_graph_matches_product": False,
                "error": "ts_quality_gate_failed",
                "topology_error": None,
                "ts_coords_flat": _coords_flat(ts_coords),
                "refined_ts_coords_flat": _coords_flat(refined_ts_coords),
                "reactant_coords_flat": _coords_flat(reactant_coords),
                "product_coords_flat": _coords_flat(product_coords),
                "forward_coords_flat": None,
                "reverse_coords_flat": None,
                "viewer_bundle_dir": None,
                "viewer_multi_xyz": None,
                "viewer_sequence_dir": None,
                "wall_time_s": 0.0,
            })
            continue

        t0 = time.time()
        irc_result = run_irc_validation(
            ts_coords=refined_ts_coords,
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
                ts_coords=refined_ts_coords,
                reverse_coords=irc_result.reverse_coords,
                forward_coords=irc_result.forward_coords,
                product_coords=product_coords,
            )

        results.append({
            "run_id": run_id,
            "sample_id": sample_id,
            "formula": formula,
            "noise_pm": args.noise_pm,
            "atomic_nums": z.detach().cpu().numpy().astype(int).tolist(),
            "ts_pick_mode": args.ts_pick,
            "ts_force_criterion": criterion_used,
            "ts_force_threshold": args.ts_force_threshold,
            "candidate_step": candidate_step,
            "candidate_n_neg_logged": candidate_nneg_logged,
            "ts_force_norm_recomputed": ts_force_norm_recomputed,
            "ts_force_max_recomputed": ts_force_max_recomputed,
            "ts_n_neg_recomputed": ts_n_neg_recomputed,
            "ts_quality_ok": ts_quality_ok,
            "refine_ts": args.refine_ts,
            "refine_steps": args.refine_steps,
            "refine_dt": args.refine_dt,
            "refine_force_criterion": args.refine_force_criterion,
            "refine_force_threshold": args.refine_force_threshold,
            "refine_converged": None if refine_result is None else refine_result.converged,
            "refine_total_steps": None if refine_result is None else refine_result.total_steps,
            "refined_force_norm": refined_force_norm,
            "refined_force_max": refined_force_max,
            "refined_n_neg": refined_n_neg,
            "refined_quality_ok": refined_quality_ok,
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
            "error": irc_result.error,
            "topology_error": irc_result.topology_error,
            "ts_coords_flat": _coords_flat(ts_coords),
            "refined_ts_coords_flat": _coords_flat(refined_ts_coords),
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
    out_path = os.path.join(output_dir, f"irc_validation_{args.noise_pm}pm.parquet")
    df.to_parquet(out_path)

    n_intended = df["intended"].sum()
    n_half = df["half_intended"].sum()
    n_topology_intended = df["topology_intended"].sum()
    n_topology_half = df["topology_half_intended"].sum()
    n_unintended = len(df) - n_intended - n_half - df["error"].notna().sum()
    n_error = df["error"].notna().sum()

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
