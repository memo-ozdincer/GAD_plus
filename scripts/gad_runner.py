#!/usr/bin/env python
"""Run GAD at a SINGLE noise level. Designed for max parallelism.

Usage:
  uv run python scripts/gad_runner.py --dt 0.003 --noise 0.05 --n-samples 300 --n-steps 1000
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import uuid

import pandas as pd
import torch
from ase import Atoms
from ase.io import write as ase_write

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from gadplus.paths import hip_checkpoint_path, scratch_dir, transition1x_h5_path


def _label_float(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def _default_label(args: argparse.Namespace) -> str:
    parts = [f"gad_dt{_label_float(args.dt)}"]
    if not args.use_projection:
        parts.append("no_projection")
    if args.use_adaptive_dt:
        parts.append(f"adaptive_{_label_float(args.dt_min)}_{_label_float(args.dt_max)}")
    if args.use_preconditioning:
        parts.append(f"precond{_label_float(args.eig_floor)}")
    if args.blend_sharpness > 0:
        parts.append(f"blend{_label_float(args.blend_sharpness)}")
    if args.descent_until_nneg > 0:
        parts.append(f"descent_until_nneg{args.descent_until_nneg}")
    if args.multimode:
        parts.append(f"multimode_{args.multimode}")
    parts.append(f"{args.force_criterion}{_label_float(args.force_threshold)}")
    return "_".join(parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dt", type=float, required=True, help="Base GAD time step")
    parser.add_argument(
        "--label",
        type=str,
        default=None,
        help="Optional label for output filenames and trajectory metadata",
    )
    parser.add_argument("--k-track", type=int, default=0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument(
        "--use-projection",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use Eckart-projected GAD dynamics",
    )
    parser.add_argument(
        "--adaptive-dt",
        dest="use_adaptive_dt",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable adaptive time stepping",
    )
    parser.add_argument("--dt-min", type=float, default=1e-4)
    parser.add_argument("--dt-max", type=float, default=0.05)
    parser.add_argument("--dt-adaptation", type=str, default="eigenvalue_clamped")
    parser.add_argument("--max-atom-disp", type=float, default=0.35)
    parser.add_argument("--min-interatomic-dist", type=float, default=0.4)
    parser.add_argument(
        "--preconditioned",
        dest="use_preconditioning",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use Hessian absolute-value preconditioning",
    )
    parser.add_argument("--eig-floor", type=float, default=0.01)
    parser.add_argument("--blend-sharpness", type=float, default=0.0)
    parser.add_argument("--descent-until-nneg", type=int, default=0)
    parser.add_argument("--multimode", type=str, default="", choices=["", "all_neg", "smooth", "top2"])
    parser.add_argument("--multimode-sharpness", type=float, default=50.0)
    parser.add_argument(
        "--purify-hessian",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Purify the projected vibrational Hessian before eigendecomposition",
    )
    parser.add_argument("--noise", type=float, default=0.0, help="Gaussian noise std (Angstrom)")
    parser.add_argument("--n-samples", type=int, default=287)
    parser.add_argument("--n-steps", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument(
        "--random-offset",
        type=int,
        default=0,
        help="Skip first N samples (for randomized sampling from full dataset)",
    )
    parser.add_argument(
        "--force-threshold",
        type=float,
        default=0.01,
        help="Convergence threshold for selected force criterion",
    )
    parser.add_argument(
        "--force-criterion",
        type=str,
        default="fmax",
        choices=["fmax", "force_norm"],
        help="Force criterion for convergence gating",
    )
    parser.add_argument(
        "--save-ts-xyz",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write converged TS geometries to a multi-frame XYZ file",
    )
    parser.add_argument(
        "--start-from",
        type=str,
        default="geodesic_mid",
        choices=["ts_noised", "reactant", "product", "midpoint", "geodesic_mid"],
        help=(
            "Initial geometry: noised TS (default), reactant, product, "
            "linear midpoint, or reactant-product geodesic midpoint."
        ),
    )
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    noise_pm = int(round(args.noise * 1000))
    method_label = args.label or _default_label(args)
    print(
        f"Device: {device} | label={method_label} | noise={noise_pm}pm | "
        f"samples={args.n_samples} | steps={args.n_steps} | dt={args.dt} "
        f"| conv={args.force_criterion}<{args.force_threshold}"
    )

    # ---- Paths ----
    try:
        ckpt_path = str(hip_checkpoint_path())
        h5_path = str(transition1x_h5_path())
    except FileNotFoundError as exc:
        sys.exit(str(exc))

    output_dir = args.output_dir or str(scratch_dir() / "runs" / "method_cmp_300")
    os.makedirs(output_dir, exist_ok=True)

    # ---- Load HIP ----
    from gadplus.calculator.hip import load_hip_calculator, make_hip_predict_fn

    calculator = load_hip_calculator(ckpt_path, device=device)
    predict_fn = make_hip_predict_fn(calculator)
    print("HIP loaded")

    # ---- Load dataset ----
    from gadplus.data.transition1x import Transition1xDataset, UsePos

    dataset = Transition1xDataset(
        h5_path,
        split=args.split,
        max_samples=args.n_samples + args.random_offset,
        transform=UsePos("pos_transition"),
    )
    print(f"Loaded {len(dataset)} samples (split={args.split})")

    # ---- Build config ----
    from gadplus.logging.trajectory import TrajectoryLogger

    from gadplus.search.gad_search import GADSearchConfig, run_gad_search

    cfg = GADSearchConfig(
        n_steps=args.n_steps,
        dt=args.dt,
        k_track=args.k_track,
        beta=args.beta,
        use_projection=args.use_projection,
        use_adaptive_dt=args.use_adaptive_dt,
        dt_min=args.dt_min,
        dt_max=args.dt_max,
        dt_adaptation=args.dt_adaptation,
        max_atom_disp=args.max_atom_disp,
        min_interatomic_dist=args.min_interatomic_dist,
        force_threshold=args.force_threshold,
        force_criterion=args.force_criterion,
        purify_hessian=args.purify_hessian,
        use_preconditioning=args.use_preconditioning,
        eig_floor=args.eig_floor,
        blend_sharpness=args.blend_sharpness,
        descent_until_nneg=args.descent_until_nneg,
        multimode=args.multimode,
        multimode_sharpness=args.multimode_sharpness,
    )

    # ---- Sample range (supports random offset into full dataset) ----
    offset = args.random_offset
    sample_indices = list(range(offset, len(dataset)))
    print(f"Sample range: [{offset}, {len(dataset)}) = {len(sample_indices)} samples")

    # ---- Pre-generate noise ----
    torch.manual_seed(args.seed)
    noise_vecs = {}
    for i in sample_indices:
        sample = dataset[i]
        noise_vecs[i] = torch.randn_like(sample.pos) * args.noise

    # ---- Run ----
    run_id = f"{method_label}_{noise_pm}pm_{uuid.uuid4().hex[:8]}"
    results = []
    ts_atoms: list[Atoms] = []
    ts_index_rows: list[dict] = []
    t_total = time.time()
    ts_xyz_path = os.path.join(output_dir, f"ts_all_{method_label}_{noise_pm}pm.xyz")
    ts_index_path = os.path.join(output_dir, f"ts_index_{method_label}_{noise_pm}pm.parquet")

    for i in sample_indices:
        sample = dataset[i]
        coords_ts = sample.pos.to(device)
        z = sample.z.to(device)
        formula = getattr(sample, "formula", f"sample_{i}")

        if args.start_from == "ts_noised":
            coords_start = coords_ts + noise_vecs[i].to(device)
            start_method_str = f"noised_ts_{noise_pm}pm"
        elif args.start_from == "reactant":
            if not hasattr(sample, "pos_reactant"):
                print(f"  [{i:3d}] {formula:>12s} | SKIP: no pos_reactant on sample")
                continue
            coords_start = sample.pos_reactant.to(device)
            start_method_str = "reactant"
        elif args.start_from == "product":
            if not hasattr(sample, "pos_product"):
                print(f"  [{i:3d}] {formula:>12s} | SKIP: no pos_product on sample")
                continue
            pos_p = sample.pos_product.to(device)
            if pos_p.abs().sum() < 1e-6:
                print(f"  [{i:3d}] {formula:>12s} | SKIP: pos_product is all zeros")
                continue
            coords_start = pos_p
            start_method_str = "product"
        elif args.start_from == "midpoint":
            if not hasattr(sample, "pos_reactant") or not hasattr(sample, "pos_product"):
                print(f"  [{i:3d}] {formula:>12s} | SKIP: midpoint needs reactant+product")
                continue
            pos_r = sample.pos_reactant.to(device)
            pos_p = sample.pos_product.to(device)
            if pos_p.abs().sum() < 1e-6:
                print(f"  [{i:3d}] {formula:>12s} | SKIP: pos_product missing")
                continue
            coords_start = 0.5 * (pos_r + pos_p)
            start_method_str = "midpoint"
        elif args.start_from == "geodesic_mid":
            if not hasattr(sample, "pos_reactant") or not hasattr(sample, "pos_product"):
                print(f"  [{i:3d}] {formula:>12s} | SKIP: geodesic_mid needs reactant+product")
                continue
            pos_r = sample.pos_reactant.to(device)
            pos_p = sample.pos_product.to(device)
            if pos_p.abs().sum() < 1e-6:
                print(f"  [{i:3d}] {formula:>12s} | SKIP: pos_product missing")
                continue
            from gadplus.geometry.interpolation import geodesic_interpolation
            from gadplus.projection import atomic_nums_to_symbols

            coords_start = geodesic_interpolation(
                pos_r, pos_p, n_images=3, atoms=atomic_nums_to_symbols(z)
            )[1]
            start_method_str = "geodesic_mid"

        logger = TrajectoryLogger(
            output_dir=output_dir,
            run_id=run_id,
            sample_id=i,
            start_method=start_method_str,
            search_method=method_label,
            formula=formula,
        )

        t0 = time.time()
        result = run_gad_search(
            predict_fn, coords_start, z, cfg, logger=logger, known_ts_coords=coords_ts
        )
        wall = time.time() - t0
        logger.flush()

        status = "CONV" if result.converged else "FAIL"
        print(
            f"  [{i:3d}] {formula:>12s} | {status} | n_neg={result.final_n_neg} "
            f"| force_norm={result.final_force_norm:.4f} "
            f"| force_max={result.final_force_max:.4f} "
            f"| steps={result.total_steps:3d} | {wall:.1f}s"
        )

        final_coords_flat = result.final_coords.reshape(-1).float().tolist()
        ts_xyz_frame = None
        if result.converged and args.save_ts_xyz:
            frame_idx = len(ts_atoms)
            ts_xyz_frame = frame_idx
            atoms = Atoms(
                numbers=z.detach().cpu().numpy(),
                positions=result.final_coords.detach().cpu().numpy(),
            )
            atoms.info["sample_id"] = int(i)
            atoms.info["formula"] = str(formula)
            atoms.info["method"] = method_label
            atoms.info["noise_pm"] = int(noise_pm)
            ts_atoms.append(atoms)
            ts_index_rows.append(
                {
                    "run_id": run_id,
                    "method": method_label,
                    "noise_pm": noise_pm,
                    "sample_id": i,
                    "formula": formula,
                    "frame_index": frame_idx,
                    "converged_step": result.converged_step,
                    "final_force_norm": result.final_force_norm,
                    "final_force_max": result.final_force_max,
                }
            )

        results.append(
            {
                "run_id": run_id,
                "method": method_label,
                "dt": args.dt,
                "noise_pm": noise_pm,
                "sample_id": i,
                "formula": formula,
                "converged": result.converged,
                "converged_step": result.converged_step,
                "total_steps": result.total_steps,
                "final_n_neg": result.final_n_neg,
                "final_force_norm": result.final_force_norm,
                "final_force_max": result.final_force_max,
                "final_energy": result.final_energy,
                "final_eig0": result.final_eig0,
                "final_coords_flat": final_coords_flat,
                "ts_xyz_path": ts_xyz_path if result.converged and args.save_ts_xyz else None,
                "ts_xyz_frame": ts_xyz_frame,
                "wall_time_s": wall,
            }
        )

    total_wall = time.time() - t_total

    # ---- Save ----
    df = pd.DataFrame(results)
    out_path = os.path.join(output_dir, f"summary_{method_label}_{noise_pm}pm.parquet")
    df.to_parquet(out_path)

    if args.save_ts_xyz and ts_atoms:
        ase_write(ts_xyz_path, ts_atoms)
        pd.DataFrame(ts_index_rows).to_parquet(ts_index_path, index=False)
        print(f"Saved TS XYZ: {ts_xyz_path} ({len(ts_atoms)} frames)")
        print(f"Saved TS index: {ts_index_path}")
    elif args.save_ts_xyz:
        print("No converged structures; TS XYZ not written.")

    n_conv = df["converged"].sum()
    rate = 100 * n_conv / len(df)
    avg_steps = df.loc[df["converged"], "converged_step"].mean()
    print(f"\n{'=' * 60}")
    print(
        f"{method_label} @ {noise_pm}pm: {n_conv}/{len(df)} ({rate:.1f}%), "
        f"avg steps={avg_steps:.0f}, wall={total_wall:.0f}s ({total_wall / 60:.1f}min)"
    )
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
