#!/usr/bin/env python
"""Run GAD or hybrid GAD/Newton on an analytic Lennard-Jones cluster."""
from __future__ import annotations

import argparse
import os
import sys
import time
import uuid
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from gadplus.calculator.lennard_jones import (
    LennardJonesParams,
    lj_atomic_nums,
    make_lj_predict_fn,
    pair_distances,
    pentagonal_bipyramid_geometry,
    random_cluster_geometry,
    shortest_pair_label,
)
from gadplus.core.adaptive_dt import cap_displacement, min_interatomic_distance
from gadplus.core.convergence import (
    count_negative_eigenvalues,
    force_max,
    force_mean,
    force_value_from_criterion,
)
from gadplus.projection import atomic_nums_to_symbols, vib_eig
from gadplus.search.gad_search import GADSearchConfig, run_gad_search
from gadplus.search.hybrid_gad_damped_eigfollownewton_eckart import (
    projected_hybrid_gad_newton_step as proj_step_damped,
)
from gadplus.search.hybrid_gad_eigfollownewton import hybrid_gad_newton_step_from_force
from gadplus.search.hybrid_gad_eigfollownewton_eckart import (
    masses_from_z,
    projected_hybrid_gad_newton_step as proj_step_plain,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark GAD and hybrid GAD/Newton on a Lennard-Jones cluster."
    )
    parser.add_argument(
        "--method",
        choices=["gad", "hybrid", "hybrid_eckart", "hybrid_damped_eckart"],
        default="gad",
    )
    parser.add_argument(
        "--start-from",
        choices=["minimum", "minimum_noised", "random", "expanded_minimum", "gaussian_origin"],
        default="gaussian_origin",
    )
    parser.add_argument("--n-atoms", type=int, default=7)
    parser.add_argument("--n-samples", type=int, default=24)
    parser.add_argument("--n-steps", type=int, default=1000)
    parser.add_argument("--noise", type=float, default=0.05)
    parser.add_argument(
        "--gaussian-origin-sigma",
        type=float,
        default=1.0,
        help="Per-coordinate stddev for --start-from gaussian_origin.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/lj"))
    parser.add_argument("--save-traj", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument(
        "--lj-compile",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable torch.compile for LJ force and Hessian kernels on CUDA.",
    )
    parser.add_argument(
        "--atomic-number",
        type=int,
        default=18,
        help="Element used only for equal-mass Eckart projection; default is argon.",
    )
    parser.add_argument("--force-threshold", type=float, default=1.0e-3)
    parser.add_argument(
        "--force-criterion",
        choices=["fmax", "force_norm"],
        default="fmax",
        help="Convergence force metric. Hybrid methods currently gate on fmax.",
    )
    parser.add_argument("--max-atom-disp", type=float, default=0.05)
    parser.add_argument("--min-interatomic-dist", type=float, default=0.75)

    parser.add_argument("--dt", type=float, default=1.0e-3, help="Regular GAD timestep.")
    parser.add_argument(
        "--k-track",
        type=int,
        default=0,
        help="Mode-tracking window (0 = always use lowest Eckart eigenvector).",
    )
    parser.add_argument("--use-projection", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-adaptive-dt", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--return-weighted-step-direction",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Return the legacy sqrt(M)-weighted projected direction instead of the unweighted Cartesian step",
    )
    parser.add_argument("--dt-min", type=float, default=1.0e-5)
    parser.add_argument("--dt-max", type=float, default=0.05)

    parser.add_argument("--gad-dt", type=float, default=1.0e-3, help="Hybrid GAD step size.")
    parser.add_argument("--trust-radius", type=float, default=0.01)
    parser.add_argument("--switch-force", type=float, default=1.0e-3)
    parser.add_argument("--switch-by-eig", choices=["true", "false"], default="false")
    parser.add_argument("--min-curvature", type=float, default=None)
    parser.add_argument(
        "--target-mode-strategy",
        choices=["fixed", "neg_force_coupling"],
        default="fixed",
    )
    parser.add_argument(
        "--high-index-descent",
        choices=["gad", "gradient", "index_controlled", "newton"],
        default="gad",
    )
    return parser.parse_args()


def make_starting_geometry(
    sample_id: int,
    start_from: str,
    n_atoms: int,
    sigma: float,
    generator: torch.Generator,
    noise: float,
    gaussian_origin_sigma: float = 1.0,
) -> tuple[torch.Tensor, str]:
    """Build one labelled LJ starting geometry plus optional Cartesian noise."""

    if start_from == "gaussian_origin":
        coords = gaussian_origin_sigma * torch.randn(
            (n_atoms, 3),
            generator=generator,
            dtype=torch.float64,
        )
        start_label = f"gaussian_origin_sigma{gaussian_origin_sigma:g}"
    elif n_atoms == 7 and start_from in {"minimum", "minimum_noised", "expanded_minimum"}:
        coords = pentagonal_bipyramid_geometry(sigma)
        start_label = "lj7_pentagonal_bipyramid"
    else:
        coords = random_cluster_geometry(n_atoms, sigma=sigma, generator=generator)
        start_label = "random_cluster"

    if start_from == "random":
        coords = random_cluster_geometry(n_atoms, sigma=sigma, generator=generator)
        start_label = "random_cluster"
    elif start_from == "expanded_minimum":
        coords = 1.15 * coords
        start_label = f"expanded_{start_label}"
    elif start_from == "minimum" and n_atoms != 7:
        start_label = "random_cluster_no_lj_minimum"

    if start_from in {"minimum_noised", "random"} and noise > 0:
        coords = coords + noise * torch.randn(coords.shape, generator=generator, dtype=coords.dtype)
        start_label = f"{start_label}_noise{noise:g}"

    # Rotate sample ordering deterministically enough to avoid identical rows for exact starts.
    if start_from in {"minimum", "expanded_minimum"} and sample_id:
        coords = torch.roll(coords, shifts=sample_id % n_atoms, dims=0)

    return coords - coords.mean(dim=0, keepdim=True), start_label


def n_neg_eckart(
    hessian: torch.Tensor,
    coords: torch.Tensor,
    atomic_nums: torch.Tensor,
) -> tuple[int, float, float]:
    """Eckart-projected index diagnostics."""

    evals, _, _ = vib_eig(hessian, coords, atomic_nums_to_symbols(atomic_nums), purify=False)
    evals = torch.sort(evals).values
    eig0 = float(evals[0].item()) if evals.numel() > 0 else 0.0
    eig1 = float(evals[1].item()) if evals.numel() > 1 else 0.0
    return count_negative_eigenvalues(evals), eig0, eig1


def info_scalar(info: dict, key: str, default=None) -> float | None:
    value = info.get(key, default)
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return None
        return float(value.detach().reshape(-1)[0].cpu().item())
    return float(value)


def method_tag(args: argparse.Namespace) -> str:
    if args.method == "gad":
        tag = (
            f"lj{args.n_atoms}_gad_dt{args.dt:g}"
            f"_eps{args.epsilon:g}_sig{args.sigma:g}"
        )
        if args.high_index_descent != "gad":
            tag = f"{tag}_hi{args.high_index_descent}"
        return tag
    switch = "swEIG" if args.switch_by_eig == "true" else "swFORCE"
    return (
        f"lj{args.n_atoms}_{args.method}_{switch}_sf{args.switch_force:g}"
        f"_dt{args.gad_dt:g}_tr{args.trust_radius:g}_eps{args.epsilon:g}_sig{args.sigma:g}"
    )


def final_diagnostics(
    predict_fn,
    coords: torch.Tensor,
    atomic_nums: torch.Tensor,
) -> tuple[dict, torch.Tensor, torch.Tensor]:
    n_atoms = coords.numel() // 3
    out = predict_fn(coords, atomic_nums, do_hessian=True, require_grad=False)
    hessian = out["hessian"].reshape(3 * n_atoms, 3 * n_atoms).double()
    forces = out["forces"].reshape(n_atoms, 3).double()
    n_neg, eig0, eig1 = n_neg_eckart(hessian, coords, atomic_nums)
    diagnostics = {
        "n_neg": n_neg,
        "eig0": eig0,
        "eig1": eig1,
        "force_max": force_max(forces),
        "force_norm": force_mean(forces),
        "energy": float(out["energy"].detach().item()),
    }
    return diagnostics, forces, hessian


def run_regular_gad(args: argparse.Namespace, predict_fn, atomic_nums: torch.Tensor) -> list[dict]:
    generator = torch.Generator().manual_seed(args.seed)
    cfg = GADSearchConfig(
        n_steps=args.n_steps,
        dt=args.dt,
        k_track=args.k_track,
        use_projection=args.use_projection,
        use_adaptive_dt=args.use_adaptive_dt,
        return_weighted_step_direction=args.return_weighted_step_direction,
        high_index_descent=args.high_index_descent,
        dt_min=args.dt_min,
        dt_max=args.dt_max,
        max_atom_disp=args.max_atom_disp,
        min_interatomic_dist=args.min_interatomic_dist,
        force_threshold=args.force_threshold,
        force_criterion=args.force_criterion,
    )

    rows = []
    for sample_id in range(args.n_samples):
        coords0, start_label = make_starting_geometry(
            sample_id,
            args.start_from,
            args.n_atoms,
            args.sigma,
            generator,
            args.noise,
            args.gaussian_origin_sigma,
        )
        t0 = time.time()
        result = run_gad_search(predict_fn, coords0, atomic_nums, cfg)
        wall = time.time() - t0

        final, forces, _ = final_diagnostics(predict_fn, result.final_coords, atomic_nums)
        distances = pair_distances(result.final_coords).detach().cpu().tolist()
        converged = (
            final["n_neg"] == 1
            and force_value_from_criterion(forces, args.force_criterion) < args.force_threshold
        )

        rows.append(
            {
                "sample_id": sample_id,
                "surface": "lennard_jones",
                "method": method_tag(args),
                "n_atoms": args.n_atoms,
                "dt": args.dt,
                "epsilon": args.epsilon,
                "sigma": args.sigma,
                "noise": args.noise,
                "gaussian_origin_sigma": args.gaussian_origin_sigma,
                "start_from": args.start_from,
                "start_method": start_label,
                "converged": converged,
                "converged_step": result.converged_step,
                "total_steps": result.total_steps,
                "final_n_neg": final["n_neg"],
                "final_eig0": final["eig0"],
                "final_eig1": final["eig1"],
                "final_force_max": final["force_max"],
                "final_force_norm": final["force_norm"],
                "final_energy": final["energy"],
                "final_short_pair": shortest_pair_label(result.final_coords),
                "final_min_distance": min(distances),
                "final_distances": distances,
                "coords_flat": result.final_coords.reshape(-1).detach().cpu().tolist(),
                "atomic_nums": atomic_nums.detach().cpu().tolist(),
                "wall_time_s": wall,
            }
        )
        status = "CONV" if converged else "FAIL"
        print(
            f"  [{sample_id:3d}] {start_label:>28s} | {status} | "
            f"n_neg={final['n_neg']} fmax={final['force_max']:.3e} steps={result.total_steps}",
            flush=True,
        )

    return rows


def run_hybrid(args: argparse.Namespace, predict_fn, atomic_nums: torch.Tensor) -> list[dict]:
    generator = torch.Generator().manual_seed(args.seed)
    masses = masses_from_z(atomic_nums, dtype=torch.float64)
    switch_by_eig = args.switch_by_eig == "true"
    rows = []

    for sample_id in range(args.n_samples):
        coords, start_label = make_starting_geometry(
            sample_id,
            args.start_from,
            args.n_atoms,
            args.sigma,
            generator,
            args.noise,
            args.gaussian_origin_sigma,
        )
        coords = coords.double()
        traj_rows = []
        t0 = time.time()
        converged = False
        converged_step = None
        final = {
            "force_max": float("nan"),
            "force_norm": float("nan"),
            "n_neg": -1,
            "eig0": float("nan"),
            "eig1": float("nan"),
            "energy": float("nan"),
            "step_norm_cart": float("nan"),
            "force_norm_internal": float("nan"),
            "target_eigval": float("nan"),
            "target_mode": -1,
            "target_force_coupling": float("nan"),
            "last_step_method": "",
        }

        for step_idx in range(args.n_steps):
            diagnostics, forces, hessian = final_diagnostics(predict_fn, coords, atomic_nums)
            fmax_v = diagnostics["force_max"]
            final.update(
                {
                    "force_max": fmax_v,
                    "force_norm": diagnostics["force_norm"],
                    "n_neg": diagnostics["n_neg"],
                    "eig0": diagnostics["eig0"],
                    "eig1": diagnostics["eig1"],
                    "energy": diagnostics["energy"],
                }
            )

            if args.save_traj:
                traj_rows.append(
                    {
                        "sample_id": sample_id,
                        "step": step_idx,
                        "energy": diagnostics["energy"],
                        "force_max": fmax_v,
                        "force_norm": diagnostics["force_norm"],
                        "n_neg": diagnostics["n_neg"],
                        "eig0": diagnostics["eig0"],
                        "eig1": diagnostics["eig1"],
                        "step_method": None,
                        "step_norm_cart": None,
                        "force_norm_internal": None,
                        "target_eigval": None,
                        "target_mode": None,
                        "coords_flat": coords.reshape(-1).detach().cpu().tolist(),
                    }
                )

            if diagnostics["n_neg"] == 1 and fmax_v < args.force_threshold:
                converged = True
                converged_step = step_idx
                break

            min_curv_kw = {} if args.min_curvature is None else {"min_curvature": args.min_curvature}
            if args.method == "hybrid":
                step, info = hybrid_gad_newton_step_from_force(
                    forces.reshape(-1),
                    hessian,
                    target_mode=0,
                    gad_dt=args.gad_dt,
                    switch_force=args.switch_force,
                    trust_radius=args.trust_radius,
                    **min_curv_kw,
                )
            elif args.method == "hybrid_eckart":
                step, info = proj_step_plain(
                    force_cart=forces,
                    hessian_cart=hessian,
                    coords=coords,
                    masses=masses,
                    target_mode=0,
                    gad_dt=args.gad_dt,
                    switch_based_on_hessian_eigval=switch_by_eig,
                    switch_force=args.switch_force,
                    trust_radius=args.trust_radius,
                    **min_curv_kw,
                )
            else:
                step, info = proj_step_damped(
                    force_cart=forces,
                    hessian_cart=hessian,
                    coords=coords,
                    masses=masses,
                    target_mode=0,
                    gad_dt=args.gad_dt,
                    switch_based_on_hessian_eigval=switch_by_eig,
                    switch_force=args.switch_force,
                    trust_radius=args.trust_radius,
                    target_mode_strategy=args.target_mode_strategy,
                    high_index_descent=args.high_index_descent,
                    **min_curv_kw,
                )

            step = cap_displacement(step.reshape_as(coords), args.max_atom_disp)
            new_coords = coords + step
            if (
                args.min_interatomic_dist > 0
                and min_interatomic_distance(new_coords) < args.min_interatomic_dist
            ):
                step = 0.5 * step
                new_coords = coords + step
            coords = new_coords.detach()

            final.update(
                {
                    "step_norm_cart": info_scalar(
                        info, "step_norm_cart", torch.linalg.vector_norm(step)
                    ),
                    "force_norm_internal": info_scalar(info, "force_norm_internal"),
                    "target_eigval": info_scalar(info, "target_eigval"),
                    "target_mode": int(info.get("target_mode", 0)),
                    "target_force_coupling": info_scalar(info, "target_force_coupling"),
                    "last_step_method": info.get("method", ""),
                }
            )
            if args.save_traj:
                traj_rows[-1].update(
                    {
                        "step_method": final["last_step_method"],
                        "step_norm_cart": final["step_norm_cart"],
                        "force_norm_internal": final["force_norm_internal"],
                        "target_eigval": final["target_eigval"],
                        "target_mode": final["target_mode"],
                    }
                )

        if args.save_traj and traj_rows:
            traj_path = args.output_dir / f"traj_{method_tag(args)}_{sample_id:04d}.parquet"
            pd.DataFrame(traj_rows).to_parquet(traj_path)

        wall = time.time() - t0
        distances = pair_distances(coords).detach().cpu().tolist()
        rows.append(
            {
                "sample_id": sample_id,
                "surface": "lennard_jones",
                "method": method_tag(args),
                "n_atoms": args.n_atoms,
                "gad_dt": args.gad_dt,
                "trust_radius": args.trust_radius,
                "switch_force": args.switch_force,
                "epsilon": args.epsilon,
                "sigma": args.sigma,
                "noise": args.noise,
                "gaussian_origin_sigma": args.gaussian_origin_sigma,
                "start_from": args.start_from,
                "start_method": start_label,
                "converged": converged,
                "converged_step": converged_step,
                "total_steps": converged_step + 1 if converged_step is not None else args.n_steps,
                "final_n_neg": final["n_neg"],
                "final_eig0": final["eig0"],
                "final_eig1": final["eig1"],
                "final_force_max": final["force_max"],
                "final_force_norm": final["force_norm"],
                "final_energy": final["energy"],
                "final_step_norm_cart": final["step_norm_cart"],
                "final_force_norm_internal": final["force_norm_internal"],
                "final_target_eigval": final["target_eigval"],
                "final_target_mode": final["target_mode"],
                "final_target_force_coupling": final["target_force_coupling"],
                "last_step_method": final["last_step_method"],
                "final_short_pair": shortest_pair_label(coords),
                "final_min_distance": min(distances),
                "final_distances": distances,
                "coords_flat": coords.reshape(-1).detach().cpu().tolist(),
                "atomic_nums": atomic_nums.detach().cpu().tolist(),
                "wall_time_s": wall,
            }
        )
        status = "CONV" if converged else "FAIL"
        print(
            f"  [{sample_id:3d}] {start_label:>28s} | {status} | "
            f"n_neg={final['n_neg']} fmax={final['force_max']:.3e} "
            f"steps={rows[-1]['total_steps']} last={final['last_step_method']}",
            flush=True,
        )

    return rows


def main() -> None:
    args = parse_args()
    if args.n_atoms < 2:
        sys.exit("--n-atoms must be at least 2")
    if args.gaussian_origin_sigma <= 0:
        sys.exit("--gaussian-origin-sigma must be positive")
    if args.method == "gad" and args.high_index_descent not in {"gad", "gradient"}:
        sys.exit("--method gad supports --high-index-descent gad or gradient")
    if args.method not in {"gad", "hybrid_damped_eckart"} and args.high_index_descent != "gad":
        sys.exit("--high-index-descent only applies to --method gad or hybrid_damped_eckart")
    if args.method != "hybrid_damped_eckart" and args.target_mode_strategy != "fixed":
        sys.exit("--target-mode-strategy only applies to hybrid_damped_eckart")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    params = LennardJonesParams(epsilon=args.epsilon, sigma=args.sigma)
    predict_fn = make_lj_predict_fn(
        params,
        n_atoms=args.n_atoms,
        compile_forces=args.lj_compile,
        compile_hessian=args.lj_compile,
    )
    atomic_nums = lj_atomic_nums(args.n_atoms, atomic_number=args.atomic_number)
    run_id = uuid.uuid4().hex[:8]
    print(
        f"LJ benchmark | method={args.method} start={args.start_from} "
        f"n_atoms={args.n_atoms} epsilon={args.epsilon:g} sigma={args.sigma:g} "
        f"samples={args.n_samples} run_id={run_id}",
        flush=True,
    )

    if args.method == "gad":
        rows = run_regular_gad(args, predict_fn, atomic_nums)
    else:
        rows = run_hybrid(args, predict_fn, atomic_nums)

    summary_path = args.output_dir / f"summary_{method_tag(args)}_{run_id}.parquet"
    pd.DataFrame(rows).to_parquet(summary_path)
    print(f"\nWrote {summary_path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
