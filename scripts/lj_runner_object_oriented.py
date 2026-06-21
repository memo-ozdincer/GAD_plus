#!/usr/bin/env python
"""Run the object-oriented projected-GAD optimizer on Lennard-Jones clusters."""
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
    pair_distances,
    pentagonal_bipyramid_geometry,
    random_cluster_geometry,
    shortest_pair_label,
)
from gadplus.search.transition_state_optimizer import TransitionStateOptimizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark the object-oriented projected GAD transition-state optimizer "
            "on a Lennard-Jones cluster."
        )
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
    parser.add_argument("--output-dir", type=Path, default=Path("runs/lj_oo"))

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
        help="Convergence force metric.",
    )
    parser.add_argument("--max-atom-disp", type=float, default=0.05)
    parser.add_argument("--min-interatomic-dist", type=float, default=0.75)

    parser.add_argument("--dt", type=float, default=1.0e-3)
    parser.add_argument("--k-track", type=int, default=8)
    parser.add_argument("--use-adaptive-dt", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--return-weighted-step-direction",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Return the legacy sqrt(M)-weighted projected direction instead of the unweighted Cartesian step",
    )
    parser.add_argument("--dt-min", type=float, default=1.0e-5)
    parser.add_argument("--dt-max", type=float, default=0.05)
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

    if start_from in {"minimum", "expanded_minimum"} and sample_id:
        coords = torch.roll(coords, shifts=sample_id % n_atoms, dims=0)

    return coords - coords.mean(dim=0, keepdim=True), start_label


def method_tag(args: argparse.Namespace) -> str:
    return (
        f"lj{args.n_atoms}_transition_state_optimizer_dt{args.dt:g}"
        f"_eps{args.epsilon:g}_sig{args.sigma:g}_hi_gradient_projected"
    )


def final_force_value(final: dict[str, float | int], force_criterion: str) -> float:
    if force_criterion == "fmax":
        return float(final["force_max"])
    return float(final["force_norm"])


def run_optimizer(args: argparse.Namespace) -> list[dict]:
    generator = torch.Generator().manual_seed(args.seed)
    optimizer = TransitionStateOptimizer(
        n_atoms=args.n_atoms,
        dt=args.dt,
        epsilon=args.epsilon,
        sigma=args.sigma,
        atomic_number=args.atomic_number,
        lj_compile=args.lj_compile,
        n_steps=args.n_steps,
        k_track=args.k_track,
        use_adaptive_dt=args.use_adaptive_dt,
        dt_min=args.dt_min,
        dt_max=args.dt_max,
        max_atom_disp=args.max_atom_disp,
        min_interatomic_dist=args.min_interatomic_dist,
        force_threshold=args.force_threshold,
        force_criterion=args.force_criterion,
        return_weighted_step_direction=args.return_weighted_step_direction,
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
        result = optimizer.optimize(coords0)
        wall = time.time() - t0

        final = optimizer.final_diagnostics(result.final_coords)
        distances = pair_distances(result.final_coords).detach().cpu().tolist()
        converged = (
            final["n_neg"] == 1
            and final_force_value(final, args.force_criterion) < args.force_threshold
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
                "atomic_nums": optimizer.atomic_nums.detach().cpu().tolist(),
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


def main() -> None:
    args = parse_args()
    if args.n_atoms < 2:
        sys.exit("--n-atoms must be at least 2")
    if args.gaussian_origin_sigma <= 0:
        sys.exit("--gaussian-origin-sigma must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_id = uuid.uuid4().hex[:8]
    print(
        f"LJ OO benchmark | method=transition_state_optimizer start={args.start_from} "
        f"n_atoms={args.n_atoms} epsilon={args.epsilon:g} sigma={args.sigma:g} "
        f"samples={args.n_samples} run_id={run_id}",
        flush=True,
    )

    rows = run_optimizer(args)
    summary_path = args.output_dir / f"summary_{method_tag(args)}_{run_id}.parquet"
    pd.DataFrame(rows).to_parquet(summary_path)
    print(f"\nWrote {summary_path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
