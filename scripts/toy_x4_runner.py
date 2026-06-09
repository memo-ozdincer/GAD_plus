#!/usr/bin/env python
"""Run GAD or hybrid GAD/Newton on the analytic X4 EVB toy surface."""
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

from gadplus.calculator.toy_x4 import (
    X4EVBParams,
    X4_PAIRS,
    adjacent_switches,
    adjacent_ts_guess,
    classify_short_bond,
    disjoint_switches,
    disjoint_ts_guess,
    make_x4_predict_fn,
    minimum_geometry,
    pair_distances,
    pair_label,
    x4_atomic_nums,
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
        description="Benchmark GAD and hybrid GAD/Newton on the X4 EVB toy surface."
    )
    parser.add_argument(
        "--method",
        choices=["gad", "hybrid", "hybrid_eckart", "hybrid_damped_eckart"],
        default="gad",
    )
    parser.add_argument(
        "--start-from",
        choices=["mixed_ts", "adjacent_ts", "disjoint_ts", "minima", "random"],
        default="mixed_ts",
    )
    parser.add_argument("--n-samples", type=int, default=24)
    parser.add_argument("--n-steps", type=int, default=1000)
    parser.add_argument("--noise", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/toy_x4"))
    parser.add_argument("--save-traj", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--beta", type=float, default=3.05)
    parser.add_argument("--active-distance", type=float, default=1.02)
    parser.add_argument("--force-threshold", type=float, default=1.0e-3)
    parser.add_argument(
        "--force-criterion",
        choices=["fmax", "force_norm"],
        default="fmax",
        help="Convergence force metric. Hybrid methods currently gate on fmax.",
    )
    parser.add_argument("--max-atom-disp", type=float, default=0.10)
    parser.add_argument("--min-interatomic-dist", type=float, default=0.35)

    parser.add_argument("--dt", type=float, default=3.0e-3, help="Regular GAD timestep.")
    parser.add_argument("--k-track", type=int, default=8)
    parser.add_argument("--use-projection", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-adaptive-dt", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--return-weighted-step-direction",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Return the legacy sqrt(M)-weighted projected direction instead of the unweighted Cartesian step",
    )
    parser.add_argument("--dt-min", type=float, default=1.0e-5)
    parser.add_argument("--dt-max", type=float, default=0.1)

    parser.add_argument("--gad-dt", type=float, default=5.0e-3, help="Hybrid GAD step size.")
    parser.add_argument("--trust-radius", type=float, default=0.02)
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
    params: X4EVBParams,
    active_distance: float,
    generator: torch.Generator,
    noise: float,
) -> tuple[torch.Tensor, str, str]:
    """Build one labelled X4 starting geometry plus optional Cartesian noise."""

    adjacent = adjacent_switches()
    disjoint = disjoint_switches()

    if start_from == "minima":
        pair = X4_PAIRS[sample_id % len(X4_PAIRS)]
        coords = minimum_geometry(pair, params)
        label = pair_label(pair)
        start_label = f"minimum_{label}"
    elif start_from == "adjacent_ts":
        first, second = adjacent[sample_id % len(adjacent)]
        coords = adjacent_ts_guess(first, second, params, active_distance)
        label = f"{pair_label(first)}-{pair_label(second)}"
        start_label = f"adjacent_ts_{label}"
    elif start_from == "disjoint_ts":
        first, second = disjoint[sample_id % len(disjoint)]
        coords = disjoint_ts_guess(first, second, params, active_distance)
        label = f"{pair_label(first)}-{pair_label(second)}"
        start_label = f"disjoint_ts_{label}"
    elif start_from == "mixed_ts":
        switches = adjacent + disjoint
        first, second = switches[sample_id % len(switches)]
        if set(first).intersection(second):
            coords = adjacent_ts_guess(first, second, params, active_distance)
            kind = "adjacent"
        else:
            coords = disjoint_ts_guess(first, second, params, active_distance)
            kind = "disjoint"
        label = f"{pair_label(first)}-{pair_label(second)}"
        start_label = f"{kind}_ts_{label}"
    else:
        pair = X4_PAIRS[sample_id % len(X4_PAIRS)]
        coords = minimum_geometry(pair, params)
        coords = coords + 0.35 * torch.randn(coords.shape, generator=generator, dtype=coords.dtype)
        label = pair_label(pair)
        start_label = f"random_from_{label}"

    if noise > 0:
        coords = coords + noise * torch.randn(coords.shape, generator=generator, dtype=coords.dtype)

    coords = coords - coords.mean(dim=0, keepdim=True)
    return coords, start_label, label


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
        return f"toy_x4_gad_dt{args.dt:g}_beta{args.beta:g}"
    switch = "swEIG" if args.switch_by_eig == "true" else "swFORCE"
    return (
        f"toy_x4_{args.method}_{switch}_sf{args.switch_force:g}"
        f"_dt{args.gad_dt:g}_tr{args.trust_radius:g}_beta{args.beta:g}"
    )


def run_regular_gad(args: argparse.Namespace, predict_fn, atomic_nums: torch.Tensor) -> list[dict]:
    params = X4EVBParams(beta=args.beta)
    generator = torch.Generator().manual_seed(args.seed)
    cfg = GADSearchConfig(
        n_steps=args.n_steps,
        dt=args.dt,
        k_track=args.k_track,
        use_projection=args.use_projection,
        use_adaptive_dt=args.use_adaptive_dt,
        return_weighted_step_direction=args.return_weighted_step_direction,
        dt_min=args.dt_min,
        dt_max=args.dt_max,
        max_atom_disp=args.max_atom_disp,
        min_interatomic_dist=args.min_interatomic_dist,
        force_threshold=args.force_threshold,
        force_criterion=args.force_criterion,
    )

    rows = []
    for sample_id in range(args.n_samples):
        coords0, start_label, switch_label = make_starting_geometry(
            sample_id,
            args.start_from,
            params,
            args.active_distance,
            generator,
            args.noise,
        )
        t0 = time.time()
        result = run_gad_search(predict_fn, coords0, atomic_nums, cfg)
        wall = time.time() - t0

        out = predict_fn(result.final_coords, atomic_nums, do_hessian=True, require_grad=False)
        hessian = out["hessian"].reshape(12, 12)
        n_neg, eig0, eig1 = n_neg_eckart(hessian, result.final_coords, atomic_nums)
        forces = out["forces"].reshape(4, 3)
        distances = pair_distances(result.final_coords).detach().cpu().tolist()
        final_short_bond = classify_short_bond(result.final_coords)
        converged = n_neg == 1 and force_value_from_criterion(forces, args.force_criterion) < args.force_threshold

        rows.append(
            {
                "sample_id": sample_id,
                "surface": "toy_x4_evb",
                "method": method_tag(args),
                "dt": args.dt,
                "beta": args.beta,
                "noise": args.noise,
                "start_from": args.start_from,
                "start_method": start_label,
                "target_switch": switch_label,
                "converged": converged,
                "converged_step": result.converged_step,
                "total_steps": result.total_steps,
                "final_n_neg": n_neg,
                "final_eig0": eig0,
                "final_eig1": eig1,
                "final_force_max": force_max(forces),
                "final_force_norm": force_mean(forces),
                "final_energy": float(out["energy"].detach().item()),
                "final_short_bond": final_short_bond,
                "final_distances": distances,
                "coords_flat": result.final_coords.reshape(-1).detach().cpu().tolist(),
                "atomic_nums": atomic_nums.detach().cpu().tolist(),
                "wall_time_s": wall,
            }
        )
        status = "CONV" if converged else "FAIL"
        print(
            f"  [{sample_id:3d}] {start_label:>20s} | {status} | "
            f"n_neg={n_neg} fmax={force_max(forces):.3e} steps={result.total_steps}",
            flush=True,
        )

    return rows


def run_hybrid(args: argparse.Namespace, predict_fn, atomic_nums: torch.Tensor) -> list[dict]:
    params = X4EVBParams(beta=args.beta)
    generator = torch.Generator().manual_seed(args.seed)
    masses = masses_from_z(atomic_nums, dtype=torch.float64)
    switch_by_eig = args.switch_by_eig == "true"
    rows = []

    for sample_id in range(args.n_samples):
        coords, start_label, switch_label = make_starting_geometry(
            sample_id,
            args.start_from,
            params,
            args.active_distance,
            generator,
            args.noise,
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
            out = predict_fn(coords, atomic_nums, do_hessian=True, require_grad=False)
            energy = out["energy"]
            forces = out["forces"].reshape(4, 3).double()
            hessian = out["hessian"].reshape(12, 12).double()

            fmax_v = force_max(forces)
            fnorm_v = force_mean(forces)
            energy_v = float(energy.detach().item())
            n_neg, eig0, eig1 = n_neg_eckart(hessian, coords, atomic_nums)
            final.update(
                {
                    "force_max": fmax_v,
                    "force_norm": fnorm_v,
                    "n_neg": n_neg,
                    "eig0": eig0,
                    "eig1": eig1,
                    "energy": energy_v,
                }
            )

            if args.save_traj:
                traj_rows.append(
                    {
                        "sample_id": sample_id,
                        "step": step_idx,
                        "energy": energy_v,
                        "force_max": fmax_v,
                        "force_norm": fnorm_v,
                        "n_neg": n_neg,
                        "eig0": eig0,
                        "eig1": eig1,
                        "step_method": None,
                        "step_norm_cart": None,
                        "force_norm_internal": None,
                        "target_eigval": None,
                        "target_mode": None,
                        "coords_flat": coords.reshape(-1).detach().cpu().tolist(),
                    }
                )

            if n_neg == 1 and fmax_v < args.force_threshold:
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
        final_short_bond = classify_short_bond(coords)
        distances = pair_distances(coords).detach().cpu().tolist()
        rows.append(
            {
                "sample_id": sample_id,
                "surface": "toy_x4_evb",
                "method": method_tag(args),
                "gad_dt": args.gad_dt,
                "trust_radius": args.trust_radius,
                "switch_force": args.switch_force,
                "beta": args.beta,
                "noise": args.noise,
                "start_from": args.start_from,
                "start_method": start_label,
                "target_switch": switch_label,
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
                "final_short_bond": final_short_bond,
                "final_distances": distances,
                "coords_flat": coords.reshape(-1).detach().cpu().tolist(),
                "atomic_nums": atomic_nums.detach().cpu().tolist(),
                "wall_time_s": wall,
            }
        )
        status = "CONV" if converged else "FAIL"
        print(
            f"  [{sample_id:3d}] {start_label:>20s} | {status} | "
            f"n_neg={final['n_neg']} fmax={final['force_max']:.3e} "
            f"steps={rows[-1]['total_steps']} last={final['last_step_method']}",
            flush=True,
        )

    return rows


def main() -> None:
    args = parse_args()
    if args.method != "hybrid_damped_eckart" and (
        args.high_index_descent != "gad" or args.target_mode_strategy != "fixed"
    ):
        sys.exit("--high-index-descent and --target-mode-strategy only apply to hybrid_damped_eckart")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    params = X4EVBParams(beta=args.beta)
    predict_fn = make_x4_predict_fn(params)
    atomic_nums = x4_atomic_nums()
    run_id = uuid.uuid4().hex[:8]
    print(
        f"X4 toy benchmark | method={args.method} start={args.start_from} "
        f"beta={args.beta:g} samples={args.n_samples} run_id={run_id}",
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
