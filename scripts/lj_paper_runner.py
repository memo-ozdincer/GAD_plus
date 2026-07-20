#!/usr/bin/env python
"""Paper-style GAD/Sella sweeps on analytic Lennard-Jones clusters.

This is the LJ analogue of the Transition1x paper experiments. It compares
GAD, hybrid GAD/Newton, and Sella on identical LJ starts, with hydrogen as the
default atom assignment for mass/Eckart bookkeeping.
"""
from __future__ import annotations

import argparse
import inspect
import os
import sys
import time
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

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
from gadplus.core.convergence import force_max, force_mean
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


class CachedPredictCalculator(Calculator):
    """ASE Calculator that caches energy, forces, and Hessian from a PredictFn."""

    implemented_properties = ["energy", "forces"]

    def __init__(self, predict_fn, atomic_nums, device: torch.device, **kwargs):
        super().__init__(**kwargs)
        self.predict_fn = predict_fn
        self.atomic_nums = atomic_nums
        self.device = device
        self._cached_coords: torch.Tensor | None = None
        self._cached_result = None
        self.n_calls = 0

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        coords = torch.tensor(self.atoms.positions, dtype=torch.float64, device=self.device)
        out = self.predict_fn(coords, self.atomic_nums, do_hessian=True, require_grad=False)
        self.n_calls += 1
        self._cached_coords = coords.clone()
        self._cached_result = out

        energy = out["energy"]
        self.results["energy"] = float(energy.detach().cpu().item())
        self.results["forces"] = out["forces"].detach().cpu().numpy().reshape(-1, 3)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method",
        choices=[
            "gad",
            "hybrid",
            "hybrid_eckart",
            "hybrid_damped_eckart",
            "sella",
        ],
        required=True,
    )
    parser.add_argument(
        "--start-from",
        choices=["minimum", "minimum_noised", "random", "expanded_minimum"],
        default="minimum_noised",
    )
    parser.add_argument("--n-atoms", type=int, default=7)
    parser.add_argument("--n-samples", type=int, default=287)
    parser.add_argument("--sample-start", type=int, default=None)
    parser.add_argument("--sample-end", type=int, default=None)
    parser.add_argument("--n-steps", type=int, default=2000)
    parser.add_argument("--noise", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/lj_paper"))
    parser.add_argument("--save-traj", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument(
        "--atomic-number",
        type=int,
        default=1,
        help="Element used only for masses/symbols. Default: H.",
    )
    parser.add_argument("--force-threshold", type=float, default=0.01)
    parser.add_argument("--max-atom-disp", type=float, default=0.05)
    parser.add_argument("--min-interatomic-dist", type=float, default=0.75)

    parser.add_argument("--dt", type=float, default=1.0e-3)
    parser.add_argument("--k-track", type=int, default=8)
    parser.add_argument("--use-projection", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-adaptive-dt", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dt-min", type=float, default=1.0e-5)
    parser.add_argument("--dt-max", type=float, default=0.05)
    parser.add_argument(
        "--blend-sharpness",
        type=float,
        default=0.0,
        help=(
            "Smoothly gate the GAD v1 inversion with sigmoid(k * lambda2). "
            "0 keeps ordinary single-mode GAD."
        ),
    )

    parser.add_argument("--gad-dt", type=float, default=1.0e-3)
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

    parser.add_argument("--sella-cartesian", action="store_true", default=False)
    parser.add_argument(
        "--sella-apply-eckart",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--sella-delta0", type=float, default=0.048)
    parser.add_argument("--sella-gamma", type=float, default=0.0)
    parser.add_argument("--sella-diag-every", type=int, default=1)
    parser.add_argument("--config-tag", type=str, default="")
    args = parser.parse_args()
    if args.blend_sharpness < 0:
        parser.error("--blend-sharpness must be nonnegative.")
    return args


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def make_starting_geometry(
    sample_id: int,
    args: argparse.Namespace,
    generator: torch.Generator,
) -> tuple[torch.Tensor, str]:
    if args.n_atoms == 7 and args.start_from in {
        "minimum",
        "minimum_noised",
        "expanded_minimum",
    }:
        coords = pentagonal_bipyramid_geometry(args.sigma)
        start_label = "lj7_pentagonal_bipyramid"
    else:
        coords = random_cluster_geometry(args.n_atoms, sigma=args.sigma, generator=generator)
        start_label = "random_cluster"

    if args.start_from == "random":
        coords = random_cluster_geometry(args.n_atoms, sigma=args.sigma, generator=generator)
        start_label = "random_cluster"
    elif args.start_from == "expanded_minimum":
        coords = 1.15 * coords
        start_label = f"expanded_{start_label}"
    elif args.start_from == "minimum" and args.n_atoms != 7:
        start_label = "random_cluster_no_lj_minimum"

    if args.start_from in {"minimum_noised", "random"} and args.noise > 0:
        coords = coords + args.noise * torch.randn(
            coords.shape, generator=generator, dtype=coords.dtype
        )
        start_label = f"{start_label}_noise{args.noise:g}"

    if args.start_from in {"minimum", "expanded_minimum"} and sample_id:
        coords = torch.roll(coords, shifts=sample_id % args.n_atoms, dims=0)

    return coords - coords.mean(dim=0, keepdim=True), start_label


def n_neg_eckart(
    hessian: torch.Tensor,
    coords: torch.Tensor,
    atomic_nums: torch.Tensor,
) -> tuple[int, float, float]:
    evals, _, _ = vib_eig(hessian, coords, atomic_nums_to_symbols(atomic_nums), purify=False)
    evals = torch.sort(evals).values
    eig0 = float(evals[0].item()) if evals.numel() > 0 else 0.0
    eig1 = float(evals[1].item()) if evals.numel() > 1 else 0.0
    return int((evals < -1.0e-4).sum().item()), eig0, eig1


def final_diagnostics(predict_fn, coords: torch.Tensor, atomic_nums: torch.Tensor):
    n_atoms = coords.numel() // 3
    out = predict_fn(coords, atomic_nums, do_hessian=True, require_grad=False)
    hessian = out["hessian"].reshape(3 * n_atoms, 3 * n_atoms).double()
    forces = out["forces"].reshape(n_atoms, 3).double()
    n_neg, eig0, eig1 = n_neg_eckart(hessian, coords, atomic_nums)
    return {
        "n_neg": n_neg,
        "eig0": eig0,
        "eig1": eig1,
        "force_max": force_max(forces),
        "force_norm": force_mean(forces),
        "energy": float(out["energy"].detach().reshape(-1)[0].item()),
    }, forces, hessian


def method_tag(args: argparse.Namespace) -> str:
    if args.method == "gad":
        blend = f"_blend{args.blend_sharpness:g}" if args.blend_sharpness > 0 else ""
        return f"lj{args.n_atoms}_gad_dt{args.dt:g}_Z{args.atomic_number}{blend}"
    if args.method == "sella":
        coord = "cart" if args.sella_cartesian else "internal"
        eck = "_eckart" if args.sella_apply_eckart else ""
        tag = f"_{args.config_tag}" if args.config_tag else ""
        return f"lj{args.n_atoms}_sella_{coord}{eck}_d{args.sella_diag_every}_Z{args.atomic_number}{tag}"
    switch = "swEIG" if args.switch_by_eig == "true" else "swFORCE"
    return (
        f"lj{args.n_atoms}_{args.method}_{switch}_sf{args.switch_force:g}"
        f"_dt{args.gad_dt:g}_tr{args.trust_radius:g}_Z{args.atomic_number}"
    )


def info_scalar(info: dict, key: str, default=None) -> float | None:
    value = info.get(key, default)
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return None
        return float(value.detach().reshape(-1)[0].cpu().item())
    return float(value)


def common_row(args, sample_id, start_label, coords, atomic_nums, final, wall, extra=None):
    distances = pair_distances(coords).detach().cpu().tolist()
    nneg1 = final["n_neg"] == 1
    row = {
        "sample_id": sample_id,
        "surface": "lennard_jones",
        "method": method_tag(args),
        "start_from": args.start_from,
        "start_method": start_label,
        "n_atoms": args.n_atoms,
        "atomic_number": args.atomic_number,
        "epsilon": args.epsilon,
        "sigma": args.sigma,
        "noise": args.noise,
        "noise_pm_equiv": int(round(args.noise * 1000)),
        "blend_sharpness": args.blend_sharpness,
        "force_threshold": args.force_threshold,
        "final_n_neg": final["n_neg"],
        "final_eig0": final["eig0"],
        "final_eig1": final["eig1"],
        "final_force_max": final["force_max"],
        "final_force_norm": final["force_norm"],
        "final_energy": final["energy"],
        "final_short_pair": shortest_pair_label(coords),
        "final_min_distance": min(distances),
        "final_distances": distances,
        "coords_flat": coords.reshape(-1).detach().cpu().tolist(),
        "atomic_nums": atomic_nums.detach().cpu().tolist(),
        "is_nneg1": nneg1,
        "conv_nneg1_fmax001": nneg1 and final["force_max"] < 0.01,
        "conv_nneg1_fmax003": nneg1 and final["force_max"] < 0.03,
        "conv_nneg1_fmax005": nneg1 and final["force_max"] < 0.05,
        "conv_nneg1_fmax0005": nneg1 and final["force_max"] < 0.005,
        "converged": nneg1 and final["force_max"] < args.force_threshold,
        "wall_time_s": wall,
    }
    if extra:
        row.update(extra)
    return row


def run_gad(args, predict_fn, atomic_nums, sample_ids: range) -> list[dict]:
    cfg = GADSearchConfig(
        n_steps=args.n_steps,
        dt=args.dt,
        k_track=args.k_track,
        use_projection=args.use_projection,
        use_adaptive_dt=args.use_adaptive_dt,
        dt_min=args.dt_min,
        dt_max=args.dt_max,
        max_atom_disp=args.max_atom_disp,
        min_interatomic_dist=args.min_interatomic_dist,
        force_threshold=args.force_threshold,
        force_criterion="fmax",
        blend_sharpness=args.blend_sharpness,
    )
    rows = []
    for sample_id in sample_ids:
        generator = torch.Generator().manual_seed(args.seed + sample_id)
        coords0, start_label = make_starting_geometry(sample_id, args, generator)
        coords0 = coords0.to(device=atomic_nums.device, dtype=torch.float64)
        t0 = time.time()
        result = run_gad_search(predict_fn, coords0, atomic_nums, cfg)
        wall = time.time() - t0
        final, _, _ = final_diagnostics(predict_fn, result.final_coords.to(atomic_nums.device), atomic_nums)
        row = common_row(
            args,
            sample_id,
            start_label,
            result.final_coords,
            atomic_nums,
            final,
            wall,
            {
                "dt": args.dt,
                "converged_step": result.converged_step,
                "total_steps": result.total_steps,
            },
        )
        rows.append(row)
        print(
            f"[{sample_id:04d}] gad {start_label} conv={row['converged']} "
            f"nneg={final['n_neg']} fmax={final['force_max']:.3e} steps={result.total_steps}",
            flush=True,
        )
    return rows


def run_hybrid(args, predict_fn, atomic_nums, sample_ids: range) -> list[dict]:
    masses = masses_from_z(atomic_nums, dtype=torch.float64)
    switch_by_eig = args.switch_by_eig == "true"
    rows = []
    for sample_id in sample_ids:
        generator = torch.Generator().manual_seed(args.seed + sample_id)
        coords, start_label = make_starting_geometry(sample_id, args, generator)
        coords = coords.to(device=atomic_nums.device, dtype=torch.float64)
        t0 = time.time()
        converged_step = None
        final = None
        last_info = {}
        for step_idx in range(args.n_steps):
            final, forces, hessian = final_diagnostics(predict_fn, coords, atomic_nums)
            if final["n_neg"] == 1 and final["force_max"] < args.force_threshold:
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
                step = step.reshape_as(coords)
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
                damped_kwargs = {}
                damped_params = inspect.signature(proj_step_damped).parameters
                if "target_mode_strategy" in damped_params:
                    damped_kwargs["target_mode_strategy"] = args.target_mode_strategy
                if "high_index_descent" in damped_params:
                    damped_kwargs["high_index_descent"] = args.high_index_descent
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
                    **min_curv_kw,
                    **damped_kwargs,
                )
            step = cap_displacement(step.reshape_as(coords), args.max_atom_disp)
            new_coords = coords + step
            if args.min_interatomic_dist > 0 and min_interatomic_distance(new_coords) < args.min_interatomic_dist:
                step = 0.5 * step
                new_coords = coords + step
            coords = new_coords.detach()
            last_info = info
        if final is None:
            final, _, _ = final_diagnostics(predict_fn, coords, atomic_nums)
        wall = time.time() - t0
        total_steps = converged_step + 1 if converged_step is not None else args.n_steps
        row = common_row(
            args,
            sample_id,
            start_label,
            coords,
            atomic_nums,
            final,
            wall,
            {
                "gad_dt": args.gad_dt,
                "trust_radius": args.trust_radius,
                "switch_force": args.switch_force,
                "converged_step": converged_step,
                "total_steps": total_steps,
                "last_step_method": last_info.get("method", ""),
                "final_step_norm_cart": info_scalar(last_info, "step_norm_cart"),
                "final_force_norm_internal": info_scalar(last_info, "force_norm_internal"),
                "final_target_eigval": info_scalar(last_info, "target_eigval"),
            },
        )
        rows.append(row)
        print(
            f"[{sample_id:04d}] {args.method} conv={row['converged']} "
            f"nneg={final['n_neg']} fmax={final['force_max']:.3e} steps={total_steps}",
            flush=True,
        )
    return rows


def make_hessian_function(calc: CachedPredictCalculator, apply_eckart: bool):
    def hessian_function(atoms):
        coords = torch.tensor(atoms.positions, dtype=torch.float64, device=calc.device)
        if calc._cached_coords is not None and torch.equal(coords, calc._cached_coords):
            hess = calc._cached_result["hessian"]
        else:
            out = calc.predict_fn(coords, calc.atomic_nums, do_hessian=True, require_grad=False)
            calc._cached_coords = coords.clone()
            calc._cached_result = out
            hess = out["hessian"]
        hess_t = hess.detach().reshape(3 * len(atoms), 3 * len(atoms)).to(torch.float64)
        if apply_eckart:
            from gadplus.projection.projection import _eckart_projector, get_mass_weights

            atomsymbols = atomic_nums_to_symbols(calc.atomic_nums)
            masses, _m3, sqrt_m, sqrt_m_inv = get_mass_weights(atomsymbols, device=hess_t.device)
            h_mw = torch.diag(sqrt_m_inv) @ hess_t @ torch.diag(sqrt_m_inv)
            p = _eckart_projector(coords.reshape(-1, 3), masses)
            h_mw = 0.5 * (p @ h_mw @ p + (p @ h_mw @ p).T)
            hess_t = torch.diag(sqrt_m) @ h_mw @ torch.diag(sqrt_m)
        return hess_t.detach().cpu().numpy().astype(np.float64)

    return hessian_function


def run_sella(args, predict_fn, atomic_nums, sample_ids: range) -> list[dict]:
    from sella import Sella

    rows = []
    for sample_id in sample_ids:
        generator = torch.Generator().manual_seed(args.seed + sample_id)
        coords0, start_label = make_starting_geometry(sample_id, args, generator)
        positions_np = coords0.detach().cpu().numpy().reshape(-1, 3)
        numbers_np = atomic_nums.detach().cpu().numpy().astype(int)
        atoms = Atoms(numbers=numbers_np, positions=positions_np)
        ase_calc = CachedPredictCalculator(predict_fn, atomic_nums, atomic_nums.device)
        atoms.calc = ase_calc
        hessian_fn = make_hessian_function(ase_calc, apply_eckart=args.sella_apply_eckart)

        t0 = time.time()
        try:
            opt = Sella(
                atoms=atoms,
                order=1,
                internal=not args.sella_cartesian,
                trajectory=None,
                logfile=None,
                delta0=args.sella_delta0,
                diag_every_n=args.sella_diag_every,
                gamma=args.sella_gamma,
                rho_inc=1.035,
                rho_dec=5.0,
                sigma_inc=1.15,
                sigma_dec=0.65,
                hessian_function=hessian_fn,
            )
            sella_converged = bool(opt.run(fmax=args.force_threshold, steps=args.n_steps))
            steps_taken = int(opt.nsteps)
            error = ""
        except Exception as exc:
            sella_converged = False
            steps_taken = args.n_steps
            error = repr(exc)
        wall = time.time() - t0

        coords = torch.tensor(atoms.positions, dtype=torch.float64, device=atomic_nums.device)
        try:
            final, _, _ = final_diagnostics(predict_fn, coords, atomic_nums)
        except Exception as exc:
            final = {
                "n_neg": -1,
                "eig0": float("nan"),
                "eig1": float("nan"),
                "force_max": float("inf"),
                "force_norm": float("inf"),
                "energy": float("nan"),
            }
            error = error or repr(exc)

        row = common_row(
            args,
            sample_id,
            start_label,
            coords,
            atomic_nums,
            final,
            wall,
            {
                "sella_converged": sella_converged,
                "sella_cartesian": args.sella_cartesian,
                "sella_apply_eckart": args.sella_apply_eckart,
                "sella_delta0": args.sella_delta0,
                "sella_gamma": args.sella_gamma,
                "sella_diag_every": args.sella_diag_every,
                "total_steps": steps_taken,
                "converged_step": steps_taken if final["n_neg"] == 1 and final["force_max"] < args.force_threshold else None,
                "n_func_evals": int(getattr(ase_calc, "n_calls", 0)),
                "error": error,
            },
        )
        rows.append(row)
        print(
            f"[{sample_id:04d}] sella conv={row['converged']} sella={sella_converged} "
            f"nneg={final['n_neg']} fmax={final['force_max']:.3e} steps={steps_taken}",
            flush=True,
        )
    return rows


def main() -> None:
    args = parse_args()
    if args.n_atoms < 2:
        sys.exit("--n-atoms must be at least 2")

    device = resolve_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    params = LennardJonesParams(epsilon=args.epsilon, sigma=args.sigma)
    predict_fn = make_lj_predict_fn(params)
    atomic_nums = lj_atomic_nums(args.n_atoms, atomic_number=args.atomic_number, device=device)

    s_start = args.sample_start if args.sample_start is not None else 0
    s_end = args.sample_end if args.sample_end is not None else args.n_samples
    sample_ids = range(s_start, min(s_end, args.n_samples))
    print(
        f"LJ paper sweep | method={args.method} tag={method_tag(args)} "
        f"start={args.start_from} n_atoms={args.n_atoms} Z={args.atomic_number} "
        f"noise={args.noise:g} samples={s_start}:{min(s_end, args.n_samples)} "
        f"device={device}",
        flush=True,
    )

    if args.method == "gad":
        rows = run_gad(args, predict_fn, atomic_nums, sample_ids)
    elif args.method == "sella":
        rows = run_sella(args, predict_fn, atomic_nums, sample_ids)
    else:
        rows = run_hybrid(args, predict_fn, atomic_nums, sample_ids)

    run_id = uuid.uuid4().hex[:8]
    suffix = f"_s{s_start}-{min(s_end, args.n_samples)}"
    noise_tag = f"{int(round(args.noise * 1000))}milli"
    out_path = args.output_dir / f"summary_{method_tag(args)}_{noise_tag}{suffix}_{run_id}.parquet"
    df = pd.DataFrame(rows)
    df.to_parquet(out_path)
    n = len(df)
    c001 = int(df["conv_nneg1_fmax001"].sum()) if n else 0
    c005 = int(df["conv_nneg1_fmax005"].sum()) if n else 0
    print(f"Wrote {out_path} ({n} rows)")
    if n:
        print(f"  n_neg=1 & fmax<0.01: {c001}/{n} ({100*c001/n:.1f}%)")
        print(f"  n_neg=1 & fmax<0.05: {c005}/{n} ({100*c005/n:.1f}%)")


if __name__ == "__main__":
    main()
