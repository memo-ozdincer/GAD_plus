"""NR+GAD flip-flop: state-based alternation between NR refinement and GAD navigation.

When starting from a noised geometry far from any saddle, the search alternates:
    - GAD phase: navigate toward an index-1 saddle (n_neg → 1)
    - NR phase: refine geometry at the saddle (force → 0)

Switching is purely state-based (no path history):
    - Switch to NR when n_neg == 1 (at a candidate saddle)
    - Switch back to GAD if n_neg != 1 (left the saddle basin)

The NR step uses spectral partitioning: ascend along the TS mode (lowest
eigenvector), descend along all others.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import torch

from gadplus.core.types import PredictFn
from gadplus.core.gad import compute_gad_vector_tracked
from gadplus.core.mode_tracking import pick_tracked_mode
from gadplus.core.newton_raphson import nr_ts_step
from gadplus.core.convergence import (
    is_ts_converged,
    force_mean,
    force_max,
    force_value_from_criterion,
)
from gadplus.core.adaptive_dt import cap_displacement, min_interatomic_distance
from gadplus.projection import vib_eig
from gadplus.projection import gad_dynamics_projected
from gadplus.projection import atomic_nums_to_symbols
from gadplus.logging.trajectory import TrajectoryLogger
from .gad_search import SearchResult


@dataclass
class NRGADConfig:
    """Configuration for NR+GAD flip-flop search."""
    max_steps: int = 500
    # GAD parameters
    gad_dt: float = 0.005
    k_track: int = 8
    use_projection: bool = True
    max_atom_disp: float = 0.35
    min_interatomic_dist: float = 0.4
    # NR parameters
    nr_max_step_component: float = 0.3
    # Convergence
    force_threshold: float = 0.01
    force_criterion: str = "fmax"
    purify_hessian: bool = False


def run_nr_gad_flipflop(
    predict_fn: PredictFn,
    coords0: torch.Tensor,
    atomic_nums: torch.Tensor,
    cfg: NRGADConfig,
    logger: Optional[TrajectoryLogger] = None,
    known_ts_coords: Optional[torch.Tensor] = None,
) -> SearchResult:
    """Run NR+GAD flip-flop to find and refine a transition state.

    Args:
        predict_fn: Energy/force/Hessian calculator.
        coords0: (N, 3) starting coordinates.
        atomic_nums: (N,) atomic numbers.
        cfg: Search configuration.
        logger: Optional trajectory logger.
        known_ts_coords: Reference TS coords for RMSD tracking.

    Returns:
        SearchResult with convergence status and final geometry.
    """
    coords = coords0.detach().clone().to(torch.float32).reshape(-1, 3)
    coords_start = coords.clone()
    coords_prev = coords.clone()
    atomsymbols = atomic_nums_to_symbols(atomic_nums)

    v_prev: Optional[torch.Tensor] = None
    t_start = time.time()

    last_n_neg = 0
    last_force_norm = float("inf")
    last_force_max = float("inf")
    last_eig0 = 0.0
    last_energy = 0.0

    for step in range(cfg.max_steps):
        out = predict_fn(coords, atomic_nums, do_hessian=True, require_grad=False)
        forces = out["forces"]
        hessian = out["hessian"]

        if forces.dim() == 3 and forces.shape[0] == 1:
            forces = forces[0]
        forces = forces.reshape(-1, 3)

        energy = float(out["energy"].detach().reshape(-1)[0].item()) if isinstance(out["energy"], torch.Tensor) else float(out["energy"])
        fn = force_mean(forces)
        fm = force_max(forces)
        f_conv = force_value_from_criterion(forces, cfg.force_criterion)

        # Vibrational eigendecomposition
        evals_vib, evecs_vib_3N, _ = vib_eig(
            hessian, coords, atomsymbols, purify=cfg.purify_hessian,
        )
        n_neg = int((evals_vib < 0).sum().item())
        eig0 = float(evals_vib[0].item()) if evals_vib.numel() > 0 else 0.0

        last_n_neg = n_neg
        last_force_norm = fn
        last_force_max = fm
        last_eig0 = eig0
        last_energy = energy

        # Decide phase: NR when n_neg == 1 (refine), GAD otherwise (navigate)
        phase = "nr" if n_neg == 1 else "gad"

        # Log
        if logger is not None:
            logger.log_step(
                step=step, phase=phase, dt_eff=cfg.gad_dt if phase == "gad" else 0.0,
                energy=energy, forces=forces, evals_vib=evals_vib, evecs_vib=evecs_vib_3N,
                coords=coords, coords_start=coords_start, coords_prev=coords_prev,
                v_prev=v_prev, known_ts_coords=known_ts_coords,
            )

        # Convergence
        if is_ts_converged(
            n_neg,
            f_conv,
            cfg.force_threshold,
            criterion=cfg.force_criterion,
        ):
            wall_time = time.time() - t_start
            if logger is not None:
                logger.flush()
            return SearchResult(
                converged=True, converged_step=step, total_steps=step + 1,
                final_coords=coords.detach().cpu(), final_energy=energy,
                final_n_neg=n_neg, final_force_norm=fn, final_force_max=fm, final_eig0=eig0,
                wall_time_s=wall_time,
            )

        coords_prev = coords.clone()

        if phase == "nr":
            # NR refinement: spectral-partitioned step
            grad = -forces.reshape(-1)
            delta_x, _info = nr_ts_step(
                grad, evals_vib, evecs_vib_3N,
                max_step_component=cfg.nr_max_step_component,
            )
            step_disp = delta_x.reshape(-1, 3).to(coords.dtype)
            step_disp = cap_displacement(step_disp, cfg.max_atom_disp)
        else:
            # GAD navigation
            if cfg.use_projection:
                k_track = min(cfg.k_track, evecs_vib_3N.shape[1])
                V_cand = evecs_vib_3N[:, :k_track].to(device=forces.device, dtype=forces.dtype)
                v_prev_local = (
                    v_prev.to(device=forces.device, dtype=forces.dtype).reshape(-1)
                    if v_prev is not None else None
                )
                v, _, _ = pick_tracked_mode(V_cand, v_prev_local, k=k_track)
                gad_vec, v_proj, _ = gad_dynamics_projected(
                    coords=coords, forces=forces, v=v, atomsymbols=atomsymbols,
                )
                v_prev = v_proj.detach().clone().reshape(-1)
            else:
                gad_vec, v_next, _ = compute_gad_vector_tracked(
                    forces, hessian, v_prev, k_track=cfg.k_track,
                )
                v_prev = v_next

            step_disp = cfg.gad_dt * gad_vec
            step_disp = cap_displacement(step_disp, cfg.max_atom_disp)

        new_coords = coords + step_disp
        if cfg.min_interatomic_dist > 0 and min_interatomic_distance(new_coords) < cfg.min_interatomic_dist:
            step_disp = step_disp * 0.5
            new_coords = coords + step_disp

        coords = new_coords.detach()

    wall_time = time.time() - t_start
    if logger is not None:
        logger.flush()

    return SearchResult(
        converged=False, converged_step=None, total_steps=cfg.max_steps,
        final_coords=coords.detach().cpu(), final_energy=last_energy,
        final_n_neg=last_n_neg, final_force_norm=last_force_norm,
        final_force_max=last_force_max,
        final_eig0=last_eig0, wall_time_s=wall_time,
    )
