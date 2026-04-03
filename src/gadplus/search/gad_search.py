"""GAD search loop — the bottom-up base.

This is the main search loop that runs GAD dynamics from a starting geometry
to find an index-1 saddle point (transition state). Features are controlled
by config flags so the same loop supports all feature levels:

    Level 0: Pure GAD (raw Hessian, fixed dt, Euler steps)
    Level 1: + Mode tracking (k_track > 0)
    Level 2: + Eckart projection (use_projection=True)
    Level 3: + Adaptive dt (use_adaptive_dt=True)

Convergence: n_neg == 1 AND force_norm < force_threshold.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch

from gadplus.core.types import PredictFn
from gadplus.core.gad import compute_gad_vector_tracked, prepare_hessian
from gadplus.core.mode_tracking import pick_tracked_mode
from gadplus.core.convergence import is_ts_converged, force_mean
from gadplus.core.adaptive_dt import compute_adaptive_dt, cap_displacement, min_interatomic_distance
from gadplus.projection.hessian import vib_eig
from gadplus.projection.gad_projected import gad_dynamics_projected_torch
from gadplus.projection.masses import atomic_nums_to_symbols
from gadplus.logging.trajectory import TrajectoryLogger


@dataclass
class GADSearchConfig:
    """Configuration for GAD search."""
    n_steps: int = 300
    dt: float = 0.005
    k_track: int = 8
    beta: float = 1.0
    use_projection: bool = False
    use_adaptive_dt: bool = False
    dt_min: float = 1e-5
    dt_max: float = 0.1
    dt_adaptation: str = "eigenvalue_clamped"
    max_atom_disp: float = 0.35
    min_interatomic_dist: float = 0.4
    force_threshold: float = 0.01
    purify_hessian: bool = False


@dataclass
class SearchResult:
    """Result of a GAD search."""
    converged: bool
    converged_step: Optional[int]
    total_steps: int
    final_coords: torch.Tensor
    final_energy: float
    final_n_neg: int
    final_force_norm: float
    final_eig0: float
    wall_time_s: float
    failure_type: Optional[str] = None


def run_gad_search(
    predict_fn: PredictFn,
    coords0: torch.Tensor,
    atomic_nums: torch.Tensor,
    cfg: GADSearchConfig,
    logger: Optional[TrajectoryLogger] = None,
    known_ts_coords: Optional[torch.Tensor] = None,
) -> SearchResult:
    """Run GAD dynamics to find an index-1 saddle point.

    Args:
        predict_fn: Energy/force/Hessian calculator.
        coords0: (N, 3) starting coordinates.
        atomic_nums: (N,) atomic numbers.
        cfg: Search configuration.
        logger: Optional trajectory logger for Parquet output.
        known_ts_coords: (N, 3) reference TS coords for RMSD tracking.

    Returns:
        SearchResult with convergence status, final geometry, and timing.
    """
    coords = coords0.detach().clone().to(torch.float32).reshape(-1, 3)
    coords_start = coords.clone()
    coords_prev = coords.clone()
    atomsymbols = atomic_nums_to_symbols(atomic_nums)

    v_prev: Optional[torch.Tensor] = None
    t_start = time.time()

    # Track last eigenvalues for result
    last_n_neg = 0
    last_force_norm = float("inf")
    last_eig0 = 0.0
    last_energy = 0.0

    for step in range(cfg.n_steps):
        step_t0 = time.time()

        # Evaluate energy, forces, Hessian
        out = predict_fn(coords, atomic_nums, do_hessian=True, require_grad=False)
        forces = out["forces"]
        hessian = out["hessian"]

        if forces.dim() == 3 and forces.shape[0] == 1:
            forces = forces[0]
        forces = forces.reshape(-1, 3)

        energy = float(out["energy"].detach().reshape(-1)[0].item()) if isinstance(out["energy"], torch.Tensor) else float(out["energy"])
        fn = force_mean(forces)

        # Vibrational eigendecomposition
        if cfg.use_projection:
            evals_vib, evecs_vib_3N, Q_vib = vib_eig(
                hessian, coords, atomsymbols, purify=cfg.purify_hessian,
            )
        else:
            num_atoms = int(forces.shape[0])
            hess = prepare_hessian(hessian, num_atoms)
            evals_vib, evecs_vib_3N = torch.linalg.eigh(hess)

        n_neg = int((evals_vib < 0).sum().item())
        eig0 = float(evals_vib[0].item()) if evals_vib.numel() > 0 else 0.0
        eig1 = float(evals_vib[1].item()) if evals_vib.numel() > 1 else 0.0

        last_n_neg = n_neg
        last_force_norm = fn
        last_eig0 = eig0
        last_energy = energy

        # Log step
        if logger is not None:
            dt_eff_for_log = cfg.dt  # will be overwritten if adaptive
            logger.log_step(
                step=step,
                phase="gad",
                dt_eff=dt_eff_for_log,
                energy=energy,
                forces=forces,
                evals_vib=evals_vib,
                evecs_vib=evecs_vib_3N,
                coords=coords,
                coords_start=coords_start,
                coords_prev=coords_prev,
                v_prev=v_prev,
                known_ts_coords=known_ts_coords,
                grad=-forces.reshape(-1) if not cfg.use_projection else None,
            )

        # Convergence check
        if is_ts_converged(n_neg, fn, cfg.force_threshold):
            wall_time = time.time() - t_start
            if logger is not None:
                logger.flush()
            return SearchResult(
                converged=True,
                converged_step=step,
                total_steps=step + 1,
                final_coords=coords.detach().cpu(),
                final_energy=energy,
                final_n_neg=n_neg,
                final_force_norm=fn,
                final_eig0=eig0,
                wall_time_s=wall_time,
            )

        # Compute GAD direction
        if cfg.use_projection:
            # Mode tracking on projected eigenvectors
            k_track = min(cfg.k_track, evecs_vib_3N.shape[1])
            V_cand = evecs_vib_3N[:, :k_track].to(device=forces.device, dtype=forces.dtype)
            v_prev_local = (
                v_prev.to(device=forces.device, dtype=forces.dtype).reshape(-1)
                if v_prev is not None else None
            )
            v, _idx, _overlap = pick_tracked_mode(V_cand, v_prev_local, k=k_track)

            gad_vec, v_proj, _info = gad_dynamics_projected_torch(
                coords=coords, forces=forces, v=v, atomsymbols=atomsymbols,
            )
            v_prev = v_proj.detach().clone().reshape(-1)
        else:
            # Raw GAD on unprojected Hessian
            gad_vec, v_next, _info = compute_gad_vector_tracked(
                forces, hessian, v_prev, k_track=cfg.k_track, beta=cfg.beta,
            )
            v_prev = v_next

        # Adaptive timestep
        if cfg.use_adaptive_dt:
            dt_eff = compute_adaptive_dt(
                cfg.dt, cfg.dt_min, cfg.dt_max, cfg.dt_adaptation, eig0,
            )
        else:
            dt_eff = cfg.dt

        # Take step with displacement capping
        step_disp = dt_eff * gad_vec
        step_disp = cap_displacement(step_disp, cfg.max_atom_disp)

        coords_prev = coords.clone()
        new_coords = coords + step_disp

        # Interatomic distance safety
        if cfg.min_interatomic_dist > 0 and min_interatomic_distance(new_coords) < cfg.min_interatomic_dist:
            step_disp = step_disp * 0.5
            new_coords = coords + step_disp

        coords = new_coords.detach()

    # Did not converge
    wall_time = time.time() - t_start
    if logger is not None:
        logger.flush()

    return SearchResult(
        converged=False,
        converged_step=None,
        total_steps=cfg.n_steps,
        final_coords=coords.detach().cpu(),
        final_energy=last_energy,
        final_n_neg=last_n_neg,
        final_force_norm=last_force_norm,
        final_eig0=last_eig0,
        wall_time_s=wall_time,
    )
