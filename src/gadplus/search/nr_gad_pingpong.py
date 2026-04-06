"""NR-GAD ping-pong: state-based switching between NR minimization and GAD ascent.

Switching rule (purely geometry-based, no path history):
    - n_neg < 2:  Use GAD to navigate toward an index-1 saddle
    - n_neg >= 2: Use NR minimization to descend away from higher-order saddle

The NR step when n_neg >= 2 is PURE Newton descent on ALL modes:
    delta_x = -H^{-1} g
No spectral partitioning, no TS mode ascent. We want to minimize all
forces to escape the higher-order saddle region and drop to n_neg == 1.

Once n_neg drops to 0 or 1, GAD takes over to navigate toward the saddle
(if n_neg == 0) or to refine it (if n_neg == 1).

Everything is geometry-based. No path history, no trust radius, no RFO.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import torch

from gadplus.core.types import PredictFn
from gadplus.core.mode_tracking import pick_tracked_mode
from gadplus.core.convergence import is_ts_converged, force_mean
from gadplus.core.adaptive_dt import compute_adaptive_dt, cap_displacement, min_interatomic_distance
from gadplus.projection import vib_eig, gad_dynamics_projected, preconditioned_gad_dynamics_projected, atomic_nums_to_symbols
from gadplus.logging.trajectory import TrajectoryLogger
from .gad_search import SearchResult


@dataclass
class NRGADPingPongConfig:
    """Configuration for NR-GAD ping-pong search."""
    max_steps: int = 500
    # GAD parameters
    gad_dt: float = 0.01
    k_track: int = 0
    use_adaptive_dt: bool = False
    dt_min: float = 1e-4
    dt_max: float = 0.05
    # NR parameters (damped descent, full Hessian inversion)
    nr_max_step: float = 0.3       # Max per-component step in NR
    nr_eig_floor: float = 1e-6     # Floor for eigenvalue inversion (regularization)
    nr_damping: float = 0.2        # Global NR step damping (0 < d <= 1)
    nr_max_step_norm: float = 0.1  # Max total NR step norm (Angstrom)
    # Descent mode when n_neg >= 2: "newton", "gradient", or "preconditioned"
    descent_mode: str = "newton"
    # Shared
    max_atom_disp: float = 0.35
    min_interatomic_dist: float = 0.4
    force_threshold: float = 0.01
    purify_hessian: bool = False


def nr_minimize_step(
    grad: torch.Tensor,
    evals_vib: torch.Tensor,
    evecs_vib: torch.Tensor,
    *,
    max_step: float = 0.3,
    eig_floor: float = 1e-6,
    damping: float = 1.0,
    max_step_norm: float = 0.1,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Damped Newton descent step: delta_x = -alpha * H^{-1} g in vibrational subspace.

    All modes are minimized (descended). No TS mode ascent.
    Eigenvalues are floored at eig_floor to avoid division by near-zero.

    The step is damped in two ways:
      1. Global damping factor `damping` (0 < damping <= 1) scales the entire step.
      2. Total step norm is capped at `max_step_norm` (Angstrom).

    Args:
        grad: (D,) gradient = -forces, in same space as evecs.
        evals_vib: (M,) vibrational eigenvalues.
        evecs_vib: (D, M) vibrational eigenvectors.
        max_step: Max per-component step magnitude.
        eig_floor: Floor for |eigenvalue| to regularize near-zero modes.
        damping: Global step damping factor (0 < damping <= 1).
        max_step_norm: Max total step norm (Angstrom). Rescales if exceeded.

    Returns:
        delta_x: (D,) displacement.
        info: Dict with diagnostics.
    """
    # Project gradient onto vibrational modes
    coeffs = evecs_vib.T @ grad  # (M,)

    # Newton step: step_i = -g_i / |lambda_i| for all modes
    # Use abs(lambda) to always descend (even for negative eigenvalues)
    safe_evals = torch.clamp(evals_vib.abs(), min=eig_floor)
    step_coeffs = -coeffs / safe_evals

    # Cap individual components
    step_coeffs = torch.clamp(step_coeffs, -max_step, max_step)

    # Apply global damping
    step_coeffs = step_coeffs * damping

    delta_x = evecs_vib @ step_coeffs

    # Cap total step norm
    step_norm = float(delta_x.norm().item())
    if step_norm > max_step_norm and step_norm > 0:
        delta_x = delta_x * (max_step_norm / step_norm)
        step_norm = max_step_norm

    info = {
        "step_norm": step_norm,
        "n_modes_used": int((step_coeffs.abs() > 1e-12).sum().item()),
        "max_eig_used": float(evals_vib.abs().max().item()),
        "damping": damping,
    }
    return delta_x, info


def run_nr_gad_pingpong(
    predict_fn: PredictFn,
    coords0: torch.Tensor,
    atomic_nums: torch.Tensor,
    cfg: NRGADPingPongConfig,
    logger: Optional[TrajectoryLogger] = None,
    known_ts_coords: Optional[torch.Tensor] = None,
) -> SearchResult:
    """Run NR-GAD ping-pong to find an index-1 saddle point.

    Switching rule:
        n_neg < 2  -> GAD (navigate toward saddle)
        n_neg >= 2 -> NR minimize (escape higher-order saddle)

    Args:
        predict_fn: Energy/force/Hessian calculator.
        coords0: (N, 3) starting coordinates.
        atomic_nums: (N,) atomic numbers.
        cfg: Configuration.
        logger: Optional trajectory logger.
        known_ts_coords: Reference TS for RMSD tracking.

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
    last_eig0 = 0.0
    last_energy = 0.0
    n_gad_steps = 0
    n_nr_steps = 0

    for step in range(cfg.max_steps):
        out = predict_fn(coords, atomic_nums, do_hessian=True, require_grad=False)
        forces = out["forces"]
        hessian = out["hessian"]

        if forces.dim() == 3 and forces.shape[0] == 1:
            forces = forces[0]
        forces = forces.reshape(-1, 3)

        energy = float(out["energy"].detach().reshape(-1)[0].item()) if isinstance(out["energy"], torch.Tensor) else float(out["energy"])
        fn = force_mean(forces)

        # Vibrational eigendecomposition (always Eckart-projected)
        evals_vib, evecs_vib_3N, _ = vib_eig(
            hessian, coords, atomsymbols, purify=cfg.purify_hessian,
        )
        n_neg = int((evals_vib < 0).sum().item())
        eig0 = float(evals_vib[0].item()) if evals_vib.numel() > 0 else 0.0

        last_n_neg = n_neg
        last_force_norm = fn
        last_eig0 = eig0
        last_energy = energy

        # Decide phase: NR when n_neg >= 2, GAD when n_neg < 2
        phase = "nr" if n_neg >= 2 else "gad"

        # Log
        if logger is not None:
            dt_eff_log = cfg.gad_dt if phase == "gad" else 0.0
            logger.log_step(
                step=step, phase=phase, dt_eff=dt_eff_log,
                energy=energy, forces=forces, evals_vib=evals_vib, evecs_vib=evecs_vib_3N,
                coords=coords, coords_start=coords_start, coords_prev=coords_prev,
                v_prev=v_prev, known_ts_coords=known_ts_coords,
            )

        # Convergence check
        if is_ts_converged(n_neg, fn, cfg.force_threshold):
            wall_time = time.time() - t_start
            if logger is not None:
                logger.flush()
            return SearchResult(
                converged=True, converged_step=step, total_steps=step + 1,
                final_coords=coords.detach().cpu(), final_energy=energy,
                final_n_neg=n_neg, final_force_norm=fn, final_eig0=eig0,
                wall_time_s=wall_time,
            )

        coords_prev = coords.clone()

        if phase == "nr":
            n_nr_steps += 1

            if cfg.descent_mode == "gradient":
                # B1: Plain gradient descent — just follow forces
                step_disp = cfg.gad_dt * forces
                step_disp = cap_displacement(step_disp, cfg.max_atom_disp)
            elif cfg.descent_mode == "preconditioned":
                # Preconditioned descent: dt · |H|⁻¹ · F (gad_blend_weight=0)
                # Uses same machinery as preconditioned GAD, but w=0 → pure descent
                n_evecs = evecs_vib_3N.shape[1]
                k_eff = min(cfg.k_track, n_evecs) if cfg.k_track > 0 else n_evecs
                V_cand = evecs_vib_3N[:, :max(k_eff, 1)].to(device=forces.device, dtype=forces.dtype)
                v_prev_local = (
                    v_prev.to(device=forces.device, dtype=forces.dtype).reshape(-1)
                    if v_prev is not None else None
                )
                v, _, _ = pick_tracked_mode(V_cand, v_prev_local, k=cfg.k_track)
                gad_vec, v_proj, _ = preconditioned_gad_dynamics_projected(
                    coords=coords, forces=forces, v=v, atomsymbols=atomsymbols,
                    evals_vib=evals_vib, evecs_vib_3N=evecs_vib_3N,
                    eig_floor=cfg.nr_eig_floor, gad_blend_weight=0.0,
                )
                v_prev = v_proj.detach().clone().reshape(-1)
                step_disp = cfg.gad_dt * gad_vec
                step_disp = cap_displacement(step_disp, cfg.max_atom_disp)
            else:
                # Default: Newton descent on all modes
                grad = -forces.reshape(-1).to(evecs_vib_3N.dtype)
                delta_x, _info = nr_minimize_step(
                    grad, evals_vib, evecs_vib_3N,
                    max_step=cfg.nr_max_step,
                    eig_floor=cfg.nr_eig_floor,
                    damping=cfg.nr_damping,
                    max_step_norm=cfg.nr_max_step_norm,
                )
                step_disp = delta_x.reshape(-1, 3).to(coords.dtype)
                step_disp = cap_displacement(step_disp, cfg.max_atom_disp)
        else:
            # GAD navigation (projected)
            n_gad_steps += 1
            n_evecs = evecs_vib_3N.shape[1]
            k_eff = min(cfg.k_track, n_evecs) if cfg.k_track > 0 else n_evecs
            V_cand = evecs_vib_3N[:, :max(k_eff, 1)].to(device=forces.device, dtype=forces.dtype)
            v_prev_local = (
                v_prev.to(device=forces.device, dtype=forces.dtype).reshape(-1)
                if v_prev is not None else None
            )
            v, _, _ = pick_tracked_mode(V_cand, v_prev_local, k=cfg.k_track)

            if cfg.descent_mode == "preconditioned":
                # Preconditioned GAD (w=1) for consistency with precond descent phase
                gad_vec, v_proj, _ = preconditioned_gad_dynamics_projected(
                    coords=coords, forces=forces, v=v, atomsymbols=atomsymbols,
                    evals_vib=evals_vib, evecs_vib_3N=evecs_vib_3N,
                    eig_floor=cfg.nr_eig_floor, gad_blend_weight=1.0,
                )
            else:
                gad_vec, v_proj, _ = gad_dynamics_projected(
                    coords=coords, forces=forces, v=v, atomsymbols=atomsymbols,
                )
            v_prev = v_proj.detach().clone().reshape(-1)

            # Adaptive or fixed dt
            if cfg.use_adaptive_dt:
                dt_eff = compute_adaptive_dt(
                    cfg.gad_dt, cfg.dt_min, cfg.dt_max, "eigenvalue_clamped", eig0,
                )
            else:
                dt_eff = cfg.gad_dt

            step_disp = dt_eff * gad_vec
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
        final_eig0=last_eig0, wall_time_s=wall_time,
    )
