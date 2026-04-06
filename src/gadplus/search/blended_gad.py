"""Lambda2-blended GAD dynamics: smooth transition between GAD and descent.

Instead of hard-switching between GAD (when n_neg<2) and descent (when n_neg>=2),
blend using a smooth sigmoid weight on the second eigenvalue lambda_2:

    weight = sigmoid(k * lambda_2)
    F_eff = weight * F_GAD + (1 - weight) * F_projected

When lambda_2 > 0 (near index-1 saddle): weight -> 1, pure GAD.
When lambda_2 < 0 (higher-order saddle): weight -> 0, pure descent.

Fully differentiable. No path history. No branching on discrete n_neg.
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
from gadplus.projection import (
    vib_eig, gad_dynamics_projected, atomic_nums_to_symbols,
    get_mass_weights,
)
from gadplus.projection.projection import _eckart_projector
from gadplus.logging.trajectory import TrajectoryLogger
from .gad_search import SearchResult


@dataclass
class BlendedGADConfig:
    """Configuration for lambda2-blended GAD search."""
    n_steps: int = 1000
    dt: float = 0.005
    k_track: int = 0
    blend_sharpness: float = 50.0  # k in sigmoid(k * lambda_2)
    max_atom_disp: float = 0.35
    min_interatomic_dist: float = 0.4
    force_threshold: float = 0.01
    purify_hessian: bool = False


def _projected_forces(coords, forces, atomsymbols):
    """Project forces into vibrational subspace (remove TR components)."""
    device = coords.device
    coords_3d = coords.reshape(-1, 3).to(torch.float64)
    f_flat = forces.reshape(-1).to(torch.float64)
    num_atoms = coords_3d.shape[0]

    masses, _, sqrt_m, sqrt_m_inv = get_mass_weights(atomsymbols, device=device)
    P = _eckart_projector(coords_3d, masses)

    # Project gradient, then convert back to force direction
    grad_mw = P @ (-sqrt_m_inv * f_flat)
    # Descent direction in Cartesian: F_proj = -sqrt_m * grad_mw_proj
    f_proj = -(sqrt_m * (P @ grad_mw)).reshape(num_atoms, 3).to(forces.dtype)
    return f_proj


def run_blended_gad(
    predict_fn: PredictFn,
    coords0: torch.Tensor,
    atomic_nums: torch.Tensor,
    cfg: BlendedGADConfig,
    logger: Optional[TrajectoryLogger] = None,
    known_ts_coords: Optional[torch.Tensor] = None,
) -> SearchResult:
    """Run lambda2-blended GAD dynamics.

    At each step:
    1. Compute vibrational eigendecomposition
    2. Get lambda_2 (second eigenvalue)
    3. weight = sigmoid(k * lambda_2)
    4. F_eff = weight * F_GAD + (1 - weight) * F_projected
    5. Step with displacement capping
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

    for step in range(cfg.n_steps):
        out = predict_fn(coords, atomic_nums, do_hessian=True, require_grad=False)
        forces = out["forces"]
        hessian = out["hessian"]

        if forces.dim() == 3 and forces.shape[0] == 1:
            forces = forces[0]
        forces = forces.reshape(-1, 3)

        energy = float(out["energy"].detach().reshape(-1)[0].item()) if isinstance(out["energy"], torch.Tensor) else float(out["energy"])
        fn = force_mean(forces)

        evals_vib, evecs_vib_3N, _ = vib_eig(
            hessian, coords, atomsymbols, purify=cfg.purify_hessian,
        )
        n_neg = int((evals_vib < 0).sum().item())
        eig0 = float(evals_vib[0].item()) if evals_vib.numel() > 0 else 0.0
        eig1 = float(evals_vib[1].item()) if evals_vib.numel() > 1 else 0.0

        last_n_neg = n_neg
        last_force_norm = fn
        last_eig0 = eig0
        last_energy = energy

        # Log
        if logger is not None:
            logger.log_step(
                step=step, phase="blended", dt_eff=cfg.dt,
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

        # Compute GAD direction
        n_evecs = evecs_vib_3N.shape[1]
        k_eff = min(cfg.k_track, n_evecs) if cfg.k_track > 0 else n_evecs
        V_cand = evecs_vib_3N[:, :max(k_eff, 1)].to(device=forces.device, dtype=forces.dtype)
        v_prev_local = (
            v_prev.to(device=forces.device, dtype=forces.dtype).reshape(-1)
            if v_prev is not None else None
        )
        v, _, _ = pick_tracked_mode(V_cand, v_prev_local, k=cfg.k_track)

        gad_vec, v_proj, _ = gad_dynamics_projected(
            coords=coords, forces=forces, v=v, atomsymbols=atomsymbols,
        )
        v_prev = v_proj.detach().clone().reshape(-1)

        # Compute projected descent direction (plain forces, Eckart-projected)
        f_proj = _projected_forces(coords, forces, atomsymbols)

        # Blend: weight = sigmoid(k * lambda_2)
        # lambda_2 = eig1 (second eigenvalue, ascending order)
        weight = torch.sigmoid(torch.tensor(cfg.blend_sharpness * eig1, dtype=forces.dtype))

        # F_eff = weight * F_GAD + (1 - weight) * F_descent
        step_disp = cfg.dt * (weight * gad_vec + (1 - weight) * f_proj)
        step_disp = cap_displacement(step_disp, cfg.max_atom_disp)

        coords_prev = coords.clone()
        new_coords = coords + step_disp

        if cfg.min_interatomic_dist > 0 and min_interatomic_distance(new_coords) < cfg.min_interatomic_dist:
            step_disp = step_disp * 0.5
            new_coords = coords + step_disp

        coords = new_coords.detach()

    wall_time = time.time() - t_start
    if logger is not None:
        logger.flush()

    return SearchResult(
        converged=False, converged_step=None, total_steps=cfg.n_steps,
        final_coords=coords.detach().cpu(), final_energy=last_energy,
        final_n_neg=last_n_neg, final_force_norm=last_force_norm,
        final_eig0=last_eig0, wall_time_s=wall_time,
    )
