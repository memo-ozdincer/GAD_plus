"""RFO-GAD: Rational Function Optimization with GAD ascent direction.

RFO solves a secular equation for the optimal Hessian shift mu:
    sum_i c_i^2 / (lambda_i - mu) + mu = 0,   mu < lambda_min

where c_i = g . v_i (gradient projected onto eigenvector i).

The step for each mode:
    h_i = -c_i / (lambda_i - mu)

For GAD: flip the sign on mode 1 (ascend along TS eigenvector):
    h_1 = +c_1 / (|lambda_1| + |mu|)
    h_i = -c_i / (lambda_i - mu)   for i > 1

Fully differentiable via implicit function theorem on the secular equation.
No trust radius, no PLS, no accept/reject.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import torch

from gadplus.core.types import PredictFn
from gadplus.core.mode_tracking import pick_tracked_mode
from gadplus.core.convergence import (
    is_ts_converged,
    force_mean,
    force_max,
    force_value_from_criterion,
)
from gadplus.core.adaptive_dt import cap_displacement, min_interatomic_distance
from gadplus.projection import (
    vib_eig, atomic_nums_to_symbols,
    get_mass_weights,
)
from gadplus.projection.projection import _eckart_projector
from gadplus.logging.trajectory import TrajectoryLogger
from .gad_search import SearchResult


@dataclass
class RFOGADConfig:
    """Configuration for RFO-GAD search."""
    n_steps: int = 1000
    dt: float = 0.005        # Overall step scaling
    max_atom_disp: float = 0.35
    min_interatomic_dist: float = 0.4
    force_threshold: float = 0.01
    force_criterion: str = "fmax"
    k_track: int = 0
    purify_hessian: bool = False
    rfo_max_iter: int = 20   # Max Newton iterations for secular equation
    rfo_tol: float = 1e-8    # Convergence tolerance for secular equation
    eig_floor: float = 1e-4  # Floor for eigenvalue regularization


def _solve_rfo_secular(
    evals: torch.Tensor,
    coeffs: torch.Tensor,
    max_iter: int = 20,
    tol: float = 1e-8,
) -> float:
    """Solve the RFO secular equation for the shift mu.

    f(mu) = sum_i c_i^2 / (lambda_i - mu) + mu = 0

    We want mu < lambda_min (for minimization). Newton iteration on f(mu).

    Args:
        evals: (M,) eigenvalues, ascending.
        coeffs: (M,) gradient projections c_i = g . v_i.
        max_iter: Max Newton iterations.
        tol: Convergence tolerance.

    Returns:
        mu: The optimal shift.
    """
    evals_f = evals.to(torch.float64)
    coeffs_f = coeffs.to(torch.float64)
    c2 = coeffs_f ** 2

    # Start below the smallest eigenvalue
    lam_min = float(evals_f[0].item())
    # Initial guess: well below lambda_min
    mu = lam_min - 1.0

    for _ in range(max_iter):
        denom = evals_f - mu  # (M,)
        # Avoid division by zero
        safe_denom = torch.where(denom.abs() < 1e-12, torch.full_like(denom, 1e-12), denom)

        f_val = (c2 / safe_denom).sum() + mu
        f_prime = (c2 / safe_denom ** 2).sum() + 1.0

        if abs(float(f_prime)) < 1e-20:
            break

        delta = float(f_val / f_prime)
        mu = mu - delta

        # Keep mu below lambda_min
        if mu >= lam_min:
            mu = lam_min - abs(lam_min) * 0.1 - 0.01

        if abs(delta) < tol:
            break

    return mu


def rfo_gad_step(
    forces: torch.Tensor,
    evals_vib: torch.Tensor,
    evecs_vib_3N: torch.Tensor,
    coords: torch.Tensor,
    atomsymbols: list[str],
    dt: float,
    rfo_max_iter: int = 20,
    rfo_tol: float = 1e-8,
    eig_floor: float = 1e-4,
) -> tuple[torch.Tensor, dict]:
    """Compute RFO-GAD step direction.

    Mode 0 (lowest, TS mode): ascend with GAD sign flip.
    All other modes: descend with RFO shift.
    """
    device = coords.device
    coords_3d = coords.reshape(-1, 3).to(torch.float64)
    f_flat = forces.reshape(-1).to(torch.float64)
    num_atoms = coords_3d.shape[0]

    masses, _, sqrt_m, sqrt_m_inv = get_mass_weights(atomsymbols, device=device)
    P = _eckart_projector(coords_3d, masses)

    # Gradient in MW vibrational space
    grad_mw = P @ (-sqrt_m_inv * f_flat)

    evecs = evecs_vib_3N.to(torch.float64)
    evals = evals_vib.to(torch.float64)

    # Project gradient onto vibrational eigenvectors
    coeffs = evecs.T @ grad_mw  # (M,)

    # Solve RFO secular equation for descent modes (i > 0)
    if evals.numel() > 1:
        evals_desc = evals[1:]
        coeffs_desc = coeffs[1:]
        mu = _solve_rfo_secular(evals_desc, coeffs_desc, rfo_max_iter, rfo_tol)
    else:
        mu = 0.0

    # Compute step coefficients
    step_coeffs = torch.zeros_like(coeffs)

    # Mode 0 (TS mode): ASCEND — flip sign, use |lambda_0| + |mu|
    lam0 = float(evals[0].item())
    denom0 = max(abs(lam0) + abs(mu), eig_floor)
    step_coeffs[0] = coeffs[0] / denom0  # ascend: +c/|lam| (not -c/lam)

    # Modes i > 0: DESCEND with RFO shift
    if evals.numel() > 1:
        denom_desc = evals[1:] - mu
        safe_denom = torch.where(
            denom_desc.abs() < eig_floor,
            torch.full_like(denom_desc, eig_floor) * torch.sign(denom_desc + 1e-20),
            denom_desc,
        )
        step_coeffs[1:] = -coeffs[1:] / safe_denom

    # Reconstruct step in MW vibrational space
    dq = evecs @ step_coeffs

    # Project to ensure vibrational subspace
    dq = P @ dq

    # Back to Cartesian
    step_cart = (sqrt_m * dq).reshape(num_atoms, 3).to(forces.dtype) * dt

    info = {
        "rfo_mu": mu,
        "lam0": lam0,
        "step_norm": float(step_cart.norm().item()),
    }
    return step_cart, info


def run_rfo_gad(
    predict_fn: PredictFn,
    coords0: torch.Tensor,
    atomic_nums: torch.Tensor,
    cfg: RFOGADConfig,
    logger: Optional[TrajectoryLogger] = None,
    known_ts_coords: Optional[torch.Tensor] = None,
) -> SearchResult:
    """Run RFO-GAD to find an index-1 saddle point."""
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

    for step in range(cfg.n_steps):
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

        if logger is not None:
            logger.log_step(
                step=step, phase="rfo_gad", dt_eff=cfg.dt,
                energy=energy, forces=forces, evals_vib=evals_vib, evecs_vib=evecs_vib_3N,
                coords=coords, coords_start=coords_start, coords_prev=coords_prev,
                v_prev=v_prev, known_ts_coords=known_ts_coords,
            )

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

        # RFO-GAD step
        step_disp, _info = rfo_gad_step(
            forces, evals_vib, evecs_vib_3N, coords, atomsymbols,
            dt=cfg.dt, rfo_max_iter=cfg.rfo_max_iter, rfo_tol=cfg.rfo_tol,
            eig_floor=cfg.eig_floor,
        )
        step_disp = cap_displacement(step_disp, cfg.max_atom_disp)

        # Track mode for logging
        n_evecs = evecs_vib_3N.shape[1]
        k_eff = min(cfg.k_track, n_evecs) if cfg.k_track > 0 else n_evecs
        V_cand = evecs_vib_3N[:, :max(k_eff, 1)].to(device=forces.device, dtype=forces.dtype)
        v_prev_local = (
            v_prev.to(device=forces.device, dtype=forces.dtype).reshape(-1)
            if v_prev is not None else None
        )
        v, _, _ = pick_tracked_mode(V_cand, v_prev_local, k=cfg.k_track)
        v_prev = v.detach().clone().reshape(-1)

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
        final_force_max=last_force_max,
        final_eig0=last_eig0, wall_time_s=wall_time,
    )
