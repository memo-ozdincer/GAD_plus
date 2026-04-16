"""Rigorous predictor-corrector IRC integrator with HIP analytical Hessian.

Hratchian-Schlegel EulerPC-inspired. Everything happens in mass-weighted
Cartesians with Eckart projection of gradient and Hessian at every step.
HIP analytical Hessian is evaluated at the current point AND at the
predictor point — exploiting the fact that HIP Hessians are O(1)-cheap
relative to forces. No BFGS approximations, no trust-region inner loops.

Per step:
  1. Compute (g_n, H_n) at q_n, mass-weighted and Eckart-projected.
  2. Adaptive arc-length step s_n clamped against sqrt(max vibrational
     eigenvalue) to avoid overshoot in stiff modes.
  3. Predictor (Euler mass-weighted steepest-descent):
       q_tilde = q_n - s_n * g_n / |g_n|
  4. Compute (g_tilde, H_tilde) at q_tilde.
  5. Corrector with midpoint Hessian curvature correction:
       g_mid = 0.5 * (g_n + g_tilde)
       H_mid = 0.5 * (H_n + H_tilde)
       g_hat = g_mid / |g_mid|
       curv  = (H_mid * g_hat  -  g_hat * (g_hat * H_mid * g_hat)) / |g_mid|
       q_{n+1} = q_n - s_n * g_hat - 0.5 * s_n^2 * curv
  6. Convergence: |g|_mw_proj < grad_tol AND all vibrational eigenvalues
     positive, held for K consecutive steps.

Initial TS kick is along the lowest *vibrational* eigenvector of the
mass-weighted Eckart-projected Hessian (not a residual TR mode).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from gadplus.projection import atomic_nums_to_symbols
from gadplus.projection.projection import (
    _eckart_projector,
    _vibrational_basis,
    get_mass_weights,
)
from gadplus.search.irc_validate import IRCResult, score_endpoints


_EIG_FLOOR = 1e-6  # Treat |eigenvalue| below this as a residual TR mode after projection


def _mw_eckart_grad_hess(
    coords_cart: torch.Tensor,
    atomic_nums: torch.Tensor,
    predict_fn,
    sqrt_m: torch.Tensor,
    sqrt_m_inv: torch.Tensor,
    masses: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Evaluate HIP at `coords_cart`, return mass-weighted + Eckart-projected
    gradient (3N,), Hessian (3N, 3N), and scalar energy.
    """
    n_atoms = coords_cart.shape[0]
    out = predict_fn(
        coords_cart, atomic_nums, do_hessian=True, require_grad=False,
    )

    energy = out["energy"]
    if isinstance(energy, torch.Tensor):
        energy = float(energy.detach().cpu().item())
    else:
        energy = float(energy)

    forces = out["forces"]
    if isinstance(forces, torch.Tensor):
        forces = forces.detach()
    else:
        forces = torch.tensor(forces)
    forces_flat = forces.reshape(-1).to(torch.float64).to(sqrt_m.device)
    g_cart = -forces_flat

    hess = out["hessian"]
    if isinstance(hess, torch.Tensor):
        H_cart = hess.detach()
    else:
        H_cart = torch.tensor(hess)
    H_cart = H_cart.reshape(3 * n_atoms, 3 * n_atoms).to(torch.float64).to(sqrt_m.device)

    # Mass-weight
    g_mw = sqrt_m_inv * g_cart
    diag_inv = torch.diag(sqrt_m_inv)
    H_mw = diag_inv @ H_cart @ diag_inv
    H_mw = 0.5 * (H_mw + H_mw.T)

    # Eckart-project
    coords_3d = coords_cart.reshape(-1, 3).to(torch.float64).to(sqrt_m.device)
    P = _eckart_projector(coords_3d, masses)
    g_mw_proj = P @ g_mw
    H_mw_proj = P @ H_mw @ P
    H_mw_proj = 0.5 * (H_mw_proj + H_mw_proj.T)

    return g_mw_proj, H_mw_proj, energy


def _vibrational_eigensystem(
    coords_cart: torch.Tensor,
    H_mw_proj: torch.Tensor,
    masses: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Diagonalize the reduced-basis (vibrational) Hessian.

    Returns ascending (eigenvalues, eigenvectors_3N) where eigenvectors_3N
    is the full 3N representation of each vibrational mode.
    """
    coords_3d = coords_cart.reshape(-1, 3).to(torch.float64).to(H_mw_proj.device)
    Q_vib, _, _ = _vibrational_basis(coords_3d, masses)
    H_red = Q_vib.T @ H_mw_proj @ Q_vib
    H_red = 0.5 * (H_red + H_red.T)
    evals, evecs_red = torch.linalg.eigh(H_red)
    evecs_3N = Q_vib @ evecs_red
    return evals, evecs_3N


def _initial_ts_kick(
    ts_coords: torch.Tensor,
    atomic_nums: torch.Tensor,
    predict_fn,
    step_size_mw: float,
    direction: str,
    sqrt_m: torch.Tensor,
    sqrt_m_inv: torch.Tensor,
    masses: torch.Tensor,
) -> tuple[torch.Tensor, dict]:
    """Step from TS along the most-negative VIBRATIONAL eigenvector in
    mass-weighted Eckart-projected space.
    """
    _g_ts, H_ts, energy_ts = _mw_eckart_grad_hess(
        ts_coords, atomic_nums, predict_fn,
        sqrt_m=sqrt_m, sqrt_m_inv=sqrt_m_inv, masses=masses,
    )
    evals_vib, evecs_vib_3N = _vibrational_eigensystem(ts_coords, H_ts, masses)

    min_eig_vib = float(evals_vib[0].cpu().item())
    v1 = evecs_vib_3N[:, 0]

    sign = 1.0 if direction == "forward" else -1.0
    q_ts_mw = sqrt_m * ts_coords.reshape(-1).to(torch.float64)
    q_next_mw = q_ts_mw + sign * step_size_mw * v1

    return q_next_mw, {
        "energy_ts": energy_ts,
        "min_vib_eig_ts": min_eig_vib,
        "n_neg_vib_ts": int((evals_vib < -_EIG_FLOOR).sum().cpu().item()),
    }


def run_irc_rigorous(
    ts_coords: torch.Tensor,
    atomic_nums: torch.Tensor,
    predict_fn,
    reactant_coords: Optional[torch.Tensor] = None,
    product_coords: Optional[torch.Tensor] = None,
    rmsd_threshold: float = 0.3,
    max_steps: int = 500,
    step_size_mw: float = 0.1,
    adaptive: bool = True,
    s_min: float = 0.01,
    s_max: float = 0.15,
    alpha_clamp: float = 0.3,
    grad_tol: float = 5e-4,
    k_hold: int = 2,
) -> IRCResult:
    """Rigorous mass-weighted predictor-corrector IRC with HIP Hessian every step.

    Args:
        ts_coords: (N, 3) TS geometry, Cartesian Angstrom.
        atomic_nums: (N,) torch tensor of atomic numbers.
        predict_fn: ``predict_fn(coords, atomic_nums, do_hessian, require_grad) -> dict``.
        reactant_coords, product_coords: reference geometries for scoring.
        rmsd_threshold: RMSD threshold for endpoint match (Angstrom).
        max_steps: Hard cap on outer steps per direction.
        step_size_mw: Nominal arc-length step in mass-weighted units (√amu·Å).
        adaptive: If True, clamp step by ``alpha_clamp / sqrt(max vib eigenvalue)``.
        s_min, s_max: Bounds on the adaptive step.
        alpha_clamp: Coefficient in the adaptive clamp; smaller = more conservative.
        grad_tol: Convergence threshold on |g|_mw (mass-weighted projected gradient norm).
        k_hold: Number of consecutive steps that must satisfy the convergence
            criterion before the loop terminates (guards against shoulder minima).

    Returns:
        IRCResult from `score_endpoints` with forward and reverse endpoints.
    """
    device = ts_coords.device
    n_atoms = ts_coords.shape[0]
    atomsymbols = atomic_nums_to_symbols(atomic_nums)
    masses, _m3, sqrt_m, sqrt_m_inv = get_mass_weights(atomsymbols, device=device)
    masses = masses.to(torch.float64)
    sqrt_m = sqrt_m.to(torch.float64)
    sqrt_m_inv = sqrt_m_inv.to(torch.float64)

    endpoints: dict[str, Optional[np.ndarray]] = {}

    for direction in ["forward", "reverse"]:
        try:
            q_mw, _ts_info = _initial_ts_kick(
                ts_coords=ts_coords,
                atomic_nums=atomic_nums,
                predict_fn=predict_fn,
                step_size_mw=step_size_mw,
                direction=direction,
                sqrt_m=sqrt_m, sqrt_m_inv=sqrt_m_inv, masses=masses,
            )

            consecutive_converged = 0
            for _step in range(max_steps):
                # Current point (mass-weighted -> Cartesian for HIP).
                x_cart = (q_mw * sqrt_m_inv).to(torch.float32).reshape(n_atoms, 3)
                g_n, H_n, _e_n = _mw_eckart_grad_hess(
                    x_cart, atomic_nums, predict_fn,
                    sqrt_m=sqrt_m, sqrt_m_inv=sqrt_m_inv, masses=masses,
                )
                g_n_norm = torch.linalg.norm(g_n)

                # Vibrational eigensystem for convergence + step-size clamp.
                evals_vib, _evecs = _vibrational_eigensystem(x_cart, H_n, masses)
                min_vib_eig = float(evals_vib[0].cpu().item())
                max_vib_eig = float(evals_vib[-1].cpu().item())

                converged_here = (
                    float(g_n_norm.cpu().item()) < grad_tol
                    and min_vib_eig > 0.0
                )
                consecutive_converged = (
                    consecutive_converged + 1 if converged_here else 0
                )
                if consecutive_converged >= k_hold:
                    break

                # Adaptive arc length.
                if adaptive and max_vib_eig > _EIG_FLOOR:
                    s_n = float(np.clip(
                        alpha_clamp / np.sqrt(max_vib_eig),
                        s_min, s_max,
                    ))
                else:
                    s_n = float(np.clip(step_size_mw, s_min, s_max))

                # Predictor: Euler SD in mass-weighted coords.
                g_hat_n = g_n / (g_n_norm + 1e-12)
                q_pred_mw = q_mw - s_n * g_hat_n

                # Compute at predictor.
                x_pred = (q_pred_mw * sqrt_m_inv).to(torch.float32).reshape(n_atoms, 3)
                g_t, H_t, _e_t = _mw_eckart_grad_hess(
                    x_pred, atomic_nums, predict_fn,
                    sqrt_m=sqrt_m, sqrt_m_inv=sqrt_m_inv, masses=masses,
                )

                # Corrector: midpoint gradient + curvature correction.
                g_mid = 0.5 * (g_n + g_t)
                H_mid = 0.5 * (H_n + H_t)
                g_mid_norm = torch.linalg.norm(g_mid) + 1e-12
                g_hat_mid = g_mid / g_mid_norm
                Hg = H_mid @ g_hat_mid
                curv = (Hg - g_hat_mid * (g_hat_mid @ Hg)) / g_mid_norm

                q_mw = q_mw - s_n * g_hat_mid - 0.5 * (s_n ** 2) * curv

            x_final = (q_mw * sqrt_m_inv).reshape(n_atoms, 3)
            endpoints[direction] = x_final.detach().cpu().numpy().astype(np.float64)
        except Exception:
            endpoints[direction] = None

    return score_endpoints(
        forward_coords=endpoints.get("forward"),
        reverse_coords=endpoints.get("reverse"),
        atomic_nums=atomic_nums,
        reactant_coords=reactant_coords,
        product_coords=product_coords,
        rmsd_threshold=rmsd_threshold,
        predict_fn=predict_fn,
    )
