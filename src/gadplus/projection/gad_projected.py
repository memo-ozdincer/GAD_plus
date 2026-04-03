"""GAD dynamics with Eckart projection.

Computes the GAD direction with consistent projection of gradient, guide vector,
and output to prevent leakage into the translation/rotation null space.
"""
from __future__ import annotations

import torch

from .masses import get_mass_weights_torch
from .eckart import eckartprojection_torch


def gad_dynamics_projected_torch(
    coords: torch.Tensor,
    forces: torch.Tensor,
    v: torch.Tensor,
    atomsymbols: list[str],
    eps: float = 1e-10,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Compute GAD dynamics with consistent Eckart projection.

    Ensures that:
    1. The gradient is projected to remove TR components.
    2. The guide vector v is projected to remove TR components.
    3. The output dq/dt is projected to stay in vibrational space.

    Args:
        coords: (N, 3) Cartesian coordinates.
        forces: (N, 3) or (3N,) forces (negative gradient).
        v: (3N,) guide vector (eigenvector of projected Hessian).
        atomsymbols: List of atom symbols.
        eps: Regularization for projector construction.

    Returns:
        gad_vec: (N, 3) GAD direction in Cartesian space.
        v_proj: (3N,) projected/normalized guide vector for tracking.
        info: Dict with diagnostic info.
    """
    device = coords.device
    dtype = torch.float64

    coords_3d = coords.reshape(-1, 3).to(dtype)
    f_flat = forces.reshape(-1).to(dtype)
    v_flat = v.reshape(-1).to(dtype)
    num_atoms = coords_3d.shape[0]

    masses_t, _, sqrt_m, sqrt_m_inv = get_mass_weights_torch(
        atomsymbols, device=device, dtype=dtype,
    )

    P = eckartprojection_torch(coords_3d, masses_t, eps=eps)

    # Project gradient (forces = -gradient)
    grad_mw = -sqrt_m_inv * f_flat
    grad_mw_proj = P @ grad_mw

    # Project guide vector v (already in MW space)
    v_proj = P @ v_flat
    v_proj = v_proj / (v_proj.norm() + 1e-12)

    # GAD: dq/dt = -grad + 2(v·grad)/(v·v) * v
    v_dot_grad = torch.dot(v_proj, grad_mw_proj)
    v_dot_v = torch.dot(v_proj, v_proj)

    dq_dt_mw = -grad_mw_proj + 2.0 * (v_dot_grad / (v_dot_v + 1e-12)) * v_proj
    dq_dt_mw = P @ dq_dt_mw  # project output

    # Convert back to Cartesian
    dq_dt_cart = sqrt_m * dq_dt_mw

    gad_vec = dq_dt_cart.reshape(num_atoms, 3).to(forces.dtype)

    info = {
        "v_dot_grad": float(v_dot_grad.item()),
        "grad_norm_mw": float(grad_mw_proj.norm().item()),
        "v_norm": float(v_proj.norm().item()),
    }
    return gad_vec, v_proj.to(v.dtype), info


def project_vector_to_vibrational_torch(
    vec: torch.Tensor,
    cart_coords: torch.Tensor,
    atomsymbols: list[str],
    eps: float = 1e-10,
) -> torch.Tensor:
    """Project a Cartesian vector to remove translation/rotation components.

    The vector is mass-weighted, projected in MW space, then un-weighted.
    """
    device = vec.device
    dtype = torch.float64

    vec_flat = vec.reshape(-1).to(dtype)
    coords_3d = cart_coords.reshape(-1, 3)

    masses_t, _, sqrt_m, sqrt_m_inv = get_mass_weights_torch(
        atomsymbols, device=device, dtype=dtype,
    )

    P = eckartprojection_torch(coords_3d, masses_t, eps=eps)

    vec_mw = sqrt_m_inv * vec_flat
    vec_mw_proj = P @ vec_mw
    vec_proj = sqrt_m * vec_mw_proj

    return vec_proj.to(vec.dtype)
