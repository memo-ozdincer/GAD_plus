"""Gentlest Ascent Dynamics (GAD) — core algorithm.

GAD inverts the force component along the lowest Hessian eigenvector:
    F_GAD = F + 2(F · v₁)v₁      (where F = -∇E are forces)

This makes the dynamics ascend along the reaction coordinate (v₁) while
descending along all other modes, converging to an index-1 saddle point.

Reference: E W, Zhou 2011, "The Gentlest Ascent Dynamics".
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from .types import PredictFn, ensure_2d_coords
from .mode_tracking import pick_tracked_mode


def prepare_hessian(hess: torch.Tensor, num_atoms: int) -> torch.Tensor:
    """Reshape raw Hessian output to (3N, 3N)."""
    if hess.dim() == 1:
        side = int(hess.numel() ** 0.5)
        return hess.view(side, side)
    if hess.dim() == 3 and hess.shape[0] == 1:
        hess = hess[0]
    if hess.dim() > 2:
        return hess.reshape(3 * num_atoms, 3 * num_atoms)
    return hess


def compute_gad_vector_tracked(
    forces: torch.Tensor,
    hessian: torch.Tensor,
    v_prev: torch.Tensor | None,
    *,
    k_track: int = 8,
    beta: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Compute the GAD direction with mode tracking.

    Args:
        forces: (N, 3) or (1, N, 3) atomic forces.
        hessian: Raw (3N, 3N) Hessian (or reshaped equivalent).
        v_prev: Previous tracked mode (3N,) or None for first step.
        k_track: Search among lowest k eigenvectors for tracking.
        beta: Smoothing factor in [0, 1]. 1.0 = no smoothing.

    Returns:
        gad_vec: (N, 3) GAD displacement direction.
        v_next: (3N,) tracked mode for next step.
        info: Dict with "mode_overlap" and "mode_index".
    """
    if forces.dim() == 3 and forces.shape[0] == 1:
        forces = forces[0]
    forces = forces.reshape(-1, 3)
    num_atoms = int(forces.shape[0])

    hess = prepare_hessian(hessian, num_atoms)
    _, evecs = torch.linalg.eigh(hess)

    if v_prev is not None:
        v_prev = v_prev.to(device=evecs.device, dtype=evecs.dtype).reshape(-1)

    v_new, j, overlap = pick_tracked_mode(evecs, v_prev, k=k_track)

    # Optional smoothing
    if v_prev is not None and float(beta) < 1.0:
        v = (1.0 - float(beta)) * v_prev + float(beta) * v_new
        v = v / (v.norm() + 1e-12)
    else:
        v = v_new

    v = v.to(device=forces.device, dtype=forces.dtype)
    v_next = v.detach().clone().reshape(-1)

    # GAD formula: F_GAD = F + 2(F · v)v  (note: F = -grad, so -F · v = grad · v)
    f_flat = forces.reshape(-1)
    gad_flat = f_flat + 2.0 * torch.dot(-f_flat, v) * v
    gad_vec = gad_flat.view(num_atoms, 3)

    info = {
        "mode_overlap": float(overlap),
        "mode_index": float(j),
    }
    return gad_vec, v_next, info


def compute_gad_vector(forces: torch.Tensor, hessian: torch.Tensor) -> torch.Tensor:
    """Compute GAD direction without mode tracking (uses lowest eigenvector)."""
    gad_vec, _, _ = compute_gad_vector_tracked(forces, hessian, None)
    return gad_vec


def gad_euler_step(
    predict_fn: PredictFn,
    coords: torch.Tensor,
    atomic_nums: torch.Tensor,
    *,
    dt: float,
    out: Optional[Dict[str, Any]] = None,
    v_prev: torch.Tensor | None = None,
    k_track: int = 8,
    beta: float = 1.0,
) -> Dict[str, Any]:
    """Take a single Euler step of GAD dynamics.

    Args:
        predict_fn: Energy/force/Hessian calculator.
        coords: (N, 3) current coordinates.
        atomic_nums: (N,) atomic numbers.
        dt: Timestep size.
        out: Pre-computed predict_fn output (avoids redundant evaluation).
        v_prev: Previous tracked mode for continuity.
        k_track: Mode tracking window size.
        beta: Mode smoothing factor.

    Returns:
        Dict with "new_coords", "gad_vec", "out", "v_next",
        "mode_overlap", "mode_index".
    """
    coords0 = ensure_2d_coords(coords)
    if out is None:
        out = predict_fn(coords0, atomic_nums, do_hessian=True, require_grad=False)

    gad_vec, v_next, info = compute_gad_vector_tracked(
        out["forces"], out["hessian"], v_prev, k_track=k_track, beta=beta,
    )
    new_coords = coords0 + dt * gad_vec

    return {
        "new_coords": new_coords,
        "gad_vec": gad_vec,
        "out": out,
        "v_next": v_next,
        **info,
    }
