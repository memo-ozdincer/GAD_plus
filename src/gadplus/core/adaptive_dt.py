"""Adaptive timestep strategies for GAD dynamics.

All strategies are state-based (no path history), making them compatible
with diffusion model integration.

Strategies:
    - none: Fixed timestep.
    - eigenvalue_clamped: dt ~ 1/clamp(|λ₀|, 0.01, 100). Small |λ| → large dt,
      large |λ| → small dt. Best general strategy (93-100% TS rate in benchmarks).
    - displacement_cap: Cap per-atom displacement to max_atom_disp.
"""
from __future__ import annotations

import math

import torch


def compute_adaptive_dt(
    dt_base: float,
    dt_min: float,
    dt_max: float,
    method: str,
    eig_0: float,
    eps: float = 1e-8,
) -> float:
    """Compute adaptive timestep using only current-state information.

    Args:
        dt_base: Base timestep.
        dt_min: Minimum allowed timestep.
        dt_max: Maximum allowed timestep.
        method: Adaptation method ("none", "eigenvalue_clamped").
        eig_0: Lowest vibrational eigenvalue.
        eps: Small constant to prevent division by zero.

    Returns:
        Effective timestep.
    """
    if method == "none":
        return dt_base

    if method == "eigenvalue_clamped":
        lam = min(max(abs(eig_0), 1e-2), 1e2)
        dt_eff = dt_base / (lam + eps)
    else:
        dt_eff = dt_base

    return float(max(dt_min, min(dt_eff, dt_max)))


def cap_displacement(
    step_disp: torch.Tensor,
    max_atom_disp: float,
) -> torch.Tensor:
    """Cap per-atom displacement to a maximum value.

    Args:
        step_disp: (N, 3) or (3N,) displacement vector.
        max_atom_disp: Maximum per-atom displacement in Angstrom.

    Returns:
        Capped displacement with same shape as input.
    """
    disp_3d = step_disp.reshape(-1, 3)
    max_actual = float(disp_3d.norm(dim=1).max().item())
    if max_actual > max_atom_disp and max_actual > 0:
        disp_3d = disp_3d * (max_atom_disp / max_actual)
    return disp_3d.reshape(step_disp.shape)


def min_interatomic_distance(coords: torch.Tensor) -> float:
    """Compute minimum interatomic distance (Angstrom).

    Args:
        coords: (N, 3) atomic coordinates.

    Returns:
        Minimum pairwise distance, or inf for single-atom systems.
    """
    c = coords.reshape(-1, 3)
    n = c.shape[0]
    if n < 2:
        return float("inf")
    diff = c.unsqueeze(0) - c.unsqueeze(1)
    dist = diff.norm(dim=2) + torch.eye(n, device=c.device, dtype=c.dtype) * 1e10
    return float(dist.min().item())
