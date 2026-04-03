"""Newton-Raphson step for transition state refinement.

Spectral-partitioned NR: maximize along the lowest eigenvector (the TS mode)
while minimizing along all other modes. No RFO, no trust radius — pure
first-principles NR in the vibrational subspace.

For a geometry near a saddle point:
    - Along v₁ (TS mode, λ₁ < 0): step UPHILL → step_1 = -g₁/|λ₁| (maximize)
    - Along v_i (i > 1, λ_i > 0): step DOWNHILL → step_i = -g_i/λ_i (minimize)

This is the P-RFO approach without the rational function shift — the simplest
possible second-order TS refinement.
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch


def nr_ts_step(
    grad: torch.Tensor,
    evals_vib: torch.Tensor,
    evecs_vib: torch.Tensor,
    *,
    max_step_component: float = 0.3,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute a spectral-partitioned NR step for TS refinement.

    Ascends along the lowest eigenmode (TS mode) and descends along all
    others. Step components are capped to prevent overshooting.

    Args:
        grad: (D,) gradient vector (negative of forces, in vibrational subspace
            or full Cartesian — must match evecs dimensionality).
        evals_vib: (M,) vibrational eigenvalues, ascending.
        evecs_vib: (D, M) corresponding eigenvectors (columns).
        max_step_component: Maximum magnitude of any single mode's contribution.

    Returns:
        delta_x: (D,) displacement step.
        info: Dict with diagnostic metrics.
    """
    coeffs = evecs_vib.T @ grad  # (M,) projections of gradient onto modes

    # Build step: for each mode, step_i = -g_i / lambda_i
    # But: TS mode (index 0, negative eigenvalue) → invert sign → ascend
    # All other modes → standard Newton descent
    step_coeffs = torch.zeros_like(coeffs)

    for i in range(len(evals_vib)):
        lam = float(evals_vib[i].item())
        g_i = float(coeffs[i].item())

        if abs(lam) < 1e-10:
            continue  # Skip near-zero modes

        if i == 0 and lam < 0:
            # TS mode: ascend (maximize energy along this direction)
            # step = +g_i / |λ_i| (positive = uphill)
            raw = g_i / abs(lam)
        else:
            # All other modes: descend (minimize energy)
            # step = -g_i / λ_i
            raw = -g_i / lam

        # Cap individual mode contributions
        raw = max(-max_step_component, min(raw, max_step_component))
        step_coeffs[i] = raw

    delta_x = evecs_vib @ step_coeffs

    info = {
        "step_norm": float(delta_x.norm().item()),
        "ts_mode_coeff": float(step_coeffs[0].item()) if len(step_coeffs) > 0 else 0.0,
        "n_active_modes": int((step_coeffs.abs() > 1e-12).sum().item()),
    }
    return delta_x, info
