"""Transition state convergence criteria.

The TS convergence criterion is:
    n_neg == 1    (exactly one negative vibrational eigenvalue)
    AND
    force_norm < 0.01 eV/A    (forces are small)

This is the mathematical definition of an index-1 saddle point.
No eigenvalue product gates, no threshold relaxation, no additional filtering.
Only Eckart projection removes rigid-body modes; everything after that is real signal.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List

import torch

# Cascade thresholds for diagnostic evaluation (never used for convergence gating)
CASCADE_THRESHOLDS: List[float] = [0.0, 1e-4, 5e-4, 1e-3, 2e-3, 5e-3, 8e-3, 1e-2]


class ConvergenceStatus(Enum):
    """Status of TS convergence check."""
    NOT_CONVERGED = "not_converged"
    TS_CONVERGED = "ts_converged"       # n_neg == 1 AND force < threshold


@dataclass
class ConvergenceState:
    """Current convergence state at a given optimization step."""
    n_neg: int                  # Number of negative vibrational eigenvalues
    force_norm: float           # Mean per-atom force norm (eV/A)
    min_eval: float             # Smallest vibrational eigenvalue
    status: ConvergenceStatus = ConvergenceStatus.NOT_CONVERGED

    # Cascade: n_neg counted at each diagnostic threshold
    cascade: dict[str, int] = field(default_factory=dict)


def is_ts_converged(
    n_neg: int,
    force_norm: float,
    force_threshold: float = 0.01,
) -> bool:
    """Check if geometry is a converged transition state.

    Args:
        n_neg: Number of negative vibrational eigenvalues after Eckart projection.
        force_norm: Mean per-atom force norm in eV/A.
        force_threshold: Force convergence threshold (default 0.01 eV/A).

    Returns:
        True if n_neg == 1 AND force_norm < force_threshold.
    """
    return n_neg == 1 and force_norm < force_threshold


def compute_cascade_n_neg(evals_vib: torch.Tensor) -> dict[str, int]:
    """Count negative eigenvalues at each diagnostic threshold.

    This is purely diagnostic — never used as a convergence gate.
    Helps distinguish "optimizer found good geometry but evaluation too strict"
    from "optimizer genuinely failed".
    """
    result: dict[str, int] = {}
    for thr in CASCADE_THRESHOLDS:
        result[f"n_neg_{thr}"] = int((evals_vib < -thr).sum().item())
    return result


def compute_eigenvalue_bands(evals_vib: torch.Tensor) -> dict[str, int]:
    """Count eigenvalues in magnitude bands for spectral analysis.

    Returns counts in 5 bands: neg_large (<-0.01), neg_small (-0.01 to 0),
    near_zero (|λ|<1e-4), pos_small (0 to 0.01), pos_large (>0.01).
    """
    return {
        "band_neg_large": int((evals_vib < -0.01).sum().item()),
        "band_neg_small": int(((evals_vib >= -0.01) & (evals_vib < 0)).sum().item()),
        "band_near_zero": int((evals_vib.abs() < 1e-4).sum().item()),
        "band_pos_small": int(((evals_vib > 0) & (evals_vib <= 0.01)).sum().item()),
        "band_pos_large": int((evals_vib > 0.01).sum().item()),
    }


def force_mean(forces: torch.Tensor) -> float:
    """Compute mean per-atom force norm."""
    if forces.dim() == 3 and forces.shape[0] == 1:
        forces = forces[0]
    f = forces.reshape(-1, 3)
    return float(f.norm(dim=1).mean().item())
