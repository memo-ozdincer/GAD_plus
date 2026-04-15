"""Transition state convergence criteria.

The TS convergence criterion is always:
    n_neg == 1    (exactly one negative vibrational eigenvalue)
    AND
    force_metric < threshold

where force_metric is selected by `criterion`:
    - "fmax": max absolute Cartesian force component
    - "force_norm": mean per-atom force norm

The default is "fmax" to match Sella's convergence convention.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List

import torch

# Cascade thresholds for diagnostic evaluation (never used for convergence gating)
CASCADE_THRESHOLDS: List[float] = [0.0, 1e-4, 5e-4, 1e-3, 2e-3, 5e-3, 8e-3, 1e-2]
FORCE_CRITERIA = {"fmax", "force_norm"}


class ConvergenceStatus(Enum):
    """Status of TS convergence check."""
    NOT_CONVERGED = "not_converged"
    TS_CONVERGED = "ts_converged"       # n_neg == 1 AND force < threshold


@dataclass
class ConvergenceState:
    """Current convergence state at a given optimization step."""
    n_neg: int                  # Number of negative vibrational eigenvalues
    force_norm: float           # Mean per-atom force norm (eV/A)
    force_max: float            # Max abs Cartesian force component (eV/A)
    min_eval: float             # Smallest vibrational eigenvalue
    force_criterion: str = "fmax"
    status: ConvergenceStatus = ConvergenceStatus.NOT_CONVERGED

    # Cascade: n_neg counted at each diagnostic threshold
    cascade: dict[str, int] = field(default_factory=dict)


def is_ts_converged(
    n_neg: int,
    force_value: float,
    force_threshold: float = 0.01,
    criterion: str = "fmax",
) -> bool:
    """Check if geometry is a converged transition state.

    Args:
        n_neg: Number of negative vibrational eigenvalues after Eckart projection.
        force_value: Value of the selected force metric in eV/A.
        force_threshold: Force convergence threshold (default 0.01 eV/A).
        criterion: Force criterion used to compute `force_value`.

    Returns:
        True if n_neg == 1 AND force_value < force_threshold.
    """
    _validate_force_criterion(criterion)
    return n_neg == 1 and force_value < force_threshold


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


def force_max(forces: torch.Tensor) -> float:
    """Compute max absolute Cartesian force component (Sella-style fmax)."""
    if forces.dim() == 3 and forces.shape[0] == 1:
        forces = forces[0]
    return float(forces.reshape(-1).abs().max().item())


def force_value_from_criterion(forces: torch.Tensor, criterion: str = "fmax") -> float:
    """Compute force value based on the selected convergence criterion."""
    _validate_force_criterion(criterion)
    if criterion == "fmax":
        return force_max(forces)
    return force_mean(forces)


def _validate_force_criterion(criterion: str) -> None:
    if criterion not in FORCE_CRITERIA:
        valid = ", ".join(sorted(FORCE_CRITERIA))
        raise ValueError(f"Unknown force criterion '{criterion}'. Expected one of: {valid}")
