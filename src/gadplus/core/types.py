"""Type definitions and protocols for GADplus.

The PredictFn protocol is the core abstraction separating algorithms from
calculator backends. All search loops and core functions accept a PredictFn
and never import HIP or any other backend directly.
"""
from __future__ import annotations

from typing import Any, Dict, Protocol

import torch


class PredictFn(Protocol):
    """Callable interface for energy/force/Hessian predictions.

    All algorithms call this to obtain energy, forces, and optionally a
    Hessian for molecular geometries. Implementations may support either
    single-geometry inputs or both single and batched inputs.

    Args:
        coords: Single geometry as (N, 3) or (3N,), or a batch as
            (B, N, 3) or (B, 3N).
        atomic_nums: (N,) atomic numbers, or optionally (B, N) for batched
            calculators that need per-system element identities.
        do_hessian: If True, compute and return the Hessian.
        require_grad: If True, returned tensors are connected to coords
            for autograd differentiation.

    Returns:
        Dict with keys "energy", "forces", and optionally "hessian".
        For single inputs, shapes are scalar, (N, 3), and (3N, 3N).
        For batched inputs, shapes are (B,), (B, N, 3), and (B, 3N, 3N).
    """

    def __call__(
        self,
        coords: torch.Tensor,
        atomic_nums: torch.Tensor,
        *,
        do_hessian: bool = True,
        require_grad: bool = False,
    ) -> Dict[str, Any]: ...


TensorDict = Dict[str, torch.Tensor]


def ensure_2d_coords(coords: torch.Tensor) -> torch.Tensor:
    """Reshape (3N,) to (N, 3) if needed. Passes (N, 3) through."""
    if coords.dim() == 1:
        return coords.reshape(-1, 3)
    return coords
