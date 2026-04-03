"""Atomic mass data and mass-weighting utilities.

Local copy of mass data to avoid hard dependency on hip.masses at import time.
Source: IUPAC 2021 atomic weights.
"""
from __future__ import annotations

import torch

# Atomic masses in AMU, keyed by lowercase element symbol
MASS_DICT: dict[str, float] = {
    "h": 1.008, "he": 4.003,
    "li": 6.941, "be": 9.012, "b": 10.81, "c": 12.011, "n": 14.007,
    "o": 15.999, "f": 18.998, "ne": 20.180,
    "na": 22.990, "mg": 24.305, "al": 26.982, "si": 28.086, "p": 30.974,
    "s": 32.065, "cl": 35.453, "ar": 39.948,
    "k": 39.098, "ca": 40.078,
    "br": 79.904, "i": 126.904,
}

# Atomic number to element symbol
Z_TO_SYMBOL: dict[int, str] = {
    1: "H", 6: "C", 7: "N", 8: "O", 9: "F", 15: "P", 16: "S",
    17: "Cl", 35: "Br", 53: "I",
}


def atomic_nums_to_symbols(atomic_nums: torch.Tensor) -> list[str]:
    """Convert atomic number tensor to element symbol list."""
    nums = atomic_nums.detach().cpu().tolist()
    return [Z_TO_SYMBOL.get(int(z), "X") for z in nums]


def _to_torch_double(array_like, device=None):
    if isinstance(array_like, torch.Tensor):
        return array_like.to(dtype=torch.float64, device=device)
    return torch.as_tensor(array_like, dtype=torch.float64, device=device)


def get_mass_weights_torch(
    atomsymbols: list[str],
    device=None,
    dtype=torch.float64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Get mass-weighting factors for a molecule.

    Returns:
        masses: (N,) atomic masses in AMU.
        masses3d: (3N,) masses repeated for each coordinate.
        sqrt_m: (3N,) sqrt(masses) for mass-weighting.
        sqrt_m_inv: (3N,) 1/sqrt(masses) for inverse mass-weighting.
    """
    masses_t = torch.tensor(
        [MASS_DICT[atom.lower()] for atom in atomsymbols],
        dtype=dtype, device=device,
    )
    masses3d_t = masses_t.repeat_interleave(3)
    sqrt_m = torch.sqrt(masses3d_t)
    sqrt_m_inv = 1.0 / sqrt_m
    return masses_t, masses3d_t, sqrt_m, sqrt_m_inv


def mass_weigh_hessian_torch(hessian: torch.Tensor, masses3d: torch.Tensor) -> torch.Tensor:
    """Mass-weight a Hessian: M^{-1/2} H M^{-1/2}."""
    h_t = _to_torch_double(hessian, device=hessian.device)
    m_t = _to_torch_double(masses3d, device=hessian.device)
    mm_sqrt_inv = torch.diag(1.0 / torch.sqrt(m_t))
    return mm_sqrt_inv @ h_t @ mm_sqrt_inv
