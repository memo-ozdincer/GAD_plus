"""Gaussian noise utilities for molecular coordinate perturbation."""

from __future__ import annotations

import torch
from torch import Tensor


def add_gaussian_noise(
    coords: Tensor,
    rms_angstrom: float,
    seed: int | None = None,
) -> Tensor:
    """Add isotropic Gaussian noise to atomic coordinates.

    Generates noise with standard deviation equal to ``rms_angstrom`` per
    Cartesian component, giving an expected per-atom RMS displacement of
    approximately ``rms_angstrom``.

    Args:
        coords:        (N, 3) or (3N,) coordinate tensor.
        rms_angstrom:  Standard deviation of Gaussian noise in Angstroms.
        seed:          If not None, ``torch.manual_seed`` is called before
                       generating noise for reproducibility.

    Returns:
        Noisy coordinates with the same shape and device as ``coords``.
    """
    if seed is not None:
        torch.manual_seed(seed)

    noise = torch.randn_like(coords) * rms_angstrom
    return coords + noise
