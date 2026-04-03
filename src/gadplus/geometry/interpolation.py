"""Interpolation between reactant and product geometries.

Provides linear and geodesic interpolation for generating initial
paths between endpoint structures (e.g. for NEB or path-based TS search).
"""

from __future__ import annotations

import warnings

import torch
from torch import Tensor


def linear_interpolation(
    reactant: Tensor,
    product: Tensor,
    n_images: int = 10,
) -> Tensor:
    """Linearly interpolate between reactant and product geometries.

    Args:
        reactant:  (N, 3) reactant coordinates.
        product:   (N, 3) product coordinates.
        n_images:  Number of interpolated images (including endpoints).

    Returns:
        (n_images, N, 3) tensor of interpolated geometries, with
        ``images[0] == reactant`` and ``images[-1] == product``.
    """
    if n_images < 2:
        raise ValueError("n_images must be >= 2 to include both endpoints.")

    alphas = torch.linspace(0.0, 1.0, n_images, device=reactant.device)
    # alphas[:, None, None] broadcasts over (N, 3)
    images = (1.0 - alphas[:, None, None]) * reactant + alphas[:, None, None] * product
    return images


def geodesic_interpolation(
    reactant: Tensor,
    product: Tensor,
    n_images: int = 10,
) -> Tensor:
    """Geodesic interpolation between reactant and product geometries.

    Uses the ``geodesic_interpolate`` package if available, which
    generates a path that minimises internal coordinate deviations.
    Falls back to linear interpolation with a warning if the package
    is not installed.

    Args:
        reactant:  (N, 3) reactant coordinates.
        product:   (N, 3) product coordinates.
        n_images:  Number of interpolated images (including endpoints).

    Returns:
        (n_images, N, 3) tensor of interpolated geometries.
    """
    try:
        from geodesic_interpolate import geodesic  # type: ignore[import-untyped]
    except ImportError:
        warnings.warn(
            "geodesic_interpolate package not found. "
            "Falling back to linear interpolation.",
            stacklevel=2,
        )
        return linear_interpolation(reactant, product, n_images)

    # geodesic_interpolate expects numpy arrays of shape (n_images, N*3)
    import numpy as np

    r_np = reactant.detach().cpu().numpy().reshape(1, -1)
    p_np = product.detach().cpu().numpy().reshape(1, -1)
    initial_path = np.concatenate([r_np, p_np], axis=0)

    path = geodesic.run_geodesic_interpolation(
        initial_path,
        n_images=n_images,
    )

    n_atoms = reactant.shape[0]
    path_tensor = torch.tensor(
        path, dtype=reactant.dtype, device=reactant.device
    ).reshape(n_images, n_atoms, 3)
    return path_tensor
