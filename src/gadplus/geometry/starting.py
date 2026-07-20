"""Starting geometry factory for transition state search.

Selects an initial geometry from the available molecular data (TS,
reactant, product, midpoint, or noised TS) and returns the coordinates
ready for optimisation.
"""

from __future__ import annotations

from enum import Enum

import torch
from torch import Tensor

from .noise import add_gaussian_noise
from .interpolation import geodesic_interpolation


class StartingGeometry(Enum):
    """Available starting geometry types."""

    TS = "ts"
    REACTANT = "reactant"
    PRODUCT = "product"
    MIDPOINT_RT = "midpoint_rt"
    NOISED_TS = "noised_ts"
    GEODESIC = "geodesic"


def make_starting_coords(
    batch,
    method: str,
    noise_rms: float = 0.0,
    seed: int | None = None,
    n_images: int = 11,
    start_fraction: float = 0.5,
) -> Tensor:
    """Build starting coordinates from a data batch.

    Args:
        batch:          Data object with ``pos_transition``, ``pos_reactant``,
                        ``pos_product``, and ``z`` attributes.
        method:         One of ``"ts"``, ``"reactant"``, ``"product"``,
                        ``"midpoint_rt"``, ``"noised_ts"``, ``"geodesic"``.
        noise_rms:      RMS noise amplitude in Angstroms (used only for
                        ``"noised_ts"``).
        seed:           Random seed for reproducible noise generation.
        n_images:       Geodesic path length (only used for ``"geodesic"``).
        start_fraction: Where on the geodesic path to pick the starting
                        image, in [0, 1] (0 = reactant, 1 = product).

    Returns:
        Coordinate tensor with the same shape as ``batch.pos_transition``.

    Raises:
        ValueError: If *method* is not a recognised geometry type.
    """
    method = method.lower()

    if method == StartingGeometry.TS.value:
        return batch.pos_transition.clone()

    if method == StartingGeometry.REACTANT.value:
        return batch.pos_reactant.clone()

    if method == StartingGeometry.PRODUCT.value:
        return batch.pos_product.clone()

    if method == StartingGeometry.MIDPOINT_RT.value:
        return 0.5 * batch.pos_reactant + 0.5 * batch.pos_transition

    if method == StartingGeometry.NOISED_TS.value:
        coords = batch.pos_transition.clone()
        return add_gaussian_noise(coords, rms_angstrom=noise_rms, seed=seed)

    if method == StartingGeometry.GEODESIC.value:
        if not (0.0 <= start_fraction <= 1.0):
            raise ValueError(f"start_fraction must be in [0, 1], got {start_fraction}")
        if n_images < 2:
            raise ValueError(f"n_images must be >= 2, got {n_images}")
        images = geodesic_interpolation(
            batch.pos_reactant,
            batch.pos_product,
            n_images=n_images,
            atomic_nums=batch.z,
        )
        idx = int(round(start_fraction * (n_images - 1)))
        coords = images[idx]
        if noise_rms > 0.0:
            coords = add_gaussian_noise(coords, rms_angstrom=noise_rms, seed=seed)
        return coords

    raise ValueError(
        f"Unknown starting geometry method '{method}'. "
        f"Choose from: {[g.value for g in StartingGeometry]}"
    )
