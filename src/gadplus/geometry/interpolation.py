"""Interpolation between reactant and product geometries.

Provides linear and geodesic interpolation for generating initial
paths between endpoint structures (e.g. for NEB or path-based TS search).

The geodesic path requires the
`geodesic-interpolate <https://github.com/virtualzx-nad/geodesic-interpolate>`_
package, installed via::

    pip install "geodesic-interpolate @ git+https://github.com/virtualzx-nad/geodesic-interpolate.git"

The fallback to linear interpolation has been removed deliberately so that a
missing install surfaces immediately instead of silently degrading results.
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor

from gadplus.projection.projection import atomic_nums_to_symbols


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
    images = (1.0 - alphas[:, None, None]) * reactant + alphas[:, None, None] * product
    return images


def _resolve_symbols(
    symbols: Sequence[str] | Tensor | None,
    atomic_nums: Tensor | None,
    n_atoms: int,
) -> list[str]:
    if symbols is not None:
        if isinstance(symbols, Tensor):
            return atomic_nums_to_symbols(symbols)
        return [str(s) for s in symbols]
    if atomic_nums is not None:
        return atomic_nums_to_symbols(atomic_nums)
    raise ValueError(
        "geodesic_interpolation requires atomic symbols (`symbols=...`) or "
        "atomic numbers (`atomic_nums=...`) — the geodesic algorithm needs "
        "element identity to build internal coordinates."
    )


def geodesic_interpolation(
    reactant: Tensor,
    product: Tensor,
    n_images: int = 10,
    *,
    symbols: Sequence[str] | Tensor | None = None,
    atomic_nums: Tensor | None = None,
    scaling: float = 1.7,
    threshold: float = 3.0,
    friction: float = 1e-2,
    tol: float = 2e-3,
    max_iter: int = 15,
    micro_iter: int = 20,
    sweep: bool | None = None,
) -> Tensor:
    """Geodesic interpolation between reactant and product geometries.

    Uses the ``geodesic-interpolate`` package (Zhu et al., Martinez group)
    to generate a path that minimises internal-coordinate distortion while
    operating in Cartesian space. The package must be installed — there is
    no linear fallback.

    Args:
        reactant:    (N, 3) reactant coordinates.
        product:     (N, 3) product coordinates.
        n_images:    Number of interpolated images (including endpoints).
        symbols:     Element symbols, length N. Either this or `atomic_nums`
                     must be supplied.
        atomic_nums: Atomic numbers tensor of shape (N,). Used to derive
                     symbols if `symbols` is not given.
        scaling:     Morse potential exponential parameter (matches CLI default).
        threshold:   Distance cutoff for pair inclusion in internal coords.
        friction:    Step-size damping inside the geodesic optimiser.
        tol:         Convergence tolerance for the smoother.
        max_iter:    Maximum macro iterations for the smoother.
        micro_iter:  Maximum micro iterations for sweeping.
        sweep:       If True, run image-by-image sweeps; if False, smooth all
                     images jointly; if None, auto-pick (sweep when N > 35).

    Returns:
        (n_images, N, 3) tensor of interpolated geometries.
    """
    from geodesic_interpolate.interpolation import redistribute
    from geodesic_interpolate.geodesic import Geodesic

    if n_images < 2:
        raise ValueError("n_images must be >= 2 to include both endpoints.")

    n_atoms = reactant.shape[0]
    syms = _resolve_symbols(symbols, atomic_nums, n_atoms)
    if len(syms) != n_atoms:
        raise ValueError(
            f"Got {len(syms)} symbols but {n_atoms} atom rows in reactant."
        )

    r_np = reactant.detach().cpu().numpy()
    p_np = product.detach().cpu().numpy()
    initial_path = [r_np, p_np]

    raw = redistribute(syms, initial_path, n_images, tol=tol * 5)

    smoother = Geodesic(
        syms, raw, scaling, threshold=threshold, friction=friction,
    )
    use_sweep = (len(syms) > 35) if sweep is None else sweep
    if use_sweep:
        smoother.sweep(tol=tol, max_iter=max_iter, micro_iter=micro_iter)
    else:
        smoother.smooth(tol=tol, max_iter=max_iter)

    path = smoother.path  # list/array of (N, 3) ndarrays, length n_images
    path_tensor = torch.tensor(
        path, dtype=reactant.dtype, device=reactant.device,
    ).reshape(n_images, n_atoms, 3)
    return path_tensor
