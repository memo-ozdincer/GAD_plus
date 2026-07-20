"""Analytic Lennard-Jones cluster potential."""
from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations
from typing import Any

import torch

from gadplus.core.types import PredictFn

Pair = tuple[int, int]


@dataclass(frozen=True)
class LennardJonesParams:
    """Parameters for a reduced-unit Lennard-Jones potential."""

    epsilon: float = 1.0
    sigma: float = 1.0


def pair_indices(n_atoms: int) -> tuple[Pair, ...]:
    """Return all unique atom pairs for an ``n_atoms`` cluster."""

    if n_atoms < 2:
        raise ValueError("Lennard-Jones clusters require at least two atoms.")
    return tuple(combinations(range(n_atoms), 2))


def pair_distances(coords: torch.Tensor) -> torch.Tensor:
    """Return all pair distances for a cluster geometry."""

    xyz = coords.reshape(-1, 3)
    pairs = pair_indices(xyz.shape[0])
    return torch.stack([torch.linalg.vector_norm(xyz[i] - xyz[j]) for i, j in pairs])


def shortest_pair_label(coords: torch.Tensor) -> str:
    """Label the closest pair with one-indexed atom numbers."""

    xyz = coords.reshape(-1, 3)
    pairs = pair_indices(xyz.shape[0])
    distances = pair_distances(xyz)
    idx = int(torch.argmin(distances).detach().cpu().item())
    i, j = pairs[idx]
    return f"{i + 1}{j + 1}"


def lj_atomic_nums(n_atoms: int, atomic_number: int = 1, device=None) -> torch.Tensor:
    """Atomic numbers for identical LJ atoms.

    The Lennard-Jones surface itself ignores element identity. The assigned
    element supplies masses for Eckart projection and Sella's coordinate model.
    Hydrogen is the default for the LJ paper-style sweeps because it avoids the
    arbitrary argon mass scale used in the earlier exploratory branch.
    """

    return torch.full((n_atoms,), atomic_number, dtype=torch.long, device=device)


class LennardJonesPredictor(PredictFn):
    """PredictFn adapter for an analytic Lennard-Jones cluster."""

    def __init__(self, params: LennardJonesParams | None = None):
        self.params = params or LennardJonesParams()

    def energy(self, coords: torch.Tensor) -> torch.Tensor:
        """Evaluate the pairwise Lennard-Jones energy."""

        p = self.params
        distances = pair_distances(coords).clamp_min(1.0e-12)
        sr6 = (p.sigma / distances) ** 6
        return 4.0 * p.epsilon * torch.sum(sr6 * (sr6 - 1.0))

    def __call__(
        self,
        coords: torch.Tensor,
        atomic_nums: torch.Tensor,
        *,
        do_hessian: bool = True,
        require_grad: bool = False,
    ) -> dict[str, Any]:
        del atomic_nums

        n_atoms = coords.numel() // 3
        x = coords.reshape(n_atoms, 3).to(dtype=torch.float64)
        if not require_grad:
            x = x.detach()
        x = x.clone().requires_grad_(True)

        energy = self.energy(x)
        grad = torch.autograd.grad(
            energy,
            x,
            create_graph=do_hessian or require_grad,
            retain_graph=True,
        )[0]
        out: dict[str, Any] = {
            "energy": energy if require_grad else energy.detach(),
            "forces": -grad if require_grad else (-grad).detach(),
        }

        if do_hessian:
            flat = x.reshape(-1)

            def energy_flat(flat_coords: torch.Tensor) -> torch.Tensor:
                return self.energy(flat_coords.reshape(n_atoms, 3))

            hessian = torch.autograd.functional.hessian(
                energy_flat,
                flat,
                create_graph=require_grad,
                vectorize=True,
            )
            hessian = 0.5 * (hessian + hessian.transpose(0, 1))
            out["hessian"] = hessian if require_grad else hessian.detach()

        return out


def make_lj_predict_fn(params: LennardJonesParams | None = None) -> PredictFn:
    """Create a PredictFn for the Lennard-Jones cluster potential."""

    return LennardJonesPredictor(params)


def pentagonal_bipyramid_geometry(sigma: float = 1.0) -> torch.Tensor:
    """Construct the relaxed D5h global minimum of the reduced LJ7 cluster.

    The five equatorial atoms and two axial atoms have two independent
    geometric parameters.  Setting every nearest-neighbour distance to the
    pair minimum is only an approximation: the axial--axial interaction
    shifts the true force-balanced height.  These dimensionless constants are
    the stationary D5h solution for ``sigma = epsilon = 1`` and scale linearly
    with ``sigma``.
    """

    ring_radius = 0.9562063084643488 * sigma
    height = 0.5738701709721903 * sigma
    ring = []
    for idx in range(5):
        theta = math.tau * idx / 5.0
        ring.append(
            [
                ring_radius * math.cos(theta),
                ring_radius * math.sin(theta),
                0.0,
            ]
        )
    coords = torch.tensor(
        [[0.0, 0.0, height], [0.0, 0.0, -height], *ring],
        dtype=torch.float64,
    )
    return center_geometry(coords)


def random_cluster_geometry(
    n_atoms: int,
    sigma: float = 1.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Construct a loose random LJ cluster without severe pair overlaps."""

    target_min = 0.9 * 2.0 ** (1.0 / 6.0) * sigma
    radius = max(1.2 * sigma, 0.55 * target_min * n_atoms ** (1.0 / 3.0))
    coords: list[torch.Tensor] = []
    max_attempts = 5000
    for _ in range(max_attempts):
        candidate = (
            2.0 * torch.rand(3, generator=generator, dtype=torch.float64) - 1.0
        ) * radius
        if torch.linalg.vector_norm(candidate) > radius:
            continue
        if not coords:
            coords.append(candidate)
        else:
            distances = torch.stack(
                [torch.linalg.vector_norm(candidate - prev) for prev in coords]
            )
            if float(distances.min().item()) >= target_min:
                coords.append(candidate)
        if len(coords) == n_atoms:
            return center_geometry(torch.stack(coords))

    cloud = torch.randn((n_atoms, 3), generator=generator, dtype=torch.float64)
    cloud = cloud / cloud.norm(dim=1, keepdim=True).clamp_min(1.0e-12)
    scales = sigma * (
        1.0 + torch.arange(n_atoms, dtype=torch.float64) / max(n_atoms - 1, 1)
    )
    return center_geometry(cloud * scales[:, None])


def center_geometry(coords: torch.Tensor) -> torch.Tensor:
    """Translate coordinates to their centroid."""

    xyz = coords.reshape(-1, 3)
    return xyz - xyz.mean(dim=0, keepdim=True)
