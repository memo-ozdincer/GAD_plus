"""Public Lennard-Jones API with analytical derivatives."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any

import torch

from gadplus.calculator.lennard_jones_analytical import LennardJonesEnergy
from gadplus.core.types import PredictFn

Pair = tuple[int, int]
LJ_RM_FACTOR = 2.0 ** (1.0 / 6.0)


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


def lj_atomic_nums(n_atoms: int, atomic_number: int = 18, device=None) -> torch.Tensor:
    """Atomic numbers for identical LJ atoms.

    Argon is the conventional default for Lennard-Jones clusters and supplies
    identical masses for Eckart projection. The potential itself ignores
    element identity.
    """

    return torch.full((n_atoms,), atomic_number, dtype=torch.long, device=device)


def params_to_analytical_rm(sigma: float) -> float:
    """Map standard LJ sigma to the analytical module's equilibrium distance."""

    return LJ_RM_FACTOR * sigma


class LennardJonesAnalyticalPredictor(PredictFn):
    """PredictFn adapter around the vectorized analytical LJ implementation."""

    def __init__(
        self,
        n_atoms: int,
        params: LennardJonesParams | None = None,
        *,
        compile_forces: bool = False,
        compile_hessian: bool = False,
    ):
        if n_atoms < 2:
            raise ValueError("Lennard-Jones clusters require at least two atoms.")

        p = params or LennardJonesParams()
        self.n_atoms = n_atoms
        self.params = p
        self._model = LennardJonesEnergy(
            n_particles=n_atoms,
            eps=p.epsilon,
            rm=params_to_analytical_rm(p.sigma),
            oscillator=False,
            compile_forces=compile_forces,
            compile_hessian=compile_hessian,
        )

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
        if n_atoms != self.n_atoms:
            raise ValueError(
                f"Analytical LJ predictor was built for {self.n_atoms} atoms, got {n_atoms}."
            )

        positions = coords.reshape(1, n_atoms, 3).to(dtype=torch.float64)
        if not require_grad:
            positions = positions.detach()

        if require_grad:
            positions = positions.clone().requires_grad_(True)

        batch = {"positions": positions}
        energy = self._model._energy_from_positions(positions)[0]
        if require_grad:
            forces = self._model.compute_forces_autograd(batch).reshape(n_atoms, 3)
        else:
            forces = self._model.compute_forces(
                batch,
                create_graph=False,
            ).reshape(n_atoms, 3)

        out: dict[str, Any] = {
            "energy": energy if require_grad else energy.detach(),
            "forces": forces if require_grad else forces.detach(),
        }

        if do_hessian:
            if require_grad:
                hessian = self._model.compute_hessian_autograd(batch, create_graph=True)[0]
            else:
                hessian = self._model.compute_hessian(batch, create_graph=False)[0]
            hessian = 0.5 * (hessian + hessian.transpose(0, 1))
            out["hessian"] = hessian if require_grad else hessian.detach()

        return out


def make_lj_predict_fn(
    params: LennardJonesParams | None = None,
    *,
    n_atoms: int,
    compile_forces: bool = False,
    compile_hessian: bool = False,
) -> PredictFn:
    """Create an analytical PredictFn for the Lennard-Jones cluster potential.

    Args:
        params: Standard reduced-unit LJ parameters ``(epsilon, sigma)``.
        n_atoms: Cluster size.
        compile_forces: Enable ``torch.compile`` for analytical forces on CUDA.
        compile_hessian: Enable ``torch.compile`` for analytical Hessians on CUDA.
    """

    return LennardJonesAnalyticalPredictor(
        n_atoms,
        params,
        compile_forces=compile_forces,
        compile_hessian=compile_hessian,
    )


def pentagonal_bipyramid_geometry(sigma: float = 1.0) -> torch.Tensor:
    """Construct the standard LJ7 pentagonal-bipyramid starting geometry."""

    r_eq = 2.0 ** (1.0 / 6.0) * sigma
    ring_radius = r_eq / (2.0 * torch.sin(torch.tensor(torch.pi / 5.0))).item()
    height = max(r_eq**2 - ring_radius**2, 0.0) ** 0.5
    ring = []
    for idx in range(5):
        theta = 2.0 * torch.pi * idx / 5.0
        ring.append(
            [
                ring_radius * torch.cos(torch.tensor(theta)).item(),
                ring_radius * torch.sin(torch.tensor(theta)).item(),
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

    # Fall back to a larger cloud if rejection sampling could not pack the cluster.
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


__all__ = [
    "LJ_RM_FACTOR",
    "LennardJonesAnalyticalPredictor",
    "LennardJonesParams",
    "center_geometry",
    "lj_atomic_nums",
    "make_lj_predict_fn",
    "pair_distances",
    "pair_indices",
    "params_to_analytical_rm",
    "pentagonal_bipyramid_geometry",
    "random_cluster_geometry",
    "shortest_pair_label",
]
