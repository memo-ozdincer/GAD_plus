"""Analytic four-identical-atom EVB toy surface.

The surface is a small 3D benchmark for saddle-search algorithms. It depends
only on the six interatomic distances, so it has the usual translational and
rotational null modes and is invariant to permutations of the four atoms.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any

import torch

from gadplus.core.types import PredictFn

Pair = tuple[int, int]

X4_PAIRS: tuple[Pair, ...] = tuple(combinations(range(4), 2))


@dataclass(frozen=True)
class X4EVBParams:
    """Parameters for the symmetric one-bond X4 EVB surface."""

    r_b: float = 1.0
    r_l: float = 1.8
    D: float = 12.0
    a: float = 3.0
    k_l: float = 5.0
    A: float = 1.0e-4
    sigma: float = 0.7
    beta: float = 3.05


def pair_label(pair: Pair) -> str:
    """Return a one-indexed bond label such as ``12``."""

    i, j = pair
    return f"{i + 1}{j + 1}"


def pair_distances(coords: torch.Tensor) -> torch.Tensor:
    """Return the six X4 pair distances in ``X4_PAIRS`` order."""

    coords = coords.reshape(4, 3)
    return torch.stack(
        [torch.linalg.vector_norm(coords[i] - coords[j]) for i, j in X4_PAIRS]
    )


def classify_short_bond(coords: torch.Tensor) -> str:
    """Classify a structure by its shortest pair distance."""

    distances = pair_distances(coords)
    idx = int(torch.argmin(distances).detach().cpu().item())
    return pair_label(X4_PAIRS[idx])


def x4_atomic_nums(device=None) -> torch.Tensor:
    """Atomic numbers for four identical atoms.

    The toy potential ignores element identity. Hydrogen is used so the existing
    Eckart projection and hybrid mass tables can supply identical masses.
    """

    return torch.ones(4, dtype=torch.long, device=device)


class X4EVBPredictor(PredictFn):
    """PredictFn adapter for the X4 EVB toy surface."""

    def __init__(self, params: X4EVBParams | None = None):
        self.params = params or X4EVBParams()

    def energy(self, coords: torch.Tensor) -> torch.Tensor:
        """Evaluate the lower-envelope EVB energy."""

        p = self.params
        coords = coords.reshape(4, 3)
        distances = pair_distances(coords)
        repulsion = p.A * torch.sum((p.sigma / distances) ** 12)

        state_energies = []
        for active_idx in range(len(X4_PAIRS)):
            r_active = distances[active_idx]
            morse = p.D * (1.0 - torch.exp(-p.a * (r_active - p.r_b))) ** 2
            inactive = torch.cat(
                [distances[:active_idx], distances[active_idx + 1 :]]
            )
            loose = 0.5 * p.k_l * torch.sum((inactive - p.r_l) ** 2)
            state_energies.append(morse + loose + repulsion)

        energies = torch.stack(state_energies)
        return -torch.logsumexp(-p.beta * energies, dim=0) / p.beta

    def __call__(
        self,
        coords: torch.Tensor,
        atomic_nums: torch.Tensor,
        *,
        do_hessian: bool = True,
        require_grad: bool = False,
    ) -> dict[str, Any]:
        del atomic_nums

        x = coords.reshape(4, 3).to(dtype=torch.float64)
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
                return self.energy(flat_coords.reshape(4, 3))

            hessian = torch.autograd.functional.hessian(
                energy_flat,
                flat,
                create_graph=require_grad,
                vectorize=True,
            )
            hessian = 0.5 * (hessian + hessian.transpose(0, 1))
            out["hessian"] = hessian if require_grad else hessian.detach()

        return out


def make_x4_predict_fn(params: X4EVBParams | None = None) -> PredictFn:
    """Create a PredictFn for the X4 EVB toy surface."""

    return X4EVBPredictor(params)


def minimum_geometry(pair: Pair, params: X4EVBParams | None = None) -> torch.Tensor:
    """Construct a near-minimum geometry with one short labelled bond."""

    p = params or X4EVBParams()
    h = (p.r_l**2 - (0.5 * p.r_b) ** 2) ** 0.5
    theta = 2.0 * torch.asin(torch.tensor(p.r_l / (2.0 * h), dtype=torch.float64))
    base = torch.tensor(
        [
            [-0.5 * p.r_b, 0.0, 0.0],
            [0.5 * p.r_b, 0.0, 0.0],
            [0.0, h, 0.0],
            [0.0, h * torch.cos(theta).item(), h * torch.sin(theta).item()],
        ],
        dtype=torch.float64,
    )
    return _assign_pair_geometry(base, pair)


def adjacent_ts_guess(
    first: Pair,
    second: Pair,
    params: X4EVBParams | None = None,
    active_distance: float = 1.02,
) -> torch.Tensor:
    """Construct a symmetric adjacent-bond switch guess.

    ``first`` and ``second`` must share exactly one atom, for example
    ``(0, 1)`` and ``(0, 2)`` for a 12 <-> 13 switch.
    """

    shared = set(first).intersection(second)
    if len(shared) != 1:
        raise ValueError("Adjacent TS guesses require two bonds sharing one atom.")

    p = params or X4EVBParams()
    s = active_distance
    cos_theta = (2.0 * s**2 - p.r_l**2) / (2.0 * s**2)
    cos_theta = max(-1.0, min(1.0, cos_theta))
    sin_theta = (1.0 - cos_theta**2) ** 0.5

    atom0 = torch.zeros(3, dtype=torch.float64)
    atom1 = torch.tensor([s, 0.0, 0.0], dtype=torch.float64)
    atom2 = torch.tensor([s * cos_theta, s * sin_theta, 0.0], dtype=torch.float64)

    x = s / 2.0
    y = (0.5 * s**2 - x * s * cos_theta) / (s * sin_theta)
    z_sq = max(p.r_l**2 - x**2 - y**2, 0.0)
    atom3 = torch.tensor([x, y, z_sq**0.5], dtype=torch.float64)
    base = torch.stack([atom0, atom1, atom2, atom3])

    shared_atom = next(iter(shared))
    end_first = next(a for a in first if a != shared_atom)
    end_second = next(a for a in second if a != shared_atom)
    remaining = next(a for a in range(4) if a not in {shared_atom, end_first, end_second})

    coords = torch.empty_like(base)
    coords[shared_atom] = base[0]
    coords[end_first] = base[1]
    coords[end_second] = base[2]
    coords[remaining] = base[3]
    return center_geometry(coords)


def disjoint_ts_guess(
    first: Pair,
    second: Pair,
    params: X4EVBParams | None = None,
    active_distance: float = 1.02,
) -> torch.Tensor:
    """Construct a symmetric disjoint-bond switch guess."""

    if set(first).intersection(second):
        raise ValueError("Disjoint TS guesses require non-overlapping bonds.")

    p = params or X4EVBParams()
    s = active_distance
    z = 0.5 * s
    y = max(p.r_l**2 - 0.5 * s**2, 0.0) ** 0.5
    base = torch.tensor(
        [
            [-0.5 * s, 0.0, 0.0],
            [0.5 * s, 0.0, 0.0],
            [0.0, y, -z],
            [0.0, y, z],
        ],
        dtype=torch.float64,
    )

    coords = torch.empty_like(base)
    coords[first[0]] = base[0]
    coords[first[1]] = base[1]
    coords[second[0]] = base[2]
    coords[second[1]] = base[3]
    return center_geometry(coords)


def center_geometry(coords: torch.Tensor) -> torch.Tensor:
    """Translate coordinates to their centroid."""

    coords = coords.reshape(4, 3)
    return coords - coords.mean(dim=0, keepdim=True)


def _assign_pair_geometry(base: torch.Tensor, pair: Pair) -> torch.Tensor:
    """Assign a base 12-minimum geometry to an arbitrary labelled pair."""

    remaining = [idx for idx in range(4) if idx not in pair]
    order = [pair[0], pair[1], remaining[0], remaining[1]]
    coords = torch.empty_like(base)
    for base_idx, atom_idx in enumerate(order):
        coords[atom_idx] = base[base_idx]
    return center_geometry(coords)


def adjacent_switches() -> list[tuple[Pair, Pair]]:
    """All unique adjacent one-bond switches."""

    switches = []
    for i, first in enumerate(X4_PAIRS):
        for second in X4_PAIRS[i + 1 :]:
            if len(set(first).intersection(second)) == 1:
                switches.append((first, second))
    return switches


def disjoint_switches() -> list[tuple[Pair, Pair]]:
    """All unique disjoint one-bond switches."""

    switches = []
    for i, first in enumerate(X4_PAIRS):
        for second in X4_PAIRS[i + 1 :]:
            if not set(first).intersection(second):
                switches.append((first, second))
    return switches
