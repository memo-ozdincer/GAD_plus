"""Public Lennard-Jones API with autograd and analytical backends."""
from __future__ import annotations

from typing import Any, Literal

import torch

from gadplus.calculator.lennard_jones_analytical import LennardJonesEnergy
from gadplus.calculator.lennard_jones_old import (
    LennardJonesParams,
    LennardJonesPredictor,
    center_geometry,
    lj_atomic_nums,
    pair_distances,
    pair_indices,
    pentagonal_bipyramid_geometry,
    random_cluster_geometry,
    shortest_pair_label,
)
from gadplus.core.types import PredictFn

LJBackend = Literal["autograd", "analytical"]
LJ_RM_FACTOR = 2.0 ** (1.0 / 6.0)


def params_to_analytical_rm(sigma: float) -> float:
    """Map standard LJ sigma to the analytical module's equilibrium distance."""

    return LJ_RM_FACTOR * sigma


class LennardJonesAnalyticalPredictor(PredictFn):
    """PredictFn adapter around the vectorized analytical LJ backend."""

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
    backend: LJBackend = "autograd",
    compile_forces: bool = False,
    compile_hessian: bool = False,
) -> PredictFn:
    """Create a PredictFn for the Lennard-Jones cluster potential.

  Args:
      params: Standard reduced-unit LJ parameters ``(epsilon, sigma)``.
      n_atoms: Cluster size. Required for the analytical backend.
      backend: ``autograd`` uses ``lennard_jones_old``; ``analytical`` uses
          the vectorized force/Hessian implementation with ``rm = 2^(1/6) sigma``.
      compile_forces: Enable ``torch.compile`` for analytical forces on CUDA.
      compile_hessian: Enable ``torch.compile`` for analytical Hessians on CUDA.
    """

    if backend == "autograd":
        return LennardJonesPredictor(params)

    return LennardJonesAnalyticalPredictor(
        n_atoms,
        params,
        compile_forces=compile_forces,
        compile_hessian=compile_hessian,
    )


__all__ = [
    "LJBackend",
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
