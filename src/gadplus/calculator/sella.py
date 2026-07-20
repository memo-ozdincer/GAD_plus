"""Backend-neutral Sella support with a fresh full Hessian at every step."""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import torch
from ase.calculators.calculator import Calculator, all_changes

from gadplus.projection.projection import (
    _eckart_projector,
    atomic_nums_to_symbols,
    get_mass_weights,
)


class FullHessianASECalculator(Calculator):
    """ASE calculator backed by one energy/force/full-Hessian predictor call.

    Sella asks the calculator for energy and forces and separately asks its
    PES for a Hessian. The cached predictor output avoids evaluating the
    backend twice at identical coordinates.
    """

    implemented_properties = ["energy", "forces"]

    def __init__(self, predict_fn: Callable, atomic_nums: torch.Tensor, device: str, **kwargs):
        super().__init__(**kwargs)
        self.predict_fn = predict_fn
        self.atomic_nums = atomic_nums.to(device)
        self.device = torch.device(device)
        self.cached_coords: Optional[torch.Tensor] = None
        self.cached_result: Optional[dict] = None
        self.n_evaluations = 0

    def evaluate(self, positions: np.ndarray) -> dict:
        coords = torch.as_tensor(positions, dtype=torch.float32, device=self.device)
        if self.cached_coords is None or not torch.equal(coords, self.cached_coords):
            self.cached_result = self.predict_fn(
                coords, self.atomic_nums, do_hessian=True, require_grad=False
            )
            self.cached_coords = coords.clone()
            self.n_evaluations += 1
        return self.cached_result

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        result = self.evaluate(self.atoms.positions)
        energy = result["energy"]
        forces = result["forces"]
        self.results["energy"] = float(torch.as_tensor(energy).detach().cpu().reshape(()).item())
        self.results["forces"] = np.asarray(
            torch.as_tensor(forces).detach().cpu().numpy(), dtype=np.float64
        ).reshape(-1, 3)


def full_hessian_function(
    calculator: FullHessianASECalculator,
    *,
    eckart_project: bool,
) -> Callable:
    """Return Sella's Hessian callback, optionally Eckart-cleaned in MW space."""

    def hessian(atoms) -> np.ndarray:
        result = calculator.evaluate(atoms.positions)
        n_atoms = len(atoms)
        hessian_t = torch.as_tensor(result["hessian"], device=calculator.device).detach()
        hessian_t = hessian_t.reshape(3 * n_atoms, 3 * n_atoms).to(torch.float64)
        hessian_t = 0.5 * (hessian_t + hessian_t.T)
        if eckart_project:
            coords = torch.as_tensor(atoms.positions, dtype=torch.float64, device=calculator.device)
            symbols = atomic_nums_to_symbols(calculator.atomic_nums)
            masses, _m3, sqrt_m, sqrt_m_inv = get_mass_weights(symbols, device=calculator.device)
            h_mw = torch.diag(sqrt_m_inv) @ hessian_t @ torch.diag(sqrt_m_inv)
            projector = _eckart_projector(coords, masses)
            h_mw = projector @ h_mw @ projector
            hessian_t = torch.diag(sqrt_m) @ (0.5 * (h_mw + h_mw.T)) @ torch.diag(sqrt_m)
        return hessian_t.cpu().numpy().astype(np.float64)

    return hessian


def refresh_hessian_after_kicks(pes) -> None:
    """Ensure Sella replaces its BFGS update with the exact Hessian every kick."""
    original_kick = pes.kick

    def patched_kick(dx, diag=False, **kwargs):
        ratio = original_kick(dx, diag=diag, **kwargs)
        if pes.hessian_function is not None:
            pes.calculate_hessian()
        return ratio

    pes.kick = patched_kick
