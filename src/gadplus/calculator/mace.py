"""MACE-OFF23 calculator adapter.

MACE-OFF23 is a transferable neutral-organic force field.  This adapter keeps
its ASE calculator on the selected device while exposing the project's
``PredictFn`` contract in eV, eV/Angstrom, and eV/Angstrom^2.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
import torch
from ase import Atoms

from gadplus.core.types import PredictFn


# MACE-OFF23's documented element set.  It is deliberately checked before a
# model call so a mixed T1x subset cannot fail halfway through a worker run.
MACE_OFF23_ATOMIC_NUMBERS = frozenset({1, 6, 7, 8, 9, 15, 16, 17, 35, 53})


class MaceOffCalculator:
    """Persistent MACE-OFF23 ASE calculator with full-Hessian support."""

    def __init__(
        self,
        model: str = "small",
        device: str = "cuda",
        default_dtype: str = "float64",
        **_,
    ):
        from mace.calculators import mace_off

        self.model = model
        self.device_str = device
        self.default_dtype = default_dtype
        self._calculator = mace_off(
            model=model,
            device=device,
            default_dtype=default_dtype,
        )

    @staticmethod
    def _validate_atomic_numbers(atomic_nums: torch.Tensor) -> np.ndarray:
        numbers = atomic_nums.detach().cpu().numpy().astype(np.int64).reshape(-1)
        unsupported = sorted(set(numbers.tolist()).difference(MACE_OFF23_ATOMIC_NUMBERS))
        if unsupported:
            raise ValueError(
                "MACE-OFF23 does not support atomic numbers "
                f"{unsupported}; supported: {sorted(MACE_OFF23_ATOMIC_NUMBERS)}"
            )
        return numbers

    def compute(
        self,
        coords: torch.Tensor,
        atomic_nums: torch.Tensor,
        do_hessian: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Evaluate one geometry through MACE's ASE calculator."""
        numbers = self._validate_atomic_numbers(atomic_nums)
        positions = coords.detach().cpu().numpy().reshape(-1, 3).astype(np.float64)
        atoms = Atoms(numbers=numbers, positions=positions)
        atoms.calc = self._calculator

        energy = float(atoms.get_potential_energy())
        forces = np.asarray(atoms.get_forces(), dtype=np.float64).reshape(-1, 3)
        result: Dict[str, torch.Tensor] = {
            "energy": torch.tensor(energy, dtype=torch.float64),
            "forces": torch.from_numpy(forces),
        }
        if do_hessian:
            hessian = np.asarray(self._calculator.get_hessian(atoms), dtype=np.float64)
            hessian = hessian.reshape(3 * len(atoms), 3 * len(atoms))
            # Preserve a real symmetric vibrational problem despite roundoff.
            hessian = 0.5 * (hessian + hessian.T)
            result["hessian"] = torch.from_numpy(hessian)
        return result


def make_mace_predict_fn(calculator: MaceOffCalculator) -> PredictFn:
    """Adapt MACE-OFF23 to the backend-neutral prediction protocol."""

    def _predict(
        coords: torch.Tensor,
        atomic_nums: torch.Tensor,
        *,
        do_hessian: bool = True,
        require_grad: bool = False,
    ) -> Dict[str, Any]:
        if require_grad:
            raise NotImplementedError(
                "MACE-OFF23 is exposed through ASE; use require_grad=False"
            )
        result = calculator.compute(coords, atomic_nums, do_hessian=do_hessian)
        result["energy"] = result["energy"].to(device=coords.device, dtype=coords.dtype)
        result["forces"] = result["forces"].to(device=coords.device, dtype=coords.dtype)
        if "hessian" in result:
            result["hessian"] = result["hessian"].to(device=coords.device)
        return result

    return _predict


def load_mace_calculator(
    model: str = "small",
    device: str = "cuda",
    default_dtype: str = "float64",
    **kwargs,
) -> MaceOffCalculator:
    """Construct a cached MACE-OFF23 calculator.

    The MACE checkpoint must be present in the MACE cache before running on
    Narval compute nodes, which intentionally have no internet access.
    """
    return MaceOffCalculator(
        model=model,
        device=device,
        default_dtype=default_dtype,
        **kwargs,
    )
