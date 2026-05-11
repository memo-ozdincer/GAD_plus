"""SCINE Sparrow calculator adapter.

Wraps SCINE Sparrow semiempirical methods (DFTB0/DFTB2/DFTB3/PM6/AM1/RM1/MNDO)
into the PredictFn interface. CPU-only, not autograd-differentiable.

Units: SCINE returns energies in Hartree, gradients in Hartree/Bohr,
Hessian in Hartree/Bohr^2. We convert to eV / eV-Angstrom / eV-Angstrom^2
to match the HIP backend.
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from gadplus.core.types import PredictFn


HARTREE_TO_EV = 27.211386245988
BOHR_TO_ANG = 0.529177210903


@contextmanager
def _suppress_stdout():
    """Suppress SCINE chatter at the file-descriptor level."""
    devnull = os.open(os.devnull, os.O_WRONLY)
    saved = os.dup(1)
    os.dup2(devnull, 1)
    try:
        yield
    finally:
        os.dup2(saved, 1)
        os.close(devnull)
        os.close(saved)


class ScineSparrowCalculator:
    """SCINE Sparrow wrapper, exposing a single-geometry compute method.

    Args:
        functional: SCINE method name. DFTB0 is the project default
            (fastest and most stable for organic molecules).
        device: Ignored; SCINE runs on CPU.
    """

    SUPPORTED_METHODS = ("DFTB0", "DFTB2", "DFTB3", "PM6", "AM1", "RM1", "MNDO")

    def __init__(self, functional: str = "DFTB0", device: str = "cpu", **_):
        import scine_sparrow  # noqa: F401  (registers the module)
        import scine_utilities

        self._scine_utilities = scine_utilities
        self.functional = functional
        self.device_str = "cpu"

        self.manager = scine_utilities.core.ModuleManager.get_instance()
        sparrow_so = Path(scine_sparrow.__file__).parent / "sparrow.module.so"
        if not sparrow_so.exists():
            raise RuntimeError(f"SCINE Sparrow module not found at {sparrow_so}")
        self.manager.load(os.fspath(sparrow_so))

        if self.manager.get("calculator", functional) is None:
            raise ValueError(
                f"Calculator '{functional}' not found. "
                f"Supported: {', '.join(self.SUPPORTED_METHODS)}"
            )

        scine_utilities.core.Log.silent()

    def _atomic_nums_to_elements(self, atomic_nums: np.ndarray):
        ElementType = self._scine_utilities.ElementType
        # SCINE stores elements as ElementType.<Symbol>. Build symbol map.
        # Use periodic table symbols for the elements we expect in T1x.
        symbols = {
            1: "H", 6: "C", 7: "N", 8: "O", 9: "F", 15: "P", 16: "S",
            17: "Cl", 35: "Br", 53: "I",
        }
        out = []
        for z in atomic_nums:
            z_int = int(z)
            sym = symbols.get(z_int)
            if sym is None:
                raise ValueError(f"Unsupported element Z={z_int} for SCINE backend")
            out.append(getattr(ElementType, sym))
        return out

    def compute(
        self,
        coords: torch.Tensor,
        atomic_nums: torch.Tensor,
        do_hessian: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Run a single SCINE singlepoint and return torch tensors in eV/Angstrom."""
        scine = self._scine_utilities

        coords_np = coords.detach().cpu().numpy().reshape(-1, 3).astype(np.float64)
        z_np = atomic_nums.detach().cpu().numpy().astype(np.int64)

        elements = self._atomic_nums_to_elements(z_np)
        positions_bohr = coords_np * scine.BOHR_PER_ANGSTROM
        structure = scine.AtomCollection(elements, positions_bohr)

        calc = self.manager.get("calculator", self.functional)
        calc.structure = structure
        props = [scine.Property.Energy, scine.Property.Gradients]
        if do_hessian:
            props.append(scine.Property.Hessian)
        calc.set_required_properties(props)

        with _suppress_stdout():
            results = calc.calculate()

        energy_ev = float(results.energy) * HARTREE_TO_EV
        gradients = np.asarray(results.gradients).reshape(-1, 3)
        # Forces = -gradient; convert Hartree/Bohr -> eV/Angstrom.
        forces_ev_ang = -gradients * (HARTREE_TO_EV / BOHR_TO_ANG)

        out: Dict[str, torch.Tensor] = {
            "energy": torch.tensor(energy_ev, dtype=torch.float64),
            "forces": torch.tensor(forces_ev_ang, dtype=torch.float64),
        }
        if do_hessian:
            hess = np.asarray(results.hessian)
            hess_ev_ang2 = hess * (HARTREE_TO_EV / (BOHR_TO_ANG ** 2))
            out["hessian"] = torch.tensor(hess_ev_ang2, dtype=torch.float64)
        return out


def make_scine_predict_fn(calculator: ScineSparrowCalculator) -> PredictFn:
    """Adapt a ScineSparrowCalculator to the PredictFn protocol."""

    def _predict(
        coords: torch.Tensor,
        atomic_nums: torch.Tensor,
        *,
        do_hessian: bool = True,
        require_grad: bool = False,
    ) -> Dict[str, Any]:
        if require_grad:
            raise NotImplementedError(
                "SCINE backend is not autograd-differentiable; use require_grad=False"
            )
        result = calculator.compute(coords, atomic_nums, do_hessian=do_hessian)
        # Match caller's dtype/device for energy/forces; keep Hessian in float64.
        target_device = coords.device
        target_dtype = coords.dtype
        result["energy"] = result["energy"].to(device=target_device, dtype=target_dtype)
        result["forces"] = result["forces"].to(device=target_device, dtype=target_dtype)
        if "hessian" in result:
            result["hessian"] = result["hessian"].to(device=target_device)
        return result

    return _predict


def load_scine_calculator(
    functional: str = "DFTB0",
    device: str = "cpu",
    **kwargs,
) -> ScineSparrowCalculator:
    """Construct a SCINE Sparrow calculator.

    Args:
        functional: SCINE method (default "DFTB0").
        device: Ignored (SCINE is CPU-only).
    """
    return ScineSparrowCalculator(functional=functional, device=device, **kwargs)
