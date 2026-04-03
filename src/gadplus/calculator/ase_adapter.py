"""ASE Calculator adapter for HIP-NN potential.

Wraps a ``predict_fn`` callable into an ASE ``Calculator`` interface,
enabling use with ASE-based tools such as Sella for IRC calculations
and ASE optimisers.
"""

from __future__ import annotations

import numpy as np
import torch
from ase.calculators.calculator import Calculator, all_changes


class HipASECalculator(Calculator):
    """ASE-compatible calculator backed by a HIP-NN predict function.

    Translates between ASE's ``Atoms`` object and the ``predict_fn``
    interface used throughout GADplus, enabling interoperability with
    ASE optimisers, Sella IRC, and other ASE-based workflows.

    Args:
        predict_fn:  Callable with signature
                     ``predict_fn(coords, atomic_nums, do_hessian, require_grad) -> dict``
                     returning ``{"energy": ..., "forces": ...}``.
        atomic_nums: Sequence of atomic numbers for the molecule.
        **kwargs:    Forwarded to ``ase.calculators.calculator.Calculator``.
    """

    implemented_properties = ["energy", "forces"]

    def __init__(self, predict_fn, atomic_nums, **kwargs):
        super().__init__(**kwargs)
        self.predict_fn = predict_fn
        self.atomic_nums = atomic_nums

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        """Compute energy and forces for the current atomic configuration.

        Reads positions from ``self.atoms``, calls ``predict_fn``, and
        stores results in ``self.results``.
        """
        super().calculate(atoms, properties, system_changes)

        coords = torch.tensor(
            self.atoms.positions, dtype=torch.float32, device="cuda"
        )
        out = self.predict_fn(
            coords, self.atomic_nums, do_hessian=False, require_grad=False
        )

        # Energy: handle both tensor and scalar returns.
        energy = out["energy"]
        if isinstance(energy, torch.Tensor):
            energy = energy.detach().cpu().item()
        self.results["energy"] = float(energy)

        # Forces: handle both tensor and numpy returns.
        forces = out["forces"]
        if isinstance(forces, torch.Tensor):
            forces = forces.detach().cpu().numpy()
        self.results["forces"] = np.asarray(forces).reshape(-1, 3)
