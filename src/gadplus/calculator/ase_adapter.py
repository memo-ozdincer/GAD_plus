"""ASE Calculator adapter for predict_fn backends.

Wraps a ``predict_fn`` callable into an ASE ``Calculator`` interface,
enabling use with ASE-based tools such as Sella for IRC calculations
and ASE optimisers. Backend-agnostic: works with HIP (GPU), SCINE (CPU),
and xTB (CPU) by deriving the torch device from the ``atomic_nums``
tensor passed at construction.
"""

from __future__ import annotations

import numpy as np
import torch
from ase.calculators.calculator import Calculator, all_changes


class HipASECalculator(Calculator):
    """ASE-compatible calculator backed by a GADplus predict_fn.

    The class is named ``HipASECalculator`` for historical compatibility but
    is now backend-agnostic: it dispatches through any ``predict_fn`` that
    follows the PredictFn protocol (HIP, SCINE, xTB).

    Args:
        predict_fn:  Callable with signature
                     ``predict_fn(coords, atomic_nums, do_hessian, require_grad) -> dict``
                     returning ``{"energy": ..., "forces": ...}``.
        atomic_nums: Atomic numbers as a torch tensor (its device is used
                     for coord placement) or any sequence convertible to one.
        **kwargs:    Forwarded to ``ase.calculators.calculator.Calculator``.
    """

    implemented_properties = ["energy", "forces"]

    def __init__(self, predict_fn, atomic_nums, **kwargs):
        super().__init__(**kwargs)
        self.predict_fn = predict_fn
        if isinstance(atomic_nums, torch.Tensor):
            self.atomic_nums = atomic_nums
        else:
            self.atomic_nums = torch.as_tensor(atomic_nums, dtype=torch.int64)
        self._device = self.atomic_nums.device

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        """Compute energy and forces for the current atomic configuration.

        Reads positions from ``self.atoms``, calls ``predict_fn``, and
        stores results in ``self.results``.
        """
        super().calculate(atoms, properties, system_changes)

        coords = torch.tensor(
            self.atoms.positions, dtype=torch.float32, device=self._device
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
