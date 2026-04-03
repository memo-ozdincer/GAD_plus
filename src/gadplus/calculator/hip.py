"""HIP (Equiformer) calculator adapter.

Wraps HIP's EquiformerTorchCalculator into the PredictFn interface.
Also provides PyG batch construction for HIP's expected input format.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from torch_geometric.data import Batch
from torch_geometric.data import Data as TGData

from gadplus.core.types import PredictFn


def coords_to_pyg_batch(
    coords: torch.Tensor,
    atomic_nums: torch.Tensor,
    *,
    device: Optional[torch.device] = None,
) -> Batch:
    """Create a single-structure PyG Batch in the format HIP expects.

    Args:
        coords: (N, 3) or (3N,) atomic coordinates.
        atomic_nums: (N,) atomic numbers.
        device: Target device (defaults to coords.device).

    Returns:
        PyG Batch with pos, z, charges, natoms, cell, pbc fields.
    """
    if coords.dim() == 1:
        coords = coords.reshape(-1, 3)

    if device is None:
        device = coords.device

    data = TGData(
        pos=torch.as_tensor(coords, dtype=torch.float32),
        z=torch.as_tensor(atomic_nums, dtype=torch.int64),
        charges=torch.as_tensor(atomic_nums, dtype=torch.int64),
        natoms=torch.tensor([int(atomic_nums.numel())], dtype=torch.int64),
        cell=None,
        pbc=torch.tensor(False, dtype=torch.bool),
    )
    return Batch.from_data_list([data]).to(device)


def make_hip_predict_fn(calculator) -> PredictFn:
    """Create a PredictFn adapter for HIP EquiformerTorchCalculator.

    Two paths:
        require_grad=False: Uses calculator.predict() (fast, no autograd).
        require_grad=True: Uses calculator.potential.forward() for autograd.

    Args:
        calculator: An EquiformerTorchCalculator instance with .potential attribute.

    Returns:
        A PredictFn callable.
    """
    model = calculator.potential

    def _predict(
        coords: torch.Tensor,
        atomic_nums: torch.Tensor,
        *,
        do_hessian: bool = True,
        require_grad: bool = False,
    ) -> Dict[str, Any]:
        device = coords.device
        batch = coords_to_pyg_batch(coords, atomic_nums, device=device)

        if require_grad:
            if not do_hessian:
                raise ValueError("HIP differentiable path expects do_hessian=True")
            with torch.enable_grad():
                _, _, out = model.forward(batch, otf_graph=True)
                return {
                    "energy": out.get("energy"),
                    "forces": out.get("forces"),
                    "hessian": out.get("hessian"),
                }

        with torch.no_grad():
            return calculator.predict(batch, do_hessian=do_hessian)

    return _predict


def load_hip_calculator(
    checkpoint_path: str,
    device: str = "cuda",
    hessian_method: str = "predict",
):
    """Load HIP calculator from checkpoint.

    Args:
        checkpoint_path: Path to HIP .ckpt file.
        device: Target device ("cuda" or "cpu").
        hessian_method: Hessian computation method ("predict").

    Returns:
        EquiformerTorchCalculator instance.
    """
    # Monkey-patch to allow inference without training dataset paths
    from hip import path_config, training_module, inference_utils

    _original = path_config.fix_dataset_path

    def _lenient(path):
        try:
            return _original(path)
        except FileNotFoundError:
            return path

    path_config.fix_dataset_path = _lenient
    training_module.fix_dataset_path = _lenient
    inference_utils.fix_dataset_path = _lenient

    from hip.equiformer_torch_calculator import EquiformerTorchCalculator

    calculator = EquiformerTorchCalculator(
        checkpoint_path=checkpoint_path,
        hessian_method=hessian_method,
        device=device,
    )
    return calculator
