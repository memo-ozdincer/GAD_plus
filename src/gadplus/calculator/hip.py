"""HIP (Equiformer) calculator adapter.

Wraps HIP's EquiformerTorchCalculator into the PredictFn interface.
Also provides PyG batch construction for HIP's expected input format.
"""
from __future__ import annotations

from typing import Any, Dict, Literal, Optional

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
                energy, forces, out = model.forward(batch, otf_graph=True)
                return {
                    "energy": energy,
                    "forces": forces,
                    "hessian": out.get("hessian"),
                }

        with torch.no_grad():
            return calculator.predict(batch, do_hessian=do_hessian)

    return _predict


def make_hip_curvature_source_predict_fn(
    calculator,
    source: Literal["predicted", "force_jacobian", "energy_hessian"],
) -> PredictFn:
    """Keep HIP's direct E/F heads fixed while choosing its curvature source.

    This is a diagnostic adapter, not a claim that the direct-force field is
    conservative. Returned Hessians are symmetrized because both GAD's
    vibrational decomposition and Sella require a symmetric curvature matrix.
    """
    model = calculator.potential

    def _negative_jacobian(values: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        rows = []
        for value in values.reshape(-1):
            rows.append(-torch.autograd.grad(value, positions, retain_graph=True)[0].reshape(-1))
        return torch.stack(rows)

    def _predict(
        coords: torch.Tensor,
        atomic_nums: torch.Tensor,
        *,
        do_hessian: bool = True,
        require_grad: bool = False,
    ) -> Dict[str, Any]:
        del require_grad
        standard_batch = coords_to_pyg_batch(coords, atomic_nums, device=coords.device)
        with torch.no_grad():
            standard = calculator.predict(
                standard_batch, do_hessian=do_hessian,
            )
        result: Dict[str, Any] = {
            "energy": standard["energy"],
            "forces": standard["forces"],
        }
        if not do_hessian:
            return result
        if source == "predicted":
            raw_hessian = standard["hessian"].reshape(coords.numel(), coords.numel())
            result["hessian_raw"] = raw_hessian
            result["hessian"] = raw_hessian
            return result
        batch = coords_to_pyg_batch(coords, atomic_nums, device=coords.device)
        batch.pos = batch.pos.detach().clone().requires_grad_(True)
        with torch.enable_grad():
            energy, direct_forces, _outputs = model.forward(batch, otf_graph=True)
            if source == "force_jacobian":
                raw_hessian = _negative_jacobian(direct_forces, batch.pos)
            else:
                energy_forces = -torch.autograd.grad(energy.sum(), batch.pos, create_graph=True)[0]
                raw_hessian = _negative_jacobian(energy_forces, batch.pos)
            result["hessian_raw"] = raw_hessian.detach()
            result["hessian"] = (0.5 * (raw_hessian + raw_hessian.T)).detach()
        return result

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
    # e3nn versions bundled with this HIP checkout serialize ``slice`` in a
    # trusted package constant. PyTorch >=2.6 now defaults torch.load to a
    # restricted weights-only unpickler, so explicitly allow that one type
    # before importing the HIP model stack.
    if hasattr(torch.serialization, "add_safe_globals"):
        torch.serialization.add_safe_globals([slice])

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
