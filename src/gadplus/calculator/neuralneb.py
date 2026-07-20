"""Conservative PaiNN adapter for the published NeuralNEB Transition1x models.

The published PaiNN model returns forces as ``-dE/dx``.  This adapter forms
the full Cartesian Hessian from that same autograd graph, rather than mixing
an energy-derived force with an independent curvature model.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

import torch

from gadplus.core.types import PredictFn


NEURALNEB_DIR = Path("/lustre07/scratch/memoozd/gadplus/external/neuralneb")
NEURALNEB_MODELS_DIR = Path(
    "/lustre07/scratch/memoozd/gadplus/models/neuralneb-t1x-8d9598b/models"
)


class NeuralNebPaiNNCalculator:
    """Published NeuralNEB PaiNN potential with exact autograd curvature."""

    def __init__(
        self,
        checkpoint: str | Path = NEURALNEB_MODELS_DIR / "painn0.sd",
        device: str = "cuda",
        cutoff: float = 5.0,
    ):
        if not NEURALNEB_DIR.is_dir():
            raise FileNotFoundError(f"NeuralNEB checkout not found: {NEURALNEB_DIR}")
        if str(NEURALNEB_DIR) not in sys.path:
            sys.path.insert(0, str(NEURALNEB_DIR))

        from neuralneb.painn import PaiNN

        self.device = torch.device(device)
        self.cutoff = float(cutoff)
        self.checkpoint = Path(checkpoint)
        if not self.checkpoint.is_file():
            raise FileNotFoundError(f"PaiNN checkpoint not found: {self.checkpoint}")

        self.model = PaiNN(3, 256, self.cutoff)
        state = torch.load(self.checkpoint, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state)
        self.model = self.model.to(self.device).eval()

    def _batch(self, coords: torch.Tensor, atomic_nums: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Build NeuralNEB's one-structure padded batch without detaching coords."""
        z = atomic_nums.to(device=self.device, dtype=torch.long).reshape(-1)
        x = coords.to(device=self.device, dtype=torch.float32).reshape(-1, 3)
        n_atoms = x.shape[0]
        if z.numel() != n_atoms:
            raise ValueError("atomic_nums and coords have inconsistent atom counts")

        distances = torch.cdist(x.detach(), x.detach())
        edge_mask = (distances < self.cutoff) & ~torch.eye(
            n_atoms, dtype=torch.bool, device=self.device
        )
        edges = edge_mask.nonzero(as_tuple=False).to(dtype=torch.long)
        return {
            "nodes": z.unsqueeze(0),
            "nodes_xyz": x.unsqueeze(0),
            "num_nodes": torch.tensor([n_atoms], dtype=torch.long, device=self.device),
            "edges": edges.unsqueeze(0),
            "edges_displacement": torch.zeros(
                (1, edges.shape[0], 3), dtype=x.dtype, device=self.device
            ),
            "cell": torch.zeros((1, 3, 3), dtype=x.dtype, device=self.device),
            "num_edges": torch.tensor([edges.shape[0]], dtype=torch.long, device=self.device),
        }

    def compute(
        self,
        coords: torch.Tensor,
        atomic_nums: torch.Tensor,
        do_hessian: bool = True,
    ) -> Dict[str, torch.Tensor]:
        batch = self._batch(coords, atomic_nums)
        raw_pos = batch["nodes_xyz"]
        raw_pos.requires_grad_(True)

        with torch.enable_grad():
            result = self.model(batch, compute_forces=True)
            energy = result["energy"].reshape(())
            forces = result["forces"].reshape(-1, 3)
            out: Dict[str, torch.Tensor] = {"energy": energy, "forces": forces}
            if do_hessian:
                rows = [
                    torch.autograd.grad(-component, raw_pos, retain_graph=True)[0].reshape(-1)
                    for component in forces.reshape(-1)
                ]
                hessian = torch.stack(rows)
                out["hessian"] = 0.5 * (hessian + hessian.T)
        return out


def make_neuralneb_predict_fn(calculator: NeuralNebPaiNNCalculator) -> PredictFn:
    """Adapt a PaiNN calculator to the backend-neutral prediction protocol."""

    def predict(
        coords: torch.Tensor,
        atomic_nums: torch.Tensor,
        *,
        do_hessian: bool = True,
        require_grad: bool = False,
    ) -> Dict[str, Any]:
        del require_grad
        result = calculator.compute(coords, atomic_nums, do_hessian=do_hessian)
        return {
            key: value.to(device=coords.device, dtype=coords.dtype)
            for key, value in result.items()
        }

    return predict


def load_neuralneb_painn_calculator(**kwargs) -> NeuralNebPaiNNCalculator:
    return NeuralNebPaiNNCalculator(**kwargs)
