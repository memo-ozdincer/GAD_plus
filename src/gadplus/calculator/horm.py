"""HORM LEFTNet E-F-H calculator adapter."""
from __future__ import annotations

import sys
from typing import Any, Dict

import torch
import torch.nn.functional as F
from torch_geometric.data import Data

from gadplus.core.types import PredictFn


HORM_DIR = "/lustre07/scratch/memoozd/external/HORM"
HORM_LEFTNET_CHECKPOINT = "/lustre07/scratch/memoozd/models/horm/left.ckpt"
HORM_LEFTNET_DF_CHECKPOINT = "/lustre07/scratch/memoozd/models/horm/left-df.ckpt"
_SUPPORTED = torch.tensor([1, 6, 7, 8])


class HormLeftNetCalculator:
    """HORM LEFTNet calculator with conservative or direct-force outputs."""

    def __init__(
        self,
        checkpoint: str = HORM_LEFTNET_CHECKPOINT,
        device: str = "cuda",
        direct_forces: bool = False,
    ):
        if HORM_DIR not in sys.path:
            sys.path.insert(0, HORM_DIR)
        from training_module import PotentialModule

        self.device = torch.device(device)
        self.model = PotentialModule.load_from_checkpoint(checkpoint, strict=False).potential
        self.model = self.model.to(self.device).eval()
        self.direct_forces = direct_forces

    def _batch(self, atomic_nums: torch.Tensor, pos: torch.Tensor) -> Data:
        z = atomic_nums.to(device=self.device, dtype=torch.long).reshape(-1)
        matches = z[:, None].eq(_SUPPORTED.to(self.device)[None, :])
        if not bool(matches.any(dim=1).all()):
            raise ValueError("HORM LEFTNet supports H/C/N/O only")
        indices = matches.to(torch.long).argmax(dim=1)
        n_atoms = z.numel()
        return Data(
            pos=pos,
            # HORM's checkpoint expects five atom-type channels.
            one_hot=F.one_hot(indices, num_classes=5).float(),
            charges=torch.zeros(n_atoms, dtype=torch.float32, device=self.device),
            batch=torch.zeros(n_atoms, dtype=torch.long, device=self.device),
            natoms=torch.tensor([n_atoms], dtype=torch.long, device=self.device),
            ae=torch.zeros(1, dtype=torch.float32, device=self.device),
        )

    def compute(self, coords: torch.Tensor, atomic_nums: torch.Tensor, do_hessian: bool = True) -> Dict[str, torch.Tensor]:
        raw_pos = coords.detach().to(self.device, torch.float32).clone().requires_grad_(True)
        batch = self._batch(atomic_nums, raw_pos)
        if self.direct_forces:
            # LEFTNet-df emits a force field independently of its energy
            # head. Its Jacobian is therefore the model's learned local
            # curvature object, not necessarily an energy Hessian.
            energy, forces = self.model.forward(batch)
        else:
            from leftnet.potential import get_edges_index, get_n_frag_switch

            centered_pos = raw_pos - raw_pos.mean(dim=0, keepdim=True)
            h = [torch.cat([batch.one_hot, batch.charges[:, None]], dim=1)]
            energy, _ = self.model._forward_autograd(
                h=h,
                pos=centered_pos,
                edge_index=get_edges_index(batch.batch, remove_self_edge=True),
                t=torch.tensor([0.0], device=self.device),
                conditions=torch.zeros(1, 1, dtype=torch.long, device=self.device),
                n_frag_switch=get_n_frag_switch([batch.natoms]),
                combined_mask=batch.batch,
                edge_attr=None,
            )
            forces = -torch.autograd.grad(energy.sum(), raw_pos, create_graph=do_hessian)[0]
        out: Dict[str, torch.Tensor] = {"energy": energy, "forces": forces}
        if do_hessian:
            # HORM's scatter kernels do not currently support a performant
            # batched VJP, so form the exact force Jacobian row by row.
            rows = [
                torch.autograd.grad(-component, raw_pos, retain_graph=True)[0].reshape(-1)
                for component in forces.reshape(-1)
            ]
            raw_hessian = torch.stack(rows)
            out["hessian_raw"] = raw_hessian
            out["hessian"] = 0.5 * (raw_hessian + raw_hessian.T)
        return out


def make_horm_predict_fn(calculator: HormLeftNetCalculator) -> PredictFn:
    def predict(coords: torch.Tensor, atomic_nums: torch.Tensor, *, do_hessian: bool = True, require_grad: bool = False) -> Dict[str, Any]:
        if require_grad:
            raise NotImplementedError("HORM adapter supplies explicit forces/Hessians")
        result = calculator.compute(coords, atomic_nums, do_hessian=do_hessian)
        return {key: value.to(device=coords.device, dtype=coords.dtype) for key, value in result.items()}
    return predict


def load_horm_leftnet_calculator(**kwargs) -> HormLeftNetCalculator:
    return HormLeftNetCalculator(**kwargs)
