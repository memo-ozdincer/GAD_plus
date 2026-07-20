#!/usr/bin/env python
"""GPU smoke test for the HORM LEFTNet E-F-H checkpoint."""
from __future__ import annotations

import argparse
import json
import sys

import torch
import torch.nn.functional as F
from torch_geometric.data import Data


HORM_DIR = "/lustre07/scratch/memoozd/external/HORM"
if HORM_DIR not in sys.path:
    sys.path.insert(0, HORM_DIR)


def make_batch(z: torch.Tensor, pos: torch.Tensor) -> Data:
    supported = torch.tensor([1, 6, 7, 8], device=z.device)
    matches = z[:, None].eq(supported[None, :])
    if not bool(matches.any(dim=1).all()):
        raise ValueError("HORM LEFTNet supports H/C/N/O only")
    indices = matches.to(torch.long).argmax(dim=1)
    n_atoms = int(z.numel())
    return Data(
        pos=pos,
        # HORM LEFTNet was trained with a five-channel atom-type encoding;
        # H/C/N/O occupy the first four channels and the fifth remains zero.
        one_hot=F.one_hot(indices, num_classes=5).to(torch.float32),
        charges=torch.zeros(n_atoms, dtype=torch.float32, device=z.device),
        batch=torch.zeros(n_atoms, dtype=torch.long, device=z.device),
        natoms=torch.tensor([n_atoms], dtype=torch.long, device=z.device),
        ae=torch.zeros(1, dtype=torch.float32, device=z.device),
    )


def predict(model, z: torch.Tensor, pos: torch.Tensor, direct_forces: bool):
    # HORM's convenience wrapper differentiates w.r.t. its internally
    # re-centered tensor.  Differentiate through the centering map instead,
    # so forces and Hessians are defined in the caller's Cartesian chart.
    from leftnet.potential import get_edges_index, get_n_frag_switch

    raw_pos = pos.detach().clone().requires_grad_(True)
    batch = make_batch(z, raw_pos)
    if direct_forces:
        energy, forces = model.forward(batch)
    else:
        centered_pos = raw_pos - raw_pos.mean(dim=0, keepdim=True)
        edge_index = get_edges_index(batch.batch, remove_self_edge=True)
        n_frag_switch = get_n_frag_switch([batch.natoms])
        h = [torch.cat([batch.one_hot, batch.charges[:, None]], dim=1).float()]
        energy, _ = model._forward_autograd(
            h=h,
            pos=centered_pos,
            edge_index=edge_index,
            t=torch.tensor([0.0], device=raw_pos.device),
            conditions=torch.zeros(1, 1, dtype=torch.long, device=raw_pos.device),
            n_frag_switch=n_frag_switch,
            combined_mask=batch.batch,
            edge_attr=None,
        )
        forces = -torch.autograd.grad(energy.sum(), raw_pos, create_graph=True)[0]
    return energy, forces, raw_pos


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--direct-forces", action="store_true")
    parser.add_argument("--sample-id", type=int, default=2)
    parser.add_argument("--h5", default="/lustre06/project/6033559/memoozd/data/transition1x.h5")
    parser.add_argument("--epsilon", type=float, default=1e-3)
    parser.add_argument(
        "--epsilons", type=float, nargs="+", default=None,
        help="Optional finite-difference sweep; overrides --epsilon.",
    )
    args = parser.parse_args()

    from transition1x import Dataloader
    from training_module import PotentialModule

    sample = None
    for index, record in enumerate(Dataloader(args.h5, datasplit="test", only_final=True)):
        if index == args.sample_id:
            sample = record
            break
    if sample is None:
        raise IndexError(f"sample {args.sample_id} was not found")
    ts = sample["transition_state"]
    z = torch.as_tensor(ts["atomic_numbers"], dtype=torch.long, device="cuda")
    pos = torch.as_tensor(ts["positions"], dtype=torch.float32, device="cuda")

    module = PotentialModule.load_from_checkpoint(args.checkpoint, strict=False)
    model = module.potential.to("cuda").eval()

    energy, forces, pos_used = predict(model, z, pos.clone(), args.direct_forces)
    flat_force = forces.reshape(-1)
    rows = []
    for force_component in flat_force:
        row = torch.autograd.grad(
            -force_component, pos_used, retain_graph=True,
        )[0].reshape(-1)
        rows.append(row)
    hessian = torch.stack(rows)

    gen = torch.Generator(device="cuda").manual_seed(7)
    direction3 = torch.randn(pos.shape, device="cuda", generator=gen)
    direction3 -= direction3.mean(dim=0, keepdim=True)
    direction = direction3.reshape(-1)
    direction /= torch.linalg.vector_norm(direction)
    direction3 = direction.reshape_as(pos_used)
    analytic_hv = hessian @ direction

    print(f"sample={args.sample_id} atoms={len(z)}")
    print(f"energy={float(energy):.8f} fmax={float(forces.abs().max()):.6e}")
    asym = (hessian - hessian.T).abs().max()
    print(f"hessian_absmax={float(hessian.abs().max()):.6e}")
    print(f"hessian_asym_max={float(asym):.6e}")
    print(f"fd_hv_absmax={float(analytic_hv.abs().max()):.6e}")
    for epsilon in args.epsilons or [args.epsilon]:
        centered_pos = pos_used.detach()
        _, f_plus, _ = predict(model, z, centered_pos + epsilon * direction3, args.direct_forces)
        _, f_minus, _ = predict(model, z, centered_pos - epsilon * direction3, args.direct_forces)
        fd_hv = (-(f_plus.reshape(-1)) + f_minus.reshape(-1)) / (2 * epsilon)
        fd_error = (fd_hv - analytic_hv).abs()
        row = {
            "epsilon_A": epsilon,
            "fd_hv_max_error": float(fd_error.max()),
            "fd_hv_rms_error": float(torch.sqrt(torch.mean(fd_error.square()))),
            "fd_hv_relative_rms_error": float(
                torch.sqrt(torch.mean(fd_error.square()))
                / torch.sqrt(torch.mean(analytic_hv.square())).clamp_min(1e-8)
            ),
        }
        print(json.dumps(row, sort_keys=True))


if __name__ == "__main__":
    main()
