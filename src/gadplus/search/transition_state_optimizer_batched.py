"""Batched transition-state vector field for projected GAD."""
from __future__ import annotations

import argparse

import torch

from gadplus.calculator.lennard_jones import lj_atomic_nums, params_to_analytical_rm
from gadplus.calculator.lennard_jones_analytical import LennardJonesEnergy
from gadplus.core.convergence import (
    NEG_EIGVAL_THRESHOLD,
    count_negative_eigenvalues,
    force_max,
    force_mean,
    force_value_from_criterion,
    is_ts_converged,
)
from gadplus.projection import (
    atomic_nums_to_symbols,
    batched_gad_dynamics_projected,
    batched_vib_eig,
    vib_eig,
)


class TransitionStateVectorfield:
    """Callable projected-GAD vector field for batched index-1 TS search.

    The vector field accepts one geometry shaped ``(N, 3)`` or ``(3N,)``, or a
    batch shaped ``(B, N, 3)`` or ``(B, 3N)``. Batched calls evaluate the
    Lennard-Jones force/Hessian and Eckart-projected vibrational eigensolve for
    the whole batch at once.
    """

    def __init__(
        self,
        *,
        n_atoms: int = 7,
        epsilon: float = 1.0,
        sigma: float = 1.0,
        atomic_number: int = 1,
        compile_forces: bool = False,
        compile_hessian: bool = False,
        oscillator: bool = False,
        purify_hessian: bool = False,
        return_weighted_step_direction: bool = False,
    ) -> None:
        if n_atoms < 2:
            raise ValueError("TransitionStateVectorfield requires at least two atoms.")

        self.n_atoms = n_atoms
        self.atomic_nums = lj_atomic_nums(n_atoms, atomic_number=atomic_number)
        self.potential_energy_surface = LennardJonesEnergy(
            n_particles=n_atoms,
            eps=epsilon,
            rm=params_to_analytical_rm(sigma),
            oscillator=oscillator,
            compile_forces=compile_forces,
            compile_hessian=compile_hessian,
        )
        self.purify_hessian = purify_hessian
        self.return_weighted_step_direction = return_weighted_step_direction
        self.last_info: dict[str, object] = {}

    def __call__(self, coords: torch.Tensor) -> torch.Tensor:
        """Return projected transition-state-search directions at ``coords``."""

        coords_batch, was_batched, return_flat = _coords_to_batch(coords, self.n_atoms)
        atomic_nums = self.atomic_nums.to(device=coords_batch.device)
        atomsymbols = atomic_nums_to_symbols(atomic_nums)
        out = self._evaluate(coords_batch)
        forces = out["forces"].reshape(coords_batch.shape[0], self.n_atoms, 3)

        evals_vib, evecs_vib_3n, _ = batched_vib_eig(
            out["hessian"],
            coords_batch,
            atomsymbols,
            purify=self.purify_hessian,
        )
        n_neg = _count_negative_eigenvalues_batch(evals_vib)
        v = evecs_vib_3n[:, :, 0].to(device=forces.device, dtype=forces.dtype)

        blend_weight = torch.where(
            n_neg > 1,
            torch.zeros_like(evals_vib[:, 0]),
            torch.ones_like(evals_vib[:, 0]),
        )
        direction_batch, _v_projected, info = batched_gad_dynamics_projected(
            coords=coords_batch,
            forces=forces,
            v=v,
            atomsymbols=atomsymbols,
            gad_blend_weight=blend_weight,
            return_weighted_step_direction=self.return_weighted_step_direction,
        )
        self.last_info = _format_info(
            energy=out["energy"],
            forces=forces,
            evals_vib=evals_vib,
            n_neg=n_neg,
            blend_weight=blend_weight,
            info=info,
            was_batched=was_batched,
        )
        return _restore_direction_shape(direction_batch, was_batched, return_flat)

    def reset(self) -> None:
        """Clear cached diagnostics before starting an independent trajectory."""

        self.last_info = {}

    def _evaluate(self, coords: torch.Tensor) -> dict[str, torch.Tensor]:
        coords_batch, _was_batched, _return_flat = _coords_to_batch(coords, self.n_atoms)
        positions = coords_batch.to(dtype=torch.float64).detach()
        batch = {"positions": positions}

        energy = self.potential_energy_surface._energy_from_positions(positions).detach()
        forces = self.potential_energy_surface.compute_forces(batch).detach()
        hessian = self.potential_energy_surface.compute_hessian(batch).detach()
        hessian = 0.5 * (hessian + hessian.transpose(-1, -2))
        return {
            "energy": energy,
            "forces": forces,
            "hessian": hessian,
        }


##################################################################################################
# Optimization
##################################################################################################


def cap_displacement(
    step_disp: torch.Tensor,
    max_atom_disp: float,
) -> torch.Tensor:
    """Cap per-atom displacement to a maximum value.

    Args:
        step_disp: (N, 3) or (3N,) displacement vector.
        max_atom_disp: Maximum per-atom displacement in Angstrom.

    Returns:
        Capped displacement with same shape as input.
    """
    disp_3d = step_disp.reshape(-1, 3)
    max_actual = float(disp_3d.norm(dim=1).max().item())
    if max_actual > max_atom_disp and max_actual > 0:
        disp_3d = disp_3d * (max_atom_disp / max_actual)
    return disp_3d.reshape(step_disp.shape)


def min_interatomic_distance(coords: torch.Tensor) -> float:
    """Compute minimum interatomic distance (Angstrom).

    Args:
        coords: (N, 3) atomic coordinates.

    Returns:
        Minimum pairwise distance, or inf for single-atom systems.
    """
    c = coords.reshape(-1, 3)
    n = c.shape[0]
    if n < 2:
        return float("inf")
    diff = c.unsqueeze(0) - c.unsqueeze(1)
    dist = diff.norm(dim=2) + torch.eye(n, device=c.device, dtype=c.dtype) * 1e10
    return float(dist.min().item())


def _scalar_energy(energy: torch.Tensor | float) -> float:
    if isinstance(energy, torch.Tensor):
        return float(energy.detach().reshape(-1)[0].item())
    return float(energy)


def _diagnostics(
    vectorfield: TransitionStateVectorfield,
    coords: torch.Tensor,
    *,
    force_criterion: str,
) -> dict[str, float | int]:
    coords_3d = coords.detach().reshape(-1, 3)
    atomic_nums = vectorfield.atomic_nums.to(device=coords_3d.device)
    out = vectorfield._evaluate(coords_3d)
    forces = out["forces"].reshape(-1, 3)
    evals_vib, _, _ = vib_eig(
        out["hessian"][0],
        coords_3d,
        atomic_nums_to_symbols(atomic_nums),
        purify=vectorfield.purify_hessian,
    )
    return {
        "n_neg": count_negative_eigenvalues(evals_vib),
        "eig0": float(evals_vib[0].item()) if evals_vib.numel() > 0 else 0.0,
        "eig1": float(evals_vib[1].item()) if evals_vib.numel() > 1 else 0.0,
        "force_max": force_max(forces),
        "force_norm": force_mean(forces),
        "force_value": force_value_from_criterion(forces, force_criterion),
        "energy": _scalar_energy(out["energy"]),
    }


def _coords_to_batch(coords: torch.Tensor, n_atoms: int) -> tuple[torch.Tensor, bool, bool]:
    xyz = coords.detach()
    dim3n = 3 * n_atoms
    if xyz.dim() == 1:
        if xyz.numel() != dim3n:
            raise ValueError(f"Expected flat coordinate length {dim3n}, got {xyz.numel()}.")
        return xyz.reshape(1, n_atoms, 3), False, True
    if xyz.dim() == 2:
        if xyz.shape == (n_atoms, 3):
            return xyz.reshape(1, n_atoms, 3), False, False
        if xyz.shape[1] == dim3n:
            return xyz.reshape(xyz.shape[0], n_atoms, 3), True, True
        raise ValueError(
            f"Expected coords shaped ({n_atoms}, 3), ({dim3n},), or (B, {dim3n}); "
            f"got {tuple(xyz.shape)}."
        )
    if xyz.dim() == 3 and xyz.shape[1:] == (n_atoms, 3):
        return xyz, True, False
    raise ValueError(
        f"Expected coords shaped ({n_atoms}, 3), ({dim3n},), (B, {n_atoms}, 3), "
        f"or (B, {dim3n}); got {tuple(xyz.shape)}."
    )


def _restore_direction_shape(
    direction_batch: torch.Tensor,
    was_batched: bool,
    return_flat: bool,
) -> torch.Tensor:
    if was_batched:
        if return_flat:
            return direction_batch.reshape(direction_batch.shape[0], -1)
        return direction_batch
    direction = direction_batch[0]
    return direction.reshape(-1) if return_flat else direction


def _count_negative_eigenvalues_batch(evals_vib: torch.Tensor) -> torch.Tensor:
    return (evals_vib < -NEG_EIGVAL_THRESHOLD).sum(dim=-1)


def _format_info(
    *,
    energy: torch.Tensor,
    forces: torch.Tensor,
    evals_vib: torch.Tensor,
    n_neg: torch.Tensor,
    blend_weight: torch.Tensor,
    info: dict[str, torch.Tensor | bool],
    was_batched: bool,
) -> dict[str, object]:
    phases = ["descent" if int(n.item()) > 1 else "gad" for n in n_neg]
    force_max_values = forces.reshape(forces.shape[0], -1).abs().max(dim=-1).values
    force_norm_values = forces.norm(dim=-1).mean(dim=-1)
    batch_info: dict[str, object] = {
        "phase": phases,
        "n_neg": n_neg.detach(),
        "eig0": evals_vib[:, 0].detach() if evals_vib.shape[1] > 0 else torch.zeros_like(n_neg),
        "eig1": evals_vib[:, 1].detach() if evals_vib.shape[1] > 1 else torch.zeros_like(n_neg),
        "energy": energy.detach(),
        "force_max": force_max_values.detach(),
        "force_norm": force_norm_values.detach(),
        "gad_blend_weight": blend_weight.detach(),
        **info,
    }
    if was_batched:
        return batch_info

    scalar_info: dict[str, object] = {}
    for key, value in batch_info.items():
        if isinstance(value, torch.Tensor):
            scalar_info[key] = float(value.reshape(-1)[0].item())
        elif isinstance(value, list):
            scalar_info[key] = value[0]
        else:
            scalar_info[key] = value
    scalar_info["n_neg"] = int(n_neg[0].item())
    return scalar_info


def _example_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Example optimization loop using TransitionStateVectorfield."
    )
    parser.add_argument("--n-atoms", type=int, default=7)
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--n-steps", type=int, default=2000)
    parser.add_argument("--dt", type=float, default=0.007)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--atomic-number", type=int, default=1)
    parser.add_argument("--gaussian-origin-sigma", type=float, default=1.0)
    parser.add_argument("--force-threshold", type=float, default=0.05)
    parser.add_argument("--force-criterion", choices=["fmax", "force_norm"], default="fmax")
    parser.add_argument("--max-atom-disp", type=float, default=0.05)
    parser.add_argument("--min-interatomic-dist", type=float, default=0.75)
    return parser.parse_args()


def _run_example() -> None:
    args = _example_args()
    if args.gaussian_origin_sigma <= 0:
        raise ValueError("--gaussian-origin-sigma must be positive.")

    generator = torch.Generator().manual_seed(args.seed)
    vectorfield = TransitionStateVectorfield(
        n_atoms=args.n_atoms,
        epsilon=args.epsilon,
        sigma=args.sigma,
        atomic_number=args.atomic_number,
    )

    print(
        "Batched TransitionStateVectorfield example | "
        "start=gaussian_origin gaussian_origin_sigma="
        f"{args.gaussian_origin_sigma:g} n_atoms={args.n_atoms} samples={args.n_samples}",
        flush=True,
    )
    for sample_id in range(args.n_samples):
        vectorfield.reset()
        coords = args.gaussian_origin_sigma * torch.randn(
            (args.n_atoms, 3),
            generator=generator,
            dtype=torch.float64,
        )
        coords = coords - coords.mean(dim=0, keepdim=True)
        converged = False
        final = _diagnostics(vectorfield, coords, force_criterion=args.force_criterion)

        for step_idx in range(args.n_steps):
            final = _diagnostics(vectorfield, coords, force_criterion=args.force_criterion)
            if is_ts_converged(
                int(final["n_neg"]),
                float(final["force_value"]),
                args.force_threshold,
                criterion=args.force_criterion,
            ):
                converged = True
                break

            direction = vectorfield(coords)
            step = cap_displacement(args.dt * direction, args.max_atom_disp)
            new_coords = coords + step
            if (
                args.min_interatomic_dist > 0
                and min_interatomic_distance(new_coords) < args.min_interatomic_dist
            ):
                step = 0.5 * step
                new_coords = coords + step
            coords = new_coords.detach()

        status = "CONV" if converged else "FAIL"
        total_steps = step_idx + 1 if args.n_steps > 0 else 0
        print(
            f"  [{sample_id:3d}] {status} | n_neg={final['n_neg']} "
            f"fmax={final['force_max']:.3e} energy={final['energy']:.6f} "
            f"steps={total_steps}",
            flush=True,
        )


if __name__ == "__main__":
    _run_example()


__all__ = ["TransitionStateVectorfield"]
