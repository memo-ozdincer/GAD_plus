"""Batched transition-state vector field for projected GAD."""
from __future__ import annotations

import argparse

import torch

from gadplus.calculator.lennard_jones import lj_atomic_nums, params_to_analytical_rm
from gadplus.calculator.lennard_jones_analytical import LennardJonesEnergy
from gadplus.core.convergence import NEG_EIGVAL_THRESHOLD
from gadplus.projection import (
    atomic_nums_to_symbols,
    batched_gad_dynamics_projected,
    batched_vib_eig,
)


class TransitionStateVectorfieldBatched:
    """Callable projected-GAD vector field for batched index-1 TS search.

    The vector field accepts only batches shaped ``(B, N, 3)`` or ``(B, 3N)``.
    Batched calls evaluate the Lennard-Jones force/Hessian and Eckart-projected
    vibrational eigensolve for the whole batch at once.
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

        coords_batch, return_flat = _coords_to_batch(coords, self.n_atoms)
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
        )
        return _restore_direction_shape(direction_batch, return_flat)

    def reset(self) -> None:
        """Clear cached diagnostics before starting an independent trajectory."""

        self.last_info = {}

    def _evaluate(self, coords: torch.Tensor) -> dict[str, torch.Tensor]:
        coords_batch, _return_flat = _coords_to_batch(coords, self.n_atoms)
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
    """Cap each batched sample's per-atom displacement to a maximum value."""

    if step_disp.dim() != 3 or step_disp.shape[-1] != 3:
        raise ValueError(f"Expected batched displacement shaped (B, N, 3), got {step_disp.shape}.")
    max_actual = step_disp.norm(dim=-1).amax(dim=-1)
    scale = torch.clamp(max_atom_disp / max_actual.clamp_min(1.0e-12), max=1.0)
    return step_disp * scale[:, None, None]


def min_interatomic_distance(coords: torch.Tensor) -> torch.Tensor:
    """Compute per-sample minimum interatomic distance for a coordinate batch."""

    if coords.dim() != 3 or coords.shape[-1] != 3:
        raise ValueError(f"Expected batched coords shaped (B, N, 3), got {coords.shape}.")
    batch_size, n_atoms, _ = coords.shape
    if n_atoms < 2:
        return torch.full((batch_size,), float("inf"), device=coords.device, dtype=coords.dtype)
    diff = coords[:, :, None, :] - coords[:, None, :, :]
    dist = diff.norm(dim=-1)
    eye = torch.eye(n_atoms, device=coords.device, dtype=torch.bool)
    return dist.masked_fill(eye[None, :, :], float("inf")).amin(dim=(1, 2))


def _diagnostics(
    vectorfield: TransitionStateVectorfieldBatched,
    coords: torch.Tensor,
    *,
    force_criterion: str,
) -> dict[str, torch.Tensor]:
    coords_batch, _return_flat = _coords_to_batch(coords, vectorfield.n_atoms)
    atomic_nums = vectorfield.atomic_nums.to(device=coords_batch.device)
    atomsymbols = atomic_nums_to_symbols(atomic_nums)
    out = vectorfield._evaluate(coords_batch)
    forces = out["forces"].reshape(coords_batch.shape[0], vectorfield.n_atoms, 3)
    evals_vib, _, _ = batched_vib_eig(
        out["hessian"],
        coords_batch,
        atomsymbols,
        purify=vectorfield.purify_hessian,
    )
    force_max_values = forces.reshape(forces.shape[0], -1).abs().amax(dim=-1)
    force_norm_values = forces.norm(dim=-1).mean(dim=-1)
    if force_criterion == "fmax":
        force_values = force_max_values
    elif force_criterion == "force_norm":
        force_values = force_norm_values
    else:
        raise ValueError(f"Unknown force criterion '{force_criterion}'.")
    return {
        "n_neg": _count_negative_eigenvalues_batch(evals_vib),
        "eig0": evals_vib[:, 0] if evals_vib.shape[1] > 0 else torch.zeros_like(force_values),
        "eig1": evals_vib[:, 1] if evals_vib.shape[1] > 1 else torch.zeros_like(force_values),
        "force_max": force_max_values,
        "force_norm": force_norm_values,
        "force_value": force_values,
        "energy": out["energy"],
    }


def _coords_to_batch(coords: torch.Tensor, n_atoms: int) -> tuple[torch.Tensor, bool]:
    xyz = coords.detach()
    dim3n = 3 * n_atoms
    if xyz.dim() == 2 and xyz.shape[1] == dim3n:
        return xyz.reshape(xyz.shape[0], n_atoms, 3), True
    if xyz.dim() == 3 and xyz.shape[1:] == (n_atoms, 3):
        return xyz, False
    raise ValueError(
        f"Expected batched coords shaped (B, {n_atoms}, 3) or (B, {dim3n}); "
        f"got {tuple(xyz.shape)}."
    )


def _restore_direction_shape(
    direction_batch: torch.Tensor,
    return_flat: bool,
) -> torch.Tensor:
    if return_flat:
        return direction_batch.reshape(direction_batch.shape[0], -1)
    return direction_batch


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
) -> dict[str, object]:
    phases = ["descent" if int(n.item()) > 1 else "gad" for n in n_neg]
    force_max_values = forces.reshape(forces.shape[0], -1).abs().max(dim=-1).values
    force_norm_values = forces.norm(dim=-1).mean(dim=-1)
    return {
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
    vectorfield = TransitionStateVectorfieldBatched(
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

    vectorfield.reset()
    coords = args.gaussian_origin_sigma * torch.randn(
        (args.n_samples, args.n_atoms, 3),
        generator=generator,
        dtype=torch.float64,
    )
    coords = coords - coords.mean(dim=1, keepdim=True)

    converged = torch.zeros(args.n_samples, dtype=torch.bool, device=coords.device)
    steps_taken = torch.zeros(args.n_samples, dtype=torch.long, device=coords.device)

    for _step_idx in range(args.n_steps):
        final = _diagnostics(vectorfield, coords, force_criterion=args.force_criterion)
        converged_now = (final["n_neg"] == 1) & (final["force_value"] < args.force_threshold)
        converged = converged | converged_now
        active = ~converged
        if not bool(active.any().item()):
            break

        active_coords = coords[active]
        direction = vectorfield(active_coords)
        step = cap_displacement(args.dt * direction, args.max_atom_disp)
        new_active_coords = active_coords + step
        if args.min_interatomic_dist > 0:
            too_close = min_interatomic_distance(new_active_coords) < args.min_interatomic_dist
            if bool(too_close.any().item()):
                backed_off = active_coords + 0.5 * step
                new_active_coords = torch.where(
                    too_close[:, None, None],
                    backed_off,
                    new_active_coords,
                )

        coords = coords.clone()
        coords[active] = new_active_coords.detach()
        steps_taken[active] += 1

    final = _diagnostics(vectorfield, coords, force_criterion=args.force_criterion)
    converged = (final["n_neg"] == 1) & (final["force_value"] < args.force_threshold)

    for sample_id in range(args.n_samples):
        status = "CONV" if bool(converged[sample_id].item()) else "FAIL"
        total_steps = int(steps_taken[sample_id].item())
        print(
            f"  [{sample_id:3d}] {status} | n_neg={int(final['n_neg'][sample_id].item())} "
            f"fmax={float(final['force_max'][sample_id].item()):.3e} "
            f"energy={float(final['energy'][sample_id].item()):.6f} "
            f"steps={total_steps}",
            flush=True,
        )


if __name__ == "__main__":
    _run_example()


__all__ = ["TransitionStateVectorfieldBatched"]
