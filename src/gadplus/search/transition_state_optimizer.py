"""Object-oriented transition-state optimizer for projected LJ GAD."""
from __future__ import annotations

import time
from dataclasses import dataclass

import torch

from gadplus.calculator.lennard_jones import (
    LennardJonesParams,
    lj_atomic_nums,
    make_lj_predict_fn,
)
from gadplus.core.adaptive_dt import (
    cap_displacement,
    compute_adaptive_dt,
    min_interatomic_distance,
)
from gadplus.core.convergence import (
    count_negative_eigenvalues,
    force_max,
    force_mean,
    force_value_from_criterion,
    is_ts_converged,
)
from gadplus.core.mode_tracking import pick_tracked_mode
from gadplus.core.types import PredictFn
from gadplus.projection import atomic_nums_to_symbols, gad_dynamics_projected, vib_eig


@dataclass(frozen=True)
class TransitionStateOptimizerConfig:
    """Settings for the dedicated projected-GAD transition-state optimizer."""

    n_steps: int = 1000
    dt: float = 1.0e-3
    k_track: int = 8
    use_adaptive_dt: bool = False
    dt_min: float = 1.0e-5
    dt_max: float = 0.05
    dt_adaptation: str = "eigenvalue_clamped"
    max_atom_disp: float = 0.05
    min_interatomic_dist: float = 0.75
    force_threshold: float = 1.0e-3
    force_criterion: str = "fmax"
    purify_hessian: bool = False
    return_weighted_step_direction: bool = False


@dataclass(frozen=True)
class TransitionStateOptimizationResult:
    """Result from a standalone transition-state optimization run."""

    converged: bool
    converged_step: int | None
    total_steps: int
    final_coords: torch.Tensor
    final_energy: float
    final_n_neg: int
    final_force_norm: float
    final_force_max: float
    final_eig0: float
    final_eig1: float
    wall_time_s: float
    failure_type: str | None = None


class TransitionStateOptimizer:
    """Projected GAD optimizer with gradient descent in high-index regions.

    This class corresponds to the ``lj_runner.py`` option combination
    ``--method gad --high-index-descent gradient --use-projection``. It keeps
    the optimization loop local to this module and stores the calculator as
    ``self.potential_energy_surface``.
    """

    def __init__(
        self,
        potential_energy_surface: PredictFn | None = None,
        *,
        n_atoms: int = 7,
        dt: float = 1.0e-3,
        epsilon: float = 1.0,
        sigma: float = 1.0,
        atomic_number: int = 18,
        atomic_nums: torch.Tensor | None = None,
        lj_compile: bool = False,
        n_steps: int = 1000,
        k_track: int = 8,
        use_adaptive_dt: bool = False,
        dt_min: float = 1.0e-5,
        dt_max: float = 0.05,
        dt_adaptation: str = "eigenvalue_clamped",
        max_atom_disp: float = 0.05,
        min_interatomic_dist: float = 0.75,
        force_threshold: float = 1.0e-3,
        force_criterion: str = "fmax",
        purify_hessian: bool = False,
        return_weighted_step_direction: bool = False,
    ) -> None:
        if n_atoms < 2:
            raise ValueError("TransitionStateOptimizer requires at least two atoms.")
        if dt <= 0:
            raise ValueError("dt must be positive.")
        if n_steps < 0:
            raise ValueError("n_steps must be non-negative.")

        self.n_atoms = n_atoms
        self.dt = dt
        self.lj_params = LennardJonesParams(epsilon=epsilon, sigma=sigma)
        self.atomic_nums = (
            atomic_nums.detach().clone()
            if atomic_nums is not None
            else lj_atomic_nums(n_atoms, atomic_number=atomic_number)
        )
        if self.atomic_nums.numel() != n_atoms:
            raise ValueError(
                f"atomic_nums must contain {n_atoms} entries, got {self.atomic_nums.numel()}."
            )

        self.potential_energy_surface = potential_energy_surface or make_lj_predict_fn(
            self.lj_params,
            n_atoms=n_atoms,
            compile_forces=lj_compile,
            compile_hessian=lj_compile,
        )
        self.config = TransitionStateOptimizerConfig(
            n_steps=n_steps,
            dt=dt,
            k_track=k_track,
            use_adaptive_dt=use_adaptive_dt,
            dt_min=dt_min,
            dt_max=dt_max,
            dt_adaptation=dt_adaptation,
            max_atom_disp=max_atom_disp,
            min_interatomic_dist=min_interatomic_dist,
            force_threshold=force_threshold,
            force_criterion=force_criterion,
            purify_hessian=purify_hessian,
            return_weighted_step_direction=return_weighted_step_direction,
        )

    def optimize(self, coords0: torch.Tensor) -> TransitionStateOptimizationResult:
        """Run the dedicated projected GAD optimizer from ``coords0``."""

        cfg = self.config
        coords = coords0.detach().clone().to(torch.float32).reshape(-1, 3)
        if coords.shape[0] != self.n_atoms:
            raise ValueError(f"Expected {self.n_atoms} atoms, got {coords.shape[0]}.")

        atomic_nums = self.atomic_nums.to(device=coords.device)
        atomsymbols = atomic_nums_to_symbols(atomic_nums)
        v_prev: torch.Tensor | None = None
        t_start = time.time()

        last_n_neg = 0
        last_force_norm = float("inf")
        last_force_max = float("inf")
        last_eig0 = 0.0
        last_eig1 = 0.0
        last_energy = 0.0

        for step_idx in range(cfg.n_steps):
            out = self.potential_energy_surface(
                coords,
                atomic_nums,
                do_hessian=True,
                require_grad=False,
            )
            forces = out["forces"]
            hessian = out["hessian"]

            if forces.dim() == 3 and forces.shape[0] == 1:
                forces = forces[0]
            forces = forces.reshape(-1, 3)

            energy = (
                float(out["energy"].detach().reshape(-1)[0].item())
                if isinstance(out["energy"], torch.Tensor)
                else float(out["energy"])
            )
            force_norm = force_mean(forces)
            force_max_value = force_max(forces)
            force_for_convergence = force_value_from_criterion(forces, cfg.force_criterion)

            evals_vib, evecs_vib_3n, _q_vib = vib_eig(
                hessian,
                coords,
                atomsymbols,
                purify=cfg.purify_hessian,
            )
            n_neg = count_negative_eigenvalues(evals_vib)
            eig0 = float(evals_vib[0].item()) if evals_vib.numel() > 0 else 0.0
            eig1 = float(evals_vib[1].item()) if evals_vib.numel() > 1 else 0.0

            last_n_neg = n_neg
            last_force_norm = force_norm
            last_force_max = force_max_value
            last_eig0 = eig0
            last_eig1 = eig1
            last_energy = energy

            if is_ts_converged(
                n_neg,
                force_for_convergence,
                cfg.force_threshold,
                criterion=cfg.force_criterion,
            ):
                return TransitionStateOptimizationResult(
                    converged=True,
                    converged_step=step_idx,
                    total_steps=step_idx + 1,
                    final_coords=coords.detach().cpu(),
                    final_energy=energy,
                    final_n_neg=n_neg,
                    final_force_norm=force_norm,
                    final_force_max=force_max_value,
                    final_eig0=eig0,
                    final_eig1=eig1,
                    wall_time_s=time.time() - t_start,
                )

            gad_vec, v_prev = self._projected_step_direction(
                coords=coords,
                forces=forces,
                evecs_vib_3n=evecs_vib_3n,
                atomsymbols=atomsymbols,
                n_neg=n_neg,
                v_prev=v_prev,
            )

            dt_eff = (
                compute_adaptive_dt(cfg.dt, cfg.dt_min, cfg.dt_max, cfg.dt_adaptation, eig0)
                if cfg.use_adaptive_dt
                else cfg.dt
            )
            step_disp = cap_displacement(dt_eff * gad_vec, cfg.max_atom_disp)
            new_coords = coords + step_disp
            if (
                cfg.min_interatomic_dist > 0
                and min_interatomic_distance(new_coords) < cfg.min_interatomic_dist
            ):
                step_disp = 0.5 * step_disp
                new_coords = coords + step_disp
            coords = new_coords.detach()

        return TransitionStateOptimizationResult(
            converged=False,
            converged_step=None,
            total_steps=cfg.n_steps,
            final_coords=coords.detach().cpu(),
            final_energy=last_energy,
            final_n_neg=last_n_neg,
            final_force_norm=last_force_norm,
            final_force_max=last_force_max,
            final_eig0=last_eig0,
            final_eig1=last_eig1,
            wall_time_s=time.time() - t_start,
        )

    def final_diagnostics(self, coords: torch.Tensor) -> dict[str, float | int]:
        """Return final energy, force, and projected-index diagnostics."""

        coords_3d = coords.detach().reshape(-1, 3)
        atomic_nums = self.atomic_nums.to(device=coords_3d.device)
        out = self.potential_energy_surface(
            coords_3d,
            atomic_nums,
            do_hessian=True,
            require_grad=False,
        )
        forces = out["forces"].reshape(-1, 3)
        hessian = out["hessian"].reshape(3 * coords_3d.shape[0], 3 * coords_3d.shape[0])
        evals_vib, _, _ = vib_eig(
            hessian,
            coords_3d,
            atomic_nums_to_symbols(atomic_nums),
            purify=self.config.purify_hessian,
        )
        eig0 = float(evals_vib[0].item()) if evals_vib.numel() > 0 else 0.0
        eig1 = float(evals_vib[1].item()) if evals_vib.numel() > 1 else 0.0
        return {
            "n_neg": count_negative_eigenvalues(evals_vib),
            "eig0": eig0,
            "eig1": eig1,
            "force_max": force_max(forces),
            "force_norm": force_mean(forces),
            "energy": float(out["energy"].detach().reshape(-1)[0].item()),
        }

    def _projected_step_direction(
        self,
        *,
        coords: torch.Tensor,
        forces: torch.Tensor,
        evecs_vib_3n: torch.Tensor,
        atomsymbols: list[str],
        n_neg: int,
        v_prev: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute projected descent for high-index states, otherwise projected GAD."""

        cfg = self.config
        k_eff = min(cfg.k_track, evecs_vib_3n.shape[1]) if cfg.k_track > 0 else evecs_vib_3n.shape[1]
        v_candidates = evecs_vib_3n[:, : max(k_eff, 1)].to(
            device=forces.device,
            dtype=forces.dtype,
        )
        v_prev_local = (
            v_prev.to(device=forces.device, dtype=forces.dtype).reshape(-1)
            if v_prev is not None
            else None
        )
        v, _idx, _overlap = pick_tracked_mode(v_candidates, v_prev_local, k=cfg.k_track)

        # This fixed blend implements high-index gradient descent for n_neg > 1,
        # then standard projected GAD once the projected index is at most one.
        blend_weight = 0.0 if n_neg > 1 else 1.0
        gad_vec, v_projected, _info = gad_dynamics_projected(
            coords=coords,
            forces=forces,
            v=v,
            atomsymbols=atomsymbols,
            gad_blend_weight=blend_weight,
            return_weighted_step_direction=cfg.return_weighted_step_direction,
        )
        return gad_vec, v_projected.detach().clone().reshape(-1)


__all__ = [
    "TransitionStateOptimizationResult",
    "TransitionStateOptimizer",
    "TransitionStateOptimizerConfig",
]
