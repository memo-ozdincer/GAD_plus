# Copyright (c) Meta Platforms, Inc. and affiliates.

from __future__ import annotations

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch

from gadplus.calculator.lj_data_utils import subtract_com_vector


def _resolve_reference_path(path: str) -> Path:
    try:
        from hydra.utils import to_absolute_path
    except ImportError:
        return Path(path).expanduser().resolve()
    return Path(to_absolute_path(path))


def lennard_jones_pair_energy(
    distances: torch.Tensor,
    eps: float = 1.0,
    rm: float = 1.0,
) -> torch.Tensor:
    return eps * ((rm / distances) ** 12 - 2 * (rm / distances) ** 6)


class LennardJonesEnergy(torch.nn.Module):
    """Graph-native Lennard-Jones cluster energy for atoms-only BMS runs.

    LJ-13 on CUDA with batch size 128:
    autograd forces:            0.839 ms
    eager analytical forces:    0.172 ms
    compiled analytical forces: 0.046 ms
    autograd/compiled speedup:  18.2x

    compiled analytical Hessian: 0.051 ms
    autograd Hessian:            37.593 ms
    autograd/compiled speedup:   743.8x
    """

    def __init__(
        self,
        n_particles: int,
        spatial_dim: int = 3,
        eps: float = 1.0,
        rm: float = 1.0,
        oscillator: bool = True,
        oscillator_scale: float = 1.0,
        energy_factor: float = 1.0,
        min_distance: float = 1e-3,
        tau: float = 1.0,
        alpha: float = 1.0,
        device: str = "cpu",
        ref_samples_path: Optional[str] = None,
        compile_forces: bool = True,
        compile_hessian: bool = True,
        compile_mode: str = "reduce-overhead",
    ):
        super().__init__()
        self.n_particles = n_particles
        self.n_spatial_dim = spatial_dim
        self.spatial_dim = spatial_dim
        self.dim = n_particles * spatial_dim
        self.eps = eps
        self.rm = rm
        self.oscillator = oscillator
        self.oscillator_scale = oscillator_scale
        self.energy_factor = energy_factor
        self.min_distance = min_distance
        self.tau = tau
        self.alpha = alpha
        self.device = device
        self.ref_samples_path = ref_samples_path
        self.compile_forces = compile_forces
        self.compile_hessian = compile_hessian
        self.compile_mode = compile_mode
        self._compiled_forces_fn = None
        self._compiled_hessian_fn = None
        self.r_max = float("inf")
        self.default_regularize = False
        self.register_buffer("atomic_numbers", torch.tensor([1], dtype=torch.long))

    def _energy_from_positions(self, positions: torch.Tensor) -> torch.Tensor:
        samples = subtract_com_vector(
            positions.reshape(-1, self.n_particles, self.n_spatial_dim),
            self.n_particles,
            self.n_spatial_dim,
        )

        diff = samples[:, :, None, :] - samples[:, None, :, :]
        pair_mask = torch.triu(
            torch.ones(
                self.n_particles,
                self.n_particles,
                dtype=torch.bool,
                device=samples.device,
            ),
            diagonal=1,
        )
        distances = torch.linalg.norm(diff[:, pair_mask, :], dim=-1)
        distances = distances.clamp_min(self.min_distance)

        energy = lennard_jones_pair_energy(distances, self.eps, self.rm).sum(dim=-1)
        energy = energy * self.energy_factor
        if self.oscillator:
            oscillator_energy = samples.pow(2).sum(dim=(-2, -1))
            energy = energy + self.oscillator_scale * oscillator_energy
        return energy

    def eval_flat(self, samples: torch.Tensor) -> torch.Tensor:
        samples = subtract_com_vector(samples, self.n_particles, self.n_spatial_dim)
        positions = samples.reshape(-1, self.n_particles, self.n_spatial_dim)
        return self._energy_from_positions(positions)

    def load_reference_samples(
        self,
        max_samples: Optional[int] = None,
        device: Optional[torch.device | str] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        if self.ref_samples_path is None:
            raise ValueError("ref_samples_path is not configured")
        samples_np = np.load(
            _resolve_reference_path(self.ref_samples_path), allow_pickle=True
        )
        samples = torch.as_tensor(samples_np, dtype=dtype or torch.get_default_dtype())
        samples = samples.reshape(-1, self.dim)
        samples = subtract_com_vector(samples, self.n_particles, self.n_spatial_dim)
        if max_samples is not None:
            samples = samples[:max_samples]
        if device is not None:
            samples = samples.to(device)
        return samples

    def __call__(self, batch, regularize: Optional[bool] = None):
        del regularize
        positions = self._batched_positions(batch)
        energy = self._energy_from_positions(positions)
        forces = self.compute_forces(batch).reshape_as(batch["positions"])

        return {
            "energy": energy.detach(),
            "forces": forces.detach() / self.tau,
            "reg_energy": torch.zeros_like(energy.detach()),
            "reg_forces": torch.zeros_like(forces.detach()),
        }

    def compute_forces_autograd(self, batch) -> torch.Tensor:
        """Compute forces from autograd for validation."""
        with torch.enable_grad():
            positions = self._batched_positions(batch).detach().requires_grad_(True)
            energy = self._energy_from_positions(positions)
            forces = -torch.autograd.grad(energy.sum(), positions)[0]
        return forces.detach()

    def compute_forces(self, batch, create_graph: bool = False) -> torch.Tensor:
        """Compute per-particle analytical forces for the LJ energy."""
        positions = self._batched_positions(batch)
        if create_graph or not self._should_compile_forces(positions):
            forces = self._compute_forces_from_positions(positions)
        else:
            forces = self._compute_forces_compiled(positions)

        if create_graph:
            return forces
        return forces.detach()

    def _should_compile_forces(self, positions: torch.Tensor) -> bool:
        return (
            self.compile_forces
            and positions.is_cuda
            and hasattr(torch, "compile")
        )

    def _compute_forces_compiled(self, positions: torch.Tensor) -> torch.Tensor:
        if self._compiled_forces_fn is None:
            self._compiled_forces_fn = torch.compile(
                self._compute_forces_from_positions,
                mode=self.compile_mode,
            )
        return self._compiled_forces_fn(positions)

    def _compute_forces_from_positions(self, positions: torch.Tensor) -> torch.Tensor:
        samples = subtract_com_vector(
            positions,
            self.n_particles,
            self.n_spatial_dim,
        )
        n_systems, n_particles, _ = samples.shape

        diff = samples[:, :, None, :] - samples[:, None, :, :]
        distances = torch.linalg.norm(diff, dim=-1)
        active_pairs = ~torch.eye(
            n_particles,
            dtype=torch.bool,
            device=samples.device,
        )[None, :, :]
        active_pairs = active_pairs & (distances > self.min_distance)
        safe_distances = torch.where(
            active_pairs,
            distances.clamp_min(self.min_distance),
            torch.ones_like(distances),
        )

        rm6 = self.rm**6
        rm12 = rm6**2
        inv_r8 = safe_distances.pow(-8)
        inv_r14 = safe_distances.pow(-14)
        pair_force_scale = (
            self.eps
            * self.energy_factor
            * (12.0 * rm12 * inv_r14 - 12.0 * rm6 * inv_r8)
        )
        pair_forces = (
            pair_force_scale[..., None]
            * diff
            * active_pairs.to(samples.dtype)[..., None]
        ).sum(dim=2)

        forces = pair_forces
        if self.oscillator:
            forces = forces - 2.0 * self.oscillator_scale * samples

        forces = forces.reshape(n_systems, n_particles, self.n_spatial_dim)
        return forces

    def _batched_positions(self, batch) -> torch.Tensor:
        return batch["positions"].reshape(
            -1, self.n_particles, self.n_spatial_dim
        )

    def compute_hessian_autograd(
        self,
        batch,
        create_graph: bool = False,
    ) -> torch.Tensor:
        """Compute per-system Hessians of the LJ energy wrt particle positions."""
        with torch.enable_grad():
            positions = self._batched_positions(batch)
            positions = positions.detach().requires_grad_(True)
            energy = self._energy_from_positions(positions)
            grad = torch.autograd.grad(
                energy.sum(),
                positions,
                create_graph=True,
            )[0]

            n_systems = positions.shape[0]
            flat_grad = grad.reshape(n_systems, self.dim)
            hessian_rows = []
            for dim_i in range(self.dim):
                grad_i = flat_grad[:, dim_i].sum()
                row_i = torch.autograd.grad(
                    grad_i,
                    positions,
                    retain_graph=create_graph or dim_i < self.dim - 1,
                    create_graph=create_graph,
                )[0].reshape(n_systems, self.dim)
                hessian_rows.append(row_i)
            hessian = torch.stack(hessian_rows, dim=1)

        if create_graph:
            return hessian
        return hessian.detach()

    def compute_hessian(
        self,
        batch,
        create_graph: bool = False,
    ) -> torch.Tensor:
        """Compute per-system analytical Hessians of the LJ energy."""
        positions = self._batched_positions(batch)
        if create_graph or not self._should_compile_hessian(positions):
            hessian = self._compute_hessian_from_positions(positions)
        else:
            hessian = self._compute_hessian_compiled(positions)

        if create_graph:
            return hessian
        return hessian.detach()

    def _should_compile_hessian(self, positions: torch.Tensor) -> bool:
        return (
            self.compile_hessian
            and positions.is_cuda
            and hasattr(torch, "compile")
        )

    def _compute_hessian_compiled(self, positions: torch.Tensor) -> torch.Tensor:
        if self._compiled_hessian_fn is None:
            self._compiled_hessian_fn = torch.compile(
                self._compute_hessian_from_positions,
                mode=self.compile_mode,
            )
        return self._compiled_hessian_fn(positions)

    def _compute_hessian_from_positions(self, positions: torch.Tensor) -> torch.Tensor:
        n_systems, n_particles, spatial_dim = positions.shape
        eye_particles = torch.eye(
            n_particles,
            dtype=torch.bool,
            device=positions.device,
        )
        eye_spatial = torch.eye(
            spatial_dim,
            dtype=positions.dtype,
            device=positions.device,
        )

        diff = positions[:, :, None, :] - positions[:, None, :, :]
        distances = torch.linalg.norm(diff, dim=-1)
        upper_pairs = torch.triu(
            torch.ones(
                n_particles,
                n_particles,
                dtype=torch.bool,
                device=positions.device,
            ),
            diagonal=1,
        )
        active_pairs = upper_pairs.unsqueeze(0) & (
            distances > self.min_distance
        )
        safe_distances = torch.where(
            active_pairs,
            distances.clamp_min(self.min_distance),
            torch.ones_like(distances),
        )

        rm6 = self.rm**6
        rm12 = rm6**2
        inv_r2 = safe_distances.pow(-2)
        inv_r8 = safe_distances.pow(-8)
        inv_r14 = safe_distances.pow(-14)
        v_prime_over_r = self.eps * (-12.0 * rm12 * inv_r14 + 12.0 * rm6 * inv_r8)
        v_second = self.eps * (156.0 * rm12 * inv_r14 - 84.0 * rm6 * inv_r8)

        outer = diff[..., :, None] * diff[..., None, :] * inv_r2[..., None, None]
        pair_hessian = (
            v_prime_over_r[..., None, None] * eye_spatial
            + (v_second - v_prime_over_r)[..., None, None] * outer
        )
        pair_hessian = pair_hessian * (
            active_pairs.to(positions.dtype) * self.energy_factor
        )[..., None, None]

        off_diagonal_blocks = -(pair_hessian + pair_hessian.transpose(1, 2))
        diagonal_blocks = pair_hessian.sum(dim=2) + pair_hessian.sum(dim=1)
        hessian_blocks = torch.where(
            eye_particles[None, :, :, None, None],
            diagonal_blocks[:, :, None, :, :],
            off_diagonal_blocks,
        )

        if self.oscillator:
            projector = torch.eye(
                n_particles,
                dtype=positions.dtype,
                device=positions.device,
            )
            projector = projector - torch.full_like(projector, 1.0 / n_particles)
            oscillator_hessian = (
                2.0
                * self.oscillator_scale
                * projector[None, :, :, None, None]
                * eye_spatial[None, None, None, :, :]
            )
            hessian_blocks = hessian_blocks + oscillator_hessian

        hessian = hessian_blocks.permute(0, 1, 3, 2, 4).reshape(
            n_systems,
            self.dim,
            self.dim,
        )
        return hessian

    def to(self, *args, **kwargs):
        module = super().to(*args, **kwargs)
        device = kwargs.get("device", None)
        if device is None and args:
            device = args[0]
        if device is not None:
            self.device = str(device)
        return module
