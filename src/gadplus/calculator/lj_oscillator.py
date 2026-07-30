"""Optional tether terms for analytical Lennard-Jones cluster energies."""
from __future__ import annotations

import torch

OSCILLATOR_MODES = frozenset({"off", "linear", "deadzone", "pair", "switch", "quartic"})


def normalize_oscillator_mode(mode: str) -> str:
    normalized = mode.strip().lower()
    if normalized not in OSCILLATOR_MODES:
        raise ValueError(
            f"Unknown oscillator mode {mode!r}; expected one of {sorted(OSCILLATOR_MODES)}"
        )
    return normalized


def cluster_length_scale(rm: float, n_particles: int) -> float:
    """Reference cluster length ``r_eq * N^(1/3)``."""

    return rm * (n_particles ** (1.0 / 3.0))


def _pair_spring_force_scale(
    distances: torch.Tensor,
    *,
    cutoff: float,
    scale: float,
) -> torch.Tensor:
    """Return ``2 k max(0, d - r_cut)`` for active pair distances."""

    return 2.0 * scale * (distances - cutoff).clamp_min(0.0)


def oscillator_energy(
    samples: torch.Tensor,
    distances: torch.Tensor,
    *,
    mode: str,
    scale: float,
    rm: float,
    n_particles: int,
    r0_factor: float,
    rcut_factor: float,
    switch_width_factor: float,
) -> torch.Tensor:
    """Batch oscillator energy contribution ``(B,)``."""

    mode = normalize_oscillator_mode(mode)
    if mode == "off" or scale == 0.0:
        return torch.zeros(samples.shape[0], device=samples.device, dtype=samples.dtype)

    r_cluster = cluster_length_scale(rm, n_particles)
    if mode == "linear":
        return scale * samples.pow(2).sum(dim=(-2, -1))
    if mode == "quartic":
        radii_sq = samples.pow(2).sum(dim=-1)
        return scale * radii_sq.pow(2).sum(dim=-1)
    if mode == "deadzone":
        r0 = r0_factor * r_cluster
        radii = samples.norm(dim=-1)
        excess = (radii - r0).clamp_min(0.0)
        return scale * excess.pow(2).sum(dim=-1)
    if mode == "pair":
        rcut = rcut_factor * r_cluster
        upper = torch.triu(
            torch.ones(n_particles, n_particles, dtype=torch.bool, device=samples.device),
            diagonal=1,
        )
        pair_dist = distances[:, upper]
        excess = (pair_dist - rcut).clamp_min(0.0)
        return scale * excess.pow(2).sum(dim=-1)
    if mode == "switch":
        r_on = r0_factor * r_cluster
        width = max(switch_width_factor * rm, 1.0e-12)
        upper = torch.triu(
            torch.ones(n_particles, n_particles, dtype=torch.bool, device=samples.device),
            diagonal=1,
        )
        d_max = distances[:, upper].amax(dim=-1)
        weight = torch.sigmoid((d_max - r_on) / width)
        return weight * scale * samples.pow(2).sum(dim=(-2, -1))

    raise AssertionError(f"Unhandled oscillator mode: {mode}")


def oscillator_forces(
    samples: torch.Tensor,
    diff: torch.Tensor,
    distances: torch.Tensor,
    *,
    mode: str,
    scale: float,
    rm: float,
    n_particles: int,
    r0_factor: float,
    rcut_factor: float,
    switch_width_factor: float,
    min_distance: float,
) -> torch.Tensor:
    """Oscillator force contribution with shape ``(B, N, 3)``."""

    mode = normalize_oscillator_mode(mode)
    if mode == "off" or scale == 0.0:
        return torch.zeros_like(samples)

    r_cluster = cluster_length_scale(rm, n_particles)
    if mode == "linear":
        return -2.0 * scale * samples
    if mode == "quartic":
        radii_sq = samples.pow(2).sum(dim=-1, keepdim=True)
        return -4.0 * scale * radii_sq * samples
    if mode == "deadzone":
        r0 = r0_factor * r_cluster
        radii = samples.norm(dim=-1, keepdim=True).clamp_min(min_distance)
        excess = (radii - r0).clamp_min(0.0)
        unit = samples / radii
        return -2.0 * scale * excess * unit
    if mode == "pair":
        rcut = rcut_factor * r_cluster
        active = torch.triu(
            torch.ones(n_particles, n_particles, dtype=torch.bool, device=samples.device),
            diagonal=1,
        )[None, :, :]
        safe = distances.clamp_min(min_distance)
        force_scale = _pair_spring_force_scale(
            safe,
            cutoff=rcut,
            scale=scale,
        )
        pair_forces = (
            force_scale[..., None]
            * diff
            / safe[..., None]
            * active.to(samples.dtype)[..., None]
        ).sum(dim=2)
        return -pair_forces
    if mode == "switch":
        r_on = r0_factor * r_cluster
        width = max(switch_width_factor * rm, 1.0e-12)
        upper = torch.triu(
            torch.ones(n_particles, n_particles, dtype=torch.bool, device=samples.device),
            diagonal=1,
        )
        d_max = distances[:, upper].amax(dim=-1)
        weight = torch.sigmoid((d_max - r_on) / width)
        return -2.0 * scale * weight[:, None, None] * samples

    raise AssertionError(f"Unhandled oscillator mode: {mode}")


def oscillator_hessian_blocks(
    samples: torch.Tensor,
    diff: torch.Tensor,
    distances: torch.Tensor,
    *,
    mode: str,
    scale: float,
    rm: float,
    n_particles: int,
    spatial_dim: int,
    r0_factor: float,
    rcut_factor: float,
    switch_width_factor: float,
    min_distance: float,
) -> torch.Tensor:
    """Oscillator Hessian block contribution ``(B, N, N, 3, 3)``."""

    mode = normalize_oscillator_mode(mode)
    dtype = samples.dtype
    device = samples.device
    eye_particles = torch.eye(n_particles, dtype=torch.bool, device=device)
    eye_spatial = torch.eye(spatial_dim, dtype=dtype, device=device)
    zero = torch.zeros(
        (samples.shape[0], n_particles, n_particles, spatial_dim, spatial_dim),
        dtype=dtype,
        device=device,
    )
    if mode == "off" or scale == 0.0:
        return zero

    r_cluster = cluster_length_scale(rm, n_particles)
    if mode in {"linear", "switch"}:
        projector = torch.eye(n_particles, dtype=dtype, device=device)
        projector = projector - torch.full_like(projector, 1.0 / n_particles)
        weight = torch.ones(samples.shape[0], dtype=dtype, device=device)
        if mode == "switch":
            r_on = r0_factor * r_cluster
            width = max(switch_width_factor * rm, 1.0e-12)
            upper = torch.triu(
                torch.ones(n_particles, n_particles, dtype=torch.bool, device=device),
                diagonal=1,
            )
            d_max = distances[:, upper].amax(dim=-1)
            weight = torch.sigmoid((d_max - r_on) / width)
        coeff = (
            2.0
            * scale
            * weight[:, None, None, None, None]
            * projector[None, :, :, None, None]
            * eye_spatial[None, None, None, :, :]
        )
        return torch.where(eye_particles[None, :, :, None, None], coeff, zero)

    if mode == "quartic":
        radii_sq = samples.pow(2).sum(dim=-1).clamp_min(min_distance**2)
        hessian_blocks = torch.zeros_like(zero)
        for atom_idx in range(n_particles):
            r = samples[:, atom_idx, :]
            r2 = radii_sq[:, atom_idx]
            block = (
                4.0 * scale * r2[:, None, None] * eye_spatial[None, :, :]
                + 8.0 * scale * r[:, :, None] * r[:, None, :]
            )
            hessian_blocks[:, atom_idx, atom_idx] = block
        return hessian_blocks

    if mode == "deadzone":
        r0 = r0_factor * r_cluster
        radii = samples.norm(dim=-1).clamp_min(min_distance)
        active = radii > r0
        unit = samples / radii.unsqueeze(-1)
        outer = unit[..., :, None] * unit[..., None, :]
        radial_coeff = (2.0 * scale * (radii - r0).clamp_min(0.0) / radii).clamp_min(0.0)
        block_diag = (
            radial_coeff[:, :, None, None] * eye_spatial[None, None, :, :]
            + 2.0 * scale * outer * active[:, :, None, None].to(dtype)
        )
        hessian_blocks = torch.zeros_like(zero)
        for atom_idx in range(n_particles):
            hessian_blocks[:, atom_idx, atom_idx] = block_diag[:, atom_idx]
        projector = torch.eye(n_particles, dtype=dtype, device=device)
        projector = projector - torch.full_like(projector, 1.0 / n_particles)
        return hessian_blocks * projector[None, :, :, None, None]

    if mode == "pair":
        rcut = rcut_factor * r_cluster
        upper_pairs = torch.triu(
            torch.ones(n_particles, n_particles, dtype=torch.bool, device=device),
            diagonal=1,
        )
        active_pairs = upper_pairs.unsqueeze(0) & (distances > rcut)
        safe = distances.clamp_min(min_distance)
        v_second = torch.full_like(safe, 2.0 * scale)
        v_prime_over_r = torch.zeros_like(safe)
        active_f = active_pairs.to(dtype)
        v_prime_over_r = torch.where(
            active_f > 0,
            2.0 * scale * (safe - rcut) / safe,
            v_prime_over_r,
        )
        inv_r2 = safe.pow(-2)
        outer = diff[..., :, None] * diff[..., None, :] * inv_r2[..., None, None]
        pair_hessian = (
            v_prime_over_r[..., None, None] * eye_spatial
            + (v_second - v_prime_over_r)[..., None, None] * outer
        )
        pair_hessian = pair_hessian * active_f[..., None, None]
        off_diagonal_blocks = -(pair_hessian + pair_hessian.transpose(1, 2))
        diagonal_blocks = pair_hessian.sum(dim=2) + pair_hessian.sum(dim=1)
        return torch.where(
            eye_particles[None, :, :, None, None],
            diagonal_blocks[:, :, None, :, :],
            off_diagonal_blocks,
        )

    raise AssertionError(f"Unhandled oscillator mode: {mode}")
