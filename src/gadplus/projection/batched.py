"""Batched Eckart projection and projected GAD helpers."""
from __future__ import annotations

import torch

from gadplus.projection.projection import get_mass_weights


def batched_purify_hessian(hessian: torch.Tensor, n_atoms: int) -> torch.Tensor:
    """Enforce translational sum rules for batched Cartesian Hessians."""

    hessian_batch = _as_batched_hessian(hessian, n_atoms).to(dtype=torch.float64)
    dim3n = 3 * n_atoms
    h_blocks = hessian_batch.reshape(-1, n_atoms, 3, n_atoms, 3)
    row_sums = h_blocks.sum(dim=(3, 4))
    h_purified = h_blocks - row_sums[:, :, :, None, None] / dim3n
    h_purified = h_purified.reshape(-1, dim3n, dim3n)
    return 0.5 * (h_purified + h_purified.transpose(-1, -2))


def batched_vib_eig(
    hessian: torch.Tensor,
    coords: torch.Tensor,
    atomsymbols: list[str],
    purify: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Batched vibrational eigendecomposition via a reduced Eckart basis.

    This vectorized path removes all six rigid-body generators for each geometry,
    so it assumes non-linear structures and returns fixed-size tensors:
    ``evals_vib`` has shape ``(B, 3N - 6)`` and ``evecs_vib_3n`` has shape
    ``(B, 3N, 3N - 6)``.
    """

    coords_batch = _as_batched_coords(coords).to(dtype=torch.float64)
    batch_size, n_atoms, _ = coords_batch.shape
    hessian_batch = _as_batched_hessian(hessian, n_atoms).to(
        device=coords_batch.device,
        dtype=torch.float64,
    )
    if hessian_batch.shape[0] != batch_size:
        raise ValueError(
            f"Expected hessian batch size {batch_size}, got {hessian_batch.shape[0]}."
        )

    masses, _m3, _sqrt_m, sqrt_m_inv = get_mass_weights(
        atomsymbols,
        device=coords_batch.device,
        dtype=torch.float64,
    )
    if masses.numel() != n_atoms:
        raise ValueError(f"Expected {n_atoms} atom symbols, got {masses.numel()}.")

    hessian_batch = batched_purify_hessian(hessian_batch, n_atoms) if purify else hessian_batch
    h_mw = sqrt_m_inv[None, :, None] * hessian_batch * sqrt_m_inv[None, None, :]

    q_vib, _q_tr = batched_vibrational_basis(coords_batch, masses)
    h_red = q_vib.transpose(-1, -2) @ h_mw @ q_vib
    h_red = 0.5 * (h_red + h_red.transpose(-1, -2))

    evals, evecs_red = torch.linalg.eigh(h_red)
    evecs_3n = q_vib @ evecs_red
    return evals, evecs_3n, q_vib


def batched_gad_dynamics_projected(
    coords: torch.Tensor,
    forces: torch.Tensor,
    v: torch.Tensor,
    atomsymbols: list[str],
    gad_blend_weight: float | torch.Tensor = 1.0,
    return_weighted_step_direction: bool = False,
    eps: float = 1e-10,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor | bool]]:
    """Batched projected GAD direction with consistent Eckart projection."""

    coords_batch = _as_batched_coords(coords).to(dtype=torch.float64)
    batch_size, n_atoms, _ = coords_batch.shape
    dim3n = 3 * n_atoms
    forces_batch = _as_batched_forces(forces, n_atoms).to(
        device=coords_batch.device,
        dtype=torch.float64,
    )
    v_batch = _as_batched_vector(v, dim3n).to(device=coords_batch.device, dtype=torch.float64)
    if forces_batch.shape[0] != batch_size:
        raise ValueError(f"Expected forces batch size {batch_size}, got {forces_batch.shape[0]}.")
    if v_batch.shape[0] != batch_size:
        raise ValueError(f"Expected guide-vector batch size {batch_size}, got {v_batch.shape[0]}.")

    masses, _m3, sqrt_m, sqrt_m_inv = get_mass_weights(
        atomsymbols,
        device=coords_batch.device,
        dtype=torch.float64,
    )
    if masses.numel() != n_atoms:
        raise ValueError(f"Expected {n_atoms} atom symbols, got {masses.numel()}.")

    projector = batched_eckart_projector(coords_batch, masses, eps=eps)
    f_flat = forces_batch.reshape(batch_size, dim3n)
    grad_mw = torch.matmul(projector, (-sqrt_m_inv[None, :] * f_flat).unsqueeze(-1)).squeeze(-1)
    v_proj = torch.matmul(projector, v_batch.unsqueeze(-1)).squeeze(-1)
    v_proj = v_proj / (v_proj.norm(dim=-1, keepdim=True) + 1e-12)

    v_dot_grad = (v_proj * grad_mw).sum(dim=-1)
    v_dot_v = (v_proj * v_proj).sum(dim=-1)
    weights = _as_batch_weight(gad_blend_weight, batch_size, coords_batch.device)
    dq = -grad_mw + 2.0 * weights[:, None] * (v_dot_grad / (v_dot_v + 1e-12))[:, None] * v_proj
    dq = torch.matmul(projector, dq.unsqueeze(-1)).squeeze(-1)

    step_scale = sqrt_m if return_weighted_step_direction else sqrt_m_inv
    gad_vec = (step_scale[None, :] * dq).reshape(batch_size, n_atoms, 3).to(forces.dtype)
    info = {
        "v_dot_grad": v_dot_grad,
        "grad_norm_mw": grad_mw.norm(dim=-1),
        "gad_blend_weight": weights,
        "return_weighted_step_direction": bool(return_weighted_step_direction),
    }
    return gad_vec, v_proj.to(v.dtype), info


def batched_project_vector_to_vibrational(
    vec: torch.Tensor,
    cart_coords: torch.Tensor,
    atomsymbols: list[str],
    eps: float = 1e-10,
) -> torch.Tensor:
    """Project a batch of Cartesian vectors to remove translation/rotation components."""

    coords_batch = _as_batched_coords(cart_coords).to(dtype=torch.float64)
    batch_size, n_atoms, _ = coords_batch.shape
    dim3n = 3 * n_atoms
    vec_batch = _as_batched_vector(vec, dim3n).to(device=coords_batch.device, dtype=torch.float64)
    if vec_batch.shape[0] != batch_size:
        raise ValueError(f"Expected vector batch size {batch_size}, got {vec_batch.shape[0]}.")

    masses, _m3, sqrt_m, sqrt_m_inv = get_mass_weights(
        atomsymbols,
        device=coords_batch.device,
        dtype=torch.float64,
    )
    projector = batched_eckart_projector(coords_batch, masses, eps=eps)
    vec_mw = sqrt_m_inv[None, :] * vec_batch
    vec_proj_mw = torch.matmul(projector, vec_mw.unsqueeze(-1)).squeeze(-1)
    return (sqrt_m[None, :] * vec_proj_mw).to(vec.dtype)


def batched_eckart_projector(
    cart_coords: torch.Tensor,
    masses: torch.Tensor,
    eps: float = 1e-10,
) -> torch.Tensor:
    """Build batched vibrational projectors in mass-weighted space."""

    b_matrix = batched_eckart_generators(cart_coords, masses, eps=eps)
    gram = b_matrix.transpose(-1, -2) @ b_matrix
    eye6 = torch.eye(6, dtype=gram.dtype, device=gram.device).expand(gram.shape[0], -1, -1)
    chol = torch.linalg.cholesky(gram + eps * eye6)
    ginv_bt = torch.cholesky_solve(b_matrix.transpose(-1, -2), chol)
    dim3n = b_matrix.shape[1]
    eye3n = torch.eye(dim3n, dtype=b_matrix.dtype, device=b_matrix.device).expand(
        b_matrix.shape[0],
        -1,
        -1,
    )
    projector = eye3n - b_matrix @ ginv_bt
    return 0.5 * (projector + projector.transpose(-1, -2))


def batched_vibrational_basis(
    cart_coords: torch.Tensor,
    masses: torch.Tensor,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build batched non-linear vibrational bases with six TR modes removed."""

    b_matrix = batched_eckart_generators(cart_coords, masses, eps=eps)
    q_full, _r = torch.linalg.qr(b_matrix, mode="reduced")
    q_tr = q_full[:, :, :6]
    u_matrix, _s, _vh = torch.linalg.svd(q_tr, full_matrices=True)
    return u_matrix[:, :, 6:], u_matrix[:, :, :6]


def batched_eckart_generators(
    cart_coords: torch.Tensor,
    masses: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Build six batched Eckart generators in mass-weighted space."""

    coords_batch = _as_batched_coords(cart_coords).to(dtype=torch.float64)
    masses = masses.to(device=coords_batch.device, dtype=torch.float64)
    batch_size, n_atoms, _ = coords_batch.shape
    if masses.numel() != n_atoms:
        raise ValueError(f"Expected {n_atoms} masses, got {masses.numel()}.")

    sqrt_m = torch.sqrt(masses)
    sqrt_m3 = sqrt_m.repeat_interleave(3)
    com = (coords_batch * masses[None, :, None]).sum(dim=1) / masses.sum()
    rel_coords = coords_batch - com[:, None, :]

    unit_axes = torch.eye(3, dtype=torch.float64, device=coords_batch.device)
    trans_cols = []
    for axis in unit_axes:
        col = sqrt_m3 * axis.repeat(n_atoms)
        col = col / (col.norm() + eps)
        trans_cols.append(col.expand(batch_size, -1))

    rx, ry, rz = rel_coords[:, :, 0], rel_coords[:, :, 1], rel_coords[:, :, 2]
    rot_axes = (
        torch.stack([torch.zeros_like(rx), -rz, ry], dim=-1),
        torch.stack([rz, torch.zeros_like(ry), -rx], dim=-1),
        torch.stack([-ry, rx, torch.zeros_like(rz)], dim=-1),
    )
    rot_cols = []
    for rot_axis in rot_axes:
        col = (rot_axis * sqrt_m[None, :, None]).reshape(batch_size, -1)
        col = col / (col.norm(dim=-1, keepdim=True) + eps)
        rot_cols.append(col)

    return torch.stack([*trans_cols, *rot_cols], dim=-1)


def _as_batched_coords(coords: torch.Tensor) -> torch.Tensor:
    if coords.dim() == 2 and coords.shape[-1] == 3:
        return coords.unsqueeze(0)
    if coords.dim() == 3 and coords.shape[-1] == 3:
        return coords
    raise ValueError(f"Expected coords shaped (N, 3) or (B, N, 3), got {tuple(coords.shape)}.")


def _as_batched_hessian(hessian: torch.Tensor, n_atoms: int) -> torch.Tensor:
    dim3n = 3 * n_atoms
    if hessian.dim() == 2 and hessian.shape == (dim3n, dim3n):
        return hessian.unsqueeze(0)
    if hessian.dim() == 3 and hessian.shape[-2:] == (dim3n, dim3n):
        return hessian
    raise ValueError(
        f"Expected hessian shaped ({dim3n}, {dim3n}) or (B, {dim3n}, {dim3n}), "
        f"got {tuple(hessian.shape)}."
    )


def _as_batched_forces(forces: torch.Tensor, n_atoms: int) -> torch.Tensor:
    dim3n = 3 * n_atoms
    if forces.dim() == 1 and forces.numel() == dim3n:
        return forces.reshape(1, n_atoms, 3)
    if forces.dim() == 2 and forces.shape == (n_atoms, 3):
        return forces.unsqueeze(0)
    if forces.dim() == 2 and forces.shape[-1] == dim3n:
        return forces.reshape(forces.shape[0], n_atoms, 3)
    if forces.dim() == 3 and forces.shape[-2:] == (n_atoms, 3):
        return forces
    raise ValueError(
        f"Expected forces shaped ({n_atoms}, 3), ({dim3n},), (B, {n_atoms}, 3), "
        f"or (B, {dim3n}), got {tuple(forces.shape)}."
    )


def _as_batched_vector(vector: torch.Tensor, dim3n: int) -> torch.Tensor:
    n_atoms = dim3n // 3
    if vector.dim() == 1 and vector.numel() == dim3n:
        return vector.reshape(1, dim3n)
    if vector.dim() == 2 and vector.shape == (n_atoms, 3):
        return vector.reshape(1, dim3n)
    if vector.dim() == 2 and vector.shape[-1] == dim3n:
        return vector
    if vector.dim() == 3 and vector.shape[-2:] == (n_atoms, 3):
        return vector.reshape(vector.shape[0], dim3n)
    raise ValueError(
        f"Expected vector shaped ({dim3n},), ({n_atoms}, 3), (B, {dim3n}), "
        f"or (B, {n_atoms}, 3), got {tuple(vector.shape)}."
    )


def _as_batch_weight(
    weight: float | torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    weights = torch.as_tensor(weight, dtype=torch.float64, device=device)
    if weights.dim() == 0:
        return weights.expand(batch_size)
    if weights.shape == (batch_size,):
        return weights
    raise ValueError(f"Expected scalar weight or ({batch_size},), got {tuple(weights.shape)}.")
