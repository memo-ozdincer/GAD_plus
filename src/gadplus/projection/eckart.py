"""Eckart projection: remove translation/rotation null modes from Hessians.

Builds the 6 Eckart generators (3 translations, 3 rotations) in mass-weighted
space, then constructs a projector P = I - B(B^TB)^{-1}B^T that removes the
rigid-body subspace.

Also provides the reduced vibrational basis Q_vib: an explicit (3N, 3N-k)
orthonormal basis for the vibrational subspace, giving a full-rank Hessian
with no zero eigenvalues and no threshold-based filtering.
"""
from __future__ import annotations

import torch

from .masses import _to_torch_double


def _center_of_mass(coords3d: torch.Tensor, masses: torch.Tensor) -> torch.Tensor:
    total_mass = torch.sum(masses)
    return (coords3d * masses[:, None]).sum(dim=0) / total_mass


def eckart_B_massweighted_torch(
    cart_coords: torch.Tensor,
    masses: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Build the 6 Eckart generators in mass-weighted space.

    Returns B: (3N, 6) matrix whose columns span translations + rotations.
    No QR decomposition — smooth gradients via normalization only.
    """
    coords = _to_torch_double(cart_coords)
    masses = _to_torch_double(masses, device=coords.device)

    xyz = coords.reshape(-1, 3)
    N = xyz.shape[0]
    sqrt_m = torch.sqrt(masses)
    sqrt_m3 = sqrt_m.repeat_interleave(3)

    com = _center_of_mass(xyz, masses)
    r = xyz - com[None, :]

    # 3 translations
    ex = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64, device=coords.device)
    ey = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64, device=coords.device)
    ez = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64, device=coords.device)
    Tcols = []
    for e in (ex, ey, ez):
        col = sqrt_m3 * e.repeat(N)
        col = col / (col.norm() + eps)
        Tcols.append(col)

    # 3 rotations (infinitesimal cross products)
    rx, ry, rz = r[:, 0], r[:, 1], r[:, 2]
    R_ex = torch.stack([torch.zeros_like(rx), -rz, ry], dim=1)
    R_ey = torch.stack([rz, torch.zeros_like(ry), -rx], dim=1)
    R_ez = torch.stack([-ry, rx, torch.zeros_like(rz)], dim=1)
    Rcols = []
    for Raxis in (R_ex, R_ey, R_ez):
        col = (Raxis * sqrt_m[:, None]).reshape(-1)
        col = col / (col.norm() + eps)
        Rcols.append(col)

    B = torch.stack(Tcols + Rcols, dim=1)
    return B


def eckartprojection_torch(
    cart_coords: torch.Tensor,
    masses: torch.Tensor,
    eps: float = 1e-10,
) -> torch.Tensor:
    """Build the vibrational projector P in mass-weighted space.

    P = I - B (B^T B + eps I)^{-1} B^T

    Returns (3N, 3N) projector that zeroes out TR components.
    """
    B = eckart_B_massweighted_torch(cart_coords, masses, eps=eps)
    G = B.transpose(0, 1) @ B
    try:
        L = torch.linalg.cholesky(G + eps * torch.eye(6, dtype=G.dtype, device=G.device))
        Ginvt_Bt = torch.cholesky_solve(B.transpose(0, 1), L)
    except RuntimeError:
        Ginvt_Bt = torch.linalg.solve(
            G + eps * torch.eye(6, dtype=G.dtype, device=G.device),
            B.transpose(0, 1),
        )
    P = torch.eye(B.shape[0], dtype=B.dtype, device=B.device) - B @ Ginvt_Bt
    P = 0.5 * (P + P.transpose(0, 1))
    return P


def build_vibrational_basis_torch(
    cart_coords: torch.Tensor,
    masses: torch.Tensor,
    eps: float = 1e-12,
    linear_tol: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Build orthonormal vibrational basis Q_vib.

    Unlike the projector P which keeps the full (3N, 3N) space with 6 near-zero
    eigenvalues, this constructs an explicit (3N, 3N-k) orthonormal basis for
    the vibrational subspace, where k = 5 (linear) or 6 (non-linear).

    Returns:
        Q_vib: (3N, 3N-k) orthonormal vibrational columns.
        Q_tr: (3N, k) orthonormal TR columns.
        k: Number of TR modes (5 or 6).
    """
    B = eckart_B_massweighted_torch(cart_coords, masses, eps=eps)

    Q_full, R = torch.linalg.qr(B, mode="reduced")
    diag_R = torch.abs(torch.diag(R))
    valid_mask = diag_R > linear_tol
    k = max(int(valid_mask.sum().item()), 1)

    Q_tr = Q_full[:, :k]

    U, _, _ = torch.linalg.svd(Q_tr, full_matrices=True)
    Q_vib = U[:, k:]

    return Q_vib, Q_tr, k
