"""Hessian projection and vibrational eigendecomposition.

The reduced-basis approach projects the (3N, 3N) raw Cartesian Hessian to a
full-rank (3N-k, 3N-k) vibrational Hessian with no zero eigenvalues. Every
eigenvalue returned is a genuine vibrational frequency — no threshold-based
filtering needed.
"""
from __future__ import annotations

import torch

from .masses import get_mass_weights_torch, mass_weigh_hessian_torch, _to_torch_double
from .eckart import build_vibrational_basis_torch


def purify_hessian_sum_rules_torch(hessian: torch.Tensor, n_atoms: int) -> torch.Tensor:
    """Enforce translational invariance sum rules on a Cartesian Hessian.

    ML-predicted Hessians violate sum_j H[i,a; j,b] = 0, causing residual
    TR eigenvalues (~5e-5) after Eckart projection. This distributes the
    row-sum error uniformly so the sum rules hold exactly.
    """
    dtype = torch.float64
    H = hessian.to(dtype=dtype)
    dim3N = 3 * n_atoms

    H_block = H.reshape(n_atoms, 3, n_atoms, 3)
    row_sums = H_block.sum(dim=(2, 3))
    correction = row_sums[:, :, None, None] / dim3N
    H_block = H_block - correction
    H_purified = H_block.reshape(dim3N, dim3N)
    H_purified = 0.5 * (H_purified + H_purified.transpose(0, 1))

    return H_purified


def reduced_basis_hessian_torch(
    hessian: torch.Tensor,
    cart_coords: torch.Tensor,
    atomsymbols: list[str],
    purify: bool = False,
) -> dict[str, torch.Tensor]:
    """Mass-weight and project Hessian to the full-rank vibrational subspace.

    Returns:
        dict with keys:
            H_red: (3N-k, 3N-k) full-rank vibrational Hessian.
            Q_vib: (3N, 3N-k) orthonormal vibrational basis.
            Q_tr:  (3N, k) orthonormal TR basis.
            k_tr:  int, number of TR modes (5 or 6).
            H_mw:  (3N, 3N) mass-weighted Hessian.
            masses, sqrt_m, sqrt_m_inv: mass tensors.
    """
    device = hessian.device
    dtype = torch.float64

    coords_3d = cart_coords.reshape(-1, 3).to(dtype)
    n_atoms = coords_3d.shape[0]

    masses_t, masses3d_t, sqrt_m, sqrt_m_inv = get_mass_weights_torch(
        atomsymbols, device=device, dtype=dtype,
    )

    H = hessian.to(dtype=dtype)
    if purify:
        H = purify_hessian_sum_rules_torch(H, n_atoms)

    H_mw = mass_weigh_hessian_torch(H, masses3d_t)
    Q_vib, Q_tr, k_tr = build_vibrational_basis_torch(coords_3d, masses_t)

    H_red = Q_vib.transpose(0, 1) @ H_mw @ Q_vib
    H_red = 0.5 * (H_red + H_red.transpose(0, 1))

    return {
        "H_red": H_red,
        "Q_vib": Q_vib,
        "Q_tr": Q_tr,
        "k_tr": k_tr,
        "H_mw": H_mw,
        "masses": masses_t,
        "sqrt_m": sqrt_m,
        "sqrt_m_inv": sqrt_m_inv,
    }


def vib_eig(
    hessian: torch.Tensor,
    coords: torch.Tensor,
    atomsymbols: list[str],
    purify: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Vibrational eigenvalues and eigenvectors via the reduced basis.

    Every returned eigenvalue is a genuine vibrational frequency. No
    threshold-based TR filtering — the TR modes are removed by construction.

    Args:
        hessian: (3N, 3N) raw Cartesian Hessian.
        coords: (N, 3) atomic coordinates.
        atomsymbols: Element symbols ['C', 'H', ...].
        purify: If True, enforce translational sum rules before projecting.

    Returns:
        evals_vib:    (3N-k,) vibrational eigenvalues, ascending.
        evecs_vib_3N: (3N, 3N-k) eigenvectors in full Cartesian space.
        Q_vib:        (3N, 3N-k) orthonormal vibrational basis.
    """
    n_atoms = coords.reshape(-1, 3).shape[0]
    hess = hessian.reshape(3 * n_atoms, 3 * n_atoms)
    rb = reduced_basis_hessian_torch(hess, coords.reshape(-1, 3), atomsymbols, purify=purify)
    evals_vib, evecs_red = torch.linalg.eigh(rb["H_red"])
    evecs_vib_3N = rb["Q_vib"] @ evecs_red
    return evals_vib, evecs_vib_3N, rb["Q_vib"]
