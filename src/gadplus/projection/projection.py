"""Differentiable Eckart projection for mass-weighted Hessians.

All operations use pure torch — autograd flows through the entire pipeline.
This is critical for HIP's require_grad=True path.

Provides:
    - Mass-weighting and Eckart projection (remove translation/rotation modes)
    - Reduced-basis vibrational Hessian (full-rank, no threshold filtering)
    - Vibrational eigendecomposition (vib_eig)
    - Projected GAD dynamics (gad_dynamics_projected_torch)
    - Vector projection to vibrational subspace
"""
from __future__ import annotations

import torch


# =============================================================================
# Mass data and utilities
# =============================================================================

MASS_DICT: dict[str, float] = {
    "h": 1.008, "he": 4.003,
    "li": 6.941, "be": 9.012, "b": 10.81, "c": 12.011, "n": 14.007,
    "o": 15.999, "f": 18.998, "ne": 20.180,
    "na": 22.990, "mg": 24.305, "al": 26.982, "si": 28.086, "p": 30.974,
    "s": 32.065, "cl": 35.453, "ar": 39.948,
    "k": 39.098, "ca": 40.078,
    "br": 79.904, "i": 126.904,
}

Z_TO_SYMBOL: dict[int, str] = {
    1: "H", 6: "C", 7: "N", 8: "O", 9: "F", 15: "P", 16: "S",
    17: "Cl", 35: "Br", 53: "I",
}


def atomic_nums_to_symbols(atomic_nums: torch.Tensor) -> list[str]:
    """Convert atomic number tensor to element symbol list."""
    nums = atomic_nums.detach().cpu().tolist()
    return [Z_TO_SYMBOL.get(int(z), "X") for z in nums]


def _to_f64(array_like, device=None):
    if isinstance(array_like, torch.Tensor):
        return array_like.to(dtype=torch.float64, device=device)
    return torch.as_tensor(array_like, dtype=torch.float64, device=device)


def get_mass_weights(
    atomsymbols: list[str], device=None, dtype=torch.float64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (masses_N, masses_3N, sqrt_m_3N, sqrt_m_inv_3N)."""
    masses = torch.tensor(
        [MASS_DICT[a.lower()] for a in atomsymbols], dtype=dtype, device=device,
    )
    m3 = masses.repeat_interleave(3)
    return masses, m3, torch.sqrt(m3), 1.0 / torch.sqrt(m3)


# =============================================================================
# Eckart projection internals
# =============================================================================

def _eckart_generators(coords: torch.Tensor, masses: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Build 6 Eckart generators (3 trans + 3 rot) in mass-weighted space. Returns (3N, 6)."""
    xyz = _to_f64(coords).reshape(-1, 3)
    masses = _to_f64(masses, device=xyz.device)
    N = xyz.shape[0]
    sqrt_m = torch.sqrt(masses)
    sqrt_m3 = sqrt_m.repeat_interleave(3)

    com = (xyz * masses[:, None]).sum(0) / masses.sum()
    r = xyz - com[None, :]

    # Translations
    cols = []
    for e in (torch.tensor([1, 0, 0], dtype=torch.float64, device=xyz.device),
              torch.tensor([0, 1, 0], dtype=torch.float64, device=xyz.device),
              torch.tensor([0, 0, 1], dtype=torch.float64, device=xyz.device)):
        c = sqrt_m3 * e.repeat(N)
        cols.append(c / (c.norm() + eps))

    # Rotations
    rx, ry, rz = r[:, 0], r[:, 1], r[:, 2]
    for Raxis in (torch.stack([torch.zeros_like(rx), -rz, ry], 1),
                  torch.stack([rz, torch.zeros_like(ry), -rx], 1),
                  torch.stack([-ry, rx, torch.zeros_like(rz)], 1)):
        c = (Raxis * sqrt_m[:, None]).reshape(-1)
        cols.append(c / (c.norm() + eps))

    return torch.stack(cols, dim=1)


def _eckart_projector(coords: torch.Tensor, masses: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    """Vibrational projector P = I - B(B^TB + εI)^{-1}B^T in MW space. Returns (3N, 3N)."""
    B = _eckart_generators(coords, masses, eps=eps)
    G = B.T @ B
    try:
        L = torch.linalg.cholesky(G + eps * torch.eye(6, dtype=G.dtype, device=G.device))
        GiBt = torch.cholesky_solve(B.T, L)
    except RuntimeError:
        GiBt = torch.linalg.solve(G + eps * torch.eye(6, dtype=G.dtype, device=G.device), B.T)
    P = torch.eye(B.shape[0], dtype=B.dtype, device=B.device) - B @ GiBt
    return 0.5 * (P + P.T)


def _vibrational_basis(coords: torch.Tensor, masses: torch.Tensor, eps: float = 1e-12, linear_tol: float = 1e-6):
    """Build orthonormal vibrational basis Q_vib (3N, 3N-k). Returns (Q_vib, Q_tr, k)."""
    B = _eckart_generators(coords, masses, eps=eps)
    Q, R = torch.linalg.qr(B, mode="reduced")
    k = max(int((torch.abs(torch.diag(R)) > linear_tol).sum().item()), 1)
    U, _, _ = torch.linalg.svd(Q[:, :k], full_matrices=True)
    return U[:, k:], U[:, :k], k


# =============================================================================
# Public API
# =============================================================================

def purify_hessian(hessian: torch.Tensor, n_atoms: int) -> torch.Tensor:
    """Enforce translational invariance sum rules on a Cartesian Hessian.

    ML Hessians violate Σ_j H[i,a;j,b]=0, causing residual TR eigenvalues.
    This distributes the row-sum error uniformly so sum rules hold exactly.
    """
    H = hessian.to(torch.float64).reshape(n_atoms, 3, n_atoms, 3)
    H = H - H.sum(dim=(2, 3))[:, :, None, None] / (3 * n_atoms)
    H = H.reshape(3 * n_atoms, 3 * n_atoms)
    return 0.5 * (H + H.T)


def vib_eig(
    hessian: torch.Tensor,
    coords: torch.Tensor,
    atomsymbols: list[str],
    purify: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Vibrational eigenvalues and eigenvectors via the reduced basis.

    Every returned eigenvalue is a genuine vibrational frequency — no
    threshold-based TR filtering needed.

    Returns:
        evals_vib:    (3N-k,) vibrational eigenvalues, ascending.
        evecs_vib_3N: (3N, 3N-k) eigenvectors in full 3N space.
        Q_vib:        (3N, 3N-k) orthonormal vibrational basis.
    """
    device = hessian.device
    coords_3d = coords.reshape(-1, 3).to(torch.float64)
    n_atoms = coords_3d.shape[0]

    masses, m3, sqrt_m, sqrt_m_inv = get_mass_weights(atomsymbols, device=device)

    H = hessian.to(torch.float64).reshape(3 * n_atoms, 3 * n_atoms)
    if purify:
        H = purify_hessian(H, n_atoms)

    # Mass-weight
    diag_inv = torch.diag(1.0 / torch.sqrt(m3))
    H_mw = diag_inv @ H @ diag_inv

    # Reduce to vibrational subspace
    Q_vib, _, _ = _vibrational_basis(coords_3d, masses)
    H_red = Q_vib.T @ H_mw @ Q_vib
    H_red = 0.5 * (H_red + H_red.T)

    evals, evecs_red = torch.linalg.eigh(H_red)
    evecs_3N = Q_vib @ evecs_red
    return evals, evecs_3N, Q_vib


def gad_dynamics_projected(
    coords: torch.Tensor,
    forces: torch.Tensor,
    v: torch.Tensor,
    atomsymbols: list[str],
    gad_blend_weight: float = 1.0,
    eps: float = 1e-10,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """GAD direction with consistent Eckart projection.

    Projects gradient, guide vector, and output to prevent TR leakage.

    Args:
        gad_blend_weight: Blend weight w in F + 2w(F·v₁)v₁.
            1.0 = full GAD (default), 0.0 = pure descent.
            Use sigmoid(k·λ₂) for smooth λ₂-blended dynamics.

    Returns:
        gad_vec: (N, 3) GAD direction in Cartesian space.
        v_proj: (3N,) projected guide vector for tracking.
        info: Diagnostic dict.
    """
    device = coords.device
    coords_3d = coords.reshape(-1, 3).to(torch.float64)
    f_flat = forces.reshape(-1).to(torch.float64)
    v_flat = v.reshape(-1).to(torch.float64)
    num_atoms = coords_3d.shape[0]

    masses, _, sqrt_m, sqrt_m_inv = get_mass_weights(atomsymbols, device=device)
    P = _eckart_projector(coords_3d, masses, eps=eps)

    grad_mw = P @ (-sqrt_m_inv * f_flat)
    v_proj = P @ v_flat
    v_proj = v_proj / (v_proj.norm() + 1e-12)

    v_dot_grad = torch.dot(v_proj, grad_mw)
    w = float(gad_blend_weight) if not isinstance(gad_blend_weight, torch.Tensor) else gad_blend_weight
    dq = P @ (-grad_mw + 2.0 * w * (v_dot_grad / (torch.dot(v_proj, v_proj) + 1e-12)) * v_proj)

    gad_vec = (sqrt_m * dq).reshape(num_atoms, 3).to(forces.dtype)
    info = {
        "v_dot_grad": float(v_dot_grad.item()),
        "grad_norm_mw": float(grad_mw.norm().item()),
        "gad_blend_weight": float(w) if isinstance(w, (int, float)) else float(w.item()),
    }
    return gad_vec, v_proj.to(v.dtype), info


def preconditioned_gad_dynamics_projected(
    coords: torch.Tensor,
    forces: torch.Tensor,
    v: torch.Tensor,
    atomsymbols: list[str],
    evals_vib: torch.Tensor,
    evecs_vib_3N: torch.Tensor,
    eig_floor: float = 0.01,
    gad_blend_weight: float = 1.0,
    eps: float = 1e-10,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Preconditioned GAD: Δx = dt · |H|⁻¹ F_blend.

    F_blend = F + 2·w·(F·v₁)v₁, then preconditioned by |H|⁻¹.

    When gad_blend_weight=1.0 (default): standard preconditioned GAD.
    When gad_blend_weight=0.0: pure preconditioned descent (no v₁ ascent).
    When gad_blend_weight=sigmoid(k·λ₂): smooth λ₂-blended dynamics.

    Mode-by-mode effect:
        v₁ (lowest): step ∝ (F·v₁·(2w-1)) / |λ₁|.
            w=1 → ascend (GAD). w=0 → descend. w=0.5 → zero force on v₁.
        vᵢ (i>1):   step ∝ (F·vᵢ) / |λᵢ|.
            Always descent, always curvature-scaled. Unaffected by blend.

    Args:
        coords: (N, 3) atomic coordinates.
        forces: (N, 3) atomic forces.
        v: (3N,) guide eigenvector for GAD.
        atomsymbols: Element symbols.
        evals_vib: (3N-k,) vibrational eigenvalues from vib_eig.
        evecs_vib_3N: (3N, 3N-k) vibrational eigenvectors in 3N MW space.
        eig_floor: Clamp |λᵢ| from below to avoid blowup near zero.
        gad_blend_weight: Blend weight w in F + 2w(F·v₁)v₁.
            1.0 = full GAD, 0.0 = pure descent. Use sigmoid(k·λ₂) for smooth blend.
        eps: Numerical stability for projector.

    Returns:
        gad_vec: (N, 3) preconditioned direction in Cartesian space.
        v_proj: (3N,) projected guide vector for tracking.
        info: Diagnostic dict with preconditioning stats.
    """
    device = coords.device
    coords_3d = coords.reshape(-1, 3).to(torch.float64)
    f_flat = forces.reshape(-1).to(torch.float64)
    v_flat = v.reshape(-1).to(torch.float64)
    num_atoms = coords_3d.shape[0]

    masses, _, sqrt_m, sqrt_m_inv = get_mass_weights(atomsymbols, device=device)
    P = _eckart_projector(coords_3d, masses, eps=eps)

    # Blended GAD direction in MW vibrational space
    # F_blend = -grad + 2·w·(grad·v)·v  →  dq = P @ (-grad + 2w·(grad·v)·v)
    grad_mw = P @ (-sqrt_m_inv * f_flat)
    v_proj = P @ v_flat
    v_proj = v_proj / (v_proj.norm() + 1e-12)

    v_dot_grad = torch.dot(v_proj, grad_mw)
    w = float(gad_blend_weight) if not isinstance(gad_blend_weight, torch.Tensor) else gad_blend_weight
    dq = P @ (-grad_mw + 2.0 * w * (v_dot_grad / (torch.dot(v_proj, v_proj) + 1e-12)) * v_proj)

    # Precondition: decompose dq into eigenvector components, scale by 1/|λᵢ|
    evecs = evecs_vib_3N.to(torch.float64)
    evals = evals_vib.to(torch.float64)

    coeffs = evecs.T @ dq  # (3N-k,)
    inv_abs_evals = 1.0 / torch.clamp(evals.abs(), min=eig_floor)
    scaled_coeffs = coeffs * inv_abs_evals

    dq_precond = evecs @ scaled_coeffs

    # Back to Cartesian
    gad_vec = (sqrt_m * dq_precond).reshape(num_atoms, 3).to(forces.dtype)

    # Diagnostics
    scale_range = float(inv_abs_evals.max().item()) / max(float(inv_abs_evals.min().item()), 1e-12)
    info = {
        "v_dot_grad": float(v_dot_grad.item()),
        "grad_norm_mw": float(grad_mw.norm().item()),
        "precond_scale_min": float(inv_abs_evals.min().item()),
        "precond_scale_max": float(inv_abs_evals.max().item()),
        "precond_scale_range": scale_range,
        "n_clamped": int((evals.abs() < eig_floor).sum().item()),
        "gad_blend_weight": float(w) if isinstance(w, (int, float)) else float(w.item()),
    }
    return gad_vec, v_proj.to(v.dtype), info


def project_vector_to_vibrational(
    vec: torch.Tensor,
    cart_coords: torch.Tensor,
    atomsymbols: list[str],
    eps: float = 1e-10,
) -> torch.Tensor:
    """Project a Cartesian vector to remove TR components. Mass-weights internally."""
    device = vec.device
    vec_flat = vec.reshape(-1).to(torch.float64)
    coords_3d = cart_coords.reshape(-1, 3)
    masses, _, sqrt_m, sqrt_m_inv = get_mass_weights(atomsymbols, device=device)
    P = _eckart_projector(coords_3d, masses, eps=eps)
    return (sqrt_m * (P @ (sqrt_m_inv * vec_flat))).to(vec.dtype)
