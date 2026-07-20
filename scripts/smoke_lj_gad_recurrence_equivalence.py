#!/usr/bin/env python
"""Audit projected-LJ GAD recurrence conventions for identical hydrogen masses.

The current canonical projected GAD applies the GAD flip in Cartesian
coordinates after Eckart projection.  The older Andreas-branch convention and
the batched LJ runner apply the corresponding mass-weighted update.  For an
identical mass ``m`` cluster, their directions differ only by ``1/m``.  Thus a
batched timestep of ``m * dt`` must reproduce the canonical Cartesian path.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from gadplus.calculator.lennard_jones import (  # noqa: E402
    lj_atomic_nums,
    make_lj_predict_fn,
    pentagonal_bipyramid_geometry,
)
from gadplus.projection import (  # noqa: E402
    atomic_nums_to_symbols,
    gad_dynamics_projected,
    vib_eig,
)


def batched_internal_basis(coords: torch.Tensor, mass: float) -> torch.Tensor:
    """Same uniform-mass Eckart basis construction as the batched runner."""
    bsz, n_atoms, _ = coords.shape
    sqrt_m = torch.as_tensor(mass, dtype=coords.dtype, device=coords.device).sqrt()
    centered = coords - coords.mean(dim=1, keepdim=True)
    cols = []
    for axis in range(3):
        col = torch.zeros_like(coords)
        col[:, :, axis] = sqrt_m
        cols.append(col.reshape(bsz, -1))
    eye = torch.eye(3, dtype=coords.dtype, device=coords.device)
    for axis in range(3):
        omega = eye[axis].reshape(1, 1, 3).expand_as(centered)
        cols.append((torch.cross(omega, centered, dim=2) * sqrt_m).reshape(bsz, -1))
    q, _ = torch.linalg.qr(torch.stack(cols, dim=2), mode="complete")
    return q[:, :, 6:]


def old_or_batched_direction(
    coords: torch.Tensor,
    forces: torch.Tensor,
    hessian: torch.Tensor,
    mass: float,
) -> torch.Tensor:
    """One old/batched projected GAD direction in Cartesian coordinates."""
    n_atoms = coords.shape[0]
    basis = batched_internal_basis(coords[None], mass)[0]
    h_mw = hessian / mass
    evals, evecs = torch.linalg.eigh(basis.T @ h_mw @ basis)
    del evals
    v = evecs[:, 0]
    f_internal = basis.T @ (forces.reshape(-1) / mass**0.5)
    d_internal = f_internal - 2.0 * torch.dot(f_internal, v) * v
    return (basis @ d_internal / mass**0.5).reshape(n_atoms, 3)


def main() -> None:
    torch.set_default_dtype(torch.float64)
    mass = 1.008
    dt = 0.003
    n_steps = 5
    generator = torch.Generator().manual_seed(20260717)
    start = pentagonal_bipyramid_geometry() + 0.08 * torch.randn(
        (7, 3), generator=generator
    )
    start = start - start.mean(dim=0, keepdim=True)
    z = lj_atomic_nums(7)
    symbols = atomic_nums_to_symbols(z)
    predictor = make_lj_predict_fn()

    canonical = start.clone()
    old_scaled_dt = start.clone()
    max_direction_error = 0.0
    for step in range(n_steps):
        out = predictor(canonical, z, do_hessian=True)
        evals, evecs, _ = vib_eig(out["hessian"], canonical, symbols)
        del evals
        canonical_direction, _, _ = gad_dynamics_projected(
            canonical, out["forces"], evecs[:, 0], symbols
        )
        old_direction = old_or_batched_direction(
            canonical, out["forces"], out["hessian"], mass
        )
        max_direction_error = max(
            max_direction_error,
            float((canonical_direction / mass - old_direction).abs().max()),
        )
        canonical = canonical + dt * canonical_direction

        old_out = predictor(old_scaled_dt, z, do_hessian=True)
        old_direction_at_old_coords = old_or_batched_direction(
            old_scaled_dt, old_out["forces"], old_out["hessian"], mass
        )
        old_scaled_dt = old_scaled_dt + (mass * dt) * old_direction_at_old_coords

    path_error = float((canonical - old_scaled_dt).abs().max())
    print(f"steps={n_steps} mass={mass:g}")
    print(f"max_direction_error_after_1_over_m_scaling={max_direction_error:.3e}")
    print(f"max_coordinate_error_after_dt_rescaling={path_error:.3e}")
    if max_direction_error > 2.0e-10 or path_error > 2.0e-9:
        raise SystemExit("Projected GAD recurrence equivalence smoke failed.")
    print("PASS: conventions are analytically equivalent for uniform hydrogen masses.")


if __name__ == "__main__":
    main()
