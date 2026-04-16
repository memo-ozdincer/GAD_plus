#!/usr/bin/env python
"""Standalone GAD transition state search.

Self-contained: loads HIP, loads Transition1x, runs Eckart-projected GAD
on noised TS geometries, prints results.

Dependencies:
    pip install torch torch_geometric hip transition1x

The core algorithm (everything between the imports and `load_hip`) is pure
PyTorch — plug in any energy/force/Hessian calculator via predict_fn.

Usage:
    # Best config, 300 samples, 10pm noise (reproduces 94.7% with fmax<0.01)
    python standalone.py --noise 0.01 --n-samples 300 --dt 0.003 --n-steps 2000

    # Quick smoke test (10 samples)
    python standalone.py

    # High noise (reproduces 55.2% at 200pm)
    python standalone.py --noise 0.20 --n-samples 300 --dt 0.003 --n-steps 2000

    # Start from linear midpoint between reactant and product (no TS knowledge)
    python standalone.py --start midpoint --n-samples 300

    # Start from geodesic midpoint (better path, needs geodesic_interpolate package)
    python standalone.py --start geodesic --n-samples 300

    # The dt=0.005 config from Round 1 (reproduces 94.3% at 10pm)
    python standalone.py --noise 0.01 --n-samples 300 --dt 0.005 --n-steps 1000

    # Mean per-atom force norm instead of fmax (our original, slightly looser criterion)
    python standalone.py --noise 0.01 --n-samples 300 --force-criterion mean
"""
from __future__ import annotations

import argparse
import os
import time
from typing import Any, Dict, Optional

import torch

# =============================================================================
# Configuration — edit these for different clusters
# =============================================================================

DEFAULT_CHECKPOINT = "path_to/hip_v2.ckpt"
DEFAULT_H5_PATH = "path_to/transition1x.h5"
DEFAULT_SPLIT = "train"

# Best-performing GAD configuration
DEFAULT_DT = 0.003
DEFAULT_N_STEPS = 2000
DEFAULT_FORCE_THRESHOLD = 0.01  # eV/A
DEFAULT_MAX_ATOM_DISP = 0.35    # A


# =============================================================================
# Atomic masses (Z → amu)
# =============================================================================

ATOMIC_MASSES: dict[int, float] = {
    1: 1.008, 2: 4.003, 3: 6.941, 4: 9.012, 5: 10.81, 6: 12.011,
    7: 14.007, 8: 15.999, 9: 18.998, 10: 20.180, 15: 30.974, 16: 32.065,
    17: 35.453, 35: 79.904, 53: 126.904,
}


def _get_masses(atomic_nums: torch.Tensor, device: torch.device) -> torch.Tensor:
    return torch.tensor(
        [ATOMIC_MASSES.get(int(z), 12.0) for z in atomic_nums.detach().cpu().tolist()],
        dtype=torch.float64, device=device,
    )


# =============================================================================
# Eckart projection — removes 6 translation/rotation modes
# =============================================================================

def _eckart_projector(coords: torch.Tensor, masses: torch.Tensor) -> torch.Tensor:
    """P = I - B(B^TB)^{-1}B^T  in mass-weighted space.  Returns (3N, 3N)."""
    N = coords.shape[0]
    sqrt_m = torch.sqrt(masses)
    sqrt_m3 = sqrt_m.repeat_interleave(3)
    com = (coords * masses[:, None]).sum(0) / masses.sum()
    r = coords - com[None, :]

    cols = []
    eye3 = torch.eye(3, dtype=torch.float64, device=coords.device)
    for e in eye3:
        c = sqrt_m3 * e.repeat(N)
        cols.append(c / (c.norm() + 1e-12))
    rx, ry, rz = r[:, 0], r[:, 1], r[:, 2]
    for Raxis in (
        torch.stack([torch.zeros_like(rx), -rz, ry], 1),
        torch.stack([rz, torch.zeros_like(ry), -rx], 1),
        torch.stack([-ry, rx, torch.zeros_like(rz)], 1),
    ):
        c = (Raxis * sqrt_m[:, None]).reshape(-1)
        cols.append(c / (c.norm() + 1e-12))
    B = torch.stack(cols, dim=1)  # (3N, 6)

    G = B.T @ B + 1e-10 * torch.eye(6, dtype=B.dtype, device=B.device)
    P = torch.eye(3 * N, dtype=torch.float64, device=coords.device) - B @ torch.linalg.solve(G, B.T)
    return 0.5 * (P + P.T)


# =============================================================================
# Vibrational eigendecomposition
# =============================================================================

def vib_eig(
    hessian: torch.Tensor, coords: torch.Tensor, masses: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Eigenvalues/vectors of the reduced vibrational Hessian.

    Returns:
        evals: (M,) ascending, M = 3N-6 (or 3N-5 for linear).
        evecs_3N: (3N, M) in mass-weighted space.
    """
    coords_3d = coords.reshape(-1, 3).to(torch.float64)
    N = coords_3d.shape[0]
    device = hessian.device

    m3 = masses.repeat_interleave(3)
    H = hessian.to(torch.float64).reshape(3 * N, 3 * N)
    H_mw = torch.diag(1.0 / torch.sqrt(m3)) @ H @ torch.diag(1.0 / torch.sqrt(m3))

    # Vibrational basis = orthogonal complement of Eckart generators
    sqrt_m = torch.sqrt(masses)
    sqrt_m3 = sqrt_m.repeat_interleave(3)
    com = (coords_3d * masses[:, None]).sum(0) / masses.sum()
    r = coords_3d - com[None, :]

    B_cols = []
    eye3 = torch.eye(3, dtype=torch.float64, device=device)
    for e in eye3:
        c = sqrt_m3 * e.repeat(N)
        B_cols.append(c / (c.norm() + 1e-12))
    rx, ry, rz = r[:, 0], r[:, 1], r[:, 2]
    for Raxis in (
        torch.stack([torch.zeros_like(rx), -rz, ry], 1),
        torch.stack([rz, torch.zeros_like(ry), -rx], 1),
        torch.stack([-ry, rx, torch.zeros_like(rz)], 1),
    ):
        c = (Raxis * sqrt_m[:, None]).reshape(-1)
        B_cols.append(c / (c.norm() + 1e-12))
    B = torch.stack(B_cols, dim=1)

    Q, R = torch.linalg.qr(B, mode="reduced")
    k = max(int((torch.abs(torch.diag(R)) > 1e-6).sum().item()), 1)
    U, _, _ = torch.linalg.svd(Q[:, :k], full_matrices=True)
    Q_vib = U[:, k:]

    H_red = Q_vib.T @ H_mw @ Q_vib
    H_red = 0.5 * (H_red + H_red.T)
    evals, evecs_red = torch.linalg.eigh(H_red)
    return evals, Q_vib @ evecs_red


# =============================================================================
# Projected GAD direction
# =============================================================================

def _gad_direction(
    coords: torch.Tensor, forces: torch.Tensor,
    v: torch.Tensor, masses: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Eckart-projected GAD:  dq = P(-g + 2(g·v)v),  dx = √m · dq.

    Returns (N,3) Cartesian GAD vector and (3N,) projected guide vector.
    """
    coords_3d = coords.reshape(-1, 3).to(torch.float64)
    f_flat = forces.reshape(-1).to(torch.float64)
    v_flat = v.reshape(-1).to(torch.float64)
    N = coords_3d.shape[0]

    m3 = masses.repeat_interleave(3)
    sqrt_m = torch.sqrt(m3)
    P = _eckart_projector(coords_3d, masses)

    grad_mw = P @ (-f_flat / sqrt_m)
    v_proj = P @ v_flat
    v_proj = v_proj / (v_proj.norm() + 1e-12)

    v_dot_grad = torch.dot(v_proj, grad_mw)
    dq = P @ (-grad_mw + 2.0 * (v_dot_grad / (torch.dot(v_proj, v_proj) + 1e-12)) * v_proj)

    gad_vec = (sqrt_m * dq).reshape(N, 3).to(forces.dtype)
    return gad_vec, v_proj.to(v.dtype)


# =============================================================================
# GAD search loop
# =============================================================================

def gad_search(
    predict_fn,
    coords: torch.Tensor,
    atomic_nums: torch.Tensor,
    *,
    dt: float = DEFAULT_DT,
    n_steps: int = DEFAULT_N_STEPS,
    force_threshold: float = DEFAULT_FORCE_THRESHOLD,
    force_criterion: str = "mean",
    max_atom_disp: float = DEFAULT_MAX_ATOM_DISP,
) -> dict[str, Any]:
    """Run Eckart-projected GAD to find an index-1 saddle point.

    Each step depends only on the current geometry — no path history.
    Fully differentiable, diffusion-compatible.

    Args:
        predict_fn: (coords, atomic_nums, do_hessian=, require_grad=) → dict
                    with keys "energy", "forces", "hessian".
        coords: (N, 3) starting geometry in Angstrom.
        atomic_nums: (N,) atomic numbers.
        dt: Euler timestep. 0.003 is optimal.
        n_steps: Step budget.
        force_threshold: Convergence threshold in eV/A.
        force_criterion: "mean" (per-atom force norm) or "fmax" (max |component|, max abs component).
        max_atom_disp: Per-atom displacement cap (A).

    Returns:
        dict with: converged, coords, energy, n_neg, force_norm, force_max, eig0, step
    """
    x = coords.detach().clone().to(torch.float32).reshape(-1, 3)
    masses = _get_masses(atomic_nums, x.device).to(torch.float64)

    energy, force_norm, force_max, n_neg, eig0 = 0.0, float("inf"), float("inf"), 0, 0.0

    for step in range(n_steps):
        out = predict_fn(x, atomic_nums, do_hessian=True, require_grad=False)
        forces = out["forces"]
        if forces.dim() == 3 and forces.shape[0] == 1:
            forces = forces[0]
        forces = forces.reshape(-1, 3)

        energy = float(
            out["energy"].detach().reshape(-1)[0].item()
            if isinstance(out["energy"], torch.Tensor) else out["energy"]
        )
        force_norm = float(forces.norm(dim=1).mean().item())
        force_max = float(forces.reshape(-1).abs().max().item())
        force_val = force_max if force_criterion == "fmax" else force_norm

        evals, evecs_3N = vib_eig(out["hessian"], x, masses)
        n_neg = int((evals < 0).sum().item())
        eig0 = float(evals[0].item()) if evals.numel() > 0 else 0.0

        if n_neg == 1 and force_val < force_threshold:
            return {"converged": True, "coords": x.detach().cpu(),
                    "energy": energy, "n_neg": n_neg, "force_norm": force_norm,
                    "force_max": force_max, "eig0": eig0, "step": step}

        # Lowest eigenvector from current Hessian — no path history
        v1 = evecs_3N[:, 0].to(device=forces.device, dtype=forces.dtype)
        v1 = v1 / (v1.norm() + 1e-12)
        gad_vec, _ = _gad_direction(x, forces, v1, masses)

        step_disp = dt * gad_vec
        d3 = step_disp.reshape(-1, 3)
        max_d = float(d3.norm(dim=1).max().item())
        if max_d > max_atom_disp and max_d > 0:
            step_disp = step_disp * (max_atom_disp / max_d)

        x = (x + step_disp).detach()

    return {"converged": False, "coords": x.detach().cpu(),
            "energy": energy, "n_neg": n_neg, "force_norm": force_norm,
            "force_max": force_max, "eig0": eig0, "step": n_steps}


# =============================================================================
# HIP calculator loader
# =============================================================================

def load_hip(checkpoint: str = DEFAULT_CHECKPOINT, device: str = "cuda"):
    """Load HIP and return a predict_fn compatible with gad_search."""
    from hip import path_config, training_module, inference_utils

    _orig = path_config.fix_dataset_path
    def _lenient(path):
        return _orig(path) if os.path.exists(path) else path
    path_config.fix_dataset_path = _lenient
    training_module.fix_dataset_path = _lenient
    inference_utils.fix_dataset_path = _lenient

    from hip.equiformer_torch_calculator import EquiformerTorchCalculator
    calc = EquiformerTorchCalculator(checkpoint_path=checkpoint,
                                     hessian_method="predict", device=device)

    from torch_geometric.data import Batch, Data as TGData

    def predict_fn(coords, atomic_nums, *, do_hessian=True, require_grad=False):
        if coords.dim() == 1:
            coords = coords.reshape(-1, 3)
        batch = Batch.from_data_list([TGData(
            pos=coords.to(torch.float32),
            z=atomic_nums.to(torch.int64),
            charges=atomic_nums.to(torch.int64),
            natoms=torch.tensor([int(atomic_nums.numel())], dtype=torch.int64),
            cell=None, pbc=torch.tensor(False, dtype=torch.bool),
        )]).to(coords.device)
        with torch.no_grad():
            return calc.predict(batch, do_hessian=do_hessian)

    return predict_fn


# =============================================================================
# Transition1x dataset loader
# =============================================================================

def load_transition1x(h5_path: str = DEFAULT_H5_PATH, split: str = DEFAULT_SPLIT,
                      max_samples: Optional[int] = None):
    """Load samples from Transition1x.

    Returns list of dicts with keys: z, pos_ts, pos_reactant, pos_product, formula.
    """
    from transition1x import Dataloader as T1xDataloader
    loader = T1xDataloader(h5_path, datasplit=split, only_final=True)
    samples = []
    for mol in loader:
        if max_samples is not None and len(samples) >= max_samples:
            break
        ts = mol["transition_state"]
        reactant = mol["reactant"]
        if len(ts["atomic_numbers"]) != len(reactant["atomic_numbers"]):
            continue
        product = mol.get("product")
        has_product = (product is not None
                       and len(product.get("atomic_numbers", [])) == len(ts["atomic_numbers"]))
        samples.append({
            "z": torch.tensor(ts["atomic_numbers"], dtype=torch.long),
            "pos_ts": torch.tensor(ts["positions"], dtype=torch.float32),
            "pos_reactant": torch.tensor(reactant["positions"], dtype=torch.float32),
            "pos_product": torch.tensor(product["positions"], dtype=torch.float32)
                         if has_product else None,
            "formula": ts.get("formula", "?"),
        })
    return samples


# =============================================================================
# Starting geometry construction
# =============================================================================

def starting_geometry(sample: dict, method: str, noise: float, device: str) -> torch.Tensor:
    """Build starting geometry from a Transition1x sample.

    Args:
        sample: dict with keys z, pos_ts, pos_reactant, pos_product.
        method: "noised_ts", "midpoint", or "geodesic".
        noise: Gaussian noise in Angstrom (only used for "noised_ts").
        device: torch device.

    Returns:
        (N, 3) starting coordinates.
    """
    if method == "noised_ts":
        ts = sample["pos_ts"].to(device)
        return ts + noise * torch.randn_like(ts)

    reactant = sample["pos_reactant"].to(device)
    product = sample["pos_product"]
    assert product is not None, "midpoint/geodesic requires product geometry"
    product = product.to(device)

    if method == "midpoint":
        return 0.5 * (reactant + product)

    if method == "geodesic":
        import numpy as np
        from geodesic_interpolate import geodesic
        r_np = reactant.detach().cpu().numpy().reshape(1, -1)
        p_np = product.detach().cpu().numpy().reshape(1, -1)
        path = geodesic.run_geodesic_interpolation(
            np.concatenate([r_np, p_np], axis=0), n_images=5,
        )
        mid = path[len(path) // 2]
        return torch.tensor(mid, dtype=torch.float32, device=device).reshape(-1, 3)

    raise ValueError(f"Unknown starting method: {method}")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standalone GAD transition state search")
    parser.add_argument("--start", choices=["noised_ts", "midpoint", "geodesic"], default="noised_ts",
                        help="Starting geometry: noised_ts (default), midpoint (R+P)/2, geodesic midpoint")
    parser.add_argument("--noise", type=float, default=0.01, help="Noise in Angstrom (0.01 = 10pm, only for noised_ts)")
    parser.add_argument("--n-samples", type=int, default=300, help="Number of samples")
    parser.add_argument("--dt", type=float, default=DEFAULT_DT, help=f"Timestep (default: {DEFAULT_DT})")
    parser.add_argument("--n-steps", type=int, default=DEFAULT_N_STEPS, help=f"Max steps (default: {DEFAULT_N_STEPS})")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for noise (default: 42)")
    parser.add_argument("--force-criterion", choices=["mean", "fmax"], default="fmax",
                        help="Force metric: 'fmax' (max component, max abs component) or 'mean' (per-atom norm)")
    parser.add_argument("--force-threshold", type=float, default=DEFAULT_FORCE_THRESHOLD,
                        help=f"Force threshold in eV/A (default: {DEFAULT_FORCE_THRESHOLD})")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--h5-path", default=DEFAULT_H5_PATH)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    noise_pm = int(round(args.noise * 1000))
    start_label = f"{args.start}" + (f" {noise_pm}pm" if args.start == "noised_ts" else "")
    print(f"GAD standalone | dt={args.dt} | steps={args.n_steps} | start={start_label} | "
          f"samples={args.n_samples} | seed={args.seed} | {args.force_criterion}<{args.force_threshold} | "
          f"device={args.device}")

    predict_fn = load_hip(args.checkpoint, args.device)
    print("HIP loaded")

    samples = load_transition1x(args.h5_path, args.split, args.n_samples)
    print(f"Loaded {len(samples)} samples from Transition1x ({args.split})")

    n_conv = 0
    t_total = time.time()

    for i, sample in enumerate(samples):
        z = sample["z"].to(args.device)
        coords_start = starting_geometry(sample, args.start, args.noise, args.device)

        t0 = time.time()
        result = gad_search(predict_fn, coords_start, z,
                            dt=args.dt, n_steps=args.n_steps,
                            force_threshold=args.force_threshold,
                            force_criterion=args.force_criterion)
        wall = time.time() - t0

        status = "CONV" if result["converged"] else "FAIL"
        if result["converged"]:
            n_conv += 1
        print(f"  [{i:3d}] {sample['formula']:>12s} | {status} | n_neg={result['n_neg']} "
              f"| fmax={result['force_max']:.4f} | step={result['step']:4d} | {wall:.1f}s")

    rate = 100 * n_conv / len(samples)
    print(f"\n{n_conv}/{len(samples)} converged ({rate:.1f}%) in {time.time()-t_total:.0f}s")
