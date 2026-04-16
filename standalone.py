"""Standalone Eckart-projected GAD transition state search.

GAD flips the force along the lowest Hessian eigenvector:
    F_GAD = F + 2(F · v₁)v₁

Convergence: n_neg == 1 AND fmax < threshold.

Dependencies: torch, torch_geometric, hip, transition1x
"""

import os
import time
import torch
torch.manual_seed(42)

# ---------------------------------------------------------------------------
# Configuration. Paths and hyperparameters
# ---------------------------------------------------------------------------
checkpoint_path = "path_to/hip_v2.ckpt"
h5_path         = "path_to/transition1x.h5"
split           = "train"
device          = "cuda" if torch.cuda.is_available() else "cpu"

n_samples       = 300      # molecules to optimize
noise           = 0.01     # Angstrom of Gaussian noise on starting geometry (0.01 = 10pm)
dt              = 0.003    # Euler timestep — 0.003 is optimal, smaller has diminishing returns
n_steps         = 2000     # step budget — enough for dt=0.003
force_threshold = 0.01     # eV/A, convergence on max |force component| (fmax)
max_atom_disp   = 0.35     # A, per-atom displacement cap per step (safety, rarely triggers)

# ---------------------------------------------------------------------------
# Atomic masses (Z → amu). Covers all of Transition1x.
# ---------------------------------------------------------------------------
mass = {1: 1.008, 6: 12.011, 7: 14.007, 8: 15.999, 9: 18.998,
        15: 30.974, 16: 32.065, 17: 35.453, 35: 79.904, 53: 126.904}

def masses_from_z(z, dev):
    return torch.tensor([mass.get(int(a), 12.0) for a in z.cpu().tolist()],
                        dtype=torch.float64, device=dev)

# ---------------------------------------------------------------------------
# Eckart projection — removes 6 translation/rotation modes from the Hessian
# ---------------------------------------------------------------------------
def eckart_projector(coords, masses):
    """P = I - B(BᵀB)⁻¹Bᵀ in mass-weighted space. Returns (3N, 3N)."""
    N = coords.shape[0]
    sq = torch.sqrt(masses)
    sq3 = sq.repeat_interleave(3)
    com = (coords * masses[:, None]).sum(0) / masses.sum()
    r = coords - com

    # 6 generators: 3 translations + 3 infinitesimal rotations
    cols = []
    for e in torch.eye(3, dtype=torch.float64, device=coords.device):
        c = sq3 * e.repeat(N)
        cols.append(c / c.norm())
    rx, ry, rz = r[:, 0], r[:, 1], r[:, 2]
    for R in (torch.stack([torch.zeros_like(rx), -rz,  ry], 1),
              torch.stack([ rz, torch.zeros_like(ry), -rx], 1),
              torch.stack([-ry,  rx, torch.zeros_like(rz)], 1)):
        c = (R * sq[:, None]).reshape(-1)
        cols.append(c / c.norm())
    B = torch.stack(cols, 1)

    G = B.T @ B + 1e-10 * torch.eye(6, dtype=B.dtype, device=B.device)
    P = torch.eye(3*N, dtype=B.dtype, device=B.device) - B @ torch.linalg.solve(G, B.T)
    return 0.5 * (P + P.T)

# ---------------------------------------------------------------------------
# Vibrational eigendecomposition (reduced-basis, 3N-6 dims)
# ---------------------------------------------------------------------------
def vib_eig(hessian, coords, masses):
    """Vibrational eigenvalues and eigenvectors via reduced-basis projection.

    H_mw = M^{-1/2} H M^{-1/2}           mass-weighted Hessian
    B = [t1 t2 t3 r1 r2 r3]              6 Eckart generators (translation + rotation)
    Q_vib = null(B)                       orthogonal complement, shape (3N, 3N-6)
    H_red = Q_vib^T H_mw Q_vib           reduced Hessian in vibrational subspace
    evals, evecs_red = eigh(H_red)        diagonalize
    evecs_3N = Q_vib @ evecs_red          lift back to full 3N space

    Returns (evals (M,), evecs (3N, M)) where M = 3N-6 (or 3N-5 for linear).
    """
    N = coords.shape[0]
    m3 = masses.repeat_interleave(3)

    # Mass-weight the Hessian: H_mw = M^{-1/2} H M^{-1/2}
    inv_sq = torch.diag(1.0 / torch.sqrt(m3))
    H_mw = inv_sq @ hessian.to(torch.float64).reshape(3*N, 3*N) @ inv_sq

    # Build Eckart generators (same as eckart_projector) to find their null space
    sq = torch.sqrt(masses)
    sq3 = sq.repeat_interleave(3)
    com = (coords.to(torch.float64) * masses[:, None]).sum(0) / masses.sum()
    r = coords.to(torch.float64) - com

    B_cols = []
    for e in torch.eye(3, dtype=torch.float64, device=hessian.device):
        c = sq3 * e.repeat(N)
        B_cols.append(c / c.norm())
    rx, ry, rz = r[:, 0], r[:, 1], r[:, 2]
    for R in (torch.stack([torch.zeros_like(rx), -rz, ry], 1),
              torch.stack([rz, torch.zeros_like(ry), -rx], 1),
              torch.stack([-ry, rx, torch.zeros_like(rz)], 1)):
        c = (R * sq[:, None]).reshape(-1)
        B_cols.append(c / c.norm())
    B = torch.stack(B_cols, 1)

    # Q_vib = orthogonal complement of Eckart generators (3N, 3N-k)
    Q, R_ = torch.linalg.qr(B, mode="reduced")
    k = max(int((torch.diag(R_).abs() > 1e-6).sum().item()), 1)
    U, _, _ = torch.linalg.svd(Q[:, :k], full_matrices=True)
    Q_vib = U[:, k:]

    # Diagonalize the reduced (3N-k) x (3N-k) vibrational Hessian
    H_red = Q_vib.T @ H_mw @ Q_vib
    H_red = 0.5 * (H_red + H_red.T)
    evals, evecs = torch.linalg.eigh(H_red)
    return evals, Q_vib @ evecs

# ---------------------------------------------------------------------------
# Projected GAD direction
# ---------------------------------------------------------------------------
def gad_direction(coords, forces, v, masses):
    """Eckart-projected GAD: dq = P(-g + 2(g·v)v), dx = √m · dq.

    Returns (N,3) GAD vector and (3N,) projected guide vector.
    """
    c = coords.reshape(-1, 3).to(torch.float64)
    f = forces.reshape(-1).to(torch.float64)
    v = v.reshape(-1).to(torch.float64)
    N = c.shape[0]

    m3 = masses.repeat_interleave(3)
    sq = torch.sqrt(m3)
    P = eckart_projector(c, masses)

    g = P @ (-f / sq)                                  # projected gradient in MW space
    vp = P @ v; vp = vp / (vp.norm() + 1e-12)         # projected, normalized guide vector
    dq = P @ (-g + 2 * torch.dot(vp, g) * vp)         # GAD formula: flip along v
    return (sq * dq).reshape(N, 3).to(forces.dtype), vp.to(forces.dtype)

# ---------------------------------------------------------------------------
# GAD search loop (Euler integration, no path history)
# ---------------------------------------------------------------------------
def gad_search(predict_fn, coords, atomic_nums):
    """Find an index-1 saddle point via Eckart-projected GAD.

    Args:
        predict_fn: (coords, atomic_nums, do_hessian=, require_grad=) -> dict
                    with keys "energy", "forces", "hessian".
        coords: (N, 3) starting geometry in Angstrom.
        atomic_nums: (N,) atomic numbers.

    Returns:
        dict with: converged, coords, energy, n_neg, fmax, step.
    """
    x = coords.detach().clone().to(torch.float32).reshape(-1, 3)
    m = masses_from_z(atomic_nums, x.device)

    for step in range(n_steps):
        out = predict_fn(x, atomic_nums, do_hessian=True, require_grad=False)
        f = out["forces"]
        if f.dim() == 3: f = f[0]
        f = f.reshape(-1, 3)
        e = float(out["energy"].detach().reshape(-1)[0]) if isinstance(out["energy"], torch.Tensor) else float(out["energy"])
        fmax = float(f.reshape(-1).abs().max())

        evals, evecs = vib_eig(out["hessian"], x, m)
        n_neg = int((evals < 0).sum())

        # Convergence: n_neg == 1 AND fmax < threshold
        if n_neg == 1 and fmax < force_threshold:
            return dict(converged=True, coords=x.cpu(), energy=e,
                        n_neg=n_neg, fmax=fmax, step=step)

        # Lowest eigenvector from current Hessian — no path history
        v1 = evecs[:, 0].to(device=f.device, dtype=f.dtype)
        v1 = v1 / (v1.norm() + 1e-12)
        gad_vec, _ = gad_direction(x, f, v1, m)
        dx = dt * gad_vec

        # Displacement cap
        d_max = float(dx.reshape(-1, 3).norm(dim=1).max())
        if d_max > max_atom_disp:
            dx = dx * (max_atom_disp / d_max)
        x = (x + dx).detach()

    return dict(converged=False, coords=x.cpu(), energy=e,
                n_neg=n_neg, fmax=fmax, step=n_steps)

# ---------------------------------------------------------------------------
# HIP calculator loader
# ---------------------------------------------------------------------------
def load_hip():
    from hip import path_config, training_module, inference_utils
    orig = path_config.fix_dataset_path
    def lenient(p):
        return orig(p) if os.path.exists(p) else p
    path_config.fix_dataset_path = lenient
    training_module.fix_dataset_path = lenient
    inference_utils.fix_dataset_path = lenient

    from hip.equiformer_torch_calculator import EquiformerTorchCalculator
    calc = EquiformerTorchCalculator(checkpoint_path=checkpoint_path,
                                     hessian_method="predict", device=device)

    from torch_geometric.data import Batch, Data as TGData
    def predict(coords, z, *, do_hessian=True, require_grad=False):
        batch = Batch.from_data_list([TGData(
            pos=coords.reshape(-1,3).to(torch.float32),
            z=z.to(torch.int64), charges=z.to(torch.int64),
            natoms=torch.tensor([int(z.numel())], dtype=torch.int64),
            cell=None, pbc=torch.tensor(False),
        )]).to(coords.device)
        with torch.no_grad():
            return calc.predict(batch, do_hessian=do_hessian)
    return predict

# ---------------------------------------------------------------------------
# Transition1x dataset loader
# ---------------------------------------------------------------------------
def load_dataset():
    from transition1x import Dataloader
    loader = Dataloader(h5_path, datasplit=split, only_final=True)
    samples = []
    for mol in loader:
        if n_samples and len(samples) >= n_samples:
            break
        ts, rx = mol["transition_state"], mol["reactant"]
        if len(ts["atomic_numbers"]) != len(rx["atomic_numbers"]):
            continue  # mismatched atom counts between reactant and TS
        samples.append(dict(
            z=torch.tensor(ts["atomic_numbers"], dtype=torch.long),
            pos=torch.tensor(ts["positions"], dtype=torch.float32),
            formula=ts.get("formula", "?"),
        ))
    return samples

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    noise_pm = int(round(noise * 1000))
    print(f"GAD | dt={dt} | steps={n_steps} | noise={noise_pm}pm | "
          f"n={n_samples} | fmax<{force_threshold} | {device}\n")

    predict_fn = load_hip()
    print("HIP loaded")

    samples = load_dataset()
    print(f"{len(samples)} samples from Transition1x ({split})\n")

    n_converged = 0
    t_start = time.time()

    for i, sample in enumerate(samples):
        atomic_nums = sample["z"].to(device)
        coords_ts = sample["pos"].to(device)
        coords_noised = coords_ts + noise * torch.randn_like(coords_ts)

        t0 = time.time()
        result = gad_search(predict_fn, coords_noised, atomic_nums)
        wall = time.time() - t0

        if result["converged"]:
            n_converged += 1
        status = "CONV" if result["converged"] else "FAIL"
        print(f"  [{i:3d}] {sample['formula']:>12s} | {status} | n_neg={result['n_neg']} "
              f"| fmax={result['fmax']:.4f} | step={result['step']:4d} | {wall:.1f}s")

    rate = 100 * n_converged / len(samples)
    print(f"\n{n_converged}/{len(samples)} converged ({rate:.1f}%) "
          f"in {time.time() - t_start:.0f}s")
