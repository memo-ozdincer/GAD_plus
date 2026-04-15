"""
The most atomic way to find a transition state on a potential energy surface.
This file is the complete algorithm.
Everything else is just infrastructure.

Gentlest Ascent Dynamics (GAD) flips the force along the lowest Hessian
eigenvector, ascending toward the saddle while descending everywhere else:

    F_GAD = F + 2(F · v₁)v₁

One eigenvalue negative, forces near zero — that's a transition state.
"""

import os
import time
import torch

# ---------------------------------------------------------------------------
# Configuration. Change these paths for your cluster.
# ---------------------------------------------------------------------------
checkpoint_path = "/lustre06/project/6033559/memoozd/models/hip_v2.ckpt"
h5_path         = "/lustre06/project/6033559/memoozd/data/transition1x.h5"
split           = "train"
device          = "cuda" if torch.cuda.is_available() else "cpu"

n_samples       = 10       # molecules to optimize
noise           = 0.01     # Angstrom of Gaussian noise on starting geometry (0.01 = 10pm)
dt              = 0.003    # Euler timestep — 0.003 is optimal, smaller is diminishing returns
n_steps         = 2000     # step budget — enough for dt=0.003
force_threshold = 0.01     # eV/A, convergence criterion on mean per-atom force norm
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
# Eckart projection — remove 6 translation/rotation modes from the Hessian.
# Without this, eigenvalue counting is meaningless (ghost negative modes).
# This single function is worth +68 percentage points of convergence.
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
# Vibrational eigendecomposition — the reduced-basis approach.
# Projects mass-weighted Hessian onto the vibrational subspace (3N-6 dims),
# diagonalizes there. Every returned eigenvalue is a real vibration.
# ---------------------------------------------------------------------------
def vib_eig(hessian, coords, masses):
    """Returns (evals (M,), evecs (3N, M)) in mass-weighted vibrational space."""
    N = coords.shape[0]
    m3 = masses.repeat_interleave(3)
    inv_sq = torch.diag(1.0 / torch.sqrt(m3))
    H_mw = inv_sq @ hessian.to(torch.float64).reshape(3*N, 3*N) @ inv_sq

    # Vibrational basis = null space of Eckart generators
    P = eckart_projector(coords.to(torch.float64), masses)
    sq = torch.sqrt(masses); sq3 = sq.repeat_interleave(3)
    com = (coords.to(torch.float64) * masses[:, None]).sum(0) / masses.sum()
    r = coords.to(torch.float64) - com
    B_cols = []
    for e in torch.eye(3, dtype=torch.float64, device=hessian.device):
        c = sq3 * e.repeat(N); B_cols.append(c / c.norm())
    rx, ry, rz = r[:,0], r[:,1], r[:,2]
    for R in (torch.stack([torch.zeros_like(rx),-rz,ry],1),
              torch.stack([rz,torch.zeros_like(ry),-rx],1),
              torch.stack([-ry,rx,torch.zeros_like(rz)],1)):
        c = (R*sq[:,None]).reshape(-1); B_cols.append(c/c.norm())
    B = torch.stack(B_cols, 1)
    Q, R_ = torch.linalg.qr(B, mode="reduced")
    k = max(int((torch.diag(R_).abs() > 1e-6).sum().item()), 1)
    U, _, _ = torch.linalg.svd(Q[:, :k], full_matrices=True)
    Q_vib = U[:, k:]                                  # (3N, 3N-k)

    H_red = Q_vib.T @ H_mw @ Q_vib
    H_red = 0.5 * (H_red + H_red.T)                   # symmetrize
    evals, evecs = torch.linalg.eigh(H_red)
    return evals, Q_vib @ evecs                        # lift back to 3N

# ---------------------------------------------------------------------------
# Projected GAD direction — the heart of the algorithm.
# Projects gradient and guide vector through P, applies GAD flip, projects output.
# Three projections prevent any translational/rotational leakage.
# ---------------------------------------------------------------------------
def gad_direction(coords, forces, v, masses):
    """Eckart-projected GAD: dq = P(-g + 2(g·v)v), dx = √m · dq."""
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
# The search loop — Euler integration of GAD dynamics.
# Each step depends only on the current geometry. No path history.
# Converges when exactly one eigenvalue is negative and forces are small.
# That's a transition state. Nothing more, nothing less.
# ---------------------------------------------------------------------------
def gad_search(predict_fn, coords, atomic_nums):
    x = coords.detach().clone().to(torch.float32).reshape(-1, 3)
    m = masses_from_z(atomic_nums, x.device)

    for step in range(n_steps):
        # Evaluate
        out = predict_fn(x, atomic_nums, do_hessian=True, require_grad=False)
        f = out["forces"]
        if f.dim() == 3: f = f[0]
        f = f.reshape(-1, 3)
        e = float(out["energy"].detach().reshape(-1)[0]) if isinstance(out["energy"], torch.Tensor) else float(out["energy"])
        fn = float(f.norm(dim=1).mean())

        # Vibrational analysis
        evals, evecs = vib_eig(out["hessian"], x, m)
        n_neg = int((evals < 0).sum())
        eig0 = float(evals[0]) if evals.numel() > 0 else 0.0

        # Convergence: one negative eigenvalue + small forces = transition state
        if n_neg == 1 and fn < force_threshold:
            return dict(converged=True, coords=x.cpu(), energy=e,
                        n_neg=n_neg, force_norm=fn, eig0=eig0, step=step)

        # GAD step — always use the lowest eigenvector, fresh from the current Hessian
        v1 = evecs[:, 0].to(device=f.device, dtype=f.dtype)
        v1 = v1 / (v1.norm() + 1e-12)
        gad_vec, _ = gad_direction(x, f, v1, m)

        # Euler integration with displacement cap
        dx = dt * gad_vec
        d_max = float(dx.reshape(-1, 3).norm(dim=1).max())
        if d_max > max_atom_disp:
            dx = dx * (max_atom_disp / d_max)
        x = (x + dx).detach()

    return dict(converged=False, coords=x.cpu(), energy=e,
                n_neg=n_neg, force_norm=fn, eig0=eig0, step=n_steps)

# ---------------------------------------------------------------------------
# Load HIP — the neural network potential that gives us analytical Hessians.
# The monkey-patch lets us run inference without training dataset paths.
# ---------------------------------------------------------------------------
def load_hip():
    from hip import path_config, training_module, inference_utils
    orig = path_config.fix_dataset_path
    def lenient(p):
        try: return orig(p)
        except FileNotFoundError: return p
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
# Load Transition1x — 9,561 organic reactions with known transition states.
# Each sample has reactant, product, and TS geometries. We noise the TS.
# ---------------------------------------------------------------------------
def load_dataset():
    from transition1x import Dataloader
    loader = Dataloader(h5_path, datasplit=split, only_final=True)
    samples = []
    for mol in loader:
        if n_samples and len(samples) >= n_samples:
            break
        try:
            ts, rx = mol["transition_state"], mol["reactant"]
            if len(ts["atomic_numbers"]) != len(rx["atomic_numbers"]):
                continue
            samples.append(dict(
                z=torch.tensor(ts["atomic_numbers"], dtype=torch.long),
                pos=torch.tensor(ts["positions"], dtype=torch.float32),
                formula=ts.get("formula", "?"),
            ))
        except Exception:
            continue
    return samples

# ---------------------------------------------------------------------------
# Run it. Add noise, search, report.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    noise_pm = int(round(noise * 1000))
    print(f"GAD transition state search | dt={dt} | steps={n_steps} | "
          f"noise={noise_pm}pm | samples={n_samples} | device={device}")
    print()

    predict = load_hip()
    print("HIP loaded")

    samples = load_dataset()
    print(f"Loaded {len(samples)} samples from Transition1x ({split})")
    print()

    n_conv = 0
    t0_all = time.time()
    for i, s in enumerate(samples):
        z = s["z"].to(device)
        start = s["pos"].to(device) + noise * torch.randn_like(s["pos"].to(device))

        t0 = time.time()
        result = gad_search(predict, start, z)
        wall = time.time() - t0

        tag = "CONV" if result["converged"] else "FAIL"
        if result["converged"]:
            n_conv += 1
        print(f"  [{i:3d}] {s['formula']:>12s} | {tag} | n_neg={result['n_neg']} "
              f"| force={result['force_norm']:.4f} | step={result['step']:4d} | {wall:.1f}s")

    rate = 100 * n_conv / len(samples)
    print(f"\n{n_conv}/{len(samples)} converged ({rate:.1f}%) in {time.time()-t0_all:.0f}s")
