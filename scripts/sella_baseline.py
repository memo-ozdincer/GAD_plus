#!/usr/bin/env python
"""Sella TS baseline: single fmax threshold at single noise level.

Matches our GAD experiments exactly: same 300 T1x samples, same noise seeds,
same noise levels. After Sella converges (or hits max_steps), runs our own
Eckart-projected vib_eig to check n_neg on the final geometry.

Usage:
  python scripts/sella_baseline.py --fmax 0.01 --noise 0.05 --n-samples 300

Three comparison modes:
  --fmax 0.03   Sella default threshold
  --fmax 0.01   Match our GAD force_threshold
  --fmax 0.001  Very tight, then retroactively apply n_neg==1 + force<0.01
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import uuid
import warnings

import numpy as np
import pandas as pd
import torch
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

warnings.filterwarnings("ignore", message=".*weights_only.*")


# ---------------------------------------------------------------------------
# HIP ASE Calculator with Hessian caching for Sella
# ---------------------------------------------------------------------------
class HipSellaCalculator(Calculator):
    """ASE Calculator that caches HIP energy+forces+Hessian in one forward pass.

    Sella calls calculate() for energy/forces, then hessian_function() for the
    Hessian. The cache avoids running HIP twice per step.
    """
    implemented_properties = ["energy", "forces"]

    def __init__(self, predict_fn, atomic_nums, device="cuda", **kwargs):
        super().__init__(**kwargs)
        self.predict_fn = predict_fn
        self.atomic_nums = atomic_nums
        self.device = device
        self._cached_coords = None
        self._cached_result = None

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        coords = torch.tensor(self.atoms.positions, dtype=torch.float32, device=self.device)

        # Always compute with Hessian so we can cache it
        out = self.predict_fn(coords, self.atomic_nums, do_hessian=True, require_grad=False)

        # Cache for hessian_function
        self._cached_coords = coords.clone()
        self._cached_result = out

        energy = out["energy"]
        if isinstance(energy, torch.Tensor):
            energy = energy.detach().cpu().item()
        self.results["energy"] = float(energy)

        forces = out["forces"]
        if isinstance(forces, torch.Tensor):
            forces = forces.detach().cpu().numpy()
        self.results["forces"] = np.asarray(forces).reshape(-1, 3)


def make_hessian_function(calc, apply_eckart=False):
    """Create a hessian_function callable for Sella that reads from cache.

    If apply_eckart=True, Eckart-projects the Hessian (removes TR modes in
    mass-weighted space) then converts back to Cartesian before passing to Sella.
    """
    def hessian_function(atoms):
        # Check cache
        coords = torch.tensor(atoms.positions, dtype=torch.float32, device=calc.device)
        if calc._cached_coords is not None and torch.equal(coords, calc._cached_coords):
            hess = calc._cached_result["hessian"]
        else:
            # Cache miss — recompute
            out = calc.predict_fn(coords, calc.atomic_nums, do_hessian=True, require_grad=False)
            hess = out["hessian"]
            calc._cached_coords = coords.clone()
            calc._cached_result = out

        if isinstance(hess, torch.Tensor):
            hess_t = hess.detach()
        else:
            hess_t = torch.tensor(hess)

        n = len(atoms)
        hess_t = hess_t.reshape(3 * n, 3 * n).to(torch.float64)

        if apply_eckart:
            # Eckart project: H_cart -> M^{-1/2} H M^{-1/2} -> P H_mw P -> M^{1/2} H_proj M^{1/2}
            from gadplus.projection.projection import get_mass_weights, _eckart_projector, atomic_nums_to_symbols
            atomsymbols = atomic_nums_to_symbols(calc.atomic_nums)
            coords_3d = coords.reshape(-1, 3).to(torch.float64)
            masses, m3, sqrt_m, sqrt_m_inv = get_mass_weights(atomsymbols, device=hess_t.device)
            # Mass-weight
            diag_inv = torch.diag(sqrt_m_inv)
            H_mw = diag_inv @ hess_t @ diag_inv
            # Project
            P = _eckart_projector(coords_3d, masses)
            H_mw_proj = P @ H_mw @ P
            H_mw_proj = 0.5 * (H_mw_proj + H_mw_proj.T)
            # Un-mass-weight back to Cartesian
            diag_m = torch.diag(sqrt_m)
            hess_t = diag_m @ H_mw_proj @ diag_m

        return hess_t.cpu().numpy().astype(np.float64)
    return hessian_function


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fmax", type=float, required=True, help="Sella force convergence threshold (eV/A)")
    parser.add_argument("--noise", type=float, required=True, help="Gaussian noise std (Angstrom)")
    parser.add_argument("--n-samples", type=int, default=300)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--internal", action="store_true", default=True,
                        help="Use internal coordinates (default: True)")
    parser.add_argument("--cartesian", action="store_true", default=False,
                        help="Use Cartesian coordinates instead of internal")
    parser.add_argument("--apply-eckart", action="store_true", default=False,
                        help="Eckart-project Hessian before passing to Sella")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    use_internal = not args.cartesian
    apply_eckart = args.apply_eckart

    device = "cuda" if torch.cuda.is_available() else "cpu"
    noise_pm = int(round(args.noise * 1000))
    fmax_str = f"{args.fmax}".replace(".", "p")
    coord_str = "internal" if use_internal else "cartesian"
    eckart_str = "_eckart" if apply_eckart else ""
    config_name = f"sella_{coord_str}{eckart_str}_fmax{fmax_str}"
    print(f"Device: {device} | Sella baseline | fmax={args.fmax} | noise={noise_pm}pm | "
          f"coords={coord_str} | eckart={apply_eckart} | samples={args.n_samples} | max_steps={args.max_steps}")

    # ---- Paths ----
    for ckpt_path in [
        "/lustre06/project/6033559/memoozd/models/hip_v2.ckpt",
        "/project/rrg-aspuru/memoozd/models/hip_v2.ckpt",
    ]:
        if os.path.exists(ckpt_path):
            break
    else:
        sys.exit("hip_v2.ckpt not found")

    for h5_path in [
        "/lustre06/project/6033559/memoozd/data/transition1x.h5",
        "/project/rrg-aspuru/memoozd/data/transition1x.h5",
    ]:
        if os.path.exists(h5_path):
            break
    else:
        sys.exit("transition1x.h5 not found")

    output_dir = args.output_dir or "/lustre07/scratch/memoozd/gadplus/runs/sella_baselines"
    os.makedirs(output_dir, exist_ok=True)

    # ---- Load HIP ----
    from gadplus.calculator.hip import load_hip_calculator, make_hip_predict_fn
    calculator = load_hip_calculator(ckpt_path, device=device)
    predict_fn = make_hip_predict_fn(calculator)
    print("HIP loaded")

    # ---- Load dataset ----
    from gadplus.data.transition1x import Transition1xDataset, UsePos
    dataset = Transition1xDataset(
        h5_path, split=args.split,
        max_samples=args.n_samples,
        transform=UsePos("pos_transition"),
    )
    print(f"Loaded {len(dataset)} samples (split={args.split})")

    # ---- Pre-generate noise (same seed as GAD experiments) ----
    torch.manual_seed(args.seed)
    noise_vecs = {}
    for i in range(len(dataset)):
        sample = dataset[i]
        noise_vecs[i] = torch.randn_like(sample.pos) * args.noise

    # ---- Import Sella ----
    from sella import Sella
    from gadplus.projection import vib_eig, atomic_nums_to_symbols

    # ---- Run ----
    run_id = f"{config_name}_{noise_pm}pm_{uuid.uuid4().hex[:8]}"
    results = []
    t_total = time.time()

    for i in range(len(dataset)):
        sample = dataset[i]
        coords_ts = sample.pos.to(device)
        z = sample.z.to(device)
        formula = getattr(sample, "formula", f"sample_{i}")

        coords_start = coords_ts + noise_vecs[i].to(device)

        # Create ASE Atoms
        positions_np = coords_start.detach().cpu().numpy().reshape(-1, 3)
        numbers_np = z.detach().cpu().numpy().flatten().astype(int)
        atoms = Atoms(numbers=numbers_np, positions=positions_np)

        # Attach calculator with Hessian caching
        ase_calc = HipSellaCalculator(predict_fn, z, device=device)
        atoms.calc = ase_calc
        hessian_fn = make_hessian_function(ase_calc, apply_eckart=apply_eckart)

        # Run Sella
        t0 = time.time()
        try:
            opt = Sella(
                atoms,
                order=1,
                internal=use_internal,
                trajectory=None,
                logfile=None,
                delta0=0.048,
                hessian_function=hessian_fn,
                diag_every_n=1,
                gamma=0.0,
                rho_inc=1.035,
                rho_dec=5.0,
                sigma_inc=1.15,
                sigma_dec=0.65,
            )
            sella_converged = opt.run(fmax=args.fmax, steps=args.max_steps)
            steps_taken = opt.nsteps
        except Exception as e:
            print(f"  [{i:3d}] {formula:>12s} | ERROR: {e}")
            sella_converged = False
            steps_taken = args.max_steps

        wall = time.time() - t0

        # Get final state — recompute to ensure cache is fresh
        try:
            final_coords = torch.tensor(atoms.positions, dtype=torch.float32, device=device)
            out_final = predict_fn(final_coords, z, do_hessian=True, require_grad=False)
            forces_t = out_final["forces"]
            if isinstance(forces_t, torch.Tensor):
                forces_np = forces_t.detach().cpu().numpy().reshape(-1, 3)
            else:
                forces_np = np.asarray(forces_t).reshape(-1, 3)
            final_fmax = float(np.max(np.abs(forces_np)))
            final_force_norm = float(np.mean(np.linalg.norm(forces_np, axis=1)))
            energy_t = out_final["energy"]
            final_energy = float(energy_t.detach().cpu().item()) if isinstance(energy_t, torch.Tensor) else float(energy_t)

            # Our convergence check: Eckart-projected n_neg + force_norm
            atomsymbols = atomic_nums_to_symbols(z)
            evals_vib, _, _ = vib_eig(out_final["hessian"], final_coords, atomsymbols)
            n_neg = int((evals_vib < 0).sum().item())
            eig0 = float(evals_vib[0].item()) if evals_vib.numel() > 0 else 0.0
            eig1 = float(evals_vib[1].item()) if evals_vib.numel() > 1 else 0.0
        except Exception as e2:
            print(f"  [{i:3d}] {formula:>12s} | EVAL ERROR: {e2}")
            final_fmax = 999.0
            final_force_norm = 999.0
            final_energy = 0.0
            n_neg = 0
            eig0 = 0.0
            eig1 = 0.0

        # Our criterion: n_neg==1 AND force_norm < 0.01
        our_converged = (n_neg == 1) and (final_force_norm < 0.01)

        status_sella = "CONV" if sella_converged else "FAIL"
        status_ours = "TS" if our_converged else "no"
        print(f"  [{i:3d}] {formula:>12s} | sella={status_sella} fmax={final_fmax:.4f} | "
              f"n_neg={n_neg} force={final_force_norm:.4f} ours={status_ours} | "
              f"steps={steps_taken:3d} | {wall:.1f}s")

        # All convergence criteria combinations
        nneg1 = (n_neg == 1)
        force_001 = (final_force_norm < 0.01)
        force_003 = (final_force_norm < 0.03)
        fmax_001 = (final_fmax < 0.01)
        fmax_003 = (final_fmax < 0.03)

        results.append({
            "method": config_name,
            "noise_pm": noise_pm,
            "sample_id": i,
            "formula": formula,
            # Raw final state
            "final_n_neg": n_neg,
            "final_fmax": final_fmax,
            "final_force_norm": final_force_norm,
            "final_energy": final_energy,
            "final_eig0": eig0,
            "final_eig1": eig1,
            "total_steps": steps_taken,
            "wall_time_s": wall,
            # Sella's own convergence
            "sella_converged": sella_converged,
            # Individual criteria
            "is_nneg1": nneg1,
            "is_force_001": force_001,
            "is_force_003": force_003,
            "is_fmax_001": fmax_001,
            "is_fmax_003": fmax_003,
            # Combined criteria
            "conv_nneg1_force001": nneg1 and force_001,      # Our GAD criterion
            "conv_nneg1_force003": nneg1 and force_003,
            "conv_nneg1_fmax001": nneg1 and fmax_001,
            "conv_nneg1_fmax003": nneg1 and fmax_003,
            "conv_sella_and_nneg1": sella_converged and nneg1,
            # Legacy compatibility
            "converged": our_converged,
            "converged_step": steps_taken if our_converged else None,
        })

    total_wall = time.time() - t_total

    # ---- Save ----
    df = pd.DataFrame(results)
    out_path = os.path.join(output_dir, f"summary_{config_name}_{noise_pm}pm.parquet")
    df.to_parquet(out_path)

    # Report — ALL criteria combinations
    n = len(df)
    print(f"\n{'='*70}")
    print(f"{config_name} @ {noise_pm}pm  ({n} samples, max_steps={args.max_steps})")
    print(f"  Wall time: {total_wall:.0f}s ({total_wall/60:.1f}min)")
    print(f"\n  Individual criteria:")
    for col, label in [
        ("sella_converged", f"Sella fmax<{args.fmax}"),
        ("is_nneg1", "n_neg==1"),
        ("is_force_001", "force_norm<0.01"),
        ("is_force_003", "force_norm<0.03"),
        ("is_fmax_001", "fmax<0.01"),
        ("is_fmax_003", "fmax<0.03"),
    ]:
        c = int(df[col].sum())
        print(f"    {label:25s}: {c:3d}/{n} ({100*c/n:5.1f}%)")
    print(f"\n  Combined criteria:")
    for col, label in [
        ("conv_nneg1_force001", "n_neg1 + force<0.01 (GAD criterion)"),
        ("conv_nneg1_force003", "n_neg1 + force<0.03"),
        ("conv_nneg1_fmax001", "n_neg1 + fmax<0.01"),
        ("conv_nneg1_fmax003", "n_neg1 + fmax<0.03"),
        ("conv_sella_and_nneg1", f"Sella fmax<{args.fmax} + n_neg1"),
    ]:
        c = int(df[col].sum())
        print(f"    {label:40s}: {c:3d}/{n} ({100*c/n:5.1f}%)")
    # Overlap: Sella vs our GAD criterion
    both = int(((df["sella_converged"]) & (df["converged"])).sum())
    sella_only = int(((df["sella_converged"]) & (~df["converged"])).sum())
    ours_only = int(((~df["sella_converged"]) & (df["converged"])).sum())
    neither = int(((~df["sella_converged"]) & (~df["converged"])).sum())
    print(f"\n  Overlap (Sella vs GAD criterion):")
    print(f"    Both={both}  Sella-only={sella_only}  Ours-only={ours_only}  Neither={neither}")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
