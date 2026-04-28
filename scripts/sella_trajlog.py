#!/usr/bin/env python
"""Sella with per-step trajectory logging.

Run a small Sella sweep that logs every step's geometry, energy, force,
and per-step displacement to a Parquet file. Used to produce a
step-size-vs-step comparison figure against GAD.

Output one trajectory parquet per sample under --output-dir, schema:
  step, sample_id, formula, energy, force_max, force_norm, disp_from_last,
  delta_trust, coords_flat

Plus a single summary parquet aggregating wall time, n_steps, converged,
n_neg, fmax for all samples.

Usage:
  python scripts/sella_trajlog.py --noise 0.10 --n-samples 50 \\
      --output-dir /lustre07/scratch/memoozd/gadplus/runs/sella_trajlog
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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore", message=".*weights_only.*")

# Reuse the calculator + hessian wrapping from sella_baseline
from sella_baseline import HipSellaCalculator, make_hessian_function  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--noise", type=float, default=0.10, help="Gaussian noise std (A)")
    ap.add_argument("--n-samples", type=int, default=50)
    ap.add_argument("--max-steps", type=int, default=2000)
    ap.add_argument("--fmax", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split", type=str, default="train")
    ap.add_argument("--cartesian", action="store_true", default=True,
                    help="Use Cartesian coords (default: True for fair comparison with GAD)")
    ap.add_argument("--apply-eckart", action="store_true", default=True)
    ap.add_argument("--config-name", type=str, default="sella_carte_eckart_traj",
                    help="Used as filename prefix")
    ap.add_argument("--delta0", type=float, default=0.048)
    ap.add_argument("--gamma", type=float, default=0.0,
                    help="Sella line-search gamma (0 disables line search)")
    ap.add_argument("--output-dir", type=str, required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    noise_pm = int(round(args.noise * 1000))
    use_internal = not args.cartesian
    apply_eckart = args.apply_eckart
    print(f"Device={device} | sella_trajlog | noise={noise_pm}pm | n={args.n_samples} | "
          f"cartesian={args.cartesian} | eckart={apply_eckart} | "
          f"delta0={args.delta0} | gamma={args.gamma}")

    for ckpt in ["/lustre06/project/6033559/memoozd/models/hip_v2.ckpt",
                 "/project/rrg-aspuru/memoozd/models/hip_v2.ckpt"]:
        if os.path.exists(ckpt):
            ckpt_path = ckpt; break
    else:
        sys.exit("ckpt not found")
    for h5 in ["/lustre06/project/6033559/memoozd/data/transition1x.h5",
               "/project/rrg-aspuru/memoozd/data/transition1x.h5"]:
        if os.path.exists(h5):
            h5_path = h5; break
    else:
        sys.exit("h5 not found")

    os.makedirs(args.output_dir, exist_ok=True)

    from gadplus.calculator.hip import load_hip_calculator, make_hip_predict_fn
    calculator = load_hip_calculator(ckpt_path, device=device)
    predict_fn = make_hip_predict_fn(calculator)

    from gadplus.data.transition1x import Transition1xDataset, UsePos
    dataset = Transition1xDataset(h5_path, split=args.split, max_samples=args.n_samples,
                                  transform=UsePos("pos_transition"))

    # Same noise-vec generation as GAD baseline (seed-matched)
    torch.manual_seed(args.seed)
    noise_vecs = {}
    for i in range(len(dataset)):
        noise_vecs[i] = torch.randn_like(dataset[i].pos) * args.noise

    from sella import Sella
    from gadplus.projection import vib_eig, atomic_nums_to_symbols

    summary_rows = []
    run_id = f"{args.config_name}_{noise_pm}pm_{uuid.uuid4().hex[:8]}"

    for i in range(len(dataset)):
        sample = dataset[i]
        coords_ts = sample.pos.to(device)
        z = sample.z.to(device)
        formula = getattr(sample, "formula", f"sample_{i}")
        coords_start = coords_ts + noise_vecs[i].to(device)

        positions_np = coords_start.detach().cpu().numpy().reshape(-1, 3)
        numbers_np = z.detach().cpu().numpy().flatten().astype(int)
        atoms = Atoms(numbers=numbers_np, positions=positions_np)
        ase_calc = HipSellaCalculator(predict_fn, z, device=device)
        atoms.calc = ase_calc
        hessian_fn = make_hessian_function(ase_calc, apply_eckart=apply_eckart)

        # Per-step log buffer
        traj_rows = []
        prev_pos = positions_np.copy()

        def log_step():
            # Called after every Sella step via opt.attach
            pos = atoms.positions.copy()
            disp = float(np.linalg.norm(pos.flatten() - prev_pos.flatten()))
            forces = atoms.get_forces()
            energy = float(atoms.get_potential_energy())
            try:
                delta_trust = float(getattr(opt, "delta", np.nan))
            except Exception:
                delta_trust = float("nan")
            traj_rows.append({
                "step": len(traj_rows),
                "sample_id": i,
                "formula": formula,
                "energy": energy,
                "force_max": float(np.max(np.abs(forces))),
                "force_norm": float(np.mean(np.linalg.norm(forces, axis=1))),
                "disp_from_last": disp,
                "delta_trust": delta_trust,
                "coords_flat": pos.flatten().astype(np.float32).tolist(),
            })
            # update prev_pos for next step
            prev_pos[:] = pos

        t0 = time.time()
        try:
            opt = Sella(
                atoms, order=1, internal=use_internal,
                trajectory=None, logfile=None,
                delta0=args.delta0, hessian_function=hessian_fn,
                diag_every_n=1, gamma=args.gamma,
                rho_inc=1.035, rho_dec=5.0,
                sigma_inc=1.15, sigma_dec=0.65,
            )
            # Log step 0 (initial state)
            log_step()
            opt.attach(log_step, interval=1)
            sella_converged = opt.run(fmax=args.fmax, steps=args.max_steps)
            steps_taken = opt.nsteps
        except Exception as e:
            print(f"  [{i:3d}] {formula:>12s} ERROR: {e}")
            sella_converged = False
            steps_taken = args.max_steps
        wall = time.time() - t0

        # Final eval (n_neg, fmax)
        try:
            final_coords = torch.tensor(atoms.positions, dtype=torch.float32, device=device)
            out = predict_fn(final_coords, z, do_hessian=True, require_grad=False)
            forces_final = out["forces"]
            if isinstance(forces_final, torch.Tensor):
                forces_final = forces_final.detach().cpu().numpy()
            forces_final = np.asarray(forces_final).reshape(-1, 3)
            final_fmax = float(np.max(np.abs(forces_final)))
            final_force_norm = float(np.mean(np.linalg.norm(forces_final, axis=1)))
            atomsymbols = atomic_nums_to_symbols(z)
            evals_vib, _, _ = vib_eig(out["hessian"], final_coords, atomsymbols)
            n_neg = int((evals_vib < 0).sum().item())
        except Exception:
            final_fmax = 999.0; final_force_norm = 999.0; n_neg = 0

        converged_ours = (n_neg == 1) and (final_fmax < 0.01)
        print(f"  [{i:3d}] {formula:>12s} | sella={'CONV' if sella_converged else 'FAIL'} "
              f"steps={steps_taken:4d} fmax={final_fmax:.4f} n_neg={n_neg} | {wall:.1f}s")

        # Write per-sample trajectory parquet
        if traj_rows:
            traj_df = pd.DataFrame(traj_rows)
            tp = os.path.join(args.output_dir,
                              f"traj_{args.config_name}_{noise_pm}pm_{run_id[-8:]}_{i}.parquet")
            traj_df.to_parquet(tp, compression="snappy")

        summary_rows.append({
            "method": args.config_name,
            "noise_pm": noise_pm,
            "sample_id": i,
            "formula": formula,
            "sella_converged": sella_converged,
            "converged": converged_ours,
            "n_neg": n_neg,
            "fmax_final": final_fmax,
            "force_norm_final": final_force_norm,
            "n_steps": steps_taken,
            "wall_s": wall,
            "delta0": args.delta0,
            "gamma": args.gamma,
        })

    sp = os.path.join(args.output_dir,
                      f"summary_{args.config_name}_{noise_pm}pm.parquet")
    pd.DataFrame(summary_rows).to_parquet(sp, compression="snappy")
    print(f"wrote {sp}")


if __name__ == "__main__":
    main()
