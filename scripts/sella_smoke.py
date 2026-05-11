"""Parallel Sella smoke runner for SCINE / xTB calculators.

Self-contained, intentionally NOT a refactor of sella_baseline.py — the
HIP baseline keeps that script untouched while we explore alternative
calculators here.

Mirrors the canonical Sella configuration used in the paper:
  Cartesian + Eckart Hessian projection, hessian_function every step,
  delta0=0.1, gamma=0.4 (Sella library defaults).

Usage:
    python scripts/sella_smoke.py \\
        --backend scine --method DFTB0 \\
        --noise 1.0 --n-samples 5 --max-steps 500 \\
        --output-dir /lustre07/scratch/memoozd/gadplus/runs/smoke_scine_sella
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import uuid
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")


def _build_predict_fn(backend: str, method: str):
    if backend == "scine":
        from gadplus.calculator.scine import (
            load_scine_calculator, make_scine_predict_fn,
        )
        return make_scine_predict_fn(
            load_scine_calculator(functional=method, device="cpu")
        )
    if backend == "xtb":
        from gadplus.calculator.xtb import load_xtb_calculator, make_xtb_predict_fn
        return make_xtb_predict_fn(
            load_xtb_calculator(method=method, device="cpu")
        )
    raise ValueError(f"Unknown backend: {backend!r}")


def _make_calculator_and_hessfn(predict_fn, atomic_nums, apply_eckart: bool):
    """Return (ASE Calculator, hessian_function). Captures `predict_fn` and
    caches Hessians within each step.
    """
    import numpy as np
    import torch
    from ase.calculators.calculator import Calculator, all_changes

    class _Calc(Calculator):
        implemented_properties = ["energy", "forces"]

        def __init__(self):
            super().__init__()
            self.predict_fn = predict_fn
            self.atomic_nums = atomic_nums
            self._cache = None
            self.n_calls = 0

        def calculate(self, atoms=None, properties=None, system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            coords = torch.tensor(
                self.atoms.positions, dtype=torch.float64, device="cpu"
            )
            out = self.predict_fn(
                coords, self.atomic_nums, do_hessian=True, require_grad=False
            )
            self.n_calls += 1
            self._cache = (coords.clone(), out)

            e = out["energy"]
            self.results["energy"] = float(
                e.detach().cpu().item() if isinstance(e, torch.Tensor) else e
            )
            f = out["forces"]
            self.results["forces"] = (
                f.detach().cpu().numpy() if isinstance(f, torch.Tensor) else np.asarray(f)
            ).reshape(-1, 3)

    calc = _Calc()

    def hessian_function(atoms):
        coords = torch.tensor(atoms.positions, dtype=torch.float64, device="cpu")
        if calc._cache is not None and torch.equal(coords, calc._cache[0]):
            hess = calc._cache[1]["hessian"]
        else:
            out = calc.predict_fn(
                coords, calc.atomic_nums, do_hessian=True, require_grad=False
            )
            hess = out["hessian"]
            calc._cache = (coords.clone(), out)
            calc.n_calls += 1

        h = hess.detach().to(torch.float64) if isinstance(hess, torch.Tensor) \
            else torch.tensor(hess, dtype=torch.float64)
        n = len(atoms)
        h = h.reshape(3 * n, 3 * n)

        if apply_eckart:
            from gadplus.projection.projection import (
                get_mass_weights, _eckart_projector, atomic_nums_to_symbols,
            )
            atomsymbols = atomic_nums_to_symbols(calc.atomic_nums)
            coords_3d = coords.reshape(-1, 3)
            masses, _m3, sqrt_m, sqrt_m_inv = get_mass_weights(
                atomsymbols, device=h.device
            )
            diag_inv = torch.diag(sqrt_m_inv)
            diag_m = torch.diag(sqrt_m)
            H_mw = diag_inv @ h @ diag_inv
            P = _eckart_projector(coords_3d, masses)
            H_mw_proj = P @ H_mw @ P
            H_mw_proj = 0.5 * (H_mw_proj + H_mw_proj.T)
            h = diag_m @ H_mw_proj @ diag_m

        return h.cpu().numpy().astype(np.float64)

    return calc, hessian_function


def _run_one_sample(args_tuple):
    (
        sample_idx, h5_path, split, backend, method, noise_ang, seed,
        max_steps, fmax, apply_eckart, internal_coords,
        delta0, gamma, diag_every, output_dir, run_id,
    ) = args_tuple

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    import numpy as np
    import torch
    torch.set_num_threads(1)
    from ase import Atoms
    from sella import Sella

    from gadplus.data.transition1x import Transition1xDataset, UsePos
    from gadplus.geometry.starting import make_starting_coords
    from gadplus.projection import vib_eig, atomic_nums_to_symbols

    ds = Transition1xDataset(
        h5_path=h5_path, split=split, max_samples=sample_idx + 1,
        transform=UsePos("pos_transition"),
    )
    sample = ds[sample_idx]
    formula = str(getattr(sample, "formula", f"sample_{sample_idx}"))
    rxn = str(getattr(sample, "rxn", ""))

    coords_start = make_starting_coords(
        sample, "noised_ts", noise_rms=noise_ang, seed=seed,
    ).to(torch.float64)
    z = sample.z.to(torch.long)

    predict_fn = _build_predict_fn(backend, method)
    calc, hessian_fn = _make_calculator_and_hessfn(predict_fn, z, apply_eckart)

    positions_np = coords_start.detach().cpu().numpy().reshape(-1, 3)
    numbers_np = z.detach().cpu().numpy().flatten().astype(int)
    atoms = Atoms(numbers=numbers_np, positions=positions_np)
    atoms.calc = calc

    t0 = time.time()
    try:
        opt = Sella(
            atoms=atoms, order=1, internal=internal_coords,
            trajectory=None, logfile=None,
            delta0=delta0, gamma=gamma,
            diag_every_n=diag_every,
            hessian_function=hessian_fn,
            rho_inc=1.035, rho_dec=5.0,
            sigma_inc=1.15, sigma_dec=0.65,
        )
        sella_converged = opt.run(fmax=fmax, steps=max_steps)
        steps_taken = int(opt.nsteps)
    except Exception as exc:
        sella_converged = False
        steps_taken = max_steps
        err = repr(exc)
    else:
        err = ""
    wall = time.time() - t0

    # Final-state evaluation via the same predict_fn (1 extra Hessian call).
    final_coords = torch.tensor(
        atoms.positions, dtype=torch.float64, device="cpu"
    )
    try:
        out_final = predict_fn(final_coords, z, do_hessian=True, require_grad=False)
        forces_t = out_final["forces"]
        forces_np = (
            forces_t.detach().cpu().numpy() if isinstance(forces_t, torch.Tensor)
            else np.asarray(forces_t)
        ).reshape(-1, 3)
        final_fmax = float(np.max(np.abs(forces_np)))
        final_force_norm = float(np.mean(np.linalg.norm(forces_np, axis=1)))
        energy_t = out_final["energy"]
        final_energy = float(
            energy_t.detach().cpu().item() if isinstance(energy_t, torch.Tensor)
            else energy_t
        )
        atomsymbols = atomic_nums_to_symbols(z)
        evals_vib, _, _ = vib_eig(out_final["hessian"], final_coords, atomsymbols)
        n_neg = int((evals_vib < 0).sum().item())
        eig0 = float(evals_vib[0].item()) if evals_vib.numel() > 0 else 0.0
    except Exception:
        final_fmax = float("nan"); final_force_norm = float("nan")
        final_energy = float("nan"); n_neg = -1; eig0 = float("nan")

    our_converged = (n_neg == 1) and (final_fmax < fmax)

    return {
        "sample_id": int(sample_idx),
        "formula": formula,
        "rxn": rxn,
        "noise_angstrom": float(noise_ang),
        "search_method": f"sella_{'internal' if internal_coords else 'cart'}"
                        f"{'_eckart' if apply_eckart else ''}_{backend}_{method}",
        "converged": bool(our_converged),
        "sella_converged": bool(sella_converged),
        "converged_step": int(steps_taken) if our_converged else -1,
        "total_steps": int(steps_taken),
        "final_n_neg": int(n_neg),
        "final_force_norm": float(final_force_norm),
        "final_force_max": float(final_fmax),
        "final_energy": float(final_energy),
        "final_eig0": float(eig0),
        "n_calculator_calls": int(getattr(calc, "n_calls", 0)),
        "wall_time_s": float(wall),
        "failure_type": err,
        # Final geometry saved as a flat list so downstream IRC validation
        # can pull TS coords directly from the summary parquet.
        "final_coords_flat": atoms.positions.reshape(-1).astype(float).tolist(),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", required=True, choices=["scine", "xtb"])
    p.add_argument("--method", default="DFTB0")
    p.add_argument("--noise", type=float, default=1.0,
                   help="Gaussian RMS noise on TS in Angstrom. 1.0 A = 100 pm.")
    p.add_argument("--n-samples", type=int, default=5)
    p.add_argument("--sample-indices", type=str, default=None,
                   help="Comma-separated 0-indexed sample IDs. Overrides --n-samples.")
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--fmax", type=float, default=0.01)
    p.add_argument("--apply-eckart", action="store_true", default=True,
                   help="Eckart-project Hessian (canonical). On by default.")
    p.add_argument("--no-eckart", dest="apply_eckart", action="store_false")
    p.add_argument("--internal", action="store_true", default=False,
                   help="Use internal coordinates (default: Cartesian).")
    p.add_argument("--delta0", type=float, default=0.1)
    p.add_argument("--gamma", type=float, default=0.4)
    p.add_argument("--diag-every", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--split", default="test")
    p.add_argument("--h5", default="/lustre06/project/6033559/memoozd/data/transition1x.h5")
    p.add_argument("--n-workers", type=int,
                   default=int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    run_id = str(uuid.uuid4())[:8]

    print(f"Backend: {args.backend} | Method: {args.method}")
    print(f"Sella: cart={not args.internal} eckart={args.apply_eckart} "
          f"delta0={args.delta0} gamma={args.gamma} diag_every={args.diag_every}")
    print(f"Samples: {args.n_samples} | Max steps: {args.max_steps} | fmax: {args.fmax}")
    print(f"Noise: {args.noise} A | Workers: {args.n_workers}")
    print(f"Output: {args.output_dir}")

    if args.sample_indices:
        indices = [int(x) for x in args.sample_indices.split(",") if x.strip()]
    else:
        indices = list(range(args.n_samples))
    task_args = [
        (
            i, args.h5, args.split, args.backend, args.method, args.noise,
            args.seed + 1000 * i, args.max_steps, args.fmax,
            args.apply_eckart, args.internal,
            args.delta0, args.gamma, args.diag_every,
            args.output_dir, run_id,
        )
        for i in indices
    ]

    results = []
    t_overall = time.time()
    with ProcessPoolExecutor(max_workers=args.n_workers) as exe:
        future_to_idx = {exe.submit(_run_one_sample, ta): ta[0] for ta in task_args}
        for fut in as_completed(future_to_idx):
            idx = future_to_idx[fut]
            try:
                r = fut.result()
            except Exception as exc:
                print(f"  [{idx}] FAILED: {exc}")
                continue
            status = "TS" if r["converged"] else f"no(n_neg={r['final_n_neg']})"
            print(f"  [{r['sample_id']}] {r['formula']} | {status} | "
                  f"fmax={r['final_force_max']:.4e} steps={r['total_steps']} "
                  f"wall={r['wall_time_s']:.1f}s")
            results.append(r)

    total_wall = time.time() - t_overall

    summary_path = os.path.join(args.output_dir, f"summary_{run_id}.parquet")
    pq.write_table(pa.Table.from_pylist(results), summary_path)

    n_total = len(results)
    n_conv = sum(1 for r in results if r["converged"])
    n_sella = sum(1 for r in results if r["sella_converged"])
    print()
    print("=" * 60)
    print(f"{args.backend}/{args.method} Sella "
          f"{'internal' if args.internal else 'cart'}"
          f"{'+Eckart' if args.apply_eckart else ''}: "
          f"{n_conv}/{n_total} TS converged (n_neg=1 ∧ fmax<{args.fmax}); "
          f"sella self-reports {n_sella}/{n_total} | total wall: {total_wall:.1f}s")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
