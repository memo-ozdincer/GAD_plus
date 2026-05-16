"""Parallel GAD smoke runner for SCINE / xTB calculators.

Self-contained (no Hydra) so we can fan a sample-level multiprocessing
pool across the cores SLURM hands us. Mirrors gad_projected canonical
settings; emits one summary_*.parquet at the end matching the layout
the existing analyzers expect.

Usage:
    python scripts/gad_smoke.py \\
        --backend scine --method DFTB0 \\
        --noise 1.0 --n-samples 5 --n-steps 500 \\
        --output-dir /lustre07/scratch/memoozd/gadplus/runs/smoke_scine_gad
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed

import pyarrow as pa
import pyarrow.parquet as pq

# Make src/ importable without installing.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def _build_predict_fn(backend: str, method: str):
    """Construct a PredictFn for the chosen backend. Called inside workers."""
    if backend == "scine":
        from gadplus.calculator.scine import (
            load_scine_calculator, make_scine_predict_fn,
        )
        calc = load_scine_calculator(functional=method, device="cpu")
        return make_scine_predict_fn(calc)
    if backend == "xtb":
        from gadplus.calculator.xtb import load_xtb_calculator, make_xtb_predict_fn
        calc = load_xtb_calculator(method=method, device="cpu")
        return make_xtb_predict_fn(calc)
    raise ValueError(f"Unknown backend: {backend!r}")


def _run_one_sample(args_tuple):
    """Worker entrypoint. Imports stay inside this fn so each subprocess
    only loads the dependencies it needs.
    """
    (
        sample_idx, h5_path, split, backend, method, noise_ang, seed,
        n_steps, dt, use_projection, force_threshold, force_criterion,
        use_adaptive_dt, dt_min, dt_max, max_atom_disp,
        use_preconditioning, eig_floor,
        k_track, descent_until_nneg, purify_hessian, multimode,
        output_dir, run_id,
    ) = args_tuple

    # Single-threaded BLAS per worker — we get parallelism from the pool.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

    import torch
    torch.set_num_threads(1)

    from gadplus.data.transition1x import Transition1xDataset, UsePos
    from gadplus.geometry.starting import make_starting_coords
    from gadplus.logging.trajectory import TrajectoryLogger
    from gadplus.logging.autopsy import classify_failure
    from gadplus.search.gad_search import GADSearchConfig, run_gad_search

    ds = Transition1xDataset(
        h5_path=h5_path, split=split, max_samples=sample_idx + 1,
        transform=UsePos("pos_transition"),
    )
    sample = ds[sample_idx]
    formula = getattr(sample, "formula", f"sample_{sample_idx}")
    rxn = getattr(sample, "rxn", "")

    coords = make_starting_coords(
        sample, "noised_ts", noise_rms=noise_ang, seed=seed,
    ).to(torch.float32)
    z = sample.z.to(torch.long)
    known_ts = sample.pos_transition.to(torch.float32) if hasattr(sample, "pos_transition") else None

    predict_fn = _build_predict_fn(backend, method)

    cfg = GADSearchConfig(
        n_steps=n_steps, dt=dt, k_track=k_track, beta=1.0,
        use_projection=use_projection,
        use_adaptive_dt=use_adaptive_dt, dt_min=dt_min, dt_max=dt_max,
        max_atom_disp=max_atom_disp,
        use_preconditioning=use_preconditioning, eig_floor=eig_floor,
        force_threshold=force_threshold, force_criterion=force_criterion,
        purify_hessian=purify_hessian,
        descent_until_nneg=descent_until_nneg,
        multimode=multimode,
    )

    start_label = f"noised_ts_noise{noise_ang:.2f}A"
    logger = TrajectoryLogger(
        output_dir=output_dir, run_id=run_id, sample_id=sample_idx,
        start_method=start_label,
        search_method=f"gad_projected_{backend}_{method}",
        rxn=rxn, formula=formula,
    )

    t0 = time.time()
    result = run_gad_search(predict_fn, coords, z, cfg, logger=logger, known_ts_coords=known_ts)
    wall = time.time() - t0

    failure_type = None
    if not result.converged and logger.rows:
        failure_type = classify_failure(logger.rows).value

    return {
        "sample_id": sample_idx,
        "formula": str(formula),
        "rxn": str(rxn),
        "start_method": start_label,
        "search_method": f"gad_projected_{backend}_{method}",
        "converged": bool(result.converged),
        "converged_step": int(result.converged_step) if result.converged_step is not None else -1,
        "total_steps": int(result.total_steps),
        "final_n_neg": int(result.final_n_neg),
        "final_force_norm": float(result.final_force_norm),
        "final_force_max": float(result.final_force_max),
        "final_energy": float(result.final_energy),
        "final_eig0": float(result.final_eig0),
        "wall_time_s": float(wall),
        "failure_type": failure_type or "",
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", required=True, choices=["scine", "xtb"])
    p.add_argument("--method", default="DFTB0",
                   help="SCINE functional (DFTB0/PM6/...) or xTB method (gfn1/gfn2)")
    p.add_argument("--noise", type=float, default=1.0,
                   help="Gaussian RMS noise on TS, in Angstrom. 1.0 A = 100 pm.")
    p.add_argument("--n-samples", type=int, default=5)
    p.add_argument("--sample-indices", type=str, default=None,
                   help="Comma-separated 0-indexed sample IDs. Overrides --n-samples.")
    p.add_argument("--n-steps", type=int, default=500)
    p.add_argument("--dt", type=float, default=0.003)
    p.add_argument("--use-projection", action="store_true", default=True)
    p.add_argument("--no-projection", dest="use_projection", action="store_false")
    p.add_argument("--force-threshold", type=float, default=0.01)
    p.add_argument("--force-criterion", default="fmax", choices=["fmax", "force_norm"])
    p.add_argument("--use-adaptive-dt", action="store_true", default=False)
    p.add_argument("--dt-min", type=float, default=1e-5)
    p.add_argument("--dt-max", type=float, default=0.1)
    p.add_argument("--max-atom-disp", type=float, default=0.35)
    p.add_argument("--use-preconditioning", action="store_true", default=False)
    p.add_argument("--eig-floor", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--split", default="test")
    p.add_argument("--h5", default="/lustre06/project/6033559/memoozd/data/transition1x.h5")
    p.add_argument("--n-workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    # New hparam knobs
    p.add_argument("--k-track", type=int, default=8)
    p.add_argument("--descent-until-nneg", type=int, default=0,
                   help="Pure descent until n_neg <= this, then lock GAD. 0 = pure GAD from start.")
    p.add_argument("--purify-hessian", action="store_true", default=False)
    p.add_argument("--multimode", default="", choices=["", "all_neg", "smooth", "top2"])
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    run_id = str(uuid.uuid4())[:8]

    print(f"Backend: {args.backend} | Method: {args.method}")
    print(f"Samples: {args.n_samples} | Steps: {args.n_steps} | dt: {args.dt}")
    print(f"Noise: {args.noise} A | Use Eckart: {args.use_projection}")
    print(f"Workers: {args.n_workers}")
    print(f"Output: {args.output_dir}")

    if args.sample_indices:
        indices = [int(x) for x in args.sample_indices.split(",") if x.strip()]
    else:
        indices = list(range(args.n_samples))
    task_args = [
        (
            i, args.h5, args.split, args.backend, args.method, args.noise,
            args.seed + 1000 * i, args.n_steps, args.dt, args.use_projection,
            args.force_threshold, args.force_criterion,
            args.use_adaptive_dt, args.dt_min, args.dt_max, args.max_atom_disp,
            args.use_preconditioning, args.eig_floor,
            args.k_track, args.descent_until_nneg, args.purify_hessian, args.multimode,
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
                results.append({
                    "sample_id": idx, "formula": "", "rxn": "",
                    "start_method": "", "search_method": "",
                    "converged": False, "converged_step": -1,
                    "total_steps": 0, "final_n_neg": -1,
                    "final_force_norm": float("nan"), "final_force_max": float("nan"),
                    "final_energy": float("nan"), "final_eig0": float("nan"),
                    "wall_time_s": float("nan"), "failure_type": f"worker_exception:{type(exc).__name__}",
                })
                continue
            status = "OK" if r["converged"] else r["failure_type"]
            print(f"  [{r['sample_id']}] {r['formula']} | {status} | "
                  f"n_neg={r['final_n_neg']} fmax={r['final_force_max']:.3e} "
                  f"wall={r['wall_time_s']:.1f}s")
            results.append(r)

    total_wall = time.time() - t_overall

    summary_path = os.path.join(args.output_dir, f"summary_{run_id}.parquet")
    pq.write_table(pa.Table.from_pylist(results), summary_path)

    n_total = len(results)
    n_conv = sum(1 for r in results if r["converged"])
    print()
    print("=" * 60)
    print(f"{args.backend}/{args.method} GAD-projected: "
          f"{n_conv}/{n_total} converged ({100 * n_conv / max(n_total, 1):.1f}%) "
          f"| total wall: {total_wall:.1f}s")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
