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

# Make src/ importable without installing.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

_PREDICT_FN_CACHE = {}


def _build_predict_fn(backend: str, method: str):
    """Construct a PredictFn for the chosen backend. Called inside workers."""
    key = (backend, method)
    if key in _PREDICT_FN_CACHE:
        return _PREDICT_FN_CACHE[key]
    if backend == "scine":
        from gadplus.calculator.scine import (
            load_scine_calculator, make_scine_predict_fn,
        )
        calc = load_scine_calculator(functional=method, device="cpu")
        predict_fn = make_scine_predict_fn(calc)
    elif backend == "xtb":
        from gadplus.calculator.xtb import load_xtb_calculator, make_xtb_predict_fn
        calc = load_xtb_calculator(method=method, device="cpu")
        predict_fn = make_xtb_predict_fn(calc)
    elif backend == "mace":
        from gadplus.calculator.mace import load_mace_calculator, make_mace_predict_fn
        calc = load_mace_calculator(model=method, device="cuda")
        predict_fn = make_mace_predict_fn(calc)
    elif backend == "horm":
        from gadplus.calculator.horm import load_horm_leftnet_calculator, make_horm_predict_fn
        calc = load_horm_leftnet_calculator(device="cuda")
        predict_fn = make_horm_predict_fn(calc)
    elif backend == "gxtb":
        from gadplus.calculator.gxtb import load_gxtb_calculator, make_gxtb_predict_fn
        calc = load_gxtb_calculator(**{k: v for k, v in {
            "executable": os.environ.get("GADPLUS_GXTB_EXE", "g-xtb/xtb-6.7.1/bin/xtb"),
            "n_threads": 1,
            "parallel": int(os.environ.get("GADPLUS_GXTB_PARALLEL", "1")),
        }.items() if v is not None})
        predict_fn = make_gxtb_predict_fn(calc)
    else:
        raise ValueError(f"Unknown backend: {backend!r}")
    _PREDICT_FN_CACHE[key] = predict_fn
    return predict_fn


def _run_one_sample(args_tuple):
    """Worker entrypoint. Imports stay inside this fn so each subprocess
    only loads the dependencies it needs.
    """
    (
        sample_idx, h5_path, split, backend, method, noise_ang, start_geometry, seed,
        n_steps, dt, use_projection, force_threshold, force_criterion,
        use_adaptive_dt, dt_min, dt_max, max_atom_disp,
        use_preconditioning, eig_floor,
        descent_until_nneg, blend_sharpness,
        output_dir, run_id,
    ) = args_tuple

    # Single-threaded BLAS per worker — we get parallelism from the pool.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

    import torch
    torch.set_num_threads(1)

    from gadplus.logging.trajectory import TrajectoryLogger
    from gadplus.logging.autopsy import classify_failure
    from gadplus.search.gad_search import GADSearchConfig, run_gad_search

    if backend in {"mace", "horm", "gxtb"}:
        if backend == "gxtb":
            from gadplus.data.direct_t1x import load_t1x_records_direct
            sample = load_t1x_records_direct(h5_path, split, [sample_idx])[sample_idx]
        else:
            from gadplus.data.direct_t1x import load_t1x_record
            sample = load_t1x_record(h5_path, split, sample_idx)
        formula = sample.formula or f"sample_{sample_idx}"
        rxn = sample.rxn
        known_ts_cpu = torch.as_tensor(sample.transition_state, dtype=torch.float32)
        if start_geometry == "labelled_ts":
            coords_cpu = known_ts_cpu.clone()
            start_label = "labelled_ts"
        elif start_geometry == "reactant":
            coords_cpu = torch.as_tensor(sample.reactant, dtype=torch.float32)
            start_label = "labelled_reactant"
        elif start_geometry == "product":
            if sample.product is None:
                raise ValueError(f"sample {sample_idx} has no compatible labelled product geometry")
            coords_cpu = torch.as_tensor(sample.product, dtype=torch.float32)
            start_label = "labelled_product"
        else:
            generator = torch.Generator(device="cpu").manual_seed(seed)
            coords_cpu = known_ts_cpu + noise_ang * torch.randn(
                known_ts_cpu.shape, generator=generator, dtype=known_ts_cpu.dtype,
            )
            start_label = f"noised_ts_noise{noise_ang:.2f}A"
        if backend in {"mace", "horm"}:
            # Keep neural-model state and outputs on CUDA. g-xTB is an
            # external CPU executable and stays on CPU.
            device = torch.device("cuda")
            coords = coords_cpu.to(device)
            known_ts = known_ts_cpu.to(device)
            z = torch.as_tensor(sample.atomic_nums, dtype=torch.long, device=device)
        else:
            coords = coords_cpu
            known_ts = known_ts_cpu
            z = torch.as_tensor(sample.atomic_nums, dtype=torch.long)
    else:
        from gadplus.data.transition1x import Transition1xDataset, UsePos
        from gadplus.geometry.starting import make_starting_coords

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
        known_ts = sample.pos_transition.to(torch.float32)

    predict_fn = _build_predict_fn(backend, method)

    cfg = GADSearchConfig(
        n_steps=n_steps, dt=dt, k_track=8, beta=1.0,
        use_projection=use_projection,
        use_adaptive_dt=use_adaptive_dt, dt_min=dt_min, dt_max=dt_max,
        max_atom_disp=max_atom_disp,
        use_preconditioning=use_preconditioning, eig_floor=eig_floor,
        descent_until_nneg=descent_until_nneg,
        blend_sharpness=blend_sharpness,
        force_threshold=force_threshold, force_criterion=force_criterion,
        purify_hessian=False,
    )

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
        "final_coords_flat": result.final_coords.reshape(-1).tolist(),
        "wall_time_s": float(wall),
        "failure_type": failure_type or "",
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", required=True, choices=["scine", "xtb", "gxtb", "mace", "horm"])
    p.add_argument("--method", default="DFTB0",
                   help="SCINE functional, xTB method, or MACE-OFF23 model size/path")
    p.add_argument("--noise", type=float, default=1.0,
                   help="Per-Cartesian-component Gaussian sigma on the labelled TS, in Angstrom (not per-atom RMS).")
    p.add_argument(
        "--start-geometry", default="noised_ts",
        choices=("noised_ts", "labelled_ts", "reactant", "product"),
        help="Initial T1x geometry; noise is used only for noised_ts.",
    )
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
    p.add_argument(
        "--descent-until-nneg", type=int, default=0,
        help="Follow projected descent until n_neg is at or below this value; 0 starts pure GAD.",
    )
    p.add_argument(
        "--blend-sharpness", type=float, default=0.0,
        help="Use sigmoid(blend_sharpness * lambda_2) for the GAD ascent weight; 0 is pure GAD.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--split", default="test")
    p.add_argument(
        "--h5", default=os.environ.get("GADPLUS_T1X_H5", "data/transition1x.h5"),
        help="Transition1x HDF5 path (or set GADPLUS_T1X_H5).",
    )
    p.add_argument("--n-workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    run_id = str(uuid.uuid4())[:8]

    print(f"Backend: {args.backend} | Method: {args.method}")
    print(f"Start: {args.start_geometry} | Noise: {args.noise} A | Use Eckart: {args.use_projection}")
    print(f"Workers: {args.n_workers}")
    print(f"Output: {args.output_dir}")

    if args.sample_indices:
        indices = [int(x) for x in args.sample_indices.split(",") if x.strip()]
    else:
        indices = list(range(args.n_samples))
    print(
        f"Samples: {len(indices)} | Steps: {args.n_steps} | dt: {args.dt} | "
        f"descent_until_nneg={args.descent_until_nneg} | blend_sharpness={args.blend_sharpness}"
    )
    task_args = [
        (
            i, args.h5, args.split, args.backend, args.method, args.noise, args.start_geometry,
            args.seed + 1000 * i, args.n_steps, args.dt, args.use_projection,
            args.force_threshold, args.force_criterion,
            args.use_adaptive_dt, args.dt_min, args.dt_max, args.max_atom_disp,
            args.use_preconditioning, args.eig_floor,
            args.descent_until_nneg, args.blend_sharpness,
            args.output_dir, run_id,
        )
        for i in indices
    ]

    results = []
    t_overall = time.time()
    # A single local run should not fork a second Python interpreter.  Besides
    # adding no throughput, that makes large shared filesystems a startup
    # bottleneck and obscures the calculator subprocess in the Slurm log.
    if args.n_workers == 1:
        completed = ((ta[0], _run_one_sample, ta) for ta in task_args)
    else:
        exe = ProcessPoolExecutor(max_workers=args.n_workers)
        future_to_idx = {exe.submit(_run_one_sample, ta): ta[0] for ta in task_args}
        completed = ((future_to_idx[fut], fut.result, None) for fut in as_completed(future_to_idx))
    try:
        for idx, run, task in completed:
            try:
                r = run(task) if task is not None else run()
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
    finally:
        if args.n_workers != 1:
            exe.shutdown()

    total_wall = time.time() - t_overall

    summary_path = os.path.join(args.output_dir, f"summary_{run_id}.parquet")
    # pyarrow's large shared objects can take minutes to page in from Lustre;
    # it is only needed after the electronic-structure work has finished.
    import pyarrow as pa
    import pyarrow.parquet as pq
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
