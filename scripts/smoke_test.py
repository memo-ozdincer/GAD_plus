#!/usr/bin/env python
"""Smoke test: 5 samples, pure GAD, verify everything works end-to-end.

Checks:
  1. HIP loads and produces energy/forces/hessian
  2. Eckart projection gives correct vibrational mode count
  3. GAD step produces valid displacement
  4. TrajectoryLogger writes Parquet
  5. Convergence check logic
  6. Timing per step (for ETA estimation)

Usage (on a GPU node):
  srun python scripts/smoke_test.py
  # or: sbatch scripts/run_smoke_test.slurm
"""
from __future__ import annotations

import os
import sys
import time

import torch

# ---- Add src to path if not installed ----
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def main():
    print("=" * 60)
    print("GAD_plus Smoke Test")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ---- 1. Import check ----
    print("\n--- Import check ---")
    t0 = time.time()
    from gadplus.core.types import PredictFn
    from gadplus.core.gad import compute_gad_vector_tracked, prepare_hessian
    from gadplus.core.mode_tracking import pick_tracked_mode
    from gadplus.core.convergence import is_ts_converged, force_mean, compute_cascade_n_neg
    from gadplus.core.adaptive_dt import compute_adaptive_dt, cap_displacement
    from gadplus.projection import vib_eig, gad_dynamics_projected, atomic_nums_to_symbols
    from gadplus.calculator.hip import load_hip_calculator, make_hip_predict_fn
    from gadplus.data.transition1x import Transition1xDataset, UsePos
    from gadplus.logging.trajectory import TrajectoryLogger
    from gadplus.logging.autopsy import classify_failure, FailureType
    from gadplus.search.gad_search import GADSearchConfig, run_gad_search
    print(f"  All imports OK ({time.time() - t0:.1f}s)")

    # ---- 2. Load HIP ----
    print("\n--- Loading HIP ---")
    # Find checkpoint
    for ckpt_path in [
        "/lustre06/project/6033559/memoozd/models/hip_v2.ckpt",
        "/project/rrg-aspuru/memoozd/models/hip_v2.ckpt",
    ]:
        if os.path.exists(ckpt_path):
            break
    else:
        print("  ERROR: hip_v2.ckpt not found")
        sys.exit(1)

    t0 = time.time()
    calculator = load_hip_calculator(ckpt_path, device=device)
    predict_fn = make_hip_predict_fn(calculator)
    hip_load_time = time.time() - t0
    print(f"  HIP loaded from {ckpt_path} ({hip_load_time:.1f}s)")

    # ---- 3. Load dataset ----
    print("\n--- Loading dataset ---")
    for h5_path in [
        "/lustre06/project/6033559/memoozd/data/transition1x.h5",
        "/project/rrg-aspuru/memoozd/data/transition1x.h5",
    ]:
        if os.path.exists(h5_path):
            break
    else:
        print("  ERROR: transition1x.h5 not found")
        sys.exit(1)

    t0 = time.time()
    dataset = Transition1xDataset(h5_path, split="test", max_samples=5, transform=UsePos("pos_transition"))
    ds_load_time = time.time() - t0
    print(f"  Loaded {len(dataset)} samples ({ds_load_time:.1f}s)")

    # ---- 4. Single predict_fn call ----
    print("\n--- Single HIP evaluation ---")
    sample = dataset[0]
    coords = sample.pos.to(device)
    atomic_nums = sample.z.to(device)
    n_atoms = coords.shape[0]
    formula = sample.formula if hasattr(sample, "formula") else "unknown"
    print(f"  Sample: {formula}, {n_atoms} atoms")

    t0 = time.time()
    out = predict_fn(coords, atomic_nums, do_hessian=True, require_grad=False)
    single_eval_time = time.time() - t0

    energy = float(out["energy"].detach().reshape(-1)[0].item()) if isinstance(out["energy"], torch.Tensor) else float(out["energy"])
    forces = out["forces"]
    if forces.dim() == 3:
        forces = forces[0]
    hessian = out["hessian"]

    print(f"  Energy: {energy:.6f} eV")
    print(f"  Force shape: {forces.shape}")
    print(f"  Hessian shape: {hessian.shape}")
    print(f"  Force norm: {force_mean(forces):.6f} eV/A")
    print(f"  Time: {single_eval_time:.3f}s")

    # ---- 5. Eckart projection ----
    print("\n--- Eckart projection ---")
    atomsymbols = atomic_nums_to_symbols(atomic_nums)
    t0 = time.time()
    evals, evecs, Q_vib = vib_eig(hessian, coords, atomsymbols)
    proj_time = time.time() - t0
    n_neg = int((evals < 0).sum().item())
    n_vib = evals.shape[0]
    expected_vib = 3 * n_atoms - 6  # nonlinear
    print(f"  Vibrational modes: {n_vib} (expected {expected_vib})")
    print(f"  n_neg: {n_neg}")
    print(f"  eig0: {evals[0].item():.6f}")
    print(f"  eig1: {evals[1].item():.6f}")
    print(f"  Time: {proj_time:.4f}s")
    assert n_vib == expected_vib or n_vib == expected_vib + 1, f"Unexpected vib mode count: {n_vib}"

    # ---- 6. GAD step ----
    print("\n--- GAD step ---")
    t0 = time.time()
    gad_vec, v_next, info = compute_gad_vector_tracked(forces, hessian, None, k_track=8)
    gad_time = time.time() - t0
    print(f"  GAD vec norm: {gad_vec.norm().item():.6f}")
    print(f"  Mode overlap: {info['mode_overlap']:.4f}")
    print(f"  Time: {gad_time:.4f}s")

    # ---- 7. Projected GAD step ----
    print("\n--- Projected GAD step ---")
    v_init = evecs[:, 0].to(forces.dtype)
    t0 = time.time()
    gad_proj, v_proj, proj_info = gad_dynamics_projected(coords, forces, v_init, atomsymbols)
    proj_gad_time = time.time() - t0
    print(f"  Projected GAD vec norm: {gad_proj.norm().item():.6f}")
    print(f"  Time: {proj_gad_time:.4f}s")

    # ---- 8. Run 5 samples through GAD search (20 steps each) ----
    print("\n--- GAD search: 5 samples × 20 steps ---")
    out_dir = "/tmp/gadplus_smoke_test"
    os.makedirs(out_dir, exist_ok=True)

    cfg = GADSearchConfig(
        n_steps=20,
        dt=0.005,
        k_track=8,
        use_projection=True,
        use_adaptive_dt=False,
        force_threshold=0.01,
    )

    step_times = []
    results = []
    for i in range(len(dataset)):
        sample = dataset[i]
        coords_i = sample.pos.to(device)
        z_i = sample.z.to(device)
        formula_i = sample.formula if hasattr(sample, "formula") else f"sample_{i}"

        # Add small noise to start away from TS
        noise = torch.randn_like(coords_i) * 0.05  # 0.5 Angstrom
        coords_noised = coords_i + noise

        logger = TrajectoryLogger(
            output_dir=out_dir, run_id="smoke", sample_id=i,
            start_method="noised_ts_50pm", search_method="gad_projected",
            formula=formula_i,
        )

        t0 = time.time()
        result = run_gad_search(predict_fn, coords_noised, z_i, cfg, logger=logger)
        run_time = time.time() - t0
        step_times.append(run_time / result.total_steps)

        status = "CONV" if result.converged else "FAIL"
        print(f"  [{i}] {formula_i:>12s} | {status} | steps={result.total_steps:3d} | "
              f"n_neg={result.final_n_neg} | force={result.final_force_norm:.4f} | "
              f"{run_time:.1f}s ({run_time/result.total_steps:.2f}s/step)")
        results.append(result)

    # ---- 9. Check Parquet output ----
    print("\n--- Parquet output ---")
    import glob
    parquets = glob.glob(os.path.join(out_dir, "*.parquet"))
    print(f"  Files written: {len(parquets)}")
    if parquets:
        import pyarrow.parquet as pq
        table = pq.read_table(parquets[0])
        print(f"  First file: {parquets[0]}")
        print(f"  Rows: {table.num_rows}, Columns: {table.num_columns}")
        print(f"  Columns: {table.column_names[:10]}...")

    # ---- 10. ETA estimation ----
    print("\n" + "=" * 60)
    print("TIMING SUMMARY")
    print("=" * 60)
    avg_step = sum(step_times) / len(step_times)
    print(f"  HIP load:           {hip_load_time:.1f}s (one-time)")
    print(f"  Dataset load (5):   {ds_load_time:.1f}s")
    print(f"  Single HIP eval:    {single_eval_time:.3f}s")
    print(f"  Avg time/step:      {avg_step:.3f}s")
    print(f"  Avg time/sample:    {avg_step * 20:.1f}s (at 20 steps)")
    print()

    # ETAs for starter #2
    steps_300 = avg_step * 300
    print(f"  === ETAs for Starter #2 ===")
    print(f"  1 sample × 300 steps:     {steps_300:.0f}s ({steps_300/60:.1f}min)")
    print(f"  50 samples × 300 steps:   {steps_300 * 50:.0f}s ({steps_300 * 50/3600:.1f}hr)")
    print(f"  300 samples × 300 steps:  {steps_300 * 300:.0f}s ({steps_300 * 300/3600:.1f}hr)")
    print(f"  9561 samples × 300 steps: {steps_300 * 9561:.0f}s ({steps_300 * 9561/3600:.1f}hr)")
    print()
    print(f"  With 500 parallel MIG jobs: {steps_300 * 9561 / 500 / 3600:.1f}hr wall time")
    print()

    n_conv = sum(1 for r in results if r.converged)
    print(f"  Convergence: {n_conv}/{len(results)} ({100*n_conv/len(results):.0f}%)")
    print(f"  (at 0.5A noise, 20 steps — this is expected to be low)")

    # Cleanup
    for f in parquets:
        os.remove(f)
    print("\nSmoke test complete.")


if __name__ == "__main__":
    main()
