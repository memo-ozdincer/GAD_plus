"""Hydra entry point for GADplus experiments.

Orchestrates: calculator loading → dataset iteration → search loop → logging.

Usage:
    # Local (single run):
    python -m gadplus.orchestration.run

    # SLURM sweep (multiple configs):
    python -m gadplus.orchestration.run --multirun \\
        hydra/launcher=submitit_slurm \\
        search=pure_gad,gad_projected,gad_adaptive_dt \\
        starting.noise_levels_pm=0,5,10,15
"""
from __future__ import annotations

import os
import time
import uuid

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch_geometric.loader import DataLoader

from gadplus.calculator.hip import load_hip_calculator, make_hip_predict_fn
from gadplus.data.transition1x import Transition1xDataset, UsePos
from gadplus.geometry.noise import add_gaussian_noise
from gadplus.geometry.starting import make_starting_coords
from gadplus.logging.trajectory import TrajectoryLogger
from gadplus.logging.autopsy import classify_failure
from gadplus.projection.masses import atomic_nums_to_symbols
from gadplus.search.gad_search import GADSearchConfig, run_gad_search
from gadplus.search.nr_gad_flipflop import NRGADConfig, run_nr_gad_flipflop


def _build_gad_config(cfg: DictConfig) -> GADSearchConfig:
    return GADSearchConfig(
        n_steps=cfg.search.get("n_steps", 300),
        dt=cfg.search.get("dt", 0.005),
        k_track=cfg.search.get("k_track", 0),
        beta=cfg.search.get("beta", 1.0),
        use_projection=cfg.search.get("use_projection", False),
        use_adaptive_dt=cfg.search.get("use_adaptive_dt", False),
        dt_min=cfg.search.get("dt_min", 1e-5),
        dt_max=cfg.search.get("dt_max", 0.1),
        dt_adaptation=cfg.search.get("dt_adaptation", "eigenvalue_clamped"),
        max_atom_disp=cfg.search.get("max_atom_disp", 0.35),
        min_interatomic_dist=cfg.search.get("min_interatomic_dist", 0.4),
        force_threshold=cfg.search.get("force_threshold", 0.01),
        purify_hessian=cfg.search.get("purify_hessian", False),
    )


def _build_nr_gad_config(cfg: DictConfig) -> NRGADConfig:
    return NRGADConfig(
        max_steps=cfg.search.get("max_steps", 500),
        gad_dt=cfg.search.get("gad_dt", 0.005),
        k_track=cfg.search.get("k_track", 8),
        use_projection=cfg.search.get("use_projection", True),
        max_atom_disp=cfg.search.get("max_atom_disp", 0.35),
        min_interatomic_dist=cfg.search.get("min_interatomic_dist", 0.4),
        nr_max_step_component=cfg.search.get("nr_max_step_component", 0.3),
        force_threshold=cfg.search.get("force_threshold", 0.01),
        purify_hessian=cfg.search.get("purify_hessian", False),
    )


@hydra.main(version_base=None, config_path="../../../configs", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    # Seed
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    # Setup MLflow (optional)
    if cfg.logging.get("mlflow_enabled", False):
        from gadplus.logging.mlflow_logger import setup_mlflow, log_run_params
        setup_mlflow(cfg.output_dir, cfg.logging.experiment_name)
        import mlflow
        mlflow.start_run(run_name=f"{cfg.search.method}_{cfg.starting.method}")
        log_run_params({
            "search_method": cfg.search.method,
            "starting_method": cfg.starting.method,
            "max_samples": cfg.max_samples,
            "seed": cfg.seed,
        })

    # Load calculator
    print(f"Loading HIP checkpoint: {cfg.calculator.checkpoint}")
    calculator = load_hip_calculator(
        checkpoint_path=cfg.calculator.checkpoint,
        device=cfg.calculator.device,
        hessian_method=cfg.calculator.hessian_method,
    )
    predict_fn = make_hip_predict_fn(calculator)

    # Load dataset
    print(f"Loading Transition1x: {cfg.calculator.h5_path}")
    dataset = Transition1xDataset(
        h5_path=cfg.calculator.h5_path,
        split=cfg.split,
        max_samples=cfg.max_samples,
        transform=UsePos("pos_transition"),
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    print(f"Loaded {len(dataset)} samples")

    # Determine noise levels
    noise_levels_pm = cfg.starting.get("noise_levels_pm", [0])
    seeds_per_level = cfg.starting.get("seeds_per_level", 1)

    # Build search config
    method = cfg.search.method
    os.makedirs(cfg.output_dir, exist_ok=True)

    results = []
    run_id = str(uuid.uuid4())[:8]

    for sample_idx, batch in enumerate(dataloader):
        formula = batch.formula[0] if hasattr(batch, "formula") else f"sample_{sample_idx}"
        rxn = batch.rxn[0] if hasattr(batch, "rxn") else ""

        for noise_pm in noise_levels_pm:
            noise_angstrom = noise_pm / 100.0  # pm -> Angstrom

            for seed_offset in range(seeds_per_level):
                seed = cfg.seed + sample_idx * 1000 + seed_offset
                start_label = f"{cfg.starting.method}_noise{noise_pm}pm_seed{seed_offset}"

                # Generate starting geometry
                coords = make_starting_coords(
                    batch, cfg.starting.method,
                    noise_rms=noise_angstrom, seed=seed,
                )
                coords = coords.to(cfg.calculator.device)
                atomic_nums = batch.z.to(cfg.calculator.device)

                # Known TS for RMSD tracking
                known_ts = batch.pos_transition.to(cfg.calculator.device) if hasattr(batch, "pos_transition") else None

                # Create logger
                logger = TrajectoryLogger(
                    output_dir=cfg.output_dir,
                    run_id=run_id,
                    sample_id=sample_idx,
                    start_method=start_label,
                    search_method=method,
                    rxn=rxn,
                    formula=formula,
                )

                # Run search
                if method == "nr_gad_flipflop":
                    search_cfg = _build_nr_gad_config(cfg)
                    result = run_nr_gad_flipflop(
                        predict_fn, coords, atomic_nums, search_cfg,
                        logger=logger, known_ts_coords=known_ts,
                    )
                else:
                    search_cfg = _build_gad_config(cfg)
                    result = run_gad_search(
                        predict_fn, coords, atomic_nums, search_cfg,
                        logger=logger, known_ts_coords=known_ts,
                    )

                # Classify failure if not converged
                if not result.converged and logger.rows:
                    failure_type = classify_failure(logger.rows)
                    result.failure_type = failure_type.value

                results.append({
                    "sample_id": sample_idx,
                    "formula": formula,
                    "rxn": rxn,
                    "start_method": start_label,
                    "search_method": method,
                    "converged": result.converged,
                    "converged_step": result.converged_step,
                    "total_steps": result.total_steps,
                    "final_n_neg": result.final_n_neg,
                    "final_force_norm": result.final_force_norm,
                    "final_energy": result.final_energy,
                    "final_eig0": result.final_eig0,
                    "wall_time_s": result.wall_time_s,
                    "failure_type": result.failure_type,
                })

                status = "CONVERGED" if result.converged else f"FAILED ({result.failure_type})"
                print(f"  [{sample_idx}] {formula} | {start_label} | {status} "
                      f"| steps={result.total_steps} | n_neg={result.final_n_neg} "
                      f"| force={result.final_force_norm:.4f} | {result.wall_time_s:.1f}s")

    # Write summary Parquet
    import pyarrow as pa
    import pyarrow.parquet as pq

    summary_path = os.path.join(cfg.output_dir, f"summary_{run_id}.parquet")
    table = pa.Table.from_pylist(results)
    pq.write_table(table, summary_path)
    print(f"\nSummary written to {summary_path}")

    # Log to MLflow
    if cfg.logging.get("mlflow_enabled", False):
        from gadplus.logging.mlflow_logger import log_run_metrics, log_artifact
        n_converged = sum(1 for r in results if r["converged"])
        log_run_metrics({
            "n_samples": len(results),
            "n_converged": n_converged,
            "convergence_rate": n_converged / max(len(results), 1),
        })
        log_artifact(summary_path)
        mlflow.end_run()

    # Print summary
    n_total = len(results)
    n_conv = sum(1 for r in results if r["converged"])
    print(f"\n{'='*60}")
    print(f"SUMMARY: {n_conv}/{n_total} converged ({100*n_conv/max(n_total,1):.1f}%)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
