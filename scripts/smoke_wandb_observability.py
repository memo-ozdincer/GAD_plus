#!/usr/bin/env python3
"""Create and export a small analytic-LJ observability smoke run."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

from gadplus.calculator.lennard_jones import (
    lj_atomic_nums,
    make_lj_predict_fn,
    pentagonal_bipyramid_geometry,
)
from gadplus.logging.pointwise import IntrinsicTrajectoryRecorder
from gadplus.logging.wandb_export import export_bundle
from gadplus.projection import atomic_nums_to_symbols, get_mass_weights, vib_eig
from gadplus.search.intrinsic_gad import IntrinsicGADConfig, run_intrinsic_gad


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        default="/scratch/memoozd/gadplus/wandb-smoke",
    )
    parser.add_argument("--project", default="gadplus-ts-mechanisms")
    parser.add_argument("--entity")
    parser.add_argument("--mode", choices=("offline", "online"), default="offline")
    parser.add_argument("--cockpit-chart-id")
    parser.add_argument("--mechanism-chart-id")
    args = parser.parse_args()

    predictor = make_lj_predict_fn()
    atomic_numbers = lj_atomic_nums(7)
    minimum = pentagonal_bipyramid_geometry()
    symbols = atomic_nums_to_symbols(atomic_numbers)
    minimum_out = predictor(minimum, atomic_numbers, do_hessian=True)
    _, modes_mw, _ = vib_eig(minimum_out["hessian"], minimum, symbols)
    _, _, _, inv_sqrt_mass = get_mass_weights(symbols)
    start = minimum + 0.26 * (inv_sqrt_mass * modes_mw[:, 0]).reshape_as(minimum)
    config = IntrinsicGADConfig(
        max_steps=30,
        spectral_temperature=0.01,
        step_fraction=0.05,
        gate_variant="competitive",
        record_history=False,
    )
    run_name = "lj7-competitive-observability-smoke-v1"
    recorder = IntrinsicTrajectoryRecorder(
        Path(args.output_root),
        run_name,
        atomic_numbers,
        config={
            **asdict(config),
            "surface": "analytic-lj",
            "sample_id": 7,
            "noise_angstrom": 0.0,
            "seed": 0,
            "selection_stage": "instrumentation-smoke",
        },
    )
    result = run_intrinsic_gad(
        predictor,
        start,
        atomic_numbers,
        config,
        observer=recorder,
    )
    bundle = recorder.flush(
        result,
        summary={
            "calculator_valid": True,
            "local_ts": result.final_n_neg == 1,
            "strict_ts": result.converged,
            "native_topology": None,
            "instrumentation_level": "full-competitive",
        },
    )
    run_id = export_bundle(
        bundle,
        project=args.project,
        entity=args.entity,
        group="observability-smoke-v1",
        job_type="competitive-gad",
        tags=("instrumentation-smoke", "analytic-lj", "competitive-gad"),
        mode=args.mode,
        cockpit_chart_id=args.cockpit_chart_id,
        mechanism_chart_id=args.mechanism_chart_id,
    )
    print(f"bundle={bundle}")
    print(f"run_id={run_id}")


if __name__ == "__main__":
    main()
