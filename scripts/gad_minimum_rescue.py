#!/usr/bin/env python3
"""Restart a collapsed projected-GAD trajectory by a soft-mode nudge.

The starting point is the final geometry from a prior GAD trajectory with
``n_neg == 0``.  It is displaced by ``sign * amplitude`` along the softest
Eckart-projected vibrational mode, mapped back to Cartesian coordinates, then
run with standard projected GAD.  Both signs must be tested; neither uses a
labelled-TS direction or a non-GAD optimizer.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time
import uuid

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--source-dir", required=True, help="Directory containing one traj_*.parquet.")
    p.add_argument("--sample-index", type=int, required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--amplitude", type=float, default=0.05, help="Cartesian RMS displacement in A.")
    p.add_argument("--sign", type=int, choices=(-1, 1), required=True)
    p.add_argument("--n-steps", type=int, default=500)
    p.add_argument("--dt", type=float, default=0.003)
    p.add_argument("--force-threshold", type=float, default=0.01)
    p.add_argument("--h5", default=os.environ.get("GADPLUS_T1X_H5"))
    p.add_argument("--split", default="test")
    return p


def main() -> None:
    args = parser().parse_args()
    if not args.h5:
        raise SystemExit("Set --h5 or GADPLUS_T1X_H5")
    paths = glob.glob(os.path.join(args.source_dir, "traj_*.parquet"))
    if len(paths) != 1:
        raise SystemExit(f"Expected exactly one source trajectory in {args.source_dir}, found {len(paths)}")
    prior = pd.read_parquet(paths[0]).sort_values("step").iloc[-1]
    if int(prior.n_neg) != 0:
        raise SystemExit(f"Source endpoint is not minimum-like: n_neg={prior.n_neg}")

    from gadplus.calculator.gxtb import load_gxtb_calculator, make_gxtb_predict_fn
    from gadplus.data.direct_t1x import load_t1x_records_direct
    from gadplus.logging.autopsy import classify_failure
    from gadplus.logging.trajectory import TrajectoryLogger
    from gadplus.projection import atomic_nums_to_symbols, get_mass_weights, vib_eig
    from gadplus.search.gad_search import GADSearchConfig, run_gad_search

    sample = load_t1x_records_direct(args.h5, args.split, [args.sample_index])[args.sample_index]
    z = torch.as_tensor(sample.atomic_nums, dtype=torch.long)
    known_ts = torch.as_tensor(sample.transition_state, dtype=torch.float32)
    endpoint = torch.as_tensor(np.asarray(prior.coords_flat), dtype=torch.float32).reshape(-1, 3)
    if endpoint.shape != known_ts.shape:
        raise SystemExit(f"Endpoint shape {endpoint.shape} does not match labelled TS {known_ts.shape}")

    predict = make_gxtb_predict_fn(load_gxtb_calculator(
        executable=os.environ["GADPLUS_GXTB_EXE"], n_threads=1,
        parallel=int(os.environ.get("GADPLUS_GXTB_PARALLEL", "1")),
    ))
    first = predict(endpoint, z, do_hessian=True, require_grad=False)
    symbols = atomic_nums_to_symbols(z)
    evals, evecs_mw, _ = vib_eig(first["hessian"], endpoint, symbols)
    n_neg_endpoint = int((evals < -1e-4).sum().item())
    if n_neg_endpoint != 0:
        raise SystemExit(f"Re-evaluated endpoint n_neg={n_neg_endpoint}; refusing a minimum-rescue label")

    # vib_eig eigenvectors are mass-weighted.  Remove the mass weighting to
    # obtain a Cartesian displacement, then normalize to the requested RMS.
    _masses, _m3, _sqrt_m, sqrt_m_inv = get_mass_weights(symbols, dtype=torch.float64)
    direction = (sqrt_m_inv * evecs_mw[:, 0]).reshape_as(endpoint).to(torch.float32)
    rms = torch.sqrt(direction.square().sum(dim=1).mean())
    coords0 = endpoint + args.sign * args.amplitude * direction / rms

    cfg = GADSearchConfig(
        n_steps=args.n_steps, dt=args.dt, k_track=8, beta=1.0,
        use_projection=True, use_adaptive_dt=False, max_atom_disp=0.35,
        force_threshold=args.force_threshold, force_criterion="fmax",
        descent_until_nneg=0, blend_sharpness=0.0, purify_hessian=False,
    )
    os.makedirs(args.output_dir, exist_ok=True)
    run_id = str(uuid.uuid4())[:8]
    logger = TrajectoryLogger(
        output_dir=args.output_dir, run_id=run_id, sample_id=args.sample_index,
        start_method=f"minimum_endpoint_softmode_{args.sign:+d}_{args.amplitude:.3f}A",
        search_method="gad_projected_gxtb_gxTB_minimum_rescue",
        rxn=sample.rxn, formula=sample.formula,
    )
    t0 = time.time()
    result = run_gad_search(predict, coords0, z, cfg, logger=logger, known_ts_coords=known_ts)
    traj_path = logger.flush()
    failure = None if result.converged else classify_failure(logger.rows).value
    row = pd.DataFrame([{
        "sample_id": args.sample_index, "formula": sample.formula, "rxn": sample.rxn,
        "source_trajectory": paths[0], "source_final_fmax": float(prior.force_max),
        "source_lambda2": float(prior.eig1), "nudge_sign": args.sign,
        "nudge_rms_ang": args.amplitude, "endpoint_n_neg": n_neg_endpoint,
        "endpoint_lambda1": float(evals[0]), "converged": result.converged,
        "converged_step": result.converged_step, "final_n_neg": result.final_n_neg,
        "final_fmax": result.final_force_max, "failure_type": failure,
        "wall_time_s": time.time() - t0, "trajectory": traj_path,
    }])
    summary = os.path.join(args.output_dir, f"summary_{run_id}.parquet")
    row.to_parquet(summary, index=False)
    print(row.to_string(index=False))
    print(f"Summary: {summary}")


if __name__ == "__main__":
    main()
