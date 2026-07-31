#!/usr/bin/env python3
"""Local minimum-shell branching sampler around failed intrinsic-GAD starts.

This is deliberately an outer sampling policy, not a modification of the
strict deterministic intrinsic-GAD map.  Each branch is constructed solely
from the current minimum geometry and its g-xTB Hessian, then optimized by the
unchanged pointwise intrinsic method.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-summary", type=Path, required=True)
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--task-id", type=int, required=True)
    p.add_argument("--h5", default=os.environ["GADPLUS_T1X_H5"])
    p.add_argument("--gxtb-executable", required=True)
    p.add_argument("--parallel", type=int, default=32)
    p.add_argument("--kappas", default="0.06,0.17,0.30")
    p.add_argument("--seed", type=int, default=20260728)
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument(
        "--gate-variant", choices=("alignment", "competitive", "gad"), required=True
    )
    p.add_argument("--direction", choices=("density", "softest"), default="density")
    return p.parse_args()


def main() -> None:
    a = args()
    from gadplus.calculator.gxtb import load_gxtb_calculator, make_gxtb_predict_fn
    from gadplus.data.direct_t1x import load_t1x_records_direct
    from gadplus.projection import atomic_nums_to_symbols, get_mass_weights, vib_eig
    from gadplus.search.intrinsic_gad import (IntrinsicGADConfig,
                                               inverse_rms_pair_length,
                                               run_intrinsic_gad,
                                               smooth_spectral_policy)
    from gadplus.search.irc_validate import score_endpoints
    from gadplus.search.native_endpoints import relax_to_minimum

    torch.set_num_threads(1)
    source = json.loads(a.source_summary.read_text())
    failures = [r for r in source["rows"] if not r["native_endpoint_topology"] and r.get("final_n_neg") == 0]
    kappas = [float(x) for x in a.kappas.split(",")]
    tasks = [(r, kappa, sign) for r in failures for kappa in kappas for sign in (-1.0, 1.0)]
    if not 0 <= a.task_id < len(tasks):
        raise ValueError(f"task-id must be in [0, {len(tasks)})")
    prior, kappa, sign = tasks[a.task_id]
    records = load_t1x_records_direct(a.h5, "test", sorted({r["sample_id"] for r in failures}))
    record = records[prior["sample_id"]]
    z = torch.as_tensor(record.atomic_nums, dtype=torch.long)
    qmin = torch.as_tensor(prior["final_coords"], dtype=torch.float64)
    calculator = load_gxtb_calculator(executable=a.gxtb_executable, n_threads=1, parallel=a.parallel)
    predict = make_gxtb_predict_fn(calculator)
    out = predict(qmin, z, do_hessian=True, require_grad=False)
    symbols = atomic_nums_to_symbols(z)
    evals, modes, _ = vib_eig(out["hessian"], qmin, symbols)
    _, _, _, inv_sqrt_mass = get_mass_weights(symbols)

    # Basis-invariant soft-low-mode sampler.  The paired signs are antithetic;
    # their shared Gaussian direction is a reproducible sampling input, not
    # trajectory state.  At tau=0.01 with a separated soft mode this recovers
    # the familiar +/- soft-mode kick.
    policy = smooth_spectral_policy(evals, 0.01)
    if a.direction == "softest":
        y = torch.zeros_like(evals)
        y[0] = 1.0
    else:
        gen = torch.Generator().manual_seed(a.seed + 1000 * int(prior["sample_id"]) + 10 * int(round(100 * prior["noise_angstrom"])) + int(round(100 * kappa)))
        xi = torch.randn(len(evals), generator=gen, dtype=torch.float64)
        y = torch.sqrt(policy.low_mode_weights) * xi
    y /= torch.linalg.vector_norm(y).clamp_min(1.0e-12)
    direction = (inv_sqrt_mass * (modes @ y)).reshape_as(qmin)
    direction /= torch.linalg.vector_norm(direction).clamp_min(1.0e-12)
    displacement = kappa * float(inverse_rms_pair_length(qmin).item())
    start = qmin + sign * displacement * direction
    result = run_intrinsic_gad(predict, start, z, IntrinsicGADConfig(
        max_steps=a.max_steps, force_threshold=0.03, force_criterion="fmax",
        step_fraction=0.01, spectral_temperature=0.01, record_history=False,
        gate_variant=a.gate_variant,
    ))
    final = result.final_coords.to(torch.float64)
    final_out = predict(final, z, do_hessian=True, require_grad=False)
    final_evals, final_modes, _ = vib_eig(final_out["hessian"], final, symbols)
    fmax = float(final_out["forces"].abs().amax().item())
    nneg = int((final_evals < -1e-4).sum().item())
    row = {"source_sample": prior["sample_id"], "source_start": prior["start"],
           "source_noise_angstrom": prior["noise_angstrom"], "kappa": kappa, "sign": sign,
           "gate_variant": a.gate_variant,
           "direction": a.direction,
           "shell_displacement_angstrom": displacement, "final_n_neg": nneg,
           "final_fmax": fmax, "steps": result.total_steps, "search_gate": nneg == 1 and fmax < 0.03,
           "native_endpoint_topology": False, "error": ""}
    if row["search_gate"]:
        _, _, _, inv = get_mass_weights(symbols)
        d = (inv * final_modes[:, 0]).reshape_as(final)
        d /= torch.linalg.vector_norm(d).clamp_min(1e-12)
        ends = [relax_to_minimum(final + s * 0.05 * d, z, predict, fmax=0.001, max_steps=500) for s in (-1., 1.)]
        reactant = json.loads((a.source_root / "native_labels" / f"sample_{prior['sample_id']}_reactant.json").read_text())
        product = json.loads((a.source_root / "native_labels" / f"sample_{prior['sample_id']}_product.json").read_text())
        score = score_endpoints(np.asarray(ends[0].coords), np.asarray(ends[1].coords), z,
                                torch.as_tensor(reactant["coords"]), torch.as_tensor(product["coords"]),
                                rmsd_threshold=0.3, predict_fn=predict)
        row["endpoint_minima"] = bool(all(e.converged and e.force_max < .001 for e in ends))
        row["native_endpoint_topology"] = bool(score.topology_intended)
    a.output_root.mkdir(parents=True, exist_ok=True)
    (a.output_root / f"task_{a.task_id:03d}.json").write_text(json.dumps(row, indent=2) + "\n")


if __name__ == "__main__":
    main()
