#!/usr/bin/env python
"""Compare g-xTB and the project's reference surface on labelled T1x TSs.

The comparison uses identical labelled TS coordinates and atom ordering.  It
reports both raw Cartesian Hessians and the project-standard Eckart treatment:
mass-weight -> remove translation/rotation -> diagonalize in the reduced
vibrational basis, plus the equivalent projected Cartesian Hessian.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def _sym(h: torch.Tensor) -> torch.Tensor:
    h = h.to(torch.float64).reshape(int(h.numel() ** 0.5), -1)
    return 0.5 * (h + h.T)


def _relative_norm(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).norm() / b.norm().clamp_min(1.0e-12))


def _projected_cartesian(hessian: torch.Tensor, coords: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    from gadplus.projection.projection import (
        _eckart_projector, atomic_nums_to_symbols, get_mass_weights,
    )

    symbols = atomic_nums_to_symbols(z)
    _masses, _m3, sqrt_m, sqrt_m_inv = get_mass_weights(
        symbols, device=hessian.device, dtype=torch.float64,
    )
    raw = _sym(hessian)
    inv_m = torch.diag(sqrt_m_inv)
    mw = _eckart_projector(coords.to(torch.float64), _masses) @ (inv_m @ raw @ inv_m)
    mw = mw @ _eckart_projector(coords.to(torch.float64), _masses)
    return torch.diag(sqrt_m) @ (0.5 * (mw + mw.T)) @ torch.diag(sqrt_m)


def _vibrational_eigenvalues(hessian: torch.Tensor, coords: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    from gadplus.projection import atomic_nums_to_symbols, vib_eig

    evals, _vectors, _basis = vib_eig(
        _sym(hessian), coords, atomic_nums_to_symbols(z), purify=False,
    )
    return evals.detach().to(torch.float64).cpu()


def _compare_pair(
    sample_id: int, record, gxtb_predict, reference_predict, reference_name: str,
) -> dict:
    coords = torch.as_tensor(record.transition_state, dtype=torch.float64)
    z = torch.as_tensor(record.atomic_nums, dtype=torch.long)
    t0 = time.perf_counter()
    g_out = gxtb_predict(coords, z, do_hessian=True, require_grad=False)
    gxtb_hessian_seconds = time.perf_counter() - t0
    t0 = time.perf_counter()
    reference_out = reference_predict(coords, z, do_hessian=True, require_grad=False)
    reference_hessian_seconds = time.perf_counter() - t0
    hg = _sym(g_out["hessian"])
    hd = _sym(reference_out["hessian"])
    hgp = _projected_cartesian(hg, coords, z)
    hdp = _projected_cartesian(hd, coords, z)
    eg = _vibrational_eigenvalues(hg, coords, z)
    ed = _vibrational_eigenvalues(hd, coords, z)
    n = min(len(eg), len(ed))
    ev_diff = eg[:n] - ed[:n]
    row = {
        "sample_id": int(sample_id),
        "formula": str(record.formula),
        "n_atoms": int(len(z)),
        "gxtb_energy_eV": float(g_out["energy"].reshape(-1)[0]),
        "reference_surface": reference_name,
        "reference_energy_eV": float(reference_out["energy"].reshape(-1)[0]),
        "energy_diff_eV": float(g_out["energy"].reshape(-1)[0] - reference_out["energy"].reshape(-1)[0]),
        "gxtb_fmax_eV_A": float(g_out["forces"].abs().max()),
        "reference_fmax_eV_A": float(reference_out["forces"].abs().max()),
        "gxtb_full_hessian_seconds": gxtb_hessian_seconds,
        "reference_full_hessian_seconds": reference_hessian_seconds,
        "gxtb_estimated_287x2000_steps_hours": gxtb_hessian_seconds * 287 * 2000 / 3600.0,
        "reference_estimated_287x2000_steps_hours": reference_hessian_seconds * 287 * 2000 / 3600.0,
        "raw_hessian_rel_frobenius": _relative_norm(hg, hd),
        "raw_hessian_max_abs_eV_A2": float((hg - hd).abs().max()),
        "projected_hessian_rel_frobenius": _relative_norm(hgp, hdp),
        "projected_hessian_max_abs_eV_A2": float((hgp - hdp).abs().max()),
        "vib_eigenvalue_rmse_eV_A2": float(torch.sqrt(torch.mean(ev_diff**2))),
        "vib_eigenvalue_max_abs_eV_A2": float(ev_diff.abs().max()),
        "gxtb_n_vib_negative": int((eg < -1.0e-4).sum()),
        "reference_n_vib_negative": int((ed < -1.0e-4).sum()),
        "gxtb_vib_eigenvalues_eV_A2": eg.tolist(),
        "reference_vib_eigenvalues_eV_A2": ed.tolist(),
    }
    print(json.dumps(row, sort_keys=True))
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", default=os.environ.get("GADPLUS_T1X_H5", "data/transition1x.h5"))
    parser.add_argument("--split", default="test")
    parser.add_argument("--sample-ids", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--gxtb-executable", default="g-xtb/xtb-6.7.1/bin/xtb")
    parser.add_argument("--reference", choices=("hip", "dxtb"), default="hip")
    parser.add_argument("--checkpoint", default="/lustre06/project/6033559/memoozd/models/hip_v2.ckpt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    from gadplus.calculator.gxtb import load_gxtb_calculator, make_gxtb_predict_fn
    from gadplus.data.direct_t1x import load_t1x_records_direct

    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = load_t1x_records_direct(args.h5, args.split, args.sample_ids)
    gxtb_predict = make_gxtb_predict_fn(load_gxtb_calculator(
        executable=args.gxtb_executable, n_threads=1,
        parallel=int(os.environ.get("GADPLUS_GXTB_PARALLEL", "1")),
    ))
    if args.reference == "hip":
        from gadplus.calculator.hip import load_hip_calculator, make_hip_predict_fn
        reference_predict = make_hip_predict_fn(load_hip_calculator(args.checkpoint, device=args.device))
    else:
        from gadplus.calculator.xtb import load_xtb_calculator, make_xtb_predict_fn
        reference_predict = make_xtb_predict_fn(load_xtb_calculator(method="gfn2", device="cpu"))
    rows = [_compare_pair(i, records[i], gxtb_predict, reference_predict, args.reference) for i in args.sample_ids]
    timing = {
        "gxtb_full_hessian_median_seconds": float(torch.tensor([r["gxtb_full_hessian_seconds"] for r in rows]).median()),
        "reference_full_hessian_median_seconds": float(torch.tensor([r["reference_full_hessian_seconds"] for r in rows]).median()),
        "gxtb_estimated_287x2000_steps_median_hours": float(torch.tensor([r["gxtb_estimated_287x2000_steps_hours"] for r in rows]).median()),
        "reference_estimated_287x2000_steps_median_hours": float(torch.tensor([r["reference_estimated_287x2000_steps_hours"] for r in rows]).median()),
    }
    print(json.dumps({"timing": timing}, sort_keys=True))
    (args.output_dir / "summary.json").write_text(json.dumps({
        "n_samples": len(rows),
        "sample_ids": args.sample_ids,
        "split": args.split,
        "timing": timing,
        "rows": rows,
    }, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
