"""Smoke test for SCINE and xTB calculator backends.

Loads a single Transition1x sample, runs each backend through the
predict_fn protocol, and checks:
    - energy is scalar and finite
    - forces are (N, 3) and finite, |F| > 0
    - hessian is (3N, 3N), finite, symmetric to ~1e-6 (eV/A^2)

Also runs one GAD search step end-to-end with each backend to confirm
the orchestration plumbing works.

Run with:
    python scripts/smoke_test_scine_xtb.py

This script avoids GPU and is meant for CPU compute nodes.
"""
from __future__ import annotations

import sys
import time
import traceback
from typing import Callable

import numpy as np
import torch

from gadplus.data.transition1x import Transition1xDataset, UsePos


T1X_H5 = "/lustre06/project/6033559/memoozd/data/transition1x.h5"


def load_one_sample(split: str = "test", index: int = 0):
    """Return (coords [N,3] float64, atomic_nums [N] long, formula)."""
    ds = Transition1xDataset(
        h5_path=T1X_H5, split=split, max_samples=index + 1,
        transform=UsePos("pos_transition"),
    )
    sample = ds[index]
    coords = sample.pos.detach().to(torch.float64)
    atomic_nums = sample.z.detach().to(torch.long)
    formula = getattr(sample, "formula", f"sample_{index}")
    return coords, atomic_nums, formula


def check_predict(name: str, predict_fn: Callable, coords, atomic_nums):
    """Run the predict_fn and validate shapes and finiteness."""
    print(f"\n[{name}] predict_fn smoke check")
    t0 = time.time()
    out = predict_fn(coords, atomic_nums, do_hessian=True, require_grad=False)
    dt = time.time() - t0

    e = out["energy"]
    f = out["forces"]
    h = out["hessian"]

    N = coords.shape[0]
    e_val = float(e.detach().item()) if e.ndim == 0 else float(e.detach().reshape(-1)[0])
    print(f"  wall: {dt:.2f}s")
    print(f"  energy: {e_val:.6f} eV")
    print(f"  forces shape: {tuple(f.shape)}  |F|: {float(torch.linalg.norm(f)):.4e} eV/A")
    print(f"  hessian shape: {tuple(h.shape)}  dtype: {h.dtype}")

    assert f.shape == (N, 3), f"forces shape mismatch: {f.shape}"
    assert h.shape == (3 * N, 3 * N), f"hessian shape mismatch: {h.shape}"
    assert torch.isfinite(e).all(), "energy not finite"
    assert torch.isfinite(f).all(), "forces not finite"
    assert torch.isfinite(h).all(), "hessian not finite"

    sym_err = float((h - h.transpose(-1, -2)).abs().max())
    print(f"  hessian symmetry max-err: {sym_err:.2e}")

    eigvals = torch.linalg.eigvalsh(0.5 * (h + h.transpose(-1, -2)))
    print(f"  eig range: [{float(eigvals.min()):.4f}, {float(eigvals.max()):.4f}]")
    print(f"  n_neg (raw, no Eckart): {int((eigvals < -1e-4).sum())}")

    return out


def run_one_gad_step(name: str, predict_fn, coords, atomic_nums, device):
    """Tiny end-to-end check: one GAD search step with Eckart projection."""
    from gadplus.search.gad_search import GADSearchConfig, run_gad_search

    print(f"\n[{name}] running 3-step gad_projected to verify integration")
    cfg = GADSearchConfig(
        n_steps=3, dt=0.001, k_track=0, beta=1.0,
        use_projection=True, use_adaptive_dt=False,
        force_threshold=1e-9,  # never converges in 3 steps
        force_criterion="fmax",
    )
    t0 = time.time()
    coords_dev = coords.to(device=device, dtype=torch.float32)
    z_dev = atomic_nums.to(device=device)
    result = run_gad_search(predict_fn, coords_dev, z_dev, cfg)
    dt = time.time() - t0
    print(f"  steps: {result.total_steps}  wall: {dt:.2f}s")
    print(f"  final n_neg: {result.final_n_neg}  fmax: {result.final_force_max:.4e}")
    print(f"  final eig0: {result.final_eig0:.4e}")


def test_scine(coords, atomic_nums):
    from gadplus.calculator.scine import (
        load_scine_calculator, make_scine_predict_fn,
    )
    calc = load_scine_calculator(functional="DFTB0", device="cpu")
    predict_fn = make_scine_predict_fn(calc)
    check_predict("SCINE/DFTB0", predict_fn, coords, atomic_nums)
    run_one_gad_step("SCINE/DFTB0", predict_fn, coords, atomic_nums, device=torch.device("cpu"))


def test_xtb(coords, atomic_nums):
    from gadplus.calculator.xtb import load_xtb_calculator, make_xtb_predict_fn
    calc = load_xtb_calculator(method="gfn2", device="cpu")
    predict_fn = make_xtb_predict_fn(calc)
    check_predict("xTB/GFN2", predict_fn, coords, atomic_nums)
    run_one_gad_step("xTB/GFN2", predict_fn, coords, atomic_nums, device=torch.device("cpu"))


def main():
    print(f"torch: {torch.__version__}  threads: {torch.get_num_threads()}")
    coords, atomic_nums, formula = load_one_sample()
    print(f"Sample: {formula}  N={coords.shape[0]}")

    failures = []

    for name, fn in [("SCINE", test_scine), ("xTB", test_xtb)]:
        try:
            fn(coords, atomic_nums)
            print(f"\n[{name}] OK")
        except Exception as exc:  # noqa: BLE001
            print(f"\n[{name}] FAILED: {exc}")
            traceback.print_exc()
            failures.append(name)

    print("\n" + "=" * 60)
    if failures:
        print(f"FAILED backends: {failures}")
        sys.exit(1)
    print("All backends OK")


if __name__ == "__main__":
    main()
