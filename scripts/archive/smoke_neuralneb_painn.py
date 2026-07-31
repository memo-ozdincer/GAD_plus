#!/usr/bin/env python
"""Conservative PaiNN backend smoke before any GAD/Sella benchmark expansion."""
from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def directional_checks(predict_fn, coords: torch.Tensor, z: torch.Tensor, eps: float) -> dict[str, float]:
    out = predict_fn(coords, z, do_hessian=True, require_grad=False)
    direction = torch.randn_like(coords)
    direction = direction / direction.norm()
    plus = predict_fn(coords + eps * direction, z, do_hessian=False, require_grad=False)
    minus = predict_fn(coords - eps * direction, z, do_hessian=False, require_grad=False)

    force_fd = -((plus["energy"] - minus["energy"]) / (2.0 * eps))
    force_dot = (out["forces"] * direction).sum()
    force_jac_fd = -((plus["forces"] - minus["forces"]) / (2.0 * eps)).reshape(-1)
    hessian_direction = out["hessian"] @ direction.reshape(-1)
    hessian = out["hessian"]
    translation = torch.zeros_like(coords)
    translation[:, 0] = 1.0 / math.sqrt(coords.shape[0])

    return {
        "force_fd_abs_error": float((force_fd - force_dot).abs()),
        "hessian_direction_rel_error": float(
            (force_jac_fd - hessian_direction).norm() / force_jac_fd.norm().clamp_min(1.0e-8)
        ),
        "hessian_relative_antisymmetry": float(
            (hessian - hessian.T).norm() / hessian.norm().clamp_min(1.0e-8)
        ),
        "translation_force_abs": float((out["forces"] * translation).sum().abs()),
        "translation_hessian_abs": float((hessian @ translation.reshape(-1)).norm()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--epsilon", type=float, default=1.0e-3)
    args = parser.parse_args()

    from gadplus.calculator.neuralneb import (
        NeuralNebPaiNNCalculator,
        NEURALNEB_MODELS_DIR,
        make_neuralneb_predict_fn,
    )

    checkpoint = Path(args.checkpoint) if args.checkpoint else NEURALNEB_MODELS_DIR / "painn0.sd"
    calculator = NeuralNebPaiNNCalculator(checkpoint=checkpoint, device=args.device)
    predict_fn = make_neuralneb_predict_fn(calculator)

    # Stable, non-collinear H2O geometry avoids a rotational degeneracy in this smoke.
    coords = torch.tensor(
        [[0.0, 0.0, 0.0], [0.9572, 0.0, 0.0], [-0.2390, 0.9270, 0.0]],
        dtype=torch.float32,
        device=args.device,
    )
    z = torch.tensor([8, 1, 1], dtype=torch.long, device=args.device)
    out = predict_fn(coords, z, do_hessian=True, require_grad=False)
    checks = directional_checks(predict_fn, coords, z, args.epsilon)

    print(f"device={args.device}")
    print(f"checkpoint={checkpoint}")
    print(f"energy_eV={float(out['energy']):.8f}")
    print(f"force_shape={tuple(out['forces'].shape)}")
    print(f"hessian_shape={tuple(out['hessian'].shape)}")
    for name, value in checks.items():
        print(f"{name}={value:.3e}")

    assert out["forces"].shape == coords.shape
    assert out["hessian"].shape == (coords.numel(), coords.numel())
    assert checks["force_fd_abs_error"] < 2.0e-4
    assert checks["hessian_direction_rel_error"] < 2.0e-3
    assert checks["hessian_relative_antisymmetry"] < 1.0e-7
    print("PAINN_EFH_SMOKE_PASS")


if __name__ == "__main__":
    main()
