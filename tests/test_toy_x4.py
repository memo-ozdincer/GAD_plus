from __future__ import annotations

import torch

from gadplus.calculator.toy_x4 import (
    X4EVBParams,
    X4_PAIRS,
    adjacent_ts_guess,
    make_x4_predict_fn,
    minimum_geometry,
    x4_atomic_nums,
)


def test_toy_x4_predict_fn_shapes_and_force_sign():
    predict_fn = make_x4_predict_fn(X4EVBParams(beta=3.05))
    coords = adjacent_ts_guess((0, 1), (0, 2))
    atomic_nums = x4_atomic_nums()

    out = predict_fn(coords, atomic_nums, do_hessian=True)

    assert out["energy"].shape == torch.Size([])
    assert out["forces"].shape == (4, 3)
    assert out["hessian"].shape == (12, 12)
    assert torch.allclose(out["hessian"], out["hessian"].T, atol=1.0e-9)

    x = coords.clone().requires_grad_(True)
    energy = predict_fn.energy(x)
    grad = torch.autograd.grad(energy, x)[0]
    assert torch.allclose(out["forces"], -grad, atol=1.0e-9)


def test_toy_x4_translation_rotation_and_permutation_invariance():
    predict_fn = make_x4_predict_fn()
    coords = minimum_geometry(X4_PAIRS[0])
    atomic_nums = x4_atomic_nums()
    energy = predict_fn(coords, atomic_nums, do_hessian=False)["energy"]

    translated = coords + torch.tensor([2.0, -1.0, 0.5], dtype=torch.float64)
    energy_translated = predict_fn(translated, atomic_nums, do_hessian=False)["energy"]
    assert torch.allclose(energy, energy_translated, atol=1.0e-10)

    theta = torch.tensor(0.7, dtype=torch.float64)
    rotation = torch.tensor(
        [
            [torch.cos(theta), -torch.sin(theta), 0.0],
            [torch.sin(theta), torch.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float64,
    )
    energy_rotated = predict_fn(coords @ rotation.T, atomic_nums, do_hessian=False)["energy"]
    assert torch.allclose(energy, energy_rotated, atol=1.0e-10)

    perm = torch.tensor([2, 0, 3, 1])
    energy_permuted = predict_fn(coords[perm], atomic_nums, do_hessian=False)["energy"]
    assert torch.allclose(energy, energy_permuted, atol=1.0e-10)
