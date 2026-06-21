from __future__ import annotations

import torch

from gadplus.calculator.lennard_jones_old import (
    LennardJonesParams,
    lj_atomic_nums,
    make_lj_predict_fn,
    pentagonal_bipyramid_geometry,
    random_cluster_geometry,
)


def test_lennard_jones_old_predict_fn_shapes_and_force_sign():
    predict_fn = make_lj_predict_fn(LennardJonesParams(epsilon=1.0, sigma=1.0))
    coords = pentagonal_bipyramid_geometry(sigma=1.0)
    atomic_nums = lj_atomic_nums(coords.shape[0])

    out = predict_fn(coords, atomic_nums, do_hessian=True)

    assert out["energy"].shape == torch.Size([])
    assert out["forces"].shape == (7, 3)
    assert out["hessian"].shape == (21, 21)
    assert torch.allclose(out["hessian"], out["hessian"].T, atol=1.0e-9)

    x = coords.clone().requires_grad_(True)
    energy = predict_fn.energy(x)
    grad = torch.autograd.grad(energy, x)[0]
    assert torch.allclose(out["forces"], -grad, atol=1.0e-9)


def test_lennard_jones_old_accepts_flat_coords():
    predict_fn = make_lj_predict_fn()
    coords = random_cluster_geometry(5, sigma=1.0, generator=torch.Generator().manual_seed(0))
    atomic_nums = lj_atomic_nums(5)

    out_2d = predict_fn(coords, atomic_nums, do_hessian=False)
    out_flat = predict_fn(coords.reshape(-1), atomic_nums, do_hessian=False)

    assert torch.allclose(out_2d["energy"], out_flat["energy"], atol=1.0e-12)
    assert torch.allclose(out_2d["forces"], out_flat["forces"], atol=1.0e-12)


def test_lennard_jones_old_translation_invariance():
    predict_fn = make_lj_predict_fn()
    coords = pentagonal_bipyramid_geometry()
    atomic_nums = lj_atomic_nums(coords.shape[0])
    energy = predict_fn(coords, atomic_nums, do_hessian=False)["energy"]

    translated = coords + torch.tensor([1.5, -2.0, 0.25], dtype=torch.float64)
    energy_translated = predict_fn(translated, atomic_nums, do_hessian=False)["energy"]
    assert torch.allclose(energy, energy_translated, atol=1.0e-10)
