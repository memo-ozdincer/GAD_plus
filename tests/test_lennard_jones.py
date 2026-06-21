from __future__ import annotations

import pytest
import torch

from gadplus.calculator.lennard_jones import (
    LennardJonesParams,
    lj_atomic_nums,
    make_lj_predict_fn,
    params_to_analytical_rm,
    pentagonal_bipyramid_geometry,
    random_cluster_geometry,
)


@pytest.mark.parametrize("backend", ["autograd", "analytical"])
def test_lennard_jones_predict_fn_shapes_and_force_sign(backend: str):
    predict_fn = make_lj_predict_fn(
        LennardJonesParams(epsilon=1.0, sigma=1.0),
        n_atoms=7,
        backend=backend,
    )
    coords = pentagonal_bipyramid_geometry(sigma=1.0)
    atomic_nums = lj_atomic_nums(coords.shape[0])

    out = predict_fn(coords, atomic_nums, do_hessian=True)

    assert out["energy"].shape == torch.Size([])
    assert out["forces"].shape == (7, 3)
    assert out["hessian"].shape == (21, 21)
    assert torch.allclose(out["hessian"], out["hessian"].T, atol=1.0e-9)

    if backend == "autograd":
        x = coords.clone().requires_grad_(True)
        energy = predict_fn.energy(x)
        grad = torch.autograd.grad(energy, x)[0]
        assert torch.allclose(out["forces"], -grad, atol=1.0e-9)


@pytest.mark.parametrize("backend", ["autograd", "analytical"])
def test_lennard_jones_accepts_flat_coords(backend: str):
    predict_fn = make_lj_predict_fn(n_atoms=5, backend=backend)
    coords = random_cluster_geometry(5, sigma=1.0, generator=torch.Generator().manual_seed(0))
    atomic_nums = lj_atomic_nums(5)

    out_2d = predict_fn(coords, atomic_nums, do_hessian=False)
    out_flat = predict_fn(coords.reshape(-1), atomic_nums, do_hessian=False)

    assert torch.allclose(out_2d["energy"], out_flat["energy"], atol=1.0e-12)
    assert torch.allclose(out_2d["forces"], out_flat["forces"], atol=1.0e-12)


@pytest.mark.parametrize("backend", ["autograd", "analytical"])
def test_lennard_jones_translation_invariance(backend: str):
    predict_fn = make_lj_predict_fn(n_atoms=7, backend=backend)
    coords = pentagonal_bipyramid_geometry()
    atomic_nums = lj_atomic_nums(coords.shape[0])
    energy = predict_fn(coords, atomic_nums, do_hessian=False)["energy"]

    translated = coords + torch.tensor([1.5, -2.0, 0.25], dtype=torch.float64)
    energy_translated = predict_fn(translated, atomic_nums, do_hessian=False)["energy"]
    assert torch.allclose(energy, energy_translated, atol=1.0e-10)


def test_lennard_jones_analytical_matches_autograd():
    params = LennardJonesParams(epsilon=1.0, sigma=1.0)
    coords = random_cluster_geometry(7, sigma=1.0, generator=torch.Generator().manual_seed(1))
    atomic_nums = lj_atomic_nums(7)

    autograd_fn = make_lj_predict_fn(params, n_atoms=7, backend="autograd")
    analytical_fn = make_lj_predict_fn(params, n_atoms=7, backend="analytical")

    out_auto = autograd_fn(coords, atomic_nums, do_hessian=True)
    out_anal = analytical_fn(coords, atomic_nums, do_hessian=True)

    assert torch.allclose(out_auto["energy"], out_anal["energy"], atol=1.0e-9, rtol=1.0e-9)
    assert torch.allclose(out_auto["forces"], out_anal["forces"], atol=1.0e-8, rtol=1.0e-8)
    assert torch.allclose(out_auto["hessian"], out_anal["hessian"], atol=1.0e-6, rtol=1.0e-6)


def test_params_to_analytical_rm_uses_equilibrium_distance():
    assert params_to_analytical_rm(1.0) == pytest.approx(2.0 ** (1.0 / 6.0))
