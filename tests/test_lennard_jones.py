from __future__ import annotations

import pytest
import torch

from gadplus.calculator.lennard_jones import (
    LennardJonesParams,
    is_lj_dissociated,
    lj_atomic_nums,
    lj_dissociation_distance_threshold,
    make_lj_predict_fn,
    pair_distances,
    params_to_analytical_rm,
    pentagonal_bipyramid_geometry,
    random_cluster_geometry,
)

def test_lennard_jones_predict_fn_shapes_and_force_sign():
    predict_fn = make_lj_predict_fn(
        LennardJonesParams(epsilon=1.0, sigma=1.0),
        n_atoms=7,
    )
    coords = pentagonal_bipyramid_geometry(sigma=1.0)
    atomic_nums = lj_atomic_nums(coords.shape[0])

    out = predict_fn(coords, atomic_nums, do_hessian=True)

    assert out["energy"].shape == torch.Size([])
    assert out["forces"].shape == (7, 3)
    assert out["hessian"].shape == (21, 21)
    assert torch.allclose(out["hessian"], out["hessian"].T, atol=1.0e-9)


def test_lennard_jones_accepts_flat_coords():
    predict_fn = make_lj_predict_fn(n_atoms=5)
    coords = random_cluster_geometry(5, sigma=1.0, generator=torch.Generator().manual_seed(0))
    atomic_nums = lj_atomic_nums(5)

    out_2d = predict_fn(coords, atomic_nums, do_hessian=False)
    out_flat = predict_fn(coords.reshape(-1), atomic_nums, do_hessian=False)

    assert torch.allclose(out_2d["energy"], out_flat["energy"], atol=1.0e-12)
    assert torch.allclose(out_2d["forces"], out_flat["forces"], atol=1.0e-12)


def test_lennard_jones_rejects_require_grad_true():
    predict_fn = make_lj_predict_fn(n_atoms=5)
    coords = random_cluster_geometry(5, sigma=1.0, generator=torch.Generator().manual_seed(0))
    atomic_nums = lj_atomic_nums(5)

    with pytest.raises(NotImplementedError, match="require_grad=True"):
        predict_fn(coords, atomic_nums, require_grad=True)


def test_lennard_jones_accepts_batched_coords():
    predict_fn = make_lj_predict_fn(n_atoms=5)
    generator = torch.Generator().manual_seed(0)
    coords = torch.stack(
        [
            random_cluster_geometry(5, sigma=1.0, generator=generator),
            random_cluster_geometry(5, sigma=1.0, generator=generator),
            random_cluster_geometry(5, sigma=1.0, generator=generator),
        ]
    )
    atomic_nums = lj_atomic_nums(5)

    out = predict_fn(coords, atomic_nums, do_hessian=True)
    singles = [predict_fn(coords_i, atomic_nums, do_hessian=True) for coords_i in coords]

    assert out["energy"].shape == (3,)
    assert out["forces"].shape == (3, 5, 3)
    assert out["hessian"].shape == (3, 15, 15)
    assert torch.allclose(out["energy"], torch.stack([single["energy"] for single in singles]))
    assert torch.allclose(out["forces"], torch.stack([single["forces"] for single in singles]))
    assert torch.allclose(out["hessian"], torch.stack([single["hessian"] for single in singles]))


def test_lennard_jones_accepts_batched_flat_coords():
    predict_fn = make_lj_predict_fn(n_atoms=4)
    generator = torch.Generator().manual_seed(1)
    coords = torch.stack(
        [
            random_cluster_geometry(4, sigma=1.0, generator=generator),
            random_cluster_geometry(4, sigma=1.0, generator=generator),
        ]
    )
    atomic_nums = lj_atomic_nums(4)

    out_3d = predict_fn(coords, atomic_nums, do_hessian=False)
    out_flat = predict_fn(coords.reshape(coords.shape[0], -1), atomic_nums, do_hessian=False)

    assert out_flat["energy"].shape == (2,)
    assert out_flat["forces"].shape == (2, 4, 3)
    assert torch.allclose(out_3d["energy"], out_flat["energy"], atol=1.0e-12)
    assert torch.allclose(out_3d["forces"], out_flat["forces"], atol=1.0e-12)


def test_lennard_jones_translation_invariance():
    predict_fn = make_lj_predict_fn(n_atoms=7)
    coords = pentagonal_bipyramid_geometry()
    atomic_nums = lj_atomic_nums(coords.shape[0])
    energy = predict_fn(coords, atomic_nums, do_hessian=False)["energy"]

    translated = coords + torch.tensor([1.5, -2.0, 0.25], dtype=torch.float64)
    energy_translated = predict_fn(translated, atomic_nums, do_hessian=False)["energy"]
    assert torch.allclose(energy, energy_translated, atol=1.0e-10)


def test_lennard_jones_requires_matching_atom_count():
    params = LennardJonesParams(epsilon=1.0, sigma=1.0)
    coords = random_cluster_geometry(7, sigma=1.0, generator=torch.Generator().manual_seed(1))
    atomic_nums = lj_atomic_nums(7)

    predict_fn = make_lj_predict_fn(params, n_atoms=5)

    with pytest.raises(ValueError, match="built for 5 atoms"):
        predict_fn(coords, atomic_nums, do_hessian=False)


def test_params_to_analytical_rm_uses_equilibrium_distance():
    assert params_to_analytical_rm(1.0) == pytest.approx(2.0 ** (1.0 / 6.0))


def test_lj_dissociation_threshold_scales_with_n_and_sigma():
    assert lj_dissociation_distance_threshold(7, 1.0) == pytest.approx(
        2.0 * (2.0 ** (1.0 / 6.0)) * (7 ** (1.0 / 3.0)),
        rel=1e-12,
    )
    assert lj_dissociation_distance_threshold(7, 2.0) == pytest.approx(
        2.0 * lj_dissociation_distance_threshold(7, 1.0),
        rel=1e-12,
    )


def test_lj_dissociation_flag_for_compact_lj7():
    coords = pentagonal_bipyramid_geometry(sigma=1.0)
    d_max = float(pair_distances(coords).max().item())
    assert not is_lj_dissociated(d_max, n_atoms=7, sigma=1.0)
    threshold = lj_dissociation_distance_threshold(7, 1.0)
    assert not is_lj_dissociated(threshold, n_atoms=7, sigma=1.0)
    assert is_lj_dissociated(threshold + 1.0e-6, n_atoms=7, sigma=1.0)
