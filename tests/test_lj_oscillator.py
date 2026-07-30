from __future__ import annotations

import pytest
import torch

from gadplus.calculator.lennard_jones import (
    make_lj_predict_fn,
    pentagonal_bipyramid_geometry,
    random_cluster_geometry,
)
from gadplus.calculator.lj_oscillator import normalize_oscillator_mode, oscillator_energy


@pytest.mark.parametrize("mode", ["linear", "deadzone", "pair", "switch", "quartic"])
def test_lj_oscillator_modes_have_finite_energy(mode: str):
    predict_fn = make_lj_predict_fn(
        n_atoms=7,
        oscillator_mode=mode,
        oscillator_scale=0.5,
    )
    coords = random_cluster_geometry(7, generator=torch.Generator().manual_seed(0))
    atomic_nums = torch.full((7,), 1, dtype=torch.long)
    out = predict_fn(coords, atomic_nums, do_hessian=True)
    assert torch.isfinite(out["energy"])
    assert torch.isfinite(out["forces"]).all()
    assert torch.isfinite(out["hessian"]).all()


def test_compact_lj7_has_no_deadzone_energy():
    coords = pentagonal_bipyramid_geometry().reshape(1, 7, 3)
    diff = coords[:, :, None, :] - coords[:, None, :, :]
    distances = torch.linalg.norm(diff, dim=-1)
    energy = oscillator_energy(
        coords,
        distances,
        mode="deadzone",
        scale=1.0,
        rm=2.0 ** (1.0 / 6.0),
        n_particles=7,
        r0_factor=1.0,
        rcut_factor=1.0,
        switch_width_factor=0.3,
    )
    assert float(energy.item()) == pytest.approx(0.0, abs=1.0e-10)


def test_pair_oscillator_only_penalizes_long_contacts():
    coords = pentagonal_bipyramid_geometry().reshape(1, 7, 3)
    diff = coords[:, :, None, :] - coords[:, None, :, :]
    distances = torch.linalg.norm(diff, dim=-1)
    compact = oscillator_energy(
        coords,
        distances,
        mode="pair",
        scale=1.0,
        rm=2.0 ** (1.0 / 6.0),
        n_particles=7,
        r0_factor=1.0,
        rcut_factor=1.0,
        switch_width_factor=0.3,
    )
    stretched = coords.clone()
    stretched[:, 0, 0] += 4.0
    diff_st = stretched[:, :, None, :] - stretched[:, None, :, :]
    distances_st = torch.linalg.norm(diff_st, dim=-1)
    stretched_energy = oscillator_energy(
        stretched,
        distances_st,
        mode="pair",
        scale=1.0,
        rm=2.0 ** (1.0 / 6.0),
        n_particles=7,
        r0_factor=1.0,
        rcut_factor=1.0,
        switch_width_factor=0.3,
    )
    assert float(stretched_energy.item()) > float(compact.item())


def test_normalize_oscillator_mode_rejects_unknown():
    with pytest.raises(ValueError, match="Unknown oscillator mode"):
        normalize_oscillator_mode("foo")
