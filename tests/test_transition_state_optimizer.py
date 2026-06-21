from __future__ import annotations

import torch

from gadplus.calculator.lennard_jones import (
    lj_atomic_nums,
    make_lj_predict_fn,
    random_cluster_geometry,
)
from gadplus.search import transition_state_optimizer as tso
from gadplus.search.transition_state_optimizer import TransitionStateVectorfield


def test_transition_state_vectorfield_stores_potential_energy_surface():
    predict_fn = make_lj_predict_fn(n_atoms=4)
    atomic_nums = lj_atomic_nums(4, atomic_number=1)

    vectorfield = TransitionStateVectorfield(
        predict_fn,
        atomic_nums=atomic_nums,
    )

    assert vectorfield.potential_energy_surface is predict_fn
    assert vectorfield.n_atoms == 4
    assert vectorfield.last_info == {}


def test_transition_state_vectorfield_returns_finite_lj_direction():
    predict_fn = make_lj_predict_fn(n_atoms=3)
    atomic_nums = lj_atomic_nums(3, atomic_number=1)
    vectorfield = TransitionStateVectorfield(predict_fn, atomic_nums=atomic_nums)
    coords = random_cluster_geometry(3, sigma=1.0, generator=torch.Generator().manual_seed(0))

    direction = vectorfield(coords)

    assert direction.shape == (3, 3)
    assert torch.isfinite(direction).all()
    assert vectorfield.last_info["phase"] in {"descent", "gad"}
    assert isinstance(vectorfield.last_info["n_neg"], int)


def test_transition_state_vectorfield_blend_switches_with_projected_index(monkeypatch):
    atomic_nums = lj_atomic_nums(2, atomic_number=1)

    def fake_predict_fn(coords, atomic_nums_arg, *, do_hessian=True, require_grad=False):
        del atomic_nums_arg, do_hessian, require_grad
        return {
            "energy": torch.tensor(0.0, dtype=coords.dtype),
            "forces": torch.ones_like(coords),
            "hessian": torch.eye(coords.numel(), dtype=coords.dtype),
        }

    vectorfield = TransitionStateVectorfield(fake_predict_fn, atomic_nums=atomic_nums)
    calls: list[float] = []

    def fake_vib_eig(hessian, coords, atomsymbols, purify=False):
        del hessian, coords, atomsymbols, purify
        evals = torch.tensor([-2.0, -1.0, 0.5, 1.0, 2.0, 3.0])
        return evals, torch.eye(6), torch.eye(6)

    def fake_count_negative_eigenvalues(evals):
        return int((evals < 0).sum().item())

    def fake_gad_dynamics_projected(
        coords,
        forces,
        v,
        atomsymbols,
        gad_blend_weight,
        return_weighted_step_direction,
    ):
        del atomsymbols, return_weighted_step_direction
        calls.append(float(gad_blend_weight))
        return forces.clone(), v.clone(), {}

    monkeypatch.setattr(tso, "vib_eig", fake_vib_eig)
    monkeypatch.setattr(tso, "count_negative_eigenvalues", fake_count_negative_eigenvalues)
    monkeypatch.setattr(tso, "gad_dynamics_projected", fake_gad_dynamics_projected)

    coords = torch.zeros((2, 3), dtype=torch.float32)
    vectorfield(coords)

    def fake_vib_eig_index1(hessian, coords, atomsymbols, purify=False):
        del hessian, coords, atomsymbols, purify
        evals = torch.tensor([-1.0, 0.5, 1.0, 2.0, 3.0, 4.0])
        return evals, torch.eye(6), torch.eye(6)

    monkeypatch.setattr(tso, "vib_eig", fake_vib_eig_index1)
    vectorfield.reset()
    vectorfield(coords)

    assert calls == [0.0, 1.0]
