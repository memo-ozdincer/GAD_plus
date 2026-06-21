from __future__ import annotations

import torch

from gadplus.calculator.lennard_jones import (
    lj_atomic_nums,
    make_lj_predict_fn,
    random_cluster_geometry,
)
from gadplus.search import transition_state_optimizer as tso
from gadplus.search.transition_state_optimizer import (
    TransitionStateOptimizationResult,
    TransitionStateOptimizer,
)


def test_transition_state_optimizer_stores_potential_energy_surface_and_dt():
    predict_fn = make_lj_predict_fn(n_atoms=4)

    optimizer = TransitionStateOptimizer(
        potential_energy_surface=predict_fn,
        n_atoms=4,
        atomic_nums=lj_atomic_nums(4, atomic_number=1),
        dt=0.007,
        n_steps=2,
    )

    assert optimizer.potential_energy_surface is predict_fn
    assert optimizer.dt == 0.007
    assert optimizer.config.dt == 0.007
    assert optimizer.config.n_steps == 2


def test_projected_step_uses_gradient_descent_only_for_high_index(monkeypatch):
    optimizer = TransitionStateOptimizer(n_atoms=2, n_steps=1)
    calls: list[float] = []

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

    monkeypatch.setattr(tso, "gad_dynamics_projected", fake_gad_dynamics_projected)
    coords = torch.zeros((2, 3), dtype=torch.float32)
    forces = torch.ones((2, 3), dtype=torch.float32)
    evecs = torch.eye(6, dtype=torch.float32)

    optimizer._projected_step_direction(
        coords=coords,
        forces=forces,
        evecs_vib_3n=evecs,
        atomsymbols=["H", "H"],
        n_neg=2,
        v_prev=None,
    )
    optimizer._projected_step_direction(
        coords=coords,
        forces=forces,
        evecs_vib_3n=evecs,
        atomsymbols=["H", "H"],
        n_neg=1,
        v_prev=None,
    )

    assert calls == [0.0, 1.0]


def test_transition_state_optimizer_runs_tiny_lj_optimization():
    optimizer = TransitionStateOptimizer(
        n_atoms=3,
        atomic_number=1,
        n_steps=1,
        dt=1.0e-3,
        min_interatomic_dist=0.1,
    )
    coords = random_cluster_geometry(3, sigma=1.0, generator=torch.Generator().manual_seed(0))

    result = optimizer.optimize(coords)

    assert isinstance(result, TransitionStateOptimizationResult)
    assert result.total_steps == 1
    assert result.final_coords.shape == (3, 3)
    assert torch.isfinite(result.final_coords).all()
