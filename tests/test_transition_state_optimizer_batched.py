import pytest
import torch

from gadplus.search.transition_state_optimizer import (
    TransitionStateVectorfield as ScalarTransitionStateVectorfield,
)
from gadplus.search.transition_state_optimizer_batched import (
    TransitionStateVectorfieldBatched,
    _run_example,
)


def test_batched_transition_state_vectorfield_matches_scalar_for_multiple_geometries():
    coords_batch = _nonlinear_lj7_batch()
    scalar_vectorfield = ScalarTransitionStateVectorfield(n_atoms=7)
    batched_vectorfield = TransitionStateVectorfieldBatched(n_atoms=7)

    batched_direction = batched_vectorfield(coords_batch)
    scalar_directions = []
    scalar_n_neg = []
    for coords in coords_batch:
        scalar_directions.append(scalar_vectorfield(coords))
        scalar_n_neg.append(scalar_vectorfield.last_info["n_neg"])
    scalar_directions = torch.stack(scalar_directions)

    assert batched_direction.shape == coords_batch.shape
    torch.testing.assert_close(batched_direction, scalar_directions, atol=1e-10, rtol=1e-10)

    assert isinstance(batched_vectorfield.last_info["n_neg"], torch.Tensor)
    torch.testing.assert_close(
        batched_vectorfield.last_info["n_neg"],
        torch.tensor(scalar_n_neg, device=coords_batch.device),
    )


def test_batched_transition_state_vectorfield_supports_flat_batch_shape():
    coords_batch = _nonlinear_lj7_batch()
    vectorfield = TransitionStateVectorfieldBatched(n_atoms=7)

    direction = vectorfield(coords_batch.reshape(coords_batch.shape[0], -1))

    assert direction.shape == (coords_batch.shape[0], coords_batch[0].numel())


def test_batched_transition_state_vectorfield_rejects_single_geometry():
    coords_batch = _nonlinear_lj7_batch()
    vectorfield = TransitionStateVectorfieldBatched(n_atoms=7)

    with pytest.raises(ValueError, match="Expected batched coords"):
        vectorfield(coords_batch[0])

    with pytest.raises(ValueError, match="Expected batched coords"):
        vectorfield(coords_batch[0].reshape(-1))


def test_batched_example_loop_smoke(monkeypatch, capsys):
    monkeypatch.setattr(
        "sys.argv",
        [
            "transition_state_optimizer_batched.py",
            "--n-samples",
            "3",
            "--n-steps",
            "1",
            "--seed",
            "1",
        ],
    )

    _run_example()

    captured = capsys.readouterr()
    assert "samples=3" in captured.out
    assert captured.out.count("[") == 3


def _nonlinear_lj7_batch() -> torch.Tensor:
    generator = torch.Generator().manual_seed(24)
    scaffold = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.1, 0.2],
            [-0.2, 1.1, 0.3],
            [0.3, -0.4, 1.2],
            [-1.0, -0.2, 0.4],
            [0.5, 0.9, -0.7],
            [-0.6, 0.4, -1.0],
        ],
        dtype=torch.float64,
    )
    noise = 0.05 * torch.randn((4, 7, 3), generator=generator, dtype=torch.float64)
    coords = scaffold.unsqueeze(0) + noise
    return coords - coords.mean(dim=1, keepdim=True)
