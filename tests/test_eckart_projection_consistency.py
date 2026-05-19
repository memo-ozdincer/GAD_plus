import pytest
import torch

from gadplus.core.convergence import count_negative_eigenvalues
from gadplus.projection import atomic_nums_to_symbols, vib_eig
from gadplus.search.hybrid_gad_damped_eigfollownewton_eckart import (
    _internal_mass_weighted_state,
    masses_from_z,
    projected_hybrid_gad_newton_step,
)


@pytest.mark.parametrize(
    ("coords", "atomic_nums"),
    [
        (
            torch.tensor(
                [
                    [0.0000, 0.0000, 0.0000],
                    [0.6291, 0.6291, 0.6291],
                    [-0.6291, -0.6291, 0.6291],
                    [-0.6291, 0.6291, -0.6291],
                    [0.6291, -0.6291, -0.6291],
                ],
                dtype=torch.float64,
            ),
            torch.tensor([6, 1, 1, 1, 1]),
        ),
        (
            torch.tensor(
                [
                    [0.0000, 0.0000, 0.0000],
                    [1.2300, 0.1100, -0.0200],
                    [-0.5200, 1.0700, 0.2600],
                    [-0.6100, -0.9400, 0.1900],
                    [1.9400, -0.5700, -0.1300],
                    [-1.3400, 0.3600, -0.3100],
                ],
                dtype=torch.float64,
            ),
            torch.tensor([6, 8, 7, 1, 1, 1]),
        ),
    ],
)
def test_runner_vib_eig_matches_hybrid_damped_eckart_internal_basis(coords, atomic_nums):
    """The runner diagnostic path and step path must see the same vibrational Hessian."""
    coords = coords.double()
    atomic_nums = atomic_nums.long()
    masses = masses_from_z(atomic_nums, device=coords.device, dtype=coords.dtype)
    atomsymbols = atomic_nums_to_symbols(atomic_nums)

    generator = torch.Generator(device=coords.device)
    generator.manual_seed(0)

    for _ in range(3):
        dim = coords.numel()
        raw_hessian = torch.randn(dim, dim, generator=generator, dtype=torch.float64)
        hessian = 0.5 * (raw_hessian + raw_hessian.T)
        forces = torch.randn(coords.shape, generator=generator, dtype=torch.float64)

        ref_eigvals, _, ref_basis = vib_eig(
            hessian,
            coords,
            atomsymbols,
            purify=False,
        )
        state = _internal_mass_weighted_state(
            force_cart=forces,
            hessian_cart=hessian,
            coords=coords,
            masses=masses,
        )
        step_eigvals = torch.linalg.eigvalsh(state["H_i"])

        ref_projector = ref_basis @ ref_basis.T
        step_projector = state["U_int"] @ state["U_int"].T

        torch.testing.assert_close(step_projector, ref_projector, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(step_eigvals, ref_eigvals, atol=1e-12, rtol=1e-12)


def test_negative_mode_count_uses_hip_frequency_cutoff():
    eigvals = torch.tensor([-1.0e-3, -5.0e-7, 0.0, 1.0e-4], dtype=torch.float64)

    assert count_negative_eigenvalues(eigvals) == 1


def test_hybrid_damped_eckart_inertia_uses_hip_frequency_cutoff():
    coords = torch.tensor(
        [
            [0.0000, 0.0000, 0.0000],
            [0.6291, 0.6291, 0.6291],
            [-0.6291, -0.6291, 0.6291],
            [-0.6291, 0.6291, -0.6291],
            [0.6291, -0.6291, -0.6291],
        ],
        dtype=torch.float64,
    )
    atomic_nums = torch.tensor([6, 1, 1, 1, 1])
    masses = masses_from_z(atomic_nums, device=coords.device, dtype=coords.dtype)
    masses3d = masses.repeat_interleave(3)
    force = torch.ones_like(coords, dtype=torch.float64)
    hessian = torch.diag(-5.0e-7 * masses3d)

    _, info = projected_hybrid_gad_newton_step(
        force_cart=force,
        hessian_cart=hessian,
        coords=coords,
        masses=masses,
        switch_based_on_hessian_eigval=True,
    )

    assert int(info["num_negative_modes"].item()) == 0
    assert not bool(info["hessian_has_clear_index1"].item())
