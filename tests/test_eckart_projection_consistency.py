import pytest
import torch

from gadplus.core.convergence import count_negative_eigenvalues
from gadplus.projection import (
    atomic_nums_to_symbols,
    batched_gad_dynamics_projected,
    batched_project_vector_to_vibrational,
    batched_vib_eig,
    gad_dynamics_projected,
    project_vector_to_vibrational,
    vib_eig,
)
from gadplus.search.hybrid_gad_damped_eigfollownewton_eckart import (
    eckart_internal_basis,
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


def test_projected_gad_returns_unweighted_step_direction_by_default():
    coords = torch.tensor(
        [
            [0.0000, 0.0000, 0.0000],
            [1.2300, 0.1100, -0.0200],
            [-0.5200, 1.0700, 0.2600],
            [-0.6100, -0.9400, 0.1900],
        ],
        dtype=torch.float64,
    )
    atomic_nums = torch.tensor([6, 8, 7, 1])
    atomsymbols = atomic_nums_to_symbols(atomic_nums)
    forces = torch.tensor(
        [
            [0.30, -0.20, 0.10],
            [-0.40, 0.50, -0.10],
            [0.20, 0.10, -0.30],
            [-0.10, -0.40, 0.20],
        ],
        dtype=torch.float64,
    )
    v = torch.arange(1, coords.numel() + 1, dtype=torch.float64)
    masses3d = masses_from_z(atomic_nums, dtype=torch.float64).repeat_interleave(3)

    unweighted, _, unweighted_info = gad_dynamics_projected(
        coords=coords,
        forces=forces,
        v=v,
        atomsymbols=atomsymbols,
    )
    weighted, _, weighted_info = gad_dynamics_projected(
        coords=coords,
        forces=forces,
        v=v,
        atomsymbols=atomsymbols,
        return_weighted_step_direction=True,
    )

    assert not unweighted_info["return_weighted_step_direction"]
    assert weighted_info["return_weighted_step_direction"]
    torch.testing.assert_close(
        weighted.reshape(-1),
        masses3d * unweighted.reshape(-1),
        atol=1e-12,
        rtol=1e-12,
    )


def test_batched_vib_eig_matches_scalar_for_multiple_geometries():
    atomic_nums = torch.tensor([6, 8, 7, 1])
    atomsymbols = atomic_nums_to_symbols(atomic_nums)
    coords_batch = torch.tensor(
        [
            [
                [0.0000, 0.0000, 0.0000],
                [1.2300, 0.1100, -0.0200],
                [-0.5200, 1.0700, 0.2600],
                [-0.6100, -0.9400, 0.1900],
            ],
            [
                [0.1000, -0.1200, 0.0900],
                [1.0400, 0.4200, -0.3300],
                [-0.6700, 0.8900, 0.5100],
                [-0.4700, -1.1900, -0.2700],
            ],
            [
                [-0.2100, 0.0700, -0.1400],
                [1.3900, -0.1800, 0.2300],
                [-0.8100, 1.2100, -0.3600],
                [-0.3700, -1.1000, 0.2700],
            ],
        ],
        dtype=torch.float64,
    )
    generator = torch.Generator(device=coords_batch.device)
    generator.manual_seed(12)
    dim = coords_batch.shape[1] * 3
    raw_hessians = torch.randn(
        coords_batch.shape[0],
        dim,
        dim,
        generator=generator,
        dtype=torch.float64,
    )
    hessians = 0.5 * (raw_hessians + raw_hessians.transpose(-1, -2))

    evals_batch, evecs_batch, q_batch = batched_vib_eig(hessians, coords_batch, atomsymbols)

    for idx, coords in enumerate(coords_batch):
        evals_scalar, evecs_scalar, q_scalar = vib_eig(hessians[idx], coords, atomsymbols)
        torch.testing.assert_close(evals_batch[idx], evals_scalar, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(
            q_batch[idx] @ q_batch[idx].T,
            q_scalar @ q_scalar.T,
            atol=1e-12,
            rtol=1e-12,
        )
        torch.testing.assert_close(
            evecs_batch[idx] @ evecs_batch[idx].T,
            evecs_scalar @ evecs_scalar.T,
            atol=1e-12,
            rtol=1e-12,
        )


def test_batched_projected_gad_matches_scalar_for_multiple_geometries():
    atomic_nums = torch.tensor([6, 8, 7, 1])
    atomsymbols = atomic_nums_to_symbols(atomic_nums)
    coords_batch = torch.tensor(
        [
            [
                [0.0000, 0.0000, 0.0000],
                [1.2300, 0.1100, -0.0200],
                [-0.5200, 1.0700, 0.2600],
                [-0.6100, -0.9400, 0.1900],
            ],
            [
                [0.1000, -0.1200, 0.0900],
                [1.0400, 0.4200, -0.3300],
                [-0.6700, 0.8900, 0.5100],
                [-0.4700, -1.1900, -0.2700],
            ],
            [
                [-0.2100, 0.0700, -0.1400],
                [1.3900, -0.1800, 0.2300],
                [-0.8100, 1.2100, -0.3600],
                [-0.3700, -1.1000, 0.2700],
            ],
        ],
        dtype=torch.float64,
    )
    forces_batch = torch.tensor(
        [
            [[0.30, -0.20, 0.10], [-0.40, 0.50, -0.10], [0.20, 0.10, -0.30], [-0.10, -0.40, 0.20]],
            [[-0.20, 0.10, 0.40], [0.30, -0.60, 0.20], [0.50, 0.20, -0.10], [-0.60, 0.30, -0.50]],
            [[0.70, -0.10, -0.20], [-0.30, 0.20, 0.60], [-0.20, -0.50, 0.30], [-0.20, 0.40, -0.70]],
        ],
        dtype=torch.float64,
    )
    guide_vectors = torch.stack(
        [
            torch.arange(1, coords_batch[0].numel() + 1, dtype=torch.float64),
            torch.linspace(-2.0, 1.0, coords_batch[0].numel(), dtype=torch.float64),
            torch.linspace(0.5, 3.0, coords_batch[0].numel(), dtype=torch.float64),
        ]
    )
    blend_weights = torch.tensor([1.0, 0.0, 0.35], dtype=torch.float64)

    gad_batch, v_proj_batch, info_batch = batched_gad_dynamics_projected(
        coords=coords_batch,
        forces=forces_batch,
        v=guide_vectors,
        atomsymbols=atomsymbols,
        gad_blend_weight=blend_weights,
    )
    vec_proj_batch = batched_project_vector_to_vibrational(
        guide_vectors.reshape(coords_batch.shape),
        coords_batch,
        atomsymbols,
    )

    for idx, coords in enumerate(coords_batch):
        gad_scalar, v_proj_scalar, info_scalar = gad_dynamics_projected(
            coords=coords,
            forces=forces_batch[idx],
            v=guide_vectors[idx],
            atomsymbols=atomsymbols,
            gad_blend_weight=blend_weights[idx],
        )
        torch.testing.assert_close(gad_batch[idx], gad_scalar, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(
            torch.outer(v_proj_batch[idx], v_proj_batch[idx]),
            torch.outer(v_proj_scalar, v_proj_scalar),
            atol=1e-12,
            rtol=1e-12,
        )
        torch.testing.assert_close(
            info_batch["v_dot_grad"][idx],
            torch.tensor(info_scalar["v_dot_grad"], dtype=torch.float64),
            atol=1e-12,
            rtol=1e-12,
        )
        vec_proj_scalar = project_vector_to_vibrational(
            guide_vectors[idx],
            coords,
            atomsymbols,
        )
        torch.testing.assert_close(vec_proj_batch[idx], vec_proj_scalar, atol=1e-12, rtol=1e-12)


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


def test_high_index_descent_index_controlled_escapes_only_extra_negative_modes():
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
    _, U_int, _ = eckart_internal_basis(coords, masses)

    num_internal = U_int.shape[1]
    eigvals = torch.arange(2.0, 2.0 + num_internal, dtype=torch.float64)
    eigvals[0] = -4.0
    eigvals[1] = -1.0

    force_internal = torch.zeros(num_internal, dtype=torch.float64)
    force_internal[0] = 8.0
    force_internal[1] = 3.0
    force_internal[2] = 6.0

    sqrt_m3 = torch.sqrt(masses).repeat_interleave(3)
    hessian_mw = U_int @ torch.diag(eigvals) @ U_int.T
    hessian_cart = sqrt_m3[:, None] * hessian_mw * sqrt_m3[None, :]
    force_cart = (sqrt_m3 * (U_int @ force_internal)).reshape_as(coords)

    step_cart, info = projected_hybrid_gad_newton_step(
        force_cart=force_cart,
        hessian_cart=hessian_cart,
        coords=coords,
        masses=masses,
        target_mode=0,
        high_index_descent="index_controlled",
        trust_radius=None,
    )

    step_internal = U_int.T @ (sqrt_m3 * step_cart.reshape(-1))
    expected = torch.zeros_like(force_internal)
    expected[0] = -force_internal[0] / eigvals[0].abs()
    expected[1] = force_internal[1] / eigvals[1].abs()
    expected[2] = force_internal[2] / eigvals[2].abs()

    assert info["method"] == "projected_index_controlled_newton"
    assert int(info["num_negative_modes"].item()) == 2
    torch.testing.assert_close(step_internal, expected, atol=1e-12, rtol=1e-12)


def test_target_mode_strategy_selects_negative_mode_by_force_coupling():
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
    _, U_int, _ = eckart_internal_basis(coords, masses)

    num_internal = U_int.shape[1]
    eigvals = torch.arange(2.0, 2.0 + num_internal, dtype=torch.float64)
    eigvals[0] = -4.0
    eigvals[1] = -1.0

    force_internal = torch.zeros(num_internal, dtype=torch.float64)
    force_internal[0] = 1.0
    force_internal[1] = 5.0
    force_internal[2] = 2.0

    sqrt_m3 = torch.sqrt(masses).repeat_interleave(3)
    hessian_mw = U_int @ torch.diag(eigvals) @ U_int.T
    hessian_cart = sqrt_m3[:, None] * hessian_mw * sqrt_m3[None, :]
    force_cart = (sqrt_m3 * (U_int @ force_internal)).reshape_as(coords)

    step_cart, info = projected_hybrid_gad_newton_step(
        force_cart=force_cart,
        hessian_cart=hessian_cart,
        coords=coords,
        masses=masses,
        target_mode=0,
        target_mode_strategy="neg_force_coupling",
        switch_based_on_hessian_eigval=True,
        gad_dt=0.01,
        trust_radius=None,
    )

    step_internal = U_int.T @ (sqrt_m3 * step_cart.reshape(-1))
    expected = 0.01 * force_internal.clone()
    expected[1] *= -1.0

    assert info["method"] == "projected_gad"
    assert info["target_mode"] == 1
    torch.testing.assert_close(
        info["target_force_coupling"],
        torch.tensor(5.0, dtype=torch.float64),
    )
    torch.testing.assert_close(step_internal, expected, atol=1e-12, rtol=1e-12)
