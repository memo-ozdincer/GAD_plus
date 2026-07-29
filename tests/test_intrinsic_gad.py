"""Unit and analytic-LJ tests for the strictly pointwise GAD map."""

from __future__ import annotations

import math
import unittest

import torch

from gadplus.calculator.lennard_jones import (
    lj_atomic_nums,
    make_lj_predict_fn,
    pentagonal_bipyramid_geometry,
)
from gadplus.projection import (
    atomic_nums_to_symbols,
    get_mass_weights,
    vib_eig,
)
from gadplus.search.intrinsic_gad import (
    IntrinsicGADConfig,
    effective_gate_weight,
    pointwise_step_coefficients,
    run_intrinsic_gad,
    smooth_spectral_policy,
)


class SpectralPolicyTests(unittest.TestCase):
    def test_experimental_gate_variants_are_local_and_selective(self) -> None:
        evals = torch.tensor([-3.0, -1.0, 2.0, 4.0], dtype=torch.float64)
        policy = smooth_spectral_policy(evals, temperature=0.01)
        soft_aligned = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64)
        diffuse_negative = torch.tensor([1.0, 4.0, 0.0, 0.0], dtype=torch.float64)
        stable_contamination = torch.tensor([1.0, 0.0, 10.0, 0.0], dtype=torch.float64)

        base = effective_gate_weight(soft_aligned, policy, "lambda2")
        pure_gad = effective_gate_weight(diffuse_negative, policy, "gad")
        aligned = effective_gate_weight(soft_aligned, policy, "alignment")
        diffuse = effective_gate_weight(diffuse_negative, policy, "competitive")
        alignment_with_stable = effective_gate_weight(
            stable_contamination, policy, "alignment"
        )
        competitive_with_stable = effective_gate_weight(
            stable_contamination, policy, "competitive"
        )
        boundary_policy = smooth_spectral_policy(
            torch.tensor([-0.01, -0.005, 2.0], dtype=torch.float64),
            temperature=0.01,
        )
        boundary_competitive = effective_gate_weight(
            diffuse_negative[:3], boundary_policy, "competitive"
        )
        boundary_guard = effective_gate_weight(
            diffuse_negative[:3], boundary_policy, "guard"
        )

        self.assertLess(float(base), 1.0e-6)
        self.assertEqual(float(pure_gad), 1.0)
        self.assertGreater(float(aligned), 1.0 - 1.0e-6)
        self.assertLess(float(diffuse), 0.1)
        self.assertLess(float(alignment_with_stable), 0.02)
        self.assertGreater(float(competitive_with_stable), 1.0 - 1.0e-6)
        self.assertGreater(float(boundary_guard), float(boundary_competitive))

    def test_positive_energy_rescaling_leaves_policy_and_step_unchanged(self) -> None:
        evals = torch.tensor([-3.0, 1.0, 4.0], dtype=torch.float64)
        gradient = torch.tensor([0.7, -0.2, 1.3], dtype=torch.float64)
        radius = 0.4

        policy = smooth_spectral_policy(evals, temperature=0.01)
        step, _ = pointwise_step_coefficients(gradient, evals, policy, radius)

        energy_scale = 37.0
        scaled_policy = smooth_spectral_policy(
            energy_scale * evals,
            temperature=0.01,
        )
        scaled_step, _ = pointwise_step_coefficients(
            energy_scale * gradient,
            energy_scale * evals,
            scaled_policy,
            radius,
        )

        torch.testing.assert_close(scaled_policy.gate, policy.gate)
        torch.testing.assert_close(
            scaled_policy.low_mode_weights,
            policy.low_mode_weights,
        )
        torch.testing.assert_close(scaled_step, step)

    def test_soft_projector_is_basis_invariant_at_degeneracy(self) -> None:
        evals = torch.tensor([1.0, 1.0, 3.0], dtype=torch.float64)
        policy = smooth_spectral_policy(evals, temperature=0.1)
        angle = 0.731
        rotation = torch.tensor(
            [
                [math.cos(angle), -math.sin(angle), 0.0],
                [math.sin(angle), math.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float64,
        )
        density = torch.diag(policy.low_mode_weights)
        rotated_density = rotation @ density @ rotation.T
        torch.testing.assert_close(rotated_density, density, atol=1e-12, rtol=1e-12)

    def test_gate_descends_at_high_index_and_flips_only_low_mode_at_index_one(self) -> None:
        gradient = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)

        high_index_evals = torch.tensor([-3.0, -1.0, 2.0], dtype=torch.float64)
        high_policy = smooth_spectral_policy(high_index_evals, temperature=0.01)
        high_step, _ = pointwise_step_coefficients(
            gradient,
            high_index_evals,
            high_policy,
            radius_mw=10.0,
        )
        self.assertLess(float(high_policy.gate.item()), 1e-8)
        self.assertTrue(bool((high_step * gradient < 0).all().item()))

        index_one_evals = torch.tensor([-3.0, 1.0, 2.0], dtype=torch.float64)
        index_one_policy = smooth_spectral_policy(index_one_evals, temperature=0.01)
        index_one_step, _ = pointwise_step_coefficients(
            gradient,
            index_one_evals,
            index_one_policy,
            radius_mw=10.0,
        )
        self.assertGreater(float(index_one_policy.gate.item()), 1.0 - 1e-8)
        self.assertGreater(float(index_one_step[0].item()), 0.0)
        self.assertTrue(bool((index_one_step[1:] * gradient[1:] < 0).all().item()))

    def test_closed_form_step_obeys_radius_bound(self) -> None:
        generator = torch.Generator().manual_seed(7)
        for _ in range(20):
            evals = torch.randn(12, generator=generator, dtype=torch.float64)
            gradient = torch.randn(12, generator=generator, dtype=torch.float64)
            radius = float(torch.rand((), generator=generator).item()) + 1.0e-3
            policy = smooth_spectral_policy(evals, temperature=0.02)
            step, _ = pointwise_step_coefficients(
                gradient,
                evals,
                policy,
                radius,
            )
            self.assertLessEqual(
                float(torch.linalg.vector_norm(step).item()),
                radius * (1.0 + 1.0e-12),
            )

    def test_uniform_mass_rescaling_leaves_cartesian_step_unchanged(self) -> None:
        cartesian_evals = torch.tensor([-4.0, 2.0, 7.0], dtype=torch.float64)
        cartesian_gradient = torch.tensor([0.8, -1.1, 0.3], dtype=torch.float64)
        locality_length = 0.2
        atom_count = 3

        cartesian_steps = []
        for mass in (1.0, 19.0):
            evals_mw = cartesian_evals / mass
            gradient_mw = cartesian_gradient / math.sqrt(mass)
            radius_mw = locality_length * math.sqrt(atom_count * mass)
            policy = smooth_spectral_policy(evals_mw, temperature=0.01)
            step_mw, _ = pointwise_step_coefficients(
                gradient_mw,
                evals_mw,
                policy,
                radius_mw,
            )
            cartesian_steps.append(step_mw / math.sqrt(mass))

        torch.testing.assert_close(cartesian_steps[0], cartesian_steps[1])


class LennardJonesBehaviorTests(unittest.TestCase):
    def test_pushed_lj7_minimum_reaches_an_index_one_saddle(self) -> None:
        predictor = make_lj_predict_fn()
        atomic_nums = lj_atomic_nums(7)
        minimum = pentagonal_bipyramid_geometry()
        symbols = atomic_nums_to_symbols(atomic_nums)
        minimum_out = predictor(minimum, atomic_nums, do_hessian=True)
        _, modes_mw, _ = vib_eig(
            minimum_out["hessian"],
            minimum,
            symbols,
        )
        _, _, _, inv_sqrt_mass = get_mass_weights(symbols)

        # This is the explicit targeting input: choose one member of the
        # exactly degenerate lowest-mode pair and push into its exit channel.
        start = minimum + 0.26 * (inv_sqrt_mass * modes_mw[:, 0]).reshape_as(minimum)
        result = run_intrinsic_gad(
            predictor,
            start,
            atomic_nums,
            IntrinsicGADConfig(max_steps=30, record_history=True),
        )

        self.assertTrue(result.converged)
        self.assertEqual(result.final_n_neg, 1)
        self.assertLess(result.final_force_max, 0.01)
        self.assertLess(result.total_steps, 20)
        self.assertEqual(result.n_evaluations, result.total_steps + 1)
        self.assertTrue(result.history)
        for record in result.history:
            self.assertLessEqual(
                record.step_rms,
                record.local_radius * (1.0 + 1.0e-12),
            )

    def test_high_index_noised_lj7_start_uses_descent_gate_and_recovers(self) -> None:
        predictor = make_lj_predict_fn()
        atomic_nums = lj_atomic_nums(7)
        minimum = pentagonal_bipyramid_geometry()
        generator = torch.Generator().manual_seed(42)
        start = minimum + 0.20 * torch.randn(
            minimum.shape,
            generator=generator,
            dtype=minimum.dtype,
        )
        start = start - start.mean(dim=0, keepdim=True)

        result = run_intrinsic_gad(
            predictor,
            start,
            atomic_nums,
            IntrinsicGADConfig(max_steps=40, record_history=True),
        )

        self.assertTrue(result.converged)
        self.assertGreaterEqual(result.history[0].n_neg, 2)
        self.assertLess(result.history[0].gate_weight, 1.0e-6)
        self.assertEqual(result.final_n_neg, 1)
        self.assertLess(result.final_force_max, 0.01)
        self.assertLess(result.total_steps, 30)


if __name__ == "__main__":
    unittest.main()
