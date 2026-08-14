import unittest

import numpy as np
import torch

from examples.best_methods import gxtb, hip, lj
from gadplus.search.intrinsic_gad import pointwise_step_coefficients, smooth_spectral_policy


class BestMethodReferenceTests(unittest.TestCase):
    def setUp(self):
        self.x = np.array([[0.0, 0.0, 0.0], [1.25, 0.0, 0.0]])
        self.gradient = np.array([0.4, -0.2, 0.1, -0.3, 0.2, -0.1])
        self.eigenvalues = np.array([-2.0, 0.4, 0.8, 1.1, 1.5, 2.0])
        self.modes = np.eye(6)
        self.total_mass = 2.016

    def _expected_intrinsic_step(self, eta, variant):
        eigenvalues = torch.tensor(self.eigenvalues, dtype=torch.float64)
        coefficients = torch.tensor(self.gradient, dtype=torch.float64)
        policy = smooth_spectral_policy(eigenvalues, 0.01)
        radius = eta * 1.25 * np.sqrt(self.total_mass)
        step, _ = pointwise_step_coefficients(
            coefficients, eigenvalues, policy, radius, variant
        )
        return self.x + step.numpy().reshape(self.x.shape)

    def test_lj_matches_production_lambda2_map(self):
        observed = lj.step(
            self.x, self.gradient, self.eigenvalues, self.modes, self.total_mass
        )
        np.testing.assert_allclose(
            observed, self._expected_intrinsic_step(0.05, "lambda2"), rtol=1e-13, atol=1e-13
        )

    def test_gxtb_matches_production_cs2_map(self):
        observed = gxtb.step(
            self.x, self.gradient, self.eigenvalues, self.modes, self.total_mass
        )
        np.testing.assert_allclose(
            observed,
            self._expected_intrinsic_step(0.01, "competitive_subspace"),
            rtol=1e-13,
            atol=1e-13,
        )

    def test_hip_is_lowest_mode_householder_euler_step(self):
        force = -self.gradient
        lowest_mode = self.modes[:, 0]
        observed = hip.step(self.x, force, lowest_mode)
        direction = force - 2.0 * np.dot(force, lowest_mode) * lowest_mode
        expected = self.x + 0.007 * direction.reshape(self.x.shape)
        np.testing.assert_allclose(observed, expected, rtol=0.0, atol=1e-15)


if __name__ == "__main__":
    unittest.main()
