"""Intrinsic lambda2 GAD: the best maintained Lennard-Jones map."""

import numpy as np


def step(x, gradient, eigenvalues, modes, total_mass):
    scale = np.sqrt(np.mean(eigenvalues**2))

    logits = -eigenvalues / (0.01 * scale)
    lowest = np.exp(logits - logits.max())
    lowest /= lowest.sum()
    ascend = np.exp(
        -np.logaddexp(0.0, -eigenvalues[1] / (0.01 * scale))
    )

    coefficients = modes.T @ gradient.reshape(-1)
    reflected = (1.0 - 2.0 * ascend * lowest) * coefficients

    distances = np.linalg.norm(x[:, None] - x[None, :], axis=-1)
    pair_distances = distances[np.triu_indices(len(x), 1)]
    length = 1.0 / np.sqrt(np.mean(1.0 / pair_distances**2))
    radius = 0.05 * length * np.sqrt(total_mass)
    regularizer = np.linalg.norm(reflected) / radius

    displacement = -reflected / np.sqrt(eigenvalues**2 + regularizer**2)
    return x + (modes @ displacement).reshape(x.shape)
