"""Competitive Soft-Spectral GAD (CS²-GAD): the best g-xTB map."""

import numpy as np


def step(x, gradient, eigenvalues, modes, total_mass):
    scale = np.sqrt(np.mean(eigenvalues**2))

    logits = -eigenvalues / (0.01 * scale)
    lowest = np.exp(logits - logits.max())
    lowest /= lowest.sum()
    lowest_subspace = lowest / lowest.max()
    negative = np.exp(
        -np.logaddexp(0.0, eigenvalues / (0.01 * scale))
    )

    coefficients = modes.T @ gradient.reshape(-1)
    activity = coefficients**2
    lowest_activity = np.sum(lowest * activity)
    other_negative_activity = np.sum(negative * (1.0 - lowest) ** 2 * activity)
    total_activity = lowest_activity + other_negative_activity
    competition = lowest_activity / total_activity if total_activity else 0.0

    lambda2_gate = np.exp(
        -np.logaddexp(0.0, -eigenvalues[1] / (0.01 * scale))
    )
    ascend = lambda2_gate + (1.0 - lambda2_gate) * competition
    reflected = (1.0 - 2.0 * ascend * lowest_subspace) * coefficients

    distances = np.linalg.norm(x[:, None] - x[None, :], axis=-1)
    pair_distances = distances[np.triu_indices(len(x), 1)]
    length = 1.0 / np.sqrt(np.mean(1.0 / pair_distances**2))
    radius = 0.01 * length * np.sqrt(total_mass)
    regularizer = np.linalg.norm(reflected) / radius

    displacement = -reflected / np.sqrt(eigenvalues**2 + regularizer**2)
    return x + (modes @ displacement).reshape(x.shape)
