"""Plain one-mode GAD: the best HIP map."""

import numpy as np


def step(x, force, lowest_mode):
    force = force.reshape(-1)
    lowest_mode = lowest_mode.reshape(-1)
    gad_force = force - 2.0 * np.dot(force, lowest_mode) * lowest_mode
    return x + 0.007 * gad_force.reshape(x.shape)
