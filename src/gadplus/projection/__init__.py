"""Differentiable Eckart projection and vibrational analysis."""

from gadplus.projection.projection import (
    MASS_DICT,
    Z_TO_SYMBOL,
    atomic_nums_to_symbols,
    get_mass_weights,
    purify_hessian,
    vib_eig,
    gad_dynamics_projected,
    multimode_gad_dynamics_projected,
    preconditioned_gad_dynamics_projected,
    project_vector_to_vibrational,
)
from gadplus.projection.batched import (
    batched_gad_dynamics_projected,
    batched_project_vector_to_vibrational,
    batched_vib_eig,
)

__all__ = [
    "MASS_DICT",
    "Z_TO_SYMBOL",
    "atomic_nums_to_symbols",
    "get_mass_weights",
    "purify_hessian",
    "vib_eig",
    "gad_dynamics_projected",
    "multimode_gad_dynamics_projected",
    "preconditioned_gad_dynamics_projected",
    "project_vector_to_vibrational",
    "batched_gad_dynamics_projected",
    "batched_project_vector_to_vibrational",
    "batched_vib_eig",
]
