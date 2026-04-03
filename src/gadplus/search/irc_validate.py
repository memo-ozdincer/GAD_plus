"""IRC (Intrinsic Reaction Coordinate) validation using Sella.

After finding a TS, run IRC forward and backward to verify that it connects
the intended reactant and product. Uses Sella's IRC optimizer with HIP
via the ASE adapter.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from ase import Atoms

from gadplus.projection import Z_TO_SYMBOL


@dataclass
class IRCResult:
    """Result of IRC validation."""
    intended: bool              # Both reactant and product matched
    half_intended: bool         # Only one endpoint matched
    forward_coords: Optional[np.ndarray]    # Final geometry from forward IRC
    reverse_coords: Optional[np.ndarray]    # Final geometry from reverse IRC
    rmsd_to_reactant: Optional[float]
    rmsd_to_product: Optional[float]
    error: Optional[str] = None


def run_irc_validation(
    ts_coords: torch.Tensor,
    atomic_nums: torch.Tensor,
    predict_fn,
    reactant_coords: Optional[torch.Tensor] = None,
    product_coords: Optional[torch.Tensor] = None,
    rmsd_threshold: float = 0.3,
    max_steps: int = 100,
) -> IRCResult:
    """Run IRC forward and backward from a converged TS.

    Args:
        ts_coords: (N, 3) TS coordinates.
        atomic_nums: (N,) atomic numbers.
        predict_fn: PredictFn for energy/forces.
        reactant_coords: (N, 3) reference reactant geometry.
        product_coords: (N, 3) reference product geometry.
        rmsd_threshold: RMSD threshold for matching endpoints.
        max_steps: Maximum IRC steps per direction.

    Returns:
        IRCResult with validation status and final geometries.
    """
    try:
        from sella import IRC
    except ImportError:
        return IRCResult(
            intended=False, half_intended=False,
            forward_coords=None, reverse_coords=None,
            rmsd_to_reactant=None, rmsd_to_product=None,
            error="Sella not installed",
        )

    from gadplus.calculator.ase_adapter import HipASECalculator

    # Build ASE Atoms from TS geometry
    coords_np = ts_coords.detach().cpu().numpy().reshape(-1, 3)
    nums = atomic_nums.detach().cpu().tolist()
    symbols = [Z_TO_SYMBOL.get(int(z), "X") for z in nums]

    atoms = Atoms(symbols=symbols, positions=coords_np)
    atoms.calc = HipASECalculator(predict_fn=predict_fn, atomic_nums=atomic_nums)

    optimizer_kwargs = {"dx": 0.1, "eta": 1e-4, "gamma": 0.4}

    # Run IRC in both directions
    endpoints = {}
    for direction in ["forward", "reverse"]:
        try:
            atoms_copy = atoms.copy()
            atoms_copy.calc = HipASECalculator(predict_fn=predict_fn, atomic_nums=atomic_nums)
            irc = IRC(atoms=atoms_copy, **optimizer_kwargs)
            irc.run(fmax=0.01, steps=max_steps, direction=direction)
            endpoints[direction] = atoms_copy.positions.copy()
        except Exception as e:
            endpoints[direction] = None

    forward_coords = endpoints.get("forward")
    reverse_coords = endpoints.get("reverse")

    # Compare endpoints to known reactant/product
    found_reactant = False
    found_product = False
    rmsd_to_reactant = None
    rmsd_to_product = None

    for endpoint in [forward_coords, reverse_coords]:
        if endpoint is None:
            continue

        if reactant_coords is not None:
            ref = reactant_coords.detach().cpu().numpy().reshape(-1, 3)
            rmsd = float(np.sqrt(np.mean((endpoint - ref) ** 2)))
            if rmsd_to_reactant is None or rmsd < rmsd_to_reactant:
                rmsd_to_reactant = rmsd
            if rmsd < rmsd_threshold:
                found_reactant = True

        if product_coords is not None:
            ref = product_coords.detach().cpu().numpy().reshape(-1, 3)
            rmsd = float(np.sqrt(np.mean((endpoint - ref) ** 2)))
            if rmsd_to_product is None or rmsd < rmsd_to_product:
                rmsd_to_product = rmsd
            if rmsd < rmsd_threshold:
                found_product = True

    intended = found_reactant and found_product
    half_intended = (found_reactant or found_product) and not intended

    return IRCResult(
        intended=intended,
        half_intended=half_intended,
        forward_coords=forward_coords,
        reverse_coords=reverse_coords,
        rmsd_to_reactant=rmsd_to_reactant,
        rmsd_to_product=rmsd_to_product,
    )
