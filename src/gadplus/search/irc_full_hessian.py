"""Sella IRC validation with a backend-supplied full Hessian at every kick."""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from ase import Atoms

from gadplus.calculator.sella import (
    FullHessianASECalculator,
    full_hessian_function,
    refresh_hessian_after_kicks,
)
from gadplus.projection import Z_TO_SYMBOL
from gadplus.search.irc_validate import IRCResult, score_endpoints


def _force_first_kick(irc) -> None:
    """Prevent ASE from accepting a converged saddle before the IRC kick."""
    stepped = {"value": False}
    original_step = irc.step
    original_converged = irc.converged
    original_gradient_converged = getattr(irc, "gradient_converged", None)

    def step(*args, **kwargs):
        stepped["value"] = True
        return original_step(*args, **kwargs)

    def converged(*args, **kwargs):
        if not stepped["value"]:
            return False
        return original_converged(*args, **kwargs)

    def gradient_converged(*args, **kwargs):
        if not stepped["value"]:
            return False
        if original_gradient_converged is not None:
            return original_gradient_converged(*args, **kwargs)
        return original_converged(*args, **kwargs)

    irc.step = step
    irc.converged = converged
    if original_gradient_converged is not None:
        irc.gradient_converged = gradient_converged


def run_irc_full_hessian(
    ts_coords: torch.Tensor,
    atomic_nums: torch.Tensor,
    predict_fn,
    reactant_coords: Optional[torch.Tensor] = None,
    product_coords: Optional[torch.Tensor] = None,
    rmsd_threshold: float = 0.3,
    max_steps: int = 500,
    dx: float = 0.1,
    eta: float = 1e-4,
    gamma: float = 0.4,
    fmax: float = 0.01,
    eckart_project: bool = True,
) -> IRCResult:
    """Run forward and reverse Sella IRC with the supplied full Hessian.

    This is intended for PES-native reference-set validation. It does not use
    either GAD or Sella's saddle-search path to create the candidate saddle.
    """
    try:
        from sella import IRC
    except ImportError:
        return score_endpoints(
            None, None, atomic_nums, reactant_coords, product_coords,
            rmsd_threshold, error="Sella not installed", predict_fn=predict_fn,
        )

    device = str(ts_coords.device) if ts_coords.is_cuda else "cpu"
    coords_np = ts_coords.detach().cpu().numpy().reshape(-1, 3)
    symbols = [Z_TO_SYMBOL.get(int(z), "X") for z in atomic_nums.detach().cpu().tolist()]
    endpoints: dict[str, Optional[np.ndarray]] = {"forward": None, "reverse": None}

    for direction in ("forward", "reverse"):
        try:
            atoms = Atoms(symbols=symbols, positions=coords_np)
            calculator = FullHessianASECalculator(predict_fn, atomic_nums, device)
            atoms.calc = calculator
            irc = IRC(
                atoms=atoms,
                dx=dx,
                eta=eta,
                gamma=gamma,
                hessian_function=full_hessian_function(
                    calculator, eckart_project=eckart_project,
                ),
            )
            refresh_hessian_after_kicks(irc.pes)
            _force_first_kick(irc)
            irc.run(fmax=fmax, steps=max_steps, direction=direction)
            endpoints[direction] = atoms.positions.copy()
        except Exception as exc:
            print(f"[full-Hessian IRC {direction} failed] {type(exc).__name__}: {exc}", flush=True)

    return score_endpoints(
        endpoints["forward"], endpoints["reverse"], atomic_nums,
        reactant_coords, product_coords, rmsd_threshold, predict_fn=predict_fn,
    )
