"""Sella IRC with HIP analytical Hessian injected every step.

Identical to the vanilla Sella IRC wiring in `irc_validate.run_irc_validation`
(same `dx=0.1, eta=1e-4, gamma=0.4` knobs, Cartesian coords, same trust
region and convergence checks) — the only change is that the Hessian
stored in Sella's `PES` object is overwritten after every `pes.kick()`
with HIP's analytical Hessian, mass-weighted, Eckart-projected, then
un-mass-weighted back to Cartesian (the recipe from `scripts/sella_baseline.py`).

This isolates the effect of Hessian quality on Sella's IRC: Sella's
default BFGS-updated Hessian is replaced with an exact HIP Hessian at
every single inner-loop iteration of every outer step.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from gadplus.projection import Z_TO_SYMBOL, atomic_nums_to_symbols
from gadplus.projection.projection import (
    _eckart_projector,
    get_mass_weights,
)
from gadplus.search.irc_validate import IRCResult, score_endpoints


class HipSellaCalculator(Calculator):
    """ASE calculator that caches HIP energy/forces/Hessian per position.

    Sella calls ``calculate()`` for energy/forces, then separately calls
    ``hessian_function(atoms)`` for the Hessian. The cache ensures HIP
    runs once per unique geometry even though Sella asks twice.
    """
    implemented_properties = ["energy", "forces"]

    def __init__(self, predict_fn, atomic_nums, device="cuda", **kwargs):
        super().__init__(**kwargs)
        self.predict_fn = predict_fn
        self.atomic_nums = atomic_nums
        self.device = device
        self._cached_coords: Optional[torch.Tensor] = None
        self._cached_result: Optional[dict] = None

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        coords = torch.tensor(self.atoms.positions, dtype=torch.float32, device=self.device)
        out = self.predict_fn(coords, self.atomic_nums, do_hessian=True, require_grad=False)
        self._cached_coords = coords.clone()
        self._cached_result = out

        energy = out["energy"]
        if isinstance(energy, torch.Tensor):
            energy = energy.detach().cpu().item()
        self.results["energy"] = float(energy)

        forces = out["forces"]
        if isinstance(forces, torch.Tensor):
            forces = forces.detach().cpu().numpy()
        self.results["forces"] = np.asarray(forces).reshape(-1, 3)


def _force_first_kick(irc):
    """Monkey-patch IRC so the first convergence check is bypassed.

    ASE's ``Optimizer.irun`` checks ``gradient_converged`` on the starting
    geometry *before* calling ``step()`` — and for Sella's IRC the first
    ``step()`` call contains the kick off the saddle. When the starting
    point is a TS already satisfying ``fmax < halt_threshold`` (e.g. a
    TS refined with Sella's own optimizer at matching fmax), the
    pre-kick check trivially passes and IRC returns immediately without
    ever kicking. This patch forces at least one ``step()`` call by
    returning ``False`` from every convergence check until ``step()``
    has run once.
    """
    has_stepped = {"v": False}
    orig_step = irc.step
    orig_converged = irc.converged
    orig_grad_conv = getattr(irc, "gradient_converged", None)

    def patched_step(*args, **kwargs):
        has_stepped["v"] = True
        return orig_step(*args, **kwargs)

    def patched_converged(*args, **kwargs):
        if not has_stepped["v"]:
            return False
        return orig_converged(*args, **kwargs)

    def patched_grad_conv(*args, **kwargs):
        if not has_stepped["v"]:
            return False
        if orig_grad_conv is not None:
            return orig_grad_conv(*args, **kwargs)
        return orig_converged(*args, **kwargs)

    irc.step = patched_step
    irc.converged = patched_converged
    if orig_grad_conv is not None:
        irc.gradient_converged = patched_grad_conv


def _make_mw_eckart_hessian_function(calc: HipSellaCalculator):
    """Return a callable(atoms)->(3N,3N) Cartesian Hessian that has been
    mass-weighted, Eckart-projected (TR modes removed), and un-mass-weighted
    back to Cartesian. Same recipe as scripts/sella_baseline.py with apply_eckart=True.
    """
    def hessian_function(atoms: Atoms) -> np.ndarray:
        coords = torch.tensor(atoms.positions, dtype=torch.float32, device=calc.device)
        if calc._cached_coords is not None and torch.equal(coords, calc._cached_coords):
            hess = calc._cached_result["hessian"]
        else:
            out = calc.predict_fn(coords, calc.atomic_nums, do_hessian=True, require_grad=False)
            hess = out["hessian"]
            calc._cached_coords = coords.clone()
            calc._cached_result = out

        if isinstance(hess, torch.Tensor):
            hess_t = hess.detach()
        else:
            hess_t = torch.tensor(hess)

        n = len(atoms)
        hess_t = hess_t.reshape(3 * n, 3 * n).to(torch.float64)

        atomsymbols = atomic_nums_to_symbols(calc.atomic_nums)
        coords_3d = coords.reshape(-1, 3).to(torch.float64)
        masses, _m3, sqrt_m, sqrt_m_inv = get_mass_weights(atomsymbols, device=hess_t.device)

        diag_inv = torch.diag(sqrt_m_inv)
        H_mw = diag_inv @ hess_t @ diag_inv

        P = _eckart_projector(coords_3d, masses)
        H_mw_proj = P @ H_mw @ P
        H_mw_proj = 0.5 * (H_mw_proj + H_mw_proj.T)

        diag_m = torch.diag(sqrt_m)
        H_cart = diag_m @ H_mw_proj @ diag_m
        return H_cart.cpu().numpy().astype(np.float64)

    return hessian_function


def _force_hessian_every_kick(pes):
    """Monkey-patch pes.kick to overwrite the BFGS-updated Hessian with
    pes.hessian_function's result after every kick (at the new position).
    """
    original_kick = pes.kick

    def patched_kick(dx, diag=False, **kwargs):
        ratio = original_kick(dx, diag=diag, **kwargs)
        if pes.hessian_function is not None:
            pes.calculate_hessian()
        return ratio

    pes.kick = patched_kick


def run_irc_sella_hip(
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
) -> IRCResult:
    """Sella IRC identical to the baseline, with HIP MW+Eckart Hessian
    injected into pes.H after every inner kick.
    """
    try:
        from sella import IRC
    except ImportError:
        return _irc_error("Sella not installed")

    device = ts_coords.device.type if hasattr(ts_coords, "device") else "cuda"
    coords_np = ts_coords.detach().cpu().numpy().reshape(-1, 3)
    nums = atomic_nums.detach().cpu().tolist()
    symbols = [Z_TO_SYMBOL.get(int(z), "X") for z in nums]

    endpoints: dict[str, Optional[np.ndarray]] = {}
    for direction in ["forward", "reverse"]:
        try:
            atoms = Atoms(symbols=symbols, positions=coords_np)
            calc = HipSellaCalculator(predict_fn=predict_fn, atomic_nums=atomic_nums, device=device)
            atoms.calc = calc
            hess_fn = _make_mw_eckart_hessian_function(calc)

            irc = IRC(
                atoms=atoms,
                dx=dx,
                eta=eta,
                gamma=gamma,
                hessian_function=hess_fn,
            )
            _force_hessian_every_kick(irc.pes)
            _force_first_kick(irc)
            irc.run(fmax=fmax, steps=max_steps, direction=direction)
            endpoints[direction] = atoms.positions.copy()
        except Exception:
            endpoints[direction] = None

    return score_endpoints(
        forward_coords=endpoints.get("forward"),
        reverse_coords=endpoints.get("reverse"),
        atomic_nums=atomic_nums,
        reactant_coords=reactant_coords,
        product_coords=product_coords,
        rmsd_threshold=rmsd_threshold,
        predict_fn=predict_fn,
    )


def _irc_error(msg: str) -> IRCResult:
    return IRCResult(
        intended=False, half_intended=False,
        topology_intended=False, topology_half_intended=False,
        forward_coords=None, reverse_coords=None,
        rmsd_to_reactant=None, rmsd_to_product=None,
        forward_rmsd_to_reactant=None, forward_rmsd_to_product=None,
        reverse_rmsd_to_reactant=None, reverse_rmsd_to_product=None,
        forward_graph_matches_reactant=False, forward_graph_matches_product=False,
        reverse_graph_matches_reactant=False, reverse_graph_matches_product=False,
        error=msg,
    )
