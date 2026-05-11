"""xTB calculator adapter (Grimme-lab GFN family).

Uses the ``dxtb`` package — Grimme group's fully differentiable PyTorch
implementation of GFN1-xTB and GFN2-xTB. Analytic energy, forces, and
Hessian; runs on CPU by default. Unlike xtb-python, dxtb is autograd-
compatible, so this backend supports ``require_grad=True``.

Units: dxtb returns energy in Hartree, forces in Hartree/Bohr, Hessian
in Hartree/Bohr^2. We convert to eV / eV-Angstrom / eV-Angstrom^2 to
match the HIP backend.

Note: GFN-FF and GFN0-xTB are NOT in dxtb (dxtb only ships GFN1 and GFN2).
For those methods, use the SCINE xtb_wrapper (separate backend, not
implemented here).
"""
from __future__ import annotations

from typing import Any, Dict

import torch

from gadplus.core.types import PredictFn


HARTREE_TO_EV = 27.211386245988
BOHR_TO_ANG = 0.529177210903
ANG_TO_BOHR = 1.0 / BOHR_TO_ANG


_METHOD_ALIASES = {
    "gfn1": "GFN1Calculator",
    "gfn1-xtb": "GFN1Calculator",
    "gfn1xtb": "GFN1Calculator",
    "gfn2": "GFN2Calculator",
    "gfn2-xtb": "GFN2Calculator",
    "gfn2xtb": "GFN2Calculator",
}


class XtbCalculator:
    """Persistent dxtb calculator: the underlying dxtb object is rebuilt
    per geometry because atomic numbers are baked in at construction.

    Args:
        method: "gfn1" or "gfn2" (default GFN2-xTB).
        device: Torch device string. dxtb runs on CPU; cuda is supported
            but rarely faster for the molecule sizes used here.
        accuracy: dxtb SCF tolerance multiplier (smaller = tighter).
        electronic_temperature: Fermi smearing in Kelvin.
    """

    def __init__(
        self,
        method: str = "gfn2",
        device: str = "cpu",
        accuracy: float = 1.0,
        electronic_temperature: float = 300.0,
        **_,
    ):
        import dxtb  # noqa: F401
        from dxtb import calculators as _dxtb_calcs

        key = method.lower().replace("_", "")
        if key not in _METHOD_ALIASES:
            raise ValueError(
                f"Unknown xtb method '{method}'. dxtb supports: gfn1, gfn2"
            )
        self._calc_cls = getattr(_dxtb_calcs, _METHOD_ALIASES[key])
        self.method = method
        self.device_str = device
        self.device = torch.device(device)
        self.dtype = torch.float64
        # dxtb's SCF tolerance is set via opts; keep verbosity quiet.
        # Keys: verbosity is popped by BaseCalculator; f_atol/x_atol/
        # fermi_etemp are forwarded to dxtb.config.Config.
        self.opts = {
            "verbosity": 0,
            "f_atol": 1e-6 * float(accuracy),
            "x_atol": 1e-6 * float(accuracy),
            "fermi_etemp": float(electronic_temperature),
        }
        # dxtb constructs internal state from `numbers` only; we cache the
        # most recent calculator so consecutive same-molecule calls reuse it.
        self._cached_numbers_key = None
        self._cached_calc = None

    def _get_calc(self, atomic_nums: torch.Tensor):
        numbers = atomic_nums.detach().to(device=self.device, dtype=torch.long)
        key = tuple(numbers.cpu().tolist())
        if key != self._cached_numbers_key:
            self._cached_calc = self._calc_cls(
                numbers,
                opts=self.opts,
                dtype=self.dtype,
                device=self.device,
            )
            self._cached_numbers_key = key
        return self._cached_calc

    def _fresh_pos(self, coords_ang: torch.Tensor) -> torch.Tensor:
        """Build a fresh requires_grad position tensor in Bohr.

        dxtb consumes autograd graphs on each derivative call, so each
        ``get_*`` invocation needs its own leaf tensor.
        """
        return (coords_ang * ANG_TO_BOHR).clone().detach().requires_grad_(True)

    def compute(
        self,
        coords: torch.Tensor,
        atomic_nums: torch.Tensor,
        do_hessian: bool = True,
        require_grad: bool = False,
    ) -> Dict[str, torch.Tensor]:
        calc = self._get_calc(atomic_nums)

        # dxtb runs in float64 with requires_grad positions. require_grad=True
        # from the caller is currently not wired all the way through (would
        # need a single shared leaf tensor for E, F, H); the search loops
        # never call xtb with require_grad=True, so we leave that as a TODO.
        coords_ang = coords.detach().to(device=self.device, dtype=self.dtype)

        # Energy uses its own leaf tensor so the graph is independent of the
        # forces/hessian graphs (dxtb releases each graph after .backward()).
        pos_e = self._fresh_pos(coords_ang)
        energy_hartree = calc.get_energy(pos_e).detach()

        pos_f = self._fresh_pos(coords_ang)
        forces_hb = calc.get_forces(pos_f).detach()

        out: Dict[str, torch.Tensor] = {
            "energy": energy_hartree * HARTREE_TO_EV,
            "forces": forces_hb * (HARTREE_TO_EV / BOHR_TO_ANG),
        }

        if do_hessian:
            pos_h = self._fresh_pos(coords_ang)
            hess_hb2 = calc.get_hessian(pos_h).detach()
            # dxtb's Hessian is (N, 3, N, 3); flatten to (3N, 3N) in
            # row-major-by-atom order matching everything else in this repo.
            if hess_hb2.dim() == 4:
                N = hess_hb2.shape[0]
                hess_hb2 = hess_hb2.reshape(3 * N, 3 * N)
            out["hessian"] = hess_hb2 * (HARTREE_TO_EV / (BOHR_TO_ANG ** 2))

        return out


def make_xtb_predict_fn(calculator: XtbCalculator) -> PredictFn:
    """Adapt an XtbCalculator to the PredictFn protocol."""

    def _predict(
        coords: torch.Tensor,
        atomic_nums: torch.Tensor,
        *,
        do_hessian: bool = True,
        require_grad: bool = False,
    ) -> Dict[str, Any]:
        if require_grad:
            raise NotImplementedError(
                "xtb (dxtb) backend does not yet support require_grad=True end-to-end; "
                "use HIP for the differentiable code path."
            )
        result = calculator.compute(
            coords, atomic_nums,
            do_hessian=do_hessian, require_grad=False,
        )
        # Cast back to caller's dtype/device for energy & forces; keep
        # Hessian in float64 (search code expects high-precision eigh).
        target_device = coords.device
        target_dtype = coords.dtype
        result["energy"] = result["energy"].to(device=target_device, dtype=target_dtype)
        result["forces"] = result["forces"].to(device=target_device, dtype=target_dtype)
        if "hessian" in result:
            result["hessian"] = result["hessian"].to(device=target_device)
        return result

    return _predict


def load_xtb_calculator(
    method: str = "gfn2",
    device: str = "cpu",
    accuracy: float = 1.0,
    electronic_temperature: float = 300.0,
    **kwargs,
) -> XtbCalculator:
    """Construct an xTB calculator backed by dxtb.

    Args:
        method: "gfn1" or "gfn2" (default GFN2-xTB).
        device: Torch device for dxtb tensors. CPU is the practical default.
        accuracy: SCF tolerance multiplier (1.0 = dxtb defaults).
        electronic_temperature: Fermi smearing in Kelvin.
    """
    return XtbCalculator(
        method=method,
        device=device,
        accuracy=accuracy,
        electronic_temperature=electronic_temperature,
        **kwargs,
    )
