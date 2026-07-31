"""g-xTB calculator adapter.

The g-xTB distribution currently ships a standalone ``xtb`` executable (with
the modified tblite engine), not a Python calculator API.  This adapter keeps
the surface compatible with the rest of GADplus by using a private temporary
working directory for each single-point and parsing the standard ``energy``,
``gradient`` and ``hessian`` files.

Input coordinates are Angstrom and output tensors use the project convention:
energy in eV, forces in eV/Angstrom, and Hessians in eV/Angstrom^2.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict

import torch

from gadplus.core.types import PredictFn


HARTREE_TO_EV = 27.211386245988
BOHR_TO_ANG = 0.529177210903
GRAD_HA_BOHR_TO_EV_ANG = HARTREE_TO_EV / BOHR_TO_ANG
HESS_HA_BOHR2_TO_EV_ANG2 = HARTREE_TO_EV / (BOHR_TO_ANG**2)

_SYMBOLS = (
    "", "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
    "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
    "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
    "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm",
    "Md", "No", "Lr",
)


def _floats(text: str) -> list[float]:
    return [float(x) for x in re.findall(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[EeDd][-+]?\d+)?", text.replace("D", "E"))]


def _read_energy(path: Path) -> float:
    values = _floats(path.read_text())
    if not values:
        raise RuntimeError(f"g-xTB energy file is empty: {path}")
    # The energy file contains one row with three equivalent energy values.
    return values[-1]


def _read_gradient(path: Path, n_atoms: int) -> torch.Tensor:
    text = path.read_text()
    body = text.split("$grad", 1)[-1].split("$end", 1)[0]
    values = _floats(body)
    # The first numeric values are cycle/SCF metadata; the final 3N values
    # are the Cartesian gradient block, which is stable across xtb versions.
    if len(values) < 3 * n_atoms:
        raise RuntimeError(f"g-xTB gradient has too few values: {path}")
    return torch.tensor(values[-3 * n_atoms:], dtype=torch.float64).reshape(n_atoms, 3)


def _read_hessian(path: Path, n_atoms: int) -> torch.Tensor:
    text = path.read_text()
    body = text.split("$hessian", 1)[-1].split("$end", 1)[0]
    dim = 3 * n_atoms
    values = _floats(body)
    # xTB's ``$hessian`` block begins with its matrix dimension.  It is a
    # header, not H[0, 0], and must not shift the Cartesian matrix.
    if values and int(values[0]) == dim:
        values = values[1:]
    if len(values) < dim * dim:
        raise RuntimeError(
            f"g-xTB Hessian has {len(values)} values; expected {dim * dim}: {path}"
        )
    return torch.tensor(values[: dim * dim], dtype=torch.float64).reshape(dim, dim)


class GxtbCalculator:
    """Single-point g-xTB calculator using the cloned binary distribution."""

    def __init__(
        self,
        executable: str | os.PathLike[str] = "g-xtb/xtb-6.7.1/bin/xtb",
        charge: int = 0,
        uhf: int | None = None,
        n_threads: int | None = None,
        parallel: int | None = None,
        timeout_s: float = 600.0,
    ):
        # Resolve before entering the per-call temporary directory; a relative
        # path such as the repository's bundled binary must remain usable.
        self.executable = str(Path(executable).expanduser().resolve())
        if not Path(self.executable).is_file():
            raise FileNotFoundError(f"g-xTB executable not found: {self.executable}")
        self.charge = int(charge)
        self.uhf = uhf
        self.n_threads = n_threads
        self.parallel = parallel
        self.timeout_s = float(timeout_s)

    def compute(
        self,
        coords: torch.Tensor,
        atomic_nums: torch.Tensor,
        do_hessian: bool = True,
        require_grad: bool = False,
    ) -> Dict[str, torch.Tensor]:
        if require_grad:
            raise NotImplementedError("g-xTB is an external executable, not autograd-compatible")
        xyz = coords.detach().to(dtype=torch.float64, device="cpu").reshape(-1, 3)
        z = atomic_nums.detach().to(device="cpu", dtype=torch.long).reshape(-1)
        if xyz.shape[0] != z.shape[0]:
            raise ValueError("coords and atomic_nums have inconsistent atom counts")
        if any(int(a) <= 0 or int(a) >= len(_SYMBOLS) for a in z):
            raise ValueError("g-xTB adapter only accepts atomic numbers 1..103")

        keep_workdir = os.environ.get("GADPLUS_GXTB_KEEP_WORKDIR", "").lower() in {
            "1", "true", "yes",
        }
        work_root = os.environ.get("GADPLUS_GXTB_WORK_ROOT")
        if work_root:
            Path(work_root).mkdir(parents=True, exist_ok=True)
        workdir = Path(tempfile.mkdtemp(prefix="gadplus_gxtb_", dir=work_root))
        try:
            xyz_path = workdir / "structure.xyz"
            lines = [str(len(z)), "GADplus g-xTB single point"]
            lines.extend(
                f"{_SYMBOLS[int(a)]} {p[0].item():.12f} {p[1].item():.12f} {p[2].item():.12f}"
                for a, p in zip(z, xyz)
            )
            xyz_path.write_text("\n".join(lines) + "\n")

            args = [self.executable, "structure.xyz", "--gxtb", "--silent", "--chrg", str(self.charge)]
            if self.parallel is not None:
                # A numerical Cartesian Hessian has at most 3N independent
                # gradient displacements.  Giving xTB more workers than that
                # merely adds launch/synchronisation overhead on small T1x
                # molecules; retain the requested value as a cap for larger
                # systems.
                parallel = min(int(self.parallel), 3 * len(z))
                args.extend(["--parallel", str(parallel)])
            if self.uhf is not None:
                args.extend(["--uhf", str(int(self.uhf))])
            env = os.environ.copy()
            if self.n_threads is not None:
                env["OMP_NUM_THREADS"] = str(int(self.n_threads))
            def run(flag: str) -> str:
                proc = subprocess.run(
                    [*args, flag], cwd=workdir, env=env, stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT, text=True, timeout=self.timeout_s, check=False,
                )
                if proc.returncode != 0:
                    raise RuntimeError(f"g-xTB {flag} failed with code {proc.returncode}:\n{proc.stdout[-4000:]}")
                return proc.stdout

            hessian = None
            if do_hessian:
                # ``--grad --hess`` silently omits the Hessian in this release.
                # ``--hess`` supplies the direct numerical Hessian but does not
                # reliably retain energy/Cartesian-gradient files, so read it
                # before the separate gradient single point below.
                hess_output = run("--hess")
                hess_path = workdir / "hessian"
                if not hess_path.is_file():
                    produced = sorted(path.name for path in workdir.iterdir())
                    retained = f"; retained workdir={workdir}" if keep_workdir else ""
                    raise RuntimeError(
                        "g-xTB --hess returned success but did not create hessian; "
                        f"produced={produced}{retained}; output:\n{hess_output[-4000:]}"
                    )
                hessian = _read_hessian(hess_path, len(z)) * HESS_HA_BOHR2_TO_EV_ANG2

            grad_output = run("--grad")
            required = [workdir / "energy", workdir / "gradient"]
            missing = [path.name for path in required if not path.is_file()]
            if missing:
                produced = sorted(path.name for path in workdir.iterdir())
                retained = f"; retained workdir={workdir}" if keep_workdir else ""
                raise RuntimeError(
                    "g-xTB --grad returned success but did not create "
                    f"{missing}; produced={produced}{retained}; output:\n{grad_output[-4000:]}"
                )

            energy = _read_energy(workdir / "energy") * HARTREE_TO_EV
            gradient = _read_gradient(workdir / "gradient", len(z))
            result: Dict[str, torch.Tensor] = {
                "energy": torch.tensor(energy, dtype=torch.float64),
                "forces": -gradient * GRAD_HA_BOHR_TO_EV_ANG,
            }
            if do_hessian:
                result["hessian"] = hessian
            return result
        finally:
            if not keep_workdir:
                shutil.rmtree(workdir, ignore_errors=True)


def make_gxtb_predict_fn(calculator: GxtbCalculator) -> PredictFn:
    """Adapt :class:`GxtbCalculator` to the project PredictFn protocol."""

    def _predict(coords: torch.Tensor, atomic_nums: torch.Tensor, *, do_hessian: bool = True,
                 require_grad: bool = False) -> Dict[str, Any]:
        result = calculator.compute(coords, atomic_nums, do_hessian=do_hessian, require_grad=require_grad)
        target_device, target_dtype = coords.device, coords.dtype
        result["energy"] = result["energy"].to(device=target_device, dtype=target_dtype)
        result["forces"] = result["forces"].to(device=target_device, dtype=target_dtype)
        if "hessian" in result:
            result["hessian"] = result["hessian"].to(device=target_device)
        return result

    return _predict


def load_gxtb_calculator(**kwargs) -> GxtbCalculator:
    return GxtbCalculator(**kwargs)
