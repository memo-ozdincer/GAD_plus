"""Calculator-native endpoint labels for IRC validation.

Cross-calculator IRC validation should distinguish two questions:

1. Does a candidate saddle connect the Transition1x reference endpoints?
2. Does it connect the minima reached from those labeled endpoints on the
   calculator being evaluated?

This module implements the second question.  It minimizes the labeled
reactant/product with the active calculator, caches the resulting minima,
and returns enough diagnostics to keep collapsed or unconverged reference
labels visible to downstream scoring.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch


CACHE_SCHEMA_VERSION = 1


@dataclass
class RelaxationResult:
    """A local-minimum relaxation result in Cartesian Angstrom coordinates."""

    coords: np.ndarray
    converged: bool
    steps: int
    force_max: float
    energy: float
    error: Optional[str] = None


@dataclass
class NativeEndpointLabels:
    """Calculator-relaxed reactant/product labels for one sample."""

    reactant: Optional[RelaxationResult]
    product: Optional[RelaxationResult]
    cache_path: Path
    cache_hit: bool


def _as_coords(coords: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(coords, torch.Tensor):
        coords = coords.detach().cpu().numpy()
    return np.ascontiguousarray(np.asarray(coords, dtype=np.float64).reshape(-1, 3))


def _as_atomic_nums(atomic_nums: torch.Tensor | np.ndarray | list[int]) -> np.ndarray:
    if isinstance(atomic_nums, torch.Tensor):
        atomic_nums = atomic_nums.detach().cpu().numpy()
    return np.ascontiguousarray(np.asarray(atomic_nums, dtype=np.int64).reshape(-1))


def _cache_key(
    *,
    functional: str,
    atomic_nums: np.ndarray,
    reactant_coords: np.ndarray,
    product_coords: Optional[np.ndarray],
    relax_fmax: float,
    max_steps: int,
) -> str:
    digest = hashlib.sha256()
    digest.update(f"native-endpoints-v{CACHE_SCHEMA_VERSION}".encode())
    digest.update(functional.upper().encode())
    digest.update(np.asarray([relax_fmax], dtype=np.float64).tobytes())
    digest.update(np.asarray([max_steps], dtype=np.int64).tobytes())
    digest.update(atomic_nums.tobytes())
    digest.update(reactant_coords.tobytes())
    if product_coords is None:
        digest.update(b"no-product")
    else:
        digest.update(product_coords.tobytes())
    return digest.hexdigest()[:20]


def _cache_path(
    cache_dir: Path,
    *,
    functional: str,
    sample_id: int,
    key: str,
) -> Path:
    return cache_dir / functional.upper() / f"sample_{sample_id:04d}_{key}.npz"


def relax_to_minimum(
    coords: np.ndarray | torch.Tensor,
    atomic_nums: torch.Tensor | np.ndarray | list[int],
    predict_fn: Callable,
    *,
    fmax: float,
    max_steps: int,
) -> RelaxationResult:
    """Relax one geometry with ASE BFGS through a PredictFn calculator."""
    from ase import Atoms
    from ase.optimize import BFGS

    from gadplus.calculator.ase_adapter import HipASECalculator

    coords_np = _as_coords(coords)
    nums_np = _as_atomic_nums(atomic_nums)
    z = torch.as_tensor(nums_np, dtype=torch.long)
    try:
        atoms = Atoms(numbers=nums_np.tolist(), positions=coords_np)
        atoms.calc = HipASECalculator(predict_fn=predict_fn, atomic_nums=z)
        optimizer = BFGS(atoms, logfile=None)
        converged = bool(optimizer.run(fmax=fmax, steps=max_steps))
        forces = np.asarray(atoms.get_forces(), dtype=np.float64)
        force_max = float(np.abs(forces).max()) if forces.size else 0.0
        energy = float(atoms.get_potential_energy())
        return RelaxationResult(
            coords=np.asarray(atoms.positions, dtype=np.float64).copy(),
            converged=converged,
            steps=int(optimizer.nsteps),
            force_max=force_max,
            energy=energy,
        )
    except Exception as exc:
        return RelaxationResult(
            coords=coords_np.copy(),
            converged=False,
            steps=0,
            force_max=float("inf"),
            energy=float("nan"),
            error=repr(exc),
        )


def _pack_result(prefix: str, result: Optional[RelaxationResult]) -> dict[str, np.ndarray]:
    if result is None:
        return {
            f"{prefix}_present": np.asarray(False),
            f"{prefix}_coords": np.empty((0, 3), dtype=np.float64),
            f"{prefix}_converged": np.asarray(False),
            f"{prefix}_steps": np.asarray(0, dtype=np.int64),
            f"{prefix}_force_max": np.asarray(np.nan, dtype=np.float64),
            f"{prefix}_energy": np.asarray(np.nan, dtype=np.float64),
            f"{prefix}_error": np.asarray(""),
        }
    return {
        f"{prefix}_present": np.asarray(True),
        f"{prefix}_coords": result.coords,
        f"{prefix}_converged": np.asarray(result.converged),
        f"{prefix}_steps": np.asarray(result.steps, dtype=np.int64),
        f"{prefix}_force_max": np.asarray(result.force_max, dtype=np.float64),
        f"{prefix}_energy": np.asarray(result.energy, dtype=np.float64),
        f"{prefix}_error": np.asarray(result.error or ""),
    }


def _scalar(value: np.ndarray) -> object:
    return value.reshape(()).item()


def _unpack_result(data: np.lib.npyio.NpzFile, prefix: str) -> Optional[RelaxationResult]:
    if not bool(_scalar(data[f"{prefix}_present"])):
        return None
    error = str(_scalar(data[f"{prefix}_error"])) or None
    return RelaxationResult(
        coords=np.asarray(data[f"{prefix}_coords"], dtype=np.float64),
        converged=bool(_scalar(data[f"{prefix}_converged"])),
        steps=int(_scalar(data[f"{prefix}_steps"])),
        force_max=float(_scalar(data[f"{prefix}_force_max"])),
        energy=float(_scalar(data[f"{prefix}_energy"])),
        error=error,
    )


def _load_cache(path: Path) -> NativeEndpointLabels:
    with np.load(path, allow_pickle=False) as data:
        if int(_scalar(data["schema_version"])) != CACHE_SCHEMA_VERSION:
            raise ValueError("native-endpoint cache schema version mismatch")
        return NativeEndpointLabels(
            reactant=_unpack_result(data, "reactant"),
            product=_unpack_result(data, "product"),
            cache_path=path,
            cache_hit=True,
        )


def _write_cache(path: Path, labels: NativeEndpointLabels) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.stem}.{os.getpid()}.tmp.npz")
    payload = {
        "schema_version": np.asarray(CACHE_SCHEMA_VERSION, dtype=np.int64),
        **_pack_result("reactant", labels.reactant),
        **_pack_result("product", labels.product),
    }
    np.savez_compressed(tmp, **payload)
    os.replace(tmp, path)


def load_or_relax_native_endpoints(
    *,
    cache_dir: str | Path,
    sample_id: int,
    functional: str,
    atomic_nums: torch.Tensor | np.ndarray | list[int],
    reactant_coords: np.ndarray | torch.Tensor,
    product_coords: Optional[np.ndarray | torch.Tensor],
    predict_fn: Callable,
    relax_fmax: float = 0.001,
    max_steps: int = 500,
) -> NativeEndpointLabels:
    """Load or generate calculator-native labels for a T1x R/P pair.

    The cache key includes the initial labeled coordinates and relaxation
    settings, so reruns cannot silently reuse labels from a different
    calculator configuration or source geometry.
    """
    cache_root = Path(cache_dir)
    nums_np = _as_atomic_nums(atomic_nums)
    reactant_np = _as_coords(reactant_coords)
    product_np = _as_coords(product_coords) if product_coords is not None else None
    key = _cache_key(
        functional=functional,
        atomic_nums=nums_np,
        reactant_coords=reactant_np,
        product_coords=product_np,
        relax_fmax=relax_fmax,
        max_steps=max_steps,
    )
    path = _cache_path(
        cache_root,
        functional=functional,
        sample_id=sample_id,
        key=key,
    )
    if path.exists():
        try:
            return _load_cache(path)
        except Exception:
            path.unlink(missing_ok=True)

    labels = NativeEndpointLabels(
        reactant=relax_to_minimum(
            reactant_np, nums_np, predict_fn, fmax=relax_fmax, max_steps=max_steps,
        ),
        product=(
            relax_to_minimum(
                product_np, nums_np, predict_fn, fmax=relax_fmax, max_steps=max_steps,
            )
            if product_np is not None
            else None
        ),
        cache_path=path,
        cache_hit=False,
    )
    _write_cache(path, labels)
    return labels
