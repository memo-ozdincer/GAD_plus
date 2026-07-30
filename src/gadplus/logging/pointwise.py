"""Exact local trajectory bundles for pointwise transition-state searches.

The recorder is deliberately independent of W&B. It receives detached
observations, writes append-free exact files at the end of a run, and can be
exported repeatedly without rerunning a calculator.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from gadplus.projection import atomic_nums_to_symbols, get_mass_weights
from gadplus.search.intrinsic_gad import IntrinsicGADObservation, IntrinsicGADResult


def _json_value(value: Any) -> Any:
    if is_dataclass(value):
        return {key: _json_value(item) for key, item in asdict(value).items()}
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


class IntrinsicTrajectoryRecorder:
    """Callable recorder adapter for :func:`run_intrinsic_gad`.

    Parameters are descriptive metadata only. They do not enter an optimizer
    update and can be changed without changing the recorded coordinates.
    """

    def __init__(
        self,
        output_root: str | Path,
        run_id: str,
        atomic_numbers: torch.Tensor,
        *,
        config: Mapping[str, Any] | None = None,
        reference_coordinates: Mapping[str, torch.Tensor] | None = None,
    ) -> None:
        self.bundle_dir = Path(output_root).resolve() / run_id
        self.run_id = str(run_id)
        self.atomic_numbers = atomic_numbers.detach().cpu().to(torch.int64).reshape(-1)
        symbols = atomic_nums_to_symbols(self.atomic_numbers)
        masses, _, _, _ = get_mass_weights(symbols, device="cpu")
        self.masses = masses.detach().cpu().to(torch.float64)
        self.config = dict(config or {})
        self.reference_coordinates = {
            str(name): coords.detach().cpu().to(torch.float64).reshape(-1, 3).numpy().copy()
            for name, coords in (reference_coordinates or {}).items()
        }
        self.rows: list[dict[str, Any]] = []
        self.coordinates: list[np.ndarray] = []
        self._start_coords: torch.Tensor | None = None
        self._previous_coords: torch.Tensor | None = None
        self._cumulative_path = 0.0

    def __call__(self, observation: IntrinsicGADObservation) -> None:
        coords = observation.coords.to(torch.float64).reshape(-1, 3)
        forces = observation.forces.to(torch.float64).reshape(-1, 3)
        evals = observation.eigenvalues.to(torch.float64).reshape(-1)
        if self._start_coords is None:
            self._start_coords = coords.clone()
        if self._previous_coords is None:
            incoming_rms = 0.0
        else:
            incoming = coords - self._previous_coords
            incoming_rms = float(torch.sqrt(torch.mean(torch.sum(incoming.square(), dim=1))))
            self._cumulative_path += incoming_rms
        displacement = coords - self._start_coords
        displacement_from_start = float(
            torch.sqrt(torch.mean(torch.sum(displacement.square(), dim=1)))
        )

        distances = torch.pdist(coords)
        centered = coords - torch.sum(coords * self.masses[:, None], dim=0) / self.masses.sum()
        radius_of_gyration = torch.sqrt(
            torch.sum(self.masses * torch.sum(centered.square(), dim=1)) / self.masses.sum()
        )
        spectral_scale = max(observation.spectral_scale, torch.finfo(torch.float64).tiny)
        eigenvalues = [float(value) for value in evals[:5].tolist()]
        eigenvalues += [math.nan] * (5 - len(eigenvalues))

        if observation.step_cart is None:
            step_cart_rms = math.nan
            max_atom_displacement = math.nan
        else:
            outgoing = observation.step_cart.to(torch.float64).reshape(-1, 3)
            outgoing_norms = torch.linalg.vector_norm(outgoing, dim=1)
            step_cart_rms = float(torch.sqrt(torch.mean(outgoing_norms.square())))
            max_atom_displacement = float(outgoing_norms.max())

        weights = observation.low_mode_weights.to(torch.float64).reshape(-1)
        row: dict[str, Any] = {
            "evaluation": int(observation.evaluation),
            "iteration": int(observation.iteration),
            "wall_time_s": float(observation.wall_time_s),
            "energy": float(observation.energy),
            "force_max": float(forces.abs().max()),
            "force_rms": float(torch.sqrt(torch.mean(forces.square()))),
            "force_mean_atom": float(torch.linalg.vector_norm(forces, dim=1).mean()),
            "gradient_norm": float(torch.linalg.vector_norm(forces)),
            "n_neg": int(observation.n_neg),
            "spectral_scale": float(observation.spectral_scale),
            "lambda1": eigenvalues[0],
            "lambda2": eigenvalues[1],
            "lambda3": eigenvalues[2],
            "lambda4": eigenvalues[3],
            "lambda5": eigenvalues[4],
            "lambda1_scaled": eigenvalues[0] / spectral_scale,
            "lambda2_scaled": eigenvalues[1] / spectral_scale,
            "lambda3_scaled": eigenvalues[2] / spectral_scale,
            "closest_pair": float(distances.min()),
            "maximum_pair": float(distances.max()),
            "radius_of_gyration": float(radius_of_gyration),
            "incoming_step_cart_rms": incoming_rms,
            "displacement_from_start": displacement_from_start,
            "cumulative_path": self._cumulative_path,
            "step_cart_rms": step_cart_rms,
            "max_atom_displacement": max_atom_displacement,
            "step_mw_rms": observation.step_mw_rms,
            "geometric_length": observation.geometric_length,
            "local_radius": observation.local_radius,
            "step_over_length": (
                step_cart_rms / observation.geometric_length
                if observation.geometric_length and math.isfinite(step_cart_rms)
                else math.nan
            ),
            "step_over_radius": (
                observation.step_mw_rms / observation.local_radius
                if observation.local_radius and observation.step_mw_rms is not None
                else math.nan
            ),
            "regularizer": observation.regularizer,
            "regularizer_scaled": (
                observation.regularizer / spectral_scale
                if observation.regularizer is not None
                else math.nan
            ),
            "lambda2_gate": float(observation.lambda2_gate),
            "effective_gate": float(observation.effective_gate),
            "soft_activity": float(observation.soft_activity),
            "extra_negative_activity": float(observation.extra_negative_activity),
            "activity_fraction": float(observation.activity_fraction),
            "activity_log10_ratio": math.log10(
                max(observation.soft_activity, torch.finfo(torch.float64).tiny)
                / max(observation.extra_negative_activity, torch.finfo(torch.float64).tiny)
            ),
            "p1": float(weights[0]) if weights.numel() else math.nan,
            "p2": float(weights[1]) if weights.numel() > 1 else math.nan,
            "spectral_entropy": float(observation.spectral_entropy),
            "lowest_reflection": float(observation.lowest_reflection),
            "terminal": bool(observation.terminal),
        }
        self.rows.append(row)
        self.coordinates.append(coords.numpy().copy())
        self._previous_coords = coords.clone()

    def flush(
        self,
        result: IntrinsicGADResult,
        *,
        summary: Mapping[str, Any] | None = None,
    ) -> Path:
        """Write the exact trajectory bundle and return its directory."""

        if not self.rows:
            raise ValueError("cannot flush an empty trajectory")
        self.bundle_dir.mkdir(parents=True, exist_ok=True)
        table = pa.Table.from_pylist(self.rows)
        pq.write_table(table, self.bundle_dir / "trajectory.parquet", compression="zstd")
        np.savez_compressed(
            self.bundle_dir / "coordinates.npz",
            coordinates=np.stack(self.coordinates),
            atomic_numbers=self.atomic_numbers.numpy(),
            masses=self.masses.numpy(),
            **self.reference_coordinates,
        )
        metadata = {
            "schema_version": 1,
            "run_id": self.run_id,
            "config": _json_value(self.config),
            "summary": {
                "converged": result.converged,
                "converged_step": result.converged_step,
                "total_steps": result.total_steps,
                "n_evaluations": result.n_evaluations,
                "final_energy": result.final_energy,
                "final_n_neg": result.final_n_neg,
                "final_force_norm": result.final_force_norm,
                "final_force_max": result.final_force_max,
                "final_eig0": result.final_eig0,
                "final_eig1": result.final_eig1,
                "final_gate_weight": result.final_gate_weight,
                "wall_time_s": result.wall_time_s,
                "failure_type": result.failure_type,
                **_json_value(dict(summary or {})),
            },
        }
        with (self.bundle_dir / "metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        return self.bundle_dir


__all__ = ["IntrinsicTrajectoryRecorder"]
