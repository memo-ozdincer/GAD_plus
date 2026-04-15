"""Per-step trajectory logging with Parquet output.

TrajectoryLogger accumulates one dict per optimization step, computing every
metric in TRAJECTORY_SCHEMA from raw tensors.  Call ``flush()`` to write to
disk as a Parquet file.

Usage::

    logger = TrajectoryLogger(output_dir, run_id, sample_id, ...)
    for step in range(max_steps):
        ...
        logger.log_step(step, phase="gad", dt_eff=dt, energy=E,
                        forces=F, evals_vib=evals, evecs_vib=evecs,
                        coords=x, coords_start=x0, coords_prev=x_prev,
                        v_prev=v_prev, known_ts_coords=ts_ref, grad=g)
    path = logger.flush()
"""
from __future__ import annotations

import os
import time
from typing import Optional

import torch
from torch import Tensor

from gadplus.core.convergence import (
    compute_cascade_n_neg,
    compute_eigenvalue_bands,
)
from gadplus.logging.schema import TRAJECTORY_SCHEMA


def _safe_float(x: Tensor | float | None) -> float | None:
    """Extract a Python float from a scalar tensor, pass through floats/None."""
    if x is None:
        return None
    if isinstance(x, Tensor):
        return float(x.item())
    return float(x)


def _force_norm(forces: Tensor) -> float:
    """Mean per-atom force norm (eV/A)."""
    f = forces.detach().reshape(-1, 3)
    return float(f.norm(dim=1).mean().item())


def _force_rms(forces: Tensor) -> float:
    """RMS of all force components."""
    f = forces.detach().reshape(-1)
    return float(f.pow(2).mean().sqrt().item())


def _force_max(forces: Tensor) -> float:
    """Max absolute Cartesian force component (fmax)."""
    f = forces.detach().reshape(-1)
    return float(f.abs().max().item())


def _displacement(a: Tensor, b: Tensor) -> float:
    """RMSD between two coordinate tensors (A)."""
    diff = (a.detach().reshape(-1, 3) - b.detach().reshape(-1, 3))
    return float(diff.pow(2).sum(dim=1).mean().sqrt().item())


def _overlap(u: Tensor, v: Tensor) -> float:
    """Absolute cosine similarity between two flat vectors."""
    u_flat = u.detach().reshape(-1).double()
    v_flat = v.detach().reshape(-1).double()
    norm_u = u_flat.norm()
    norm_v = v_flat.norm()
    if norm_u < 1e-15 or norm_v < 1e-15:
        return 0.0
    return float((u_flat @ v_flat).abs() / (norm_u * norm_v))


# Map cascade threshold floats to schema column names
_CASCADE_KEY_MAP = {
    "n_neg_0.0": "n_neg_0",
    "n_neg_0.0001": "n_neg_1e4",
    "n_neg_0.0005": "n_neg_5e4",
    "n_neg_0.001": "n_neg_1e3",
    "n_neg_0.002": "n_neg_2e3",
    "n_neg_0.005": "n_neg_5e3",
    "n_neg_0.008": "n_neg_8e3",
    "n_neg_0.01": "n_neg_1e2",
}


class TrajectoryLogger:
    """Accumulate per-step metrics and flush to Parquet.

    All heavy computation (norms, overlaps, cascade, bands) happens inside
    ``log_step`` so the caller only needs to pass raw tensors.

    Parameters
    ----------
    output_dir : str
        Directory where parquet files are written.
    run_id : str
        Unique identifier for this run (e.g. a UUID or timestamp tag).
    sample_id : int
        Index of the sample within the run.
    start_method : str
        How the starting geometry was generated (e.g. "kick", "diffusion").
    search_method : str
        Optimization method (e.g. "gad", "nr_gad_hybrid").
    rxn : str
        Reaction identifier from the dataset.
    formula : str
        Chemical formula (e.g. "C3H8O").
    """

    def __init__(
        self,
        output_dir: str,
        run_id: str,
        sample_id: int,
        start_method: str,
        search_method: str,
        rxn: str = "",
        formula: str = "",
    ) -> None:
        self.meta = {
            "run_id": run_id,
            "sample_id": sample_id,
            "rxn": rxn,
            "formula": formula,
            "start_method": start_method,
            "search_method": search_method,
        }
        self.rows: list[dict] = []
        self.output_dir = output_dir
        self._start_time = time.time()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log_step(
        self,
        step: int,
        phase: str,
        dt_eff: float,
        energy: float,
        forces: Tensor,
        evals_vib: Tensor,
        evecs_vib: Tensor,
        coords: Tensor,
        coords_start: Tensor,
        coords_prev: Tensor,
        v_prev: Optional[Tensor] = None,
        known_ts_coords: Optional[Tensor] = None,
        grad: Optional[Tensor] = None,
    ) -> None:
        """Record one optimization step.

        Parameters
        ----------
        step : int
            Optimization step index (0-based).
        phase : str
            Current phase, ``"gad"`` or ``"nr"``.
        dt_eff : float
            Effective step size used this iteration.
        energy : float
            Potential energy (eV).
        forces : Tensor
            Atomic forces, shape ``(n_atoms, 3)`` or ``(1, n_atoms, 3)``.
        evals_vib : Tensor
            Eckart-projected vibrational eigenvalues, sorted ascending.
        evecs_vib : Tensor
            Corresponding eigenvectors, columns match ``evals_vib``.
        coords : Tensor
            Current coordinates, shape ``(n_atoms, 3)``.
        coords_start : Tensor
            Starting coordinates for displacement tracking.
        coords_prev : Tensor
            Previous step coordinates.
        v_prev : Tensor, optional
            Previous step's tracked eigenvector (for mode overlap).
        known_ts_coords : Tensor, optional
            Reference TS coordinates (for distance-to-TS metric).
        grad : Tensor, optional
            Energy gradient (negative of forces if not supplied).
        """
        wall_time = time.time() - self._start_time
        evals = evals_vib.detach()

        # ── Forces ───────────────────────────────────────────────────
        force_norm = _force_norm(forces)
        force_max = _force_max(forces)
        force_rms = _force_rms(forces)

        # ── Eigenvalue basics ────────────────────────────────────────
        n_neg = int((evals < 0).sum().item())
        eig0 = float(evals[0].item()) if evals.numel() > 0 else 0.0
        eig1 = float(evals[1].item()) if evals.numel() > 1 else 0.0
        # Product of the two lowest eigenvalues — large negative means
        # good TS character (one negative, one positive)
        eig_product = eig0 * eig1

        # Bottom 6 eigenvalues for quick spectral inspection
        n_bottom = min(6, evals.numel())
        bottom_spectrum = [float(evals[i].item()) for i in range(n_bottom)]

        # ── Cascade n_neg ────────────────────────────────────────────
        cascade_raw = compute_cascade_n_neg(evals)
        cascade = {}
        for raw_key, schema_key in _CASCADE_KEY_MAP.items():
            cascade[schema_key] = cascade_raw.get(raw_key, 0)

        # ── Band populations ─────────────────────────────────────────
        bands = compute_eigenvalue_bands(evals)

        # ── Mode tracking ────────────────────────────────────────────
        v0 = evecs_vib[:, 0] if evecs_vib.dim() == 2 and evecs_vib.shape[1] > 0 else None

        if v_prev is not None and v0 is not None:
            # Overlap of previous tracked vector with current lowest mode
            mode_overlap = _overlap(v_prev, v0)
            # Find which current mode best matches the previous vector
            if evecs_vib.dim() == 2:
                overlaps = torch.tensor([
                    _overlap(v_prev, evecs_vib[:, j])
                    for j in range(min(evecs_vib.shape[1], 10))
                ])
                mode_index = int(overlaps.argmax().item())
                eigvec_continuity = float(overlaps.max().item())
            else:
                mode_index = 0
                eigvec_continuity = mode_overlap
        else:
            mode_overlap = None
            mode_index = None
            eigvec_continuity = None

        # ── Gradient-mode overlaps ───────────────────────────────────
        if grad is None and forces is not None:
            grad = -forces.detach()
        if grad is not None and v0 is not None:
            grad_v0_overlap = _overlap(grad, v0)
            v1 = (evecs_vib[:, 1]
                   if evecs_vib.dim() == 2 and evecs_vib.shape[1] > 1
                   else None)
            grad_v1_overlap = _overlap(grad, v1) if v1 is not None else None
        else:
            grad_v0_overlap = None
            grad_v1_overlap = None

        # ── Displacements ────────────────────────────────────────────
        disp_from_start = _displacement(coords, coords_start)
        disp_from_last = _displacement(coords, coords_prev)
        dist_to_known_ts = (
            _displacement(coords, known_ts_coords)
            if known_ts_coords is not None
            else None
        )

        # ── Coordinates ──────────────────────────────────────────────
        coords_flat = coords.detach().reshape(-1).float().tolist()

        # ── Assemble row ─────────────────────────────────────────────
        row = {
            **self.meta,
            "step": step,
            "phase": phase,
            "dt_eff": dt_eff,
            "wall_time_s": wall_time,
            "energy": energy,
            "force_norm": force_norm,
            "force_max": force_max,
            "force_rms": force_rms,
            "n_neg": n_neg,
            "eig0": eig0,
            "eig1": eig1,
            "eig_product": eig_product,
            "bottom_spectrum": bottom_spectrum,
            **cascade,
            **bands,
            "mode_overlap": mode_overlap,
            "mode_index": mode_index,
            "eigvec_continuity": eigvec_continuity,
            "grad_v0_overlap": grad_v0_overlap,
            "grad_v1_overlap": grad_v1_overlap,
            "disp_from_start": disp_from_start,
            "disp_from_last": disp_from_last,
            "dist_to_known_ts": dist_to_known_ts,
            "coords_flat": coords_flat,
        }
        self.rows.append(row)

    def flush(self) -> str:
        """Write accumulated rows to a Parquet file.

        Returns
        -------
        str
            Absolute path to the written Parquet file.
        """
        import pyarrow as pa
        import pyarrow.parquet as pq

        table = pa.Table.from_pylist(self.rows, schema=TRAJECTORY_SCHEMA)
        path = os.path.join(
            self.output_dir,
            f"traj_{self.meta['run_id']}_{self.meta['sample_id']}.parquet",
        )
        os.makedirs(os.path.dirname(path), exist_ok=True)
        pq.write_table(table, path)
        return path

    def __len__(self) -> int:
        return len(self.rows)
