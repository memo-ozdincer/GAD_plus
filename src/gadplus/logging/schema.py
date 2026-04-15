"""Parquet schema definitions for trajectory and summary data.

Two schemas:
  TRAJECTORY_SCHEMA — one row per optimization step per sample.
  SUMMARY_SCHEMA    — one row per sample (final outcome).

All eigenvalue fields assume Eckart-projected vibrational eigenvalues.
Cascade n_neg fields match the thresholds in core.convergence.CASCADE_THRESHOLDS.
"""
from __future__ import annotations

import pyarrow as pa

TRAJECTORY_SCHEMA = pa.schema([
    # ── Identifiers ──────────────────────────────────────────────────────
    ("run_id", pa.string()),
    ("sample_id", pa.int32()),
    ("rxn", pa.string()),
    ("formula", pa.string()),
    ("start_method", pa.string()),
    ("search_method", pa.string()),

    # ── Step info ────────────────────────────────────────────────────────
    ("step", pa.int32()),
    ("phase", pa.string()),           # "gad" or "nr"
    ("dt_eff", pa.float64()),
    ("wall_time_s", pa.float64()),

    # ── Energy surface ───────────────────────────────────────────────────
    ("energy", pa.float64()),
    ("force_norm", pa.float64()),
    ("force_max", pa.float64()),
    ("force_rms", pa.float64()),

    # ── Eigenvalue spectrum ──────────────────────────────────────────────
    ("n_neg", pa.int32()),
    ("eig0", pa.float64()),
    ("eig1", pa.float64()),
    ("eig_product", pa.float64()),
    ("bottom_spectrum", pa.list_(pa.float64())),

    # ── Cascade n_neg at thresholds ──────────────────────────────────────
    ("n_neg_0", pa.int32()),
    ("n_neg_1e4", pa.int32()),
    ("n_neg_5e4", pa.int32()),
    ("n_neg_1e3", pa.int32()),
    ("n_neg_2e3", pa.int32()),
    ("n_neg_5e3", pa.int32()),
    ("n_neg_8e3", pa.int32()),
    ("n_neg_1e2", pa.int32()),

    # ── Eigenvalue band populations ──────────────────────────────────────
    ("band_neg_large", pa.int32()),
    ("band_neg_small", pa.int32()),
    ("band_near_zero", pa.int32()),
    ("band_pos_small", pa.int32()),
    ("band_pos_large", pa.int32()),

    # ── Mode tracking ────────────────────────────────────────────────────
    ("mode_overlap", pa.float64()),
    ("mode_index", pa.int32()),
    ("eigvec_continuity", pa.float64()),

    # ── Gradient-mode overlaps (bottleneck detector) ─────────────────────
    ("grad_v0_overlap", pa.float64()),
    ("grad_v1_overlap", pa.float64()),

    # ── Displacements ────────────────────────────────────────────────────
    ("disp_from_start", pa.float64()),
    ("disp_from_last", pa.float64()),
    ("dist_to_known_ts", pa.float64()),

    # ── Coordinates ──────────────────────────────────────────────────────
    ("coords_flat", pa.list_(pa.float32())),
])

SUMMARY_SCHEMA = pa.schema([
    ("run_id", pa.string()),
    ("sample_id", pa.int32()),
    ("rxn", pa.string()),
    ("formula", pa.string()),
    ("start_method", pa.string()),
    ("search_method", pa.string()),
    ("converged", pa.bool_()),
    ("converged_step", pa.int32()),
    ("total_steps", pa.int32()),
    ("final_n_neg", pa.int32()),
    ("final_force_norm", pa.float64()),
    ("final_force_max", pa.float64()),
    ("final_energy", pa.float64()),
    ("final_eig0", pa.float64()),
    ("wall_time_total_s", pa.float64()),
    ("failure_type", pa.string()),
])
