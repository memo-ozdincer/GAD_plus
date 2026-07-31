"""Replay exact local trajectory bundles into optional W&B observability.

No function in this module is imported by an optimizer. The W&B SDK is loaded
only inside :func:`export_bundle`, which keeps the core package usable without
the optional dependency and makes offline export the safe cluster default.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

COCKPIT_FIELDS = (
    "evaluation",
    "force_ratio_display",
    "force_max",
    "n_neg",
    "lambda1",
    "lambda2",
    "lambda3",
    "lambda1_scaled",
    "lambda2_scaled",
    "lambda3_scaled",
    "spectral_scale",
    "terminal",
    "distance_to_terminal_display",
    "distance_to_terminal",
    "distance_to_labelled_ts_display",
    "distance_to_labelled_ts",
    "terminal_progress_raw",
    "step_over_radius_display",
    "step_over_radius",
    "step_over_length_display",
    "step_over_length",
    "step_mw_rms",
    "max_atom_displacement",
    "energy",
    "energy_from_start",
    "wall_time_s",
)

COMPETITIVE_FIELDS = (
    "evaluation",
    "lambda2_gate",
    "effective_gate",
    "activity_fraction",
    "soft_activity",
    "extra_negative_activity",
    "activity_log10_ratio",
    "p1",
    "p2",
    "spectral_entropy",
    "lowest_reflection",
    "lambda2_scaled",
    "n_neg",
    "regularizer_scaled",
    "step_over_radius",
    "local_radius",
)


def deterministic_run_id(parts: Sequence[object], *, length: int = 20) -> str:
    """Return a stable W&B-safe identifier for a scientific run identity."""

    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:length]


def kabsch_rmsd(coords: np.ndarray, reference: np.ndarray) -> float:
    """Rigid-motion-aligned RMSD for coordinates with fixed atom identity."""

    xyz = np.asarray(coords, dtype=np.float64)
    ref = np.asarray(reference, dtype=np.float64)
    if xyz.shape != ref.shape or xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("Kabsch inputs must have matching (N, 3) shapes")
    xyz0 = xyz - xyz.mean(axis=0)
    ref0 = ref - ref.mean(axis=0)
    left, _, right_t = np.linalg.svd(xyz0.T @ ref0)
    correction = np.eye(3)
    correction[-1, -1] = np.sign(np.linalg.det(left @ right_t))
    rotation = left @ correction @ right_t
    residual = xyz0 @ rotation - ref0
    return float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))


def enrich_rows(
    rows: Sequence[Mapping[str, Any]],
    coordinates: np.ndarray,
    *,
    force_threshold: float,
    labelled_ts: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    """Add normalized and hindsight-only fields used by the dashboard."""

    if force_threshold <= 0:
        raise ValueError("force_threshold must be positive")
    if len(rows) != len(coordinates):
        raise ValueError("trajectory rows and coordinates must have equal length")
    terminal = coordinates[-1]
    terminal_distances = np.asarray(
        [kabsch_rmsd(coords, terminal) for coords in coordinates], dtype=np.float64
    )
    labelled_distances = None
    if labelled_ts is not None:
        labelled_distances = np.asarray(
            [kabsch_rmsd(coords, labelled_ts) for coords in coordinates], dtype=np.float64
        )

    enriched: list[dict[str, Any]] = []
    start_energy = float(rows[0]["energy"])
    best_terminal = math.inf
    best_labelled = math.inf
    terminal_d0 = float(terminal_distances[0])
    labelled_d0 = float(labelled_distances[0]) if labelled_distances is not None else math.nan
    for index, original in enumerate(rows):
        row = dict(original)
        best_terminal = min(best_terminal, float(terminal_distances[index]))
        row["energy_from_start"] = float(row["energy"]) - start_energy
        row["force_ratio"] = float(row["force_max"]) / force_threshold
        row["force_ratio_display"] = max(row["force_ratio"], 1.0e-15)
        row["distance_to_terminal"] = float(terminal_distances[index])
        row["distance_to_terminal_display"] = max(row["distance_to_terminal"], 1.0e-15)
        row["best_distance_to_terminal"] = best_terminal
        raw_progress = (
            1.0 - float(terminal_distances[index]) / terminal_d0
            if terminal_d0 > 0
            else (1.0 if index == len(rows) - 1 else 0.0)
        )
        row["terminal_progress_raw"] = raw_progress
        row["terminal_progress_clipped"] = min(1.0, max(0.0, raw_progress))
        if labelled_distances is not None:
            distance = float(labelled_distances[index])
            best_labelled = min(best_labelled, distance)
            row["distance_to_labelled_ts"] = distance
            row["best_distance_to_labelled_ts"] = best_labelled
            labelled_progress = 1.0 - distance / labelled_d0 if labelled_d0 > 0 else 0.0
            row["labelled_progress_raw"] = labelled_progress
            row["labelled_progress_clipped"] = min(1.0, max(0.0, labelled_progress))
            row["distance_to_labelled_ts_display"] = max(distance, 1.0e-15)
        else:
            row["distance_to_labelled_ts"] = None
            row["best_distance_to_labelled_ts"] = None
            row["labelled_progress_raw"] = None
            row["labelled_progress_clipped"] = None
            row["distance_to_labelled_ts_display"] = None
        for field in ("step_over_radius", "step_over_length"):
            value = row.get(field)
            row[f"{field}_display"] = (
                max(float(value), 1.0e-15)
                if value is not None and math.isfinite(float(value))
                else None
            )
        enriched.append(row)
    return enriched


def event_preserving_indices(
    rows: Sequence[Mapping[str, Any]],
    *,
    max_rows: int = 600,
) -> list[int]:
    """Select a compact view while retaining scientific transition events."""

    count = len(rows)
    if max_rows < 2:
        raise ValueError("max_rows must be at least two")
    if count <= max_rows:
        return list(range(count))

    mandatory = {0, count - 1}
    for index in range(1, count):
        previous, current = rows[index - 1], rows[index]
        if current.get("n_neg") != previous.get("n_neg"):
            mandatory.add(index)
            mandatory.add(index - 1)
        prev_lambda2 = float(previous.get("lambda2", math.nan))
        curr_lambda2 = float(current.get("lambda2", math.nan))
        finite_crossing = math.isfinite(prev_lambda2) and math.isfinite(curr_lambda2)
        sign_crossing = (prev_lambda2 < 0 <= curr_lambda2) or (curr_lambda2 < 0 <= prev_lambda2)
        if finite_crossing and sign_crossing:
            mandatory.add(index)
            mandatory.add(index - 1)
        if bool(current.get("terminal", False)):
            mandatory.add(index)

    force = np.asarray([float(row.get("force_max", math.nan)) for row in rows])
    gate = np.asarray([float(row.get("effective_gate", math.nan)) for row in rows])
    step = np.asarray([float(row.get("step_cart_rms", math.nan)) for row in rows])
    for index in range(1, count - 1):
        if (
            np.isfinite(force[index])
            and force[index] <= force[index - 1]
            and force[index] <= force[index + 1]
        ):
            mandatory.add(index)
    for values in (np.abs(np.diff(gate)), step):
        finite = np.where(np.isfinite(values))[0]
        if finite.size:
            top = finite[np.argsort(values[finite])[-min(64, finite.size) :]]
            mandatory.update(int(index + (1 if values.size == count - 1 else 0)) for index in top)

    if len(mandatory) >= max_rows:
        # Transition events take precedence; keep evenly spaced mandatory
        # points when pathological oscillation alone exceeds the view budget.
        ordered = sorted(mandatory)
        positions = np.linspace(0, len(ordered) - 1, max_rows, dtype=int)
        return [ordered[position] for position in positions]

    remaining_budget = max_rows - len(mandatory)
    candidates = [index for index in range(count) if index not in mandatory]
    positions = np.linspace(0, len(candidates) - 1, remaining_budget, dtype=int)
    mandatory.update(candidates[position] for position in positions)
    return sorted(mandatory)


def load_bundle(bundle_dir: str | Path) -> tuple[list[dict[str, Any]], np.ndarray, dict[str, Any]]:
    """Load one exact local trajectory bundle."""

    root = Path(bundle_dir).resolve()
    rows = pq.read_table(root / "trajectory.parquet").to_pylist()
    with np.load(root / "coordinates.npz") as arrays:
        coordinates = arrays["coordinates"].copy()
    with (root / "metadata.json").open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    return rows, coordinates, metadata


def log_trajectory_table(
    run: Any,
    rows: Sequence[Mapping[str, Any]],
    *,
    cockpit_chart_id: str | None = None,
    mechanism_chart_id: str | None = None,
    mechanism_key: str = "competitive_mechanism",
    mechanism_fields: Sequence[str] = COMPETITIVE_FIELDS,
) -> None:
    """Attach one exact, queryable trajectory table and optional Vega panels.

    This deliberately runs *after* history logging.  The table is a compact
    visualization view of an already recorded trajectory, never an input to
    the search.  Keeping all common cockpit columns (including null columns)
    makes the same chart honest across GAD and Sella: a panel is absent where
    the corresponding optimizer did not record or define that quantity.
    """

    import wandb

    columns = sorted({key for row in rows for key in row} | set(COCKPIT_FIELDS))

    def table_value(value: Any) -> Any:
        if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
            return None
        if isinstance(value, np.generic):
            return value.item()
        return value

    data = [[table_value(row.get(column)) for column in columns] for row in rows]
    table = wandb.Table(columns=columns, data=data)
    payload: dict[str, Any] = {"trajectory_view": table}
    if cockpit_chart_id:
        payload["trajectory_cockpit"] = wandb.plot_table(
            vega_spec_name=cockpit_chart_id,
            data_table=table,
            fields={field: field for field in COCKPIT_FIELDS},
        )
    if mechanism_chart_id:
        payload[mechanism_key] = wandb.plot_table(
            vega_spec_name=mechanism_chart_id,
            data_table=table,
            fields={field: field for field in mechanism_fields},
        )
    run.log(payload)

    # Plain W&B line charts are intentionally duplicated from the richer Vega
    # cockpit. They are a robust, immediately visible per-run fallback when a
    # custom chart is unavailable in a particular workspace/browser.
    def line_data(field: str, *, absolute: bool = False) -> tuple[list[float], list[float]]:
        xs: list[float] = []
        ys: list[float] = []
        for row in rows:
            x, value = row.get("evaluation"), row.get(field)
            if x is None or value is None:
                continue
            try:
                x_float, y_float = float(x), float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(x_float) and math.isfinite(y_float):
                xs.append(x_float)
                ys.append(abs(y_float) if absolute else y_float)
        return xs, ys

    force_x, force_y = line_data("force_ratio_display")
    force_y = [math.log10(value) for value in force_y]
    if force_x:
        run.log({
            "trajectory_force_graph": wandb.plot.line_series(
                force_x, [force_y], keys=("log10(fmax / tolerance)",),
                title="Stationarity (per-run; 0 is the force tolerance)",
                xname="Hessian evaluation",
            )
        })
    spectra = [line_data(field, absolute=True) for field in ("lambda1", "lambda2", "lambda3")]
    present = [(x, y, label) for (x, y), label in zip(spectra, ("|lambda1|", "|lambda2|", "|lambda3|")) if x]
    if present:
        run.log({
            "trajectory_curvature_graph": wandb.plot.line_series(
                [item[0] for item in present], [item[1] for item in present],
                keys=[item[2] for item in present],
                title="Lowest vibrational-curvature magnitudes (per run)",
                xname="Hessian evaluation",
            )
        })
    distance_series = [
        (*line_data(field), label)
        for field, label in (
            ("distance_to_terminal_display", "RMSD to terminal candidate"),
            ("distance_to_labelled_ts_display", "RMSD to labelled TS"),
        )
    ]
    present_distance = [(x, y, label) for x, y, label in distance_series if x]
    if present_distance:
        run.log({
            "trajectory_hindsight_distance_graph": wandb.plot.line_series(
                [item[0] for item in present_distance], [item[1] for item in present_distance],
                keys=[item[2] for item in present_distance],
                title="Hindsight closeness to TS/candidate (not used by optimizer)",
                xname="Hessian evaluation",
            )
        })
    step_series = [
        (*line_data(field), label)
        for field, label in (
            ("step_cart_rms", "Cartesian RMS step"),
            ("disp_from_last", "Cartesian displacement"),
            ("max_atom_displacement", "maximum atom displacement"),
        )
    ]
    present_step = [(x, y, label) for x, y, label in step_series if x]
    if present_step:
        run.log({
            "trajectory_step_graph": wandb.plot.line_series(
                [item[0] for item in present_step], [item[1] for item in present_step],
                keys=[item[2] for item in present_step],
                title="Per-step displacement (per run)",
                xname="Hessian evaluation",
            )
        })


def export_bundle(
    bundle_dir: str | Path,
    *,
    project: str = "gadplus-ts-mechanisms",
    entity: str | None = None,
    group: str | None = None,
    job_type: str = "competitive-gad",
    tags: Sequence[str] = (),
    mode: str = "offline",
    labelled_ts: np.ndarray | None = None,
    max_view_rows: int = 600,
    cockpit_chart_id: str | None = None,
    mechanism_chart_id: str | None = None,
) -> str:
    """Export a local bundle to W&B and return the deterministic run ID."""

    try:
        import wandb
    except ImportError as error:
        raise RuntimeError(
            "W&B export requires the optional 'observability' dependencies"
        ) from error

    root = Path(bundle_dir).resolve()
    rows, coordinates, metadata = load_bundle(root)
    if labelled_ts is None:
        with np.load(root / "coordinates.npz") as arrays:
            if "labelled_ts" in arrays:
                labelled_ts = arrays["labelled_ts"].copy()
    config = dict(metadata.get("config", {}))
    force_threshold = float(config.get("force_threshold", 0.01))
    enriched = enrich_rows(
        rows,
        coordinates,
        force_threshold=force_threshold,
        labelled_ts=labelled_ts,
    )
    indices = event_preserving_indices(enriched, max_rows=max_view_rows)
    view = [enriched[index] for index in indices]
    identity = [
        metadata.get("run_id", root.name),
        config.get("sample_id", ""),
        config.get("noise_angstrom", ""),
        config.get("seed", ""),
        job_type,
    ]
    run_id = deterministic_run_id(identity)
    run = wandb.init(
        project=project,
        entity=entity,
        group=group,
        job_type=job_type,
        tags=list(tags),
        id=run_id,
        resume="allow",
        mode=mode,
        config=config,
        name=str(metadata.get("run_id", root.name)),
    )
    run.define_metric("trajectory/evaluation")
    run.define_metric("trajectory/*", step_metric="trajectory/evaluation")
    for row in view:
        payload: dict[str, Any] = {"trajectory/evaluation": row["evaluation"]}
        for key, value in row.items():
            if key == "evaluation" or value is None:
                continue
            if isinstance(value, (bool, int, float, np.number)):
                numeric = value.item() if isinstance(value, np.generic) else value
                if not isinstance(numeric, float) or math.isfinite(numeric):
                    payload[f"trajectory/{key}"] = numeric
        run.log(payload)

    log_trajectory_table(
        run,
        view,
        cockpit_chart_id=cockpit_chart_id,
        mechanism_chart_id=mechanism_chart_id,
    )
    run.summary.update(metadata.get("summary", {}))
    run.summary["view_rows"] = len(view)
    run.summary["exact_rows"] = len(enriched)
    artifact = wandb.Artifact(f"trajectory-{run_id}", type="ts-trajectory")
    artifact.add_file(str(root / "trajectory.parquet"))
    artifact.add_file(str(root / "coordinates.npz"))
    artifact.add_file(str(root / "metadata.json"))
    run.log_artifact(artifact)
    run.finish()
    return run_id


__all__ = [
    "COCKPIT_FIELDS",
    "COMPETITIVE_FIELDS",
    "deterministic_run_id",
    "enrich_rows",
    "event_preserving_indices",
    "export_bundle",
    "kabsch_rmsd",
    "load_bundle",
    "log_trajectory_table",
]
