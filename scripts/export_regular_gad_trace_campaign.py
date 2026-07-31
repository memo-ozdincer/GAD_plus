#!/usr/bin/env python3
"""Export the exact ordinary-GAD Parquet trajectories of one 287-start cell.

The legacy projected-GAD runner already records a rich per-step trajectory.
This adapter deliberately replays that immutable record rather than
re-evaluating the calculator, so W&B is an observational layer and cannot
change a search result.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from gadplus.logging.wandb_export import event_preserving_indices, kabsch_rmsd, log_trajectory_table


def _run_id(parts: tuple[object, ...]) -> str:
    return hashlib.sha256("\x1f".join(map(str, parts)).encode()).hexdigest()[:20]


def _finite(value: object) -> bool:
    """Whether a Parquet scalar is safely loggable as a numeric time series."""

    if value is None or isinstance(value, (str, bytes, bool)) or not np.isscalar(value):
        return False
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _number_or_none(value: object) -> float | None:
    return float(value) if _finite(value) else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign_root", type=Path)
    parser.add_argument("--noise", type=float, required=True)
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--project", default="gadplus-ts-mechanisms")
    parser.add_argument("--entity", default=os.environ.get("WANDB_ENTITY"))
    parser.add_argument("--group", required=True)
    parser.add_argument("--mode", choices=("online", "offline"), default="online")
    parser.add_argument("--start-index", type=int, default=0, help="0-based inclusive summary index")
    parser.add_argument("--stop-index", type=int, help="0-based exclusive summary index")
    parser.add_argument("--max-view-rows", type=int, default=600)
    parser.add_argument("--max-runs", type=int, help="Evenly select at most this many trajectories from the requested range.")
    parser.add_argument(
        "--cockpit-chart-id",
        default="memo-ozdincer-university-of-toronto/gadplus-trajectory-cockpit-v3",
    )
    parser.add_argument(
        "--mechanism-chart-id",
        default="memo-ozdincer-university-of-toronto/gadplus-regular-gad-mechanism-v2",
    )
    args = parser.parse_args()

    import wandb

    summaries = sorted(args.campaign_root.glob("task_*/summary_*.parquet"))
    if len(summaries) != 287:
        raise SystemExit(f"expected 287 task summaries, found {len(summaries)}")
    stop = len(summaries) if args.stop_index is None else args.stop_index
    if not 0 <= args.start_index <= stop <= len(summaries):
        raise SystemExit("invalid --start-index/--stop-index range")
    positions = list(range(args.start_index, stop))
    if args.max_runs is not None and len(positions) > args.max_runs:
        if args.max_runs < 1:
            raise SystemExit("--max-runs must be positive")
        positions = sorted({round(item * (len(positions) - 1) / (args.max_runs - 1)) for item in range(args.max_runs)}) if args.max_runs > 1 else [0]
        positions = [args.start_index + item for item in positions]
    for summary_index in positions:
        index, summary_path = summary_index + 1, summaries[summary_index]
        summary = pq.read_table(summary_path).to_pylist()[0]
        trace_matches = sorted(summary_path.parent.glob("traj_*.parquet"))
        if len(trace_matches) != 1:
            print(f"[{index}/287] skip sample {summary['sample_id']}: trace missing/ambiguous")
            continue
        trace_path = trace_matches[0]
        trace = pq.read_table(trace_path).to_pylist()
        config = {
            "dataset": "Transition1x",
            "calculator": "g-xTB",
            "optimizer": "regular_gad",
            "sample_id": int(summary["sample_id"]),
            "formula": summary.get("formula", ""),
            "rxn": summary.get("rxn", ""),
            "noise_angstrom": args.noise,
            "max_steps": args.budget,
            "dt": 0.003,
            "instrumentation_level": "exact-projected-gad-steps",
        }
        run = wandb.init(
            project=args.project, entity=args.entity, group=args.group,
            job_type="regular-gad", tags=("evaluation", "Transition1x", "g-xTB", "regular-gad"),
            id=_run_id((args.group, config["sample_id"], args.noise, "regular-gad")),
            resume="allow", mode=args.mode,
            config=config,
            name=f"regular-gad-{config['sample_id']:03d}-{args.noise:.2f}A",
        )
        run.define_metric("trajectory/step")
        run.define_metric("trajectory/*", step_metric="trajectory/step")
        coordinates = np.asarray([entry["coords_flat"] for entry in trace], dtype=float)
        coordinates = coordinates.reshape(len(trace), -1, 3)
        terminal = coordinates[-1]
        terminal_distance = np.asarray([kabsch_rmsd(xyz, terminal) for xyz in coordinates])
        terminal_d0 = float(terminal_distance[0])
        start_energy = float(trace[0]["energy"])
        view_rows: list[dict[str, object]] = []
        for position, row in enumerate(trace):
            spectrum = list(row.get("bottom_spectrum") or [])
            low = [float(value) for value in spectrum[:3] if _finite(value)]
            scale = float(np.sqrt(np.mean(np.square(low)))) if low else 1.0
            scale = max(scale, 1.0e-15)
            known_distance = row.get("dist_to_known_ts")
            normalized = {
                "evaluation": int(row["step"]),
                "force_max": float(row["force_max"]),
                "force_ratio_display": max(float(row["force_max"]) / 0.01, 1.0e-15),
                "n_neg": int(row["n_neg"]),
                "lambda1": float(spectrum[0]) if len(spectrum) > 0 else None,
                "lambda2": float(spectrum[1]) if len(spectrum) > 1 else None,
                "lambda3": float(spectrum[2]) if len(spectrum) > 2 else None,
                "lambda1_scaled": float(spectrum[0]) / scale if len(spectrum) > 0 else None,
                "lambda2_scaled": float(spectrum[1]) / scale if len(spectrum) > 1 else None,
                "lambda3_scaled": float(spectrum[2]) / scale if len(spectrum) > 2 else None,
                "spectral_scale": scale,
                "terminal": position == len(trace) - 1,
                "distance_to_terminal": float(terminal_distance[position]),
                "distance_to_terminal_display": max(float(terminal_distance[position]), 1.0e-15),
                "terminal_progress_raw": 1.0 - float(terminal_distance[position]) / terminal_d0 if terminal_d0 else float(position == len(trace) - 1),
                "distance_to_labelled_ts": float(known_distance) if _finite(known_distance) else None,
                "distance_to_labelled_ts_display": max(float(known_distance), 1.0e-15) if _finite(known_distance) else None,
                "step_mw_rms": None,
                "max_atom_displacement": None,
                "step_over_radius": None,
                "step_over_radius_display": None,
                "step_over_length": None,
                "step_over_length_display": None,
                "energy": float(row["energy"]),
                "energy_from_start": float(row["energy"]) - start_energy,
                "wall_time_s": float(row["wall_time_s"]),
                "dt_eff": _number_or_none(row.get("dt_eff")),
                "mode_overlap": _number_or_none(row.get("mode_overlap")),
                "eigvec_continuity": _number_or_none(row.get("eigvec_continuity")),
                "grad_v0_overlap": _number_or_none(row.get("grad_v0_overlap")),
                "grad_v1_overlap": _number_or_none(row.get("grad_v1_overlap")),
                "disp_from_last": _number_or_none(row.get("disp_from_last")),
                "step_cart_rms": _number_or_none(row.get("disp_from_last")),
            }
            view_rows.append(normalized)
            payload: dict[str, object] = {"trajectory/step": int(row["step"])}
            for key, value in row.items():
                if key in {"step", "coords_flat", "bottom_spectrum", "run_id", "sample_id", "rxn", "formula", "start_method", "search_method"}:
                    continue
                if _finite(value):
                    payload[f"trajectory/{key}"] = value
            # Keep the first few curvatures searchable as distinct scalars.
            for mode, value in enumerate(row.get("bottom_spectrum") or []):
                if mode >= 6 or not _finite(value):
                    continue
                payload[f"trajectory/lambda_{mode + 1}"] = value
            for key, value in normalized.items():
                if key != "evaluation" and _finite(value):
                    payload[f"trajectory/{key}"] = value
            run.log(payload)
        view_rows = [view_rows[item] for item in event_preserving_indices(view_rows, max_rows=args.max_view_rows)]
        log_trajectory_table(
            run,
            view_rows,
            cockpit_chart_id=args.cockpit_chart_id,
            mechanism_chart_id=args.mechanism_chart_id,
            mechanism_key="regular_gad_mechanism",
            mechanism_fields=(
                "evaluation", "dt_eff", "disp_from_last", "mode_overlap",
                "eigvec_continuity", "grad_v0_overlap", "grad_v1_overlap",
                "lambda1", "lambda2", "n_neg",
            ),
        )
        run.summary.update({
            "calculator_valid": not bool(summary.get("final_eval_error")),
            "local_ts": bool(summary.get("converged")),
            "final_n_neg": int(summary["final_n_neg"]),
            "final_force_max": float(summary["final_force_max"]),
            "total_steps": int(summary["total_steps"]),
            "failure_type": summary.get("failure_type", ""),
            "trace_steps": len(trace),
        })
        artifact = wandb.Artifact(f"regular-gad-trace-{run.id}", type="ts-trajectory")
        artifact.add_file(str(summary_path))
        artifact.add_file(str(trace_path))
        run.log_artifact(artifact)
        run.finish()
        print(f"[{index}/287] sample {config['sample_id']} -> {run.id}")


if __name__ == "__main__":
    main()
