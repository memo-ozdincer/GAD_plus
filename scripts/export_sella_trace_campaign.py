#!/usr/bin/env python3
"""Replay exact local Sella traces into the matched W&B comparison group."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from gadplus.logging.wandb_export import event_preserving_indices, kabsch_rmsd, log_trajectory_table


def _run_id(parts: tuple[object, ...]) -> str:
    text = "\x1f".join(map(str, parts)).encode()
    return hashlib.sha256(text).hexdigest()[:20]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign_root", type=Path)
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
        default="memo-ozdincer-university-of-toronto/gadplus-sella-mechanism-v2",
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
        row = pq.read_table(summary_path).to_pylist()[0]
        trace_path = Path(str(row.get("sella_trace_path", "")))
        if not trace_path.is_file():
            print(f"[{index}/287] skip sample {row['sample_id']}: no trace")
            continue
        with np.load(trace_path) as trace:
            fields = {key: trace[key] for key in trace.files if key != "coordinates"}
            coordinates = trace["coordinates"].copy()
            count = len(fields["evaluation"])
        config = {
            "dataset": "Transition1x",
            "calculator": "g-xTB",
            "optimizer": "sella",
            "sample_id": int(row["sample_id"]),
            "formula": row.get("formula", ""),
            "rxn": row.get("rxn", ""),
            "noise_angstrom": float(row["noise_angstrom"]),
            "max_steps": int(row["total_steps"]),
            "instrumentation_level": "local-sella-evaluations",
        }
        run = wandb.init(
            project=args.project,
            entity=args.entity,
            group=args.group,
            job_type="sella",
            tags=("evaluation", "Transition1x", "g-xTB", "sella"),
            id=_run_id((args.group, config["sample_id"], config["noise_angstrom"], "sella")),
            resume="allow",
            mode=args.mode,
            config=config,
            name=f"sella-{config['sample_id']:03d}-{config['noise_angstrom']:.2f}A",
        )
        run.define_metric("trajectory/evaluation")
        run.define_metric("trajectory/*", step_metric="trajectory/evaluation")
        coordinates = np.asarray(coordinates, dtype=float)
        terminal = coordinates[-1]
        terminal_distance = np.asarray([kabsch_rmsd(xyz, terminal) for xyz in coordinates])
        terminal_d0 = float(terminal_distance[0])
        start_energy = float(fields["energy"][0])
        step_cart_rms = np.zeros(count, dtype=float)
        if count > 1:
            displacement = coordinates[1:] - coordinates[:-1]
            step_cart_rms[1:] = np.sqrt(np.mean(np.sum(displacement * displacement, axis=2), axis=1))
        view_rows: list[dict[str, object]] = []
        for position in range(count):
            payload = {"trajectory/evaluation": int(fields["evaluation"][position])}
            for key, values in fields.items():
                value = values[position].item()
                if isinstance(value, float) and not np.isfinite(value):
                    continue
                payload[f"trajectory/{key}"] = value
            lambdas = [float(fields[name][position]) for name in ("lambda1", "lambda2", "lambda3")]
            scale = max(float(np.sqrt(np.mean(np.square(lambdas)))), 1.0e-15)
            normalized = {
                "evaluation": int(fields["evaluation"][position]),
                "force_max": float(fields["force_max"][position]),
                "force_ratio_display": max(float(fields["force_max"][position]) / 0.01, 1.0e-15),
                "n_neg": int(fields["n_neg"][position]),
                "lambda1": lambdas[0], "lambda2": lambdas[1], "lambda3": lambdas[2],
                "lambda1_scaled": lambdas[0] / scale,
                "lambda2_scaled": lambdas[1] / scale,
                "lambda3_scaled": lambdas[2] / scale,
                "spectral_scale": scale,
                "terminal": position == count - 1,
                "distance_to_terminal": float(terminal_distance[position]),
                "distance_to_terminal_display": max(float(terminal_distance[position]), 1.0e-15),
                "terminal_progress_raw": 1.0 - float(terminal_distance[position]) / terminal_d0 if terminal_d0 else float(position == count - 1),
                "distance_to_labelled_ts": None,
                "distance_to_labelled_ts_display": None,
                "step_mw_rms": None, "max_atom_displacement": None,
                "step_cart_rms": float(step_cart_rms[position]),
                "step_over_radius": None, "step_over_radius_display": None,
                "step_over_length": None, "step_over_length_display": None,
                "energy": float(fields["energy"][position]),
                "energy_from_start": float(fields["energy"][position]) - start_energy,
                "wall_time_s": float(fields["wall_time_s"][position]),
                "force_rms": float(fields["force_rms"][position]),
            }
            view_rows.append(normalized)
            for key, value in normalized.items():
                if key != "evaluation" and value is not None and np.isscalar(value) and np.isfinite(float(value)):
                    payload[f"trajectory/{key}"] = value
            run.log(payload)
        view_rows = [view_rows[item] for item in event_preserving_indices(view_rows, max_rows=args.max_view_rows)]
        log_trajectory_table(
            run,
            view_rows,
            cockpit_chart_id=args.cockpit_chart_id,
            mechanism_chart_id=args.mechanism_chart_id,
            mechanism_key="sella_mechanism",
            mechanism_fields=(
                "evaluation", "force_max", "force_rms", "energy_from_start",
                "wall_time_s", "lambda1_scaled", "lambda2_scaled", "lambda3_scaled", "n_neg",
            ),
        )
        run.summary.update(
            {
                "calculator_valid": not bool(row.get("final_eval_error")),
                "local_ts": bool(row.get("converged")),
                "final_n_neg": int(row["final_n_neg"]),
                "final_force_max": float(row["final_force_max"]),
                "total_steps": int(row["total_steps"]),
                "trace_evaluations": count,
            }
        )
        artifact = wandb.Artifact(f"sella-trace-{run.id}", type="ts-trajectory")
        artifact.add_file(str(trace_path))
        artifact.add_file(str(summary_path))
        run.log_artifact(artifact)
        run.finish()
        print(f"[{index}/287] sample {config['sample_id']} -> {run.id}")


if __name__ == "__main__":
    main()
