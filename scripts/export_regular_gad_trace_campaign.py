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
    args = parser.parse_args()

    import wandb

    summaries = sorted(args.campaign_root.glob("task_*/summary_*.parquet"))
    if len(summaries) != 287:
        raise SystemExit(f"expected 287 task summaries, found {len(summaries)}")
    stop = len(summaries) if args.stop_index is None else args.stop_index
    if not 0 <= args.start_index <= stop <= len(summaries):
        raise SystemExit("invalid --start-index/--stop-index range")
    for index, summary_path in enumerate(summaries[args.start_index:stop], start=args.start_index + 1):
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
        for row in trace:
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
            run.log(payload)
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
