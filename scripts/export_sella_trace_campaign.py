#!/usr/bin/env python3
"""Replay exact local Sella traces into the matched W&B comparison group."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


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
    args = parser.parse_args()
    import wandb

    summaries = sorted(args.campaign_root.glob("task_*/summary_*.parquet"))
    if len(summaries) != 287:
        raise SystemExit(f"expected 287 task summaries, found {len(summaries)}")
    stop = len(summaries) if args.stop_index is None else args.stop_index
    if not 0 <= args.start_index <= stop <= len(summaries):
        raise SystemExit("invalid --start-index/--stop-index range")
    for index, summary_path in enumerate(summaries[args.start_index:stop], start=args.start_index + 1):
        row = pq.read_table(summary_path).to_pylist()[0]
        trace_path = Path(str(row.get("sella_trace_path", "")))
        if not trace_path.is_file():
            print(f"[{index}/287] skip sample {row['sample_id']}: no trace")
            continue
        with np.load(trace_path) as trace:
            fields = {key: trace[key] for key in trace.files if key != "coordinates"}
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
        for position in range(count):
            payload = {"trajectory/evaluation": int(fields["evaluation"][position])}
            for key, values in fields.items():
                value = values[position].item()
                if isinstance(value, float) and not np.isfinite(value):
                    continue
                payload[f"trajectory/{key}"] = value
            run.log(payload)
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
