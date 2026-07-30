#!/usr/bin/env python3
"""Publish the completed matched g-xTB grid as one compact W&B summary run.

Individual optimizer runs retain exact trajectories and artifacts.  This
script deliberately publishes only 12 campaign rows, so the overview remains
fast while every displayed rate retains its explicit denominator.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path


def _load_aggregator(script_root: Path):
    spec = importlib.util.spec_from_file_location(
        "matched_grid_aggregate", script_root / "aggregate_t1x_gxtb_noise_grid.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load grid aggregator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _raw_count(root: Path, family: str) -> int:
    pattern = "tasks/task_*.json" if family in {"competitive", "competitive_subspace"} else "task_*/summary_*.parquet"
    return len(list(root.glob(pattern)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--project", default="gadplus-ts-mechanisms")
    parser.add_argument("--entity", default=os.environ.get("WANDB_ENTITY"))
    parser.add_argument("--group", required=True)
    parser.add_argument("--mode", choices=("online", "offline"), default="online")
    args = parser.parse_args()

    specs = json.loads(args.manifest.read_text())
    for campaign in specs:
        raw = Path(campaign["raw_root"])
        topology = Path(campaign["topology_root"])
        n_raw = _raw_count(raw, campaign["family"])
        n_topology = len(list(topology.glob("task_*.json")))
        if n_raw != 287 or n_topology != 287:
            raise SystemExit(
                f"incomplete {campaign['family']} {campaign['noise_angstrom']}: "
                f"raw={n_raw}/287, topology={n_topology}/287"
            )

    aggregate = _load_aggregator(Path(__file__).parent)
    rows = sorted(
        (aggregate._summary(campaign) for campaign in specs),
        key=lambda row: (row["noise_angstrom"], row["method"]),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "summary.json"
    markdown_path = args.output_dir / "SUMMARY.md"
    json_path.write_text(json.dumps(rows, indent=2) + "\n")
    markdown_path.write_text(aggregate._markdown(rows))

    import wandb

    run = wandb.init(
        project=args.project, entity=args.entity, group=args.group,
        job_type="matched-grid-summary", mode=args.mode,
        config={
            "dataset": "Transition1x test split", "calculator": "g-xTB",
            "starts_per_cell": 287, "local_force_threshold_eV_per_A": 0.03,
            "strict_force_threshold_eV_per_A": 0.01,
            "topology": "native labelled two-branch endpoint classifier",
            "cells": len(rows),
        },
        name="matched-noise-grid-summary",
        tags=("evaluation", "Transition1x", "g-xTB", "matched-grid", "topology"),
    )
    columns = list(rows[0])
    table = wandb.Table(columns=columns, data=[[row[column] for column in columns] for row in rows])
    run.log({"summary/noise_grid": table})
    # Histories make the standard responsive W&B line/scatter views useful;
    # the table remains the authoritative all-method comparison.
    for row in rows:
        run.log({
            "grid/noise_angstrom": row["noise_angstrom"],
            "grid/method": row["method"],
            "grid/native_topology_per_start": row["native_topology_per_start"],
            "grid/native_topology_per_local": row["native_topology_per_local"],
            "grid/local_index1_per_start": row["local_index1"] / row["starts"],
            "grid/calculator_valid_per_start": row["calculator_valid"] / row["starts"],
        })
    artifact = wandb.Artifact("t1x-gxtb-matched-noise-grid", type="benchmark-summary")
    artifact.add_file(str(args.manifest))
    artifact.add_file(str(json_path))
    artifact.add_file(str(markdown_path))
    run.log_artifact(artifact)
    run.summary.update({"cells": len(rows), "complete": True})
    run.finish()
    print(markdown_path)


if __name__ == "__main__":
    main()
