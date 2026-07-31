#!/usr/bin/env python3
"""Export every exact trajectory bundle in a completed campaign to W&B."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from gadplus.logging.wandb_export import export_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign_root", type=Path)
    parser.add_argument("--project", default="gadplus-ts-mechanisms")
    parser.add_argument("--entity", default=os.environ.get("WANDB_ENTITY"))
    parser.add_argument("--group", required=True)
    parser.add_argument("--job-type", default="competitive-gad")
    parser.add_argument("--mode", choices=("offline", "online"), default="online")
    parser.add_argument("--start-index", type=int, default=0, help="0-based inclusive bundle index")
    parser.add_argument("--stop-index", type=int, help="0-based exclusive bundle index")
    parser.add_argument("--max-runs", type=int, help="Evenly select at most this many trajectories from the requested range.")
    parser.add_argument(
        "--max-view-rows", type=int, default=600,
        help="Maximum event-preserving points in each interactive trajectory table.",
    )
    parser.add_argument(
        "--cockpit-chart-id",
        default="memo-ozdincer-university-of-toronto/gadplus-trajectory-cockpit-v3",
    )
    parser.add_argument(
        "--mechanism-chart-id",
        default="memo-ozdincer-university-of-toronto/gadplus-competitive-mechanism-v2",
    )
    args = parser.parse_args()

    bundles = sorted(
        path.parent for path in args.campaign_root.glob("trajectories/*/metadata.json")
    )
    if not bundles:
        raise SystemExit(f"no trajectory bundles found under {args.campaign_root}")
    stop = len(bundles) if args.stop_index is None else args.stop_index
    if not 0 <= args.start_index <= stop <= len(bundles):
        raise SystemExit("invalid --start-index/--stop-index range")
    positions = list(range(args.start_index, stop))
    if args.max_runs is not None and len(positions) > args.max_runs:
        if args.max_runs < 1:
            raise SystemExit("--max-runs must be positive")
        positions = sorted({round(item * (len(positions) - 1) / (args.max_runs - 1)) for item in range(args.max_runs)}) if args.max_runs > 1 else [0]
        positions = [args.start_index + item for item in positions]
    failures = []
    for index in positions:
        bundle = bundles[index]
        try:
            run_id = export_bundle(
                bundle,
                project=args.project,
                entity=args.entity,
                group=args.group,
                job_type=args.job_type,
                tags=("evaluation", "Transition1x", "g-xTB", args.job_type),
                mode=args.mode,
                cockpit_chart_id=args.cockpit_chart_id,
                mechanism_chart_id=args.mechanism_chart_id,
                max_view_rows=args.max_view_rows,
            )
            print(f"[{index + 1}/{len(bundles)}] {bundle.name} -> {run_id}")
        except Exception as error:  # noqa: BLE001 - continue an independent export batch.
            failures.append((bundle, error))
            print(f"[{index + 1}/{len(bundles)}] ERROR {bundle}: {type(error).__name__}: {error}")
    if failures:
        raise SystemExit(f"{len(failures)} of {len(positions)} selected exports failed")


if __name__ == "__main__":
    main()
