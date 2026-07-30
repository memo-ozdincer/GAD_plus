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
    parser.add_argument(
        "--cockpit-chart-id",
        default="memo-ozdincer-university-of-toronto/gadplus-trajectory-cockpit-v1",
    )
    parser.add_argument(
        "--mechanism-chart-id",
        default="memo-ozdincer-university-of-toronto/gadplus-competitive-mechanism-v1",
    )
    args = parser.parse_args()

    bundles = sorted(
        path.parent for path in args.campaign_root.glob("trajectories/*/metadata.json")
    )
    if not bundles:
        raise SystemExit(f"no trajectory bundles found under {args.campaign_root}")
    failures = []
    for index, bundle in enumerate(bundles, start=1):
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
            )
            print(f"[{index}/{len(bundles)}] {bundle.name} -> {run_id}")
        except Exception as error:  # noqa: BLE001 - continue an independent export batch.
            failures.append((bundle, error))
            print(f"[{index}/{len(bundles)}] ERROR {bundle}: {type(error).__name__}: {error}")
    if failures:
        raise SystemExit(f"{len(failures)} of {len(bundles)} exports failed")


if __name__ == "__main__":
    main()
