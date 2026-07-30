#!/usr/bin/env python3
"""Replay one exact GADplus trajectory bundle into W&B."""

from __future__ import annotations

import argparse
import os

from gadplus.logging.wandb_export import export_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle_dir")
    parser.add_argument("--project", default="gadplus-ts-mechanisms")
    parser.add_argument("--entity", default=os.environ.get("WANDB_ENTITY"))
    parser.add_argument("--group")
    parser.add_argument("--job-type", default="competitive-gad")
    parser.add_argument("--tag", action="append", default=[])
    parser.add_argument("--mode", choices=("offline", "online", "disabled"), default="offline")
    parser.add_argument("--max-view-rows", type=int, default=2500)
    parser.add_argument("--cockpit-chart-id", default=os.environ.get("GADPLUS_WANDB_COCKPIT"))
    parser.add_argument("--mechanism-chart-id", default=os.environ.get("GADPLUS_WANDB_COMPETITIVE"))
    args = parser.parse_args()
    run_id = export_bundle(
        args.bundle_dir,
        project=args.project,
        entity=args.entity,
        group=args.group,
        job_type=args.job_type,
        tags=args.tag,
        mode=args.mode,
        max_view_rows=args.max_view_rows,
        cockpit_chart_id=args.cockpit_chart_id,
        mechanism_chart_id=args.mechanism_chart_id,
    )
    print(run_id)


if __name__ == "__main__":
    main()
