#!/usr/bin/env python3
"""Register versioned GADplus Vega chart presets with W&B."""

from __future__ import annotations

import argparse
import json
import os
from importlib.resources import files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entity", default=os.environ.get("WANDB_ENTITY"))
    parser.add_argument("--access", choices=("private", "public"), default="private")
    parser.add_argument("--version", default="v1")
    args = parser.parse_args()

    import wandb

    api = wandb.Api()
    entity = args.entity or api.default_entity
    if not entity:
        raise SystemExit("Set WANDB_ENTITY or pass --entity")

    chart_root = files("gadplus.logging").joinpath("vega")
    definitions = (
        (
            "gadplus-trajectory-cockpit",
            "GADplus trajectory cockpit",
            "trajectory_cockpit.json",
        ),
        (
            "gadplus-competitive-mechanism",
            "GADplus competitive mechanism",
            "competitive_mechanism.json",
        ),
    )
    for base_name, display_name, filename in definitions:
        with chart_root.joinpath(filename).open(encoding="utf-8") as handle:
            specification = json.load(handle)
        chart_id = api.create_custom_chart(
            entity=entity,
            name=f"{base_name}-{args.version}",
            display_name=f"{display_name} ({args.version})",
            spec_type="vega2",
            access=args.access,
            spec=specification,
        )
        print(chart_id)


if __name__ == "__main__":
    main()
