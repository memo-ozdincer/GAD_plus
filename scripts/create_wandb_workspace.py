#!/usr/bin/env python3
"""Create the reproducible GADplus W&B workspace view."""

from __future__ import annotations

import argparse
import os

from gadplus.logging.wandb_export import COCKPIT_FIELDS, COMPETITIVE_FIELDS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entity", default=os.environ.get("WANDB_ENTITY"))
    parser.add_argument("--project", default="gadplus-ts-mechanisms")
    parser.add_argument("--name", default="GADplus TS mechanisms v1")
    parser.add_argument(
        "--cockpit-chart-id",
        default="memo-ozdincer-university-of-toronto/gadplus-trajectory-cockpit-v1",
    )
    parser.add_argument(
        "--mechanism-chart-id",
        default="memo-ozdincer-university-of-toronto/gadplus-competitive-mechanism-v1",
    )
    args = parser.parse_args()

    import wandb
    import wandb_workspaces.reports.v2 as wr
    import wandb_workspaces.workspaces as ws

    entity = args.entity or wandb.Api().default_entity
    if not entity:
        raise SystemExit("Set WANDB_ENTITY or pass --entity")

    outcome_panels = []
    for position, (metric, title) in enumerate(
        (
            ("calculator_valid", "Calculator-valid starts"),
            ("local_ts", "Local TS / valid starts"),
            ("strict_ts", "Strict TS / valid starts"),
            ("native_topology", "Native topology / valid starts"),
            ("n_evaluations", "Median Hessian evaluations"),
            ("final_force_max", "Final fmax"),
        )
    ):
        aggregate = "median" if metric in {"n_evaluations", "final_force_max"} else "mean"
        outcome_panels.append(
            wr.ScalarChart(
                title=title,
                metric=metric,
                groupby_aggfunc=aggregate,
                groupby_rangefunc="none",
                font_size="medium",
                layout=wr.Layout(x=(position % 3) * 8, y=(position // 3) * 6, w=8, h=6),
            )
        )

    cockpit = wr.CustomChart(
        query={"summaryTable": {"tableKey": "trajectory_cockpit_table"}},
        chart_name=args.cockpit_chart_id,
        chart_fields={field: field for field in COCKPIT_FIELDS},
        layout=wr.Layout(x=0, y=0, w=24, h=34),
    )
    mechanism = wr.CustomChart(
        query={"summaryTable": {"tableKey": "competitive_mechanism_table"}},
        chart_name=args.mechanism_chart_id,
        chart_fields={field: field for field in COMPETITIVE_FIELDS},
        layout=wr.Layout(x=0, y=0, w=24, h=26),
    )
    common_diagnostics = (
        wr.LinePlot(
            title="Force ratio (native full-fidelity panel)",
            x="trajectory/evaluation",
            y=["trajectory/force_ratio_display"],
            log_y=True,
            smoothing_type="none",
            ignore_outliers=False,
            point_visualization_method="bucketing-gorilla",
            layout=wr.Layout(x=0, y=0, w=12, h=8),
        ),
        wr.LinePlot(
            title="Normalized lowest spectrum",
            x="trajectory/evaluation",
            y=[
                "trajectory/lambda1_scaled",
                "trajectory/lambda2_scaled",
                "trajectory/lambda3_scaled",
            ],
            smoothing_type="none",
            ignore_outliers=False,
            point_visualization_method="bucketing-gorilla",
            line_colors={
                "trajectory/lambda1_scaled": "#D55E00",
                "trajectory/lambda2_scaled": "#CC79A7",
                "trajectory/lambda3_scaled": "#009E73",
            },
            layout=wr.Layout(x=12, y=0, w=12, h=8),
        ),
        wr.LinePlot(
            title="Competitive gate and activity",
            x="trajectory/evaluation",
            y=[
                "trajectory/lambda2_gate",
                "trajectory/effective_gate",
                "trajectory/activity_fraction",
            ],
            range_y=(0.0, 1.0),
            smoothing_type="none",
            ignore_outliers=False,
            point_visualization_method="bucketing-gorilla",
            layout=wr.Layout(x=0, y=8, w=12, h=8),
        ),
        wr.LinePlot(
            title="Hindsight distance to terminal",
            x="trajectory/evaluation",
            y=["trajectory/distance_to_terminal_display"],
            log_y=True,
            smoothing_type="none",
            ignore_outliers=False,
            point_visualization_method="bucketing-gorilla",
            layout=wr.Layout(x=12, y=8, w=12, h=8),
        ),
    )

    workspace = ws.Workspace(
        entity=entity,
        project=args.project,
        name=args.name,
        auto_generate_panels=False,
        settings=ws.WorkspaceSettings(
            x_axis="trajectory/evaluation",
            smoothing_type="none",
            smoothing_weight=0,
            ignore_outliers=False,
            sort_panels_alphabetically=False,
            tooltip_number_of_runs="single",
            max_runs=10,
            point_visualization_method="bucketing",
        ),
        sections=[
            ws.Section(name="Campaign outcomes", panels=outcome_panels, is_open=True),
            ws.Section(name="Trajectory cockpit", panels=[cockpit], is_open=True, pinned=True),
            ws.Section(
                name="Competitive GAD mechanism",
                panels=[mechanism],
                is_open=True,
                pinned=True,
            ),
            ws.Section(
                name="Native full-fidelity diagnostics",
                panels=list(common_diagnostics),
                is_open=False,
            ),
            ws.Section(
                name="Exact trajectory records",
                panels=[
                    wr.WeavePanelSummaryTable(
                        table_name="trajectory_view",
                        layout=wr.Layout(x=0, y=0, w=24, h=18),
                    )
                ],
                is_open=False,
            ),
        ],
    )
    workspace.save()
    print(workspace.url)


if __name__ == "__main__":
    main()
