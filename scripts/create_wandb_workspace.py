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
        "--trajectory-only",
        action="store_true",
        help="Create a one-run inspector instead of an aggregate comparison workspace.",
    )
    parser.add_argument(
        "--cockpit-chart-id",
        default="memo-ozdincer-university-of-toronto/gadplus-trajectory-cockpit-v3",
    )
    parser.add_argument(
        "--mechanism-chart-id",
        default="memo-ozdincer-university-of-toronto/gadplus-competitive-mechanism-v2",
    )
    parser.add_argument(
        "--regular-gad-chart-id",
        default="memo-ozdincer-university-of-toronto/gadplus-regular-gad-mechanism-v2",
    )
    parser.add_argument(
        "--sella-chart-id",
        default="memo-ozdincer-university-of-toronto/gadplus-sella-mechanism-v2",
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
        query={"summaryTable": {"tableKey": "trajectory_view"}},
        chart_name=args.cockpit_chart_id,
        chart_fields={field: field for field in COCKPIT_FIELDS},
        layout=wr.Layout(x=0, y=0, w=24, h=34),
    )
    mechanism = wr.CustomChart(
        query={"summaryTable": {"tableKey": "trajectory_view"}},
        chart_name=args.mechanism_chart_id,
        chart_fields={field: field for field in COMPETITIVE_FIELDS},
        layout=wr.Layout(x=0, y=0, w=24, h=26),
    )
    regular_mechanism = wr.CustomChart(
        query={"summaryTable": {"tableKey": "trajectory_view"}},
        chart_name=args.regular_gad_chart_id,
        chart_fields={
            field: field for field in (
                "evaluation", "dt_eff", "disp_from_last", "mode_overlap",
                "eigvec_continuity", "grad_v0_overlap", "grad_v1_overlap",
                "lambda1", "lambda2", "n_neg",
            )
        },
        layout=wr.Layout(x=0, y=0, w=24, h=26),
    )
    sella_mechanism = wr.CustomChart(
        query={"summaryTable": {"tableKey": "trajectory_view"}},
        chart_name=args.sella_chart_id,
        chart_fields={
            field: field for field in (
                "evaluation", "force_max", "force_rms", "energy_from_start",
                "wall_time_s", "lambda1_scaled", "lambda2_scaled", "lambda3_scaled", "n_neg",
            )
        },
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

    trajectory_sections = [
        ws.Section(name="Trajectory cockpit — select one run", panels=[cockpit], is_open=True, pinned=True),
        ws.Section(name="Competitive GAD mechanism", panels=[mechanism], is_open=True, pinned=True),
        ws.Section(name="Ordinary GAD mechanism", panels=[regular_mechanism], is_open=True, pinned=True),
        ws.Section(name="Sella mechanism", panels=[sella_mechanism], is_open=True, pinned=True),
        ws.Section(
            name="Exact trajectory record",
            panels=[wr.WeavePanelSummaryTable(table_name="trajectory_view", layout=wr.Layout(x=0, y=0, w=24, h=18))],
            is_open=False,
        ),
    ]
    # The overview must remain fast: it contains only scalar population
    # summaries. High-resolution charts and tables are deliberately loaded
    # only from the one-run inspector (or directly from a run page).
    full_sections = [ws.Section(name="Campaign outcomes", panels=outcome_panels, is_open=True)]

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
            max_runs=1 if args.trajectory_only else 10,
            point_visualization_method="bucketing",
        ),
        sections=trajectory_sections if args.trajectory_only else full_sections,
    )
    workspace.save()
    print(workspace.url)


if __name__ == "__main__":
    main()
