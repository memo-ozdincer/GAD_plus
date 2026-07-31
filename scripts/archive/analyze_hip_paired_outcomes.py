#!/usr/bin/env python
"""Summarize matched HIP GAD/Sella outcome classes from stored trajectories."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gad-dir", type=Path, required=True)
    parser.add_argument("--sella-dir", type=Path, required=True)
    parser.add_argument("--noise-pm", type=int, default=150)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def summary_path(directory: Path, noise_pm: int) -> Path:
    paths = sorted(directory.glob(f"summary_*_{noise_pm}pm.parquet"))
    if len(paths) != 1:
        raise RuntimeError(f"Expected one {noise_pm} pm summary in {directory}, found {paths}")
    return paths[0]


def trajectory_path(directory: Path, noise_pm: int, sample_id: int) -> Path:
    paths = sorted(directory.glob(f"traj_*_{noise_pm}pm_*_{sample_id}.parquet"))
    if len(paths) != 1:
        raise RuntimeError(f"Expected one trajectory for sample {sample_id}, found {paths}")
    return paths[0]


def first_or_nan(frame: pd.DataFrame, column: str) -> float:
    if column not in frame or frame.empty:
        return float("nan")
    return float(frame.iloc[0][column])


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    gad = pd.read_parquet(summary_path(args.gad_dir, args.noise_pm)).set_index("sample_id")
    sella = pd.read_parquet(summary_path(args.sella_dir, args.noise_pm)).set_index("sample_id")
    paired = gad[["converged", "converged_step", "total_steps"]].rename(
        columns={"converged": "gad_converged"}
    ).join(sella[["converged", "n_steps"]].rename(columns={"converged": "sella_converged"}))
    paired["outcome"] = np.select(
        [
            paired.gad_converged & paired.sella_converged,
            paired.gad_converged & ~paired.sella_converged,
            ~paired.gad_converged & paired.sella_converged,
        ],
        ["both", "gad_only", "sella_only"],
        default="neither",
    )

    rows: list[dict] = []
    columns = [
        "step", "n_neg", "eig0", "eig1", "force_max", "disp_from_last",
        "mode_overlap", "dist_to_known_ts",
    ]
    for sample_id, outcome in paired.outcome.items():
        trajectory = pd.read_parquet(
            trajectory_path(args.gad_dir, args.noise_pm, int(sample_id)), columns=columns
        ).sort_values("step")
        initial = trajectory.iloc[0]
        first_20 = trajectory.iloc[:20]
        first_100 = trajectory.iloc[:100]
        index_one = trajectory.loc[trajectory.n_neg == 1, "step"]
        rows.append(
            {
                "sample_id": int(sample_id),
                "outcome": outcome,
                "initial_n_neg": int(initial.n_neg),
                "initial_eig0": float(initial.eig0),
                "initial_eig1": float(initial.eig1),
                "initial_eigengap": float(initial.eig1 - initial.eig0),
                "initial_fmax": float(initial.force_max),
                "first_step_norm": float(trajectory.iloc[1].disp_from_last)
                if len(trajectory) > 1 else float("nan"),
                "first_index_one_step": float(index_one.iloc[0]) if len(index_one) else float("nan"),
                "early_mode_overlap_median": float(first_20.mode_overlap.iloc[1:].median()),
                "min_distance_to_reference_ts_first100": float(first_100.dist_to_known_ts.min()),
            }
        )
    per_sample = pd.DataFrame(rows)
    metrics = [column for column in per_sample.columns if column not in {"sample_id", "outcome"}]
    grouped = per_sample.groupby("outcome")[metrics].median()
    grouped.insert(0, "n", per_sample.groupby("outcome").size())
    grouped = grouped.reindex(["both", "gad_only", "sella_only", "neither"])
    per_sample.to_parquet(args.output_dir / "per_sample.parquet", index=False)
    grouped.to_csv(args.output_dir / "outcome_medians.csv")
    payload = {
        "noise_pm_label": args.noise_pm,
        "counts": {key: int(value) for key, value in paired.outcome.value_counts().items()},
        "median_metrics": json.loads(grouped.to_json(orient="index")),
        "inputs": {"gad_dir": str(args.gad_dir), "sella_dir": str(args.sella_dir)},
    }
    (args.output_dir / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
