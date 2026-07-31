#!/usr/bin/env python3
"""Publish compact, parameter-explicit g-xTB and LJ method tables to W&B.

Each row is a *method × noise × panel* aggregate, never an individual
trajectory.  The readable label contains the essential controls; the complete
JSON parameter payload is kept in a separate table column for click-through
inspection without crowding the outcome columns.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


GXTB_METHODS: dict[str, tuple[str, dict[str, Any]]] = {
    "regular_gad": ("ordinary GAD [dt=0.003]", {"dt": 0.003, "mode": "instantaneous lowest", "projection": "Eckart"}),
    "competitive": ("competitive GAD [τs=0.01, η=0.01]", {"variant": "competitive", "spectral_temperature": 0.01, "step_fraction": 0.01}),
    "competitive_subspace": ("CS²-GAD [τs=0.01, η=0.01]", {"variant": "competitive_subspace", "spectral_temperature": 0.01, "step_fraction": 0.01}),
    "sella": ("Sella [Cartesian + Eckart]", {"coordinates": "Cartesian", "projection": "Eckart", "calculator": "g-xTB"}),
}

LJ_METHODS: dict[str, tuple[str, dict[str, Any]]] = {
    "ordinary_gad": ("ordinary GAD [dt=0.005, cap=0.005]", {"dt": 0.005, "max_atom_displacement": 0.005, "max_evaluations": 8000, "mode": "instantaneous lowest"}),
    "hard_gate": ("hard λ₂ gate [dt=0.005, cap=0.005]", {"gate": "λ2 >= 0", "dt": 0.005, "max_atom_displacement": 0.005, "max_evaluations": 8000}),
    "historical_lambda2": ("smooth λ₂ gate [k=50, dt=0.005, cap=0.005]", {"gate": "sigmoid(50 λ2)", "dt": 0.005, "max_atom_displacement": 0.005, "max_evaluations": 8000}),
    "intrinsic": ("pointwise intrinsic GAD [τs=0.01, η=0.05]", {"spectral_temperature": 0.01, "step_fraction": 0.05, "max_evaluations": 200}),
}


def _gxtb_rows(path: Path) -> list[dict[str, Any]]:
    rows = json.loads(path.read_text())
    output = []
    for row in rows:
        label, hparams = GXTB_METHODS[row["method"]]
        output.append({
            "surface": "Transition1x / g-xTB", "method": row["method"], "method_label": label,
            "noise": row["noise_angstrom"], "noise_unit": "Å", "panel": "matched test split",
            "starts": row["starts"], "calculator_valid": row["calculator_valid"],
            "local_index1": row["local_index1"], "strict_index1": row["strict_index1"],
            "two_branch_endpoints": row["endpoint_minima"], "labelled_topology": row["native_topology"],
            "two_branch_rate": row["endpoint_minima"] / row["starts"],
            "labelled_topology_rate": row["native_topology_per_start"],
            "budget": row["budget_updates"], "hyperparameters": json.dumps(hparams, sort_keys=True),
        })
    return output


def _lj_rows(path: Path) -> list[dict[str, Any]]:
    rows = json.loads(path.read_text())
    output = []
    for row in rows:
        label, hparams = LJ_METHODS[row["method"]]
        output.append({
            "surface": "analytic reduced LJ7", "method": row["method"], "method_label": label,
            "noise": row["noise_sigma"], "noise_unit": "σ", "panel": row["panel"],
            "starts": row["n"], "calculator_valid": row["n"] - row["exceptions"],
            "local_index1": row["strict"], "strict_index1": row["strict"],
            "two_branch_endpoints": row["downhill_valid"], "labelled_topology": row["correct_event"],
            "two_branch_rate": row["downhill_valid_rate"], "labelled_topology_rate": row["correct_event_rate"],
            "budget": hparams["max_evaluations"],
            "median_evaluations": row["median_strict_evaluations"],
            "hyperparameters": json.dumps(hparams, sort_keys=True),
        })
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gxtb-summary", type=Path, default=Path("/scratch/memoozd/gadplus/analysis/t1x-gxtb-matched-noise-grid/summary.json"))
    parser.add_argument("--lj-summary", type=Path, default=Path("/scratch/memoozd/gadplus/runs/lj-method-progression-1946071/summary.json"))
    parser.add_argument("--project", default="gadplus-ts-mechanisms")
    parser.add_argument("--entity", default=os.environ.get("WANDB_ENTITY"))
    parser.add_argument("--group", default="method-comparison-tables")
    parser.add_argument("--mode", choices=("online", "offline"), default="online")
    args = parser.parse_args()

    import wandb

    gxtb, lj = _gxtb_rows(args.gxtb_summary), _lj_rows(args.lj_summary)
    columns = sorted({key for row in gxtb + lj for key in row})
    run = wandb.init(
        project=args.project, entity=args.entity, group=args.group, job_type="method-comparison-tables",
        name="current-method-comparison-tables", mode=args.mode,
        config={"gxtb_source": str(args.gxtb_summary), "lj_source": str(args.lj_summary), "table_granularity": "method × noise × panel"},
        tags=("evaluation", "methods-table", "g-xTB", "LJ7"),
    )
    run.log({
        "methods/gxtb": wandb.Table(columns=columns, data=[[row.get(key) for key in columns] for row in gxtb]),
        "methods/lj7": wandb.Table(columns=columns, data=[[row.get(key) for key in columns] for row in lj]),
    })
    run.summary.update({"gxtb_rows": len(gxtb), "lj7_rows": len(lj), "complete": True})
    run.finish()


if __name__ == "__main__":
    main()
