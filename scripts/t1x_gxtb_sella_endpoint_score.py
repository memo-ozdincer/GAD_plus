#!/usr/bin/env python3
"""Score g-xTB Sella saddles with the intrinsic pilot's native endpoint test."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sella-root", type=Path, required=True)
    p.add_argument(
        "--competitive-root", type=Path,
        help="Use intrinsic pilot task JSON instead of Sella parquet summaries.",
    )
    p.add_argument("--native-label-root", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--task-id", type=int, required=True)
    p.add_argument("--h5", default=os.environ["GADPLUS_T1X_H5"])
    p.add_argument("--gxtb-executable", required=True)
    p.add_argument("--parallel", type=int, default=32)
    p.add_argument("--fmax", type=float, default=0.03)
    return p.parse_args()


def candidates(root: Path, fmax: float, competitive_root: Path | None = None) -> list[dict]:
    if competitive_root is not None:
        rows = [
            json.loads(path.read_text())
            for path in sorted((competitive_root / "tasks").glob("task_*.json"))
        ]
        return [
            {
                "sample_id": row["sample_id"],
                "noise_angstrom": row["noise_angstrom"],
                "final_coords_flat": np.asarray(row["final_coords"], dtype=float).reshape(-1).tolist(),
            }
            for row in sorted(rows, key=lambda r: r.get("sample_id", -1))
            if row.get("search_gate") and "final_coords" in row
        ]
    rows: list[dict] = []
    for path in sorted(root.glob("task_*/summary_*.parquet")):
        rows.extend(pq.read_table(path).to_pylist())
    return [
        row for row in sorted(
            rows,
            key=lambda r: (r["sample_id"], float(r.get("noise_angstrom", 0.0))),
        )
        if int(row["final_n_neg"]) == 1 and float(row["final_force_max"]) < fmax
    ]


def main() -> None:
    a = parse_args()
    from gadplus.calculator.gxtb import load_gxtb_calculator, make_gxtb_predict_fn
    from gadplus.data.direct_t1x import load_t1x_records_direct
    from gadplus.projection import atomic_nums_to_symbols, get_mass_weights, vib_eig
    from gadplus.search.irc_validate import score_endpoints
    from gadplus.search.native_endpoints import relax_to_minimum

    torch.set_num_threads(1)
    rows = candidates(a.sella_root, a.fmax, a.competitive_root)
    if a.task_id < 0:
        raise ValueError("task-id must be nonnegative")
    if a.task_id >= len(rows):
        a.output_root.mkdir(parents=True, exist_ok=True)
        (a.output_root / f"task_{a.task_id:03d}.json").write_text(json.dumps({
            "task_id": a.task_id, "search_gate": False, "skipped": True,
            "native_endpoint_topology": False, "endpoint_minima": False,
            "error": "padding task",
        }, indent=2) + "\n")
        return
    source = rows[a.task_id]
    sid = int(source["sample_id"])
    record = load_t1x_records_direct(a.h5, "test", [sid])[sid]
    z = torch.as_tensor(record.atomic_nums, dtype=torch.long)
    qts = torch.as_tensor(source["final_coords_flat"], dtype=torch.float64).reshape(-1, 3)
    predict = make_gxtb_predict_fn(load_gxtb_calculator(
        executable=a.gxtb_executable, n_threads=1, parallel=a.parallel,
    ))
    row = {
        "task_id": a.task_id, "sample_id": sid,
        # Legacy projected-GAD summaries predate the noise column.  The
        # endpoint calculation itself is independent of that label.
        "noise_angstrom": source.get("noise_angstrom"),
        "search_gate": True, "endpoint_minima": False,
        "native_endpoint_topology": False, "error": "",
    }
    try:
        out = predict(qts, z, do_hessian=True, require_grad=False)
        symbols = atomic_nums_to_symbols(z)
        _evals, modes, _ = vib_eig(out["hessian"], qts, symbols)
        _, _, _, inv = get_mass_weights(symbols)
        direction = (inv * modes[:, 0]).reshape_as(qts)
        direction /= torch.linalg.vector_norm(direction).clamp_min(1e-12)
        ends = [relax_to_minimum(qts + sign * .05 * direction, z, predict, fmax=.001, max_steps=500)
                for sign in (-1., 1.)]
        reactant = json.loads((a.native_label_root / f"sample_{sid}_reactant.json").read_text())
        product = json.loads((a.native_label_root / f"sample_{sid}_product.json").read_text())
        score = score_endpoints(
            np.asarray(ends[0].coords), np.asarray(ends[1].coords), z,
            torch.as_tensor(reactant["coords"]), torch.as_tensor(product["coords"]),
            rmsd_threshold=.3, predict_fn=predict,
        )
        row["endpoint_minima"] = bool(all(e.converged and e.force_max < .001 for e in ends))
        row["native_endpoint_topology"] = bool(score.topology_intended)
    except Exception as exc:  # A calculator failure is an outcome.
        row["error"] = f"{type(exc).__name__}: {exc}"
    a.output_root.mkdir(parents=True, exist_ok=True)
    (a.output_root / f"task_{a.task_id:03d}.json").write_text(json.dumps(row, indent=2) + "\n")


if __name__ == "__main__":
    main()
