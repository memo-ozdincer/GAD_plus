#!/usr/bin/env python
"""Apply the common T1x n_neg/fmax and IRC_TOPO gates to g-xTB candidates."""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def _to_jsonable(value):
    if isinstance(value, np.generic):
        return value.item()
    return value


def _candidate_rows(paths: list[Path], fmax: float):
    for path in paths:
        for row in pq.read_table(path).to_pylist():
            n_neg = int(row.get("final_n_neg", -1))
            final_fmax = float(row.get("final_force_max", float("inf")))
            coords = row.get("final_coords_flat")
            gate = n_neg == 1 and np.isfinite(final_fmax) and final_fmax < fmax
            if gate and coords:
                yield path, row, np.asarray(coords, dtype=np.float64).reshape(-1, 3)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--h5", default=os.environ.get("GADPLUS_T1X_H5", "data/transition1x.h5"),
        help="Transition1x HDF5 path (or set GADPLUS_T1X_H5).",
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--fmax", type=float, default=0.01)
    parser.add_argument("--irc-steps", type=int, default=500)
    parser.add_argument("--irc-fmax", type=float, default=0.01)
    parser.add_argument("--relax-steps", type=int, default=500)
    parser.add_argument("--relax-fmax", type=float, default=0.001)
    parser.add_argument("--rmsd-threshold", type=float, default=0.3)
    parser.add_argument("--max-validate", type=int, default=0)
    parser.add_argument(
        "--candidate-index", type=int, default=None,
        help="Validate one zero-based TS-gate-passing row; for Slurm arrays.",
    )
    parser.add_argument(
        "--gxtb-executable",
        default=os.environ.get("GADPLUS_GXTB_EXE", "g-xtb/xtb-6.7.1/bin/xtb"),
    )
    args = parser.parse_args()

    from gadplus.calculator.gxtb import load_gxtb_calculator, make_gxtb_predict_fn
    from gadplus.data.direct_t1x import load_t1x_records_direct
    from gadplus.search.irc_full_hessian import run_irc_full_hessian
    from gadplus.search.irc_validate import score_endpoints
    from gadplus.search.native_endpoints import relax_to_minimum

    candidates = list(_candidate_rows(args.summary, args.fmax))
    if args.candidate_index is not None:
        if args.candidate_index < 0 or args.candidate_index >= len(candidates):
            raise IndexError(
                f"candidate index {args.candidate_index} is outside 0..{len(candidates) - 1}"
            )
        candidates = [candidates[args.candidate_index]]
    elif args.max_validate:
        candidates = candidates[: args.max_validate]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not candidates:
        pq.write_table(pa.Table.from_pylist([]), args.output_dir / "irc_topo.parquet")
        print("No candidates passed n_neg=1 and fmax gate", flush=True)
        return

    ids = [int(row["sample_id"]) for _path, row, _coords in candidates]
    records = load_t1x_records_direct(args.h5, args.split, ids)
    predict_fn = make_gxtb_predict_fn(load_gxtb_calculator(
        executable=args.gxtb_executable, n_threads=1,
        parallel=int(os.environ.get("GADPLUS_GXTB_PARALLEL", "1")),
    ))
    results = []
    for source, candidate, ts_coords in candidates:
        started = time.monotonic()
        sid = int(candidate["sample_id"])
        record = records[sid]
        z = torch.as_tensor(record.atomic_nums, dtype=torch.long)
        reactant = torch.as_tensor(record.reactant, dtype=torch.float64)
        product = (torch.as_tensor(record.product, dtype=torch.float64)
                   if record.product is not None else None)
        row = {
            "source_summary": str(source), "sample_id": sid,
            "formula": record.formula, "ts_gate_nneg_fmax": True,
            "irc_topology_intended": False, "endpoint_minima": False,
            "irc_topo_accepted": False, "error": "",
        }
        try:
            irc = run_irc_full_hessian(
                ts_coords=torch.as_tensor(ts_coords, dtype=torch.float64),
                atomic_nums=z, predict_fn=predict_fn,
                reactant_coords=reactant, product_coords=product,
                rmsd_threshold=args.rmsd_threshold, max_steps=args.irc_steps,
                fmax=args.irc_fmax, eckart_project=True,
            )
            forward = (relax_to_minimum(irc.forward_coords, z, predict_fn,
                                         fmax=args.relax_fmax, max_steps=args.relax_steps)
                       if irc.forward_coords is not None else None)
            reverse = (relax_to_minimum(irc.reverse_coords, z, predict_fn,
                                         fmax=args.relax_fmax, max_steps=args.relax_steps)
                       if irc.reverse_coords is not None else None)
            score = score_endpoints(
                forward.coords if forward else None, reverse.coords if reverse else None,
                z, reactant, product, args.rmsd_threshold, predict_fn=predict_fn,
            )
            row.update({
                "irc_topology_intended": bool(score.topology_intended),
                "irc_topology_half_intended": bool(score.topology_half_intended),
                "forward_n_neg_vib": score.forward_n_neg_vib,
                "reverse_n_neg_vib": score.reverse_n_neg_vib,
                "forward_relax_converged": bool(forward and forward.converged),
                "reverse_relax_converged": bool(reverse and reverse.converged),
            })
            row["endpoint_minima"] = bool(
                forward and reverse and forward.converged and reverse.converged
                and score.forward_n_neg_vib == 0 and score.reverse_n_neg_vib == 0
            )
            row["irc_topo_accepted"] = bool(
                row["ts_gate_nneg_fmax"] and row["irc_topology_intended"]
                and row["endpoint_minima"]
            )
        except Exception as exc:
            row["error"] = repr(exc)
        row["wall_time_s"] = time.monotonic() - started
        results.append(row)
        print(json.dumps(row, sort_keys=True, default=_to_jsonable), flush=True)
        pq.write_table(pa.Table.from_pylist(results), args.output_dir / "irc_topo.parquet")

    accepted = sum(bool(row["irc_topo_accepted"]) for row in results)
    print(json.dumps({"accepted": accepted, "total": len(results)}), flush=True)


if __name__ == "__main__":
    main()
