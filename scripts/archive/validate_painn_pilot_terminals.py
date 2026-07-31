#!/usr/bin/env python
"""IRC_TOPO-validate strict terminal successes from a paired PaiNN pilot."""
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


def _jsonable(value):
    return value.item() if isinstance(value, np.generic) else value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--irc-max-steps", type=int, default=500)
    parser.add_argument("--relax-fmax", type=float, default=0.005)
    parser.add_argument("--relax-max-steps", type=int, default=500)
    parser.add_argument("--rmsd-threshold", type=float, default=0.3)
    args = parser.parse_args()

    from gadplus.calculator.neuralneb import (
        NEURALNEB_MODELS_DIR,
        NeuralNebPaiNNCalculator,
        make_neuralneb_predict_fn,
    )
    from gadplus.search.irc_full_hessian import run_irc_full_hessian
    from gadplus.search.irc_validate import bond_graphs_match, coords_to_bond_graph, score_endpoints
    from gadplus.search.native_endpoints import relax_to_minimum

    input_rows = pq.read_table(args.summary).to_pylist()
    strict_rows = [row for row in input_rows if bool(row["strict_converged"])]
    if not strict_rows:
        raise ValueError("No strict successes to validate")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(args.checkpoint) if args.checkpoint else NEURALNEB_MODELS_DIR / "painn0.sd"
    predict_fn = make_neuralneb_predict_fn(
        NeuralNebPaiNNCalculator(checkpoint=checkpoint, device=args.device)
    )
    candidate_cache: dict[str, dict[str, np.ndarray]] = {}
    rows: list[dict] = []
    for source in strict_rows:
        started = time.monotonic()
        name = str(source["candidate_file"])
        if name not in candidate_cache:
            with np.load(args.candidate_dir / name, allow_pickle=False) as data:
                candidate_cache[name] = {key: data[key].copy() for key in ("atomic_numbers", "reactant", "product")}
        candidate = candidate_cache[name]
        atomic_nums = torch.as_tensor(candidate["atomic_numbers"], dtype=torch.long, device=args.device)
        reactant = torch.as_tensor(candidate["reactant"], dtype=torch.float64)
        product = torch.as_tensor(candidate["product"], dtype=torch.float64)
        terminal = torch.as_tensor(source["coords_flat"], dtype=torch.float32, device=args.device).reshape(-1, 3)
        row = {
            "candidate_file": name,
            "sample_id": int(source["sample_id"]),
            "method": str(source["method"]),
            "noise_pm": float(source["noise_pm"]),
            "seed": int(source["seed"]),
            "strict_converged": True,
            "terminal_topology_intended": False,
            "terminal_rmsd_intended": False,
            "terminal_endpoint_minima": False,
            "terminal_accepted": False,
            "error": "",
        }
        try:
            irc = run_irc_full_hessian(
                ts_coords=terminal,
                atomic_nums=atomic_nums,
                predict_fn=predict_fn,
                reactant_coords=reactant,
                product_coords=product,
                rmsd_threshold=args.rmsd_threshold,
                max_steps=args.irc_max_steps,
                eckart_project=True,
            )
            forward = relax_to_minimum(
                irc.forward_coords, atomic_nums, predict_fn,
                fmax=args.relax_fmax, max_steps=args.relax_max_steps,
            ) if irc.forward_coords is not None else None
            reverse = relax_to_minimum(
                irc.reverse_coords, atomic_nums, predict_fn,
                fmax=args.relax_fmax, max_steps=args.relax_max_steps,
            ) if irc.reverse_coords is not None else None
            score = score_endpoints(
                forward.coords if forward else None,
                reverse.coords if reverse else None,
                atomic_nums, reactant, product, args.rmsd_threshold, predict_fn=predict_fn,
            )
            reference_graph_same = bond_graphs_match(
                coords_to_bond_graph(reactant, atomic_nums),
                coords_to_bond_graph(product, atomic_nums),
            )
            connectivity = bool(score.intended if reference_graph_same else score.topology_intended)
            endpoint_minima = bool(
                forward and reverse and forward.converged and reverse.converged
                and score.forward_n_neg_vib == 0 and score.reverse_n_neg_vib == 0
            )
            row.update(
                terminal_topology_intended=bool(score.topology_intended),
                terminal_rmsd_intended=bool(score.intended),
                terminal_endpoint_minima=endpoint_minima,
                reference_graph_same=bool(reference_graph_same),
                terminal_accepted=bool(connectivity and endpoint_minima),
                forward_force_max=float(forward.force_max) if forward else None,
                reverse_force_max=float(reverse.force_max) if reverse else None,
            )
            for key, value in dataclasses.asdict(score).items():
                if key not in {"forward_coords", "reverse_coords"}:
                    row[f"irc_{key}"] = _jsonable(value)
        except Exception as exc:
            row["error"] = repr(exc)
        row["wall_time_s"] = time.monotonic() - started
        rows.append(row)
        pq.write_table(pa.Table.from_pylist(rows), args.output_dir / "summary.parquet")
        with (args.output_dir / "summary.json").open("w") as handle:
            json.dump(rows, handle, indent=2, sort_keys=True, default=_jsonable)
        print(json.dumps(row, sort_keys=True, default=_jsonable), flush=True)
    print(json.dumps({"terminal_accepted": sum(row["terminal_accepted"] for row in rows), "strict_rows": len(rows)}))


if __name__ == "__main__":
    main()
