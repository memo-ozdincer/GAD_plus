#!/usr/bin/env python
"""Validate calculator-native NEB saddles by full-Hessian IRC and relaxation.

Candidates are produced independently by endpoint relaxation plus NEB. This
script only decides whether their two downhill branches return to the two
calculator-native minima stored in each candidate file.
"""
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


def _as_jsonable(value):
    if isinstance(value, np.generic):
        return value.item()
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("neuralneb", "horm"), default="neuralneb")
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--irc-max-steps", type=int, default=500)
    parser.add_argument("--irc-fmax", type=float, default=0.01)
    parser.add_argument("--relax-fmax", type=float, default=0.005)
    parser.add_argument("--relax-max-steps", type=int, default=500)
    parser.add_argument("--rmsd-threshold", type=float, default=0.3)
    args = parser.parse_args()

    from gadplus.search.irc_full_hessian import run_irc_full_hessian
    from gadplus.search.irc_validate import (
        bond_graphs_match,
        coords_to_bond_graph,
        score_endpoints,
    )
    from gadplus.search.native_endpoints import relax_to_minimum

    candidate_paths = sorted(args.candidate_dir.glob("candidate_*.npz"))
    if not candidate_paths:
        raise FileNotFoundError(f"No candidates under {args.candidate_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.backend == "neuralneb":
        from gadplus.calculator.neuralneb import (
            NEURALNEB_MODELS_DIR,
            NeuralNebPaiNNCalculator,
            make_neuralneb_predict_fn,
        )

        checkpoint = Path(args.checkpoint) if args.checkpoint else NEURALNEB_MODELS_DIR / "painn0.sd"
        predict_fn = make_neuralneb_predict_fn(
            NeuralNebPaiNNCalculator(checkpoint=checkpoint, device=args.device)
        )
    else:
        from gadplus.calculator.horm import (
            HORM_LEFTNET_CHECKPOINT,
            HormLeftNetCalculator,
            make_horm_predict_fn,
        )

        checkpoint = Path(args.checkpoint) if args.checkpoint else Path(HORM_LEFTNET_CHECKPOINT)
        predict_fn = make_horm_predict_fn(
            HormLeftNetCalculator(checkpoint=str(checkpoint), device=args.device)
        )

    rows: list[dict] = []
    for candidate_path in candidate_paths:
        started = time.monotonic()
        with np.load(candidate_path, allow_pickle=False) as candidate:
            sample_id = int(candidate["sample_id"].reshape(()).item())
            atomic_nums = torch.as_tensor(candidate["atomic_numbers"], dtype=torch.long, device=args.device)
            ts_coords = torch.as_tensor(candidate["coords"], dtype=torch.float32, device=args.device)
            reactant = torch.as_tensor(candidate["reactant"], dtype=torch.float64)
            product = torch.as_tensor(candidate["product"], dtype=torch.float64)

        row = {
            "candidate": candidate_path.name,
            "sample_id": sample_id,
            "irc_completed": False,
            "forward_relaxed": False,
            "reverse_relaxed": False,
            "native_topology_intended": False,
            "native_rmsd_intended": False,
            "endpoint_minima": False,
            "accepted": False,
            "error": "",
        }
        try:
            irc = run_irc_full_hessian(
                ts_coords=ts_coords,
                atomic_nums=atomic_nums,
                predict_fn=predict_fn,
                reactant_coords=reactant,
                product_coords=product,
                rmsd_threshold=args.rmsd_threshold,
                max_steps=args.irc_max_steps,
                fmax=args.irc_fmax,
                eckart_project=True,
            )
            row["irc_completed"] = bool(
                irc.forward_coords is not None and irc.reverse_coords is not None
            )
            forward = (
                relax_to_minimum(
                    irc.forward_coords, atomic_nums, predict_fn,
                    fmax=args.relax_fmax, max_steps=args.relax_max_steps,
                ) if irc.forward_coords is not None else None
            )
            reverse = (
                relax_to_minimum(
                    irc.reverse_coords, atomic_nums, predict_fn,
                    fmax=args.relax_fmax, max_steps=args.relax_max_steps,
                ) if irc.reverse_coords is not None else None
            )
            row.update(
                forward_relaxed=bool(forward and forward.converged),
                reverse_relaxed=bool(reverse and reverse.converged),
                forward_force_max=(float(forward.force_max) if forward else None),
                reverse_force_max=(float(reverse.force_max) if reverse else None),
                forward_relax_steps=(int(forward.steps) if forward else None),
                reverse_relax_steps=(int(reverse.steps) if reverse else None),
            )
            score = score_endpoints(
                forward.coords if forward else None,
                reverse.coords if reverse else None,
                atomic_nums,
                reactant,
                product,
                args.rmsd_threshold,
                predict_fn=predict_fn,
            )
            reference_graph_same = bond_graphs_match(
                coords_to_bond_graph(reactant, atomic_nums),
                coords_to_bond_graph(product, atomic_nums),
            )
            score_fields = dataclasses.asdict(score)
            for name, value in score_fields.items():
                if name not in {"forward_coords", "reverse_coords"}:
                    row[f"irc_{name}"] = _as_jsonable(value)
            row["native_topology_intended"] = bool(score.topology_intended)
            row["native_rmsd_intended"] = bool(score.intended)
            row["reference_graph_same"] = bool(reference_graph_same)
            # IRC_TOPO is decisive when the two native references are
            # topologically distinct. If they share a graph (a conformer or
            # stereochemical path), topology cannot distinguish the endpoints
            # and the RMSD identity criterion remains necessary.
            row["connectivity_intended"] = bool(
                score.intended if reference_graph_same else score.topology_intended
            )
            row["endpoint_minima"] = bool(
                forward and reverse and forward.converged and reverse.converged
                and score.forward_n_neg_vib == 0 and score.reverse_n_neg_vib == 0
            )
            row["accepted"] = bool(
                row["connectivity_intended"] and row["endpoint_minima"]
            )
            np.savez_compressed(
                args.output_dir / f"{candidate_path.stem}_irc.npz",
                forward_raw=(irc.forward_coords if irc.forward_coords is not None else np.empty((0, 3))),
                reverse_raw=(irc.reverse_coords if irc.reverse_coords is not None else np.empty((0, 3))),
                forward_relaxed=(forward.coords if forward else np.empty((0, 3))),
                reverse_relaxed=(reverse.coords if reverse else np.empty((0, 3))),
            )
        except Exception as exc:
            row["error"] = repr(exc)
        row["wall_time_s"] = time.monotonic() - started
        rows.append(row)
        pq.write_table(pa.Table.from_pylist(rows), args.output_dir / "summary.parquet")
        with (args.output_dir / "summary.json").open("w") as handle:
            json.dump(rows, handle, indent=2, sort_keys=True, default=_as_jsonable)
        print(json.dumps(row, sort_keys=True, default=_as_jsonable), flush=True)

    print(json.dumps({"accepted": sum(row["accepted"] for row in rows), "total": len(rows)}))


if __name__ == "__main__":
    main()
