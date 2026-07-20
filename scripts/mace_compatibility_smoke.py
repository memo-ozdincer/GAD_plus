#!/usr/bin/env python
"""Smoke-test MACE-OFF23 against a small T1x subset on its own PES.

For each selected labelled transition state this records a full MACE Hessian,
a directional finite-difference check, and DFT-quality-free MACE relaxation of
the labelled reactant/product.  It is intentionally a compatibility screen,
not an optimizer benchmark.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def _parse_ids(value: str | None, n_samples: int) -> list[int]:
    if value is None:
        return list(range(n_samples))
    ids = [int(token) for token in value.split(",") if token.strip()]
    if not ids:
        raise ValueError("--sample-ids did not contain any ids")
    return ids


def _load_t1x_records(h5_path: str, split: str, max_sample_id: int) -> list[dict]:
    """Mirror Transition1xDataset's filtered ordering without torch_geometric."""
    from transition1x import Dataloader

    records = []
    for molecule in Dataloader(h5_path, datasplit=split, only_final=True):
        if len(records) > max_sample_id:
            break
        try:
            ts = molecule["transition_state"]
            reactant = molecule["reactant"]
            if len(ts["atomic_numbers"]) != len(reactant["atomic_numbers"]):
                continue
            product = molecule.get("product")
            has_product = (
                product is not None
                and len(product.get("atomic_numbers", [])) == len(ts["atomic_numbers"])
            )
            records.append({
                "atomic_nums": np.asarray(ts["atomic_numbers"], dtype=np.int64),
                "ts": np.asarray(ts["positions"], dtype=np.float64),
                "reactant": np.asarray(reactant["positions"], dtype=np.float64),
                "product": (
                    np.asarray(product["positions"], dtype=np.float64)
                    if has_product
                    else None
                ),
                "formula": str(ts.get("formula", "")),
                "rxn": str(ts.get("rxn", "")),
            })
        except Exception:
            continue
    if len(records) <= max_sample_id:
        raise IndexError(f"Requested sample {max_sample_id}, loaded {len(records)} records")
    return records


def _directional_hessian_error(predict_fn, coords, atomic_nums, hessian, epsilon):
    generator = torch.Generator(device="cpu").manual_seed(17)
    direction = torch.randn(coords.numel(), generator=generator, dtype=torch.float64)
    direction /= torch.linalg.vector_norm(direction)
    direction_3d = direction.reshape_as(coords)
    plus = predict_fn(coords + epsilon * direction_3d, atomic_nums, do_hessian=False)
    minus = predict_fn(coords - epsilon * direction_3d, atomic_nums, do_hessian=False)
    # H is d(grad)/dx and grad = -force.
    fd_hv = (-(plus["forces"].reshape(-1).double()) + minus["forces"].reshape(-1).double()) / (2 * epsilon)
    analytic_hv = hessian.double() @ direction
    error = fd_hv - analytic_hv
    return float(error.abs().max()), float(torch.sqrt(torch.mean(error.square())))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="small")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-samples", type=int, default=3)
    parser.add_argument("--sample-ids", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--h5", default="/lustre06/project/6033559/memoozd/data/transition1x.h5")
    parser.add_argument("--fd-epsilon", type=float, default=1e-3)
    parser.add_argument("--relax-fmax", type=float, default=0.01)
    parser.add_argument("--max-relax-steps", type=int, default=200)
    args = parser.parse_args()

    from gadplus.calculator.mace import (
        MACE_OFF23_ATOMIC_NUMBERS,
        load_mace_calculator,
        make_mace_predict_fn,
    )
    from gadplus.projection import atomic_nums_to_symbols, vib_eig
    from gadplus.search.native_endpoints import relax_to_minimum

    ids = _parse_ids(args.sample_ids, args.n_samples)
    records = _load_t1x_records(args.h5, args.split, max(ids))

    calculator = load_mace_calculator(model=args.model, device=args.device)
    predict_fn = make_mace_predict_fn(calculator)
    rows = []
    for sample_id in ids:
        sample = records[sample_id]
        z = torch.as_tensor(sample["atomic_nums"], dtype=torch.long)
        coords = torch.as_tensor(sample["ts"], dtype=torch.float64)
        row = {
            "sample_id": sample_id,
            "formula": sample["formula"],
            "rxn": sample["rxn"],
            "supported_elements": bool(set(z.tolist()).issubset(MACE_OFF23_ATOMIC_NUMBERS)),
            "ts_energy": None,
            "ts_force_max": None,
            "ts_n_neg": None,
            "hessian_symmetric_max": None,
            "fd_hessian_max_error": None,
            "fd_hessian_rms_error": None,
            "reactant_minimum": False,
            "product_minimum": False,
            "reactant_force_max": None,
            "product_force_max": None,
            "reactant_steps": None,
            "product_steps": None,
            "error": "",
        }
        try:
            if not row["supported_elements"]:
                raise ValueError(f"unsupported elements: {sorted(set(z.tolist()))}")
            out = predict_fn(coords, z, do_hessian=True)
            hessian = out["hessian"].reshape(3 * len(z), 3 * len(z))
            evals, _, _ = vib_eig(hessian, coords, atomic_nums_to_symbols(z))
            row["ts_energy"] = float(out["energy"])
            row["ts_force_max"] = float(out["forces"].abs().max())
            row["ts_n_neg"] = int((evals < -1e-4).sum())
            row["hessian_symmetric_max"] = float((hessian - hessian.T).abs().max())
            fd_max, fd_rms = _directional_hessian_error(
                predict_fn, coords, z, hessian, args.fd_epsilon,
            )
            row["fd_hessian_max_error"] = fd_max
            row["fd_hessian_rms_error"] = fd_rms

            reactant = relax_to_minimum(
                torch.as_tensor(sample["reactant"], dtype=torch.float64), z, predict_fn,
                fmax=args.relax_fmax, max_steps=args.max_relax_steps,
            )
            row["reactant_minimum"] = bool(reactant.converged and reactant.force_max <= args.relax_fmax)
            row["reactant_force_max"] = reactant.force_max
            row["reactant_steps"] = reactant.steps
            if sample["product"] is not None:
                product = relax_to_minimum(
                    torch.as_tensor(sample["product"], dtype=torch.float64), z, predict_fn,
                    fmax=args.relax_fmax, max_steps=args.max_relax_steps,
                )
                row["product_minimum"] = bool(product.converged and product.force_max <= args.relax_fmax)
                row["product_force_max"] = product.force_max
                row["product_steps"] = product.steps
        except Exception as exc:
            row["error"] = repr(exc)
        print(
            f"[{sample_id:>3}] {row['formula']:>14} supported={row['supported_elements']} "
            f"nneg={row['ts_n_neg']} Rmin={row['reactant_minimum']} "
            f"Pmin={row['product_minimum']} error={row['error']}",
            flush=True,
        )
        rows.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
