#!/usr/bin/env python
"""Create VS Code viewer bundles from saved IRC validation results."""

from __future__ import annotations

import argparse
import os

import duckdb
import numpy as np
from ase import Atoms

from visualize_3d import _write_viewer_bundle


def _coords_from_flat(values) -> np.ndarray | None:
    if values is None:
        return None
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return None
    return arr.reshape(-1, 3)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-parquet", type=str, required=True)
    parser.add_argument("--run-id", type=str, required=True)
    parser.add_argument("--sample-id", type=int, required=True)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    df = duckdb.execute(
        f"""
        SELECT *
        FROM '{args.results_parquet}'
        WHERE run_id = '{args.run_id}' AND sample_id = {args.sample_id}
        LIMIT 1
        """
    ).df()
    if len(df) == 0:
        raise ValueError("Requested IRC result row not found")

    row = df.iloc[0]
    z = None
    for flat_key in [
        "ts_coords_flat",
        "reactant_coords_flat",
        "product_coords_flat",
        "forward_coords_flat",
        "reverse_coords_flat",
    ]:
        coords = _coords_from_flat(row.get(flat_key))
        if coords is not None:
            n_atoms = len(coords)
            break
    else:
        raise ValueError("No coordinate payload found in result row")

    # Infer atomic numbers from the first available XYZ payload is impossible,
    # so this script expects the validation parquet to include them in the future.
    # For now, reuse the TS row count and default to carbon only if unavailable.
    if "atomic_nums" in row and row["atomic_nums"] is not None:
        z = np.asarray(row["atomic_nums"], dtype=int)
    else:
        raise ValueError(
            "Validation parquet does not contain atomic numbers; rerun IRC validation with the current script"
        )

    frames = []
    for label, key in [
        ("reactant_ref", "reactant_coords_flat"),
        ("irc_reverse_endpoint", "reverse_coords_flat"),
        ("ts_input", "ts_coords_flat"),
        ("irc_forward_endpoint", "forward_coords_flat"),
        ("product_ref", "product_coords_flat"),
    ]:
        coords = _coords_from_flat(row.get(key))
        if coords is None:
            continue
        atoms = Atoms(numbers=z.tolist(), positions=coords)
        atoms.info["comment"] = label
        frames.append(atoms)

    if not frames:
        raise ValueError("No IRC frames available to visualize")

    output_dir = args.output_dir or os.path.join(
        os.path.dirname(args.results_parquet), "viewer_reexports"
    )
    formula = str(row.get("formula", f"sample_{args.sample_id}"))
    bundle_dir, multi_xyz, sequence_dir = _write_viewer_bundle(
        output_dir, args.run_id, args.sample_id, formula, frames
    )
    print(f"Saved viewer bundle: {bundle_dir}")
    print(f"Protein Viewer XYZ: {multi_xyz}")
    print(f"Nano Protein Viewer frames: {sequence_dir}")


if __name__ == "__main__":
    main()
