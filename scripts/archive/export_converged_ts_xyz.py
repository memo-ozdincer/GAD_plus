#!/usr/bin/env python
"""Export converged final TS geometries to a multi-frame XYZ from Parquet runs.

This works retroactively on existing run directories containing
`summary_*.parquet` and `traj_*.parquet`.

Example:
  python scripts/export_converged_ts_xyz.py \
      --run-dir /lustre07/scratch/memoozd/gadplus/runs/noise_survey_300 \
      --split train
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import duckdb
import pandas as pd
from ase import Atoms
from ase.io import write as ase_write

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def _auto_path(candidates: list[str], label: str) -> str:
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"{label} not found in candidates: {candidates}")


def _load_summary(summary_glob: str) -> pd.DataFrame:
    df = duckdb.execute(f"SELECT * FROM '{summary_glob}'").df()
    if len(df) == 0:
        raise ValueError(f"No summary rows found in {summary_glob}")
    return df


def _load_final_coords_map(traj_glob: str) -> dict[tuple[str, int], list[float]]:
    q = f"""
    SELECT run_id, sample_id, coords_flat
    FROM (
        SELECT run_id, sample_id, step, coords_flat,
               ROW_NUMBER() OVER (PARTITION BY run_id, sample_id ORDER BY step DESC) AS rn
        FROM '{traj_glob}'
    ) t
    WHERE rn = 1
    """
    df = duckdb.execute(q).df()
    out: dict[tuple[str, int], list[float]] = {}
    for row in df.itertuples(index=False):
        out[(str(row.run_id), int(row.sample_id))] = list(row.coords_flat)
    return out


def _load_atomic_numbers(h5_path: str, split: str, max_sample_id: int):
    from gadplus.data.transition1x import Transition1xDataset, UsePos

    ds = Transition1xDataset(
        h5_path,
        split=split,
        max_samples=max_sample_id + 1,
        transform=UsePos("pos_transition"),
    )
    z_cache = {}
    for sid in range(min(len(ds), max_sample_id + 1)):
        z_cache[sid] = ds[sid].z.detach().cpu().numpy()
    return z_cache


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--summary-pattern", type=str, default="summary_*.parquet")
    parser.add_argument("--traj-pattern", type=str, default="traj_*.parquet")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--h5-path", type=str, default=None)
    parser.add_argument("--output-xyz", type=str, default=None)
    parser.add_argument("--output-index", type=str, default=None)
    parser.add_argument("--only-converged", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    summary_glob = os.path.join(args.run_dir, args.summary_pattern)
    traj_glob = os.path.join(args.run_dir, args.traj_pattern)

    if not glob.glob(summary_glob):
        raise FileNotFoundError(f"No summary files found: {summary_glob}")
    if not glob.glob(traj_glob):
        raise FileNotFoundError(f"No trajectory files found: {traj_glob}")

    summary_df = _load_summary(summary_glob)
    if args.only_converged and "converged" in summary_df.columns:
        summary_df = summary_df[summary_df["converged"] == True].copy()  # noqa: E712
    if len(summary_df) == 0:
        raise ValueError("No rows selected for export")

    final_coords_map = _load_final_coords_map(traj_glob)
    max_sample_id = int(summary_df["sample_id"].max())

    h5_path = args.h5_path or _auto_path(
        [
            "/lustre06/project/6033559/memoozd/data/transition1x.h5",
            "/project/rrg-aspuru/memoozd/data/transition1x.h5",
        ],
        "Transition1x HDF5",
    )
    z_cache = _load_atomic_numbers(h5_path, args.split, max_sample_id)

    atoms_list = []
    index_rows = []
    missing = 0

    for row in summary_df.itertuples(index=False):
        run_id = str(getattr(row, "run_id", ""))
        sample_id = int(getattr(row, "sample_id"))
        key = (run_id, sample_id)

        coords_flat = final_coords_map.get(key)
        if coords_flat is None:
            missing += 1
            continue

        z = z_cache.get(sample_id)
        if z is None:
            missing += 1
            continue

        coords = pd.Series(coords_flat, dtype=float).to_numpy().reshape(-1, 3)
        atoms = Atoms(numbers=z, positions=coords)

        method = getattr(row, "search_method", getattr(row, "method", "unknown"))
        formula = getattr(row, "formula", f"sample_{sample_id}")
        noise_pm = getattr(row, "noise_pm", None)

        atoms.info["run_id"] = run_id
        atoms.info["sample_id"] = int(sample_id)
        atoms.info["formula"] = str(formula)
        atoms.info["method"] = str(method)
        if noise_pm is not None:
            atoms.info["noise_pm"] = int(noise_pm)

        frame_index = len(atoms_list)
        atoms_list.append(atoms)
        index_rows.append(
            {
                "frame_index": frame_index,
                "run_id": run_id,
                "sample_id": sample_id,
                "formula": formula,
                "method": method,
                "noise_pm": noise_pm,
            }
        )

    if len(atoms_list) == 0:
        raise ValueError("No geometries exported; check run_dir and split")

    output_xyz = args.output_xyz or os.path.join(args.run_dir, "converged_ts_all.xyz")
    output_index = args.output_index or os.path.join(args.run_dir, "converged_ts_index.parquet")

    ase_write(output_xyz, atoms_list)
    pd.DataFrame(index_rows).to_parquet(output_index, index=False)

    print(f"Exported {len(atoms_list)} TS frames")
    print(f"Missing rows: {missing}")
    print(f"XYZ: {output_xyz}")
    print(f"Index: {output_index}")


if __name__ == "__main__":
    main()
