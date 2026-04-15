#!/usr/bin/env python
"""Batch-export viewer bundles for flagship GAD trajectories.

Focuses on the best-performing small-step GAD runs and writes VS Code
viewer-ready bundles for representative cases across noise levels.
"""

from __future__ import annotations

import argparse
import os
import sys

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from visualize_3d import (  # noqa: E402
    _auto_path,
    _build_atoms_frames,
    _coords_from_flat,
    _downsample_indices,
    _load_traj,
    _write_viewer_bundle,
)


def _load_dataset_cache(h5_path: str, split: str, sample_ids: list[int]) -> dict[int, tuple[np.ndarray, str]]:
    from gadplus.data.transition1x import Transition1xDataset, UsePos

    max_sample_id = max(sample_ids)
    ds = Transition1xDataset(
        h5_path,
        split=split,
        max_samples=max_sample_id + 1,
        transform=UsePos("pos_transition"),
    )
    out: dict[int, tuple[np.ndarray, str]] = {}
    for sample_id in sorted(set(sample_ids)):
        sample = ds[sample_id]
        z = sample.z.detach().cpu().numpy().astype(int)
        formula = str(getattr(sample, "formula", f"sample_{sample_id}"))
        out[sample_id] = (z, formula)
    return out


def _pick_rows(
    summary_glob: str,
    method: str,
    noise_levels: list[int],
    picks: list[str],
) -> pd.DataFrame:
    out = []
    summary_cols = set(
        duckdb.execute(f"DESCRIBE SELECT * FROM '{summary_glob}'").df()["column_name"].tolist()
    )
    method_col = "search_method" if "search_method" in summary_cols else "method"
    for noise_pm in noise_levels:
        df = duckdb.execute(
            f"""
            SELECT *
            FROM '{summary_glob}'
            WHERE {method_col} = '{method}' AND noise_pm = {noise_pm}
            """
        ).df()
        if len(df) == 0:
            continue

        converged = df[df["converged"] == True].copy()  # noqa: E712
        failed = df[df["converged"] == False].copy()  # noqa: E712

        for pick in picks:
            row = None
            if pick == "fast" and len(converged) > 0:
                idx = converged["converged_step"].idxmin()
                row = converged.loc[idx]
            elif pick == "slow" and len(converged) > 0:
                idx = converged["converged_step"].idxmax()
                row = converged.loc[idx]
            elif pick == "failure" and len(failed) > 0:
                if "total_steps" in failed.columns:
                    idx = failed["total_steps"].idxmax()
                    row = failed.loc[idx]
                else:
                    row = failed.iloc[0]

            if row is not None:
                row = row.copy()
                row["viz_pick"] = pick
                out.append(row)

    if not out:
        return pd.DataFrame()
    return pd.DataFrame(out).drop_duplicates(subset=["method", "noise_pm", "sample_id", "viz_pick"])


def _attach_run_ids(selection_df: pd.DataFrame, traj_glob: str) -> pd.DataFrame:
    traj_map = duckdb.execute(
        f"""
        SELECT
            search_method AS method,
            regexp_extract(start_method, '([0-9]+)pm', 1) AS noise_pm_str,
            sample_id,
            min(run_id) AS run_id
        FROM '{traj_glob}'
        GROUP BY 1, 2, 3
        """
    ).df()
    traj_map["noise_pm"] = pd.to_numeric(traj_map["noise_pm_str"], errors="coerce").astype("Int64")
    traj_map = traj_map.drop(columns=["noise_pm_str"])

    merged = selection_df.copy()
    if "method" not in merged.columns and "search_method" in merged.columns:
        merged["method"] = merged["search_method"]
    merged = merged.merge(
        traj_map,
        on=["method", "noise_pm", "sample_id"],
        how="left",
    )
    if merged["run_id"].isna().any():
        missing = merged[merged["run_id"].isna()][["method", "noise_pm", "sample_id"]]
        raise ValueError(f"Could not resolve run_id for rows:\n{missing.to_string(index=False)}")
    return merged


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--traj-dir", type=str, required=True)
    parser.add_argument("--method", type=str, default="gad_small_dt")
    parser.add_argument("--noise-levels", type=int, nargs="+", default=[10, 50, 100, 150, 200])
    parser.add_argument("--picks", type=str, nargs="+", default=["fast", "slow", "failure"])
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--h5-path", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--max-frames", type=int, default=300)
    parser.add_argument("--stride", type=int, default=None)
    args = parser.parse_args()

    summary_glob = os.path.join(args.traj_dir, "summary_*.parquet")
    traj_glob = os.path.join(args.traj_dir, "traj_*.parquet")
    output_dir = args.output_dir or os.path.join(args.traj_dir, "flagship_visualizations")
    os.makedirs(output_dir, exist_ok=True)

    selection_df = _pick_rows(summary_glob, args.method, args.noise_levels, args.picks)
    if len(selection_df) == 0:
        raise ValueError("No visualization targets matched the requested method/noise/picks")
    selection_df = _attach_run_ids(selection_df, traj_glob)

    sample_ids = selection_df["sample_id"].astype(int).tolist()
    h5_path = args.h5_path or _auto_path(
        [
            "/lustre06/project/6033559/memoozd/data/transition1x.h5",
            "/project/rrg-aspuru/memoozd/data/transition1x.h5",
        ],
        "Transition1x HDF5",
    )
    dataset_cache = _load_dataset_cache(h5_path, args.split, sample_ids)

    manifest_rows = []
    for row in selection_df.itertuples(index=False):
        run_id = str(row.run_id)
        sample_id = int(row.sample_id)
        pick = str(row.viz_pick)
        noise_pm = int(row.noise_pm)

        z, formula = dataset_cache[sample_id]
        traj_df = _load_traj(traj_glob, run_id, sample_id)
        frame_idx = _downsample_indices(len(traj_df), args.max_frames, args.stride)
        traj_df = traj_df.iloc[frame_idx].reset_index(drop=True)
        coords_series = [_coords_from_flat(v, len(z)) for v in traj_df["coords_flat"].tolist()]
        atoms_frames = _build_atoms_frames(coords_series, z, traj_df)

        case_dir = os.path.join(output_dir, f"{args.method}_{noise_pm}pm_{pick}")
        bundle_dir, multi_xyz, sequence_dir = _write_viewer_bundle(
            case_dir,
            run_id=run_id,
            sample_id=sample_id,
            formula=formula,
            atoms_frames=atoms_frames,
        )

        manifest_rows.append(
            {
                "method": args.method,
                "noise_pm": noise_pm,
                "pick": pick,
                "run_id": run_id,
                "sample_id": sample_id,
                "formula": formula,
                "converged": bool(getattr(row, "converged")),
                "converged_step": getattr(row, "converged_step", None),
                "bundle_dir": bundle_dir,
                "multi_xyz": multi_xyz,
                "sequence_dir": sequence_dir,
                "n_frames": len(atoms_frames),
            }
        )

    manifest_path = os.path.join(output_dir, f"{args.method}_viewer_manifest.parquet")
    pd.DataFrame(manifest_rows).to_parquet(manifest_path, index=False)
    print(f"Wrote {len(manifest_rows)} viewer bundles")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
