#!/usr/bin/env python
"""Retroactively backfill fmax-based convergence metrics from existing Parquet outputs.

This script avoids rerunning optimization trajectories. It reads existing
`summary_*.parquet` and `traj_*.parquet`, extracts final coordinates for each
sample, evaluates HIP once at that geometry, and appends:

- final_force_norm_recomputed
- final_force_max_recomputed
- final_n_neg_recomputed (Eckart-projected vibrational Hessian)
- conv_nneg1_force001_recomputed
- conv_nneg1_fmax001_recomputed
- conv_nneg1_fmax003_recomputed

Typical usage on Narval:

  python scripts/backfill_fmax.py \
      --results-dir /lustre07/scratch/memoozd/gadplus/runs/noise_survey_300 \
      --split train

Writes new files by default with suffix `_with_fmax.parquet`.
Use `--in-place` to overwrite original summaries.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from dataclasses import dataclass
from typing import Any

import duckdb
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


@dataclass
class FinalCoordRecord:
    run_id: str
    sample_id: int
    search_method: str
    start_method: str
    final_step: int
    coords_flat: list[float]


def _force_mean_local(forces: torch.Tensor) -> float:
    if forces.dim() == 3 and forces.shape[0] == 1:
        forces = forces[0]
    f = forces.reshape(-1, 3)
    return float(f.norm(dim=1).mean().item())


def _force_max_local(forces: torch.Tensor) -> float:
    if forces.dim() == 3 and forces.shape[0] == 1:
        forces = forces[0]
    return float(forces.reshape(-1).abs().max().item())


def _auto_path(candidates: list[str], label: str) -> str:
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"{label} not found in candidates: {candidates}")


def _load_final_coords(traj_glob: str) -> list[FinalCoordRecord]:
    query = f"""
        SELECT
            run_id,
            sample_id,
            search_method,
            start_method,
            step AS final_step,
            coords_flat
        FROM (
            SELECT
                run_id,
                sample_id,
                search_method,
                start_method,
                step,
                coords_flat,
                ROW_NUMBER() OVER (
                    PARTITION BY run_id, sample_id
                    ORDER BY step DESC
                ) AS rn
            FROM '{traj_glob}'
        ) t
        WHERE rn = 1
    """
    df = duckdb.execute(query).df()
    records: list[FinalCoordRecord] = []
    for row in df.itertuples(index=False):
        records.append(
            FinalCoordRecord(
                run_id=str(row.run_id),
                sample_id=int(row.sample_id),
                search_method=str(row.search_method),
                start_method=str(row.start_method),
                final_step=int(row.final_step),
                coords_flat=list(row.coords_flat),
            )
        )
    return records


def _build_lookups(records: list[FinalCoordRecord]) -> dict[str, Any]:
    by_run_sample: dict[tuple[str, int], FinalCoordRecord] = {}
    by_method_start_sample: dict[tuple[str, str, int], list[FinalCoordRecord]] = {}
    by_method_sample: dict[tuple[str, int], list[FinalCoordRecord]] = {}

    for rec in records:
        by_run_sample[(rec.run_id, rec.sample_id)] = rec
        by_method_start_sample.setdefault(
            (rec.search_method, rec.start_method, rec.sample_id), []
        ).append(rec)
        by_method_sample.setdefault((rec.search_method, rec.sample_id), []).append(rec)

    return {
        "by_run_sample": by_run_sample,
        "by_method_start_sample": by_method_start_sample,
        "by_method_sample": by_method_sample,
    }


def _row_get(row: pd.Series, key: str, default: Any = None) -> Any:
    return row[key] if key in row and pd.notna(row[key]) else default


def _infer_method_and_start(row: pd.Series) -> tuple[str | None, str | None]:
    method = _row_get(row, "search_method", None)
    if method is None:
        method = _row_get(row, "method", None)

    start_method = _row_get(row, "start_method", None)
    if start_method is None and "noise_pm" in row and pd.notna(row["noise_pm"]):
        start_method = f"noised_ts_{int(row['noise_pm'])}pm"

    return method, start_method


def _pick_best_candidate(
    candidates: list[FinalCoordRecord],
    row: pd.Series,
) -> FinalCoordRecord | None:
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]

    # Prefer matching final_step+1 == total_steps when available.
    total_steps = _row_get(row, "total_steps", None)
    if total_steps is not None:
        filtered = [c for c in candidates if c.final_step + 1 == int(total_steps)]
        if len(filtered) == 1:
            return filtered[0]
        if filtered:
            candidates = filtered

    # Deterministic fallback: highest run_id lexicographically.
    return sorted(candidates, key=lambda c: c.run_id)[-1]


def _match_final_coords(
    row: pd.Series,
    lookups: dict[str, Any],
) -> tuple[list[float] | None, str]:
    sample_id = int(row["sample_id"])

    # 1) If present in summary row itself.
    if "final_coords_flat" in row and isinstance(row["final_coords_flat"], list):
        if len(row["final_coords_flat"]) > 0:
            return list(row["final_coords_flat"]), "summary.final_coords_flat"

    # 2) Strongest key: (run_id, sample_id)
    run_id = _row_get(row, "run_id", None)
    if run_id is not None:
        rec = lookups["by_run_sample"].get((str(run_id), sample_id))
        if rec is not None:
            return rec.coords_flat, "traj.run_id+sample_id"

    # 3) Method/start/sample fallback
    method, start_method = _infer_method_and_start(row)
    if method is not None and start_method is not None:
        cands = lookups["by_method_start_sample"].get((method, start_method, sample_id), [])
        rec = _pick_best_candidate(cands, row)
        if rec is not None:
            return rec.coords_flat, "traj.method+start+sample"

    # 4) Method/sample fallback
    if method is not None:
        cands = lookups["by_method_sample"].get((method, sample_id), [])
        rec = _pick_best_candidate(cands, row)
        if rec is not None:
            return rec.coords_flat, "traj.method+sample"

    return None, "unmatched"


def _build_dataset(h5_path: str, split: str, max_sample_id: int):
    from gadplus.data.transition1x import Transition1xDataset, UsePos

    return Transition1xDataset(
        h5_path,
        split=split,
        max_samples=max_sample_id + 1,
        transform=UsePos("pos_transition"),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir",
        type=str,
        required=True,
        help="Directory with summary_*.parquet and traj_*.parquet",
    )
    parser.add_argument("--summary-pattern", type=str, default="summary_*.parquet")
    parser.add_argument("--traj-pattern", type=str, default="traj_*.parquet")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--h5-path", type=str, default=None)
    parser.add_argument("--ckpt-path", type=str, default=None)
    parser.add_argument("--suffix", type=str, default="_with_fmax")
    parser.add_argument("--in-place", action="store_true")
    parser.add_argument(
        "--max-rows", type=int, default=None, help="Optional smoke-test limit per summary file"
    )
    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    summary_glob = os.path.join(args.results_dir, args.summary_pattern)
    traj_glob = os.path.join(args.results_dir, args.traj_pattern)
    summary_files = sorted(glob.glob(summary_glob))
    traj_files = sorted(glob.glob(traj_glob))

    if not summary_files:
        raise FileNotFoundError(f"No summary parquet files found: {summary_glob}")
    if not traj_files:
        raise FileNotFoundError(f"No trajectory parquet files found: {traj_glob}")

    ckpt_path = args.ckpt_path or _auto_path(
        [
            "/lustre06/project/6033559/memoozd/models/hip_v2.ckpt",
            "/project/rrg-aspuru/memoozd/models/hip_v2.ckpt",
        ],
        "HIP checkpoint",
    )
    h5_path = args.h5_path or _auto_path(
        [
            "/lustre06/project/6033559/memoozd/data/transition1x.h5",
            "/project/rrg-aspuru/memoozd/data/transition1x.h5",
        ],
        "Transition1x HDF5",
    )

    print(f"Device: {device}")
    print(f"Results dir: {args.results_dir}")
    print(f"Found {len(summary_files)} summary files and {len(traj_files)} trajectory files")

    from gadplus.calculator.hip import load_hip_calculator, make_hip_predict_fn
    from gadplus.projection import vib_eig, atomic_nums_to_symbols

    calculator = load_hip_calculator(ckpt_path, device=device)
    predict_fn = make_hip_predict_fn(calculator)

    final_records = _load_final_coords(traj_glob)
    lookups = _build_lookups(final_records)
    print(f"Loaded {len(final_records)} final-coordinate records from trajectories")

    # Build one dataset covering all needed sample ids.
    max_sample_id = 0
    for sf in summary_files:
        sdf = pd.read_parquet(sf, columns=["sample_id"])
        if len(sdf) > 0:
            max_sample_id = max(max_sample_id, int(sdf["sample_id"].max()))

    dataset = _build_dataset(h5_path, args.split, max_sample_id)
    print(f"Loaded dataset split={args.split}, size={len(dataset)}")

    z_cache: dict[int, torch.Tensor] = {}
    for sid in range(max_sample_id + 1):
        z_cache[sid] = dataset[sid].z.to(device)

    for summary_path in summary_files:
        df = pd.read_parquet(summary_path)
        if args.max_rows is not None:
            df = df.head(args.max_rows).copy()

        missing_coords = 0
        rows_out: list[dict[str, Any]] = []

        print(f"\nBackfilling: {os.path.basename(summary_path)} ({len(df)} rows)")
        for idx, row in df.iterrows():
            sid = int(row["sample_id"])
            coords_flat, source = _match_final_coords(row, lookups)

            out_row = dict(row)
            out_row["retro_coords_source"] = source

            if coords_flat is None:
                missing_coords += 1
                out_row["final_force_norm_recomputed"] = None
                out_row["final_force_max_recomputed"] = None
                out_row["final_n_neg_recomputed"] = None
                out_row["conv_nneg1_force001_recomputed"] = None
                out_row["conv_nneg1_fmax001_recomputed"] = None
                out_row["conv_nneg1_fmax003_recomputed"] = None
                rows_out.append(out_row)
                continue

            z = z_cache[sid]
            coords = torch.tensor(coords_flat, dtype=torch.float32, device=device).reshape(-1, 3)
            out = predict_fn(coords, z, do_hessian=True, require_grad=False)
            forces = out["forces"]
            hessian = out["hessian"]

            if forces.dim() == 3 and forces.shape[0] == 1:
                forces = forces[0]
            forces = forces.reshape(-1, 3)

            fn = _force_mean_local(forces)
            fm = _force_max_local(forces)

            evals_vib, _, _ = vib_eig(
                hessian,
                coords,
                atomic_nums_to_symbols(z),
                purify=False,
            )
            n_neg = int((evals_vib < 0).sum().item())

            out_row["final_force_norm_recomputed"] = fn
            out_row["final_force_max_recomputed"] = fm
            out_row["final_n_neg_recomputed"] = n_neg
            out_row["conv_nneg1_force001_recomputed"] = bool(n_neg == 1 and fn < 0.01)
            out_row["conv_nneg1_fmax001_recomputed"] = bool(n_neg == 1 and fm < 0.01)
            out_row["conv_nneg1_fmax003_recomputed"] = bool(n_neg == 1 and fm < 0.03)
            rows_out.append(out_row)

            if (idx + 1) % 50 == 0:
                print(f"  processed {idx + 1}/{len(df)}")

        out_df = pd.DataFrame(rows_out)
        if args.in_place:
            out_path = summary_path
        else:
            root, ext = os.path.splitext(summary_path)
            out_path = f"{root}{args.suffix}{ext}"
        out_df.to_parquet(out_path, index=False)

        matched = len(out_df) - missing_coords
        print(f"  wrote: {out_path} | matched={matched}/{len(out_df)} | missing={missing_coords}")

        if matched > 0:
            rate_fmax = 100.0 * out_df["conv_nneg1_fmax001_recomputed"].fillna(False).mean()
            rate_force = 100.0 * out_df["conv_nneg1_force001_recomputed"].fillna(False).mean()
            print(
                f"  conv rates: nneg1+fmax<0.01={rate_fmax:.1f}% | nneg1+force<0.01={rate_force:.1f}%"
            )


if __name__ == "__main__":
    main()
