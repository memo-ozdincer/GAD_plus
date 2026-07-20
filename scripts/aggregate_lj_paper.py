#!/usr/bin/env python
"""Aggregate partitioned LJ paper-sweep parquet files."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    files = sorted(args.root.rglob("summary_*.parquet"))
    if not files:
        raise SystemExit(f"No summary_*.parquet files under {args.root}")

    df = pd.concat([pd.read_parquet(path) for path in files], ignore_index=True)
    if "is_nneg1" not in df.columns and "final_n_neg" in df.columns:
        df["is_nneg1"] = df["final_n_neg"] == 1
    if "total_steps" not in df.columns:
        df["total_steps"] = pd.NA
    if "wall_time_s" not in df.columns:
        df["wall_time_s"] = pd.NA
    out = args.output or (args.root / "lj_paper_aggregate.parquet")
    df.to_parquet(out)

    keys = ["start_from", "method", "noise"]
    summary = (
        df.groupby(keys, dropna=False)
        .agg(
            n=("sample_id", "count"),
            conv001=("conv_nneg1_fmax001", "sum"),
            conv005=("conv_nneg1_fmax005", "sum"),
            nneg1=("is_nneg1", "sum"),
            median_steps=("total_steps", "median"),
            mean_wall_s=("wall_time_s", "mean"),
        )
        .reset_index()
    )
    summary["conv001_pct"] = 100.0 * summary["conv001"] / summary["n"]
    summary["conv005_pct"] = 100.0 * summary["conv005"] / summary["n"]
    csv = out.with_suffix(".csv")
    summary.to_csv(csv, index=False)

    print(f"Wrote {out} ({len(df)} rows from {len(files)} files)")
    print(f"Wrote {csv}")
    print(summary.sort_values(["noise", "conv001_pct"], ascending=[True, False]).to_string(index=False))


if __name__ == "__main__":
    main()
