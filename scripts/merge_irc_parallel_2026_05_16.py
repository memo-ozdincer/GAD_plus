#!/usr/bin/env python
"""Merge the 4 sample-range partitions per IRC cell into a single result parquet,
then compute IRC TOPO %, RMSD-intended %, and recovery_pp for each cell.

Run after job 61165468 finishes. Idempotent — also runs partial when only some
partitions have landed.

Outputs:
  - /lustre07/scratch/memoozd/gadplus/runs/irc_*/merged_irc.parquet
  - prints IRC TOPO summary table for the PDF
"""
from __future__ import annotations

import glob
import os

import duckdb
import pandas as pd

RUNS = "/lustre07/scratch/memoozd/gadplus/runs"
con = duckdb.connect()

CELLS = [
    {
        "label": "Sella d=3 @ 200pm",
        "method_tag": "Sella cartesian Eckart untuned Hess.Freq.=3",
        "dir": f"{RUNS}/irc_sella_libdef_d3_2026_05_16",
        "noise_pm": 200,
        "raw_conv_pct": 23.3,   # from headline Table 1
    },
    {
        "label": "Sella internal d=1 @ 200pm",
        "method_tag": "Sella internal tuned Hess.Freq.=1",
        "dir": f"{RUNS}/irc_sella_internal_2026_05_16/200pm",
        "noise_pm": 200,
        "raw_conv_pct": 13.9,
    },
    {
        "label": "Sella libdef midpoint @ 0pm",
        "method_tag": "Sella libdef (midpoint @ 0pm)",
        "dir": f"{RUNS}/irc_sella_midpoint_2026_05_16",
        "noise_pm": 0,
        "raw_conv_pct": 46.7,
    },
]


def merge_cell(cell):
    parts = sorted(glob.glob(f"{cell['dir']}/p*/irc_validation_*.parquet"))
    if not parts:
        print(f"  {cell['label']}: NO partitions yet")
        return None
    quoted = ", ".join(f"'{p}'" for p in parts)
    df = con.execute(f"""
        WITH src AS (
            SELECT * FROM read_parquet([{quoted}], union_by_name=true)
        )
        SELECT * FROM src
        QUALIFY ROW_NUMBER() OVER (PARTITION BY sample_id ORDER BY wall_time_s) = 1
        ORDER BY sample_id
    """).df()
    out = f"{cell['dir']}/merged_irc.parquet"
    df.to_parquet(out)
    n_all = len(df)
    converged = df["source_gad_converged"].sum() if "source_gad_converged" in df.columns else None
    n_topo_intended = df["topology_intended"].sum() if "topology_intended" in df.columns else 0
    n_intended = df["intended"].sum() if "intended" in df.columns else 0
    # IRC TOPO % is computed over the converged subset (n=287 if everything converged)
    # but the standard project metric is over the ENTIRE sample set, where unconverged ones
    # are automatically not-intended. So denominator = 287.
    denom = 287
    topo_pct = 100 * n_topo_intended / denom
    intended_pct = 100 * n_intended / denom
    recovery_pp = topo_pct - cell["raw_conv_pct"]
    print(f"  {cell['label']}: merged {len(parts)} partitions, {n_all} samples")
    print(f"    converged (from survey): {converged}")
    print(f"    IRC TOPO-intended:   {n_topo_intended:>3.0f}/{denom} = {topo_pct:5.1f}%")
    print(f"    IRC RMSD-intended:   {n_intended:>3.0f}/{denom} = {intended_pct:5.1f}%")
    print(f"    Recovery (TOPO−raw): {recovery_pp:+.1f} pp")
    print(f"    Wrote: {out}")
    return {
        "label": cell["label"], "method_tag": cell["method_tag"],
        "noise_pm": cell["noise_pm"], "n_partitions": len(parts), "n_samples": n_all,
        "topo_pct": topo_pct, "intended_pct": intended_pct,
        "raw_conv_pct": cell["raw_conv_pct"], "recovery_pp": recovery_pp,
    }


def main():
    rows = []
    for cell in CELLS:
        print(f"\n=== {cell['label']} ===")
        r = merge_cell(cell)
        if r is not None:
            rows.append(r)
    if rows:
        out = pd.DataFrame(rows)
        out_path = "/lustre06/project/6033559/memoozd/GAD_plus/analysis_2026_04_29/irc_followup_2026_05_16.csv"
        out.to_csv(out_path, index=False)
        print(f"\nWrote summary table: {out_path}")
        print(out.to_string(index=False))


if __name__ == "__main__":
    main()
