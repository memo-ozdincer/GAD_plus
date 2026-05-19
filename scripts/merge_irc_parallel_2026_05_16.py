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
    {
        "label": "Sella internal d=1 @ 150pm",
        "method_tag": "Sella internal tuned Hess.Freq.=1",
        "dir": f"{RUNS}/irc_sella_internal_2026_05_16/150pm",
        "noise_pm": 150,
        # Final raw conv from pooled_summary_150pm.parquet (job 61166201): 26.83%.
        "raw_conv_pct": 26.8,
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
    n_converged = int(df["source_gad_converged"].sum()) if "source_gad_converged" in df.columns else 0
    n_topo_intended = int(df["topology_intended"].sum()) if "topology_intended" in df.columns else 0
    n_intended = int(df["intended"].sum()) if "intended" in df.columns else 0
    # Two views:
    # (a) per-converged TOPO rate (n_topo / n_converged) — projectable + intrinsic
    # (b) full-N TOPO % (n_topo / 287) — project standard, only meaningful when n_all == 287
    per_conv_topo = 100 * n_topo_intended / n_converged if n_converged > 0 else 0.0
    full_n_topo = 100 * n_topo_intended / 287 if n_all == 287 else None
    full_n_intended = 100 * n_intended / 287 if n_all == 287 else None
    print(f"  {cell['label']}: merged {len(parts)} partitions, {n_all} samples")
    print(f"    source-converged: {n_converged}")
    print(f"    IRC TOPO-intended:  {n_topo_intended}/{n_converged} converged = {per_conv_topo:.1f}% per-conv")
    print(f"    IRC RMSD-intended:  {n_intended}/{n_converged} converged = "
          f"{100*n_intended/n_converged if n_converged > 0 else 0:.1f}% per-conv")
    if full_n_topo is not None:
        recovery_pp = full_n_topo - cell["raw_conv_pct"]
        print(f"    FULL n=287 IRC TOPO: {n_topo_intended}/287 = {full_n_topo:.1f}%   "
              f"(recovery vs raw {cell['raw_conv_pct']:.1f}%: {recovery_pp:+.1f} pp)")
    else:
        print(f"    PARTIAL: {n_all}/287 — wait for all 4 partitions for the full-N TOPO %")
    print(f"    Wrote: {out}")
    return {
        "label": cell["label"], "method_tag": cell["method_tag"],
        "noise_pm": cell["noise_pm"], "n_partitions": len(parts), "n_samples": n_all,
        "n_converged": n_converged, "n_topo": n_topo_intended, "n_intended": n_intended,
        "per_conv_topo_pct": per_conv_topo,
        "full_n_topo_pct": full_n_topo, "full_n_intended_pct": full_n_intended,
        "raw_conv_pct": cell["raw_conv_pct"],
        "recovery_pp": (full_n_topo - cell["raw_conv_pct"]) if full_n_topo is not None else None,
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
