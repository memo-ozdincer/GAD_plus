#!/usr/bin/env python
"""Merge main + safety-net partitioned summary parquets into single canonical
pooled summaries that IRC validation can consume.

Run after SLURM job 61087603 + 61088001 cells complete.

Output paths (will overwrite if existing):
  test_hessfreq/sella_carteck_libdef_d3/pooled_summary_200pm.parquet
  test_set/sella_internal_default/pooled_summary_200pm.parquet
"""
import glob
import os

import duckdb

RUNS = "/lustre07/scratch/memoozd/gadplus/runs"
con = duckdb.connect()

POOLED_CELLS = [
    {
        "name": "Sella d=3 @ 200pm",
        "main_glob":   f"{RUNS}/test_hessfreq/sella_carteck_libdef_d3/summary*200pm.parquet",
        "safety_glob": f"{RUNS}/safetynet/sella_carteck_libdef_d3/summary*200pm*.parquet",
        "out": f"{RUNS}/test_hessfreq/sella_carteck_libdef_d3/pooled_summary_200pm.parquet",
    },
    {
        "name": "Sella internal @ 200pm",
        "main_glob":   f"{RUNS}/test_set/sella_internal_default/summary*200pm.parquet",
        "safety_glob": f"{RUNS}/safetynet/sella_internal_default/summary*200pm*.parquet",
        "out": f"{RUNS}/test_set/sella_internal_default/pooled_summary_200pm.parquet",
    },
    {
        # 150pm refill (job 61166201) writes 4 partitions to safetynet/ with _s0-72,
        # _s72-144, etc. suffixes. Pool them into canonical 150pm summary.
        "name": "Sella internal @ 150pm",
        "main_glob":   f"{RUNS}/test_set/sella_internal_default/summary*150pm.parquet",
        "safety_glob": f"{RUNS}/safetynet/sella_internal_default/summary*150pm*.parquet",
        "out": f"{RUNS}/test_set/sella_internal_default/pooled_summary_150pm.parquet",
    },
]


def pool(cell):
    parquets = []
    for g in [cell["main_glob"], cell["safety_glob"]]:
        for p in glob.glob(g):
            if "pooled_summary" in p:
                continue
            parquets.append(p)
    if not parquets:
        print(f"  {cell['name']}: NO PARQUETS YET — skip")
        return
    print(f"  {cell['name']}: pooling from {len(parquets)} sources")
    for p in parquets:
        print(f"    - {p}")
    quoted = ", ".join(f"'{p}'" for p in parquets)
    df = con.execute(f"""
        WITH src AS (
            SELECT *
            FROM read_parquet([{quoted}], union_by_name=true)
        )
        SELECT * FROM src
        QUALIFY ROW_NUMBER() OVER (PARTITION BY sample_id ORDER BY total_steps DESC) = 1
        ORDER BY sample_id
    """).df()
    df.to_parquet(cell["out"])
    print(f"    -> wrote {cell['out']}  ({len(df)} unique samples)")


def main():
    for cell in POOLED_CELLS:
        pool(cell)


if __name__ == "__main__":
    main()
