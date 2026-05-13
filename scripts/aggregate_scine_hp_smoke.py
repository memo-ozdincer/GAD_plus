"""Aggregate the SCINE GAD hparam smoke sweep.

Reads each config's summary parquet, prints a table sorted by strict-conv%
descending so the winning config is obvious. Also prints fmax distribution
to see whether non-converged samples are at the plateau or stuck far away.

Usage:
    python scripts/aggregate_scine_hp_smoke.py <smoke_run_dir>
e.g.
    python scripts/aggregate_scine_hp_smoke.py \\
        /lustre07/scratch/memoozd/gadplus/runs/smoke_scine_hp_60857961
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np
import pyarrow.parquet as pq


def summarize(parquet_path: str):
    df = pq.read_table(parquet_path).to_pandas()
    n = len(df)
    n_conv = int(df["converged"].sum())
    n_nneg1 = int((df["final_n_neg"] == 1).sum())
    fmax = df["final_force_max"].to_numpy()
    return {
        "n": n,
        "n_conv": n_conv,
        "n_nneg1": n_nneg1,
        "conv_pct": 100 * n_conv / max(n, 1),
        "nneg1_pct": 100 * n_nneg1 / max(n, 1),
        "fmax_median": float(np.median(fmax)),
        "fmax_p25": float(np.percentile(fmax, 25)),
        "fmax_p75": float(np.percentile(fmax, 75)),
        "wall_median": float(np.median(df["wall_time_s"].to_numpy())),
    }


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    root = sys.argv[1]

    rows = []
    for cfg_dir in sorted(os.listdir(root)):
        cfg_path = os.path.join(root, cfg_dir)
        if not os.path.isdir(cfg_path):
            continue
        parquets = glob.glob(os.path.join(cfg_path, "summary_*.parquet"))
        if not parquets:
            print(f"  {cfg_dir}: no summary yet")
            continue
        s = summarize(parquets[0])
        rows.append((cfg_dir, s))

    rows.sort(key=lambda r: -r[1]["conv_pct"])

    print(f"\n{'config':<28} {'N':>3} {'conv%':>6} {'nneg1%':>7} "
          f"{'fmax_p25':>9} {'fmax_med':>9} {'fmax_p75':>9} {'wall_med_s':>11}")
    print("-" * 90)
    for cfg, s in rows:
        print(f"{cfg:<28} {s['n']:>3} {s['conv_pct']:>5.1f}% "
              f"{s['nneg1_pct']:>6.1f}% {s['fmax_p25']:>9.4f} "
              f"{s['fmax_median']:>9.4f} {s['fmax_p75']:>9.4f} "
              f"{s['wall_median']:>10.1f}")


if __name__ == "__main__":
    main()
