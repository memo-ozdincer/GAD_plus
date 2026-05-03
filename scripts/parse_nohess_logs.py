#!/usr/bin/env python
"""Parse Sella no-Hess slurm logs into a partial-conv table.

12 cells timed out at 12h before writing summaries, but per-sample
stdout reports CONV/FAIL plus n_neg, fmax, force_norm, steps, walltime.

Output: analysis_2026_04_29/sella_nohess_partial.csv with columns
method, noise_pm, n_completed, n_conv, conv_pct, conv_pct_lb_287,
median_steps, median_wall_s.

Note: This is partial coverage (each cell processed only ~25 of 287
samples in 12h). The conv rate over completed samples is the best
estimate; we also report a lower-bound assuming all unprocessed
samples failed (rate = n_conv / 287).
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

LOGDIR = Path("/lustre07/scratch/memoozd/gadplus/logs")
OUT = Path("/lustre06/project/6033559/memoozd/GAD_plus/analysis_2026_04_29")
OUT.mkdir(exist_ok=True, parents=True)

LINE_RE = re.compile(
    r"\[\s*(\d+)\]\s+\S+\s+\|\s+sella=(CONV|FAIL)\s+fmax=([\d.eE+-]+)"
    r"\s+\|\s+n_neg=(\d+)\s+force=([\d.eE+-]+)\s+ours=(\S+)\s+\|\s+steps=(\d+)"
    r"\s+\|\s+([\d.]+)s"
)
HEADER_RE = re.compile(r"task=(\d+)\s+tag=(\w+)\s+.*noise=([\d.]+)")


def parse_log(path: Path):
    samples = []
    method = None
    noise_pm = None
    with path.open() as f:
        for line in f:
            if method is None:
                m = HEADER_RE.search(line)
                if m:
                    method = m.group(2)
                    noise_pm = int(round(float(m.group(3)) * 1000))
                    continue
            m = LINE_RE.search(line)
            if m:
                samples.append({
                    "sample_id": int(m.group(1)),
                    "conv": m.group(2) == "CONV",
                    "fmax": float(m.group(3)),
                    "n_neg": int(m.group(4)),
                    "force_norm": float(m.group(5)),
                    "ours": m.group(6),
                    "steps": int(m.group(7)),
                    "wall_s": float(m.group(8)),
                })
    return method, noise_pm, samples


def main():
    rows = []
    for f in sorted(LOGDIR.glob("testsellanohess_*.out")):
        method, noise_pm, samples = parse_log(f)
        if method is None or not samples:
            continue
        df = pd.DataFrame(samples)
        n_done = len(df)
        n_conv = int(df["conv"].sum())
        n_ours_ts = int((df["ours"] == "TS").sum())
        rows.append({
            "method": method,
            "noise_pm": noise_pm,
            "n_completed": n_done,
            "n_conv": n_conv,
            "conv_pct_partial": 100 * n_conv / n_done,
            "conv_pct_lb": 100 * n_conv / 287,
            "n_ours_TS": n_ours_ts,
            "ours_TS_pct_partial": 100 * n_ours_ts / n_done,
            "median_steps": float(df["steps"].median()),
            "median_wall_s": float(df["wall_s"].median()),
            "median_fmax": float(df["fmax"].median()),
        })

    out = pd.DataFrame(rows).sort_values(["method", "noise_pm"]).reset_index(drop=True)
    out.to_csv(OUT / "sella_nohess_partial.csv", index=False)
    print(f"wrote {OUT/'sella_nohess_partial.csv'}")
    print()
    print(out.to_string(index=False))

    print()
    print("=== Sella conv (Sella's own gate) over completed samples ===")
    print(f"{'method':<22} {'10':>6} {'30':>6} {'50':>6} {'100':>6} {'150':>6} {'200':>6}")
    for m in ["carteck_nohess", "internal_nohess"]:
        row = [f"{m:<22}"]
        for n in [10, 30, 50, 100, 150, 200]:
            sub = out[(out["method"] == m) & (out["noise_pm"] == n)]
            if len(sub):
                pct = sub.iloc[0]["conv_pct_partial"]
                done = sub.iloc[0]["n_completed"]
                row.append(f"{pct:5.1f}/{done:>2}")
            else:
                row.append("   -- ")
        print(" ".join(row))

    print()
    print("=== n_neg=1 ∧ force<0.01 (our gate) over completed samples ===")
    print(f"{'method':<22} {'10':>6} {'30':>6} {'50':>6} {'100':>6} {'150':>6} {'200':>6}")
    for m in ["carteck_nohess", "internal_nohess"]:
        row = [f"{m:<22}"]
        for n in [10, 30, 50, 100, 150, 200]:
            sub = out[(out["method"] == m) & (out["noise_pm"] == n)]
            if len(sub):
                pct = sub.iloc[0]["ours_TS_pct_partial"]
                done = sub.iloc[0]["n_completed"]
                row.append(f"{pct:5.1f}/{done:>2}")
            else:
                row.append("   -- ")
        print(" ".join(row))


if __name__ == "__main__":
    main()
