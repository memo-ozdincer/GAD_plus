"""Aggregate SCINE main-run summaries into a CSV that mirrors
analysis_2026_04_29/noise_sweep_with_irc.csv (the HIP headline table).

Output columns: family, noise_pm, conv_pct, topo_pct
  - conv_pct: strict GAD criterion (n_neg==1 ∧ fmax<0.01) — what we
    actually optimize for and what the HIP headline reports as "conv_pct".
  - topo_pct: looser, "right topology, force could be tighter"
    (n_neg==1 ∧ fmax<0.05). The IRC TOPO metric isn't applicable here
    (no IRC validation run yet), so we report the topology-pass at a
    looser fmax as a proxy. Documented explicitly in the README.
"""
from __future__ import annotations

import csv
import glob
import os
import sys

import pyarrow.parquet as pq


# Full-grid runs (6 noise levels each).
# Updated 2026-05-12: 15k-step budget at dt=0.007 matched to HIP-level
# performance. The 2k-step rows are kept in a second dict below for
# the "step budget matters" table.
RUNS_FULL = {
    "SCINE/DFTB0 + plain GAD dt=0.007 15k":
        "/lustre07/scratch/memoozd/gadplus/runs/main_scine_gad15k_60865063",
    "SCINE/DFTB0 + Sella libdef 15k":
        "/lustre07/scratch/memoozd/gadplus/runs/main_scine_sella15k_60868140",
    "SCINE/DFTB0 + plain GAD dt=0.005 2k (legacy)":
        "/lustre07/scratch/memoozd/gadplus/runs/main_scine_gad_60772085",
    "SCINE/DFTB0 + plain GAD dt=0.005 no-Eckart 2k (legacy)":
        "/lustre07/scratch/memoozd/gadplus/runs/main_scine_gad_neck_60774250",
    "SCINE/DFTB0 + Sella libdef 2k (legacy)":
        "/lustre07/scratch/memoozd/gadplus/runs/main_scine_sella_60772086",
}

# xTB-favorable panels: single noise level (10 pm), top-30 indices.
RUNS_XTB_PANEL = {
    "xTB/GFN1 + plain GAD (top-30 favorable)":
        "/lustre07/scratch/memoozd/gadplus/runs/main_xtb_favorable_60774467/gad",
    "xTB/GFN1 + Sella libdef (top-30 favorable)":
        "/lustre07/scratch/memoozd/gadplus/runs/main_xtb_favorable_60774467/sella",
}

# IRC validation directories: per-noise irc_validation_<noise>pm_<tag>.parquet
IRC_RUNS = {
    "SCINE/DFTB0 + plain GAD dt=0.007 15k":
        ("/lustre07/scratch/memoozd/gadplus/runs/scine_irc15k_60865129/gad", "gad"),
    "SCINE/DFTB0 + Sella libdef 15k":
        ("/lustre07/scratch/memoozd/gadplus/runs/scine_sella_irc15k_60869134/sella", "sella"),
    "SCINE/DFTB0 + plain GAD dt=0.005 2k (legacy)":
        ("/lustre07/scratch/memoozd/gadplus/runs/scine_irc_60776605/gad", "gad"),
    "SCINE/DFTB0 + Sella libdef 2k (legacy)":
        ("/lustre07/scratch/memoozd/gadplus/runs/scine_sella_irc_60777076/sella", "sella"),
}

NOISES_PM = [10, 30, 50, 100, 150, 200]
OUT_CSV = "analysis_2026_04_29/noise_sweep_scine_xtb.csv"
TOPO_FMAX = 0.05


def cell(parquet_path: str) -> tuple[int, int, int, int]:
    """Return (n_total, n_strict_conv, n_n_neg_1, n_topo_loose)."""
    df = pq.read_table(parquet_path).to_pandas()
    n = len(df)
    n_strict = int(df["converged"].sum())
    n_neg1 = int((df["final_n_neg"] == 1).sum())
    n_topo_loose = int(((df["final_n_neg"] == 1) & (df["final_force_max"] < TOPO_FMAX)).sum())
    return n, n_strict, n_neg1, n_topo_loose


def irc_topo(family: str, pm: int):
    """Look up TOPO-intended count + N validated for (family, noise)."""
    if family not in IRC_RUNS:
        return None, None
    base, tag = IRC_RUNS[family]
    path = os.path.join(base, f"irc_validation_{pm}pm_{tag}.parquet")
    if not os.path.exists(path):
        return None, None
    df = pq.read_table(path).to_pandas()
    n = len(df)
    if n == 0:
        return 0, 0
    n_topo = int(df["topo_intended"].sum())
    return n_topo, n


def main():
    rows = []
    print(f"{'family':<48} {'noise':>5} {'N':>4} {'conv%':>6} {'n_neg=1%':>9} {'topo<0.05%':>11} {'IRC TOPO%':>10}")
    for family, base in RUNS_FULL.items():
        for pm in NOISES_PM:
            p = glob.glob(os.path.join(base, f"noise{pm}pm/summary_*.parquet"))
            if not p:
                print(f"  WARN: missing {family} @ {pm}pm")
                continue
            n, n_strict, n_n1, n_topo = cell(p[0])
            conv_pct = 100.0 * n_strict / n
            n_neg1_pct = 100.0 * n_n1 / n
            topo_pct = 100.0 * n_topo / n
            # IRC TOPO over ALL samples (not just converged), so it's
            # comparable to conv_pct as an end-to-end success metric.
            irc_topo_n, irc_n_val = irc_topo(family, pm)
            if irc_n_val is not None:
                irc_topo_pct = 100.0 * irc_topo_n / max(n, 1)
                irc_str = f"{irc_topo_pct:>9.1f}%"
            else:
                irc_topo_pct = None
                irc_str = "        - "
            rows.append({
                "family": family,
                "noise_pm": pm,
                "n_total": n,
                "conv_pct": conv_pct,
                "n_neg1_pct": n_neg1_pct,
                "topo_pct": topo_pct,
                "irc_topo_pct_over_all": (
                    irc_topo_pct if irc_topo_pct is not None else float("nan")
                ),
                "irc_n_validated": irc_n_val if irc_n_val is not None else 0,
                "irc_n_topo_intended": irc_topo_n if irc_topo_n is not None else 0,
            })
            print(f"{family:<48} {pm:>5} {n:>4} {conv_pct:>5.1f}% {n_neg1_pct:>8.1f}% {topo_pct:>10.1f}% {irc_str}")

    for family, base in RUNS_XTB_PANEL.items():
        p = glob.glob(os.path.join(base, "summary_*.parquet"))
        if not p:
            print(f"  WARN: missing {family} (panel)")
            continue
        n, n_strict, n_n1, n_topo = cell(p[0])
        conv_pct = 100.0 * n_strict / n
        n_neg1_pct = 100.0 * n_n1 / n
        topo_pct = 100.0 * n_topo / n
        rows.append({
            "family": family,
            "noise_pm": 10,  # panel is 10pm only
            "n_total": n,
            "conv_pct": conv_pct,
            "n_neg1_pct": n_neg1_pct,
            "topo_pct": topo_pct,
        })
        print(f"{family:<48} {'10':>5} {n:>4} {conv_pct:>5.1f}% {n_neg1_pct:>8.1f}% {topo_pct:>10.1f}%")

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w") as f:
        w = csv.DictWriter(f, fieldnames=[
            "family", "noise_pm", "n_total",
            "conv_pct", "n_neg1_pct", "topo_pct",
            "irc_topo_pct_over_all", "irc_n_validated", "irc_n_topo_intended",
        ])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nWrote {OUT_CSV}")


if __name__ == "__main__":
    sys.path.insert(0, "src")
    main()
