"""Cross-calculator sample-level TOPO comparison: HIP vs SCINE/DFTB0.

Per sample at each noise level, classify HIP-TOPO and SCINE-TOPO outcomes,
then build the four-cell confusion table for each (method × noise).

Output: analysis_2026_04_29/cross_calc_topo_compare.csv

Question answered: among samples where SCINE finds the right saddle (per
IRC TOPO), does HIP also find it? Conversely, samples where HIP succeeds —
how many of those translate to SCINE? Sets the bound for "expected
agreement" given a different PES.
"""
from __future__ import annotations

import csv
import glob
import os

import pyarrow.parquet as pq


NOISES = [10, 30, 50, 100, 150, 200]


HIP_GAD_DIR = "/lustre07/scratch/memoozd/gadplus/runs/test_irc/gad_dt005_fmax"
HIP_SELLA_DIR = "/lustre07/scratch/memoozd/gadplus/runs/test_irc/sella_carteck_libdef"
SCINE_GAD_DIR = "/lustre07/scratch/memoozd/gadplus/runs/scine_irc15k_60865129/gad"
SCINE_SELLA_DIR = "/lustre07/scratch/memoozd/gadplus/runs/scine_sella_irc15k_60869134/sella"


def load_topo(path: str, topo_col: str) -> set[int]:
    """Return the set of sample_ids that achieved TOPO-intended."""
    if not os.path.exists(path):
        return set()
    df = pq.read_table(path).to_pandas()
    if len(df) == 0 or topo_col not in df.columns:
        return set()
    return set(int(s) for s in df[df[topo_col]]["sample_id"].tolist())


def confusion(set_hip: set[int], set_scine: set[int]) -> dict:
    """Two-way confusion for binary TOPO outcomes."""
    both = set_hip & set_scine
    hip_only = set_hip - set_scine
    scine_only = set_scine - set_hip
    return {
        "n_hip_topo": len(set_hip),
        "n_scine_topo": len(set_scine),
        "n_both": len(both),
        "n_hip_only": len(hip_only),
        "n_scine_only": len(scine_only),
    }


def main():
    rows = []
    print(f"{'method':<8} {'noise':>5} | {'HIP':>5} {'SCINE':>5} {'both':>5} "
          f"{'HIPonly':>7} {'SCINEonly':>9} {'P(SCINE|HIP)':>13} {'P(HIP|SCINE)':>13}")
    for method, hip_dir, hip_fn_pattern, scine_dir, scine_fn_pattern in [
        ("GAD",   HIP_GAD_DIR,   "irc_validation_sella_hip_allendpoints_{pm}pm.parquet",
                  SCINE_GAD_DIR, "irc_validation_{pm}pm_gad.parquet"),
        ("Sella", HIP_SELLA_DIR, "irc_validation_sella_hip_allendpoints_{pm}pm.parquet",
                  SCINE_SELLA_DIR, "irc_validation_{pm}pm_sella.parquet"),
    ]:
        for pm in NOISES:
            hip_set = load_topo(os.path.join(hip_dir, hip_fn_pattern.format(pm=pm)),
                                topo_col="topology_intended")
            scine_set = load_topo(os.path.join(scine_dir, scine_fn_pattern.format(pm=pm)),
                                  topo_col="topo_intended")
            c = confusion(hip_set, scine_set)
            p_scine_given_hip = c["n_both"] / c["n_hip_topo"] if c["n_hip_topo"] else float("nan")
            p_hip_given_scine = c["n_both"] / c["n_scine_topo"] if c["n_scine_topo"] else float("nan")
            print(f"{method:<8} {pm:>5} | "
                  f"{c['n_hip_topo']:>5} {c['n_scine_topo']:>5} {c['n_both']:>5} "
                  f"{c['n_hip_only']:>7} {c['n_scine_only']:>9} "
                  f"{p_scine_given_hip:>12.2f} {p_hip_given_scine:>12.2f}")
            rows.append({"method": method, "noise_pm": pm, **c,
                         "P_scine_topo_given_hip_topo": p_scine_given_hip,
                         "P_hip_topo_given_scine_topo": p_hip_given_scine})

    out = "analysis_2026_04_29/cross_calc_topo_compare.csv"
    with open(out, "w") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
