#!/usr/bin/env python
"""IRC validation sensitivity: re-score existing IRC parquets at multiple
RMSD thresholds and bond-cutoff scales.

The original IRC run used rmsd_threshold=0.3 Å and cutoff_scale=1.2 ×
covalent_radius. This script re-scores using the existing endpoint
coordinates without re-running IRC (which is expensive).

Re-scores:
  - RMSD-intended at thresholds {0.3, 0.4, 0.5, 0.7} Å
  - TOPO-intended at cutoff_scales {1.1, 1.2, 1.3, 1.4}

Output: analysis_2026_04_29/irc_sensitivity.csv
"""
from __future__ import annotations

import os
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import torch

RUNS = Path("/lustre07/scratch/memoozd/gadplus/runs/test_irc")
OUT = Path("/lustre06/project/6033559/memoozd/GAD_plus/analysis_2026_04_29")
OUT.mkdir(exist_ok=True, parents=True)

NOISES = [10, 30, 50, 100, 150, 200]

METHODS = {
    "GAD dt=0.003 (5k)":  "gad_dt003_fmax",
    "GAD dt=0.005 (5k)":  "gad_dt005_fmax",
    "GAD dt=0.007 (5k)":  "gad_dt007_fmax",
    "Sella cart+Eckart, delta0=0.10 gamma=0.40 H/step": "sella_carteck_libdef",
    "Sella cart+Eckart, delta0=0.048 gamma=0 H/step": "sella_carteck_default",
    "Sella internal, delta0=0.048 gamma=0 H/step": "sella_internal_default",
}

# Import scoring infrastructure from gadplus
import sys
sys.path.insert(0, "/lustre06/project/6033559/memoozd/GAD_plus/src")
from gadplus.search.irc_validate import coords_to_bond_graph, bond_graphs_match
from gadplus.geometry.alignment import aligned_rmsd_by_element


def reload_data(parquet_path: Path) -> pd.DataFrame:
    """Pull endpoint coords + reference R/P coords + atomic_nums from parquet."""
    df = duckdb.execute(f"""
      SELECT sample_id, atomic_nums, ts_coords_flat, reactant_coords_flat,
             product_coords_flat, forward_coords_flat, reverse_coords_flat
      FROM '{parquet_path}' WHERE forward_coords_flat IS NOT NULL
    """).df()
    return df


def score_one(row, rmsd_thresh: float, cutoff_scale: float) -> dict:
    nums = np.asarray(row["atomic_nums"], dtype=np.int64)
    n = len(nums)
    fwd = np.asarray(row["forward_coords_flat"], dtype=np.float64).reshape(n, 3)
    rev = np.asarray(row["reverse_coords_flat"], dtype=np.float64).reshape(n, 3)
    rea = np.asarray(row["reactant_coords_flat"], dtype=np.float64).reshape(n, 3)
    pro_flat = row["product_coords_flat"]
    if pro_flat is None or np.all(np.asarray(pro_flat) == 0):
        pro = None
    else:
        pro = np.asarray(pro_flat, dtype=np.float64).reshape(n, 3)
    nums_t = torch.tensor(nums)

    # RMSD scoring
    fr = aligned_rmsd_by_element(fwd, rea, nums)
    rr = aligned_rmsd_by_element(rev, rea, nums)
    fp = aligned_rmsd_by_element(fwd, pro, nums) if pro is not None else None
    rp = aligned_rmsd_by_element(rev, pro, nums) if pro is not None else None
    found_R = (fr < rmsd_thresh) or (rr < rmsd_thresh)
    found_P = (fp is not None and fp < rmsd_thresh) or (rp is not None and rp < rmsd_thresh)
    rmsd_intended = found_R and found_P

    # TOPO scoring at this cutoff_scale
    try:
        Gr = coords_to_bond_graph(rea, nums_t, cutoff_scale=cutoff_scale)
        Gp = coords_to_bond_graph(pro, nums_t, cutoff_scale=cutoff_scale) if pro is not None else None
        Gf = coords_to_bond_graph(fwd, nums_t, cutoff_scale=cutoff_scale)
        Gv = coords_to_bond_graph(rev, nums_t, cutoff_scale=cutoff_scale)
        # Direction-agnostic: (fwd~R AND rev~P) OR (fwd~P AND rev~R)
        topo_intended = (
            (bond_graphs_match(Gf, Gr) and bond_graphs_match(Gv, Gp))
            or (bond_graphs_match(Gf, Gp) and bond_graphs_match(Gv, Gr))
        )
    except Exception:
        topo_intended = False

    return {"rmsd_intended": rmsd_intended, "topo_intended": topo_intended}


def main():
    rows = []
    rmsd_thresholds = [0.3, 0.4, 0.5, 0.7]
    cutoff_scales = [1.1, 1.2, 1.3, 1.4]

    for label, mdir in METHODS.items():
        for noise in NOISES:
            p = RUNS / mdir / f"irc_validation_sella_hip_allendpoints_{noise}pm.parquet"
            if not p.exists(): continue
            try:
                df = reload_data(p)
            except Exception as e:
                print(f"err {label} {noise}pm: {e}"); continue
            if not len(df): continue
            n_total = len(df)
            print(f"{label} {noise}pm: scoring {n_total} samples...")

            # For each (sample × threshold combo), score; aggregate
            counts = {}  # (rmsd_thresh, cutoff) -> [n_rmsd_int, n_topo_int]
            for thr in rmsd_thresholds:
                for cut in cutoff_scales:
                    counts[(thr, cut)] = [0, 0]

            for _, row in df.iterrows():
                nums = np.asarray(row["atomic_nums"], dtype=np.int64)
                n = len(nums)
                # Handle NA endpoints: skip the sample if either direction failed.
                fwd_flat = row["forward_coords_flat"]
                rev_flat = row["reverse_coords_flat"]
                if fwd_flat is None or rev_flat is None: continue
                try:
                    fwd = np.asarray(list(fwd_flat), dtype=np.float64).reshape(n, 3)
                    rev = np.asarray(list(rev_flat), dtype=np.float64).reshape(n, 3)
                except Exception:
                    continue
                rea = np.asarray(list(row["reactant_coords_flat"]), dtype=np.float64).reshape(n, 3)
                pro_flat = row["product_coords_flat"]
                if pro_flat is None:
                    pro = None
                else:
                    pro_arr = np.asarray(list(pro_flat), dtype=np.float64).reshape(n, 3)
                    pro = pro_arr if not np.all(pro_arr == 0) else None
                nums_t = torch.tensor(nums)
                # TOPO scoring per cutoff
                topo_per_cut = {}
                for cut in cutoff_scales:
                    try:
                        Gr = coords_to_bond_graph(rea, nums_t, cutoff_scale=cut)
                        Gp = coords_to_bond_graph(pro, nums_t, cutoff_scale=cut) if pro is not None else None
                        Gf = coords_to_bond_graph(fwd, nums_t, cutoff_scale=cut)
                        Gv = coords_to_bond_graph(rev, nums_t, cutoff_scale=cut)
                        topo = (
                            (bond_graphs_match(Gf, Gr) and bond_graphs_match(Gv, Gp))
                            or (bond_graphs_match(Gf, Gp) and bond_graphs_match(Gv, Gr))
                        )
                    except Exception:
                        topo = False
                    topo_per_cut[cut] = topo

                # RMSD depends only on threshold (not cutoff)
                fr = aligned_rmsd_by_element(fwd, rea, nums)
                rr = aligned_rmsd_by_element(rev, rea, nums)
                fp = aligned_rmsd_by_element(fwd, pro, nums) if pro is not None else None
                rp = aligned_rmsd_by_element(rev, pro, nums) if pro is not None else None

                for thr in rmsd_thresholds:
                    found_R = (fr < thr) or (rr < thr)
                    found_P = (fp is not None and fp < thr) or (rp is not None and rp < thr)
                    rmsd_int = found_R and found_P
                    for cut in cutoff_scales:
                        counts[(thr, cut)][0] += int(rmsd_int)
                        counts[(thr, cut)][1] += int(topo_per_cut[cut])

            for (thr, cut), (n_r, n_t) in counts.items():
                rows.append({
                    "method": label, "noise_pm": noise, "n_total": n_total,
                    "rmsd_threshold": thr, "cutoff_scale": cut,
                    "rmsd_intended_pct": 100*n_r/n_total,
                    "topo_intended_pct": 100*n_t/n_total,
                })

    out = pd.DataFrame(rows)
    out.to_csv(OUT / "irc_sensitivity.csv", index=False)
    print(f"\nwrote {OUT/'irc_sensitivity.csv'} ({len(out)} rows)")

    # Summary tables
    print("\n=== TOPO-intended % at default cutoff=1.2 (canonical), various RMSD-N/A ===")
    print("(TOPO doesn't depend on RMSD threshold)")
    sub = out[(out["cutoff_scale"]==1.2) & (out["rmsd_threshold"]==0.3)]
    print(sub.pivot(index="method", columns="noise_pm", values="topo_intended_pct").round(1).to_string())

    print("\n=== TOPO-intended % at cutoff=1.3 ===")
    sub = out[(out["cutoff_scale"]==1.3) & (out["rmsd_threshold"]==0.3)]
    print(sub.pivot(index="method", columns="noise_pm", values="topo_intended_pct").round(1).to_string())

    print("\n=== RMSD-intended % at threshold=0.3 (canonical) ===")
    sub = out[(out["rmsd_threshold"]==0.3) & (out["cutoff_scale"]==1.2)]
    print(sub.pivot(index="method", columns="noise_pm", values="rmsd_intended_pct").round(1).to_string())

    print("\n=== RMSD-intended % at threshold=0.5 (relaxed) ===")
    sub = out[(out["rmsd_threshold"]==0.5) & (out["cutoff_scale"]==1.2)]
    print(sub.pivot(index="method", columns="noise_pm", values="rmsd_intended_pct").round(1).to_string())


if __name__ == "__main__":
    main()
