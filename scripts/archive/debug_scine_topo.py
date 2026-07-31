"""Debug why SCINE IRC TOPO rates are low.

Per-sample diagnostic at 10pm noise:
  1. For each SCINE-GAD-converged-but-not-TOPO sample, retrieve the
     converged TS coords.
  2. Re-run Sella IRC forward + reverse with the SCINE predict_fn.
  3. Build bond graphs at: T1x reactant, T1x product, SCINE IRC forward
     endpoint, SCINE IRC reverse endpoint. Print the edge sets so we
     can see WHY graphs differ.
  4. Sweep cutoff_scale ∈ {1.10, 1.15, 1.20, 1.25, 1.30, 1.35, 1.40}
     and report at which cutoff the IRC endpoint matches T1x R/P.
  5. Also relax T1x reactant + product under DFTB0 (short BFGS) and
     compare bond graphs of HIP-R vs DFTB0-R, HIP-P vs DFTB0-P.

Output: stdout table + analysis_2026_04_29/scine_topo_debug_10pm.csv
"""
from __future__ import annotations

import csv
import os
import sys
import argparse

import numpy as np
import pyarrow.parquet as pq
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


CUTOFFS = [1.10, 1.15, 1.20, 1.25, 1.30, 1.35, 1.40, 1.50]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--max-samples", type=int, default=20,
                   help="Number of SCINE-converged-not-TOPO samples to probe")
    p.add_argument("--summary", default="/lustre07/scratch/memoozd/gadplus/runs/main_scine_gad15k_60865063/noise10pm")
    p.add_argument("--irc-result", default="/lustre07/scratch/memoozd/gadplus/runs/scine_irc15k_60865129/gad/irc_validation_10pm_gad.parquet")
    p.add_argument("--out", default="analysis_2026_04_29/scine_topo_debug_10pm.csv")
    args = p.parse_args()

    from gadplus.calculator.scine import load_scine_calculator, make_scine_predict_fn
    from gadplus.data.transition1x import Transition1xDataset, UsePos
    from gadplus.search.irc_validate import (
        run_irc_validation, coords_to_bond_graph, bond_graphs_match,
    )

    print("Loading SCINE-GAD summary and IRC results...")
    import glob
    sp = glob.glob(os.path.join(args.summary, "summary_*.parquet"))[0]
    summary = pq.read_table(sp).to_pandas()
    irc = pq.read_table(args.irc_result).to_pandas()

    merged = summary.merge(irc[["sample_id", "topo_intended"]], on="sample_id")
    fail = merged[merged["converged"] & (~merged["topo_intended"])].head(args.max_samples)
    success = merged[merged["converged"] & merged["topo_intended"]].head(3)
    print(f"  {len(fail)} converged-not-TOPO probes, {len(success)} converged-TOPO controls")

    print("\nLoading SCINE calculator + dataset...")
    predict_fn = make_scine_predict_fn(load_scine_calculator("DFTB0"))
    ds = Transition1xDataset(
        h5_path="/lustre06/project/6033559/memoozd/data/transition1x.h5",
        split="test", max_samples=287,
        transform=UsePos("pos_transition"),
    )

    rows = []
    for category, df in [("FAIL", fail), ("SUCCESS", success)]:
        for _, r in df.iterrows():
            sid = int(r["sample_id"])
            print(f"\n=== {category} sample {sid} ({r['formula']}) ===")
            sample = ds[sid]
            z = sample.z.to(torch.long)
            R = sample.pos_reactant.detach().cpu().numpy() if hasattr(sample, "pos_reactant") else None
            P = sample.pos_product.detach().cpu().numpy() if hasattr(sample, "pos_product") else None
            if R is None or P is None:
                print(f"  skip — no R/P")
                continue

            # Get GAD-converged TS coords from trajectory parquet
            traj_path = os.path.join(args.summary, f"traj_*_{sid}.parquet")
            tp = glob.glob(traj_path)
            if not tp:
                print(f"  skip — no traj for {sid}")
                continue
            traj = pq.read_table(tp[0]).to_pandas()
            cstep = int(r["converged_step"]) if r["converged_step"] >= 0 else -1
            if cstep >= 0 and (traj["step"] == cstep).any():
                target = traj[traj["step"] == cstep].iloc[0]
            else:
                target = traj.iloc[-1]
            ts_coords = np.asarray(target["coords_flat"], dtype=np.float64).reshape(-1, 3)

            # Run IRC
            ts_t = torch.tensor(ts_coords, dtype=torch.float64)
            R_t = torch.tensor(R, dtype=torch.float64)
            P_t = torch.tensor(P, dtype=torch.float64)
            res = run_irc_validation(
                ts_coords=ts_t, atomic_nums=z, predict_fn=predict_fn,
                reactant_coords=R_t, product_coords=P_t,
                rmsd_threshold=0.3, max_steps=500,
            )
            fwd = res.forward_coords
            rev = res.reverse_coords

            # Spectral diagnostic at endpoints (already in res)
            print(f"  fwd_min_eig={res.forward_min_vib_eig}, fwd_n_neg={res.forward_n_neg_vib}")
            print(f"  rev_min_eig={res.reverse_min_vib_eig}, rev_n_neg={res.reverse_n_neg_vib}")
            print(f"  fwd RMSD R/P: {res.forward_rmsd_to_reactant:.3f}/{res.forward_rmsd_to_product:.3f}")
            print(f"  rev RMSD R/P: {res.reverse_rmsd_to_reactant:.3f}/{res.reverse_rmsd_to_product:.3f}")
            print(f"  default-1.2 match (fwd→R, fwd→P, rev→R, rev→P): "
                  f"{res.forward_graph_matches_reactant}/{res.forward_graph_matches_product}/"
                  f"{res.reverse_graph_matches_reactant}/{res.reverse_graph_matches_product}")

            # Cutoff sweep
            sweep = {}
            for c in CUTOFFS:
                try:
                    R_g = coords_to_bond_graph(R, z, cutoff_scale=c)
                    P_g = coords_to_bond_graph(P, z, cutoff_scale=c)
                    fwd_g = coords_to_bond_graph(fwd, z, cutoff_scale=c) if fwd is not None else None
                    rev_g = coords_to_bond_graph(rev, z, cutoff_scale=c) if rev is not None else None
                    fwd_R = bond_graphs_match(fwd_g, R_g)
                    fwd_P = bond_graphs_match(fwd_g, P_g)
                    rev_R = bond_graphs_match(rev_g, R_g)
                    rev_P = bond_graphs_match(rev_g, P_g)
                    topo = (fwd_R and rev_P) or (fwd_P and rev_R)
                    # Edge counts at this cutoff
                    R_e = R_g.number_of_edges()
                    P_e = P_g.number_of_edges()
                    fwd_e = fwd_g.number_of_edges() if fwd_g is not None else -1
                    rev_e = rev_g.number_of_edges() if rev_g is not None else -1
                    sweep[c] = (topo, fwd_R, fwd_P, rev_R, rev_P, R_e, P_e, fwd_e, rev_e)
                except Exception as exc:
                    sweep[c] = (False, False, False, False, False, -1, -1, -1, -1)

            # Print cutoff sweep table
            print(f"  cutoff  topo  fwd_e/R_e  rev_e/P_e  (fwdR,fwdP,revR,revP)")
            for c in CUTOFFS:
                t, fR, fP, rR, rP, Re, Pe, fe, re = sweep[c]
                mark = "←TOPO" if t else ""
                print(f"    {c:.2f}  {str(t):>5}    {fe:>3}/{Re:>3}    {re:>3}/{Pe:>3}    ({fR},{fP},{rR},{rP}) {mark}")

            rows.append({
                "sample_id": sid,
                "formula": str(r["formula"]),
                "category": category,
                "default_topo": bool(res.topology_intended),
                "fwd_min_eig": res.forward_min_vib_eig,
                "rev_min_eig": res.reverse_min_vib_eig,
                "fwd_n_neg": res.forward_n_neg_vib,
                "rev_n_neg": res.reverse_n_neg_vib,
                **{f"topo_cut_{c:.2f}": sweep[c][0] for c in CUTOFFS},
                "R_edges_1.20": sweep[1.20][5],
                "P_edges_1.20": sweep[1.20][6],
                "fwd_edges_1.20": sweep[1.20][7],
                "rev_edges_1.20": sweep[1.20][8],
            })

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    if rows:
        with open(args.out, "w") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"\nWrote {args.out}")

    # Summary: at which cutoff does TOPO succeed for the most FAIL samples?
    print("\n=== TOPO recovery vs cutoff for FAIL samples ===")
    fail_rows = [r for r in rows if r["category"] == "FAIL"]
    for c in CUTOFFS:
        n = sum(1 for r in fail_rows if r[f"topo_cut_{c:.2f}"])
        print(f"  cutoff={c:.2f}: {n}/{len(fail_rows)} now TOPO")


if __name__ == "__main__":
    main()
