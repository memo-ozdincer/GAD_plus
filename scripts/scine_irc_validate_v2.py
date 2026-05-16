"""IRC validation v2 — adds post-IRC steepest descent.

Debug showed ~25-30% of SCINE-converged-not-TOPO samples have one IRC
direction that hit the 500-step cap or `fmax<0.01` termination while
still on a ridge (n_neg=1 at endpoint). Bond-graph at a ridge endpoint
is unreliable.

Fix: after Sella IRC stops, run a few hundred ASE BFGS steps to force
the endpoint into a real minimum (n_neg=0, fmax → 0). Then build the
bond graph at the relaxed endpoint.

CLI matches scripts/scine_irc_validate.py for drop-in usage.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import glob
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def _build_predict_fn():
    from gadplus.calculator.scine import (
        load_scine_calculator, make_scine_predict_fn,
    )
    return make_scine_predict_fn(load_scine_calculator("DFTB0"))


def _relax_endpoint(coords_np, atomic_nums, predict_fn, fmax, max_steps):
    """ASE BFGS minimization from `coords_np`. Returns the relaxed
    coords as numpy. Falls back to input coords on any failure."""
    from ase import Atoms
    from ase.optimize import BFGS
    from gadplus.projection import Z_TO_SYMBOL
    from gadplus.calculator.ase_adapter import HipASECalculator

    try:
        nums = atomic_nums.detach().cpu().tolist()
        symbols = [Z_TO_SYMBOL.get(int(z), "X") for z in nums]
        atoms = Atoms(symbols=symbols, positions=coords_np)
        atoms.calc = HipASECalculator(predict_fn=predict_fn, atomic_nums=atomic_nums)
        BFGS(atoms, logfile=None).run(fmax=fmax, steps=max_steps)
        return atoms.positions.copy()
    except Exception:
        return coords_np.copy()


def _validate_one(args_tuple):
    (
        sample_id, ts_coords_np, atomic_nums_list,
        reactant_np, product_np, formula, rxn,
        max_irc_steps, max_relax_steps, relax_fmax, rmsd_threshold,
    ) = args_tuple

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    torch.set_num_threads(1)

    from gadplus.search.irc_validate import run_irc_validation, score_endpoints

    ts_coords = torch.tensor(ts_coords_np, dtype=torch.float64).reshape(-1, 3)
    z = torch.tensor(atomic_nums_list, dtype=torch.long)
    R = torch.tensor(reactant_np, dtype=torch.float64).reshape(-1, 3) if reactant_np is not None else None
    P = torch.tensor(product_np, dtype=torch.float64).reshape(-1, 3) if product_np is not None else None

    predict_fn = _build_predict_fn()

    t0 = time.time()
    try:
        # Phase 1: standard Sella IRC
        res0 = run_irc_validation(
            ts_coords=ts_coords, atomic_nums=z, predict_fn=predict_fn,
            reactant_coords=R, product_coords=P,
            rmsd_threshold=rmsd_threshold, max_steps=max_irc_steps,
        )
        fwd0 = res0.forward_coords
        rev0 = res0.reverse_coords

        # Phase 2: force each endpoint to a minimum via BFGS
        fwd_relaxed = (
            _relax_endpoint(fwd0, z, predict_fn, relax_fmax, max_relax_steps)
            if fwd0 is not None else None
        )
        rev_relaxed = (
            _relax_endpoint(rev0, z, predict_fn, relax_fmax, max_relax_steps)
            if rev0 is not None else None
        )

        # Phase 3: rescore against R/P with relaxed endpoints
        res = score_endpoints(
            forward_coords=fwd_relaxed,
            reverse_coords=rev_relaxed,
            atomic_nums=z,
            reactant_coords=R, product_coords=P,
            rmsd_threshold=rmsd_threshold,
            predict_fn=predict_fn,
        )
        err = ""
    except Exception as exc:
        err = repr(exc)
        res = None
        res0 = None

    wall = time.time() - t0

    if res is None:
        return {
            "sample_id": int(sample_id),
            "formula": str(formula), "rxn": str(rxn),
            "topo_intended": False, "topo_half": False,
            "topo_intended_phase1": False,
            "forward_match_reactant": False, "forward_match_product": False,
            "reverse_match_reactant": False, "reverse_match_product": False,
            "fwd_n_neg_vib": -1, "rev_n_neg_vib": -1,
            "fwd_min_vib_eig": float("nan"), "rev_min_vib_eig": float("nan"),
            "wall_time_s": float(wall), "error": err,
        }

    topo_intended = bool(
        (res.forward_graph_matches_reactant and res.reverse_graph_matches_product)
        or (res.forward_graph_matches_product and res.reverse_graph_matches_reactant)
    )
    n_matches = (int(res.forward_graph_matches_reactant)
                 + int(res.forward_graph_matches_product)
                 + int(res.reverse_graph_matches_reactant)
                 + int(res.reverse_graph_matches_product))
    topo_half = (not topo_intended) and (n_matches >= 1)
    topo_phase1 = bool(res0.topology_intended) if res0 else False

    return {
        "sample_id": int(sample_id),
        "formula": str(formula), "rxn": str(rxn),
        "topo_intended": topo_intended,
        "topo_half": topo_half,
        "topo_intended_phase1": topo_phase1,
        "forward_match_reactant": bool(res.forward_graph_matches_reactant),
        "forward_match_product":  bool(res.forward_graph_matches_product),
        "reverse_match_reactant": bool(res.reverse_graph_matches_reactant),
        "reverse_match_product":  bool(res.reverse_graph_matches_product),
        "fwd_n_neg_vib": int(res.forward_n_neg_vib) if res.forward_n_neg_vib is not None else -1,
        "rev_n_neg_vib": int(res.reverse_n_neg_vib) if res.reverse_n_neg_vib is not None else -1,
        "fwd_min_vib_eig": float(res.forward_min_vib_eig) if res.forward_min_vib_eig is not None else float("nan"),
        "rev_min_vib_eig": float(res.reverse_min_vib_eig) if res.reverse_min_vib_eig is not None else float("nan"),
        "wall_time_s": float(wall), "error": err,
    }


def _ts_coords_from_trajectory(traj_path, converged_step):
    df = pq.read_table(traj_path).to_pandas()
    if "step" in df.columns:
        mask = df["step"] == converged_step
        target = df[mask].iloc[0] if mask.any() else df.iloc[-1]
    else:
        target = df.iloc[-1]
    return np.asarray(target["coords_flat"], dtype=np.float64).reshape(-1, 3)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--summary-parquet", required=True)
    p.add_argument("--traj-dir", required=True)
    p.add_argument("--noise-pm", type=int, required=True)
    p.add_argument("--method-tag", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--max-irc-steps", type=int, default=500)
    p.add_argument("--max-relax-steps", type=int, default=500)
    p.add_argument("--relax-fmax", type=float, default=0.001)
    p.add_argument("--rmsd-threshold", type=float, default=0.3)
    p.add_argument("--max-validate", type=int, default=0)
    p.add_argument("--n-workers", type=int,
                   default=int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
    p.add_argument("--h5", default="/lustre06/project/6033559/memoozd/data/transition1x.h5")
    p.add_argument("--split", default="test")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    summary = pq.read_table(args.summary_parquet).to_pandas()
    summary = summary[summary["converged"]].reset_index(drop=True)
    if args.max_validate and args.max_validate < len(summary):
        summary = summary.head(args.max_validate)
    print(f"Converged samples to validate: {len(summary)}")

    if len(summary) == 0:
        empty = os.path.join(args.output_dir, f"irc_validation_v2_{args.noise_pm}pm_{args.method_tag}.parquet")
        pq.write_table(pa.Table.from_pylist([]), empty)
        return

    from gadplus.data.transition1x import Transition1xDataset, UsePos
    ds = Transition1xDataset(
        args.h5, split=args.split,
        max_samples=int(summary["sample_id"].max()) + 1,
        transform=UsePos("pos_transition"),
    )

    task_args = []
    for _, row in summary.iterrows():
        sid = int(row["sample_id"])
        ts_coords = None
        if "final_coords_flat" in row.index and row["final_coords_flat"] is not None:
            try:
                ts_coords = np.asarray(row["final_coords_flat"], dtype=np.float64).reshape(-1, 3)
            except Exception:
                ts_coords = None
        if ts_coords is None:
            traj_paths = [
                os.path.join(args.traj_dir, fn)
                for fn in os.listdir(args.traj_dir)
                if fn.startswith("traj_") and fn.endswith(f"_{sid}.parquet")
            ]
            if not traj_paths:
                continue
            ts_coords = _ts_coords_from_trajectory(traj_paths[0], int(row["converged_step"]))
        sample = ds[sid]
        atomic_nums = sample.z.detach().cpu().tolist()
        R = sample.pos_reactant.detach().cpu().numpy() if hasattr(sample, "pos_reactant") and sample.pos_reactant is not None else None
        P = sample.pos_product.detach().cpu().numpy() if hasattr(sample, "pos_product") and sample.pos_product is not None else None
        if P is not None and float(np.abs(P).sum()) < 1e-6:
            P = None
        task_args.append((
            sid, ts_coords, atomic_nums, R, P,
            str(row["formula"]), str(row.get("rxn", "")),
            args.max_irc_steps, args.max_relax_steps, args.relax_fmax,
            args.rmsd_threshold,
        ))

    print(f"Dispatching {len(task_args)} samples on {args.n_workers} workers")
    results = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.n_workers) as exe:
        for fut in as_completed([exe.submit(_validate_one, ta) for ta in task_args]):
            try:
                r = fut.result()
            except Exception as exc:
                print(f"  worker failed: {exc}")
                continue
            tag = "TOPO" if r["topo_intended"] else ("half" if r["topo_half"] else "miss")
            note = ""
            if r["topo_intended"] and not r["topo_intended_phase1"]:
                note = " ★(saved by relax)"
            print(f"  [{r['sample_id']:>3}] {r['formula']:>14}  {tag}  "
                  f"fwd_n_neg={r['fwd_n_neg_vib']} rev_n_neg={r['rev_n_neg_vib']}  "
                  f"wall={r['wall_time_s']:.1f}s{note}")
            results.append(r)

    out = os.path.join(args.output_dir, f"irc_validation_v2_{args.noise_pm}pm_{args.method_tag}.parquet")
    pq.write_table(pa.Table.from_pylist(results), out)
    n = len(results)
    n_topo = sum(1 for r in results if r["topo_intended"])
    n_topo1 = sum(1 for r in results if r["topo_intended_phase1"])
    saved = sum(1 for r in results if r["topo_intended"] and not r["topo_intended_phase1"])
    print(f"\n=== {args.method_tag} @ {args.noise_pm}pm ===")
    print(f"  N={n}  TOPO(v2)={n_topo} ({100*n_topo/max(n,1):.1f}%)  "
          f"TOPO(v1,IRC only)={n_topo1}  saved by relax: {saved}")
    print(f"  wall={time.time()-t0:.1f}s")
    print(f"  Wrote {out}")


if __name__ == "__main__":
    main()
