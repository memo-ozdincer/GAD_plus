"""Parallel IRC validation on SCINE-converged TSs.

For each converged sample in the SCINE main-result parquets, read the
matching per-sample trajectory parquet, pull the geometry at the
recorded `converged_step`, and run Sella IRC forward + reverse with the
SCINE predict_fn. Score with bond-graph isomorphism against the
known reactant/product.

Output: irc_validation_<noise>pm_<method>.parquet with one row per
converged sample, columns matching the HIP-side irc_validation parquets.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def _build_predict_fn():
    from gadplus.calculator.scine import (
        load_scine_calculator, make_scine_predict_fn,
    )
    return make_scine_predict_fn(load_scine_calculator("DFTB0"))


def _validate_one(args_tuple):
    (
        sample_id, ts_coords_np, atomic_nums_list,
        reactant_np, product_np, formula, rxn,
        max_irc_steps, rmsd_threshold,
    ) = args_tuple

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import torch
    torch.set_num_threads(1)

    from gadplus.search.irc_validate import run_irc_validation

    ts_coords = torch.tensor(ts_coords_np, dtype=torch.float64).reshape(-1, 3)
    z = torch.tensor(atomic_nums_list, dtype=torch.long)
    reactant = torch.tensor(reactant_np, dtype=torch.float64).reshape(-1, 3) if reactant_np is not None else None
    product = torch.tensor(product_np, dtype=torch.float64).reshape(-1, 3) if product_np is not None else None

    predict_fn = _build_predict_fn()

    t0 = time.time()
    try:
        res = run_irc_validation(
            ts_coords=ts_coords,
            atomic_nums=z,
            predict_fn=predict_fn,
            reactant_coords=reactant,
            product_coords=product,
            rmsd_threshold=rmsd_threshold,
            max_steps=max_irc_steps,
        )
        err = res.error or ""
        # TOPO-intended = (fwd matches R & rev matches P) OR (fwd matches P & rev matches R)
        topo_intended = bool(
            (res.forward_graph_matches_reactant and res.reverse_graph_matches_product)
            or (res.forward_graph_matches_product and res.reverse_graph_matches_reactant)
        )
        # TOPO-half = exactly one of the two ends matches one of R/P
        n_matches = int(res.forward_graph_matches_reactant) + int(res.forward_graph_matches_product) \
                  + int(res.reverse_graph_matches_reactant) + int(res.reverse_graph_matches_product)
        topo_half = (not topo_intended) and (n_matches >= 1)
    except Exception as exc:
        err = repr(exc)
        topo_intended = False
        topo_half = False
        res = None

    wall = time.time() - t0

    return {
        "sample_id": int(sample_id),
        "formula": str(formula),
        "rxn": str(rxn),
        "topo_intended": topo_intended,
        "topo_half": topo_half,
        "forward_match_reactant": bool(res.forward_graph_matches_reactant) if res else False,
        "forward_match_product":  bool(res.forward_graph_matches_product)  if res else False,
        "reverse_match_reactant": bool(res.reverse_graph_matches_reactant) if res else False,
        "reverse_match_product":  bool(res.reverse_graph_matches_product)  if res else False,
        "wall_time_s": float(wall),
        "error": err,
    }


def _ts_coords_from_trajectory(traj_path: str, converged_step: int) -> tuple[np.ndarray, list[int], str, str, np.ndarray | None, np.ndarray | None]:
    """Read per-sample trajectory, return (coords_at_step, atomic_nums,
    formula, rxn, reactant_coords, product_coords). Reactant/product live
    in the *dataset* — return None here and let caller fetch from h5.
    """
    import pyarrow.parquet as pq
    df = pq.read_table(traj_path).to_pandas()
    if len(df) == 0:
        raise ValueError(f"empty trajectory: {traj_path}")
    step_col = "step"
    if step_col not in df.columns:
        # fallback: row index
        target = df.iloc[converged_step if converged_step >= 0 else -1]
    else:
        mask = df[step_col] == converged_step
        if not mask.any():
            target = df.iloc[-1]
        else:
            target = df[mask].iloc[0]
    coords = np.asarray(target["coords_flat"], dtype=np.float64).reshape(-1, 3)
    formula = str(target.get("formula", ""))
    rxn = str(target.get("rxn", ""))
    return coords, formula, rxn


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--summary-parquet", required=True,
                   help="Path to a SCINE main summary_*.parquet")
    p.add_argument("--traj-dir", required=True,
                   help="Directory containing the per-sample traj_*.parquet")
    p.add_argument("--noise-pm", type=int, required=True)
    p.add_argument("--method-tag", required=True,
                   help="Identifier for output filename (e.g. 'gad', 'sella')")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--max-irc-steps", type=int, default=500)
    p.add_argument("--rmsd-threshold", type=float, default=0.3)
    p.add_argument("--max-validate", type=int, default=0,
                   help="Cap on number of samples to validate. 0 = all converged.")
    p.add_argument("--n-workers", type=int,
                   default=int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
    p.add_argument("--h5", default="/lustre06/project/6033559/memoozd/data/transition1x.h5")
    p.add_argument("--split", default="test")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading summary: {args.summary_parquet}")
    summary = pq.read_table(args.summary_parquet).to_pandas()
    summary = summary[summary["converged"]].reset_index(drop=True)
    if args.max_validate and args.max_validate < len(summary):
        summary = summary.head(args.max_validate)
    print(f"Converged samples to validate: {len(summary)}")
    if len(summary) == 0:
        print("Nothing to do — skipping IRC.")
        # Still write empty parquet to mark completion.
        empty_path = os.path.join(args.output_dir, f"irc_validation_{args.noise_pm}pm_{args.method_tag}.parquet")
        pq.write_table(pa.Table.from_pylist([]), empty_path)
        return

    # Load reactant/product from h5 for the relevant samples
    from gadplus.data.transition1x import Transition1xDataset, UsePos
    ds = Transition1xDataset(
        args.h5, split=args.split,
        max_samples=int(summary["sample_id"].max()) + 1,
        transform=UsePos("pos_transition"),
    )

    task_args = []
    for _, row in summary.iterrows():
        sid = int(row["sample_id"])
        # Prefer final_coords_flat from the summary (Sella runs save it there).
        # Fall back to the per-sample trajectory (GAD runs).
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
                print(f"  WARN: no coords for sample {sid}")
                continue
            ts_coords, _f, _r = _ts_coords_from_trajectory(
                traj_paths[0], int(row["converged_step"]),
            )
        sample = ds[sid]
        atomic_nums = sample.z.detach().cpu().tolist()
        reactant_np = sample.pos_reactant.detach().cpu().numpy() \
            if hasattr(sample, "pos_reactant") and sample.pos_reactant is not None else None
        product_np = sample.pos_product.detach().cpu().numpy() \
            if hasattr(sample, "pos_product") and sample.pos_product is not None else None
        if product_np is not None and float(np.abs(product_np).sum()) < 1e-6:
            product_np = None
        task_args.append((
            sid, ts_coords, atomic_nums, reactant_np, product_np,
            str(row["formula"]), str(row.get("rxn", "")),
            args.max_irc_steps, args.rmsd_threshold,
        ))

    print(f"Submitting {len(task_args)} IRC tasks to {args.n_workers} workers")
    results = []
    t_overall = time.time()
    with ProcessPoolExecutor(max_workers=args.n_workers) as exe:
        for fut in as_completed([exe.submit(_validate_one, ta) for ta in task_args]):
            try:
                r = fut.result()
            except Exception as exc:
                print(f"  worker failed: {exc}")
                continue
            tag = "TOPO" if r["topo_intended"] else ("half" if r["topo_half"] else "miss")
            print(f"  [{r['sample_id']:>3}] {r['formula']:>14}  {tag}  wall={r['wall_time_s']:.1f}s")
            results.append(r)

    out_path = os.path.join(
        args.output_dir,
        f"irc_validation_{args.noise_pm}pm_{args.method_tag}.parquet",
    )
    pq.write_table(pa.Table.from_pylist(results), out_path)
    total_wall = time.time() - t_overall
    n = len(results)
    n_topo = sum(1 for r in results if r["topo_intended"])
    n_half = sum(1 for r in results if r["topo_half"])
    print()
    print("=" * 60)
    print(f"IRC validation @ {args.noise_pm}pm ({args.method_tag}):")
    print(f"  N validated: {n}")
    print(f"  TOPO-intended: {n_topo} ({100*n_topo/max(n,1):.1f}%)")
    print(f"  TOPO-half:     {n_half} ({100*n_half/max(n,1):.1f}%)")
    print(f"  Wall: {total_wall:.1f}s")
    print(f"  Wrote {out_path}")


if __name__ == "__main__":
    main()
