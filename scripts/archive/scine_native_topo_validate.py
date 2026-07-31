#!/usr/bin/env python
"""Validate SCINE IRCs against calculator-native reactant/product labels.

For each converged candidate saddle, this script:

1. Relaxes the labeled T1x reactant/product on SCINE DFTB0 and caches them.
2. Runs the candidate's SCINE IRC in both directions.
3. Relaxes both IRC endpoints to SCINE minima.
4. Scores the endpoints against both the original T1x labels and the cached
   DFTB0-native labels.

The native score is deliberately conservative. A sample is evaluable only if
both native reference labels relaxed to minima and remain distinct. This keeps
calculator-PES relabeling from turning a collapsed or ill-defined reaction
into an artificial success.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
from pathlib import Path
import sys
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def parse_sample_ids(value: str | None) -> set[int] | None:
    if value is None:
        return None
    ids = {int(token) for token in value.split(",") if token.strip()}
    if not ids:
        raise ValueError("--sample-ids did not contain any ids")
    return ids


def _build_predict_fn(functional: str):
    from gadplus.calculator.scine import load_scine_calculator, make_scine_predict_fn

    return make_scine_predict_fn(load_scine_calculator(functional))


def _load_t1x_endpoint_records(h5_path: str, split: str, max_sample_id: int) -> list[dict]:
    """Load T1x R/P records without importing torch_geometric.

    Sample ids in the SCINE summaries follow ``Transition1xDataset``'s
    filtered-record order, rather than the raw HDF5 iterator index. Mirror
    that filtering here so native labels use the same reaction as the
    candidate saddle while avoiding the heavy torch_geometric import path.
    """
    from transition1x import Dataloader

    records = []
    loader = Dataloader(h5_path, datasplit=split, only_final=True)
    for molecule in loader:
        if len(records) > max_sample_id:
            break
        try:
            ts = molecule["transition_state"]
            reactant = molecule["reactant"]
            if len(ts["atomic_numbers"]) != len(reactant["atomic_numbers"]):
                continue
            product = molecule.get("product")
            has_product = (
                product is not None
                and len(product.get("atomic_numbers", [])) == len(ts["atomic_numbers"])
            )
            records.append({
                "atomic_nums": np.asarray(ts["atomic_numbers"], dtype=np.int64),
                "reactant": np.asarray(reactant["positions"], dtype=np.float64),
                "product": (
                    np.asarray(product["positions"], dtype=np.float64)
                    if has_product
                    else None
                ),
            })
        except Exception:
            continue
    if len(records) <= max_sample_id:
        raise IndexError(
            f"Requested sample {max_sample_id}, but only {len(records)} valid T1x records were loaded"
        )
    return records


def _coords_from_summary_or_trajectory(row, traj_dir: str | None) -> np.ndarray:
    final_coords = row.get("final_coords_flat")
    if final_coords is not None:
        try:
            return np.asarray(final_coords, dtype=np.float64).reshape(-1, 3)
        except Exception:
            pass

    if not traj_dir:
        raise ValueError("candidate has no final_coords_flat and --traj-dir was not supplied")

    sample_id = int(row["sample_id"])
    candidates = sorted(Path(traj_dir).glob(f"traj_*_{sample_id}.parquet"))
    if not candidates:
        raise FileNotFoundError(f"no trajectory found for sample {sample_id} in {traj_dir}")
    traj = pq.read_table(candidates[0]).to_pandas()
    if traj.empty:
        raise ValueError(f"empty trajectory: {candidates[0]}")
    converged_step = row.get("converged_step")
    if converged_step is not None and "step" in traj.columns:
        matches = traj[traj["step"] == int(converged_step)]
        if not matches.empty:
            return np.asarray(matches.iloc[0]["coords_flat"], dtype=np.float64).reshape(-1, 3)
    return np.asarray(traj.iloc[-1]["coords_flat"], dtype=np.float64).reshape(-1, 3)


def _minimum_ok(result, fmax: float) -> bool:
    return bool(
        result is not None
        and np.isfinite(result.force_max)
        and result.force_max <= 1.05 * fmax
    )


def _reference_pair_status(reactant, product, atomic_nums, collapse_rmsd: float):
    """Return whether native R/P define a distinct two-sided reaction."""
    from gadplus.geometry.alignment import aligned_rmsd_by_element
    from gadplus.search.irc_validate import bond_graphs_match, coords_to_bond_graph

    if reactant is None or product is None:
        return False, False, None, "missing_product"
    try:
        nums = np.asarray(atomic_nums, dtype=np.int64)
        rmsd = float(aligned_rmsd_by_element(reactant.coords, product.coords, nums))
        graph_same = bool(
            bond_graphs_match(
                coords_to_bond_graph(reactant.coords, torch.as_tensor(nums)),
                coords_to_bond_graph(product.coords, torch.as_tensor(nums)),
            )
        )
    except Exception as exc:
        return False, False, None, f"reference_graph_error:{exc}"

    collapsed = graph_same and rmsd <= collapse_rmsd
    if collapsed:
        return False, graph_same, rmsd, "collapsed_native_references"
    return True, graph_same, rmsd, ""


def _score_one(task):
    (
        row_dict,
        atomic_nums_list,
        source_reactant_np,
        source_product_np,
        functional,
        cache_dir,
        max_irc_steps,
        max_relax_steps,
        relax_fmax,
        rmsd_threshold,
        collapse_rmsd,
    ) = task

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    torch.set_num_threads(1)

    from gadplus.geometry.alignment import aligned_rmsd_by_element
    from gadplus.search.irc_validate import run_irc_validation, score_endpoints
    from gadplus.search.native_endpoints import (
        load_or_relax_native_endpoints,
        relax_to_minimum,
    )

    sample_id = int(row_dict["sample_id"])
    formula = str(row_dict.get("formula", ""))
    rxn = str(row_dict.get("rxn", ""))
    ts_coords_np = np.asarray(row_dict["_candidate_coords"], dtype=np.float64).reshape(-1, 3)
    z = torch.as_tensor(atomic_nums_list, dtype=torch.long)
    source_r = np.asarray(source_reactant_np, dtype=np.float64).reshape(-1, 3)
    source_p = (
        np.asarray(source_product_np, dtype=np.float64).reshape(-1, 3)
        if source_product_np is not None
        else None
    )
    t0 = time.time()

    result = {
        "sample_id": sample_id,
        "formula": formula,
        "rxn": rxn,
        "source_topology_intended": False,
        "source_rmsd_intended": False,
        "native_evaluable": False,
        "native_reference_reason": "",
        "native_reference_cache_hit": False,
        "native_reference_graph_same": None,
        "native_reference_rmsd": None,
        "native_reference_reactant_minimum": False,
        "native_reference_product_minimum": False,
        "native_reference_reactant_fmax": None,
        "native_reference_product_fmax": None,
        "native_reference_reactant_steps": None,
        "native_reference_product_steps": None,
        "native_reference_reactant_source_rmsd": None,
        "native_reference_product_source_rmsd": None,
        "native_endpoint_minima": False,
        "native_topology_intended": False,
        "native_rmsd_intended": False,
        "native_intended": False,
        "native_forward_match_reactant": False,
        "native_forward_match_product": False,
        "native_reverse_match_reactant": False,
        "native_reverse_match_product": False,
        "candidate_forward_fmax": None,
        "candidate_reverse_fmax": None,
        "candidate_forward_steps": None,
        "candidate_reverse_steps": None,
        "wall_time_s": None,
        "error": "",
    }

    try:
        predict_fn = _build_predict_fn(functional)
        labels = load_or_relax_native_endpoints(
            cache_dir=cache_dir,
            sample_id=sample_id,
            functional=functional,
            atomic_nums=z,
            reactant_coords=source_r,
            product_coords=source_p,
            predict_fn=predict_fn,
            relax_fmax=relax_fmax,
            max_steps=max_relax_steps,
        )
        native_r = labels.reactant
        native_p = labels.product
        result["native_reference_cache_hit"] = labels.cache_hit
        result["native_reference_reactant_minimum"] = _minimum_ok(native_r, relax_fmax)
        result["native_reference_product_minimum"] = _minimum_ok(native_p, relax_fmax)
        if native_r is not None:
            result["native_reference_reactant_fmax"] = native_r.force_max
            result["native_reference_reactant_steps"] = native_r.steps
            result["native_reference_reactant_source_rmsd"] = float(
                aligned_rmsd_by_element(native_r.coords, source_r, np.asarray(atomic_nums_list))
            )
        if native_p is not None and source_p is not None:
            result["native_reference_product_fmax"] = native_p.force_max
            result["native_reference_product_steps"] = native_p.steps
            result["native_reference_product_source_rmsd"] = float(
                aligned_rmsd_by_element(native_p.coords, source_p, np.asarray(atomic_nums_list))
            )

        if not result["native_reference_reactant_minimum"]:
            result["native_reference_reason"] = "reactant_not_minimized"
        elif not result["native_reference_product_minimum"]:
            result["native_reference_reason"] = "product_not_minimized"
        else:
            evaluable, graph_same, ref_rmsd, reason = _reference_pair_status(
                native_r, native_p, atomic_nums_list, collapse_rmsd,
            )
            result["native_evaluable"] = evaluable
            result["native_reference_graph_same"] = graph_same
            result["native_reference_rmsd"] = ref_rmsd
            result["native_reference_reason"] = reason

        irc = run_irc_validation(
            ts_coords=torch.as_tensor(ts_coords_np, dtype=torch.float64),
            atomic_nums=z,
            predict_fn=predict_fn,
            reactant_coords=None,
            product_coords=None,
            rmsd_threshold=rmsd_threshold,
            max_steps=max_irc_steps,
            logfile=None,
        )
        fwd = (
            relax_to_minimum(
                irc.forward_coords, z, predict_fn, fmax=relax_fmax, max_steps=max_relax_steps,
            )
            if irc.forward_coords is not None
            else None
        )
        rev = (
            relax_to_minimum(
                irc.reverse_coords, z, predict_fn, fmax=relax_fmax, max_steps=max_relax_steps,
            )
            if irc.reverse_coords is not None
            else None
        )
        result["candidate_forward_fmax"] = fwd.force_max if fwd is not None else None
        result["candidate_reverse_fmax"] = rev.force_max if rev is not None else None
        result["candidate_forward_steps"] = fwd.steps if fwd is not None else None
        result["candidate_reverse_steps"] = rev.steps if rev is not None else None
        result["native_endpoint_minima"] = _minimum_ok(fwd, relax_fmax) and _minimum_ok(rev, relax_fmax)

        fwd_coords = fwd.coords if fwd is not None else None
        rev_coords = rev.coords if rev is not None else None
        source_score = score_endpoints(
            forward_coords=fwd_coords,
            reverse_coords=rev_coords,
            atomic_nums=z,
            reactant_coords=torch.as_tensor(source_r, dtype=torch.float64),
            product_coords=(torch.as_tensor(source_p, dtype=torch.float64) if source_p is not None else None),
            rmsd_threshold=rmsd_threshold,
        )
        result["source_topology_intended"] = bool(source_score.topology_intended)
        result["source_rmsd_intended"] = bool(source_score.intended)

        if result["native_evaluable"] and result["native_endpoint_minima"]:
            native_score = score_endpoints(
                forward_coords=fwd_coords,
                reverse_coords=rev_coords,
                atomic_nums=z,
                reactant_coords=torch.as_tensor(native_r.coords, dtype=torch.float64),
                product_coords=torch.as_tensor(native_p.coords, dtype=torch.float64),
                rmsd_threshold=rmsd_threshold,
            )
            result["native_topology_intended"] = bool(native_score.topology_intended)
            result["native_rmsd_intended"] = bool(native_score.intended)
            result["native_forward_match_reactant"] = bool(native_score.forward_graph_matches_reactant)
            result["native_forward_match_product"] = bool(native_score.forward_graph_matches_product)
            result["native_reverse_match_reactant"] = bool(native_score.reverse_graph_matches_reactant)
            result["native_reverse_match_product"] = bool(native_score.reverse_graph_matches_product)
            # For topologically distinct references, require both graph and
            # geometric identity. For conformationally distinct references,
            # graph identity is insufficient, so require the RMSD criterion.
            result["native_intended"] = bool(
                native_score.intended
                and (
                    bool(result["native_reference_graph_same"])
                    or native_score.topology_intended
                )
            )
    except Exception as exc:
        result["error"] = repr(exc)

    result["wall_time_s"] = time.time() - t0
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-parquet", required=True)
    parser.add_argument(
        "--traj-dir",
        default=None,
        help="Needed only when the summary has no final_coords_flat column.",
    )
    parser.add_argument("--noise-pm", type=int, required=True)
    parser.add_argument("--method-tag", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--native-cache-dir", type=Path, required=True)
    parser.add_argument("--functional", default="DFTB0")
    parser.add_argument("--max-irc-steps", type=int, default=500)
    parser.add_argument("--max-relax-steps", type=int, default=500)
    parser.add_argument("--relax-fmax", type=float, default=0.001)
    parser.add_argument("--rmsd-threshold", type=float, default=0.3)
    parser.add_argument(
        "--collapse-rmsd",
        type=float,
        default=0.05,
        help="Same-graph native labels closer than this are not a two-sided reaction.",
    )
    parser.add_argument("--max-validate", type=int, default=0)
    parser.add_argument("--sample-ids", default=None)
    parser.add_argument(
        "--n-workers",
        type=int,
        default=int(os.environ.get("SLURM_CPUS_PER_TASK", "8")),
    )
    parser.add_argument("--h5", default="/lustre06/project/6033559/memoozd/data/transition1x.h5")
    parser.add_argument("--split", default="test")
    args = parser.parse_args()
    if args.n_workers < 1:
        parser.error("--n-workers must be positive")

    selected_ids = parse_sample_ids(args.sample_ids)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = pq.read_table(args.summary_parquet).to_pandas()
    summary = summary[summary["converged"].astype(bool)].copy()
    if selected_ids is not None:
        summary = summary[summary["sample_id"].astype(int).isin(selected_ids)]
    if args.max_validate:
        summary = summary.head(args.max_validate)
    summary = summary.reset_index(drop=True)
    print(f"Converged candidates selected: {len(summary)}")

    out_path = args.output_dir / f"native_irc_validation_{args.noise_pm}pm_{args.method_tag}.parquet"
    if summary.empty:
        pq.write_table(pa.Table.from_pylist([]), out_path)
        print(f"Wrote empty {out_path}")
        return

    source_records = _load_t1x_endpoint_records(
        args.h5,
        args.split,
        int(summary["sample_id"].max()),
    )

    tasks = []
    for _, row in summary.iterrows():
        sample_id = int(row["sample_id"])
        source = source_records[sample_id]
        row_dict = row.to_dict()
        try:
            row_dict["_candidate_coords"] = _coords_from_summary_or_trajectory(row, args.traj_dir).tolist()
        except Exception as exc:
            print(f"  skip {sample_id}: cannot load candidate coordinates: {exc}")
            continue
        tasks.append((
            row_dict,
            source["atomic_nums"].tolist(),
            source["reactant"],
            source["product"],
            args.functional,
            os.fspath(args.native_cache_dir),
            args.max_irc_steps,
            args.max_relax_steps,
            args.relax_fmax,
            args.rmsd_threshold,
            args.collapse_rmsd,
        ))

    print(
        f"Validating {len(tasks)} candidates with {args.n_workers} workers; "
        f"native cache: {args.native_cache_dir}",
        flush=True,
    )
    start = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=args.n_workers) as executor:
        futures = [executor.submit(_score_one, task) for task in tasks]
        for future in as_completed(futures):
            try:
                result = future.result()
            except Exception as exc:
                print(f"  worker failed: {exc}")
                continue
            status = "NATIVE" if result["native_intended"] else "miss"
            source = "source" if result["source_topology_intended"] else ""
            print(
                f"  [{result['sample_id']:>3}] {result['formula']:>14} "
                f"{status:>6} {source:>6} eval={result['native_evaluable']} "
                f"wall={result['wall_time_s']:.1f}s",
                flush=True,
            )
            results.append(result)

    results.sort(key=lambda row: row["sample_id"])
    pq.write_table(pa.Table.from_pylist(results), out_path)
    n = len(results)
    evaluable = [row for row in results if row["native_evaluable"]]
    native = sum(bool(row["native_intended"]) for row in results)
    source = sum(bool(row["source_topology_intended"]) for row in results)
    print()
    print(f"Native IRC validation: {args.method_tag} @ {args.noise_pm}pm")
    print(f"  candidates: {n}")
    print(f"  native-evaluable: {len(evaluable)} ({100 * len(evaluable) / max(n, 1):.1f}%)")
    print(f"  native intended: {native}/{n} ({100 * native / max(n, 1):.1f}%)")
    print(f"  source TOPO intended: {source}/{n} ({100 * source / max(n, 1):.1f}%)")
    print(f"  wall: {time.time() - start:.1f}s")
    print(f"  wrote: {out_path}")


if __name__ == "__main__":
    main()
