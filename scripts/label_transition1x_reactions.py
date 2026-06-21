#!/usr/bin/env python
"""Label non-training Transition1x reactions by reactant/product bond changes.

The labels are intentionally topology-first. Named reaction families are
best-effort automatic hints and should be manually curated before using them as
chemistry ground truth.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from ase import Atoms
from ase.data import chemical_symbols
from ase.neighborlist import natural_cutoffs, neighbor_list

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from gadplus.data.transition1x import Transition1xDataset  # noqa: E402
from gadplus.paths import scratch_dir, transition1x_h5_path  # noqa: E402


TRAIN_SPLITS = {"train", "training"}


def _as_bool(value) -> bool:
    if hasattr(value, "detach"):
        value = value.detach().cpu().reshape(-1)[0].item()
    return bool(value)


def _as_string(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.bytes_):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _split_keys(h5_path: Path) -> list[str]:
    import h5py

    with h5py.File(h5_path, "r") as h5:
        return [key for key in h5.keys() if key not in {"data"}]


def _non_train_splits(h5_path: Path) -> list[str]:
    return [split for split in _split_keys(h5_path) if split.lower() not in TRAIN_SPLITS]


def _atomic_numbers(sample) -> np.ndarray:
    return sample.z.detach().cpu().numpy().astype(int).reshape(-1)


def _coords(sample, attr: str) -> np.ndarray:
    return getattr(sample, attr).detach().cpu().numpy().reshape(-1, 3)


def _bond_edges(
    coords: np.ndarray,
    atomic_nums: np.ndarray,
    cutoff_scale: float,
) -> set[tuple[int, int]]:
    atoms = Atoms(numbers=atomic_nums.tolist(), positions=np.asarray(coords, dtype=float))
    cutoffs = natural_cutoffs(atoms, mult=cutoff_scale)
    i_idx, j_idx = neighbor_list("ij", atoms, cutoffs)
    return {
        (int(i), int(j)) if int(i) < int(j) else (int(j), int(i))
        for i, j in zip(i_idx.tolist(), j_idx.tolist(), strict=False)
        if int(i) != int(j)
    }


def _components(n_atoms: int, edges: Iterable[tuple[int, int]]) -> list[set[int]]:
    adjacency: list[set[int]] = [set() for _ in range(n_atoms)]
    for i, j in edges:
        adjacency[i].add(j)
        adjacency[j].add(i)

    seen: set[int] = set()
    comps: list[set[int]] = []
    for start in range(n_atoms):
        if start in seen:
            continue
        stack = [start]
        comp: set[int] = set()
        seen.add(start)
        while stack:
            node = stack.pop()
            comp.add(node)
            for nbr in adjacency[node]:
                if nbr not in seen:
                    seen.add(nbr)
                    stack.append(nbr)
        comps.append(comp)
    return comps


def _ring_count(n_atoms: int, edges: set[tuple[int, int]]) -> int:
    # Cyclomatic number for an undirected graph: E - V + C.
    return max(0, len(edges) - n_atoms + len(_components(n_atoms, edges)))


def _element_pair(edge: tuple[int, int], atomic_nums: np.ndarray) -> str:
    symbols = sorted((chemical_symbols[int(atomic_nums[edge[0]])], chemical_symbols[int(atomic_nums[edge[1]])]))
    return "-".join(symbols)


def _compact_pair_label(pair: str) -> str:
    return pair.replace("-", "")


def _bond_records(edges: Iterable[tuple[int, int]], atomic_nums: np.ndarray) -> list[dict[str, object]]:
    return [
        {"i": int(i), "j": int(j), "element_pair": _element_pair((i, j), atomic_nums)}
        for i, j in sorted(edges)
    ]


def _component_sets(components: list[set[int]]) -> set[frozenset[int]]:
    return {frozenset(comp) for comp in components}


def _topology_class(
    reactant_components: list[set[int]],
    product_components: list[set[int]],
    formed: set[tuple[int, int]],
    broken: set[tuple[int, int]],
    has_product: bool,
) -> str:
    if not has_product or (not formed and not broken):
        return "unknown"

    n_reactant = len(reactant_components)
    n_product = len(product_components)
    if n_reactant > n_product:
        return "association"
    if n_reactant < n_product:
        return "fragmentation_plus_rearrangement" if formed else "dissociation"
    if _component_sets(reactant_components) == _component_sets(product_components):
        return "intramolecular_rearrangement"
    return "exchange"


def _heavy_atom_edges(edges: set[tuple[int, int]], atomic_nums: np.ndarray) -> set[tuple[int, int]]:
    return {edge for edge in edges if atomic_nums[edge[0]] != 1 and atomic_nums[edge[1]] != 1}


def _hydrogen_transfer_family(
    formed: set[tuple[int, int]],
    broken: set[tuple[int, int]],
    reactant_edges: set[tuple[int, int]],
    product_edges: set[tuple[int, int]],
    atomic_nums: np.ndarray,
) -> str | None:
    if _heavy_atom_edges(reactant_edges, atomic_nums) != _heavy_atom_edges(product_edges, atomic_nums):
        return None

    formed_by_atom: dict[int, list[tuple[int, int]]] = defaultdict(list)
    broken_by_atom: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for edge in formed:
        for atom in edge:
            formed_by_atom[atom].append(edge)
    for edge in broken:
        for atom in edge:
            broken_by_atom[atom].append(edge)

    moving_hydrogens = [
        atom
        for atom, z in enumerate(atomic_nums.tolist())
        if z == 1 and len(formed_by_atom[atom]) == 1 and len(broken_by_atom[atom]) == 1
    ]
    if len(moving_hydrogens) != 1:
        return None

    h_atom = moving_hydrogens[0]
    formed_partner = next(atom for atom in formed_by_atom[h_atom][0] if atom != h_atom)
    broken_partner = next(atom for atom in broken_by_atom[h_atom][0] if atom != h_atom)
    hetero = {7, 8, 9, 15, 16, 17}
    if int(atomic_nums[formed_partner]) in hetero and int(atomic_nums[broken_partner]) in hetero:
        return "proton_transfer"
    return "H_shift"


def _reaction_family(
    topology_class: str,
    formed: set[tuple[int, int]],
    broken: set[tuple[int, int]],
    reactant_edges: set[tuple[int, int]],
    product_edges: set[tuple[int, int]],
    atomic_nums: np.ndarray,
    delta_ring_count: int,
) -> tuple[str, str]:
    h_family = _hydrogen_transfer_family(
        formed, broken, reactant_edges, product_edges, atomic_nums
    )
    if h_family:
        return h_family, "auto"

    n_formed = len(formed)
    n_broken = len(broken)
    formed_pairs = Counter(_element_pair(edge, atomic_nums) for edge in formed)
    if (
        topology_class == "association"
        and delta_ring_count > 0
        and n_formed >= 2
        and formed_pairs.get("C-C", 0) >= 2
    ):
        return "cycloaddition", "auto"
    if topology_class == "association" or n_formed > n_broken:
        return "addition", "auto"
    if topology_class == "dissociation" or n_broken > n_formed:
        return "elimination", "auto"
    if n_formed == 1 and n_broken == 1:
        return "substitution", "auto"
    if topology_class == "intramolecular_rearrangement" and n_formed == n_broken:
        return "isomerization", "auto"
    return "unknown", "auto"


def _reaction_center(
    formed: set[tuple[int, int]],
    broken: set[tuple[int, int]],
    atomic_nums: np.ndarray,
) -> tuple[list[int], list[str]]:
    atoms = sorted({atom for edge in formed | broken for atom in edge})
    elements = sorted({chemical_symbols[int(atomic_nums[atom])] for atom in atoms})
    return atoms, elements


def label_sample(
    sample,
    sample_id: int,
    split: str,
    cutoff_scale: float,
) -> dict[str, object]:
    atomic_nums = _atomic_numbers(sample)
    n_atoms = int(atomic_nums.size)
    has_product = _as_bool(sample.has_product)

    reactant_edges = _bond_edges(_coords(sample, "pos_reactant"), atomic_nums, cutoff_scale)
    if has_product:
        product_edges = _bond_edges(_coords(sample, "pos_product"), atomic_nums, cutoff_scale)
    else:
        product_edges = set()

    formed = product_edges - reactant_edges if has_product else set()
    broken = reactant_edges - product_edges if has_product else set()
    reactant_components = _components(n_atoms, reactant_edges)
    product_components = _components(n_atoms, product_edges) if has_product else []
    topology = _topology_class(reactant_components, product_components, formed, broken, has_product)

    reactant_ring_count = _ring_count(n_atoms, reactant_edges)
    product_ring_count = _ring_count(n_atoms, product_edges) if has_product else 0
    delta_ring_count = product_ring_count - reactant_ring_count
    reaction_center_atoms, reaction_center_elements = _reaction_center(formed, broken, atomic_nums)
    reaction_family, reaction_family_confidence = _reaction_family(
        topology,
        formed,
        broken,
        reactant_edges,
        product_edges,
        atomic_nums,
        delta_ring_count,
    )

    formed_counts = Counter(_compact_pair_label(_element_pair(edge, atomic_nums)) for edge in formed)
    broken_counts = Counter(_compact_pair_label(_element_pair(edge, atomic_nums)) for edge in broken)

    row: dict[str, object] = {
        "split": split,
        "sample_id": int(sample_id),
        "formula": _as_string(sample.formula),
        "rxn": _as_string(sample.rxn),
        "n_atoms": n_atoms,
        "has_product": has_product,
        "n_components_reactant": len(reactant_components),
        "n_components_product": len(product_components) if has_product else None,
        "topology_class": topology,
        "n_bonds_formed": len(formed),
        "n_bonds_broken": len(broken),
        "formed_bonds": json.dumps(_bond_records(formed, atomic_nums), sort_keys=True),
        "broken_bonds": json.dumps(_bond_records(broken, atomic_nums), sort_keys=True),
        "reaction_center_atoms": json.dumps(reaction_center_atoms),
        "n_reaction_center_atoms": len(reaction_center_atoms),
        "reaction_center_elements": json.dumps(reaction_center_elements),
        "reactant_ring_count": reactant_ring_count,
        "product_ring_count": product_ring_count if has_product else None,
        "ring_formed": delta_ring_count > 0,
        "ring_broken": delta_ring_count < 0,
        "delta_ring_count": delta_ring_count if has_product else None,
        "reaction_family": reaction_family,
        "reaction_family_confidence": reaction_family_confidence,
    }
    for pair, count in formed_counts.items():
        row[f"formed_{pair}"] = int(count)
    for pair, count in broken_counts.items():
        row[f"broken_{pair}"] = int(count)
    return row


def write_outputs(rows: list[dict[str, object]], output_dir: Path, stem: str) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    count_cols = [
        col
        for col in df.columns
        if (col.startswith("formed_") or col.startswith("broken_"))
        and col not in {"formed_bonds", "broken_bonds"}
    ]
    for col in count_cols:
        df[col] = df[col].fillna(0).astype(int)

    parquet_path = output_dir / f"{stem}.parquet"
    csv_path = output_dir / f"{stem}.csv"
    df.to_parquet(parquet_path, index=False)
    df.to_csv(csv_path, index=False)
    return parquet_path, csv_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", type=Path, default=transition1x_h5_path())
    parser.add_argument(
        "--splits",
        nargs="*",
        default=None,
        help="Splits to scan. Defaults to every top-level split except train/training.",
    )
    parser.add_argument("--max-samples-per-split", type=int, default=None)
    parser.add_argument("--cutoff-scale", type=float, default=1.2)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=scratch_dir() / "runs" / "reaction_labels",
    )
    parser.add_argument("--stem", default="non_train_reaction_labels")
    args = parser.parse_args()

    splits = args.splits or _non_train_splits(args.h5)
    splits = [split for split in splits if split.lower() not in TRAIN_SPLITS]
    if not splits:
        raise SystemExit("No non-training splits selected.")

    rows: list[dict[str, object]] = []
    for split in splits:
        print(f"Loading split={split}", flush=True)
        dataset = Transition1xDataset(
            str(args.h5),
            split=split,
            max_samples=args.max_samples_per_split,
        )
        print(f"  loaded {len(dataset)} valid samples", flush=True)
        for sample_id, sample in enumerate(dataset):
            rows.append(
                label_sample(
                    sample=sample,
                    sample_id=sample_id,
                    split=split,
                    cutoff_scale=args.cutoff_scale,
                )
            )

    parquet_path, csv_path = write_outputs(rows, args.output_dir, args.stem)
    print(f"Wrote {len(rows)} rows", flush=True)
    print(f"Parquet: {parquet_path}", flush=True)
    print(f"CSV: {csv_path}", flush=True)
    if rows:
        summary = pd.DataFrame(rows).groupby(["split", "topology_class"]).size()
        print(summary.to_string(), flush=True)


if __name__ == "__main__":
    main()
