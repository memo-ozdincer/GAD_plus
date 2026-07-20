"""Lightweight Transition1x records without the torch_geometric wrapper."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class DirectT1xRecord:
    atomic_nums: np.ndarray
    transition_state: np.ndarray
    reactant: np.ndarray
    product: np.ndarray | None
    formula: str
    rxn: str


def load_t1x_records_direct(
    h5_path: str, split: str, sample_ids: Iterable[int],
) -> dict[int, DirectT1xRecord]:
    """Read requested final structures without materializing the full split."""
    import h5py

    wanted = set(sample_ids)
    if not wanted:
        return {}
    records: dict[int, DirectT1xRecord] = {}
    record_id = 0
    with h5py.File(h5_path, "r") as handle:
        for formula, reactions in handle[split].items():
            for rxn, molecule in reactions.items():
                try:
                    ts = molecule["transition_state"]
                    reactant = molecule["reactant"]
                    atomic_nums = np.asarray(ts["atomic_numbers"], dtype=np.int64)
                    if atomic_nums.size != len(reactant["atomic_numbers"]):
                        continue
                    product = molecule.get("product")
                    has_product = (
                        product is not None
                        and atomic_nums.size == len(product["atomic_numbers"])
                    )
                    if record_id in wanted:
                        records[record_id] = DirectT1xRecord(
                            atomic_nums=atomic_nums,
                            transition_state=np.asarray(ts["positions"][0], dtype=np.float64),
                            reactant=np.asarray(reactant["positions"][0], dtype=np.float64),
                            product=(
                                np.asarray(product["positions"][0], dtype=np.float64)
                                if has_product else None
                            ),
                            formula=str(formula),
                            rxn=str(rxn),
                        )
                    record_id += 1
                    if len(records) == len(wanted):
                        return records
                except (KeyError, IndexError, TypeError):
                    continue
    missing = sorted(wanted - records.keys())
    raise IndexError(f"Requested unavailable T1x sample IDs: {missing}")


def load_t1x_record(h5_path: str, split: str, sample_id: int) -> DirectT1xRecord:
    """Load one record in Transition1xDataset's filtered sample-id order."""
    from transition1x import Dataloader

    records_seen = 0
    for molecule in Dataloader(h5_path, datasplit=split, only_final=True):
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
            if records_seen == sample_id:
                return DirectT1xRecord(
                    atomic_nums=np.asarray(ts["atomic_numbers"], dtype=np.int64),
                    transition_state=np.asarray(ts["positions"], dtype=np.float64),
                    reactant=np.asarray(reactant["positions"], dtype=np.float64),
                    product=(
                        np.asarray(product["positions"], dtype=np.float64)
                        if has_product
                        else None
                    ),
                    formula=str(ts.get("formula", "")),
                    rxn=str(ts.get("rxn", "")),
                )
            records_seen += 1
        except Exception:
            continue
    raise IndexError(f"Requested sample {sample_id}, loaded {records_seen} valid T1x records")
