"""Transition1x HDF5 dataset loader for transition state geometries.

Provides a PyTorch Dataset wrapper around the Transition1x database,
loading reactant, product, and transition state geometries along with
energies and forces.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from transition1x import Dataloader as T1xDataloader


class Transition1xDataset(Dataset):
    """Loads transition state data from the Transition1x HDF5 file.

    Each sample is a ``torch_geometric.data.Data`` object with attributes:
        z             – atomic numbers (N,)
        pos_transition – TS geometry (N, 3)
        pos_reactant  – reactant geometry (N, 3)
        pos_product   – product geometry (N, 3), zeros if unavailable
        energy        – TS energy, scalar
        forces        – TS forces (N, 3)
        has_product   – bool tensor (1,)
        rxn           – reaction SMILES string
        formula       – molecular formula string

    Args:
        h5_path:     Path to the Transition1x HDF5 file.
        split:       Dataset split, e.g. ``"test"``, ``"train"``, ``"validation"``.
        max_samples: If set, stop loading after this many valid samples.
        transform:   Optional callable applied to each sample on ``__getitem__``.
    """

    def __init__(
        self,
        h5_path: str,
        split: str = "test",
        max_samples: Optional[int] = None,
        transform=None,
    ):
        self.transform = transform
        self.samples: list[Data] = []
        loader = T1xDataloader(h5_path, datasplit=split, only_final=True)

        for idx, mol in enumerate(loader):
            if max_samples is not None and len(self.samples) >= max_samples:
                break
            try:
                ts = mol["transition_state"]
                reactant = mol["reactant"]

                # Skip if atom counts differ between reactant and TS.
                if len(ts["atomic_numbers"]) != len(reactant["atomic_numbers"]):
                    continue

                # Product is optional; zero-fill if missing or mismatched.
                product = mol.get("product")
                has_product = (
                    product is not None
                    and len(product.get("atomic_numbers", [])) == len(ts["atomic_numbers"])
                )
                if has_product:
                    pos_product = torch.tensor(product["positions"], dtype=torch.float)
                else:
                    pos_product = torch.zeros_like(
                        torch.tensor(ts["positions"], dtype=torch.float)
                    )

                data = Data(
                    z=torch.tensor(ts["atomic_numbers"], dtype=torch.long),
                    pos_transition=torch.tensor(ts["positions"], dtype=torch.float),
                    pos_reactant=torch.tensor(reactant["positions"], dtype=torch.float),
                    pos_product=pos_product,
                    has_product=torch.tensor([has_product], dtype=torch.bool),
                    energy=torch.tensor(ts["wB97x_6-31G(d).energy"], dtype=torch.float),
                    forces=torch.tensor(ts["wB97x_6-31G(d).forces"], dtype=torch.float),
                    rxn=ts["rxn"],
                    formula=ts["formula"],
                )
                self.samples.append(data)
            except Exception as e:
                print(f"[WARN] Skipping idx={idx}: {e}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Data:
        data = self.samples[idx]
        if self.transform is not None:
            data = self.transform(data)
        return data


class UsePos:
    """Transform that copies a named position attribute to ``data.pos``.

    Useful for setting which geometry (TS, reactant, product) is used as
    the primary ``pos`` field expected by downstream code.

    Args:
        attr: Name of the attribute to copy, e.g. ``"pos_transition"``.
    """

    def __init__(self, attr: str = "pos_transition"):
        self.attr = attr

    def __call__(self, data: Data) -> Data:
        pos = getattr(data, self.attr, None)
        if pos is None:
            raise ValueError(f"Data missing '{self.attr}'. Keys: {list(data.keys())}")
        data.pos = pos
        return data
