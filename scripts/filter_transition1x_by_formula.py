#!/usr/bin/env python
"""Create a Transition1x-shaped HDF5 containing only selected formulas."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import h5py

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from gadplus.paths import transition1x_h5_path


DEFAULT_SPLITS = ("data", "train", "val", "test")


def _formula_slug(formula: str) -> str:
    return "".join(ch.lower() for ch in formula if ch.isalnum())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=transition1x_h5_path(),
        help="Source Transition1x HDF5 path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Destination HDF5 path. Defaults to data/transition1x_<formula>.h5.",
    )
    parser.add_argument(
        "--formula",
        default="C3H8O",
        help="Formula group to copy from each Transition1x split.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(DEFAULT_SPLITS),
        help="Top-level HDF5 splits to preserve.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the output file if it already exists.",
    )
    return parser.parse_args()


def copy_formula_dataset(
    input_path: Path,
    output_path: Path,
    formula: str,
    splits: list[str],
    overwrite: bool = False,
) -> dict[str, int]:
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"{output_path} already exists; pass --overwrite to replace it.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    part_path = output_path.with_suffix(output_path.suffix + ".part")
    if part_path.exists():
        part_path.unlink()

    counts: dict[str, int] = {}
    try:
        with h5py.File(input_path, "r") as src, h5py.File(part_path, "w") as dst:
            for key, value in src.attrs.items():
                dst.attrs[key] = value

            for split in splits:
                dst_split = dst.create_group(split)
                counts[split] = 0
                if split not in src:
                    print(f"[WARN] Source is missing split {split!r}; created empty group.")
                    continue

                src_split = src[split]
                for key, value in src_split.attrs.items():
                    dst_split.attrs[key] = value

                if formula not in src_split:
                    continue

                src.copy(src_split[formula], dst_split, name=formula)
                counts[split] = len(src_split[formula])

        os.replace(part_path, output_path)
    except Exception:
        if part_path.exists():
            part_path.unlink()
        raise

    return counts


def main() -> None:
    args = parse_args()
    output = args.output or Path("data") / f"transition1x_{_formula_slug(args.formula)}.h5"

    counts = copy_formula_dataset(
        input_path=args.input.expanduser(),
        output_path=output.expanduser(),
        formula=args.formula,
        splits=args.splits,
        overwrite=args.overwrite,
    )
    with h5py.File(output, "r") as dst:
        unique_rxns = sorted(
            {
                rxn
                for split in counts
                if counts[split] > 0
                for rxn in dst[split][args.formula].keys()
            }
        )

    print(f"Saved {output}")
    for split in args.splits:
        print(f"{split}: {counts.get(split, 0)} reactions")
    print(f"unique reactions: {len(unique_rxns)}")


if __name__ == "__main__":
    main()
