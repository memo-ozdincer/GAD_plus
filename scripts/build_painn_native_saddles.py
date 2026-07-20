#!/usr/bin/env python
"""Construct a small calculator-native saddle set from T1x endpoints.

This is a reference-set builder, not a GAD or Sella benchmark.  It therefore
uses neither optimizer to decide whether a structure enters the set.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from ase import Atoms
from ase.mep import NEB
from ase.mep.neb import NEBOptimizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def _as_tensor(coords: np.ndarray, device: str) -> torch.Tensor:
    return torch.as_tensor(coords, dtype=torch.float32, device=device)


def _write_summary(rows: list[dict], output_dir: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    pq.write_table(pa.Table.from_pylist(rows), output_dir / "summary.parquet")
    with (output_dir / "summary.json").open("w") as handle:
        json.dump(rows, handle, indent=2, sort_keys=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("neuralneb", "horm"), default="neuralneb")
    parser.add_argument("--n-samples", type=int, default=3)
    parser.add_argument(
        "--start-sample",
        type=int,
        default=0,
        help="first dataset sample to examine; enables disjoint construction shards",
    )
    parser.add_argument(
        "--max-examined",
        type=int,
        default=None,
        help="optional cap on endpoint pairs examined; use for resource-bounded smokes",
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--images", type=int, default=7)
    parser.add_argument("--endpoint-fmax", type=float, default=0.02)
    parser.add_argument("--neb-fmax", type=float, default=0.05)
    parser.add_argument("--endpoint-max-steps", type=int, default=250)
    parser.add_argument("--neb-max-steps", type=int, default=150)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--h5-path", default="/lustre06/project/6033559/memoozd/data/transition1x.h5")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    if args.images < 3:
        raise ValueError("--images must include at least two endpoints and one interior image")

    from gadplus.calculator.ase_adapter import HipASECalculator
    from gadplus.data.transition1x import Transition1xDataset, UsePos
    from gadplus.projection import atomic_nums_to_symbols, vib_eig
    from gadplus.search.native_endpoints import load_or_relax_native_endpoints

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.backend == "neuralneb":
        from gadplus.calculator.neuralneb import (
            NEURALNEB_MODELS_DIR,
            NeuralNebPaiNNCalculator,
            make_neuralneb_predict_fn,
        )

        checkpoint = Path(args.checkpoint) if args.checkpoint else NEURALNEB_MODELS_DIR / "painn0.sd"
        calculator = NeuralNebPaiNNCalculator(checkpoint=checkpoint, device=args.device)
        predict_fn = make_neuralneb_predict_fn(calculator)
        functional = "neuralneb-painn0"
    else:
        from gadplus.calculator.horm import (
            HORM_LEFTNET_CHECKPOINT,
            HormLeftNetCalculator,
            make_horm_predict_fn,
        )

        checkpoint = Path(args.checkpoint) if args.checkpoint else Path(HORM_LEFTNET_CHECKPOINT)
        calculator = HormLeftNetCalculator(checkpoint=str(checkpoint), device=args.device)
        predict_fn = make_horm_predict_fn(calculator)
        functional = f"horm-{checkpoint.stem}"
    dataset = Transition1xDataset(args.h5_path, split=args.split, transform=UsePos("pos_transition"))
    print(f"loaded={len(dataset)} split={args.split} device={args.device} checkpoint={checkpoint}")

    rows: list[dict] = []
    accepted = 0
    examined = 0
    for sample_id, sample in enumerate(dataset):
        if sample_id < args.start_sample:
            continue
        if accepted >= args.n_samples:
            break
        if args.max_examined is not None and examined >= args.max_examined:
            break
        examined += 1
        row: dict = {"sample_id": sample_id, "formula": str(sample.formula), "accepted": False}
        if not bool(sample.has_product.item()) or not bool(sample.pos_product.abs().sum() > 0):
            row["reason"] = "missing_product"
            rows.append(row)
            continue

        z_cpu = sample.z.cpu()
        labels = load_or_relax_native_endpoints(
            cache_dir=output_dir / "endpoint_cache",
            sample_id=sample_id,
            functional=functional,
            atomic_nums=z_cpu,
            reactant_coords=sample.pos_reactant,
            product_coords=sample.pos_product,
            predict_fn=predict_fn,
            relax_fmax=args.endpoint_fmax,
            max_steps=args.endpoint_max_steps,
        )
        reactant, product = labels.reactant, labels.product
        row.update(
            endpoint_cache_hit=labels.cache_hit,
            reactant_converged=bool(reactant and reactant.converged),
            product_converged=bool(product and product.converged),
            reactant_fmax=float(reactant.force_max if reactant else np.inf),
            product_fmax=float(product.force_max if product else np.inf),
        )
        if not reactant or not product or not reactant.converged or not product.converged:
            row["reason"] = "native_endpoint_relaxation_failed"
            rows.append(row)
            continue

        numbers = z_cpu.numpy().astype(int)
        images = [Atoms(numbers=numbers, positions=reactant.coords)]
        images.extend(Atoms(numbers=numbers, positions=reactant.coords) for _ in range(args.images - 2))
        images.append(Atoms(numbers=numbers, positions=product.coords))
        z_device = z_cpu.to(args.device)
        for image in images:
            image.calc = HipASECalculator(predict_fn=predict_fn, atomic_nums=z_device)

        try:
            neb = NEB(images, climb=True, method="improvedtangent")
            neb.interpolate(method="idpp")
            optimizer = NEBOptimizer(neb, logfile=None)
            neb_converged = bool(optimizer.run(fmax=args.neb_fmax, steps=args.neb_max_steps))
            energies = np.asarray([image.get_potential_energy() for image in images])
            max_image_index = int(np.argmax(energies[1:-1])) + 1
            candidate = images[max_image_index]
            candidate_coords = _as_tensor(candidate.positions, args.device)
            out = predict_fn(candidate_coords, z_device, do_hessian=True, require_grad=False)
            force_max = float(out["forces"].abs().max().item())
            hessian = out["hessian"].reshape(3 * len(candidate), 3 * len(candidate))
            eigvals, _evecs, _basis = vib_eig(
                hessian, candidate_coords, atomic_nums_to_symbols(z_device)
            )
            n_neg = int((eigvals < -1.0e-4).sum().item())
            row.update(
                neb_converged=neb_converged,
                neb_steps=int(optimizer.nsteps),
                max_image_index=max_image_index,
                energy_eV=float(out["energy"].item()),
                force_max_eV_per_A=force_max,
                n_neg=n_neg,
                eig0_eV_per_A2=float(eigvals[0].item()),
            )
            accepted_here = neb_converged and force_max <= args.neb_fmax and n_neg == 1
            row["accepted"] = accepted_here
            row["reason"] = "accepted" if accepted_here else "not_stationary_index_one"
            if accepted_here:
                np.savez_compressed(
                    output_dir / f"candidate_{accepted:03d}_sample_{sample_id:04d}.npz",
                    sample_id=np.asarray(sample_id, dtype=np.int64),
                    atomic_numbers=numbers,
                    coords=candidate.positions,
                    reactant=reactant.coords,
                    product=product.coords,
                    hessian=hessian.detach().cpu().numpy(),
                    eigvals=eigvals.detach().cpu().numpy(),
                )
                accepted += 1
        except Exception as exc:
            row["reason"] = "neb_or_hessian_error"
            row["error"] = repr(exc)
        rows.append(row)
        _write_summary(rows, output_dir)
        print(json.dumps(row, sort_keys=True))

    _write_summary(rows, output_dir)
    print(f"accepted={accepted}/{args.n_samples} examined={examined}")


if __name__ == "__main__":
    main()
