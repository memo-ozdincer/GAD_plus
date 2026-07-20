#!/usr/bin/env python
"""Evaluate HIP at the T1x-labelled TS, zero optimisation steps.

For every sample in the chosen split, this loads HIP, evaluates
energy/forces/Hessian at ``pos_transition``, computes the Eckart-projected
vibrational n_neg (with the canonical ``< -1e-4`` threshold) and fmax
(``.abs().max()``), and tallies the fraction of labels that already satisfy
the TS convergence criterion **without any optimisation**.

This is the "true-overlap" rate: how often does HIP agree with the T1x label
that a geometry is an index-1 saddle, out of the box?

Output: ``summary_eval_labelled_ts_{split}.parquet`` and a short text tally.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--n-samples", type=int, default=287)
    parser.add_argument("--force-threshold", type=float, default=0.01)
    parser.add_argument("--neg-eig-threshold", type=float, default=1e-4,
                        help="eigenvalues counted as negative when < -threshold")
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | split={args.split} | n={args.n_samples} | "
          f"thr_fmax={args.force_threshold} | thr_eig=-{args.neg_eig_threshold}")

    # ── Paths ────────────────────────────────────────────────────────────
    for ckpt in ("/lustre06/project/6033559/memoozd/models/hip_v2.ckpt",
                 "/project/rrg-aspuru/memoozd/models/hip_v2.ckpt"):
        if os.path.exists(ckpt):
            break
    else:
        sys.exit("hip_v2.ckpt not found")
    for h5 in ("/lustre06/project/6033559/memoozd/data/transition1x.h5",
               "/project/rrg-aspuru/memoozd/data/transition1x.h5"):
        if os.path.exists(h5):
            break
    else:
        sys.exit("transition1x.h5 not found")

    os.makedirs(args.output_dir, exist_ok=True)

    # ── HIP + dataset ────────────────────────────────────────────────────
    from gadplus.calculator.hip import load_hip_calculator, make_hip_predict_fn
    calc = load_hip_calculator(ckpt, device=device)
    predict_fn = make_hip_predict_fn(calc)
    print("HIP loaded")

    from gadplus.data.transition1x import Transition1xDataset, UsePos
    dataset = Transition1xDataset(
        h5_path=h5, split=args.split, max_samples=args.n_samples,
        transform=UsePos("pos_transition"),
    )
    print(f"Loaded {len(dataset)} samples (split={args.split})")

    # ── Eval loop ────────────────────────────────────────────────────────
    from gadplus.projection import vib_eig, atomic_nums_to_symbols
    from gadplus.core.convergence import force_max, force_mean

    rows = []
    t0 = time.time()
    for i in range(len(dataset)):
        s = dataset[i]
        coords = s.pos.to(device).to(torch.float32).reshape(-1, 3)
        z = s.z.to(device)
        formula = getattr(s, "formula", f"sample_{i}")

        out = predict_fn(coords, z, do_hessian=True, require_grad=False)
        forces = out["forces"]
        if forces.dim() == 3 and forces.shape[0] == 1:
            forces = forces[0]
        forces = forces.reshape(-1, 3)
        hessian = out["hessian"]
        energy = float(out["energy"].detach().reshape(-1)[0].item())

        atomsymbols = atomic_nums_to_symbols(z)
        evals_vib, _, _ = vib_eig(hessian, coords, atomsymbols, purify=False)

        n_neg = int((evals_vib < -args.neg_eig_threshold).sum().item())
        eig0 = float(evals_vib[0].item()) if evals_vib.numel() else 0.0
        eig1 = float(evals_vib[1].item()) if evals_vib.numel() > 1 else 0.0
        fn = force_mean(forces)
        fm = force_max(forces)
        overlaps = (n_neg == 1) and (fm < args.force_threshold)

        rows.append({
            "sample_id": i,
            "formula": formula,
            "energy": energy,
            "force_norm": fn,
            "force_max": fm,
            "n_neg": n_neg,
            "eig0": eig0,
            "eig1": eig1,
            "overlaps_ts": overlaps,
        })
        mark = "OVERLAPS" if overlaps else "no-overlap"
        if i < 10 or i % 20 == 0:
            print(f"  [{i:3d}] {formula:>12s} | {mark:10s} | n_neg={n_neg} | "
                  f"fmax={fm:.4f} | fn={fn:.4f} | eig0={eig0:.4f}")

    wall = time.time() - t0
    df = pd.DataFrame(rows)
    out_path = os.path.join(args.output_dir, f"summary_eval_labelled_ts_{args.split}.parquet")
    df.to_parquet(out_path)

    n_total = len(df)
    n_overlap = int(df["overlaps_ts"].sum())
    n_nneg1 = int((df["n_neg"] == 1).sum())
    n_fmax_ok = int((df["force_max"] < args.force_threshold).sum())

    print()
    print("=" * 70)
    print(f"Labelled-TS overlap eval | split={args.split} | n={n_total} | "
          f"wall={wall:.1f}s")
    print(f"  n_neg == 1 AND fmax < {args.force_threshold:g}  : "
          f"{n_overlap}/{n_total} = {100*n_overlap/n_total:.1f}%   <-- TRUE-OVERLAP")
    print(f"  n_neg == 1 (force ignored)                       : "
          f"{n_nneg1}/{n_total} = {100*n_nneg1/n_total:.1f}%")
    print(f"  fmax < {args.force_threshold:g} (n_neg ignored)            : "
          f"{n_fmax_ok}/{n_total} = {100*n_fmax_ok/n_total:.1f}%")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
