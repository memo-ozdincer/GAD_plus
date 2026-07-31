"""Identify T1x test samples where xTB GFN1's saddle is closest to HIP's
(measured by fmax at the HIP-TS geometry). The lower the fmax at HIP-TS,
the closer xTB's local saddle to HIP's, so the more meaningful a Sella/GAD
search on xTB starting from noised HIP-TS will be.

Output: a TXT file with sorted (sample_idx, fmax) lines, and the top-K
favorable indices that the main-panel scripts will use.
"""
from __future__ import annotations

import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import torch


sys.path.insert(0, "src")


def _evaluate_one(args):
    idx, h5_path = args
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    torch.set_num_threads(1)
    from gadplus.data.transition1x import Transition1xDataset, UsePos
    from gadplus.calculator.xtb import load_xtb_calculator, make_xtb_predict_fn

    ds = Transition1xDataset(
        h5_path, split="test", max_samples=idx + 1,
        transform=UsePos("pos_transition"),
    )
    s = ds[idx]
    coords = s.pos.to(torch.float64)
    z = s.z

    predict_fn = make_xtb_predict_fn(load_xtb_calculator(method="gfn1"))
    out = predict_fn(coords, z, do_hessian=False, require_grad=False)
    f = out["forces"]
    return idx, str(getattr(s, "formula", f"sample_{idx}")), float(f.abs().max().item())


def main():
    h5 = "/lustre06/project/6033559/memoozd/data/transition1x.h5"
    n_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", "8"))
    n_samples = 287

    tasks = [(i, h5) for i in range(n_samples)]
    rows = []
    print(f"Evaluating xTB-GFN1 fmax at HIP-TS for {n_samples} samples ({n_workers} workers)")
    with ProcessPoolExecutor(max_workers=n_workers) as exe:
        for fut in as_completed([exe.submit(_evaluate_one, t) for t in tasks]):
            idx, formula, fmax = fut.result()
            rows.append((idx, formula, fmax))
            if len(rows) % 25 == 0:
                print(f"  done: {len(rows)}/{n_samples}")

    rows.sort(key=lambda r: r[2])
    out_path = "analysis_2026_04_29/xtb_gfn1_fmax_at_hipts.csv"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write("sample_idx,formula,fmax_eV_per_A\n")
        for idx, formula, fmax in rows:
            f.write(f"{idx},{formula},{fmax:.6f}\n")
    print(f"Wrote {out_path}")

    K = 30
    top = rows[:K]
    print(f"\nTop-{K} xTB-favorable samples (lowest fmax at HIP-TS):")
    for idx, formula, fmax in top:
        print(f"  idx={idx:>3}  {formula:>14}  fmax={fmax:.3f}")
    idx_list_path = "analysis_2026_04_29/xtb_favorable_top30.txt"
    with open(idx_list_path, "w") as f:
        f.write(",".join(str(t[0]) for t in top) + "\n")
    print(f"\nTop-{K} indices saved to {idx_list_path}")


if __name__ == "__main__":
    main()
