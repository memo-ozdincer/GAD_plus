#!/bin/bash
#SBATCH --account=rrg-aspuru
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=/lustre07/scratch/memoozd/gadplus/logs/probe_xtb_gfn1_%j.out
#SBATCH --error=/lustre07/scratch/memoozd/gadplus/logs/probe_xtb_gfn1_%j.err
#
# Probe: do xTB GFN1 forces at the HIP TS look like HIP/DFTB0, or like GFN2?
# If forces are ~1-3 eV/A at zero noise: GFN1 is a usable surrogate, proceed
# with main runs. If forces are ~10 eV/A like GFN2: xTB is fundamentally
# different from the HIP-trained T1x TS set; drop from main results.

set -euo pipefail
module purge
module load StdEnv/2023 python/3.11
PROJECT_DIR="/lustre06/project/6033559/memoozd/GAD_plus"
cd "$PROJECT_DIR"
source .venv/bin/activate
export PYTHONPATH="$PROJECT_DIR/src:$PYTHONPATH"
export PYTHONUNBUFFERED=1 WANDB_DISABLED=true MPLBACKEND=Agg
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

python -u - <<'PY'
import sys, torch
sys.path.insert(0, "src")
from gadplus.data.transition1x import Transition1xDataset, UsePos

ds = Transition1xDataset(
    "/lustre06/project/6033559/memoozd/data/transition1x.h5",
    split="test", max_samples=10,
    transform=UsePos("pos_transition"),
)

print(f"=== Forces & Hessian at HIP-TS, no noise, first {len(ds)} test samples ===")
print(f"{'idx':>3} {'formula':>12} {'method':>12} {'fmax':>10} {'|F|':>10}  eig range")

from gadplus.calculator.scine import load_scine_calculator, make_scine_predict_fn
from gadplus.calculator.xtb import load_xtb_calculator, make_xtb_predict_fn

scine = make_scine_predict_fn(load_scine_calculator("DFTB0"))
xtb1  = make_xtb_predict_fn(load_xtb_calculator(method="gfn1"))
xtb2  = make_xtb_predict_fn(load_xtb_calculator(method="gfn2"))

for i in range(len(ds)):
    s = ds[i]
    coords = s.pos.to(torch.float64); z = s.z
    formula = getattr(s, "formula", f"sample_{i}")
    for name, pred in [("DFTB0", scine), ("xTB-GFN1", xtb1), ("xTB-GFN2", xtb2)]:
        out = pred(coords, z, do_hessian=True, require_grad=False)
        f = out["forces"]; h = out["hessian"]
        h = 0.5 * (h + h.transpose(-1, -2))
        eigs = torch.linalg.eigvalsh(h)
        print(f"{i:>3} {formula:>12} {name:>12} {f.abs().max().item():>10.3f} {torch.linalg.norm(f).item():>10.3f}  [{eigs.min().item():>7.2f}, {eigs.max().item():>7.2f}]")
PY
