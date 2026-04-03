# GAD_plus

Clean, publishable GAD-based transition state search with HIP neural network potential. Bottom-up design: pure GAD is the base, each feature is added and benchmarked independently. Narval-first.

## Project structure

```
src/gadplus/
  core/              # Pure algorithms, zero I/O
  projection/        # Eckart/mass-weighting, reduced-basis Hessian
  calculator/        # HIP predict_fn, ASE adapter for Sella IRC
  geometry/          # Kabsch+Hungarian alignment, noise, interpolation
  search/            # GAD loop, NR+GAD flip-flop, IRC validation
  logging/           # Parquet trajectories, MLflow offline, failure autopsy
  data/              # Transition1x dataset loader
  orchestration/     # Hydra entry point
configs/             # Hydra: search/, starting/, calculator/, cluster/
scripts/             # SLURM scripts, DuckDB analysis, env setup
```

## Key concepts

- **predict_fn interface**: All algorithms use `predict_fn(coords, atomic_nums, do_hessian, require_grad) -> dict`.
- **HIP only**: GPU ML potential (Equiformer). No SCINE.
- **Eckart projection**: `vib_eig()` returns full-rank (3N-k) vibrational Hessian. No threshold filtering.
- **TS convergence**: **n_neg == 1 AND force_norm < 0.01 eV/A**. Non-negotiable.

## Cluster setup

### Narval (primary) — A100 MIG slicing

Narval's MIG lets you slice a single A100 into isolated mini-GPUs. HIP inference on small molecules uses <2GB VRAM, so a `2g.10gb` slice (10GB, 2/8 compute) is plenty. This packs ~14 independent jobs per physical A100.

```
Account:   def-aspuru
GPU:       a100_2g.10gb:1 (MIG slice)
CPU:       2 cores per job
RAM:       8GB per job
Project:   /lustre06/project/6033559/memoozd
Scratch:   /lustre07/scratch/memoozd
```

### Trillium (secondary) — H100 full GPUs

Override with `cluster=trillium` for large molecules or training.

```
Account:   rrg-aspuru
GPU:       1× H100-SXM (full)
CPU:       12 cores per job
RAM:       64GB per job
Project:   /project/rrg-aspuru/memoozd
Scratch:   /scratch/memoozd
```

## Running experiments

```bash
# Setup (once, on Narval)
bash scripts/setup_env.sh

# Activate
source .venv/bin/activate

# Single run on a reserved node
salloc --account=def-aspuru --gpus=a100_2g.10gb:1 --cpus-per-task=2 --mem=8G --time=6:00:00
srun python -m gadplus.orchestration.run search=gad_projected max_samples=50

# Submit as SLURM job
sbatch scripts/run_narval.slurm

# Sweep: all methods × all noise levels (launches hundreds of MIG jobs)
bash scripts/run_narval_sweep.sh

# Overnight reserved-node workflow
salloc --account=def-aspuru --gpus=a100:1 --cpus-per-task=12 --mem=64G --time=12:00:00
bash scripts/run_narval_reserved.sh

# Analysis
python scripts/analyze.py /lustre07/scratch/memoozd/gadplus/runs/

# On Trillium instead
python -m gadplus.orchestration.run cluster=trillium search=gad_projected max_samples=50
```

## Bottom-up feature levels

| Level | Config | What's added |
|-------|--------|-------------|
| 0 | `pure_gad` | Raw Hessian, fixed dt, Euler steps |
| 1 | `gad_tracked` | + Mode tracking (k=8) |
| 2 | `gad_projected` | + Eckart projection |
| 3 | `gad_adaptive_dt` | + Eigenvalue-clamped dt |
| 4 | `nr_gad_flipflop` | + NR refinement at saddle |

## Key paths (Narval)

- HIP checkpoint: `/lustre06/project/6033559/memoozd/models/hip_v2.ckpt`
- Transition1x: `/lustre06/project/6033559/memoozd/data/transition1x.h5`
- Output: `/lustre07/scratch/memoozd/gadplus/runs/`
- MLflow: `file:///lustre07/scratch/memoozd/gadplus/mlruns`

## Performance notes

- Threading is pinned: OMP=1, torch=2 (avoids contention on MIG shared nodes)
- Dataset is loaded once into memory; samples processed sequentially on GPU
- Each sample's trajectory is flushed to its own Parquet file (crash-safe)
- DuckDB analysis works on partial results (glob over whatever Parquet files exist)
- No internet on compute nodes: MLflow uses `file://` URI, no wandb

## Don't

- Don't use anything other than n_neg==1 + force<0.01 as TS convergence
- Don't skip Eckart projection when computing vibrational eigenvalues
- Don't add features without independent benchmarking justification
- Don't use path-based state (trajectory history) in optimizers
- Don't import HIP in core/ or projection/ — use the predict_fn interface
- Don't add Co-Authored-By lines to git commits
