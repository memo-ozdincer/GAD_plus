# GAD_plus

Clean, publishable GAD-based transition state search with HIP neural network potential. Bottom-up design: pure GAD is the base, each feature is added and benchmarked independently. Narval-first.

## Project structure

```
src/gadplus/
  core/                    # Pure algorithms, zero I/O
    gad.py                   # GAD vector computation, Euler step
    mode_tracking.py         # Eigenvector continuity across steps
    newton_raphson.py        # Spectral-partitioned NR for TS refinement
    convergence.py           # n_neg==1 AND force<0.01, cascade analysis
    adaptive_dt.py           # Eigenvalue-clamped timestep, displacement cap
    types.py                 # PredictFn protocol
  projection/              # Single-file differentiable Eckart projection
    projection.py            # Mass-weighting, Eckart generators, reduced-basis Hessian,
                             # vib_eig, gad_dynamics_projected — all pure torch, autograd-safe
  calculator/              # HIP interface
    hip.py                   # make_hip_predict_fn, load_hip_calculator, coords_to_pyg_batch
    ase_adapter.py           # ASE Calculator wrapper for Sella IRC
  geometry/                # Molecular geometry utilities
    alignment.py             # Kabsch + Hungarian (RMSD with atom permutation symmetry)
    noise.py                 # Gaussian noise injection
    interpolation.py         # Linear + geodesic interpolation
    starting.py              # StartingGeometry factory
  search/                  # Search loops (all state-based, no path history)
    gad_search.py            # Main GAD loop (levels 0-3)
    nr_gad_flipflop.py       # NR+GAD alternation (level 4)
    irc_validate.py          # Sella IRC validation
  logging/                 # Trajectory logging + failure analysis
    trajectory.py            # TrajectoryLogger -> Parquet (40-field per-step schema)
    mlflow_logger.py         # MLflow offline (file://) wrapper
    autopsy.py               # 6-class failure classification
    schema.py                # PyArrow schema definitions
  data/                    # Dataset loading
    transition1x.py          # Transition1xDataset, UsePos
  orchestration/           # Hydra entry point
    run.py                   # @hydra.main
configs/                   # Hydra config tree
scripts/                   # SLURM scripts, DuckDB analysis, env setup
```

## TS convergence criterion

A transition state is defined by exactly **one negative eigenvalue in the Hessian** (Morse index 1). The only additional requirement is low force norm. This is non-negotiable:

```
converged = (n_neg == 1) AND (force_norm < threshold)
```

- `n_neg` is counted on the **Eckart-projected vibrational Hessian** (reduced-basis, full-rank)
- No eigenvalue product gates, no threshold relaxation on the n_neg check
- No `tr_threshold` filtering — only Eckart projection removes TR modes
- Force threshold: 0.01 eV/A (loose) to 0.0001 eV/A (tight)
- The eigenvalue cascade (n_neg at thresholds 0, 1e-4, ..., 1e-2) is **diagnostic only**
- State-based optimizers preferred (no path history) for diffusion model compatibility

## Hessian

Always use the **direct analytical Hessian from HIP** (via `do_hessian=True`). The Eckart projection in `projection/projection.py` is fully differentiable — all operations are pure torch, autograd flows through the entire pipeline. This matters for HIP's `require_grad=True` path.

## predict_fn interface

All algorithms use `predict_fn(coords, atomic_nums, do_hessian, require_grad) -> dict` with keys `energy`, `forces`, `hessian`. Backend lives in `calculator/hip.py`. Core algorithms never import HIP directly.

## Bottom-up feature levels

Each is a separate Hydra config, benchmarked independently:

| Level | Config | What's added |
|-------|--------|-------------|
| 0 | `pure_gad` | Raw Hessian, fixed dt, Euler steps |
| 1 | `gad_tracked` | + Mode tracking (k=8) |
| 2 | `gad_projected` | + Eckart projection (reduced-basis) |
| 3 | `gad_adaptive_dt` | + Eigenvalue-clamped adaptive dt |
| 4 | `nr_gad_flipflop` | + NR refinement when n_neg==1 |

## Dataset

Transition1x HDF5. Three splits available:

| Split | Samples | Notes |
|-------|---------|-------|
| train | 9,561 | Default. We didn't train on T1x, so fair game |
| test | 287 | Small eval set |
| val | 225 | Validation set |

Override with `split=test` or `split=val` on the CLI.

## Cluster setup

### Narval (primary) — A100 MIG slicing

HIP inference on small molecules is sequential and uses <2GB VRAM. A full A100 would sit 95% idle. MIG slices give the same per-job speed with much faster queue dispatch and better cluster utilization.

```
Cluster:   Narval (Calcul Québec, Alliance Canada)
Login:     narval.alliancecan.ca
Account:   rrg-aspuru
GPU:       a100_2g.10gb:1 (MIG slice: 10GB VRAM, 2/8 compute)
CPU:       4 cores per job
RAM:       16GB per job
Project:   /lustre06/project/6033559/memoozd
Scratch:   /lustre07/scratch/memoozd
Policy:    Min 1h jobs, max 7 days, max 1000 queued. No internet on compute.
```

Batch MIG jobs start within minutes. Interactive `salloc` has ~3hr wait (smaller pool) — prefer `sbatch`.

### Trillium (secondary) — H100 full GPUs

Override with `cluster=trillium`. Useful for larger molecules or model training.

```
Account:   rrg-aspuru
GPU:       1× H100-SXM (full, 80GB)
CPU:       12 cores per job
RAM:       64GB per job
Project:   /project/rrg-aspuru/memoozd
Scratch:   /scratch/memoozd
```

## Running experiments

### First-time setup (Narval)

```bash
ssh narval.alliancecan.ca
cd /lustre06/project/6033559/memoozd/GAD_plus
bash scripts/setup_env.sh
```

This creates the venv, installs GAD_plus + HIP + transition1x, and creates scratch dirs.

### Quick test

```bash
source .venv/bin/activate
python -c "from gadplus.core.convergence import is_ts_converged; print('OK')"
```

### Single SLURM job (sbatch — starts in minutes)

```bash
sbatch scripts/run_narval.slurm
```

### Sweep: all methods × all noise levels

Launches up to 500 independent MIG jobs via Hydra + Submitit:

```bash
bash scripts/run_narval_sweep.sh

# Or manually:
python -m gadplus.orchestration.run --multirun \
    hydra/launcher=submitit_slurm \
    search=pure_gad,gad_tracked,gad_projected,gad_adaptive_dt,nr_gad_flipflop \
    starting.noise_levels_pm=0,1,3,5,10,15 \
    max_samples=300
```

### Overnight reserved-node workflow

Reserve a full A100 node, run experiments sequentially with `srun`. Claude Code checks every hour, fixes bugs, resubmits:

```bash
salloc --account=rrg-aspuru --gpus=a100:1 --cpus-per-task=12 --mem=64G --time=12:00:00
bash scripts/run_narval_reserved.sh
```

### Interactive single run (on reserved node)

```bash
srun --overlap python -m gadplus.orchestration.run \
    search=gad_projected starting=noised_ts max_samples=50
```

### Switch to Trillium

```bash
python -m gadplus.orchestration.run cluster=trillium search=gad_projected max_samples=50
```

## Hydra configuration

### Config structure

```
configs/
  config.yaml              # Top-level: defaults, seed, max_samples, split, output_dir
  search/                  # pure_gad.yaml, gad_tracked.yaml, gad_projected.yaml, etc.
  starting/                # noised_ts.yaml, reactant.yaml, geodesic.yaml
  calculator/hip.yaml      # Checkpoint path, device, h5_path (uses ${cluster.project_dir})
  cluster/                 # narval.yaml, trillium.yaml
  logging/default.yaml     # MLflow experiment name, log settings
  hydra/launcher/          # submitit_slurm.yaml (SLURM launcher config)
```

### Common overrides

```bash
# Change search method
python -m gadplus.orchestration.run search=nr_gad_flipflop

# Change noise levels
python -m gadplus.orchestration.run starting.noise_levels_pm="[0,5,15]"

# Change dataset split and sample count
python -m gadplus.orchestration.run split=test max_samples=50

# Change cluster
python -m gadplus.orchestration.run cluster=trillium

# Multirun sweep (launches SLURM array jobs)
python -m gadplus.orchestration.run --multirun \
    hydra/launcher=submitit_slurm \
    search=pure_gad,gad_projected \
    starting.noise_levels_pm=0,5,10

# Override output directory
python -m gadplus.orchestration.run output_dir=/lustre07/scratch/memoozd/gadplus/runs/my_experiment
```

### All paths are portable via `${cluster.project_dir}` and `${cluster.scratch_dir}`

The calculator config references `${cluster.project_dir}/models/hip_v2.ckpt` — switching clusters automatically updates all paths.

## Analyzing results

### DuckDB analysis (scripts/analyze.py)

Reads all Parquet files from experiment runs and computes aggregate statistics:

```bash
python scripts/analyze.py /lustre07/scratch/memoozd/gadplus/runs/

# Or a specific run directory
python scripts/analyze.py /lustre07/scratch/memoozd/gadplus/runs/20250403_143000/
```

Reports:
- **Convergence by method & noise** — success rate, avg steps, avg wall time
- **Failure autopsy** — distribution of 6 failure types per method
- **Hardest samples** — which molecules have the lowest convergence rate

### Raw DuckDB queries

```python
import duckdb

# All converged TS across all runs
df = duckdb.execute("""
    SELECT * FROM '/lustre07/scratch/memoozd/gadplus/runs/*/summary_*.parquet'
    WHERE converged AND final_n_neg = 1 AND final_force_norm < 0.01
""").df()

# Eigenvalue trajectory for a specific run
df = duckdb.execute("""
    SELECT step, n_neg, eig0, eig1, force_norm, mode_overlap, phase
    FROM '/lustre07/scratch/memoozd/gadplus/runs/*/traj_*.parquet'
    WHERE run_id = 'abc12345'
    ORDER BY step
""").df()

# Convergence rate by noise level
df = duckdb.execute("""
    SELECT start_method, search_method,
           COUNT(*) as total,
           SUM(CASE WHEN converged THEN 1 ELSE 0 END) as n_conv,
           ROUND(100.0 * n_conv / total, 1) as rate
    FROM '/lustre07/scratch/memoozd/gadplus/runs/*/summary_*.parquet'
    GROUP BY ALL ORDER BY ALL
""").df()
```

### Per-step trajectory data (Parquet schema, 40 fields)

Each GAD step logs: energy, force_norm, force_rms, n_neg, eig0, eig1, eig_product, bottom_spectrum (6 eigenvalues), cascade n_neg at 8 thresholds, 5 eigenvalue band populations, mode_overlap, mode_index, eigvec_continuity, grad_v0_overlap, grad_v1_overlap (bottleneck detector), disp_from_start, disp_from_last, dist_to_known_ts, coords_flat.

### Failure autopsy classification

Failed runs are classified into 6 types:
1. **Ghost modes** — all negative eigenvalues in [-1e-4, 0), not real
2. **Almost converged** — n_neg ≤ 2 and force small, just missed threshold
3. **Oscillating** — eigenvalues fluctuate without net progress
4. **Energy plateau** — energy stagnant over last 100 steps
5. **Genuinely stuck** — n_neg unchanged for >50% of trajectory
6. **Drifting** — improving eigenvalues but too slowly

## Key paths (Narval)

```
Project:      /lustre06/project/6033559/memoozd/GAD_plus
Venv:         /lustre06/project/6033559/memoozd/GAD_plus/.venv
HIP:          /lustre06/project/6033559/memoozd/hip
Transition1x: /lustre06/project/6033559/memoozd/transition1x
HIP ckpt:     /lustre06/project/6033559/memoozd/models/hip_v2.ckpt
T1x data:     /lustre06/project/6033559/memoozd/data/transition1x.h5
Runs output:  /lustre07/scratch/memoozd/gadplus/runs/
MLflow:       /lustre07/scratch/memoozd/gadplus/mlruns/
SLURM logs:   /lustre07/scratch/memoozd/gadplus/logs/
```

## Key paths (Trillium)

```
Project:      /project/rrg-aspuru/memoozd/GAD_plus
HIP ckpt:     /project/rrg-aspuru/memoozd/models/hip_v2.ckpt
T1x data:     /project/rrg-aspuru/memoozd/data/transition1x.h5
Runs output:  /scratch/memoozd/gadplus/runs/
```

## Performance notes

- Threading pinned in orchestration/run.py: OMP=1, torch=2 (avoids contention on MIG)
- Dataset loaded once into memory; samples processed sequentially on GPU
- Each sample's trajectory flushed to its own Parquet file (crash-safe, partial results usable)
- DuckDB analysis works on partial results (glob over whatever exists)
- No internet on compute nodes: MLflow uses `file://` URI
- Batch MIG jobs start in minutes; interactive salloc waits ~3hrs (use sbatch)

## Autonomous overnight workflow

The typical pattern for running experiments:

1. **Reserve a node** for 6-12 hours with `salloc`
2. **Start experiments** via `srun` within the reservation
3. **Claude Code checks** every ~hour: reads logs, runs `analyze.py`, identifies failures
4. **Fix and resubmit**: fix bugs in the code, re-run failed experiments on the same node
5. **Morning**: review the summary Parquet and failure autopsy

The Parquet-per-sample design means partial runs always produce usable data. No database locks, no corruption from interrupted jobs.

## Experiments

### Overview

All experiments use HIP on Transition1x (train split, 9561 samples). No SCINE. No Sella/KinBot benchmark — just T1x. Bottom-up: benchmark pure GAD first, then add features one at a time.

### Experiment 1: Noised TS (primary)

Start from known TS geometry + Gaussian noise. Like noisyTS.tex Figure 3.

- Noise levels: 0, 1, 3, 5, 7, 10, 12, 15 pm
- 10 noise seeds per level
- Compare: GAD levels 0-4 vs Sella-BFGS-Internal vs Sella-Hessian-Internal vs Sella-Hessian-Cartesian

### Experiment 2: Geodesic interpolation reactant→product

Start from interpolated geometry between reactant and product (not from noised TS). Tests whether GAD can find the TS from a path guess.

### Experiment 3: Random starting point + NR+GAD

Start from random geometry. Use NR+GAD flip-flop. Hardest starting condition.

### Experiment 4 (secondary): Relaxation basin mapping

Start from known TS, add increasing noise, check if we return to the SAME TS. Maps the basin of attraction around each saddle point. How far can we go before we land on a different TS?

### Metrics for every experiment

1. **Converged**: `n_neg == 1 AND force_norm < threshold` (use Sella's threshold for fair comparison)
2. **Intended (IRC validation)**: From converged TS, run Sella IRC forward + backward. Compare endpoints to known reactant/product via RMSD. Both match → intended. One matches → half-intended. Neither → unintended.
3. **Frequency analysis**: Eckart-projected vibrational eigenvalues confirm n_neg == 1.

IRC pseudocode:
```python
from sella import IRC
atoms.calc = HipASECalculator(predict_fn, atomic_nums)
optimizer_kwargs = {"dx": 0.1, "eta": 1e-4, "gamma": 0.4, "keep_going": True}
forward = IRC(atoms, **optimizer_kwargs).run(direction="forward")
reverse = IRC(atoms, **optimizer_kwargs).run(direction="reverse")
# Compare forward/reverse endpoints to known reactant/product via RMSD
```

### Baselines

- **Sella TS-BFGS** (quasi-Newton, internal coordinates) — standard QN baseline
- **Sella full Hessian** (internal coordinates) — gold standard, expensive
- **Sella full Hessian** (Cartesian coordinates) — tests coordinate system effect
- **Our GAD levels 0-4** — pure GAD through NR+GAD flip-flop

### Visualizations and logging

Every run produces:
- **Per-step Parquet** with 40 fields (energy, forces, eigenvalues, mode tracking, displacements, etc.)
- **Eigenvalue evolution plots**: eig0, eig1, eig_product vs step (key for seeing when n_neg hits 1)
- **Force convergence plot**: force_norm vs step
- **Energy trajectory**: energy vs step
- **Mode overlap plot**: tracking quality across steps (detects mode crossings)
- **Gradient-mode overlap**: |grad · v_i| / |grad| — bottleneck detector for stuck modes
- **Displacement plots**: from start, from last step, to known TS
- **Failure autopsy**: 6-class classification for every failed run

Aggregate visualizations (like noisyTS.tex figures):
- **Intended count vs noise level** (with 95% CI from seeds) — the main result figure
- **TS failure count vs noise level** (with 95% CI)
- **Wall time distribution** (violin/box plots per method)
- **Convergence rate table**: method × noise level × starting geometry

Paper page: like https://bestquark.github.io/springs-and-sticks/
GIF: random geometry → TS via GAD dynamics

## Don't

- Don't use anything other than n_neg==1 + force<threshold as TS convergence
- Don't skip Eckart projection when computing vibrational eigenvalues
- Don't add eigenvalue product gates or tr_threshold filtering
- Don't add features without independent benchmarking justification
- Don't use path-based state (trajectory history) in optimizers
- Don't import HIP in core/ or projection/ — use the predict_fn interface
- Don't add Co-Authored-By lines to git commits
- Don't use `salloc` when `sbatch` would be faster (MIG queue is lightly loaded)
