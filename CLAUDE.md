# GADplus

Clean, publishable implementation of GAD-based transition state search with HIP neural network potential. Bottom-up design: pure GAD is the base, each feature (mode tracking, Eckart projection, adaptive dt, NR flip-flop) is added and benchmarked independently.

## Project structure

```
src/gadplus/
  core/              # Pure algorithms, zero I/O
    gad.py             # GAD vector computation, Euler step
    mode_tracking.py   # Eigenvector continuity across steps
    newton_raphson.py  # Spectral-partitioned NR for TS refinement
    convergence.py     # n_neg==1 AND force<0.01, cascade analysis
    adaptive_dt.py     # Eigenvalue-clamped timestep, displacement cap
    types.py           # PredictFn protocol
  projection/        # Eckart/mass-weighting
    masses.py          # MASS_DICT, get_mass_weights_torch
    eckart.py          # Eckart generators, projector, vibrational basis
    hessian.py         # reduced_basis_hessian, vib_eig, purify_sum_rules
    gad_projected.py   # GAD dynamics with Eckart projection
  calculator/        # HIP interface
    hip.py             # make_hip_predict_fn, load_hip_calculator
    ase_adapter.py     # ASE Calculator wrapper for Sella IRC
  geometry/          # Molecular geometry utilities
    alignment.py       # Kabsch + Hungarian
    noise.py           # Gaussian noise
    interpolation.py   # Linear + geodesic interpolation
    starting.py        # StartingGeometry factory
  search/            # Search loops (all state-based, no path history)
    gad_search.py      # Main GAD loop (levels 0-3)
    nr_gad_flipflop.py # NR+GAD alternation (level 4)
    irc_validate.py    # Sella IRC validation
  logging/           # Trajectory logging + failure analysis
    trajectory.py      # TrajectoryLogger -> Parquet
    mlflow_logger.py   # MLflow offline (file://) wrapper
    autopsy.py         # 6-class failure classification
    schema.py          # PyArrow schema definitions
  data/              # Dataset loading
    transition1x.py    # Transition1xDataset, UsePos
  orchestration/     # Hydra entry point
    run.py             # @hydra.main
configs/             # Hydra config tree
scripts/             # Analysis (DuckDB), env setup
```

## Key concepts

- **predict_fn interface**: `predict_fn(coords, atomic_nums, do_hessian, require_grad) -> dict` with keys `energy`, `forces`, `hessian`. All algorithms use this; backend lives in `calculator/hip.py`.
- **HIP only**: GPU-accelerated ML potential (Equiformer). No SCINE in this codebase.
- **Eckart projection**: Raw Hessians have 5-6 rigid-body null modes. `vib_eig()` in `projection/hessian.py` returns a full-rank (3N-k, 3N-k) vibrational Hessian — no threshold filtering needed.
- **TS convergence**: **n_neg == 1 AND force_norm < 0.01 eV/A**. Non-negotiable. No eigenvalue product gates, no threshold relaxation.

## Bottom-up feature levels

| Level | Config | Feature | Description |
|-------|--------|---------|-------------|
| 0 | `pure_gad` | Baseline | Raw Hessian, fixed dt, Euler steps |
| 1 | `gad_tracked` | + Mode tracking | k=8 candidate eigenvector tracking |
| 2 | `gad_projected` | + Eckart projection | Reduced-basis vibrational Hessian |
| 3 | `gad_adaptive_dt` | + Adaptive dt | Eigenvalue-clamped timestep |
| 4 | `nr_gad_flipflop` | + NR refinement | State-based NR when n_neg==1 |

## Running experiments

```bash
# Setup
source /project/rrg-aspuru/memoozd/GADplus/.venv/bin/activate
export PYTHONPATH=/project/rrg-aspuru/memoozd/GADplus:$PYTHONPATH

# Local single run
python -m gadplus.orchestration.run search=gad_projected starting=noised_ts

# SLURM sweep (all methods x all noise levels)
python -m gadplus.orchestration.run --multirun \
    hydra/launcher=submitit_slurm \
    search=pure_gad,gad_tracked,gad_projected,gad_adaptive_dt,nr_gad_flipflop \
    starting.noise_levels_pm=0,1,3,5,10,15

# Analysis
python scripts/analyze.py /scratch/memoozd/gadplus/runs/
```

## Key paths

- HIP checkpoint: `/project/rrg-aspuru/memoozd/models/hip_v2.ckpt`
- Transition1x: `/project/rrg-aspuru/memoozd/data/transition1x.h5`
- Output: `/scratch/memoozd/gadplus/runs/`
- MLflow: `file:///scratch/memoozd/gadplus/mlruns`
- SLURM account: `rrg-aspuru`

## Don't

- Don't use anything other than n_neg==1 + force<0.01 as TS convergence
- Don't skip Eckart projection when computing vibrational eigenvalues
- Don't add features without independent benchmarking justification
- Don't use path-based state (trajectory history) in optimizers
- Don't import HIP in core/ or projection/ — use the predict_fn interface
