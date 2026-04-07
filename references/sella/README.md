# Sella Integration Reference (from ts-tools)

Sella is a trust-region saddle point optimizer from the ASE ecosystem that uses RS-P-RFO (Restricted-Step Partitioned Rational Function Optimization). These files document our integration with HIP (GPU ML potential) and SCINE (CPU semi-empirical) calculators.

**Bottom line:** Sella achieves 66.7% (SCINE) / 53.3% (HIP) TS rate at 1.0 A noise on Transition1x, vs. our Multi-Mode GAD at 100%. Sella is a useful baseline but significantly worse for noisy starting geometries.

## File inventory

### Core integration (what matters)

| File | What it does |
|------|-------------|
| `ase_calculators.py` | ASE Calculator wrappers for HIP and SCINE. The key class is `HipASECalculator` which caches Hessians to avoid double-compute. `create_hessian_function()` creates a callable that Sella uses for exact Hessians at each step. |
| `sella_ts.py` | Core runner: `run_sella_ts()` creates Sella optimizer with P-RFO, runs it, returns results. `validate_ts_eigenvalues()` does post-optimization n_neg==1 check via Eckart-projected vibrational analysis. |
| `sella_experiment.py` | Full experiment loop with W&B logging, trajectory parsing, and plotting. Handles dataset loading, parallelization, and result aggregation. |

### HIP-specific (GPU)

| File | What it does |
|------|-------------|
| `hip_sella.py` | Thin entrypoint that forces `--calculator=hip` |
| `hip_sella_hpo.py` | Grid search HPO for HIP |
| `hip_sella_hpo_parallel.py` | Parallel Bayesian HPO (Optuna) across multiple GPUs |
| `hip_sella_hpo_parallel_v2.py` | V2: adds eigenvalue-based early stopping (prune if n_neg doesn't improve for 500 steps) |

### SCINE-specific (CPU)

| File | What it does |
|------|-------------|
| `scine_sella.py` | Thin entrypoint that forces `--calculator=scine` and CPU device |
| `scine_sella_hpo.py` | Grid search over FMAX values |
| `scine_sella_hpo_bayesian.py` | Bayesian HPO for SCINE |
| `scine_sella_hpo_parallel.py` | Parallel Bayesian HPO for SCINE |

### Diagnostics

| File | What it does |
|------|-------------|
| `analyze_hessian_discrepancy.py` | Analyze HIP vs SCINE Hessian differences |
| `slurm_templates/debug_hip_sella.py` | Test HIP Hessian consistency (predicted vs numerical), monitor rho/trust radius |

## How HIP + Sella works

### Data flow

```
HipASECalculator.calculate(atoms)
  -> HIP.predict(batch, do_hessian=True)    # energy, forces, Hessian in one call
  -> cache result (so hessian_function doesn't recompute)
  -> return energy, forces to ASE

hip_hessian_function(atoms)                  # called by Sella separately
  -> check cache (same coords? reuse result)
  -> optionally apply Eckart projection
  -> return (3N, 3N) Cartesian Hessian as numpy
```

### Sella optimizer setup

```python
from sella import Sella

opt = Sella(
    atoms,
    order=1,                    # index-1 saddle (TS)
    internal=True,              # use internal coordinates (bonds/angles/dihedrals)
    delta0=0.048,               # initial trust radius (Wander et al. 2024)
    hessian_function=hess_fn,   # exact Hessian from HIP at every step
    diag_every_n=1,             # refresh Hessian every step
    gamma=0.0,                  # tightest eigensolver convergence
    # Trust radius management (Wander et al. 2024, arXiv:2410.01650v2)
    rho_inc=1.035,              # grow threshold
    rho_dec=5.0,                # shrink threshold
    sigma_inc=1.15,             # grow factor
    sigma_dec=0.65,             # shrink factor
)
opt.run(fmax=0.03, steps=200)
```

### P-RFO algorithm (what Sella does internally)

Partitioned RFO splits the step into maximization along the TS mode and minimization along all others:

**TS mode (v_0, lowest eigenvector):** augmented Hessian maximization
```
[lambda_0  g_0] [h_0]       [h_0]
[g_0       1  ] [ 1 ] = mu+ [ 1 ]
```
Solve for mu+ > lambda_0 (uphill shift).

**All other modes (v_1, v_2, ...):** standard RFO minimization
```
[lambda_1  ...  g_1] [h_1]       [h_1]
[...       ...  ...] [...] = mu- [...]
[g_1       ...   1 ] [ 1 ]       [ 1 ]
```
Solve for mu- < lambda_1 (downhill shift).

**Total step:** Delta_x = sum(h_i * v_i), subject to trust radius ||Delta_x|| <= delta.

### Eckart projection option

When `apply_eckart=True`, the Hessian goes through:
```
H_cart -> M^{-1/2} H M^{-1/2} -> P_vib H_mw P_vib -> M^{1/2} H_proj M^{1/2}
```
This removes translation/rotation modes before passing to Sella. Found to help with HIP's ML-predicted Hessians which can have spurious rigid-body components.

## HPO search space (HIP Bayesian)

```python
delta0:     [0.03, 0.8]     # log scale — initial trust radius
rho_dec:    [3.0, 80.0]     # shrink threshold
rho_inc:    [1.01, 1.1]     # grow threshold
sigma_dec:  [0.5, 0.95]     # shrink factor
sigma_inc:  [1.1, 1.8]      # grow factor
fmax:       [1e-4, 1e-2]    # log scale — force convergence
apply_eckart: [True, False]
```

## Results summary

### Sella HPO (Transition1x, sigma = 1.0 A noise)

| Calculator | Best trial | Global avg | # Trials |
|-----------|-----------|------------|----------|
| SCINE     | 66.7%     | 47.0%      | 176      |
| HIP       | 53.3%     | 30.1%      | 181      |

### Compared to our methods

| Method | Best TS rate | Noise | Notes |
|--------|-------------|-------|-------|
| Multi-Mode GAD (SCINE) | **100%** | 1.0 A | 500 Optuna trials |
| Multi-Mode GAD (HIP) | 93.3% | 1.0 A | 102 Optuna trials |
| Sella (SCINE) | 66.7% | 1.0 A | Best of 176 trials |
| Sella (HIP) | 53.3% | 1.0 A | Best of 181 trials |
| v2 Kicking (SCINE) | **100%** | **2.0 A** | Full pipeline |

### Key findings from Wander et al. (2024)

- Exact Hessians improve Sella convergence: 65% -> 93% (on their dataset, clean starts)
- Trust radius defaults (delta0=0.048, etc.) are critical for GNN potentials
- `diag_every_n=1` recommended when exact Hessians are available

## Running on cluster

```bash
# HIP single run
sbatch slurm_templates/hip_sella.slurm

# HIP HPO (parallel, 4 GPUs)
sbatch slurm_templates/hip_sella_hpo_parallel_v2.slurm

# SCINE single run
sbatch slurm_templates/scine_sella.slurm
```

## Dependencies

- `sella` (pip install sella)
- `ase` (Atomic Simulation Environment)
- `hip` (for HIP calculator)
- `scine-sparrow` (for SCINE calculator)
- `optuna` (for HPO)

## Origin

All files extracted from `ts-tools/src/experiments/Sella/`. The original code imports from `ts-tools` internal modules (`src.dependencies.*`); these won't run standalone without that infrastructure.
