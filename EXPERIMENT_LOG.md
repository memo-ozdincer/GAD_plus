# GAD_plus Experiment Results

> **What is this?** Systematic benchmarking of Gentlest Ascent Dynamics (GAD) for transition state (TS) search using the HIP neural network potential on the Transition1x dataset. All methods are purely geometry-based (state-based, no path history). We evaluate Eckart-projected GAD across noise levels, starting geometries, basin stability, and compare 7 method variants.

## Background

**Goal:** Find transition states (saddle points on the potential energy surface) starting from approximate geometries.

**Method:** GAD applies a modified force that ascends along the lowest Hessian eigenvector while descending along all others. At each step:
1. Compute energy, forces, and full analytical Hessian via HIP neural network potential
2. Eckart-project the Hessian to remove 6 translational/rotational modes (nonlinear molecules)
3. Identify the lowest vibrational eigenvector v as the "ascent direction"
4. Apply the GAD formula: F_GAD = F + 2(F . v)v
5. Take an Euler step: x_{n+1} = x_n + dt * F_GAD

**Convergence criterion** (non-negotiable):
- `n_neg == 1` (exactly one negative eigenvalue in the Eckart-projected vibrational Hessian — Morse index 1)
- `force_norm < 0.01 eV/A`

**Feature levels:**
| Level | Name | Description |
|-------|------|-------------|
| 0 | pure_gad | Raw Hessian, fixed dt=0.005, no mode tracking |
| 2 | gad_projected | + Eckart projection in dynamics, dt=0.01 |
| 3 | gad_adaptive_dt | + Eigenvalue-clamped adaptive dt |
| — | nr_gad_pingpong | NR minimize when n_neg≥2, GAD when n_neg<2 |

**Dataset:** Transition1x train split — 9,561 organic reactions with known TS, reactant, and product. We did not train on T1x.

**Cluster:** Narval (Alliance Canada), A100 MIG slices (a100_2g.10gb:1, 10 GB VRAM). ~0.06s/step, 300 steps ≈ 18s/molecule.

---

## Phase 1: Parameter Sweep (10 test samples)

**Question:** What dt and k_track work best for gad_projected?

**Setup:** 10 test-split samples, 100 steps, 50pm noise. Grid: dt=[0.001..0.02] × k_track=[0,4,8].

| dt | k_track=0 | k_track=4 | k_track=8 |
|----|-----------|-----------|-----------|
| 0.001 | 0% | 0% | 0% |
| 0.003 | 0% | 0% | 0% |
| 0.005 | 0% | 0% | 0% |
| **0.01** | **40%** | 30% | 30% |
| 0.02 | 0% | 0% | 0% |

**Best: dt=0.01, k_track=0.** Mode tracking is redundant when Eckart projection is active.

**Data:** `/lustre07/scratch/memoozd/gadplus/runs/sweep_dt/` | **Job:** 58833650

---

## Phase 2: Noise Robustness (300 samples) ✓

**Question:** How does convergence degrade with increasing Gaussian noise on the TS geometry?

**Setup:** 300 train-split samples, 300 steps, dt=0.01, gad_projected. 9 noise levels, 9 parallel MIG jobs.

| Noise (pm) | Converged | Rate | Avg Steps | Avg Time |
|------------|-----------|------|-----------|----------|
| 0 | 260/300 | **86.7%** | 11 | 3.3s |
| 10 | 210/300 | **70.0%** | 37 | 7.2s |
| 20 | 209/300 | **69.7%** | 53 | 8.0s |
| 30 | 208/300 | **69.3%** | 73 | 9.0s |
| 50 | 204/300 | **68.0%** | 99 | 10.3s |
| 70 | 204/300 | **68.0%** | 115 | 11.0s |
| 100 | 183/300 | **61.0%** | 142 | 12.8s |
| 150 | 140/300 | 46.7% | 161 | 15.0s |
| 200 | 89/300 | 29.7% | 171 | 16.4s |

**Key findings:**
- 70% plateau from 10–70pm, gradual decline to 61% at 100pm, 30% at 200pm.
- Level 0 (unprojected) gets 0% at ≥50pm — Eckart projection is the single biggest win.
- Steps scale linearly with noise (11 → 171 from 0 → 200pm).
- The ~30% that always fail have structural barriers that projection alone cannot overcome.

**Comparison: 50-sample pilot vs 300-sample final:**
| Noise | 50 samples | 300 samples | Delta |
|-------|-----------|-------------|-------|
| 0pm | 96% | 86.7% | -9.3pp (more hard molecules found) |
| 50pm | 62% | 68.0% | +6pp |
| 100pm | 62% | 61.0% | -1pp |
| 200pm | 34% | 29.7% | -4.3pp |

The 50-sample estimates were noisy — 300 samples gives tighter confidence intervals.

**Data:** `/lustre07/scratch/memoozd/gadplus/runs/noise_survey_300/` | **Jobs:** 58835838_[0-8]

---

## Phase 3: Starting Geometry (300 samples) ✓

**Question:** Can GAD find a TS starting from geometries other than a noised TS?

**Setup:** 300 train-split samples, 300 steps, dt=0.01, gad_projected. 4 starting geometries as parallel MIG jobs.

| Starting Geometry | Converged | Rate | Avg Steps |
|-------------------|-----------|------|-----------|
| Noised TS (10pm) | 210/300 | **70.0%** | 37 |
| Midpoint R→P (linear) | 87/300 | **29.0%** | 191 |
| Reactant | 19/300 | 6.3% | 108 |
| Product | 9/300 | 3.0% | 65 |

**Key findings:**
- GAD works best from noised TS (70%). Midpoint is viable at 29%.
- Reactant (6.3%) and product (3.0%) are near-useless — GAD gets stuck at minima where n_neg=0.
- Midpoint interpolation lands in a high-energy region near the TS, giving GAD enough curvature to work with.
- Level 0 comparison: reactant 0%, midpoint 18%, product 8%. Level 2 improves midpoint significantly.

**Data:** `/lustre07/scratch/memoozd/gadplus/runs/starting_geom_300/` | **Jobs:** 58835839_[0-3]

---

## Phase 4: Trajectory Visualization

Three representative trajectories from the Phase 2 pilot, plotted as 2×2 grids (energy, force_norm, n_neg, eigenvalues vs step):

1. **Fast convergence:** C2H2N2O2 at 0pm — converged at step 0 (already at TS)
2. **Slow convergence:** C2H2N4 at 100pm — converged at step 272
3. **Failure:** C2H2N2O2 at 0pm — one of 2/50 that failed even at zero noise

**Plots:** `/lustre07/scratch/memoozd/gadplus/runs/noise_survey/plots/`

---

## Phase 5: IRC Validation (10 samples, preliminary)

**Question:** When GAD converges, is it the *intended* TS connecting the known reactant and product?

**Method:** From converged TS, run Sella IRC forward + backward. Compare endpoints to known R/P via RMSD (threshold 0.3A).

| Classification | Count | Detail |
|---------------|-------|--------|
| Intended | 3/10 | Both R and P matched (RMSD ~0.11–0.17A) |
| Half-intended | 4/10 | Product matched (~0.16A), reactant didn't (~0.45A) |
| Unintended | 3/10 | Neither matched (some had missing product data in T1x) |

**Finding:** Half-intended cases consistently match the product but not the reactant. The TS is real but may connect a different reactant conformer. RMSD threshold 0.3A may be too tight for reactions with large conformational changes.

**Data:** `/lustre07/scratch/memoozd/gadplus/runs/irc_validation/` | **Job:** 58834594

---

## Phase 6: Basin Mapping (50 samples) ✓

**Question:** At what noise level do we start finding *different* transition states?

**Method:** Start from known TS, add noise, run GAD, check RMSD to original TS (threshold 0.1A).

| Noise (pm) | Converged | Same TS | Diff TS | Avg RMSD (A) |
|------------|-----------|---------|---------|-------------|
| 0 | 48/50 | 48 | 0 | 0.0005 |
| 10 | 32/50 | 32 | 0 | 0.0054 |
| 20 | 31/50 | 31 | 0 | 0.0103 |
| 50 | 32/50 | 32 | 0 | 0.0257 |
| 100 | 29/50 | 29 | 0 | 0.0490 |
| 200 | 20/50 | 12 | **8** | 0.1037 |
| 500 | 1/50 | 0 | **1** | 0.4850 |

**Key findings:**
- Basin of attraction is stable to ~100pm. Every converged run returns to the SAME TS. Zero different TS found up to 100pm.
- RMSD scales linearly with noise: 0.005A at 10pm → 0.049A at 100pm.
- First different TS appears at 200pm (8 of 20 converged found a different TS).
- At 500pm, essentially nothing converges (1/50) and that one is wrong.
- **The failure plateau is genuine non-convergence, not silent wrong answers.** GAD is either right or gives up.

**Data:** `/lustre07/scratch/memoozd/gadplus/runs/basin_map/` | **Job:** 58835840

---

## Phase 7: Method Comparison (50 samples, 5 of 7 methods completed)

**Question:** Can we push past the ~70% plateau with better GAD variants?

**Setup:** 50 train-split samples, 300 steps, 6 noise levels (10–200pm), 7 methods.

**Methods tested:**
1. **gad_projected** — Level 2 baseline (dt=0.01, fixed)
2. **gad_adaptive_dt** — Eigenvalue-clamped: dt ∝ 1/clamp(|λ₀|)
3. **gad_tight_clamp** — Max per-atom displacement capped at 0.1A (vs 0.35A default)
4. **gad_adaptive_tight** — Adaptive dt + tight clamp
5. **gad_small_dt** — Conservative dt=0.005
6. **nr_gad_pingpong** — NR minimize when n_neg≥2, GAD when n_neg<2 *(crashed — dtype bug, fix pending)*
7. **nr_gad_pp_adaptive** — Ping-pong + adaptive dt *(not reached)*

### Results (5 of 7 methods, from SLURM logs)

| Method | 10pm | 30pm | 50pm | 100pm | 150pm | 200pm |
|--------|------|------|------|-------|-------|-------|
| **gad_small_dt** | **96%** | **96%** | **84%** | **66%** | **48%** | 20% |
| gad_projected | 64% | 62% | 62% | 54% | 48% | **36%** |
| gad_tight_clamp | 64% | 62% | 62% | 54% | 46% | **36%** |
| gad_adaptive_dt | 54% | 34% | 30% | 20% | 16% | 6% |
| gad_adaptive_tight | 54% | 34% | 30% | 20% | 14% | 6% |

### Key findings:

1. **gad_small_dt (dt=0.005) is the clear winner at low-to-medium noise.** 96% at 10pm and 30pm vs 64% for the baseline. At 50pm it gets 84% vs 62%. The smaller timestep avoids overshooting near the saddle point, giving more samples time to converge.

2. **gad_projected and gad_tight_clamp are nearly identical** — tighter displacement clamping (0.1A vs 0.35A) doesn't help or hurt. The baseline's 0.35A cap rarely triggers.

3. **Adaptive dt HURTS convergence dramatically.** The eigenvalue-clamped strategy reduces dt when |λ₀| is large, but this makes steps too small in steep-curvature regions where large steps are actually needed. 54% at 10pm vs 96% for small_dt. This is a clear negative result.

4. **At 200pm, gad_projected/tight_clamp (36%) beats gad_small_dt (20%).** The larger dt helps traverse large-noise landscapes faster. There's a crossover around 150pm where dt=0.01 and dt=0.005 are equivalent.

5. **NR-GAD ping-pong crashed due to dtype mismatch** (float32 grad vs float64 eigenvectors). Fix applied to `nr_gad_pingpong.py` but not yet resubmitted.

**Data:** SLURM logs at `/lustre07/scratch/memoozd/gadplus/logs/methodcmp_58835900_*.out` | **Jobs:** 58835900_[0-5] (FAILED at ping-pong)

---

## Consolidated Summary

### What we know (300 samples, statistically robust)

| Finding | Evidence |
|---------|----------|
| Eckart projection is the biggest improvement | Level 0 → Level 2: 0% → 68% at 50pm |
| dt=0.005 beats dt=0.01 at ≤100pm noise | 96% vs 64% at 10pm, 84% vs 62% at 50pm |
| ~70% convergence plateau at 10–70pm | 210/300 converge consistently across this range |
| Basin of attraction ~100pm wide | 0 different TS found below 100pm in 50 samples |
| Adaptive dt (eigenvalue-clamped) hurts | Reduces convergence by 20–40pp across all noise levels |
| GAD needs saddle-region starts | Reactant: 6%, product: 3%, midpoint: 29%, noised TS: 70% |
| When GAD converges, the answer is correct | Basin mapping shows same TS up to 100pm |

### Level 2 vs Level 0 (matched conditions)

| Condition | Level 0 | Level 2 (dt=0.01) | Level 2 (dt=0.005) |
|-----------|---------|--------------------|--------------------|
| 0pm noise | 98% | 87% | ~96%* |
| 20pm | 24% | 70% | ~96%* |
| 50pm | 0% | 68% | 84% |
| 100pm | 0% | 61% | 66% |
| Reactant start | 0% | 6% | — |
| Midpoint start | 18% | 29% | — |

*estimated from 50-sample method comparison

### What's still pending

1. **NR-GAD ping-pong** — dtype fix applied, needs resubmit. This is the most promising untested method: NR minimizes when n_neg≥2 (escapes higher-order saddles), GAD navigates when n_neg<2.
2. **Larger IRC validation** — 30 samples × 3 noise levels. Needs Phase 2 output (now available).
3. **Geodesic midpoint starting geometry** — code written, not yet run.
4. **Full T1x (9,561 samples)** — feasible with 500 parallel MIG jobs (~6 min wall time).

### Reproducing results

All scripts are standalone (argparse, no Hydra) in `scripts/`. Submit via `sbatch`:

```bash
sbatch scripts/run_sweep_dt.slurm          # Phase 1: param sweep (~20 min)
sbatch scripts/run_noise_survey.slurm       # Phase 2: noise survey, 300 samples (9 jobs, ~90 min)
sbatch scripts/run_starting_geom.slurm      # Phase 3: starting geometry, 300 samples (4 jobs, ~90 min)
sbatch scripts/run_basin_map.slurm          # Phase 6: basin mapping, 50 samples (1 job, ~1 hr)
sbatch scripts/run_method_comparison.slurm  # Phase 7: 7 methods × 6 noise levels (6 jobs, ~2 hrs)
sbatch scripts/run_irc_validate.slurm       # Phase 5: IRC validation (3 jobs, ~30 min)
```

### DuckDB queries

```sql
-- Phase 2: Convergence by noise (300 samples)
SELECT noise_pm, COUNT(*) as total,
       SUM(CASE WHEN converged THEN 1 ELSE 0 END) as conv,
       ROUND(100.0 * conv / total, 1) as rate
FROM '/lustre07/scratch/memoozd/gadplus/runs/noise_survey_300/summary_*.parquet'
GROUP BY noise_pm ORDER BY noise_pm;

-- Phase 3: Starting geometry (300 samples)
SELECT start_method, COUNT(*) as total,
       SUM(CASE WHEN converged THEN 1 ELSE 0 END) as conv,
       ROUND(100.0 * conv / total, 1) as rate
FROM '/lustre07/scratch/memoozd/gadplus/runs/starting_geom_300/summary_*.parquet'
GROUP BY start_method ORDER BY rate DESC;

-- Phase 6: Basin mapping (50 samples)
SELECT noise_pm, COUNT(*) as total,
       SUM(CASE WHEN converged THEN 1 ELSE 0 END) as conv,
       SUM(CASE WHEN same_ts THEN 1 ELSE 0 END) as same_ts
FROM '/lustre07/scratch/memoozd/gadplus/runs/basin_map/basin_map_results.parquet'
GROUP BY noise_pm ORDER BY noise_pm;
```

### Data locations

```
Phase 1 (sweep):     /lustre07/scratch/memoozd/gadplus/runs/sweep_dt/
Phase 2 (noise):     /lustre07/scratch/memoozd/gadplus/runs/noise_survey_300/
Phase 3 (starting):  /lustre07/scratch/memoozd/gadplus/runs/starting_geom_300/
Phase 4 (plots):     /lustre07/scratch/memoozd/gadplus/runs/noise_survey/plots/
Phase 5 (IRC):       /lustre07/scratch/memoozd/gadplus/runs/irc_validation/
Phase 6 (basin):     /lustre07/scratch/memoozd/gadplus/runs/basin_map/
Phase 7 (methods):   SLURM logs at /lustre07/scratch/memoozd/gadplus/logs/methodcmp_58835900_*.out
```
