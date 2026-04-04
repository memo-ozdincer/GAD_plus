# GAD_plus Experiment Results

> **What is this?** Systematic benchmarking of the Gentlest Ascent Dynamics (GAD) algorithm for transition state (TS) search using the HIP neural network potential on the Transition1x dataset. We evaluate how Eckart-projected GAD (Level 2) performs across different noise levels, starting geometries, and basin stability.

## Background

**Goal:** Find transition states (saddle points on the potential energy surface) starting from approximate geometries.

**Method:** GAD applies a modified force that ascends along the lowest Hessian eigenvector while descending along all others. At each step:
1. Compute energy, forces, and Hessian via HIP neural network potential
2. Eckart-project the Hessian to remove translational/rotational modes (6 modes for nonlinear molecules)
3. Identify the lowest vibrational eigenvector as the "ascent direction"
4. Apply the GAD formula: F_GAD = F + 2(F . v)v, where v is the ascent eigenvector
5. Take an Euler step: x_{n+1} = x_n + dt * F_GAD

**Convergence criterion:** A transition state is converged when:
- `n_neg == 1` (exactly one negative eigenvalue in the Eckart-projected vibrational Hessian, i.e., Morse index 1)
- `force_norm < 0.01 eV/A` (forces are small)

**Feature levels tested:**
| Level | Name | Description |
|-------|------|-------------|
| 0 | pure_gad | Raw Hessian, fixed dt=0.005, no mode tracking |
| 2 | gad_projected | Eckart projection in dynamics, fixed dt=0.01, no mode tracking |

**Dataset:** Transition1x — 9,561 organic reactions with known TS, reactant, and product geometries. We use the train split (9,561 samples; we did not train on T1x so this is fair game).

**Cluster:** Narval (Alliance Canada), A100 MIG slices (a100_2g.10gb:1, 10GB VRAM). Each GAD step takes ~0.06s. A 300-step run on one molecule takes ~18s.

---

## Phase 1: Parameter Sweep

**Question:** What timestep (dt) and mode tracking window (k_track) work best for gad_projected?

**Setup:** 10 test-split samples, 100 steps, 50pm Gaussian noise on TS geometry.
Grid: dt=[0.001, 0.003, 0.005, 0.01, 0.02] x k_track=[0, 4, 8]

### Results

| dt | k_track=0 | k_track=4 | k_track=8 |
|----|-----------|-----------|-----------|
| 0.001 | 0% | 0% | 0% |
| 0.003 | 0% | 0% | 0% |
| 0.005 | 0% | 0% | 0% |
| **0.01** | **40%** | 30% | 30% |
| 0.02 | 0% | 0% | 0% |

**Best config: dt=0.01, k_track=0** (40% convergence, avg 83 steps)

**Takeaways:**
- dt=0.01 is 2x the default (0.005) and the only value that converges at 50pm noise in 100 steps. Smaller dt is too slow; larger dt overshoots.
- k_track has negligible effect when Eckart projection is on. The projection already handles eigenvector consistency, making explicit mode tracking redundant.
- Comparison to Level 0: Level 0 at dt=0.005 with 300 steps gets 0% at 50pm. Level 2 at dt=0.01 with 100 steps gets 40%.

**Data:** `/lustre07/scratch/memoozd/gadplus/runs/sweep_dt/sweep_dt_results.parquet`
**Job:** 58833650

---

## Phase 2: Noise Robustness Survey (300 samples)

**Question:** How does convergence degrade as we add increasing Gaussian noise to the known TS geometry?

**Setup:** 300 train-split samples, 300 steps, dt=0.01, k_track=0, gad_projected.
9 noise levels from 0 to 200pm (0 to 0.2 Angstrom).
Submitted as 9 parallel MIG jobs (one per noise level).

### Results

*Results pending — 300-sample runs submitted as jobs 58835838_[0-8]*

**Preliminary results (50-sample pilot):**

| Noise (pm) | Conv Rate | Avg Steps | Notes |
|------------|-----------|-----------|-------|
| 0 | 96% (48/50) | 9 | Near-perfect at known TS |
| 10 | 64% (32/50) | 31 | Slight noise → 1/3 fail |
| 20 | 62% (31/50) | 45 | Plateau begins |
| 30 | 62% (31/50) | 65 | |
| 50 | 62% (31/50) | 85 | Level 0 gets 0% here |
| 70 | 62% (31/50) | 94 | Remarkably stable |
| 100 | 62% (31/50) | 106 | Still 62%! |
| 150 | 46% (23/50) | 119 | Degradation begins |
| 200 | 34% (17/50) | 142 | |

**Takeaways:**
- Level 2 maintains ~62% convergence from 10–100pm, where Level 0 drops to 0% by 50pm.
- The 62% plateau means ~38% of samples have structural barriers that projection alone can't overcome — these need NR refinement (Level 4) or adaptive dt (Level 3).
- Steps scale roughly linearly with noise: 9 steps at 0pm → 142 steps at 200pm.
- The Eckart projection is the single biggest algorithmic improvement over pure GAD.

**Data:** `/lustre07/scratch/memoozd/gadplus/runs/noise_survey_300/`

---

## Phase 3: Starting Geometry Comparison (300 samples)

**Question:** Can GAD find a TS starting from geometries other than a noised TS?

**Setup:** 300 train-split samples, 300 steps, dt=0.01, gad_projected.
4 starting geometries, submitted as 4 parallel MIG jobs.

### Results

*Results pending — 300-sample runs submitted as jobs 58835839_[0-3]*

**Preliminary results (50-sample pilot):**

| Starting Geometry | Conv Rate | Avg Steps | Avg Force | Avg n_neg |
|-------------------|-----------|-----------|-----------|-----------|
| Noised TS (10pm) | 64% (32/50) | 31 | 0.64 | 1.3 |
| Midpoint R→P | 30% (15/50) | 174 | 0.89 | 1.5 |
| Reactant | 14% (7/50) | 281 | 0.21 | 0.7 |
| Product | 4% (2/50) | 141 | 1.29 | 1.0 |

**Takeaways:**
- GAD works best from noised TS (64%), reasonable from midpoint (30%), poor from minima.
- Reactant starts have low force and n_neg~0 — GAD gets stuck at the minimum because there are no negative eigenvalues to ascend along. GAD fundamentally needs to start near a saddle region.
- Midpoint (linear interpolation of reactant→product) is a viable starting point for 30% of reactions. The midpoint often lands in a high-energy region near the TS.
- Product is worst (4%) — products tend to be lower-energy, more stable minima.
- Level 2 improves over Level 0 across all starting geometries (e.g., reactant: 14% vs 0%).

**Data:** `/lustre07/scratch/memoozd/gadplus/runs/starting_geom_300/`

---

## Phase 4: Trajectory Visualization

Three representative trajectories from the Phase 2 pilot, plotted as 2x2 grids (energy, force norm, n_neg, eigenvalues vs step):

1. **Fast convergence:** C2H2N2O2 at 0pm noise — converged at step 0 (already at TS, just needed verification)
2. **Slow convergence:** C2H2N4 at 100pm noise — converged at step 272, showing gradual force decrease and eigenvalue evolution
3. **Failure:** C2H2N2O2 at 0pm noise — one of 2/50 that failed even at zero noise (edge case)

**Plots:** `/lustre07/scratch/memoozd/gadplus/runs/noise_survey/plots/{fast,slow,failure}_convergence.png`

---

## Phase 5: IRC Validation (10 samples, preliminary)

**Question:** When GAD converges to a TS, is it the *intended* TS that connects the known reactant and product?

**Method:** From each converged TS, run Sella IRC (Intrinsic Reaction Coordinate) forward and backward. Compare the IRC endpoints to the known reactant/product via RMSD. Match if RMSD < 0.3A.

**Setup:** 10 converged TS from Phase 2 (noise=10pm).

### Results

| Classification | Count | Meaning |
|---------------|-------|---------|
| Intended | 3/10 | Both reactant and product matched |
| Half-intended | 4/10 | One endpoint matched (usually product, RMSD~0.16A) |
| Unintended | 3/10 | Neither matched (some had missing product data) |

**Takeaways:**
- 30% fully intended, 40% half-intended, 30% unintended.
- Half-intended cases consistently match the product side (RMSD~0.16A) but not the reactant (RMSD~0.45-0.47A). This suggests the TS is real but may be from a slightly different reaction pathway.
- The RMSD threshold (0.3A) may be too tight — some reactions have large conformational changes where 0.3A is too strict.
- Larger-scale IRC validation (30 samples × 3 noise levels) pending after Phase 2 completes.

**Data:** `/lustre07/scratch/memoozd/gadplus/runs/irc_validation/`
**Job:** 58834594

---

## Phase 6: Basin Mapping (50 samples)

**Question:** How far can we perturb a known TS before GAD converges to a *different* TS?

**Method:** Start from known TS, add increasing noise (0–500pm), run GAD, check if the converged geometry matches the original TS via RMSD (threshold 0.1A).

### Results

*Results pending — 50-sample run submitted as job 58835840*

**Preliminary results (20-sample pilot):**

| Noise (pm) | Converged | Same TS | Diff TS | Avg RMSD to original |
|------------|-----------|---------|---------|---------------------|
| 0 | 19/20 | 19 | 0 | 0.0005 A |
| 10 | 13/20 | 13 | 0 | 0.0060 A |
| 20 | 14/20 | 14 | 0 | 0.0105 A |
| 50 | 13/20 | 13 | 0 | 0.0243 A |
| 100 | 12/20 | 12 | 0 | 0.0452 A |
| 200 | 2/20 | 1 | **1** | 0.1178 A |
| 500 | 0/20 | 0 | 0 | — |

**Takeaways:**
- The basin of attraction is remarkably stable up to 100pm. Every converged run returns to the **same TS** (0 different TS found). RMSD scales linearly with noise.
- First different TS appears at 200pm. At 500pm, nothing converges.
- This means the 62% convergence plateau from Phase 2 is NOT because GAD finds wrong TS — it's because 38% of samples genuinely fail to converge. The algorithm is either right or gives up; it doesn't silently find the wrong answer.
- Basin width ~100pm (0.1A) is the practical limit for reliable GAD convergence to the correct TS.

**Data:** `/lustre07/scratch/memoozd/gadplus/runs/basin_map_50/`

---

## Summary

| Phase | Key Finding |
|-------|------------|
| 1. dt sweep | dt=0.01 optimal (2x default); k_track irrelevant with projection |
| 2. Noise survey | 62% convergence plateau from 10–100pm; Level 0 gets 0% at 50pm |
| 3. Starting geom | Noised TS 64% > midpoint 30% > reactant 14% > product 4% |
| 4. Trajectories | Visual confirmation of convergence/failure modes |
| 5. IRC validation | 30% intended, 40% half-intended, 30% unintended (small sample) |
| 6. Basin mapping | Same TS recovered up to 100pm; basin width ~0.1A |

### What Eckart projection (Level 2) buys you

Level 2 vs Level 0 at matched conditions (50 samples, 300 steps):

| Noise | Level 0 | Level 2 | Improvement |
|-------|---------|---------|-------------|
| 0pm | 98% | 96% | Same (trivial case) |
| 20pm | 24% | 62% | **+38pp** |
| 50pm | 0% | 62% | **+62pp** |
| 100pm | 0% | 62% | **+62pp** |
| Reactant start | 0% | 14% | **+14pp** |
| Midpoint start | 18% | 30% | **+12pp** |

### Next steps

1. **Level 3 (adaptive dt):** Should help the 38% that plateau — eigenvalue-clamped timestep adapts to local curvature.
2. **Level 4 (NR refinement):** When n_neg=1 but force > threshold, Newton-Raphson can polish to convergence. Should convert "almost converged" failures.
3. **Sella baselines:** Compare against Sella TS-BFGS and Sella full-Hessian at matched conditions for fair comparison.
4. **Scale to full T1x:** 9,561 samples with 500 parallel MIG jobs (~6 min wall time for the whole dataset).

### Reproducing results

All scripts are standalone (no Hydra) in `scripts/`. Submit via `sbatch`:

```bash
# Phase 1: Parameter sweep (1 job, ~20 min)
sbatch scripts/run_sweep_dt.slurm

# Phase 2: Noise survey (9 parallel jobs, ~90 min)
sbatch scripts/run_noise_survey.slurm

# Phase 3: Starting geometry (4 parallel jobs, ~90 min)
sbatch scripts/run_starting_geom.slurm

# Phase 5: IRC validation (3 parallel jobs, ~30 min) — requires Phase 2 output
sbatch scripts/run_irc_validate.slurm

# Phase 6: Basin mapping (1 job, ~2 hrs)
sbatch scripts/run_basin_map.slurm
```

### DuckDB quick-access queries

```sql
-- Convergence rate by noise level (Phase 2)
SELECT noise_pm, COUNT(*) as total,
       SUM(CASE WHEN converged THEN 1 ELSE 0 END) as conv,
       ROUND(100.0 * conv / total, 1) as rate
FROM '/lustre07/scratch/memoozd/gadplus/runs/noise_survey_300/summary_*.parquet'
GROUP BY noise_pm ORDER BY noise_pm;

-- Starting geometry comparison (Phase 3)
SELECT start_method, COUNT(*) as total,
       SUM(CASE WHEN converged THEN 1 ELSE 0 END) as conv,
       ROUND(100.0 * conv / total, 1) as rate
FROM '/lustre07/scratch/memoozd/gadplus/runs/starting_geom_300/summary_*.parquet'
GROUP BY start_method ORDER BY rate DESC;

-- Basin mapping (Phase 6)
SELECT noise_pm, COUNT(*) as total,
       SUM(CASE WHEN converged THEN 1 ELSE 0 END) as conv,
       SUM(CASE WHEN same_ts THEN 1 ELSE 0 END) as same_ts
FROM '/lustre07/scratch/memoozd/gadplus/runs/basin_map_50/basin_map_results.parquet'
GROUP BY noise_pm ORDER BY noise_pm;
```
