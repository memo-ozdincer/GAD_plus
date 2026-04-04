# GAD_plus Experiment Log

## Phase 1: Parameter sweep — 2026-04-03

Config: gad_projected (Level 2, Eckart on), 10 test samples, 100 steps, noise=0.05A (50pm), seed=42
Grid: dt=[0.001, 0.003, 0.005, 0.01, 0.02] x k_track=[0, 4, 8]

| dt | k_track | Conv | Rate | Avg Steps |
|----|---------|------|------|-----------|
| 0.01 | 0 | 4/10 | 40% | 83 |
| 0.01 | 4 | 3/10 | 30% | 82 |
| 0.01 | 8 | 3/10 | 30% | 83 |
| all others | * | 0/10 | 0% | — |

Finding: dt=0.01 is the only value that converges at 50pm noise in 100 steps. 2x the old default (0.005). k_track has negligible effect — projection handles eigenvector consistency, making explicit mode tracking redundant.

Comparison to Level 0 (from RESULTS_2026-04-03.md): Level 0 at dt=0.005 with 300 steps got 0% at 50pm. Level 2 at dt=0.01 with only 100 steps gets 40%. Eckart projection in dynamics + larger dt is a major improvement.

Best params: **dt=0.01, k_track=0**
Next: Phase 2 noise robustness survey with these params.
Data: /lustre07/scratch/memoozd/gadplus/runs/sweep_dt/sweep_dt_results.parquet
Job: 58833650

## Phase 2: Noise robustness survey — 2026-04-03

Config: gad_projected (Level 2), dt=0.01, k_track=0, 50 train samples, 300 steps, seed=42
Noise levels: 0, 10, 20, 30, 50, 70, 100, 150, 200 pm (0 to 0.2A)

| Noise (pm) | Conv | Rate | Avg Steps | Avg Time |
|------------|------|------|-----------|----------|
| 0 | 48/50 | 96.0% | 9 | 1.3s |
| 10 | 32/50 | 64.0% | 31 | 8.0s |
| 20 | 31/50 | 62.0% | 45 | 8.9s |
| 30 | 31/50 | 62.0% | 65 | 9.7s |
| 50 | 31/50 | 62.0% | 85 | 10.9s |
| 70 | 31/50 | 62.0% | 94 | 10.7s |
| 100 | 31/50 | 62.0% | 106 | 12.9s |
| 150 | 23/50 | 46.0% | 119 | 13.6s |
| 200 | 17/50 | 34.0% | 142 | 20.1s |

Finding: Level 2 maintains ~62% convergence across 10–100pm (where Level 0 drops to 0% by 50pm). Plateau at 62% suggests ~38% of samples have structural barriers that projection alone can't overcome. Degradation starts at 150pm. Steps scale linearly with noise (9→142 steps from 0→200pm).

Comparison to Level 0: At 20pm, Level 0 gets 24% vs Level 2 gets 62%. At 50pm+, Level 0 gets 0% vs Level 2 stays at 62%. Eckart projection is the single biggest improvement.

Next: Phase 3 starting geometry comparison.
Data: /lustre07/scratch/memoozd/gadplus/runs/noise_survey/
Jobs: 58833911_[0-8]

## Phase 3: Starting geometry comparison — 2026-04-03

Config: gad_projected (Level 2), dt=0.01, k_track=0, 50 train samples, 300 steps
Noised TS uses 10pm noise. Same 50 samples across all start types.

| Start | Conv | Rate | Avg Steps | Avg Time | Avg Force | Avg n_neg |
|-------|------|------|-----------|----------|-----------|-----------|
| Noised TS (10pm) | 32/50 | 64.0% | 31 | 8.1s | 0.6435 | 1.3 |
| Midpoint R→P | 15/50 | 30.0% | 174 | 16.8s | 0.8947 | 1.5 |
| Reactant | 7/50 | 14.0% | 281 | 20.0s | 0.2091 | 0.7 |
| Product | 2/50 | 4.0% | 141 | 22.6s | 1.2917 | 1.0 |

Finding: GAD works best from noised TS (64%), reasonable from midpoint (30%), poor from reactant (14%) and product (4%). Reactant has low force (stuck at minimum, n_neg~0.7). Midpoint is 2x better than Level 0's 18% — projection helps here too.

Comparison to Level 0: Noised TS 64% vs 24% (at 20pm). Midpoint 30% vs 18%. Reactant 14% vs 0%. Product 4% vs 8% (similar — both bad).

Next: Phase 4 trajectory visualization, then Phase 5 IRC validation.
Data: /lustre07/scratch/memoozd/gadplus/runs/starting_geom/
Jobs: 58833927_[0-3]

## Phase 4: Trajectory visualization — 2026-04-04

Selected 3 representative trajectories from Phase 2:
- Fast convergence: C2H2N2O2 at 0pm noise, converged at step 0 (already at TS)
- Slow convergence: C2H2N4 at 100pm noise, converged at step 272
- Failure: C2H2N2O2 at 0pm noise, failed (2 of 50 at 0pm didn't converge)

Plots saved: /lustre07/scratch/memoozd/gadplus/runs/noise_survey/plots/
  - fast_convergence.png
  - slow_convergence.png
  - failure.png

Next: Phase 5 IRC validation on converged TS from Phase 2.

## Phase 5: IRC validation — 2026-04-04

Config: 10 converged TS from Phase 2 at noise=10pm, IRC 100 steps, RMSD threshold=0.3A

| Category | Count | Notes |
|----------|-------|-------|
| Intended | 3 | Both reactant and product matched (RMSD < 0.3A) |
| Half-intended | 4 | One endpoint matched, other didn't |
| Unintended | 3 | Neither matched (2 had missing product data → NaN RMSD) |
| Error | 0 | All IRC runs completed successfully |

Per-sample: Intended samples had RMSD~0.1-0.17A to both endpoints. Half-intended consistently matched product (RMSD~0.16A) but not reactant (RMSD~0.45-0.47A). Samples with NaN RMSD likely had missing product geometry in T1x data.

Finding: 30% fully intended, 70% partial or unintended. The half-intended cases suggest GAD finds a real TS but from a slightly rotated basin. RMSD threshold 0.3A may be too tight for some reactions with large conformational changes.

Data: /lustre07/scratch/memoozd/gadplus/runs/irc_validation/irc_validation_10pm.parquet
Job: 58834594

## Phase 6: Basin mapping — 2026-04-04

Config: 20 train samples, 7 noise levels (0-500pm), 300 steps, dt=0.01, RMSD threshold=0.1A

| Noise (pm) | Conv | Same TS | Diff TS | Avg RMSD |
|------------|------|---------|---------|----------|
| 0 | 19/20 | 19 | 0 | 0.0005 |
| 10 | 13/20 | 13 | 0 | 0.0060 |
| 20 | 14/20 | 14 | 0 | 0.0105 |
| 50 | 13/20 | 13 | 0 | 0.0243 |
| 100 | 12/20 | 12 | 0 | 0.0452 |
| 200 | 2/20 | 1 | 1 | 0.1178 |
| 500 | 0/20 | 0 | 0 | — |

Finding: Basin of attraction is remarkably stable up to 100pm — when GAD converges, it always returns to the SAME TS (0 different TS found up to 100pm). RMSD scales linearly with noise (0.005A at 10pm → 0.045A at 100pm). First different TS appears at 200pm. At 500pm nothing converges.

Key insight: GAD's basin of attraction is ~100pm wide. Below this threshold, convergence determines success; above it, the landscape becomes unpredictable. The 62% convergence plateau from Phase 2 is NOT because GAD finds different TS — it's because 38% of samples genuinely fail to converge.

Data: /lustre07/scratch/memoozd/gadplus/runs/basin_map/basin_map_results.parquet
Job: 58834606

---

## Summary of all phases

| Phase | Key Result |
|-------|-----------|
| 1. Parameter sweep | dt=0.01 optimal (2x default), k_track irrelevant with projection |
| 2. Noise robustness | 62% plateau from 10-100pm (Level 0: 0% at 50pm) |
| 3. Starting geometry | Noised TS 64% > midpoint 30% > reactant 14% > product 4% |
| 4. Trajectories | Visualized fast/slow/failure cases |
| 5. IRC validation | 30% intended, 40% half-intended, 30% unintended |
| 6. Basin mapping | Same TS recovered up to 100pm; basin width ~100pm |

**Main finding:** Eckart projection (Level 2) is the single biggest improvement over pure GAD (Level 0). Combined with dt=0.01, it extends the working range from <20pm to ~100pm noise. The 38% failure rate is structural — these samples need NR refinement (Level 4) or adaptive dt (Level 3).
