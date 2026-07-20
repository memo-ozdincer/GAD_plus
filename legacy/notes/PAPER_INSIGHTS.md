# GAD+ Paper Insights & Roadmap

> Running document of paper-level insights, narrative positioning, comparison with noisyTS (LMHE paper), and what's needed for a solid publication.

## Core Narrative

**Thesis:** In the MLIP regime where analytical Hessians are cheap (~0.06s), you don't need ML eigenvector prediction or quasi-Newton updates. Just compute the exact Hessian, Eckart-project it, and step carefully with GAD. The simplicity is the feature — fewer failure modes, comparable wall time, much wider noise tolerance.

**Their paper (Wu et al., LMHE):** "How to avoid computing the Hessian" → train a neural net to predict the leftmost eigenvector, use ensemble UQ for fallback.

**Our paper:** "You don't need to avoid it" → with fast MLIPs, compute the full Hessian every step. Use the simplest possible dynamics (first-order Euler + Eckart projection). No PRFO, no trust radius, no QN updates, no eigenvector prediction, no ensemble.

## Comparison with noisyTS / LMHE Paper

### Their setup
- 240 Sella benchmark reactions (combustion, 7-25 atoms)
- NewtonNet PES (MLIP), Cartesian coordinates via Sella
- Noise range: 0-15 pm (very narrow!)
- 10 noise seeds → 95% CI
- Methods: QN (TS-BFGS + Jacobi-Davidson), Full Hessian, LMHE single, LMHE ensemble
- Optimizer: RS-PRFO with trust radius (Sella), internal coordinates
- Validation: IRC → graph isomorphism (Open Babel + VF2)
- Categories: intended / partially intended / unintended / no reaction / TS error

### Our setup
- 300 Transition1x reactions (organic, similar size)
- HIP PES (MLIP), Cartesian coordinates + Eckart projection
- Noise range: 0-200 pm (13x wider)
- 1 noise seed (42) — need more for CIs
- Methods: 7 GAD variants (projected, small_dt, adaptive, NR-GAD pingpong, etc.)
- Optimizer: Forward Euler + displacement cap (trivially simple)
- Validation: IRC → RMSD threshold (less rigorous than graph isomorphism)
- Categories: converged / not converged (simpler than theirs)

### Reading their Figure 3 (noise robustness, 240 reactions)

| Noise | Full Hessian intended | LMHE ensemble intended | QN intended | Full Hessian failures | LMHE single failures |
|-------|----------------------|----------------------|-------------|----------------------|---------------------|
| 0pm | ~105/240 (44%) | ~105/240 (44%) | ~100/240 (42%) | ~5 | ~10 |
| 5pm | ~95 (40%) | ~95 (40%) | ~80 (33%) | ~5 | ~15 |
| 10pm | ~75 (31%) | ~75 (31%) | ~60 (25%) | ~5 | ~30 |
| 15pm | ~55 (23%) | ~55 (23%) | ~45 (19%) | ~5 | ~50 |

### Reading their Figure 4 (wall time)
- QN: peaks ~2s (cheapest, but worst robustness)
- LMHE: peaks ~5s
- Full Hessian: peaks ~5s but heavy tail to 30s+, outlier at 464s

### Direct comparison at overlap (0-15pm)

| Metric | Their Full Hessian | Their LMHE ensemble | Our gad_small_dt |
|--------|-------------------|--------------------|--------------------|
| 0pm | ~44% intended | ~44% intended | 87% converged* |
| 10pm | ~31% intended | ~31% intended | 94% converged* |
| 15pm | ~23% intended | ~23% intended | ~94% converged* |
| Failures at 15pm | ~2% (5/240) | ~4% (10/240) | ~6% (18/300) |
| Wall time/sample | 5-30s | 3-8s | 8-18s |

*converged ≠ intended — our IRC on 10 samples showed only 30% intended (but RMSD-based, likely undercount)

### Key observations from the comparison

1. **Their "intended" bar is much higher than our "converged."** We're measuring different things. Their 44% intended at 0pm includes IRC + graph isomorphism. Many of our 87% converged TS are valid saddle points connecting different pathways. To compare fairly we need graph isomorphism validation.

2. **Their QN collapses fast (42%→19% from 0→15pm).** Our GAD holds 94% convergence across that same range. GAD dynamics are inherently more robust than QN for navigating noisy landscapes — simpler dynamics = fewer ways to go wrong.

3. **Their single LMHE failure explosion (10→50 at 15pm) is the failure mode we avoid entirely.** We use the exact Hessian every step. Their ensemble mechanism solves a problem we don't have.

4. **Wall time is comparable.** Their Full Hessian: 5-30s. Ours: 8-18s. Same ballpark despite 1000 Euler steps vs ~25-50 PRFO steps. Each of their steps is more expensive but more efficient (second-order convergence).

5. **They test 0-15pm. We test 0-200pm.** Their "stressed" at 15pm is our "trivially close." We show the method works at 13x their noise range.

## Surprising Results from Our Experiments

1. **dt=0.005 beats dt=0.01** — smaller step avoids overshooting the narrow saddle region where n_neg transitions 2→1. Phase 1 said 0.01 was optimal at 100 steps, but with 1000 steps, 0.005 is strictly better.

2. **Adaptive dt hurts** — eigenvalue-clamped strategy makes steps too small in steep-curvature regions where large steps are needed. Clear negative result.

3. **NR-GAD ping-pong hurts** — pure Newton descent when n_neg≥2 overshoots. The step is too aggressive. Damped version (α=0.1-0.3) currently being tested.

4. **Basin of attraction is rock solid to 100pm** — 172 converged runs, 0 wrong TS. First wrong TS at 200pm. GAD is either right or gives up. No silent failures. This is a safety property worth highlighting.

5. **~13% of molecules never converge even at 0pm noise** — 40/300 fail at zero noise. Pathological Hessian spectra or near-degenerate eigenvalues. These need different approaches.

6. **Tight displacement clamping does literally nothing** — 0.1A vs 0.35A cap, <1pp difference. The default cap never triggers.

7. **Midpoint interpolation R→P works 29% of the time** — surprisingly viable as a starting geometry without knowing the TS.

## What We Need for the Paper

### Essential (must have)

| # | What | Why | Effort |
|---|------|-----|--------|
| 1 | **Sella baselines on our 300 T1x** | Direct comparison: QN vs Full Hessian vs GAD on same data | 1 day |
| 2 | **Graph isomorphism IRC on all converged** | Makes "intended" rate comparable to LMHE paper | 1 day |
| 3 | **Multiple noise seeds (10×)** | 95% CIs for credibility, matches their protocol | 1 day (parallel MIG) |
| 4 | **Full T1x (9,561 samples)** | Statistical power, publishable sample size | 6 min wall time |

### Strong (should have)

| # | What | Why | Effort |
|---|------|-----|--------|
| 5 | **GAD+ on Sella 240** | Cross-dataset: same benchmark they used | 1 day |
| 6 | **NewtonNet PES test** | PES-agnostic claim (not just "works on HIP") | 2-3 days (setup) |
| 7 | **Wall time / scaling analysis** | Efficiency story: breakdown by component | Half day |

### Bulletproof (nice to have)

| # | What | Why | Effort |
|---|------|-----|--------|
| 8 | **Third dataset** (Baker TS set or similar) | Generalization beyond organic chemistry | 1 week |
| 9 | **xTB or cheap DFT comparison** | Shows method isn't MLIP-limited | 1 week |
| 10 | **Per-molecule difficulty correlation** | Do same molecules fail for GAD+ and Sella? Complementary failure modes? | Half day |

### Specific baselines to run

- **Sella TS-BFGS (internal coords)** — their QN baseline, on our T1x data
- **Sella Full Hessian (internal coords)** — their gold standard, on our T1x data
- **Sella Full Hessian (Cartesian)** — tests internal vs Cartesian coordinate effect
- These are already in CLAUDE.md as planned baselines (Experiment 1 in the experiment plan)

### IRC upgrade needed

- Switch from RMSD threshold to **graph isomorphism** (Open Babel → VF2)
- Report in their 5 categories: intended / partially intended / unintended / no reaction / TS error
- Run on ALL converged TS, not just 10-30
- Need full intended rates at every noise level to make Figure 3a-style comparison plots

### Plots to match their format

- Intended count vs noise (their Figure 3a style) — needs full IRC
- Failure count vs noise (their Figure 3b) — we have this
- Wall time distribution violin/histogram (their Figure 4a)
- Step count distribution (analogous to their Figure 4b gradient evaluations)
- All with 95% CI from 10+ seeds

## Method Insights for Future Development

### What's working
- Eckart projection: non-negotiable, +68pp improvement
- Small fixed dt (0.005): simple, robust, best overall
- Direct analytical Hessian: cheap on MLIPs, eliminates eigenvector prediction failures
- Forward Euler: surprisingly effective when stepping carefully

### What's not working (yet)
- Adaptive dt: too conservative near saddle points
- NR-GAD ping-pong (undamped): overshoots
- Starting from equilibrium geometries: GAD needs saddle-region starts

### What to try next
- Damped NR-GAD (α=0.1-0.3): currently running, fix the overshoot
- Noise-adaptive dt scheduling: start dt=0.01 (traverse landscape), switch to dt=0.005 (fine control near saddle)
- Multiple restarts from different noise seeds: cheap way to push past 94%
- Internal coordinates: might help with the ~6% that fail even at low noise

## Data Inventory

```
Current results (all at /lustre07/scratch/memoozd/gadplus/runs/):
  noise_survey_300/     Phase 2: 300 samples × 9 noise, dt=0.01, 300 steps
  starting_geom_300/    Phase 3: 300 samples × 4 starts, dt=0.01, 300 steps
  basin_map/            Phase 6: 50 samples × 7 noise levels
  method_cmp_300/       Phase 7: 300 samples × 7 methods × 6 noise, 1000 steps
  irc_validation/       Phase 5: 10 samples IRC at 10pm
  targeted/             Running: damped NR-GAD, high step counts, randomized samples
  geodesic_mid/         Running: geodesic midpoint starting geometry

Reference:
  references/noisyTS.tex   Wu et al. LMHE paper
```
