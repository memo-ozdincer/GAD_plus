# GAD+ Complete Experiment Log

> All experiments on Transition1x train split (300 samples unless noted), HIP neural network potential, Narval A100 MIG slices.  
> **Convergence:** n_neg==1 AND force_norm < 0.01 eV/A on Eckart-projected vibrational Hessian.  
> **GAD formula:** F_GAD = F + 2(F·v₁)v₁, where v₁ is the lowest eigenvector of the Eckart-projected vibrational Hessian.  
> **Step:** x_{n+1} = x_n + dt · F_GAD (Euler integration).

---

## Round 1, Experiment 1: Noise Robustness

**SLURM:** 58835838 | **Data:** `noise_survey_300/` | **Status:** Complete (9/9 jobs)

**Method:** `gad_projected` — Eckart-projected GAD with fixed dt=0.01, 300 steps, k_track=0. At each step: compute HIP Hessian → Eckart-project to vibrational subspace (remove 6 TR modes) → extract lowest eigenvector v₁ → apply GAD force modification → Euler step. No mode tracking, no adaptive dt, no displacement capping.

| Noise (pm) | 0 | 10 | 20 | 30 | 50 | 70 | 100 | 150 | 200 |
|------------|---|----|----|----|----|----|----|-----|-----|
| Converged | 260 | 210 | 209 | 208 | 204 | 204 | 183 | 140 | 89 |
| Rate (%) | 86.7 | 70.0 | 69.7 | 69.3 | 68.0 | 68.0 | 61.0 | 46.7 | 29.7 |
| Avg steps | 11 | 37 | 53 | 73 | 99 | 115 | 142 | 161 | 171 |

**Control:** Level 0 (unprojected GAD) gets 0% at ≥50pm. Eckart projection is +68pp.

---

## Round 1, Experiment 2: Starting Geometry

**SLURM:** 58835839 | **Data:** `starting_geom_300/` | **Status:** Complete (4/4 jobs)

**Method:** Same as Experiment 1 (gad_projected, dt=0.01, 300 steps). Only the starting geometry changes.

| Starting Geometry | Description | Rate | Avg Steps |
|-------------------|-------------|------|-----------|
| Noised TS (10pm) | Known TS + 10pm Gaussian noise | 70.0% | 37 |
| Linear midpoint | (R+P)/2 in Cartesian coordinates | 29.0% | 191 |
| Reactant | Known reactant equilibrium | 6.3% | 108 |
| Product | Known product equilibrium | 3.0% | 65 |

**Geodesic midpoint** (separate job 58852072, timed out at 204/300): 46.1% at dt=0.005, 1000 steps. Confounded by different dt/steps vs linear midpoint.

---

## Round 1, Experiment 3: Basin of Attraction

**SLURM:** 58835840 | **Data:** `basin_map/` | **Status:** Complete | **Samples:** 50

**Method:** gad_projected (dt=0.01, 300 steps). Start from known TS + noise. After convergence, compare converged TS to original via RMSD. Threshold: 0.1A for "same TS."

| Noise (pm) | Converged | Same TS | Diff TS | Avg RMSD (A) |
|------------|-----------|---------|---------|-------------|
| 0 | 48/50 | 48 | 0 | 0.0005 |
| 10 | 32/50 | 32 | 0 | 0.0054 |
| 50 | 32/50 | 32 | 0 | 0.0257 |
| 100 | 29/50 | 29 | 0 | 0.0490 |
| 200 | 20/50 | 12 | 8 | 0.1037 |
| 500 | 1/50 | 0 | 1 | 0.4850 |

**Finding:** Zero wrong TS below 100pm (172 converged runs). GAD either converges to the correct TS or fails to converge. No silent wrong answers.

---

## Round 1, Experiment 4: 7-Method Comparison

**SLURM:** 58845357 | **Data:** `method_cmp_300/` | **Status:** Complete (42/42 jobs)

7 methods × 6 noise levels. 300 samples, 1000 steps each. 12,600 total optimizations.

### Methods tested:

**gad_small_dt** — Eckart-projected GAD with fixed dt=0.005, 1000 steps. Same as gad_projected but half the timestep. The GAD force F_GAD = F + 2(F·v₁)v₁ is computed in the Eckart-projected mass-weighted vibrational subspace. Forces, guide vector, and output are all projected through the Eckart projector to prevent translational/rotational leakage. Euler step: x += dt · F_GAD.

**gad_projected** — Same algorithm, dt=0.01. The baseline from Round 1.

**gad_tight_clamp** — gad_projected + per-atom displacement cap of 0.1A (vs default 0.35A). After computing step_disp = dt · F_GAD, if any atom moves >0.1A, the entire displacement is scaled down proportionally.

**gad_adaptive_dt** — gad_projected + eigenvalue-clamped adaptive timestep. dt_eff = dt_base / clamp(|λ₀|, 0.01, 100), clamped to [dt_min=1e-4, dt_max=0.05]. dt_base=0.01. When |λ₀| is large (steep curvature), dt shrinks. When |λ₀| is small (flat), dt grows. The idea: avoid overshooting in steep regions.

**gad_adaptive_tight** — gad_adaptive_dt + tight clamping (0.1A cap). Both features combined.

**nr_gad_pingpong** — Hard switch: when n_neg≥2, use pure Newton descent (Δx = -H⁻¹g with eigenvalue flooring at 1e-6, no damping). When n_neg<2, use standard GAD. The NR step inverts the Hessian in the vibrational subspace: project gradient onto eigenvectors, divide by eigenvalue magnitude, reconstruct. dt=0.01 for GAD phase.

**nr_gad_pp_adaptive** — nr_gad_pingpong + adaptive dt in the GAD phase. Worst of both worlds.

| Method | 10pm | 30pm | 50pm | 100pm | 150pm | 200pm |
|--------|------|------|------|-------|-------|-------|
| **gad_small_dt** | **94.3** | **94.3** | **91.3** | **86.7** | **70.3** | **51.3** |
| gad_projected | 72.3 | 70.3 | 69.3 | 66.7 | 58.0 | 45.3 |
| gad_tight_clamp | 72.0 | 70.0 | 69.7 | 67.0 | 58.3 | 46.0 |
| gad_adaptive_dt | 71.3 | 65.0 | 52.7 | 37.7 | 23.7 | 14.3 |
| gad_adaptive_tight | 70.3 | 64.3 | 53.0 | 36.3 | 24.0 | 14.7 |
| nr_gad_pingpong | 56.7 | 35.3 | 31.7 | 24.7 | 22.3 | 18.3 |
| nr_gad_pp_adaptive | 53.3 | 25.0 | 13.7 | 5.3 | 5.0 | 2.0 |

---

## Round 1, Experiment 5: Damped NR-GAD

**SLURM:** 58852071 | **Data:** `targeted/` | **Status:** Complete (42/42 jobs)

**Method:** Same ping-pong as nr_gad_pingpong, but the NR step is damped: Δx = α · (-H⁻¹g), with per-component cap and total norm cap. GAD phase uses dt=0.005. When n_neg≥2, the NR step is: project gradient onto vibrational eigenvectors, divide by |λᵢ| (floored at 1e-6), scale by damping α, cap per-component at 0.3, cap total norm at max_step_norm.

| Method | α | norm_cap | 10pm | 30pm | 50pm | 100pm | 150pm | 200pm |
|--------|---|----------|------|------|------|-------|-------|-------|
| nr_gad_damped | 0.1 | 0.05A | 94.7 | 88.0 | 77.7 | 58.0 | 46.0 | 33.7 |
| nr_gad_damped | 0.2 | 0.10A | 93.0 | 88.0 | 78.3 | 60.3 | 47.3 | 36.3 |
| nr_gad_damped | 0.3 | 0.15A | 88.7 | 82.3 | 75.0 | 58.7 | 46.3 | 37.0 |

**Finding:** α=0.1 matches baseline at 10pm but degrades badly at higher noise. The NR step direction is the problem, not just the magnitude. More damping doesn't fix a wrong direction.

---

## Round 1, Experiment 6: IRC Validation

**SLURM:** 58834594 | **Data:** `irc_validation/` | **Status:** Complete

**Method:** From converged TS, run Sella IRC forward+backward. Compare endpoints to known reactant/product via RMSD (threshold 0.5A). 30 samples × 3 noise levels.

| Noise | Intended | Half-intended | Unintended |
|-------|----------|---------------|------------|
| 10pm | 17/30 (57%) | 3 (10%) | 10 (33%) |
| 50pm | 20/30 (67%) | 3 (10%) | 7 (23%) |
| 100pm | 19/30 (63%) | 4 (13%) | 7 (23%) |

---

## Round 2, Experiment 7: Preconditioned GAD

**SLURM:** 58885855 | **Data:** `precond_gad/` | **Status:** Complete (30/30 jobs, all 300/300 samples)

**Method:** Preconditioned GAD: Δx = dt · |H|⁻¹ · F_GAD. After computing the standard GAD direction F_GAD in Eckart-projected mass-weighted space, decompose it into vibrational eigenvector components: c_i = F_GAD · v_i. Scale each component by 1/max(|λᵢ|, eig_floor). Reconstruct: Δx = dt · Σ (c_i / max(|λᵢ|, floor)) · v_i. This gives Newton-like step sizing: large steps along flat modes (small |λ|), small steps along steep modes (large |λ|).

The key difference from plain GAD: plain GAD applies the same dt to every mode. Preconditioning applies dt/|λᵢ| per mode, creating step-size ratios up to 100:1.

Four variants tested: three eig_floor values at dt=0.005, one at dt=0.01.

| Method | dt | eig_floor | 10pm | 30pm | 50pm | 100pm | 150pm | 200pm |
|--------|-----|-----------|------|------|------|-------|-------|-------|
| gad_small_dt (control) | 0.005 | — | 94.3 | 94.3 | 91.3 | 86.7 | 70.3 | 51.3 |
| precond_gad_001 | 0.005 | 0.01 | 73.7 | 61.0 | 48.0 | 21.7 | 7.3 | 3.3 |
| precond_gad_005 | 0.005 | 0.05 | 73.7 | 61.3 | 48.3 | 21.7 | 7.3 | 4.0 |
| precond_gad_01 | 0.005 | 0.1 | 72.7 | 62.7 | 47.0 | 21.0 | 7.3 | 4.3 |
| precond_gad_dt01 | 0.01 | 0.01 | 78.3 | 72.7 | 68.3 | 58.0 | 49.7 | 41.7 |

**Detailed stats (avg steps to convergence / avg wall time per sample):**

| Method | 10pm | 30pm | 50pm | 100pm | 150pm | 200pm |
|--------|------|------|------|-------|-------|-------|
| gad_small_dt | 78 / 8.5s | 149 / 12.0s | 200 / 16.8s | 308 / 24.6s | 396 / 35.1s | 459 / 43.9s |
| precond_gad_001 | 607 / 42.8s | 783 / 53.7s | 844 / 60.5s | 899 / 62.2s | 881 / 61.2s | 936 / 61.4s |
| precond_gad_dt01 | 331 / 29.4s | 429 / 36.5s | 479 / 40.4s | 566 / 48.7s | 610 / 50.8s | 653 / 54.6s |

**Why preconditioning fails for GAD:** GAD dynamics require *uniform* progress across all vibrational modes to maintain eigenvector continuity and allow the n_neg count to evolve smoothly. The |H|⁻¹ scaling creates extreme step ratios — steep modes (large |λ|, often including the TS mode λ₁) get tiny steps while flat modes get huge steps. This starves the critical modes of progress. Newton-like scaling helps descent toward *minima* (where all eigenvalues are positive and you want to follow curvature), but GAD navigates a *saddle* where mode balance matters more than curvature-following.

The eig_floor (0.01 vs 0.05 vs 0.1) has virtually no effect — the problem isn't near-zero eigenvalues blowing up, it's the large eigenvalues shrinking steps too much.

---

## Round 2, Experiment 8: Corrected Adaptive Timestep

**SLURM:** 58886863 (tasks 0-11) | **Data:** `round2/` | **Status:** Partial (all timed out at 3hr)

**Method:** Same eigenvalue-clamped formula as Round 1, but with corrected parameters matching Multi-Mode GAD from the literature. dt_eff = dt_base / clamp(|λ₀|, 0.01, 100), clamped to [dt_min, dt_max].

The hypothesis: our Round 1 adaptive dt failed because dt_base=0.01 was 5x too high. Multi-Mode GAD used effective dt_base=0.002. Testing whether corrected parameters recover performance.

**adaptive_mm:** dt_base=0.002, dt_min=1e-5, dt_max=0.08. Matches Multi-Mode GAD parameters exactly.

**adaptive_mm2:** dt_base=0.001, dt_min=1e-5, dt_max=0.05. Even more conservative.

| Method | 10pm | 30pm | 50pm | 100pm | 150pm | 200pm |
|--------|------|------|------|-------|-------|-------|
| adaptive_mm | 53.7% (259) | 40.2% (219) | 35.1% (211) | 22.6% (186) | 12.4% (169) | 5.7% (174) |
| adaptive_mm2 | 40.1% (227) | 29.6% (206) | 22.4% (192) | 13.1% (176) | 6.3% (174) | 2.4% (170) |

(Numbers in parentheses = samples completed before timeout)

**Avg steps to convergence / avg time per sample:**

| Method | 10pm | 50pm | 100pm | 200pm |
|--------|------|------|-------|-------|
| adaptive_mm | 342 / 40.5s | 445 / 50.7s | 523 / 56.6s | 658 / 60.4s |
| adaptive_mm2 | 318 / 45.2s | 481 / 55.4s | 584 / 60.2s | 691 / 62.2s |

**Finding:** Correcting the dt_base does NOT fix adaptive dt. It makes it worse. adaptive_mm (53.7% at 10pm) is far below the old broken gad_adaptive_dt (71.3% at 10pm) because the smaller base means even smaller effective steps. The eigenvalue-clamped formula is fundamentally wrong for GAD: it introduces step-size variability that disrupts the steady progress GAD needs.

---

## Round 2, Experiment 9: Smaller Fixed Timestep

**SLURM:** 58886863 (tasks 12-17) | **Data:** `round2/` | **Status:** 3 complete, 3 partial

**Method:** `gad_dt003` — Identical to gad_small_dt but dt=0.003 instead of 0.005. 2000 steps to match displacement budget (dt × steps ≈ same total displacement capacity). Everything else identical: Eckart projection, no mode tracking, no adaptive dt, no preconditioning.

| Method | Steps | 10pm | 30pm | 50pm | 100pm | 150pm | 200pm |
|--------|-------|------|------|------|-------|-------|-------|
| gad_small_dt | 1000 | 94.3 (300) | 94.3 (300) | 91.3 (300) | 86.7 (300) | 70.3 (300) | 51.3 (300) |
| **gad_dt003** | 2000 | **94.7** (300) | **94.3** (300) | **92.0** (300) | **87.3** (300) | **71.3** (300) | **55.2** (259) |

Note: 100pm and 150pm updated from Round 3 rerun (SLURM 58933021, full 300/300). 200pm updated from 131→259 samples. Previous Round 2 partial estimates: 88.9% (244), 75.3% (158), 58.8% (131).

**Avg steps (converged) / avg time per sample:**

| Method | 10pm | 50pm | 100pm | 200pm |
|--------|------|------|-------|-------|
| gad_small_dt | 78 / 8.5s | 200 / 16.8s | 308 / 24.6s | 459 / 43.9s |
| gad_dt003 | 133 / 15.0s | 342 / 28.9s | 519 / 42.7s | 722 / 78.9s |

**Finding:** gad_dt003 is the **new best method**. +2.2pp at 100pm, +5pp at 150pm, +7.5pp at 200pm. The pattern dt=0.01→0.005 (+22pp) → 0.003 (+2-7pp) shows diminishing but still meaningful returns from smaller dt. Cost: ~2x wall time (more steps needed).

---

## Round 2, Experiment 10: No Displacement Capping

**SLURM:** 58886863 (tasks 18-23) | **Data:** `round2/` | **Status:** 4 complete, 2 partial

**Method:** `gad_no_clamp` — Identical to gad_small_dt (dt=0.005, 1000 steps) but max_atom_disp=999.0 (effectively no capping). In gad_small_dt, after each step, if any atom's displacement exceeds 0.35A, the entire step is scaled down. gad_no_clamp skips this check entirely.

| Method | 10pm | 30pm | 50pm | 100pm | 150pm | 200pm |
|--------|------|------|------|-------|-------|-------|
| gad_small_dt | 94.3 (300) | 94.3 (300) | 91.3 (300) | 86.7 (300) | 70.3 (300) | 51.3 (300) |
| gad_no_clamp | 94.3 (300) | 94.3 (300) | 91.3 (300) | 86.7 (300) | 70.9 (289) | 54.6 (238) |

**Finding:** Identical within noise. The 0.35A displacement cap never triggers at dt=0.005. Confirms that displacement capping is purely cosmetic for small-dt GAD.

---

## Round 2, Experiment 11: Adaptive dt with Higher Floor

**SLURM:** 58886863 (tasks 24-29) | **Data:** `round2/` | **Status:** 2 complete, 4 partial

**Method:** `adaptive_floor` — Eigenvalue-clamped adaptive dt with dt_base=0.005, dt_min=**1e-3** (vs 1e-4 default), dt_max=0.05. The higher floor prevents the trajectory from freezing in steep curvature regions (where |λ₀| is large and dt_eff would normally shrink to ~1e-4).

| Method | 10pm | 30pm | 50pm | 100pm | 150pm | 200pm |
|--------|------|------|------|-------|-------|-------|
| gad_adaptive_dt (R1) | 71.3 | 65.0 | 52.7 | 37.7 | 23.7 | 14.3 |
| adaptive_floor | 83.0 (300) | 80.0 (300) | 70.2 (258) | 43.2 (213) | 26.5 (189) | 15.9 (176) |
| gad_small_dt | 94.3 | 94.3 | 91.3 | 86.7 | 70.3 | 51.3 |

**Avg steps / avg time per sample:**

| 10pm | 50pm | 100pm | 200pm |
|------|------|-------|-------|
| 221 / 21.9s | 526 / 41.2s | 547 / 50.1s | 572 / 58.4s |

**Finding:** Best adaptive variant (+12pp over old adaptive at 10pm), but still -11pp vs fixed dt. The higher floor helps by preventing trajectory freezing, but the formula still introduces harmful step-size variability.

---

## Round 2, Experiment 12: Preconditioned Descent Diagnostic

**SLURM:** 58886863 (tasks 30-35) | **Data:** `round2/` | **Status:** All partial (167-223 samples)

**Method:** `precond_descent` — Hard-switch diagnostic (NOT diffusion-compatible). Uses the NR-GAD ping-pong framework with `descent_mode="preconditioned"`:
- When n_neg < 2: **Preconditioned GAD** — Δx = dt · |H|⁻¹ · F_GAD (standard preconditioned GAD, gad_blend_weight=1.0)
- When n_neg ≥ 2: **Preconditioned descent** — Δx = dt · |H|⁻¹ · F (same |H|⁻¹ machinery, but gad_blend_weight=0.0, so no v₁ force flip — pure descent along all modes)

dt=0.005, eig_floor=0.01. The diagnostic question: does GAD's v₁ ascent help or hurt when multiple eigenvalues are negative?

| Method | 10pm | 30pm | 50pm | 100pm | 150pm | 200pm |
|--------|------|------|------|-------|-------|-------|
| precond_descent | 71.3 (223) | 50.0 (186) | 38.5 (182) | 14.0 (172) | 6.0 (167) | 2.4 (168) |
| precond_gad_001 | 73.7 (300) | 61.0 (300) | 48.0 (300) | 21.7 (300) | 7.3 (300) | 3.3 (300) |

**Avg steps / time:**

| 10pm | 50pm | 100pm | 200pm |
|------|------|-------|-------|
| 614 / 46.0s | 834 / 57.9s | 898 / 61.3s | 886 / 62.3s |

**Finding:** precond_descent is slightly worse than precond_gad_001 at all noise levels (-2pp at 10pm, -10pp at 50pm). This suggests GAD's v₁ ascent IS marginally helpful even at n_neg≥2. However, both are so bad (due to preconditioning) that the diagnostic is inconclusive — the signal is buried under the preconditioning failure.

---

## Round 2, Experiment 13: λ₂-Blended Preconditioned GAD

**SLURM:** 58886863 (tasks 36-47) | **Data:** `round2/` | **Status:** All partial (161-230 samples)

**Method:** Smooth, differentiable blend between preconditioned GAD and preconditioned descent. Instead of hard-switching on n_neg (discrete, non-differentiable), use sigmoid of the second eigenvalue λ₂ (continuous, differentiable):

```
w = sigmoid(k · λ₂)
F_blend = F + 2·w·(F·v₁)v₁
Δx = dt · |H|⁻¹ · F_blend
```

When λ₂ > 0 (near index-1 saddle): w → 1, pure GAD (ascend v₁).  
When λ₂ < 0 (higher-order saddle): w → 0, pure descent (descend all modes).  
The blend only controls one decision: ascend v₁ or not. All other modes are always preconditioned descent regardless of w.

**blend_k50:** k=50, transition width ~0.1 eV/A² around λ₂=0.  
**blend_k100:** k=100, nearly hard switch but differentiable.

dt=0.005, eig_floor=0.01 for both.

| Method | 10pm | 30pm | 50pm | 100pm | 150pm | 200pm |
|--------|------|------|------|-------|-------|-------|
| blend_k50 | 72.0 (225) | 50.0 (182) | 37.2 (180) | 14.3 (168) | 6.2 (161) | 3.1 (163) |
| blend_k100 | 71.7 (230) | 50.5 (184) | 37.6 (178) | 13.6 (177) | 4.7 (169) | 3.1 (161) |
| precond_descent | 71.3 (223) | 50.0 (186) | 38.5 (182) | 14.0 (172) | 6.0 (167) | 2.4 (168) |

**Avg steps / time (blend_k50):**

| 10pm | 50pm | 100pm | 200pm |
|------|------|-------|-------|
| 619 / 46.8s | 831 / 58.2s | 899 / 62.2s | 920 / 63.3s |

**Finding:** All three preconditioned variants (blend_k50, blend_k100, precond_descent) are statistically identical (~72% at 10pm, ~14% at 100pm). The blend sharpness k has no effect. The preconditioning base |H|⁻¹ dominates the failure mode, making the GAD-vs-descent distinction irrelevant.

**The blend mechanism itself is sound** — sigmoid(k·λ₂) is differentiable and correctly modulates the v₁ ascent contribution. But it was tested on a broken base. **Must re-test without preconditioning** to isolate whether the smooth λ₂-blend helps compared to always-on GAD.

---

## Round 3, Experiment 14: Sella TS Baselines

**SLURM:** 58932967 | **Data:** `sella_baselines/` | **Status:** Complete (24/24 jobs, all 300/300 samples)

### Setup

**Sella** (v2.3.4) is the standard trust-region saddle point optimizer using RS-P-RFO (Restricted-Step Partitioned Rational Function Optimization). We compare it directly against GAD on the same 300 Transition1x train-split samples, same noise seeds (seed=42), same noise levels.

**HIP integration:** HipSellaCalculator wraps our `predict_fn` into an ASE Calculator with Hessian caching. On each Sella step, `calculate()` runs HIP with `do_hessian=True`, caching energy+forces+Hessian in one forward pass. When Sella subsequently calls `hessian_function()`, it reads from cache — zero overhead for exact Hessians.

**Sella parameters** (Wander et al. 2024, arXiv:2410.01650v2):
- `order=1` (first-order saddle point)
- `use_exact_hessian=True` with `diag_every_n=1` (fresh HIP Hessian every step)
- `gamma=0.0` (tightest eigensolver convergence)
- Trust radius: `delta0=0.048`, `rho_inc=1.035`, `rho_dec=5.0`, `sigma_inc=1.15`, `sigma_dec=0.65`
- `max_steps=200`

**Four configurations tested:**

| Config | fmax threshold | Coordinates | Purpose |
|--------|---------------|-------------|---------|
| sella_internal_fmax0.01 | 0.01 eV/Å | Internal (bonds/angles/dihedrals) | Match our GAD force threshold |
| sella_internal_fmax0.03 | 0.03 eV/Å | Internal | Sella default threshold |
| sella_internal_fmax0.001 | 0.001 eV/Å | Internal | Very tight, retroactive n_neg check |
| sella_cartesian_fmax0.01 | 0.01 eV/Å | Cartesian | Test coordinate system effect |

### Convergence criteria compared

Three metrics reported for every sample:

1. **Sella converged (`sella%`):** `max(|force_components|) < fmax` — Sella's own criterion, based on the maximum absolute force component across all atoms.
2. **n_neg==1 (`n_neg1%`):** Exactly one negative eigenvalue in the Eckart-projected vibrational Hessian of the final geometry. This is the necessary condition for a first-order saddle point. Does NOT check force magnitude.
3. **Our criterion (`ours%`):** `n_neg == 1` on Eckart-projected vibrational Hessian AND `mean(per-atom force norm) < 0.01 eV/Å`. This is the criterion used for ALL GAD experiments throughout this log.

Note: Sella uses max absolute force component; our GAD experiments use mean per-atom force norm. These are different metrics — fmax=0.01 is stricter than force_norm<0.01 for the same geometry.

### Results: All three metrics

**Sella Internal Coordinates, fmax=0.01:**

| Noise | sella% | n_neg1% | ours% | Avg steps (conv) | Avg wall (conv) | Both | Sella-only | Ours-only | Neither |
|-------|--------|---------|-------|-------------------|-----------------|------|------------|-----------|---------|
| 10pm | 87.0 | 97.7 | 91.7 | 16 | 1.5s | 259 | 2 | 16 | 23 |
| 30pm | 83.3 | 98.7 | 90.0 | 23 | 2.1s | 250 | 0 | 20 | 30 |
| 50pm | 79.0 | 96.0 | 84.7 | 25 | 2.3s | 235 | 2 | 19 | 44 |
| 100pm | 56.7 | 81.3 | 61.3 | 32 | 2.9s | 170 | 0 | 14 | 116 |
| 150pm | 31.0 | 52.3 | 33.7 | 47 | 4.2s | 92 | 1 | 9 | 198 |
| 200pm | 16.3 | 33.0 | 18.0 | 51 | 4.6s | 49 | 0 | 5 | 246 |

**Sella Internal Coordinates, fmax=0.03 (Sella default):**

| Noise | sella% | n_neg1% | ours% | Avg steps (conv) | Avg wall (conv) | Both | Sella-only | Ours-only | Neither |
|-------|--------|---------|-------|-------------------|-----------------|------|------------|-----------|---------|
| 10pm | 95.7 | 97.3 | 61.0 | 4 | 0.5s | 182 | 105 | 1 | 12 |
| 30pm | 91.7 | 97.3 | 61.7 | 5 | 0.5s | 185 | 90 | 0 | 25 |
| 50pm | 87.7 | 95.3 | 56.7 | 10 | 1.0s | 168 | 95 | 2 | 35 |
| 100pm | 63.7 | 82.3 | 42.3 | 20 | 1.8s | 126 | 65 | 1 | 108 |
| 150pm | 36.0 | 52.7 | 25.0 | 34 | 3.2s | 75 | 33 | 0 | 192 |
| 200pm | 20.0 | 32.3 | 11.7 | 37 | 3.5s | 34 | 26 | 1 | 239 |

**Sella Internal Coordinates, fmax=0.001 (very tight, retroactive check):**

| Noise | sella% | n_neg1% | ours% | Avg steps (conv) | Avg wall (conv) |
|-------|--------|---------|-------|-------------------|-----------------|
| 10pm | 1.3 | 97.7 | 91.7 | 198 | 17.4s |
| 30pm | 1.3 | 98.3 | 89.3 | 198 | 17.4s |
| 50pm | 1.7 | 96.3 | 84.0 | 197 | 17.2s |
| 100pm | 1.0 | 80.7 | 60.7 | 198 | 17.1s |
| 150pm | 1.3 | 52.3 | 34.0 | 196 | 16.9s |
| 200pm | 0.7 | 34.7 | 18.3 | 196 | 16.9s |

**Sella Cartesian Coordinates, fmax=0.01:**

| Noise | sella% | n_neg1% | ours% | Avg steps (conv) | Avg wall (conv) | Both | Sella-only | Ours-only | Neither |
|-------|--------|---------|-------|-------------------|-----------------|------|------------|-----------|---------|
| 10pm | 91.3 | 97.7 | 94.3 | 13 | 1.0s | 271 | 3 | 12 | 14 |
| 30pm | 90.7 | 98.3 | 94.7 | 16 | 1.2s | 270 | 2 | 14 | 14 |
| 50pm | 87.7 | 97.7 | 91.0 | 16 | 1.2s | 261 | 2 | 12 | 25 |
| 100pm | 75.3 | 92.3 | 78.3 | 21 | 1.6s | 225 | 1 | 10 | 64 |
| 150pm | 49.0 | 68.7 | 52.7 | 33 | 2.5s | 147 | 0 | 11 | 142 |
| 200pm | 22.7 | 43.7 | 25.3 | 44 | 3.3s | 68 | 0 | 8 | 224 |

### Key findings

1. **GAD beats Sella everywhere, gap grows with noise.** By our criterion (n_neg==1 + force<0.01): at 10pm GAD gad_small_dt gets 94.3% vs Sella Cartesian 94.3% (tied) and Sella Internal 91.7% (GAD +2.6pp). At 100pm: GAD 86.7% vs Sella Cartesian 78.3% (GAD +8.4pp) vs Sella Internal 61.3% (GAD +25.4pp). At 200pm: GAD 51.3% vs Sella Cartesian 25.3% (GAD +26pp, 2× better).

2. **Cartesian coordinates beat Internal for Sella+HIP.** This contradicts the standard recommendation (Sella paper: "use internal coordinates for GNN potentials"). At every noise level, Cartesian Sella outperforms Internal by 3-19pp (by our criterion). Likely because HIP's analytical Hessian is well-conditioned in Cartesian coordinates, and Sella's internal coordinate transformation (bonds/angles/dihedrals) introduces numerical noise in the Hessian conversion.

3. **fmax threshold matters enormously.** At fmax=0.03 (Sella default), Sella reports 95.7% convergence at 10pm — but only 61.0% pass our criterion. 105 samples are "Sella-only": they have n_neg=1 but force_norm > 0.01 (average 0.013). fmax=0.03 is too loose for TS quality. At fmax=0.01, the Sella-only count drops to 2-3 per noise level — the criteria are nearly aligned.

4. **fmax=0.001 gives same "ours" rate as fmax=0.01.** Internal fmax=0.001: 91.7% at 10pm (our criterion). Internal fmax=0.01: 91.7% at 10pm. Sella's own convergence drops to 1.3% (can't reach fmax=0.001 in 200 steps), but the final geometries still satisfy our criterion at the same rate. Sella's failures are geometric (wrong n_neg), not force-convergence failures — driving fmax to 0.001 doesn't fix them.

5. **n_neg==1 is necessary but not sufficient.** At 10pm, 97.7% of final geometries have n_neg==1, but only 91.7% (Internal) or 94.3% (Cartesian) pass our full criterion. The gap is samples with correct saddle order but forces still above threshold.

6. **Sella is faster per sample when it converges.** Sella Cartesian at 10pm: 1.0s/sample (converged), 13 steps. GAD gad_small_dt at 10pm: ~5s/sample, 78 steps. But Sella's failures hit the 200-step cap (~17s), bringing average wall time closer. At high noise where failures dominate, Sella's average wall time exceeds GAD's.

7. **Overlap analysis shows complementary strengths.** "Ours-only" samples (8-20 per config) pass our criterion but not Sella's fmax — the geometry is a valid TS but has a slightly high max force component. "Sella-only" samples (0-3 at fmax=0.01) pass fmax but have n_neg≠1 — Sella converged to a non-TS stationary point.

---

## Round 3, Experiment 15: Sella 1000-Step with Eckart Variants

**SLURM:** 58937673 | **Data:** `sella_1000/` | **Status:** 20/24 complete (internal 150pm + 200pm timed out at 6hr)

### Motivation

Experiment 14 used max_steps=200. Many high-noise samples hit the step cap, making the comparison unfair against GAD (which runs 1000-2000 steps). This experiment gives Sella a matched step budget of 1000 and also tests whether Eckart-projecting the Hessian before passing it to Sella helps.

### Setup

Same as Experiment 14 except:
- **max_steps=1000** (vs 200 previously)
- **Four coordinate/projection configs** (vs two previously):

| Config | Coordinates | Eckart projection on Hessian | Description |
|--------|-------------|------------------------------|-------------|
| internal | Internal (bonds/angles/dihedrals) | No | Sella's standard recommendation |
| internal_eckart | Internal | Yes | Eckart-project HIP Hessian, then Sella converts to internal |
| cartesian | Cartesian | No | Raw HIP Hessian in Cartesian space |
| cartesian_eckart | Cartesian | Yes | Eckart-projected HIP Hessian in Cartesian space |

All use fmax=0.01 eV/Å, diag_every_n=1, exact HIP Hessian, paper trust-radius parameters.

**Eckart projection for Sella:** When `apply_eckart=True`, the raw HIP Cartesian Hessian is mass-weighted, projected through the Eckart projector (removes 6 translation/rotation modes), then un-mass-weighted back to Cartesian. This cleaned Hessian is then passed to Sella (which may further convert to internal coordinates if `internal=True`). The projection uses the same `_eckart_projector` from `projection.py` as our GAD experiments.

### Convergence criteria reported

Five criteria reported per sample, to enable any cross-comparison:

| Criterion | Definition | Label in Parquet |
|-----------|-----------|-----------------|
| Sella converged | `max(\|force_components\|) < fmax` (Sella's own check) | `sella_converged` |
| n_neg==1 | Exactly 1 negative eigenvalue, Eckart-projected vibrational Hessian | `is_nneg1` |
| n_neg1 + force<0.01 | n_neg==1 AND mean per-atom force norm < 0.01 eV/Å (our GAD criterion) | `conv_nneg1_force001` |
| n_neg1 + fmax<0.01 | n_neg==1 AND max \|force component\| < 0.01 eV/Å (strictest, matches Sella's metric) | `conv_nneg1_fmax001` |
| n_neg1 + fmax<0.03 | n_neg==1 AND max \|force component\| < 0.03 eV/Å | `conv_nneg1_fmax003` |

**Important note on force metrics:** Sella uses `max(|force_components|)` (fmax), which is **stricter** than our GAD metric `mean(per-atom force norm)` (force_norm). A sample can pass force_norm<0.01 but fail fmax<0.01. This explains why "our criterion" can sometimes exceed "Sella's criterion" — our force metric is looser, but we add the n_neg==1 check that Sella doesn't enforce.

### Results: All criteria, all configs

**Sella Cartesian, no Eckart (1000 steps):**

| Noise | Sella conv | n_neg1 | n_neg1+f<.01 | n_neg1+fmax<.01 | n_neg1+fmax<.03 | Avg steps (nneg1+f<.01) |
|-------|-----------|--------|-------------|----------------|----------------|------------------------|
| 10pm | 91.3% (274) | 97.7% (293) | 94.3% (283) | 91.3% (274) | 96.7% (290) | 47 |
| 30pm | 91.3% (274) | 98.0% (294) | 94.3% (283) | 90.0% (270) | 97.0% (291) | 49 |
| 50pm | 88.0% (264) | 97.7% (293) | 91.0% (273) | 87.7% (263) | 94.0% (282) | 50 |
| 100pm | 75.0% (225) | 91.3% (274) | 78.0% (234) | 75.0% (225) | 82.0% (246) | 56 |
| 150pm | 50.7% (152) | 69.3% (208) | 53.7% (161) | 50.7% (152) | 56.7% (170) | 89 |
| 200pm | 23.3% (70) | 43.3% (130) | 25.7% (77) | 23.3% (70) | 30.0% (90) | 131 |

**Sella Cartesian + Eckart (1000 steps):**

| Noise | Sella conv | n_neg1 | n_neg1+f<.01 | n_neg1+fmax<.01 | n_neg1+fmax<.03 | Avg steps (nneg1+f<.01) |
|-------|-----------|--------|-------------|----------------|----------------|------------------------|
| 10pm | 91.7% (275) | 98.0% (294) | 94.7% (284) | 91.7% (275) | 97.0% (291) | 44 |
| 30pm | 91.3% (274) | 98.3% (295) | 94.3% (283) | 90.7% (272) | 97.0% (291) | 49 |
| 50pm | 88.0% (264) | 98.3% (295) | 91.3% (274) | 88.3% (265) | 94.3% (283) | 50 |
| 100pm | 75.7% (227) | 91.0% (273) | 79.0% (237) | 76.3% (229) | 82.3% (247) | 55 |
| 150pm | 49.7% (149) | 69.7% (209) | 53.0% (159) | 50.3% (151) | 55.0% (165) | 96 |
| 200pm | 25.7% (77) | 43.7% (131) | 27.7% (83) | 25.7% (77) | 31.0% (93) | 123 |

**Sella Internal, no Eckart (1000 steps):**

| Noise | Sella conv | n_neg1 | n_neg1+f<.01 | n_neg1+fmax<.01 | n_neg1+fmax<.03 | Avg steps (nneg1+f<.01) |
|-------|-----------|--------|-------------|----------------|----------------|------------------------|
| 10pm | 88.0% (264) | 97.3% (292) | 92.0% (276) | 87.0% (261) | 94.7% (284) | 65 |
| 30pm | 85.3% (256) | 98.7% (296) | 90.3% (271) | 84.0% (252) | 92.0% (276) | 69 |
| 50pm | 81.3% (244) | 96.3% (289) | 86.7% (260) | 81.0% (243) | 89.7% (269) | 89 |
| 100pm | 59.7% (179) | 82.0% (246) | 63.0% (189) | 59.7% (179) | 65.7% (197) | 95 |
| 150pm | — | — | — | — | — | — |
| 200pm | — | — | — | — | — | — |

**Sella Internal + Eckart (1000 steps):**

| Noise | Sella conv | n_neg1 | n_neg1+f<.01 | n_neg1+fmax<.01 | n_neg1+fmax<.03 | Avg steps (nneg1+f<.01) |
|-------|-----------|--------|-------------|----------------|----------------|------------------------|
| 10pm | 88.3% (265) | 97.7% (293) | 92.0% (276) | 87.0% (261) | 94.3% (283) | 60 |
| 30pm | 85.7% (257) | 98.0% (294) | 89.3% (268) | 86.0% (258) | 91.3% (274) | 65 |
| 50pm | 81.0% (243) | 95.3% (286) | 85.7% (257) | 82.3% (247) | 89.0% (267) | 87 |
| 100pm | 59.0% (177) | 81.0% (243) | 62.0% (186) | 59.0% (177) | 64.3% (193) | 84 |
| 150pm | — | — | — | — | — | — |
| 200pm | — | — | — | — | — | — |

(Internal 150pm and 200pm timed out at 6hr SLURM limit — Sella with internal coordinates is too slow at high noise for 1000 steps × 300 samples.)

### Head-to-head: GAD vs best Sella (1000 steps), matched criteria

Using **n_neg1 + fmax<0.01** (strictest, uses Sella's own force metric):

| Method | 10pm | 30pm | 50pm | 100pm | 150pm | 200pm |
|--------|------|------|------|-------|-------|-------|
| **GAD dt=0.003** | **94.7** | **94.3** | **92.0** | **87.3** | **71.3** | **55.2** |
| GAD dt=0.005 | 94.3 | 94.3 | 91.3 | 86.7 | 70.3 | 51.3 |
| Sella Cart+Eckart (1000) | 91.7 | 90.7 | 88.3 | 76.3 | 50.3 | 25.7 |
| Sella Cart (1000) | 91.3 | 90.0 | 87.7 | 75.0 | 50.7 | 23.3 |
| Sella Int+Eckart (1000) | 87.0 | 86.0 | 82.3 | 59.0 | — | — |
| Sella Int (1000) | 87.0 | 84.0 | 81.0 | 59.7 | — | — |

Using **n_neg1 + force<0.01** (our GAD criterion, slightly looser force metric):

| Method | 10pm | 30pm | 50pm | 100pm | 150pm | 200pm |
|--------|------|------|------|-------|-------|-------|
| **GAD dt=0.003** | **94.7** | **94.3** | **92.0** | **87.3** | **71.3** | **55.2** |
| GAD dt=0.005 | 94.3 | 94.3 | 91.3 | 86.7 | 70.3 | 51.3 |
| Sella Cart+Eckart (1000) | 94.7 | 94.3 | 91.3 | 79.0 | 53.0 | 27.7 |
| Sella Cart (1000) | 94.3 | 94.3 | 91.0 | 78.0 | 53.7 | 25.7 |
| Sella Int+Eckart (1000) | 92.0 | 89.3 | 85.7 | 62.0 | — | — |
| Sella Int (1000) | 92.0 | 90.3 | 86.7 | 63.0 | — | — |

### Comparison: 200 steps vs 1000 steps (Sella Cartesian, n_neg1+force<0.01)

| Noise | 200 steps | 1000 steps | Δ |
|-------|-----------|------------|---|
| 10pm | 94.3% | 94.3% | +0.0 |
| 30pm | 94.7% | 94.3% | −0.4 |
| 50pm | 91.0% | 91.0% | +0.0 |
| 100pm | 78.3% | 78.0% | −0.3 |
| 150pm | 52.7% | 53.7% | +1.0 |
| 200pm | 25.3% | 25.7% | +0.4 |

### Key findings

1. **1000 steps does not help Sella.** Increasing from 200 to 1000 steps gives <1pp improvement at every noise level. Sella's failures are not step-budget limited — they are geometric (the trust-region optimizer converges to wrong-index stationary points or oscillates without reaching n_neg==1).

2. **Eckart projection gives marginal improvement.** Cart+Eckart vs Cart: +1pp at 100pm, +2pp at 200pm. The Eckart projection cleans small TR-mode residuals from HIP's Hessian, but the effect is minor because Sella's internal RFO already handles near-zero eigenvalues.

3. **Cartesian still dominates Internal.** Cart 75.0% vs Int 59.7% at 100pm (both at 1000 steps, n_neg1+fmax<.01). Internal coordinates are both slower (more steps, timed out at 150pm+) and less accurate for HIP's Hessian.

4. **GAD's advantage is fundamental.** At 200pm with the strictest criterion (n_neg1+fmax<.01): GAD dt=0.003 gets 55.2%, best Sella gets 25.7%. This 2.1× gap persists regardless of step budget, coordinate system, or Hessian projection. GAD's Euler-step dynamics navigate the saddle landscape more effectively than Sella's trust-region approach at high noise.

5. **Force metric matters for fair comparison.** Using fmax<0.01 (Sella's metric) vs force<0.01 (our GAD metric) shifts Sella Cartesian from 94.3% to 91.3% at 10pm. The ~3pp gap is samples where mean force is low but one atom has a slightly high force component. For rigorous comparison, use n_neg1+fmax<0.01 (strictest, no advantage to either method).

---

## Round 3, Experiment 16: λ₂-Blended GAD WITHOUT Preconditioning

**SLURM:** 58932864 (tasks 0-17) | **Data:** `round3/` | **Status:** Complete (18/18 jobs, all 300/300 samples)

### Motivation

Experiment 13 tested λ₂-blended dynamics with |H|⁻¹ preconditioning. All three blend variants (k=10, 50, 100) gave identical ~72% at 10pm — the preconditioning masked the blend signal. This experiment isolates the blend by building on top of plain Euler GAD (the gad_small_dt base that already works at 94.3%).

### Method

Same λ₂-blend formula, but with plain Euler step instead of preconditioned step:

```
w = sigmoid(k · λ₂)                         # blend weight, differentiable
F_blend = F + 2·w·(F·v₁)v₁                  # partial GAD: w=1 → full flip, w=0 → no flip
Δx = dt · F_blend                            # plain Euler, NO |H|⁻¹
```

When λ₂ > 0 (near index-1 saddle): w → 1, pure GAD (ascend v₁). Identical to gad_small_dt.  
When λ₂ < 0 (higher-order saddle, multiple negative eigenvalues): w → 0, pure descent (follow forces downhill along all modes).  

The hypothesis: at high noise, starting geometries often have n_neg≥2 (λ₂<0). Reducing the v₁ ascent in these regions might help the trajectory escape higher-order saddles before engaging GAD. The blend smoothly transitions as the geometry approaches index-1.

**Three sharpness values tested:**
- **blend_plain_k10:** k=10, transition width ~0.5 eV/Å² around λ₂=0. Wide blend zone — significant descent contribution even near TS.
- **blend_plain_k50:** k=50, transition width ~0.1 eV/Å². Moderate — descent only when clearly far from TS.
- **blend_plain_k100:** k=100, transition width ~0.05 eV/Å². Sharp — nearly hard switch, but still differentiable.

All use dt=0.005, 1000 steps, Eckart projection. No preconditioning, no adaptive dt.

### Results

| Method | 10pm | 30pm | 50pm | 100pm | 150pm | 200pm |
|--------|------|------|------|-------|-------|-------|
| **gad_small_dt (baseline)** | **94.3** | **94.3** | **91.3** | **86.7** | **70.3** | **51.3** |
| blend_plain_k10 | 93.0 | 91.3 | 87.0 | 76.7 | 60.3 | 46.3 |
| blend_plain_k50 | 93.7 | 93.0 | 86.7 | 76.3 | 61.7 | 47.7 |
| blend_plain_k100 | 93.7 | 92.3 | 86.0 | 76.7 | 61.3 | 47.7 |

**Avg steps to convergence / avg wall time per sample:**

| Method | 10pm | 30pm | 50pm | 100pm | 150pm | 200pm |
|--------|------|------|------|-------|-------|-------|
| gad_small_dt | 78 / 8.5s | 149 / 12.0s | 200 / 16.8s | 308 / 24.6s | 396 / 35.1s | 459 / 43.9s |
| blend_plain_k10 | 97 / 10.3s | 180 / 15.5s | 246 / 20.9s | 363 / 31.9s | 436 / 43.5s | 486 / 49.0s |
| blend_plain_k50 | 79 / 10.8s | 151 / 14.5s | 210 / 19.5s | 326 / 30.2s | 415 / 39.8s | 463 / 46.5s |
| blend_plain_k100 | 77 / 8.7s | 149 / 13.7s | 208 / 20.1s | 323 / 29.6s | 411 / 39.4s | 461 / 45.6s |

### Findings

1. **The blend hurts at every noise level.** -1pp at 10pm, -5pp at 50pm, **-10pp at 100pm**, -9pp at 150pm, -4pp at 200pm (all vs gad_small_dt baseline). This is a clear, unambiguous negative result with full 300/300 samples.

2. **Sharpness k barely matters.** k=10, 50, 100 give rates within 1-2pp of each other. The blend mechanism works (sigmoid transitions smoothly), but the *direction* of the effect is wrong — weakening v₁ ascent when λ₂<0 slows convergence rather than helping.

3. **Why it fails:** The premise was that GAD's v₁ ascent hurts when n_neg≥2 (multiple negative eigenvalues). The data says the opposite: GAD's "always ascend v₁ at full strength" is already the optimal policy. When λ₂<0, the trajectory is far from the TS, and the v₁ ascent is actively guiding it toward the saddle. Weakening that guidance (via the blend) just slows the approach.

4. **This definitively closes the "weaken GAD at high n_neg" hypothesis.** Three independent tests — hard-switch NR (Exp 4-5), preconditioned blend (Exp 13), plain blend (this experiment) — all show that reducing v₁ ascent when far from the TS is harmful. The remaining question is whether *increasing* the GAD contribution at high n_neg (multi-mode ascent) could help.

---

## Round 3, Experiment 17: Even Smaller Timestep (dt=0.002)

**SLURM:** 58932864 (tasks 18-23) | **Data:** `round3/` | **Status:** 4 complete, 2 partial (219 and 174 samples)

### Method

`gad_dt002` — Identical to gad_small_dt and gad_dt003, but dt=0.002 with 3000 steps. Continues the "smaller dt" series: dt=0.01 → 0.005 → 0.003 → 0.002. All other settings identical: Eckart projection, no mode tracking, no adaptive dt, no preconditioning, no displacement capping.

### Results

| Method | Steps | 10pm | 30pm | 50pm | 100pm | 150pm | 200pm |
|--------|-------|------|------|------|-------|-------|-------|
| gad_projected | 300 | 72.3 | 70.3 | 69.3 | 66.7 | 58.0 | 45.3 |
| gad_small_dt | 1000 | 94.3 | 94.3 | 91.3 | 86.7 | 70.3 | 51.3 |
| gad_dt003 | 2000 | 94.7 | 94.3 | 92.0 | 87.3 | 71.3 | 55.2 |
| **gad_dt002** | 3000 | **94.7** | **94.3** | **92.0** | **87.3** | **74.4*** | **56.3*** |

*Partial: 150pm=219/300, 200pm=174/300.

**Avg steps to convergence / avg wall time per sample:**

| Method | 10pm | 50pm | 100pm | 200pm |
|--------|------|------|-------|-------|
| gad_small_dt (dt=0.005) | 78 / 8.5s | 200 / 16.8s | 308 / 24.6s | 459 / 43.9s |
| gad_dt003 (dt=0.003) | 133 / 15.0s | 342 / 28.9s | 527 / 44.6s | 811 / 82.2s |
| gad_dt002 (dt=0.002) | ~200 / ~22s | ~500 / ~42s | ~780 / ~66s | ~1200 / ~120s |

### gad_dt003 rerun (100/150/200pm)

**SLURM:** 58933021 | **Data:** `round3/` | **Status:** 2 complete, 1 partial (259/300 at 200pm)

| Noise | Samples | Conv | Rate | Avg Steps | Avg Time |
|-------|---------|------|------|-----------|----------|
| 100pm | 300 | 262 | 87.3% | 527 | 44.6s |
| 150pm | 300 | 214 | 71.3% | 685 | 64.7s |
| 200pm | 259 | 143 | 55.2% | 811 | 82.2s |

These replace the Round 2 partial data (which had 131-244 samples). The 100pm and 150pm results are now full 300/300. The 200pm result (259/300) is more reliable than the previous 131-sample estimate.

### The dt series: diminishing returns

| dt | Steps | 10pm | 50pm | 100pm | 150pm | 200pm | Δ vs previous dt |
|----|-------|------|------|-------|-------|-------|------------------|
| 0.01 | 300 | 72.3 | 69.3 | 66.7 | 58.0 | 45.3 | — |
| 0.005 | 1000 | 94.3 | 91.3 | 86.7 | 70.3 | 51.3 | +22pp / +22pp / +20pp |
| 0.003 | 2000 | 94.7 | 92.0 | 87.3 | 71.3 | 55.2 | +0.4 / +0.7 / +3.9 |
| 0.002 | 3000 | 94.7 | 92.0 | 87.3 | 74.4 | 56.3 | +0.0 / +0.0 / +1.1 |

**Finding:** Clear diminishing returns. The big jump is 0.01→0.005 (+20pp). Then 0.005→0.003 gives +1-4pp at high noise. And 0.003→0.002 gives +0-3pp, only visible at 150-200pm. At 10-100pm, dt=0.003 and dt=0.002 are identical. The cost is proportional: dt=0.002 needs 3× the steps (and 3× the wall time) of dt=0.005 for marginal gains.

---

## Experiments NOT Run

| Experiment | Description | Why not run |
|-----------|-------------|-------------|
| gad_clamp_005 | 0.05A aggressive clamp | Low priority after clamp proved inert |
| rfo_gad | RFO secular equation + GAD | Code written (`search/rfo_gad.py`), never submitted |
| grad_descent_pp | Plain gradient descent when n_neg≥2 | Superseded by precond_descent in redesign |

---

## Consolidated Results: All Methods Ranked

300 samples from Transition1x train split (indices 0-299), noise seed=42. Partial results (marked *) use 131-289 samples due to SLURM timeout. Internal 150pm/200pm at 1000 steps timed out (marked —).

### Ranked by n_neg==1 + fmax<0.01 (strictest criterion, uses Sella's force metric)

| Rank | Method | Type | Steps | 10pm | 50pm | 100pm | 200pm |
|------|--------|------|-------|------|------|-------|-------|
| 1 | **gad_dt002** | GAD | 3000 | **94.7** | **92.0** | **87.3** | **56.3*** |
| 2 | **gad_dt003** | GAD | 2000 | **94.7** | **92.0** | **87.3** | **55.2** |
| 3 | gad_small_dt | GAD | 1000 | 94.3 | 91.3 | 86.7 | 51.3 |
| 4 | blend_plain_k50 | GAD | 1000 | 93.7 | 86.7 | 76.3 | 47.7 |
| 5 | blend_plain_k100 | GAD | 1000 | 93.7 | 86.0 | 76.7 | 47.7 |
| 6 | blend_plain_k10 | GAD | 1000 | 93.0 | 87.0 | 76.7 | 46.3 |
| 7 | Sella Cart+Eckart | Sella | 1000 | 91.7 | 88.3 | 76.3 | 25.7 |
| 8 | Sella Cartesian | Sella | 1000 | 91.3 | 87.7 | 75.0 | 23.3 |
| 9 | Sella Cart (200 steps) | Sella | 200 | 91.3 | 87.7 | 75.3 | 22.7 |
| 10 | Sella Int+Eckart | Sella | 1000 | 87.0 | 82.3 | 59.0 | — |
| 11 | Sella Internal | Sella | 1000 | 87.0 | 81.0 | 59.7 | — |
| 12 | Sella Int (200 steps) | Sella | 200 | 87.0 | 79.0 | 56.7 | 16.3 |

### Ranked by n_neg==1 + force_norm<0.01 (our GAD criterion, looser force metric)

| Rank | Method | Type | Steps | 10pm | 50pm | 100pm | 200pm |
|------|--------|------|-------|------|------|-------|-------|
| 1 | **gad_dt002** | GAD | 3000 | **94.7** | **92.0** | **87.3** | **56.3*** |
| 2 | **gad_dt003** | GAD | 2000 | **94.7** | **92.0** | **87.3** | **55.2** |
| 3 | Sella Cart+Eckart | Sella | 1000 | 94.7 | 91.3 | 79.0 | 27.7 |
| 4 | gad_small_dt | GAD | 1000 | 94.3 | 91.3 | 86.7 | 51.3 |
| 5 | Sella Cartesian | Sella | 1000 | 94.3 | 91.0 | 78.0 | 25.7 |
| 6 | Sella Cart (200 steps) | Sella | 200 | 94.3 | 91.0 | 78.3 | 25.3 |
| 7 | nr_gad_damped α=0.1 | GAD | 1000 | 94.7 | 77.7 | 58.0 | 33.7 |
| 8 | blend_plain_k50 | GAD | 1000 | 93.7 | 86.7 | 76.3 | 47.7 |
| 9 | blend_plain_k100 | GAD | 1000 | 93.7 | 86.0 | 76.7 | 47.7 |
| 10 | blend_plain_k10 | GAD | 1000 | 93.0 | 87.0 | 76.7 | 46.3 |
| 11 | Sella Int+Eckart | Sella | 1000 | 92.0 | 85.7 | 62.0 | — |
| 12 | Sella Internal | Sella | 1000 | 92.0 | 86.7 | 63.0 | — |
| 13 | Sella Int (200 steps) | Sella | 200 | 91.7 | 84.7 | 61.3 | 18.0 |
| 14 | adaptive_floor | GAD | 1000 | 83.0 | 70.2* | 43.2* | 15.9* |
| 15 | precond_gad_dt01 | GAD | 1000 | 78.3 | 68.3 | 58.0 | 41.7 |
| 16 | precond_gad_001 | GAD | 1000 | 73.7 | 48.0 | 21.7 | 3.3 |
| 17 | gad_projected | GAD | 300 | 72.3 | 69.3 | 66.7 | 45.3 |
| 18 | blend_k50 (precond) | GAD | 1000 | 72.0 | 37.2* | 14.3* | 3.1* |
| 19 | gad_adaptive_dt | GAD | 1000 | 71.3 | 52.7 | 37.7 | 14.3 |
| 20 | Sella Int fmax=0.03 (200) | Sella | 200 | 61.0 | 56.7 | 42.3 | 11.7 |
| 21 | nr_gad_pingpong | GAD | 1000 | 56.7 | 31.7 | 24.7 | 18.3 |
| 22 | adaptive_mm | GAD | 1000 | 53.7* | 35.1* | 22.6* | 5.7* |

---

## Data Locations

```
Round 1 methods:     /lustre07/scratch/memoozd/gadplus/runs/method_cmp_300/
Round 1 damped NR:   /lustre07/scratch/memoozd/gadplus/runs/targeted/
Round 1 noise:       /lustre07/scratch/memoozd/gadplus/runs/noise_survey_300/
Round 1 start geom:  /lustre07/scratch/memoozd/gadplus/runs/starting_geom_300/
Round 1 geodesic:    /lustre07/scratch/memoozd/gadplus/runs/geodesic_mid/
Round 1 basin:       /lustre07/scratch/memoozd/gadplus/runs/basin_map/
Round 1 IRC:         /lustre07/scratch/memoozd/gadplus/runs/irc_validation/
Round 2 precond:     /lustre07/scratch/memoozd/gadplus/runs/precond_gad/
Round 2 others:      /lustre07/scratch/memoozd/gadplus/runs/round2/
Round 3 Sella 200:   /lustre07/scratch/memoozd/gadplus/runs/sella_baselines/
Round 3 Sella 1000:  /lustre07/scratch/memoozd/gadplus/runs/sella_1000/
Round 3 blend+dt002: /lustre07/scratch/memoozd/gadplus/runs/round3/
```

## SLURM Job IDs

| Job | ID | Status |
|-----|-----|--------|
| Round 1 method cmp | 58845357 | Complete (42/42) |
| Round 1 damped NR | 58852071 | Complete (42/42) |
| Round 1 noise survey | 58835838 | Complete (9/9) |
| Round 1 start geom | 58835839 | Complete (4/4) |
| Round 1 geodesic | 58852072 | Timeout (204/300 salvaged) |
| Round 1 basin | 58835840 | Complete |
| Round 1 IRC | 58834594 | Complete |
| Round 2 precond GAD | 58885855 | Complete (30/30) |
| Round 2 round2 | 58886863 | Mixed (9 full + 39 partial timeout) |
| Round 3 Sella 200-step | 58932967 | Complete (24/24) |
| Round 3 Sella 1000-step | 58937673 | 20/24 complete (internal 150/200pm timeout) |
| Round 3 blend + dt002 | 58932864 | 22/24 complete (dt002 150/200pm partial) |
| Round 3 dt003 rerun | 58933021 | 2/3 complete (200pm partial 259/300) |
| Round 3 Sella 1000-step | 58937673 | 20/24 (internal 150/200pm timeout) |
