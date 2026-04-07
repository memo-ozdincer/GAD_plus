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
| **gad_dt003** | 2000 | **94.7** (300) | **94.3** (300) | **92.0** (300) | **88.9** (244) | **75.3** (158) | **58.8** (131) |

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

## Experiments NOT Run

| Experiment | Description | Why not run |
|-----------|-------------|-------------|
| gad_dt002 | dt=0.002, 3000 steps | Dropped to limit job count |
| gad_clamp_005 | 0.05A aggressive clamp | Low priority after clamp proved inert |
| blend_k10 | Wide blend zone, k=10 | Dropped (k50/k100 already identical) |
| rfo_gad | RFO secular equation + GAD | Code written (`search/rfo_gad.py`), never submitted |
| grad_descent_pp | Plain gradient descent when n_neg≥2 | Superseded by precond_descent in redesign |
| blend WITHOUT precond | λ₂-blend on plain GAD | Not yet implemented — highest priority next |

---

## Consolidated Results: All Methods Ranked

| Rank | Method | 10pm | 50pm | 100pm | 200pm | Key feature |
|------|--------|------|------|-------|-------|-------------|
| 1 | **gad_dt003** | **94.7** | **92.0** | **88.9** | **58.8** | dt=0.003, 2000 steps |
| 2 | gad_small_dt | 94.3 | 91.3 | 86.7 | 51.3 | dt=0.005, 1000 steps |
| 3 | gad_no_clamp | 94.3 | 91.3 | 86.7 | 54.6 | Same as #2, no cap |
| 4 | nr_gad_damped α=0.1 | 94.7 | 77.7 | 58.0 | 33.7 | Damped NR when n_neg≥2 |
| 5 | adaptive_floor | 83.0 | 70.2 | 43.2 | 15.9 | Adaptive dt, high floor |
| 6 | precond_gad_dt01 | 78.3 | 68.3 | 58.0 | 41.7 | |H|⁻¹ at dt=0.01 |
| 7 | precond_gad_001 | 73.7 | 48.0 | 21.7 | 3.3 | |H|⁻¹ at dt=0.005 |
| 8 | gad_projected | 72.3 | 69.3 | 66.7 | 45.3 | dt=0.01 baseline |
| 9 | blend_k50 | 72.0 | 37.2 | 14.3 | 3.1 | λ₂-blend + |H|⁻¹ |
| 10 | precond_descent | 71.3 | 38.5 | 14.0 | 2.4 | |H|⁻¹ descent diagnostic |
| 11 | gad_adaptive_dt | 71.3 | 52.7 | 37.7 | 14.3 | Adaptive dt (old params) |
| 12 | nr_gad_pingpong | 56.7 | 31.7 | 24.7 | 18.3 | Undamped NR |
| 13 | adaptive_mm | 53.7 | 35.1 | 22.6 | 5.7 | Multi-Mode GAD params |
| 14 | adaptive_mm2 | 40.1 | 22.4 | 13.1 | 2.4 | Even smaller base |

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
