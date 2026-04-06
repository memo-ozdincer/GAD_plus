# GAD+ Round 2 Experiment Plan

> Standalone document for a fresh agent. Contains full context for each experiment: what it is, why we're trying it, exact implementation details, what to compare against, and what success looks like. All methods must be **diffusion-compatible** (differentiable end-to-end, no path history, no branching on discrete values, no conditional accept/reject).

## Background for the Fresh Agent

### What we're doing
We're searching for transition states (saddle points) on molecular potential energy surfaces using Gentlest Ascent Dynamics (GAD). GAD modifies atomic forces to ascend along the lowest Hessian eigenvector while descending along all others:

```
F_GAD = F + 2(F · v₁) v₁
x_{n+1} = x_n + dt * F_GAD
```

where v₁ is the lowest eigenvector of the Eckart-projected vibrational Hessian. A TS is converged when n_neg=1 (exactly one negative eigenvalue) AND force_norm < 0.01 eV/Å.

### What we know so far
- **Best method:** `gad_small_dt` — Eckart-projected GAD, dt=0.005, 1000 steps. Gets **94.3% at 10pm, 91.3% at 50pm, 86.7% at 100pm** noise (300 samples).
- **Eckart projection is essential:** Without it, 0% convergence at ≥50pm.
- **Adaptive dt (eigenvalue-clamped) hurt:** Our implementation with dt_base=0.01 reduced convergence by 20-30pp. But this was because the base was 5x too high — a prior codebase (Multi-Mode GAD) used dt_base=0.002 effectively and it worked.
- **NR-GAD ping-pong hurt:** Pure Newton descent when n_neg≥2 overshoots. Even with damping (α=0.1-0.3), it's worse than pure GAD. The NR step formula (H⁻¹g with eigenvalue flooring) is the problem, not the switching logic.
- **Displacement clamping has zero effect:** 0.1Å vs 0.35Å cap, <1pp difference.
- **Basin of attraction ~100pm:** GAD either finds the correct TS or fails. No silent wrong answers.
- **Convergence ceiling ~96%:** 3000 steps gains only 1-2pp over 1000.
- **Preconditioned GAD (currently running):** Δx = dt · |H|⁻¹ F_GAD. Mode-wise Hessian scaling. Converges but takes more steps at dt=0.005 due to step shrinkage along steep modes. The dt=0.01 variant is the key one to watch.

### Diffusion compatibility constraint
These methods will be used inside a differentiable pipeline (adjoint sampling, diffusion generation). Requirements:
- **No path history:** Every step depends only on current geometry, energy, forces, Hessian. No storing previous steps.
- **No branching on discrete values:** Can't do `if n_neg >= 2: use NR`. Can use smooth functions of continuous quantities like eigenvalues.
- **No conditional accept/reject:** Can't do "accept step if energy decreased." Every step must be taken unconditionally.
- **Differentiable throughout:** The entire x₀ → x_final mapping must have gradients flowing through it.

### Infrastructure
- **Cluster:** Narval (Alliance Canada), A100 MIG slices (a100_2g.10gb:1)
- **Script:** `scripts/method_single.py --method METHOD --noise NOISE --n-samples 300 --n-steps 1000`
- **Data:** Transition1x train split, 300 samples, HIP neural network potential
- **Output:** Parquet summaries + trajectories to `/lustre07/scratch/memoozd/gadplus/runs/`
- **Baseline comparison:** `method_cmp_300/summary_gad_small_dt_*.parquet` (already computed)

### Key source files
- `src/gadplus/search/gad_search.py` — Main GAD loop, `GADSearchConfig`, `run_gad_search()`
- `src/gadplus/search/nr_gad_pingpong.py` — NR-GAD hybrid, `NRGADPingPongConfig`, `run_nr_gad_pingpong()`
- `src/gadplus/core/adaptive_dt.py` — `compute_adaptive_dt()`, eigenvalue-clamped formula
- `src/gadplus/core/gad.py` — GAD vector computation
- `src/gadplus/projection/projection.py` — Eckart projection, `vib_eig()`, `gad_dynamics_projected()`, `preconditioned_gad_dynamics_projected()`
- `scripts/method_single.py` — Standalone experiment runner with method configs

---

## Experiment A: Pure GAD Improvements

### A1. Corrected Adaptive Timestep

**What:** Our eigenvalue-clamped adaptive dt failed badly (-20pp). But a prior codebase (Multi-Mode GAD) used the same formula with different parameters and it worked. The difference: their effective dt base was 0.002 (dt=0.02 × scale_factor=0.1), ours was 0.01. At |λ₀|=1.0, theirs gives dt=0.002, ours gives dt=0.01 — 5x larger steps in steep regions.

**Formula:**
```
dt_eff = dt_base / clamp(|λ₀|, 0.01, 100)
dt_eff = clamp(dt_eff, dt_min, dt_max)
```

**Configs to test:**
| Name | dt_base | dt_min | dt_max | Notes |
|------|---------|--------|--------|-------|
| adaptive_mm | 0.002 | 1e-5 | 0.08 | Match Multi-Mode GAD parameters exactly |
| adaptive_mm2 | 0.001 | 1e-5 | 0.05 | Even more conservative base |

**Implementation:** Change dt_base in `METHOD_CONFIGS`, pass dt_min/dt_max through to `GADSearchConfig`. The `compute_adaptive_dt()` function in `adaptive_dt.py` already implements the formula — just needs different input parameters.

**Compare against:** `gad_small_dt` (94.3% at 10pm) and `gad_adaptive_dt` (71.3% at 10pm, the old broken version with dt_base=0.01).

**Success looks like:** If adaptive_mm gets ≥90% at 10pm, we've recovered adaptive dt from a failed method to a viable one, and we have data-dependent step sizing without any non-differentiable components.

### A2. Smaller Fixed dt

**What:** We found dt=0.005 beats dt=0.01 by 20+pp. The Multi-Mode GAD HPO found dt=0.0036 optimal on a different PES (SCINE). Is there headroom below 0.005?

**Configs:**
| Name | dt | Steps | Displacement budget |
|------|-----|-------|-------------------|
| gad_dt003 | 0.003 | 2000 | 6.0 |
| gad_dt002 | 0.002 | 3000 | 6.0 |
| (baseline) gad_small_dt | 0.005 | 1000 | 5.0 |

**Implementation:** Add entries to `METHOD_CONFIGS` with different dt values. Run with `--n-steps 2000` or `--n-steps 3000`.

**Compare against:** `gad_small_dt` at 1000 steps.

**Success looks like:** +2-5pp at 50-100pm noise. Diminishing returns expected but worth quantifying.

### A3. Clamping Extremes

**What:** You want to see aggressive clamping vs no clamping definitively.

**Configs:**
| Name | max_atom_disp | Notes |
|------|---------------|-------|
| gad_no_clamp | 999.0 | Effectively no clamping at all |
| gad_clamp_005 | 0.05 | Very aggressive, half of tight_clamp |
| (baseline) gad_small_dt | 0.35 | Current default |

**Implementation:** Just change `max_disp` in `METHOD_CONFIGS`.

**Compare against:** `gad_small_dt` (0.35Å) and `gad_tight_clamp` (0.1Å, already tested: zero effect).

**Success looks like:** Confirming that clamping is truly inert across the full range. If aggressive clamping hurts at high noise, that tells us the trajectories do occasionally need large displacements.

### A4. Adaptive dt with Floor Fix

**What:** Our adaptive dt might fail partly because there's no meaningful dt floor. When |λ₀| is very large (steep saddle region), dt = 0.01/100 = 0.0001, which is effectively frozen. Adding a higher floor (1e-3 instead of 1e-4) ensures the trajectory always makes progress.

**Config:**
| Name | dt_base | dt_min | dt_max |
|------|---------|--------|--------|
| adaptive_floor | 0.005 | 1e-3 | 0.05 |

**Implementation:** Set dt_min=1e-3 in `GADSearchConfig`.

**Compare against:** `gad_adaptive_dt` (dt_base=0.01, dt_min=1e-4) which got 71.3% at 10pm.

---

## Experiment B: Improved NR-GAD Hybrid

### B1. Gradient Descent Instead of Newton When n_neg≥2

**What:** Our NR-GAD fails because the NR step (H⁻¹g) overshoots. What if we replace NR with plain gradient descent? When n_neg≥2, just take Δx = dt · F (forces, not F_GAD) at the same dt=0.005. No eigenvalue inversion, no blowup. This tests whether the problem is the NR step direction or just "not doing GAD when n_neg≥2."

**Implementation:** In `nr_gad_pingpong.py`, when phase=="nr", instead of calling `nr_minimize_step()`, do:
```python
step_disp = cfg.gad_dt * forces.reshape(-1, 3)  # plain gradient descent
step_disp = cap_displacement(step_disp, cfg.max_atom_disp)
```

**Compare against:** `gad_small_dt` (pure GAD, 94.3%), `nr_gad_damped_01` (damped NR, 94.7% at 10pm but 77.7% at 50pm).

**Success looks like:** If this matches or beats pure GAD, the NR step direction was always the problem. If it's worse, then not doing GAD when n_neg≥2 is inherently harmful — GAD's modified force handles the n_neg transition better than any descent method.

### B2. Preconditioned Descent When n_neg≥2

**What:** Instead of Newton (H⁻¹g) or plain gradient (F), use preconditioned gradient descent: Δx = dt · |H|⁻¹ · F. This gives mode-wise curvature-aware step scaling (like Newton) but uses |H| instead of H, so negative eigenvalues are handled correctly (always descend). Same formula as your preconditioned GAD, but applied to unmodified forces instead of F_GAD.

**Implementation:** When phase=="nr":
```python
# Project forces into vibrational subspace
coeffs = evecs_vib.T @ forces_flat  # (M,)
safe_evals = torch.clamp(evals_vib.abs(), min=eig_floor)
precond_coeffs = coeffs / safe_evals
step_disp = (evecs_vib @ precond_coeffs).reshape(-1, 3) * cfg.gad_dt
```

**Compare against:** B1 (plain gradient descent) and baseline (damped NR).

**Success looks like:** Better than B1 because curvature information helps, better than current NR because no sign-flip blowup.

---

## Experiment C: Smooth-Blended GAD/Descent (No Switching)

### C1. λ₂-Blended Dynamics

**What:** Instead of a hard switch between GAD and descent based on n_neg (which is discrete and non-differentiable), use a smooth blend controlled by the second eigenvalue λ₂. When λ₂ > 0, we're near an index-1 saddle and should use GAD. When λ₂ < 0, we're at a higher-order saddle and should descend.

**Formula:**
```
weight = sigmoid(k * λ₂)        # smooth, differentiable in λ₂
F_eff = weight * F_GAD + (1 - weight) * F
Δx = dt * F_eff
```

When λ₂ ≫ 0 (near TS): weight → 1, pure GAD.
When λ₂ ≪ 0 (higher-order saddle): weight → 0, pure descent.
At the transition (λ₂ ≈ 0): smooth blend.

**Why λ₂ and not n_neg:** λ₂ is a continuous, differentiable function of the Hessian. n_neg is a discrete count. sigmoid(k·λ₂) is smooth; any function of n_neg has jumps.

**Parameters to test:**
| Name | k (sharpness) | Notes |
|------|---------------|-------|
| blend_k10 | 10 | Gentle blend, wide transition zone |
| blend_k50 | 50 | Sharper, ~0.1 eV/Å² transition width |
| blend_k100 | 100 | Nearly hard switch but still differentiable |

**Implementation:** New dynamics function. At each step:
1. Compute vib_eig → get evals, evecs, n_neg, λ₁, λ₂
2. Compute F_GAD via gad_dynamics_projected (standard)
3. Compute weight = sigmoid(k * λ₂)
4. F_eff = weight * F_GAD + (1 - weight) * F_projected
5. Δx = dt * F_eff, cap displacement

This is a single search loop — no phase switching, no NR, no ping-pong. Just one dynamics with a smooth blend.

**Compare against:** `gad_small_dt` (pure GAD, which is weight=1 always) and `nr_gad_damped_01` (hard switch at n_neg=2).

**Success looks like:** Matching pure GAD at low noise (where n_neg is usually <2 anyway, so weight≈1) while improving at high noise (100-200pm) where many starting geometries have λ₂ < 0.

### Why this is the most important experiment

Pure GAD at 94% is already excellent at low noise. The 6% failures are molecules where n_neg oscillates between 1 and 2+ without converging. The blended dynamics would smoothly modulate the GAD contribution based on how far we are from index-1, potentially stabilizing these oscillations. And it's fully differentiable — sigmoid(k·λ₂) has clean gradients through the Hessian eigenvalue.

---

## Experiment D: Differentiable RFO-GAD

### D1. RFO Step Direction with GAD Ascent

**What:** RFO (Rational Function Optimization) solves for the optimal Hessian shift μ via a secular equation, giving data-dependent step scaling that adapts to the local eigenvalue spectrum. Unlike our preconditioned GAD which uses a fixed eig_floor, RFO's shift is computed fresh each step from the actual eigenvalues.

**The secular equation:**
```
Σ cᵢ² / (λᵢ - μ) + μ = 0,    μ < λ_min
```
where cᵢ = g · vᵢ (gradient projection onto eigenvector i). Solved via Newton iteration (differentiable).

**The step for each mode:**
```
hᵢ = -cᵢ / (λᵢ - μ)
```

**For GAD adaptation:** On the ascent mode (v₁, the lowest eigenvector), flip the sign:
```
h₁ = +c₁ / (|λ₁| + |μ|)     # ascend along TS mode
hᵢ = -cᵢ / (λᵢ - μ)          # descend along all other modes (i > 1)
```

**Final step:**
```
Δx = dt * Σ hᵢ vᵢ,  capped at max_atom_disp
```

**Why this might beat preconditioned GAD:** Preconditioned GAD scales by 1/max(|λᵢ|, floor) with a fixed floor. RFO scales by 1/(λᵢ - μ) with μ solved from the spectrum. When eigenvalues are well-separated, μ is small → aggressive steps on flat modes. When eigenvalues cluster near zero, μ is large → conservative steps. The adaptation is automatic and spectrum-aware, no hyperparameter tuning.

**Differentiability:** The secular equation is smooth in {λᵢ, cᵢ}. Solving it via Newton iteration is differentiable (implicit function theorem — dμ/dλᵢ exists and is smooth). No trust radius, no PLS, no acceptance criterion. The output Δx is a smooth function of (coords, forces, Hessian).

**Implementation:**
1. Add `_solve_rfo_secular(evals, coeffs)` → returns μ. Newton iteration on the secular equation, ~5-10 iterations.
2. Add `rfo_gad_step(forces, evals_vib, evecs_vib, dt)` → computes step using RFO shift with GAD sign flip on mode 1.
3. New method config in `method_single.py`.

**Compare against:** `precond_gad_001` (fixed floor=0.01) and `gad_small_dt` (no preconditioning).

**Success looks like:** Better than preconditioned GAD because the shift adapts, comparable wall time (secular equation is cheap — eigendecomposition dominates). The question is whether RFO's adaptive shift gives meaningful improvement over a well-tuned fixed floor.

---

## Priority Order

1. **C1 (λ₂-blended dynamics)** — Most novel, fully differentiable, directly addresses the n_neg oscillation failure mode. If this works, it's the paper's main method.
2. **A1 (corrected adaptive dt)** — Quick test of whether our adaptive dt failure was just wrong parameters. One number change.
3. **A6 (preconditioned GAD)** — Already running. Watch the dt=0.01 variant especially.
4. **B1 (gradient descent when n_neg≥2)** — Diagnostic: is the NR direction the problem or is "not doing GAD" the problem?
5. **D1 (RFO-GAD)** — Most sophisticated, biggest implementation effort. Save for after C1/A1 results are in.
6. **A2/A3 (smaller dt)** — Incremental, diminishing returns expected.
7. **A3 (clamping extremes)** — Confirmatory, low priority.
8. **A4 (dt floor fix)** — Incremental.

## Baseline Numbers for Comparison

From `method_cmp_300/` (300 samples, 1000 steps):

| Method | 10pm | 30pm | 50pm | 100pm | 150pm | 200pm |
|--------|------|------|------|-------|-------|-------|
| gad_small_dt (dt=0.005) | 94.3 | 94.3 | 91.3 | 86.7 | 70.3 | 51.3 |
| gad_projected (dt=0.01) | 72.3 | 70.3 | 69.3 | 66.7 | 58.0 | 45.3 |
| gad_adaptive_dt (dt=0.01, broken) | 71.3 | 65.0 | 52.7 | 37.7 | 23.7 | 14.3 |
| nr_gad_pingpong (undamped) | 56.7 | 35.3 | 31.7 | 24.7 | 22.3 | 18.3 |

From `targeted/` (300 samples, 1000 steps):

| Method | 10pm | 30pm | 50pm | 100pm | 150pm | 200pm |
|--------|------|------|------|-------|-------|-------|
| nr_gad_damped α=0.1 | 94.7 | 88.0 | 77.7 | 58.0 | 46.0 | 33.7 |
| nr_gad_damped α=0.2 | 93.0 | 88.0 | 78.3 | 60.3 | 47.3 | 36.3 |
| nr_gad_damped α=0.3 | 88.7 | 82.3 | 75.0 | 58.7 | 46.3 | 37.0 |

From salvaged trajectory files (partial, 200pm):

| Config | Samples | Rate |
|--------|---------|------|
| gad_small_dt 2000 steps | 275/300 | 58.2% |
| gad_small_dt 3000 steps | 224/300 | 60.7% |
