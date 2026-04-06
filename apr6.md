# GAD+ Status — April 6, 2026 (Updated)

Active experiments running. This covers what changed since the morning handoff.

---

## What's Currently Running

### 1. Preconditioned GAD (SLURM 58885855) — still running

30 MIG jobs, ~25+ min in. 5 methods × 6 noise levels. See original apr6.md for details.
Output: `/lustre07/scratch/memoozd/gadplus/runs/precond_gad/`

### 2. Round 2 Experiments (SLURM 58886863) — just submitted

48 MIG jobs. 8 methods × 6 noise levels. 300 samples, 1000 steps (except gad_dt003 at 2000 steps).
Output: `/lustre07/scratch/memoozd/gadplus/runs/round2/`

**Methods:**

| Task IDs | Method | What it is | Diffusion-compatible? |
|----------|--------|------------|----------------------|
| 0-5 | adaptive_mm | Corrected adaptive dt, base=0.002 (Multi-Mode GAD params) | Yes |
| 6-11 | adaptive_mm2 | Even more conservative adaptive dt, base=0.001 | Yes |
| 12-17 | gad_dt003 | Smaller fixed dt=0.003, 2000 steps | Yes |
| 18-23 | gad_no_clamp | No displacement cap (max_disp=999) | Yes |
| 24-29 | adaptive_floor | Adaptive dt with higher floor (dt_min=1e-3) | Yes |
| 30-35 | **precond_descent** | Precond descent when n_neg≥2, precond GAD when n_neg<2 | No (diagnostic) |
| 36-41 | **blend_k50** | Blended precond GAD, sigmoid(50·λ₂) | **Yes** |
| 42-47 | **blend_k100** | Blended precond GAD, sigmoid(100·λ₂) | **Yes** |

---

## What Changed: Blended Preconditioned GAD

The key insight from today's design session: the blend shouldn't be between raw GAD and raw descent — it should be between **preconditioned GAD** and **preconditioned descent**:

```
w = sigmoid(k · λ₂)
F_blend = F + 2·w·(F·v₁)v₁           # partial GAD: w=1 → full flip, w=0 → no flip
Δx = dt · |H|⁻¹ · F_blend             # always preconditioned
```

Mode by mode:
- **v₁ (lowest):** step ∝ (F·v₁·(2w-1)) / |λ₁|. w=1 (near TS) → ascend. w=0 (far) → descend. Smooth.
- **vᵢ (i>1):** step ∝ (F·vᵢ) / |λᵢ|. Always descent, always curvature-scaled. Unaffected by blend.

**The blend only controls one decision:** ascend v₁ or not. Everything else is preconditioned descent.

### Implementation

Modified `preconditioned_gad_dynamics_projected()` in `projection.py` — added `gad_blend_weight` parameter (default 1.0 = standard GAD). The fixed `2.0` in the GAD formula is now `2.0 * w`.

Added `blend_sharpness` to `GADSearchConfig`. When >0, computes `w = sigmoid(k * λ₂)` at each step and passes it as `gad_blend_weight`.

### precond_descent diagnostic

Hard-switch version: uses `NRGADPingPongConfig` with `descent_mode="preconditioned"`. When n_neg≥2, calls `preconditioned_gad_dynamics_projected(gad_blend_weight=0.0)` (pure precond descent). When n_neg<2, calls with `gad_blend_weight=1.0` (full precond GAD).

Tests whether GAD's v₁ ascent helps or hurts when n_neg≥2. If precond_descent beats precond_GAD at high noise, then v₁ far from the saddle isn't the right mode to ascend — supports the blend approach.

---

## Code Changes Since Morning

| File | Change |
|------|--------|
| `src/gadplus/projection/projection.py` | `preconditioned_gad_dynamics_projected` gains `gad_blend_weight` param |
| `src/gadplus/search/gad_search.py` | `GADSearchConfig.blend_sharpness`, computes `w=sigmoid(k·λ₂)` |
| `src/gadplus/search/nr_gad_pingpong.py` | `descent_mode="preconditioned"` uses same precond function with w=0 |
| `src/gadplus/search/blended_gad.py` | Created (standalone blended GAD — superseded by integrated approach) |
| `src/gadplus/search/rfo_gad.py` | Created (RFO-GAD — lower priority, not in current batch) |
| `scripts/method_single.py` | 8 new Round 2 method configs |
| `scripts/run_round2.slurm` | 48-job array script |

---

## Cancelled Jobs

- 58886605 — first round2 attempt, had import error (`_eckart_projector` not exported)
- 58886708 — second round2 attempt, cancelled to redesign blended experiments

---

## Baseline Numbers for Comparison

| Method | 10pm | 30pm | 50pm | 100pm | 150pm | 200pm |
|--------|------|------|------|-------|-------|-------|
| **gad_small_dt (dt=0.005)** | **94.3** | **94.3** | **91.3** | **86.7** | **70.3** | **51.3** |
| gad_projected (dt=0.01) | 72.3 | 70.3 | 69.3 | 66.7 | 58.0 | 45.3 |
| gad_adaptive_dt (dt=0.01, broken) | 71.3 | 65.0 | 52.7 | 37.7 | 23.7 | 14.3 |
| nr_gad_damped α=0.1 | 94.7 | 88.0 | 77.7 | 58.0 | 46.0 | 33.7 |

---

## What to Do When Results Come In

1. Check for errors: `cat /lustre07/scratch/memoozd/gadplus/logs/round2_58886863_*.err | tail -5`
2. Check completions: `ls /lustre07/scratch/memoozd/gadplus/runs/round2/summary_*.parquet | wc -l` (expect 48)
3. Analyze:
```sql
SELECT method, noise_pm, COUNT(*) as total,
       SUM(CASE WHEN converged THEN 1 ELSE 0 END) as conv,
       ROUND(100.0 * conv / total, 1) as rate
FROM '/lustre07/scratch/memoozd/gadplus/runs/round2/summary_*.parquet'
GROUP BY method, noise_pm ORDER BY method, noise_pm;
```
4. Key comparisons:
   - `blend_k50` and `blend_k100` vs `gad_small_dt` — does blending help?
   - `precond_descent` vs `gad_small_dt` — does v₁ ascent hurt at n_neg≥2?
   - `adaptive_mm` vs `gad_adaptive_dt` — did correcting dt_base fix adaptive dt?
5. Update EXPERIMENT_LOG.md and EXPERIMENTS.tex with results
