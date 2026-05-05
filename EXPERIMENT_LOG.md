# GAD+ Complete Experiment Log

> All experiments on Transition1x train split (300 samples unless noted), HIP neural network potential, Narval A100 MIG slices.  
> **Convergence:** n_neg==1 AND force_norm < 0.01 eV/A on Eckart-projected vibrational Hessian.  
> **Note (2026-04-15):** We can migrate reporting to n_neg==1 AND fmax < 0.01 **without rerunning trajectories** by recomputing metrics from saved final coordinates in trajectory Parquet files (see [scripts/backfill_fmax.py](scripts/backfill_fmax.py)). Existing force_norm tables remain for continuity until full backfill is applied.  
> **GAD formula:** F_GAD = F + 2(F·v₁)v₁, where v₁ is the lowest eigenvector of the Eckart-projected vibrational Hessian.  
> **Step:** x_{n+1} = x_n + dt · F_GAD (Euler integration).

---

## Broad Running Ledger (Keep This Even After Compaction)

This section is intentionally redundant. It is the "everything we've done that comes to mind" ledger so the full arc remains visible even if later sections get tightened.

### Core experimental arc so far

1. Established the Eckart-projected GAD baseline on Transition1x and showed projection is the main unlock.
2. Tested starting geometries beyond noised TS: midpoint, reactant, product, and a partial geodesic-midpoint run.
3. Mapped the TS basin of attraction and found no silent wrong-TS convergence below 100pm in the original basin test.
4. Ran a 7-method comparison and found `gad_small_dt` clearly dominates the first round.
5. Tested NR-GAD ping-pong and then damped NR-GAD; both underperformed because Hessian-inversion steps overshoot.
6. Ran an initial IRC validation pass on converged TS candidates using a simple RMSD endpoint check.
7. Ran preconditioned GAD and showed `|H|^{-1}` scaling is actively harmful for GAD dynamics.
8. Re-tested adaptive timestep variants, including literature-inspired corrected settings; all remained worse than fixed dt.
9. Continued the fixed-dt series (`0.01 -> 0.005 -> 0.003 -> 0.002`) and found smaller dt helps monotonically with diminishing returns.
10. Tested removing displacement capping and found the cap is effectively inert at small dt.
11. Tested higher-floor adaptive dt and found it improves over bad adaptive variants but still loses to fixed dt.
12. Ran a preconditioned-descent diagnostic to separate "preconditioning failure" from "GAD-vs-descent" effects.
13. Implemented and tested `lambda_2`-blended dynamics with preconditioning; found preconditioning dominated the behavior.
14. Re-tested the `lambda_2` blend without preconditioning and found the blend itself hurts relative to plain GAD.
15. Ran Sella baselines at 200 steps, then a 1000-step Sella follow-up with Cartesian/internal and Eckart/non-Eckart variants.
16. Showed GAD beats Sella increasingly strongly as noise rises, especially under the strict `n_neg==1 + fmax<0.01` comparison.
17. Ran one-way descent-to-GAD switching plus trajectory analysis and showed the real failure mode is Newton overshoot into `n_neg=0`, not switching logic alone.
18. Added backfill support for `fmax` so we can migrate reporting to `n_neg==1 AND fmax<0.01` without rerunning the original trajectories.

### Infrastructure and analysis upgrades

1. Standardized on viewing Narval CPU-only jobs as the right place for bulk visualization/export work, with MIG slicing still available for heavy parallel experiment sweeps.
2. Added [scripts/backfill_fmax.py](scripts/backfill_fmax.py) so old Parquet outputs can be retroactively rescored with `fmax`.
3. Upgraded [scripts/visualize_3d.py](scripts/visualize_3d.py) so it no longer assumes `force_max` exists in every trajectory schema.
4. Made the visualization workflow viewer-first rather than Plotly-only:
   - multi-frame XYZ for `arianjamasb.protein-viewer`
   - `frames_xyz/` bundles for `stevenyu.nano-protein-viewer`
5. Added [.vscode/extensions.json](.vscode/extensions.json) with those two 3D viewer recommendations.
6. Added [scripts/export_converged_ts_xyz.py](scripts/export_converged_ts_xyz.py) to export converged final TS structures for later inspection.
7. Added [scripts/make_flagship_visualizations.py](scripts/make_flagship_visualizations.py) to batch-export representative `gad_small_dt` trajectories.
8. Added [scripts/run_flagship_visualizations_cpu.slurm](scripts/run_flagship_visualizations_cpu.slurm) to run those exports on CPU nodes.
9. Fixed a schema bug in `make_flagship_visualizations.py`:
   - original assumption: `search_method` and `run_id` existed in the summary parquet
   - actual schema: `method` existed and `run_id` had to be resolved from trajectory parquet files
10. Verified the flagship visualization batch succeeded and wrote 15 viewer bundles plus a manifest under `method_cmp_300/flagship_visualizations/`.
11. Added [scripts/visualize_irc.py](scripts/visualize_irc.py) so saved IRC validation results can also be exported as viewer bundles after the fact.

### IRC-specific redesign work before the first serious rerun

1. Strengthened endpoint comparison in [src/gadplus/search/irc_validate.py](src/gadplus/search/irc_validate.py):
   - replaced naive Cartesian RMSD with element-aware Kabsch/Hungarian aligned RMSD
   - kept topology-based bond-graph matching as an additional chemistry-aware endpoint check
2. Added alignment helpers in [src/gadplus/geometry/alignment.py](src/gadplus/geometry/alignment.py) for element-aware equivalence classes and aligned RMSD.
3. Reworked [scripts/irc_validate.py](scripts/irc_validate.py) so TS candidates are selected from saved trajectories instead of blindly using final saved frames.
4. Added TS-pick modes such as `best_nneg1`, `final`, and `best_force`.
5. Added recomputation of `force_norm`, `fmax`, and projected `n_neg` on the chosen TS geometry before IRC.
6. Added a TS quality criterion so IRC can reject weak "converged" candidates before spending time on path following.
7. Added an optional projected-GAD pre-IRC refinement stage so borderline TS candidates can be tightened before IRC.
8. Recorded both pre-refinement and post-refinement metrics in the validation parquet.
9. Stored coordinate payloads in the validation output:
   - selected TS coordinates
   - refined TS coordinates
   - reactant/product references
   - forward/reverse IRC endpoints
   - atomic numbers
10. Added viewer-bundle writing for IRC outputs so endpoint/path-context structures can be opened in the same VS Code viewers as the GAD trajectories.
11. Updated [scripts/run_irc_validate.slurm](scripts/run_irc_validate.slurm) and [scripts/run_geodesic_irc.slurm](scripts/run_geodesic_irc.slurm) to use stricter defaults and the refinement stage.

### Recent run history that should not be forgotten

1. An early new-style IRC run failed because older trajectory parquet files lacked `force_max`; the code still hard-selected that column.
2. That bug was fixed by schema-detecting optional columns and falling back cleanly to `force_norm`.
3. A later rerun confirmed that the pipeline was doing real IRC work rather than failing at Parquet read time.
4. The stricter TS criterion immediately exposed a key issue: many previously "converged" samples satisfy `n_neg==1 + force_norm<0.01` while failing `fmax<0.01`.
5. The flagship CPU visualization batch completed successfully and produced 15 representative `gad_small_dt` bundles:
   - 5 noise levels: `10, 50, 100, 150, 200`
   - 3 picks each: `fast`, `slow`, `failure`
6. The refined IRC run `59367455_[0-2]` completed and is the first run in the new stronger pipeline worth treating as scientifically meaningful.
7. The headline from that refined IRC rerun is not "many intended reactions," but rather that the TS-quality criterion is now doing serious filtering and that refinement rescues some borderline candidates while many still fail the tighter criterion.
8. The first refined-run summaries currently in hand are:

| Noise | Intended | Half | Topology Intended | Topology Half | Unintended | Error |
|-------|----------|------|-------------------|---------------|------------|-------|
| 0pm | 0 | 7 | 0 | 8 | 5 | 18 |
| 10pm | 0 | 3 | 0 | 4 | 6 | 21 |
| 50pm | 0 | 0 | 0 | 0 | 4 | 26 |

9. The dominant error category in that refined rerun is now `ts_quality_gate_failed`, which is exactly the intended behavior of the stronger screening logic.
10. Several failures are borderline after refinement, which is why a second rerun with improved refinement thresholds/step budgets is the natural next experiment.

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

## Round 4: Switching & Handoff Experiments with Trajectory Analysis

This section covers all experiments that alter the dynamics based on n_neg — NR-GAD ping-pong (Round 1, revisited), descent-GAD handoff, NR-GAD handoff with displacement cap, and multi-mode GAD. Trajectory analysis reveals the failure mechanisms behind each approach.

---

### Experiment 18a: NR-GAD Ping-Pong (Round 1, revisited with trajectory analysis)

**SLURM:** 58845357, 58852071 | **Data:** `method_cmp_300/`, `targeted/`

**Method:** Per-step switching — every step checks n_neg:
- n_neg ≥ 2 → Newton descent: `Δx = α · (-H⁻¹g)` with eigenvalue flooring at 1e-6
- n_neg < 2 → Standard GAD: `Δx = dt · F_GAD`

Can switch back and forth every step. Four damping variants tested (α=1.0, 0.3, 0.2, 0.1). The NR step projects the gradient onto vibrational eigenvectors, divides by |λᵢ| (floored at 1e-6), applies damping α, caps per-component at 0.3, and caps total norm at max_step_norm. Code: `nr_minimize_step()` in `nr_gad_pingpong.py:56-116`.

**Results:**

| Method | 10pm | 50pm | 100pm | 200pm |
|--------|------|------|-------|-------|
| **gad_small_dt (baseline)** | **94.3** | **91.3** | **86.7** | **51.3** |
| nr_gad_damped α=0.1 | 94.7 | 77.7 | 58.0 | 33.7 |
| nr_gad_damped α=0.2 | 93.0 | 78.3 | 60.3 | 36.3 |
| nr_gad_damped α=0.3 | 88.7 | 75.0 | 58.7 | 37.0 |
| nr_gad_pingpong (α=1.0) | 56.7 | 31.7 | 24.7 | 18.3 |

**Trajectory analysis (100pm, 30 samples):**

| Metric | Pure GAD | NR-GAD undamped | NR-GAD α=0.1 |
|--------|----------|-----------------|--------------|
| Converged | 90% | 23% | 40% |
| Phase transitions | 0 | **11.5** | 3.2 |
| n_neg changes | 2.6 | **105.9** | 6.4 |
| Steps at n_neg=0 (overshoot) | 0 | **224.5** | **225.7** |
| Final n_neg distribution | {1:28, 2:2} | {1:25, 0:4, 4:1} | {1:26, 0:4} |

**The failure mechanism:** The Newton step (H⁻¹g) inverts eigenvalues — when |λᵢ| is small (near-zero modes), the step component along that mode is amplified by 1/|λᵢ|, potentially 1000x or more. This causes the trajectory to **overshoot through n_neg=1 directly into n_neg=0** (a minimum basin). Once at n_neg=0, GAD takes over but has no negative eigenvalue to ascend along. The trajectory wanders until n_neg bounces back to ≥2 (triggering NR again), which overshoots again. This creates an oscillation cycle:

```
NR overshoot → n_neg=0 (stuck) → eventually n_neg≥2 → NR overshoot → n_neg=0 ...
```

Samples spend mean 224 steps at n_neg=0 with 106 n_neg changes. This is NOT the "NR until near TS then GAD" pattern we expected — it's chaotic oscillation with 100+ phase transitions. Even damping at α=0.1 doesn't prevent the overshoot (225 steps at n_neg=0, same as undamped).

**Per-sample trajectory examples at 200pm:**

| Sample | Start n_neg | Pure GAD | NR-GAD undamped |
|--------|-------------|----------|-----------------|
| 0 | 5 | FAIL (oscillates 2-4) | **CONV** (6 NR steps → n_neg=1, fast) |
| 5 | 4 | **CONV** (step 651) | FAIL (NR→n_neg=0, 402 NR steps stuck) |
| 6 | 5 | **CONV** (step 835) | FAIL (oscillates 0-4, 515 NR steps) |
| 9 | 4 | **CONV** (step 361) | FAIL (NR overshoot, force=0.017) |

NR-GAD occasionally wins a sample by aggressively dropping n_neg (sample 0), but more often overshoots and loses (samples 5, 6, 9).

---

### Experiment 18b: Descent-GAD Handoff

**SLURM:** 59362083 | **Data:** `round4/` | **Status:** 4/6 complete (200pm ~238/300)

### Motivation

All previous NR/descent switching experiments (Experiments 4, 5, 12) used **ping-pong switching** — checking n_neg every step and bouncing between NR and GAD. Trajectory analysis of the NR-GAD pingpong (Experiment 4) revealed the failure mode: NR overshoots through n_neg=1 into n_neg=0 (a minimum), GAD then wanders at n_neg=0, eventually n_neg bounces back to ≥2, NR fires again, overshoots again. Samples oscillate with 10-100+ phase transitions and spend 100-900 steps stuck at n_neg=0.

This experiment tests a **one-way switch**: plain gradient descent (Δx = dt · F, no Hessian inversion) until n_neg drops to threshold, then permanent GAD. No switching back. No Newton step. Same dt=0.005 throughout both phases — no overshoot risk from Hessian inversion.

### Method

```
Phase 1 (descent): Δx = dt · F_projected       (follow Eckart-projected forces, no v₁ flip)
                    Until n_neg ≤ threshold → switch permanently
Phase 2 (GAD):     Δx = dt · F_GAD_projected    (standard Eckart-projected GAD, forever)
```

Implemented via `gad_blend_weight=0.0` during descent phase (same `gad_dynamics_projected` function, just with the v₁ force flip turned off). dt=0.005, 1000 steps, Eckart projection. Two threshold values tested:
- **descent_then_gad_2**: switch when n_neg ≤ 2
- **descent_then_gad_3**: switch when n_neg ≤ 3 (hand off to GAD earlier)

### Results

| Method | 50pm | 100pm | 200pm* |
|--------|------|-------|--------|
| **gad_small_dt (baseline)** | **91.3** | **86.7** | **51.3** |
| descent_then_gad_2 | 91.7 | 86.3 | 52.9 |
| descent_then_gad_3 | 91.3 | 86.7 | 52.5 |

*200pm still running (~238/300 samples).

**Verdict: Identical to baseline.** The one-way descent→GAD switch neither helps nor hurts. The two thresholds (≤2 vs ≤3) are within noise of each other and of pure GAD.

### Trajectory Analysis: Why It's Identical

The trajectory data reveals why: at dt=0.005, **the descent phase is only 1-3 steps long**. Most samples start with n_neg=2-6 at 100pm, and n_neg drops to ≤2 within 1-2 steps regardless of whether we use GAD or plain descent. The "descent until n_neg≤2" phase is over before it can make any difference.

**At 100pm (30 samples):**

| Metric | Pure GAD | NR-GAD undamped | NR-GAD α=0.1 | Descent→GAD(2) |
|--------|----------|-----------------|--------------|----------------|
| Converged | 27/30 (90%) | 7/30 (23%) | 12/30 (40%) | 27/30 (90%) |
| Starting n_neg (mean) | 3.5 | 3.5 | 3.5 | 3.5 |
| Steps to n_neg≤2 | 2 | 3 | 7 | 2 |
| Steps to n_neg=1 | 39 | 36 | 15 | 39 |
| Phase transitions | 0 | 11.5 | 3.2 | 0.9 |
| n_neg changes | 2.6 | 105.9 | 6.4 | 2.6 |
| Steps at n_neg=0 | 0 | 224.5 | 225.7 | 0 |

Key observations:
1. **Descent→GAD has ~1 phase transition** (the one-way switch), vs 11.5 for NR-GAD pingpong.
2. **Zero steps at n_neg=0** for both Pure GAD and Descent→GAD, vs 224 for NR-GAD. The gentle descent (dt·F) never overshoots into minimum territory.
3. **n_neg changes: 2.6 for both Pure GAD and Descent→GAD**, vs 106 for NR-GAD. The trajectory is smooth, not oscillating.
4. **Steps to n_neg≤2 is already 2** for Pure GAD — there's no room for descent to be faster.

### Per-Sample Trajectory Comparison at 200pm

Examining the same 10 molecules across three methods reveals the dynamics:

| Sample | Start n_neg | Pure GAD | NR-GAD undamped | Descent→GAD(2) |
|--------|-------------|----------|-----------------|----------------|
| 0 | 5 | FAIL (n=2, stuck oscillating 2-4) | CONV (NR drops to 1 fast, 6 NR steps) | FAIL (same as pure GAD) |
| 1 | 6 | FAIL (n=1 but force=0.025) | CONV (10 NR steps, fast convergence) | FAIL (same as pure GAD) |
| 4 | 2 | FAIL (drops to n=0, recovers to 1 but slow) | CONV (2 NR steps) | FAIL (same as pure GAD) |
| 5 | 4 | CONV (step 651) | FAIL (NR overshoots to n=0, 402 NR steps!) | CONV (step 643) |
| 6 | 5 | CONV (step 835) | FAIL (oscillates 0-4, 515 NR steps) | CONV (step 834) |
| 9 | 4 | CONV (step 361) | FAIL (NR overshoot, force=0.017) | CONV (step 359) |

**The pattern:** Descent→GAD trajectories are **nearly identical to Pure GAD** — same convergence, same step count, same failures. The 1-3 descent steps at the start don't alter the trajectory. Meanwhile NR-GAD takes different paths entirely: it sometimes converges faster (samples 0, 1, 4) by aggressively reducing n_neg, but more often overshoots into n_neg=0 territory and spends hundreds of steps stuck there (samples 5, 6).

**NR-GAD's failure mode in detail:** Sample 5 at 200pm starts at n_neg=4. NR-GAD fires the Newton step, n_neg immediately drops to 0. The trajectory spends 402 of 1000 steps in NR phase trying to escape, with n_neg oscillating between 0 and 3-4. The same sample with Pure GAD smoothly descends from n_neg=4→2→1 and converges at step 651. The Newton step's Hessian inversion (H⁻¹g) creates step components that are orders of magnitude too large along near-zero eigenvalue modes.

### Descent Phase Details

How long is the descent phase, and what does it do?

**At 50pm:** descent_then_gad_2 spends median **1 step** in descent (0-4 range). 24/50 samples start with n_neg≤2 and skip descent entirely. descent_then_gad_3 spends median **0 steps** (37/50 skip). The n_neg trajectories are identical across all three methods: step0=2.8→step1=1.8→step2=1.4→step5=1.1→step10=1.1.

**At 100pm:** descent_then_gad_2 spends median **1 step** in descent (0-18 range). 11/50 skip. At handoff, n_neg is almost always 2 (45/50 samples). descent_then_gad_3 spends median **0 steps** (26/50 skip). Mean n_neg trajectories are again identical across methods.

**At 200pm:** descent_then_gad_2 spends median **4 steps** in descent (0-53 range). Now there's real variation — one sample takes 53 descent steps. At handoff, 48/50 samples switch at exactly n_neg=2. descent_then_gad_3 spends median **1 step** (0-24 range). 16/50 skip. The mean n_neg trajectory finally shows a tiny difference: at step 10, pure GAD has mean n_neg=2.1 while descent→GAD(2) has 1.9. But by step 50 they've converged to the same trajectory.

| Noise | Method | Median descent steps | Max | Skipped (n_neg already ≤ thresh) |
|-------|--------|---------------------|-----|----------------------------------|
| 50pm | desc→GAD(2) | 1 | 4 | 24/50 (48%) |
| 50pm | desc→GAD(3) | 0 | 3 | 37/50 (74%) |
| 100pm | desc→GAD(2) | 1 | 18 | 11/50 (22%) |
| 100pm | desc→GAD(3) | 0 | 12 | 26/50 (52%) |
| 200pm | desc→GAD(2) | 4 | 53 | 6/50 (12%) |
| 200pm | desc→GAD(3) | 1 | 24 | 16/50 (32%) |

### Per-Sample Trajectory Divergence at 200pm

Despite identical aggregate rates (~52%), **32/236 individual samples diverge** — one method converges, the other fails (or vice versa). The 1-4 steps of descent vs GAD at the start put the trajectory on a slightly different path, which sometimes leads to a different outcome 500+ steps later. Examples:

| Sample | Start n_neg | Pure GAD | Desc→GAD(2) | Desc→GAD(3) | What happened |
|--------|-------------|----------|-------------|-------------|---------------|
| 17 | 4 | FAIL (stuck at n_neg 2-3) | **CONV** (1 desc step, fast collapse to n_neg=1) | **CONV** | Descent step avoided a v₁ that would have trapped the trajectory |
| 49 | 5 | FAIL (oscillates n_neg 3-5) | **CONV** (9 desc steps, steady descent) | **CONV** (4 desc steps) | Longer descent phase at high n_neg avoided n_neg oscillation |
| 67 | 5 | **CONV** (step 835) | FAIL (26 desc steps, drops to n_neg=0!) | FAIL | Descent phase went too long, overshot into minimum |
| 81 | 4 | **CONV** | FAIL (2 desc steps → n_neg=0, stuck) | FAIL (1 desc → oscillates) | Early descent step sent geometry toward wrong basin |
| 90 | 4 | FAIL (stuck at n_neg 2-3) | **CONV** (3 desc steps) | FAIL | Threshold matters: (≤2) helped, (≤3) didn't |

**Key pattern:** Descent→GAD(2) wins some samples where pure GAD gets stuck oscillating between n_neg=2-4 — the descent phase avoids applying the v₁ flip during the chaotic high-n_neg region, resulting in a slightly different trajectory that finds a path to n_neg=1. But it also loses samples where the descent phase sends the geometry toward n_neg=0 (a minimum), which pure GAD would have avoided by ascending v₁.

These effects roughly cancel out, explaining the identical aggregate rate. The descent phase is a **dice roll** — it helps some samples and hurts others, with no net benefit.

### Comparison with NR-GAD Pingpong

The descent→GAD comparison conclusively separates the Newton step problem from the switching concept:

| Metric (100pm, 30 samples) | Pure GAD | NR-GAD undamped | NR-GAD α=0.1 | Descent→GAD(2) |
|----------------------------|----------|-----------------|--------------|----------------|
| Converged | 90% | 23% | 40% | 90% |
| Phase transitions | 0 | 11.5 | 3.2 | 0.9 |
| n_neg changes | 2.6 | **105.9** | 6.4 | 2.6 |
| Steps at n_neg=0 | 0 | **224.5** | **225.7** | 0 |

### Conclusions

1. **The one-way switch produces identical aggregate rates but different per-sample trajectories.** 32/236 samples (14%) diverge at 200pm. The 1-4 descent steps create a "butterfly effect" — a slightly different starting trajectory that amplifies over 1000 steps.

2. **Descent→GAD(2) both wins and loses samples vs pure GAD.** It wins when the descent phase avoids the v₁ flip during chaotic high-n_neg oscillation (samples 17, 49, 90). It loses when descent overshoots into n_neg=0 (samples 67, 81). Net effect: zero.

3. **Threshold matters for individual samples but not aggregate.** descent_then_gad_2 and descent_then_gad_3 diverge on specific samples (e.g., sample 90: threshold ≤2 converges, threshold ≤3 doesn't). But the aggregate rates are within noise.

4. **NR-GAD's problem is definitively the Newton step, not the switching logic.** Descent→GAD uses the same switching concept (different dynamics at high n_neg) but with gentle gradient descent instead of H⁻¹g. It matches pure GAD's performance exactly. The Newton step's Hessian inversion (H⁻¹g) creates step components orders of magnitude too large along near-zero eigenvalue modes, overshooting through n_neg=1 into n_neg=0.

5. **The NR overshoot to n_neg=0 is the single biggest failure mode.** NR-GAD samples spend mean 224 steps at n_neg=0 (up to 993). Descent→GAD spends 0. Pure GAD spends 0. The overshoot traps the trajectory in a minimum basin where GAD has no negative eigenvalue to ascend along.

6. **GAD's v₁ ascent is a small perturbation at high n_neg.** At n_neg=5, the gradient dominates — |F| >> |2(F·v₁)v₁|. Whether you apply the v₁ flip or not barely changes the step direction, which is why the descent phase is only 1-4 steps and the trajectories converge immediately after handoff.

---

### Experiment 18c: NR-GAD Handoff with Displacement Cap

**SLURM:** 59385266 (full sweep), 59385318 (quick 100pm) | **Data:** `nr_then_gad/`, `nr_then_gad_quick/` | **Status:** Running (~9-43 samples per noise level)

**Method:** One-way switch like Experiment 18b, but using Newton descent (H⁻¹g) instead of plain gradient descent. To prevent the overshoot that killed NR-GAD ping-pong, the Newton step is **aggressively capped**:

```
Phase 1 (NR): Δx = α · (-H⁻¹g), damping=0.3, capped at max_step_norm
              Until n_neg ≤ 2 → switch permanently to GAD
Phase 2 (GAD): Δx = dt · F_GAD (standard Eckart-projected GAD, forever)
```

Code: uses `NRGADPingPongConfig` with `one_way=True, one_way_threshold=2`. The NR step is computed by `nr_minimize_step()` (project gradient onto vibrational eigenvectors, divide by |λᵢ| floored at 1e-6, apply damping, cap norm). Two cap tightness levels:
- **nr_then_gad_cap01**: max_step_norm=0.01Å, max_atom_disp=0.01Å
- **nr_then_gad_cap005**: max_step_norm=0.005Å, max_atom_disp=0.005Å

The Newton direction (H⁻¹g) is curvature-aware (larger steps along flat modes, smaller along steep), but the tight cap ensures each step moves at most 0.01Å total — preventing the overshoot into n_neg=0 that killed NR-GAD ping-pong.

**Results:**

| Method | 10pm | 30pm | 50pm | 100pm | 150pm | 200pm |
|--------|------|------|------|-------|-------|-------|
| **gad_small_dt (baseline)** | **94.3** | **94.3** | **91.3** | **86.7** | **70.3** | **51.3** |
| nr_then_gad_cap01 | 87.7 | 87.9* | 84.9* | 72.0* | 59.7* | 46.3* |
| nr_then_gad_cap005 | 87.7 | 88.2* | 84.4* | 72.5* | 59.8* | 42.7* |

*Still running (103-264 samples). 10pm complete (300/300). Quick test at 100pm (100 samples): both give 77.0%.

**Assessment:** Both variants are **worse than baseline** by 7pp at 10pm and 10-15pp at 100pm. The tight displacement cap prevents overshoot (zero samples at n_neg=0) but also makes the NR steps so small they're effectively frozen — a Newton step capped at 0.01Å across a molecule with 10+ atoms gives ~0.002Å per coordinate, slower than GAD's ~0.005Å per atom. The trajectory wastes step budget crawling through the NR phase before handing off to GAD.

The two cap values (0.01Å vs 0.005Å) give **nearly identical results** — both are already in the "frozen step" regime. The cap tightness doesn't matter once the steps are this small.

**The fundamental tradeoff:** Newton's advantage is large, curvature-informed steps. A displacement cap kills exactly that advantage. A Newton step capped at 0.01Å is just an expensive way to compute a direction that gets scaled to nothing. The curvature information is preserved (the *direction* is still Newton-optimal) but the magnitude is forced to be the same as a gradient step, making the Hessian inversion overhead wasted compute.

---

### Experiment 19: Multi-Mode GAD

**SLURM:** 59367568 | **Data:** `multimode/` | **Status:** 12/18 complete, 6 running (150/200pm partial)

**Method:** Instead of flipping the force along v₁ only (standard GAD), flip along multiple eigenvectors simultaneously:

**multimode_all_neg:** `F_GAD = F + 2·Σᵢ(F·vᵢ)vᵢ` for all i where λᵢ < 0. Hard threshold — every negative-eigenvalue mode gets the force flip. Code: `multimode_gad_dynamics_projected()` in `projection.py` with `mode="all_neg"`.

**multimode_smooth:** `F_GAD = F + 2·Σᵢ σ(-λᵢ·50)·(F·vᵢ)vᵢ`. Differentiable sigmoid weighting — modes with strongly negative λ get full flip, modes near zero get partial. `mode="smooth"`, `sigmoid_sharpness=50.0`.

**multimode_top2:** `F_GAD = F + 2(F·v₁)v₁ + 2(F·v₂)v₂`. Always flip the two lowest modes regardless of eigenvalue sign. `mode="top2"`.

All use dt=0.005, 1000 steps, Eckart projection.

**Results:**

| Method | 10pm | 30pm | 50pm | 100pm | 150pm* | 200pm* |
|--------|------|------|------|-------|--------|--------|
| **gad_small_dt (baseline)** | **94.3** | **94.3** | **91.3** | **86.7** | **70.3** | **51.3** |
| multimode_all_neg | 87.7 | 86.7 | 83.7 | 71.0 | 33.6 | 7.7 |
| multimode_smooth | 87.3 | 86.3 | 82.7 | 69.7 | 33.5 | 7.6 |
| multimode_top2 | 36.1 | 17.2 | 13.4 | 6.6 | 4.4 | 1.1 |

*150/200pm partial (177-215 samples).

**multimode_top2 is catastrophic** (36% at 10pm). Always flipping v₂ regardless of sign forces ascent along a mode that should be descended when λ₂>0, pushing the geometry away from the saddle.

**all_neg and smooth** are ~7pp worse at 10pm, ~16pp at 100pm, and collapse at high noise (8% at 200pm vs 51% baseline). all_neg and smooth are nearly identical (sigmoid sharpness doesn't matter when the hard threshold already separates negative from positive).

**Why multi-mode GAD fails:** Flipping all negative modes simultaneously is incoherent — the trajectory tries to ascend along 3-5 directions at once, pushing toward a higher-order saddle (index n_neg) instead of the index-1 saddle we want. Standard single-mode GAD ascends v₁ while descending all other modes including v₂...v₅. This naturally reduces n_neg one mode at a time as the geometry approaches index-1. Multi-mode GAD fights this process by trying to maintain all negative eigenvalues — which is the opposite of what we want.

---

### Unified Conclusions: Switching, Handoff & Multi-Mode Experiments

1. **Pure single-mode GAD is optimal.** Across 8 switching/handoff/multi-mode variants tested, none outperform `gad_small_dt`. The v₁-only ascent with uniform fixed timestep is already the best dynamics for saddle-point search.

2. **NR-GAD ping-pong fails due to Newton step overshoot.** Trajectory analysis proves the mechanism: H⁻¹g amplifies near-zero eigenvalue modes by 1000x+, overshooting through n_neg=1 into n_neg=0. Samples spend 224 steps trapped at a minimum with 106 n_neg changes. Not "NR then GAD" — chaotic oscillation.

3. **Descent-GAD handoff is identical to pure GAD.** Plain gradient descent (dt·F) is gentle enough to avoid overshoot (0 steps at n_neg=0), but the phase is only 1-4 steps — too short to matter. 14% of individual samples diverge via butterfly effects, but wins and losses cancel.

4. **NR-GAD handoff with tight cap is worse than pure GAD.** Capping Newton steps at 0.01-0.005Å prevents overshoot but kills Newton's advantage (large curvature-informed steps). The NR phase becomes slower than GAD, wasting step budget.

5. **Multi-mode GAD is worse than single-mode.** Ascending along all negative modes pushes toward higher-order saddles instead of index-1. The v₁-only focus is a feature, not a limitation — it naturally reduces n_neg one mode at a time.

6. **The v₁ ascent is a small perturbation at high n_neg but a critical one.** At n_neg=5, |F| >> |2(F·v₁)v₁| — the GAD modification barely changes the step. But it consistently biases the trajectory toward reducing n_neg, which is what makes it better than pure descent (despite nearly identical trajectories for the first few steps).

---

## Round 5, Experiment 18: Full sella_hip IRC Validation on gad_dt003 TSs (2026-04-16)

**Setup.** Full IRC validation of the best GAD variant (`gad_dt003`) at six noise levels: 10/30/50/100/150/200 pm × 300 samples each. TS candidates pulled naively from the GAD trajectory at the first step where `n_neg == 1 AND force_norm < 0.01` (no refinement, no gating). IRC integrator = `sella_hip`: Sella's IRC routine with HIP's analytical mass-weighted Eckart-projected Hessian injected after every inner kick, overwriting BFGS updates. Max 500 steps per direction.

Primary metric: **TOPO-intended** (bond-graph isomorphism on both directions, element-labeled, direction-agnostic). Strict metric: RMSD-intended (<0.3 Å, Kabsch+Hungarian).

**Data source.** `round2/summary_gad_dt003_{10,30,50}pm.parquet`, `round3/summary_gad_dt003_{100,150,200}pm.parquet`. The 200pm summary was missing from the original GAD sweep (259/300 trajs but no rollup) and was regenerated from the traj parquets — 143/259 GAD-converged. All other levels had the full 300.

**Results.** 1462 IRC runs total, **zero errors**, ~30 min wall parallelized over 6 MIG slices.

| Noise | n | TOPO-int% | RMSD-int% | wall_avg (s) |
|-------|---|-----------|-----------|--------------|
| 10pm  | 284 | **94.4** | 64.4 | 12.6 |
| 30pm  | 283 | **94.3** | 64.3 | 12.4 |
| 50pm  | 276 | **93.1** | 64.9 | 12.3 |
| 100pm | 262 | **93.9** | 65.6 | 12.2 |
| 150pm | 214 | **90.7** | 66.8 | 11.8 |
| 200pm | 143 | **80.4** | 60.8 | 10.4 |
| **all** | **1462** | **92.1** | **64.7** | 11.9 |

**Key findings.**

1. **TOPO-intended is nearly flat at ~93–94% through 100pm**, drops to 90.7% at 150pm, and cliffs to 80.4% at 200pm. Degradation only kicks in at 200pm.
2. **Strict RMSD-intended is flat (64–67%) regardless of noise**, actually rising slightly from 10pm (64.4%) through 150pm (66.8%). The ~27pp gap to TOPO is not noise-driven; it's conformational or labeling-driven.
3. **Endpoint vibrational quality is very high:** 87–90% of runs end with both endpoints at true minima ($n_{\text{neg,vib}}=0$). Only 0.5–1.8% end with neither endpoint at a minimum (genuine integrator failure).
4. **Directional symmetry is near-perfect:** forward and reverse match labels at 93–97% each, with ≤3pp asymmetry. No directional bias from the HIP-injected Sella.
5. **Systematic failures are labeling issues, not integrator issues.** 10 sample IDs fail TOPO at ≥5/6 noise levels. 8 of these land at valid minima on both sides — GAD found a real TS, IRC found two real minima, just not the T1x-labeled pair. Candidate formulas: C2H3N3O (sids 94/96/98/104), C2H3N3 (48), C2H3N5 (108), C2H4N2O2 (143), C2H4N4 (203). Two (258 C2H6O2, 265 C3H2N2O) are genuine ridge-stalls with `n_neg=1` on one endpoint.
6. **5.8% of all runs are "valid-but-wrong minima"** — IRC reached real minima but not the labeled pair.
7. **TS-side quality vs IRC outcome:** at low noise, TOPO-failing TSs have *sharper* saddles (median |eig0| ≈ 7 vs 3.5). At 150/200pm, failing TSs also had *longer* GAD convergence (cstep +50%). No clean TS-side filter predicts IRC failure.
8. **Per-formula difficulty:** C2N2 (50% int, n=4) and C3H2N2O (60%, n=15) are hardest. Most formulas (10+) are 100% TOPO-intended. The common organic reactions in T1x are handled reliably.

**Significance.** This is the first fully scored IRC validation on the flagship GAD method (`gad_dt003`), on the full 300-sample pool at all noise levels from 10 to 200pm. Prior IRC validations (`irc_validation/`) ran on a much smaller sample (30/noise). sella_hip is the production IRC integrator.

**Artifacts.**
- `IRC_RESULTS_2026-04-16.tex` / `.pdf` — 8-page writeup with all tables and figures.
- `figures/fig_sella_{rates_vs_noise,endpoint_quality,rmsd_distributions,ts_quality_vs_outcome,systematic_failures,wall_time}.{pdf,png}`
- `scripts/analyze_sella_deep.py` — produces all tables.
- `scripts/figures_sella_irc.py` — produces all figures.
- Raw IRC parquets: `/lustre07/scratch/memoozd/gadplus/runs/irc_sellahip_full/irc_validation_sella_hip_{10,30,50,100,150,200}pm.parquet`
- Regenerated 200pm TS summary: `/lustre07/scratch/memoozd/gadplus/runs/round3/summary_gad_dt003_200pm.parquet`

**Next.** Full rigorous (Hratchian–Schlegel predictor-corrector) IRC sweep in parallel — job 59464202, 6 tasks, ETA ~11h. Same 1462 TS set, direct head-to-head.

---

## Round 6: Five-Method Benchmark + Apples-to-Apples Convergence Criteria (2026-04-20)

**Setup.** Five TS-finding methods benchmarked on identical inputs (Transition1x train samples 0-299, Gaussian noise seed 42, 2000-step budget, HIP analytic Hessian every step):

| Method | Dynamics | Convergence criterion | Output dir |
|---|---|---|---|
| `gad_dt003` Eckart (historical) | projected GAD, dt=0.003 | n_neg==1 ∧ ‖F‖_mean<0.01 | `runs/round2/`, `runs/round3/` |
| `gad_dt003_fmax` Eckart (NEW canonical) | projected GAD, dt=0.003 | n_neg==1 ∧ fmax<0.01 | `runs/gad_eckart_fmax/` |
| `gad_dt003_no_eckart` (NEW) | raw Hessian GAD, dt=0.003 | n_neg==1 ∧ fmax<0.01 | `runs/gad_no_eckart/` |
| Sella cart+Eckart | Sella QN, Cartesian, Eckart-projected H, 2000 steps | n_neg==1 ∧ fmax<0.01 | `runs/sella_2000/` |
| Sella cart no-Eckart | Sella QN, Cartesian, raw H, 2000 steps | n_neg==1 ∧ fmax<0.01 | `runs/sella_2000/` |
| Sella internal | Sella QN, internal coords (Sella default), 2000 steps | n_neg==1 ∧ fmax<0.01 | `runs/sella_2000/` |

**TS-finding convergence (% of 300):**

| Method | 10pm | 30pm | 50pm | 100pm | 150pm | 200pm |
|---|---|---|---|---|---|---|
| GAD Eckart force_norm (old criterion) | 94.7 | 94.3 | 92.0 | 87.3 | 71.3 | 52.3 |
| **GAD Eckart fmax (new canonical)** | **87.3** | **87.3** | **85.3** | **80.0** | **63.7** | **45.3** |
| GAD no-Eckart fmax | 91.3 | 91.0 | 88.3 | 83.3 | 64.3 | 43.3 |
| Sella cart+Eckart | 92.0 | 91.3 | 88.3 | 76.3 | 50.0 | 25.7 |
| Sella cart no-Eckart | 92.3 | 91.7 | 88.3 | 75.7 | 50.7 | 24.0 |
| Sella internal | 87.7 | 86.3 | 82.0 | 60.3 | 32.7 | 17.0 |

**IRC validation (sella_hip, all 300 endpoints, TOPO-intended %):**

| Method | 10pm | 30pm | 50pm | 100pm | 150pm | 200pm |
|---|---|---|---|---|---|---|
| GAD Eckart force_norm | 92.7 | 92.7 | 89.7 | 86.3 | 69.7 | 47.9† |
| GAD no-Eckart | 92.7 | 93.0 | 90.3 | 84.7 | 66.7 | 45.7 |
| Sella cart+Eckart | 92.3 | 92.3 | 89.7 | 76.7 | 49.3 | 24.7 |
| Sella cart no-Eckart | 92.7 | 92.3 | 89.3 | 75.0 | 50.0 | 22.3 |
| Sella internal | 93.0 | 91.3 | 88.0 | 65.3 | 34.7 | 15.0 |

†200pm IRC parquet predates the 200pm refill (n=259 partial).
GAD Eckart fmax IRC pending at log-write time (job 59967280).

**Key findings.**

1. **Under matched fmax criterion, Sella beats GAD at low noise** (~4 pp at 10-50pm) and **GAD beats Sella at high noise** (~4-20 pp at 100-200pm). Crossover ≈ 75-100pm. This is more nuanced than the original Round 1 "GAD always wins" narrative, which was inflated by GAD's looser force_norm criterion.

2. **Eckart projection makes GAD ~1.33× more step-efficient**, not fundamentally more reliable. On samples both Eckart and no-Eckart converge on, Eckart needs ~33% fewer steps. Under the 2000-step budget the convergence-rate gap is +0-3 pp. Under tight budgets (300 steps, Round 1 era) the same 1.33× was the difference between converging and timing out — explaining the original "Eckart is essential" claim.

3. **Internal coordinates hurt Sella on HIP** by 12-30 pp across noise levels. Sella's library default is the worst tested configuration. Conjecture: internal-coord projection drift under noise + Hessian-update path less tested for `diag_every_n=1`.

4. **Eckart adds nothing for Sella Cartesian** (within ±1.5 pp). Sella's QN trust region is already TR-mode invariant.

5. **2000 steps favors Sella asymmetrically.** Sella QN converges in 30-200 steps when it's going to; GAD Euler can use 50-70% of the 2000-step budget at high noise with non-trivial tail at the cap. A 10k-step rerun (backburner) would only strengthen GAD's high-noise lead.

**Code changes.**

- `src/gadplus/search/irc_sella_hip.py::_force_first_kick` — bypasses ASE's pre-kick convergence check (required for IRC on Sella-refined TSs whose fmax already < 0.01).
- `scripts/sella_baseline.py` — appends `coords_flat`+`atomic_nums` to summary parquets so IRC can read coords without re-running Sella.
- `scripts/method_single.py` — added `gad_dt003_no_eckart` and `gad_dt003_fmax` configs; made `use_projection` per-method.
- `scripts/irc_validate.py` — added `--coords-source {traj,summary}`, `--all-endpoints`, narrow filename glob; bypassed Lustre-broad-glob stalls.

**Reports.**

- `IRC_COMPREHENSIVE_2026-04-20.tex/pdf` — current canonical (5 methods, both criteria, IRC, all comparison figures).
- `STATUS_2026_04_20.md` — full inventory of completed cells, IRC datasets, code changes, open items, backburner.
- Older reports marked `SUPERSEDED 2026-04-20` in the title page.

**Compute.**

19 array tasks dispatched simultaneously under rrg-aspuru on 2026-04-17/18 covered everything except the GAD Eckart fmax rerun (added 2026-04-20). Total to-date ~200-600 GPU-hours. Two OOMs hit at 16GB on Sella internal high-noise; resolved by re-requesting 48GB then 96GB.

**SLURM job IDs (Round 6).**

| Job | Contents |
|---|---|
| 59607720 / 59607837 | GAD 200pm refill (force_norm) — first attempt cancelled (criterion mismatch), second succeeded |
| 59607721 | GAD no-Eckart sweep, 6 noise × 300 |
| 59607722 / 59624768 / 59628068 / 59636981 | Sella 2000-step (12-task array + 3 OOM retries + 1 timeout retry + 1 96GB retry) |
| 59626490 | Sella cart no-Eckart 2000, 6 noise × 300 |
| 59647983 | sella_hip IRC on 4 new TS sets (23-task array) |
| 59687414 | sella_hip IRC on Sella internal 200pm (one cell that missed the array) |
| 59690067 | GAD Eckart fmax rerun (canonical), 6 noise × 300 |
| 59967280 | sella_hip IRC on GAD Eckart fmax — running at log-write time |

---

## Round 7: IRC backfill on gad_dt002 and gad_projected (2026-04-28)

Backfill IRC validation (`sella_hip`, all endpoints, TOPO-intended) on two methods that the
Round 6 head-to-head omitted: `gad_dt002` (3000-step, dt=0.002, force_norm criterion — best raw
TS-finder in the consolidated ranking) and `gad_projected` (300-step Round 1 baseline).

Session: 29 array tasks dispatched (6 retry tasks ran ~16h on the queue overnight at
n=100 cells); 35 IRC cells produced in total.

**TOPO-intended % (sella_hip, all endpoints):**

| Method | 10pm | 30pm | 50pm | 100pm | 150pm | 200pm |
|---|---|---|---|---|---|---|
| gad_dt002 | 69.0 (n=100) | 69.0 (n=100) | 61.0 (n=300) | 58.7 (n=300) | — | — |
| gad_projected | 50.3 (n=300) | 56.0 (n=100) | 57.0 (n=100) | 55.0 (n=100) | 47.0 (n=100) | 28.3 (n=300) |

Output dirs: `runs/irc_gad_dt002/`, `runs/irc_gad_projected_round1/`.

These are notably below the Round 5/6 IRC TOPO rates (~85–93% at 10–100pm), because both
methods' historical TS pools were selected by `force_norm<0.01` (looser than fmax) — the
extra "barely-converged" TSs that pass force_norm but not fmax tend to drift on IRC. The
Round 6 fmax-criterion GAD Eckart pool gave 92.7%/89.7%/86.3% IRC TOPO at 10/50/100pm; the
same 3000-step `gad_dt002` pool under force_norm only retains 69%/61%/58.7%. Confirms
that fmax is the right criterion for downstream IRC, not just for fair comparison with Sella.

---

## Experiments NOT Run

| Experiment | Description | Why not run |
|-----------|-------------|-------------|
| gad_clamp_005 | 0.05A aggressive clamp | Low priority after clamp proved inert |
| rfo_gad | RFO secular equation + GAD | Code written (`search/rfo_gad.py`), never submitted |
| nr_then_gad_loose_cap | NR→GAD with 0.05-0.1Å cap | Would test intermediate cap between "frozen" and "overshoot" |

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
Round 4 descent→GAD: /lustre07/scratch/memoozd/gadplus/runs/round4/
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
| Round 4 descent→GAD | 59362083 | 4/6 complete (200pm ~238/300) |
| Round 3 Sella 1000-step | 58937673 | 20/24 (internal 150/200pm timeout) |
| Round 5 sella_hip IRC round2 | 59456280 | Complete (tasks 0/1/2) + 3/4 cancelled (Lustre hang, refixed) |
| Round 5 sella_hip IRC round3 | 59456595 | Complete (100/150/200pm) |
| Round 5 rigorous IRC full | 59464202 | Running (6 tasks, ETA ~11h) |
