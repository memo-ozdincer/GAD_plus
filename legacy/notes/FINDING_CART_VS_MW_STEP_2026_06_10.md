# Finding: Cartesian GAD step vs mass-weighted GAD step (2026-06-10)

## What changed

`src/gadplus/projection/projection.py` (commit `61119c2`) now builds the GAD
**step** in **Cartesian** coordinates instead of mass-weighted (MW) coordinates,
for all three projected-GAD functions:

- `gad_dynamics_projected`
- `multimode_gad_dynamics_projected`
- `preconditioned_gad_dynamics_projected`

### Before (mass-weighted step)
The step was computed in MW space and returned via `gad_vec = sqrt_m * dq`,
i.e. the dynamics ran in mass-weighted coordinates.

### After (Cartesian step)
The step is built in Cartesian space using the canonical Cartesian GAD flip from
`core/gad.py::compute_gad_vector_tracked`:

```
F_GAD = F_cart + 2·w·(−F_cart·v_cart)·v_cart      (w=1 full GAD, w=0 pure descent)
```

where
- `F_cart = sqrt_m * (P @ (sqrt_m_inv * f))` — TR-projected Cartesian force
  (same convention as `project_vector_to_vibrational`),
- `v_cart = normalize(sqrt_m_inv * (P @ v))` — Cartesian image of the MW guide vector.

No `sqrt_m * dq` back-transform. For `multimode`/`preconditioned`, the per-mode
flip / `1/|λ|` scaling act along the Cartesian mode directions
`U = normalize(M^{-1/2}·v_i)`.

### Invariants preserved (unchanged)
- Eckart projection still applied (`_eckart_projector`).
- `vib_eig` and **n_neg** still computed on the MW Eckart-projected vibrational
  Hessian — TS-convergence semantics untouched.
- The returned `v_proj` for mode-tracking continuity is still in MW space
  (so tracking still compares against the MW eigenvectors from `vib_eig`).
- Differentiability: autograd verified to flow through all three (forces→gad_vec).

## Result: train-300 A/B (method `gad_dt003_fmax`, fmax<0.01, 2000 steps)

Convergence = `final_n_neg == 1 AND final_force_max < 0.01`.

| noise (pm) | n | MW conv % | Cart conv % | Δ (pp) | MW avg steps | Cart avg steps |
|---|---|---|---|---|---|---|
| 10  | 300 | 87.3 | 86.3 | −1.0 | 413  | 458  |
| 30  | 300 | 87.3 | 85.7 | −1.6 | 561  | 612  |
| 50  | 300 | 85.3 | 83.0 | −2.3 | 703  | 749  |
| 100 | 300 | 80.0 | 77.0 | −3.0 | 948  | 989  |
| 150 | 300 | 63.7 | 61.0 | −2.7 | 1269 | 1267 |
| 200 | 300 | 45.3 | 45.0 | −0.3 | 1527 | 1519 |
| **overall** | **1800** | **74.83** | **73.0** | **−1.83** | | |

**Takeaway:** the Cartesian step is consistently *slightly worse* than the
mass-weighted step (−1.8pp overall, peaking at −3.0pp @ 100pm) and needs a few
more steps on average. Mass-weighting was modestly helping convergence — it
biases steps toward lighter atoms (H), which apparently helps reach the saddle.

### Ground-truth pointers (train-300)
- MW (baseline, original code): `runs/gad_eckart_fmax/summary_gad_dt003_fmax_*pm.parquet`
- Cart (commit 61119c2):        `runs/gad_eckart_fmax_cart/summary_gad_dt003_fmax_*pm.parquet`
  - Submitted by `scripts/run_gad_eckart_fmax_cart.slurm`, SLURM array `62743290`.
  - Logs: `/lustre07/scratch/memoozd/gadplus/logs/gadfmaxcart_62743290_*.{out,err}`
- (All paths relative to `/lustre07/scratch/memoozd/gadplus/`.)

## Result: test-split A/B (287 samples, method `gad_dt003_fmax`, fmax<0.01, 2000 steps)

Per the test-split-only rule for new sweeps, both arms were rerun on the T1x
**test** split (287 samples). Convergence = `final_n_neg == 1 AND final_force_max < 0.01`.

| noise (pm) | n | MW conv % | Cart conv % | Δ (pp) | MW avg steps | Cart avg steps |
|---|---|---|---|---|---|---|
| 10  | 287 | 88.2 | 88.2 |  0.0 | 442  | 462  |
| 30  | 287 | 87.1 | 86.1 | −1.0 | 604  | 633  |
| 50  | 287 | 84.7 | 82.6 | −2.1 | 764  | 800  |
| 100 | 287 | 69.3 | 66.9 | −2.4 | 1177 | 1201 |
| 150 | 287 | 51.6 | 51.6 |  0.0 | 1487 | 1470 |
| 200 | 287 | 34.8 | 38.3 | **+3.5** | 1689 | 1658 |
| **overall** | **1722** | **69.28** | **68.93** | **−0.35** | | |

**Takeaway:** same direction as train-300 but much milder — overall only −0.35pp
(vs −1.83pp on train). Cartesian still loses in the mid-noise range (50–100pm,
−2 to −2.4pp) but **ties at 10/150pm and wins at 200pm (+3.5pp)**, so the overall
gap nearly washes out on the held-out test set. Mass-weighting's edge is real but
small and concentrated at moderate noise.

| Arm | Code | Output dir | SLURM | Script |
|---|---|---|---|---|
| MW   | worktree @ `be59e16` (`sqrt_m*dq`), PYTHONPATH override | `runs/gad_eckart_fmax_mw_test/`   | `62789330` | `scripts/run_gad_eckart_fmax_mw_test.slurm` |
| Cart | main @ `61119c2`                                       | `runs/gad_eckart_fmax_cart_test/` | `62789329` | `scripts/run_gad_eckart_fmax_cart_test.slurm` |

The MW arm ran from the pinned worktree
`/lustre07/scratch/memoozd/gadplus/worktrees/mw_baseline` (detached at
`be59e16`), reusing the main venv for dependencies but shadowing the editable
`gadplus` install via `PYTHONPATH=$WT/src` so it imports the pre-Cartesian code.
Logs: `/lustre07/scratch/memoozd/gadplus/logs/gadfmax{cart,mw}test_*.{out,err}`.

## Why is the difference so small? (the labels were wrong)

The near-zero A/B gap is diagnostic: **the old `sqrt_m * dq` path was never doing
mass-weighted *dynamics* in the per-atom step.**

- **Pure descent (w=0): old and new are bit-identical** (max abs diff ~3e-11).
  The old descent step is `gad_vec = sqrt_m ⊙ (P @ (sqrt_m_inv ⊙ f)) = M^½ P M^{-½} f`.
  The mass factors cancel inside the vibrational subspace → it collapses to the
  **Cartesian** projected force, mass-independent. `f_cart` is the same expression.
- So the **descent term — which dominates the trajectory** (the flip shrinks near
  convergence) — was already Cartesian in both versions.
- The only thing that ever differed is the **flip term's** mass weighting
  (old: `∝ sqrt_m·v_proj`; new: `∝ normalize(sqrt_m_inv·v_proj)`). Full-GAD
  old-vs-new cosine ≈ 0.86 with identical base → tiny net effect.

A **genuinely mass-weighted** step integrates `q̇` in `q = M^½ x`, so the physical
displacement is `dx = M^{-½} dq = sqrt_m_inv ⊙ dq` — the *inverse* of what the old
code did. On a C/N/O/H example (`sqrt_m * dq` vs `sqrt_m_inv * dq`):

| comparison | cosine | ‖·‖ ratio |
|---|---|---|
| old (`sqrt_m·dq`) vs new Cartesian | +0.86 | 0.87 |
| old vs true-MW (`sqrt_m_inv·dq`)   | +0.74 | 1.47 |
| new vs true-MW                     | +0.66 | 1.68 |

So both arms of the A/B above were really "Cartesian descent + slightly different
flip." The genuinely mass-weighted variant (`sqrt_m_inv·dq`) is ~35° off and ~1.5×
larger, and has **not** been benchmarked — it corresponds to the "just fix the
back-transform" option that was not taken. To actually test whether mass-weighting
helps, A/B should be `sqrt_m_inv·dq` (true MW) vs `f_cart` (Cartesian).

## Decisions / caveats
- For `preconditioned`, the `1/|λ|` scaling now acts on Cartesian mode directions
  `U` (not perfectly Euclidean-orthonormal), so the preconditioner is `U|Λ|⁻¹Uᵀ`
  rather than an exact spectral inverse. Chosen for consistency with the
  plain/multimode Cartesian step and to preserve the `O(F/|λ|)` step scale
  (the `sqrt_m_inv·dq` alternative shrinks steps by ~1/m and breaks `eig_floor`).
- Only `gad_dt003_fmax` (plain GAD) has been rerun so far. multimode/precond
  variants changed too but are not yet rerun.
