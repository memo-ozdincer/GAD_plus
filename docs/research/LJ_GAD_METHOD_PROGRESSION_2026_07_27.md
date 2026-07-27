# Paired LJ7 comparison of the GAD method progression

## Result in one sentence

On the stated analytic LJ7 capture-basin benchmark, the current pointwise
intrinsic method was the strongest of the four documented formulations: it
reached the strict TS gate and two valid downhill endpoints for all
`288/288` paired starts, including all starts at `0.40 sigma`, with a median
of `17` Hessian evaluations. The hard and historical smooth gates each reached
`283/288` with a median of `240` evaluations; ordinary GAD reached `115/288`.

This is evidence about this controlled equal-mass LJ7 problem only. It does
not establish chemical performance, IRC topology recovery, or a benefit from
mass-metric stepping for unequal-mass systems.

## Question and specified comparison

The derivation in `docs/POINTWISE_INTRINSIC_GAD.md` gives four successive
local fields:

1. **Ordinary GAD:** full one-mode ascent at every geometry.
2. **Hard gate:** force descent if `lambda_2 < 0`; otherwise full GAD.
3. **Historical smooth gate:** the same interpolation with
   `w = sigmoid(50 lambda_2)`.
4. **Pointwise intrinsic GAD:** normalized smooth spectral density, normalized
   `lambda_2` gate, and a closed-form regularized eigenvector-following step.

The claim tested here was intentionally narrow: on common LJ7 starts, does the
last formulation improve strict TS recovery and evaluation cost over its three
predecessors? The experiment did not tune a method after seeing its outcomes.

## Controlled protocol

The potential was analytic reduced-unit LJ7 (`epsilon=sigma=1`), with seven
identical particles assigned the same hydrogen mass (`1.008`) for the Eckart
projection. A deterministic index-one reference saddle was generated from a
mode-pushed pentagonal-bipyramid minimum.

Two panels used exactly the same seeded Cartesian Gaussian perturbations:

- `saddle`: perturb the reference saddle directly;
- `pushed`: perturb the mode-pushed minimum used to reach that saddle.

For each of `48` seeds, each panel, and each noise level `0.10`, `0.20`, and
`0.40 sigma`, all four methods received the identical start: `4 x 2 x 3 x 48
= 1152` trajectories.

The three Euler predecessors used their documented historical controls:

```text
instantaneous lowest mode (no mode tracking)
dt = 0.005
per-atom displacement cap = 0.005
maximum = 8000 Hessian evaluations
```

The hard gate was the *instantaneous* pointwise rule, not the historical
one-way descent-then-lock heuristic. Intrinsic GAD used
`spectral_temperature=0.01`, `step_fraction=0.05`, and a `200`-update limit.
Its reported evaluation count includes the final convergence evaluation.

Every candidate used the common strict terminal criterion

```math
n_{\mathrm{neg}}^{(10^{-4})}=1
\qquad\text{and}\qquad
\lVert F\rVert_\infty<0.01.
```

Every strict candidate was then displaced in both signs of its projected
unstable mode and each branch was minimized by analytic-gradient L-BFGS-B. A
candidate was endpoint-valid only when both endpoints had projected index zero
and `fmax < 1e-5`. This is the LJ `IRC_TOPO`-like endpoint screen used in this
project; it is not a discretized IRC calculation.

## Results

### Aggregate across panels and noise

| Method | Strict TS | Two valid endpoints | Median evaluations among strict TS | Median evaluations over all starts |
|---|---:|---:|---:|---:|
| ordinary GAD | 115/288 (39.9%) | 115/288 (39.9%) | 494 | 8000 |
| hard gate | 283/288 (98.3%) | 283/288 (98.3%) | 240 | 241 |
| historical smooth `lambda_2` gate | 283/288 (98.3%) | 283/288 (98.3%) | 240 | 241 |
| pointwise intrinsic GAD | 288/288 (100.0%) | 288/288 (100.0%) | 17 | 17 |

No endpoint-valid candidate from any method was near-flat under
`lambda_2 / s_H < 0.01` or fragmented under the declared `1.5 sigma`
connectivity graph. Thus the comparison is not being driven by accepting
visibly poor intrinsic candidates.

### Recovery as noise increases

Each cell below pools the `48` saddle-centered and `48` pushed starts.

| Method | `0.10 sigma` | `0.20 sigma` | `0.40 sigma` |
|---|---:|---:|---:|
| ordinary GAD | 64/96 (66.7%) | 34/96 (35.4%) | 17/96 (17.7%) |
| hard gate | 96/96 (100.0%) | 96/96 (100.0%) | 91/96 (94.8%) |
| historical smooth `lambda_2` gate | 96/96 (100.0%) | 96/96 (100.0%) | 91/96 (94.8%) |
| pointwise intrinsic GAD | 96/96 (100.0%) | 96/96 (100.0%) | 96/96 (100.0%) |

At `0.40 sigma`, the median evaluations among successful trajectories were
`1270` for ordinary GAD, `474` for the hard gate, `469` for the historical
smooth gate, and `29` for intrinsic GAD. The current formulation therefore
won both the success criterion and the evaluation-cost criterion on the
highest planned noise cell.

The hard and smooth gates had identical success/failure outcomes on all 288
paired starts. Their finite sigmoid transition affected the trajectory length
on 30 starts, but not this benchmark's terminal classification. That equality
is a useful result rather than evidence that the two fields are mathematically
identical: most relevant LJ7 geometries were sufficiently far from
`lambda_2=0` for the sigmoid to be nearly binary.

### Event selectivity is secondary

The reference event was recovered less often as noise increased, even when the
strict TS and endpoint tests passed. This is expected in a diversity-oriented
terminal optimizer: it indicates capture by other valid local index-one
basins. The study therefore ranks methods first by strict TS and endpoint
validity, then by cost; it does not treat reference-event recovery as a
primary measure of quality.

## What “best” means here

Within this exact protocol, pointwise intrinsic GAD is best supported because
it is the only method that simultaneously has:

1. 100% strict and endpoint-valid recovery through `0.40 sigma`;
2. no accepted near-flat or fragmented candidates under the declared screens;
3. a median of 17 Hessian evaluations, about 14 times fewer than either gated
   Euler predecessor; and
4. a strictly local, position-only update without mode tracking, fixed Euler
   step tuning, or a posteriori clipping.

This is not a universal dominance theorem. The Euler methods received their
best documented LJ controls and a much larger iteration budget; they may be
preferable under other computational-cost models or on other surfaces.

## Mass-convention limitation

The historical Euler fields in this comparison are Cartesian final updates:
they use mass weighting to form the Eckart projection and vibrational modes,
then step in the projected Cartesian GAD force. The intrinsic update is
genuinely mass-metric because its closed-form step is formed in mass-weighted
coordinates and back-transformed by `M^{-1/2}`.

All LJ7 particles here have the same mass, so `M^{-1/2}` is a scalar multiple
of the identity and its dynamical effect is absorbable into a step scale. This
study therefore does **not** test whether final-step mass weighting is better.
A heterogeneous-mass LJ ablation or a molecular experiment is required for
that question; neither was run here.

## Reproducibility

- Slurm job: `1946071`, account `def-aspuru`, one packed 192-core CPU node;
  completed in 6m21s with no task exceptions.
- Raw artifacts:
  `/scratch/memoozd/gadplus/runs/lj-method-progression-1946071/`.
- Runner: `scripts/lj_method_progression_sweep.py`.
- Launcher: `scripts/run_lj_method_progression_sweep.slurm`.
- Code commit: `d69827f` (the subsequent documentation clarification is
  separate).

No Transition1x calculation, data preparation, or model evaluation was
performed as part of this comparison.
