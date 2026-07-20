# Lennard-Jones GAD/Sella Notes, July 2026

This is a short handoff note for the LJ work. It is deliberately scoped to the checks and findings from the recent GAD-vs-Sella debugging pass, not a full experiment report.

## Implementation

- The LJ implementation is local, in `src/gadplus/calculator/lennard_jones.py`.
- It is not the SCINE/HIP path. The LJ potential, forces, and Hessian are implemented directly for reduced LJ-style clusters.
- The paper-style LJ experiment matrix is represented in the codebase via `scripts/lj_paper_runner.py`, `scripts/aggregate_lj_paper.py`, `scripts/run_lj_paper_sweep*.slurm`, plus the newer batched GAD runner.
- The batched GAD runner is `scripts/lj_batched_gad_runner.py`; the Slurm wrapper is `scripts/run_lj_batched_gad.slurm`.
- Atom mass enters the Eckart/vibrational construction. In all benchmark data
  discussed here, that weighting was removed before applying the Cartesian
  GAD displacement. The intended LJ atom type is hydrogen. Heavier masses
  were tested under the same convention and did not fix the failure mode.

## Smoke Checks Done

- Predictor returns a full raw Hessian with shape `3N x 3N`; for LJ7 this was `21 x 21`.
- LJ predictor Hessian was compared against finite differences. The rough finite-difference check near a steep geometry gave max absolute error around `6.3e-4` and RMS around `6.6e-5`, consistent with numerical FD noise rather than a structural Hessian bug.
- Batched LJ GAD was compared against the same predictor path:
  - Force max difference was about `7e-15`.
  - Hessian max difference was about `1.5e-11`.
- The Sella/ASE path was compared against batched LJ on the same geometry:
  - Energy difference was zero.
  - Force difference was at most about `4.7e-10`.
  - Raw Hessian difference was at most about `3.0e-8`.
- Sella's Eckart-projected Hessian callback preserved the vibrational spectrum relative to the raw Hessian path to within about `9e-8` in the checked case.
- Net: there is no current evidence that Sella and GAD are using materially different LJ potentials, forces, or raw Hessians.

## GAD Implementation / Branch Audit, July 17

- The benchmark convention is to construct the Eckart/vibrational problem
  with masses and then un-mass-weight the resulting direction before applying
  a Cartesian GAD displacement. The implementation that retained the mass
  weighting errored and produced no benchmark data.
- A historical older-branch/batched recurrence can nevertheless be written in
  a mass-weighted-coordinate form and was checked as implementation
  provenance.
- For a uniform-mass LJ cluster these are not different algorithms.  If every
  atom has mass `m`, the old/batched Cartesian direction is exactly the current
  direction divided by `m`; choosing `dt_batched = m * dt_current` makes their
  coordinate recurrences identical.  This equivalence does not extend without
  qualification to mixed-mass molecular systems, where mass weighting changes
  relative coordinate scales.
- `scripts/smoke_lj_gad_recurrence_equivalence.py` checked a deterministic,
  noised LJ7 hydrogen-mass start for five consecutive analytic-Hessian steps.
  The maximum direction error after the required `1/m` scaling was `7.299e-12`
  and the maximum coordinate error after timestep rescaling was `8.332e-14` A.
- Hydrogen has `m = 1.008`, so leaving the same numeric `dt` across the two
  conventions changes the effective timestep by only 0.8%.  It cannot explain
  the large pure-GAD versus Sella gap on LJ.  It does mean all future LJ tables
  must record which runner/convention supplied `dt`; cross-runner comparisons
  should rescale the batched value by the stated hydrogen mass.
- Together with the analytic-Hessian, finite-difference, Sella calculator,
  and un-mass-weighting checks above, this removes the identified
  implementation-level causes of the LJ result. The remaining evidence
  supports the high-index,
  stiff-entry-region explanation rather than a bad GAD recurrence.

## Sella Caveat

- The LJ comparison so far is mainly transition-state quality: `n_neg = 1` and low `fmax`.
- It is not yet the same as the paper's main IRC/topology endpoint criterion unless an LJ-specific IRC/topology classifier is added.
- This matters because Sella can converge to a valid first-order saddle that is not the intended saddle under an IRC/topology definition.

## GAD Failure Pattern

- The main observed failure mode is not a missing Hessian or an obviously wrong LJ implementation.
- Failures often show a large raw displacement/step event, then remain capped or trapped for many steps instead of entering the small convergence basin.
- Successful traces tend to leave the capped regime earlier.
- Capping/ramping helped but did not make pure GAD competitive with Sella on the LJ TS-quality metric.

## Replay Diagnostic, July 9

- A replay of selected `noise=0.20`, `dt=0.005`, `max_atom_disp=0.005` traces compared clean successes, late/capped successes, and hard failures.
- In the full sample-level summaries, all starts hit the cap at step 0. The distinction is persistence:
  - Successful traces had median `cap_hits = 217`.
  - Failed traces had median `cap_hits = 1021`.
  - 24 failed traces were capped for all 8000 steps; no successful trace was capped for its whole run.
- The first 100 replay steps did not show the hard failures immediately stepping in a completely wrong direction. Early steps often moved toward the low-noise reference/minimum, but hard failures stayed capped every step and retained many negative modes and high force.
- Reducing the cap from `0.005` to `0.001` kept cumulative motion toward the low-noise reference for the replayed traces through 100 steps, but it also slowed relaxation strongly and left forces/negative modes much larger at step 100.
- Current interpretation: the LJ failure is a trust-region/basin-entry problem. A smaller cap prevents obvious overshoot, but a fixed tiny cap is too slow. A useful fix likely needs adaptive step acceptance/trust radius rather than only a global displacement cap.

## Retuning Results

- Original high-noise GAD success rates were roughly:
  - Noise `0.15`: about `41.5%`.
  - Noise `0.20`: about `29.3%`.
- Best fixed-cap hydrogen retune improved this to roughly:
  - Noise `0.15`: about `51.2%`.
  - Noise `0.20`: about `38.3%`.
- Best ramped-cap hydrogen retune was roughly:
  - Noise `0.15`: about `50.5%`.
  - Noise `0.20`: about `39.7%`.
- Heavier mass settings, including carbon-like and argon-like choices, were worse in the tested matrix.
- The practical takeaway is that step-size/overstep control helps but does not remove the pure-GAD LJ failure mode.

## Deep Follow-Up: Geometry, Noise, and High-Index Recovery

### Corrected LJ7 reference geometry

- `pentagonal_bipyramid_geometry()` had used the pair-equilibrium distance for all nearest neighbors. That is a close geometric approximation but not the force-balanced LJ7 D5h minimum: at zero noise it had `E = -16.4741583` and `fmax = 1.95618`.
- It now uses the relaxed reduced-unit D5h parameters `ring_radius = 0.9562063084643488 * sigma` and `height = 0.5738701709721903 * sigma`. The corrected structure has `E = -16.5053841680`, `fmax = 1.06e-7`, and zero vibrational negative modes.
- This correction matters for the literal zero/very-low-noise setup, but it is not the high-noise explanation: pure-GAD results at noise `0.15` stayed at `51.2%` after the correction.

### What the large LJ fmax means

- `fmax` is the maximum Cartesian force component. On LJ it is dominated by the closest compressed pair, not a global measure of saddle proximity. Direct numeric comparison with HIP `fmax` is also not dimensionally meaningful: LJ uses reduced `epsilon / sigma` force units while HIP uses eV/A.
- The important signal is the dynamic range. With the corrected LJ7 start, at noise `0.20` the initial median `fmax` is `2.045e3`, p95 is `5.56e5`, and p99 is `2.63e7`; median initial `n_neg` is 8. At noise `0.03`, the initial median `fmax` is only about 10.
- The noise is independent per Cartesian component. A `0.20 * sigma` standard deviation therefore gives an expected per-atom displacement norm of `sqrt(3) * 0.20 = 0.346 * sigma`. In a 100k-start Monte Carlo check, 54.93% of starts had a closest pair below `0.75 * sigma`, and 20.16% below `0.60 * sigma`.
- The repulsive wall scales approximately as `F ~ r^-13`, so those overlap tails generate the extreme force and curvature tail. The displacement cap removes the magnitude information, making many samples take the same `0.005` step; this is why lowering a global dt alone had little effect while capped.

### The actual pure-GAD issue

- An index-1 GAD flow is locally stable only when there is one negative Hessian mode. At an index-k point with `k > 1`, all unflipped negative modes are unstable under single-mode GAD.
- High-noise LJ starts are commonly index 5--8 before relaxation, whereas the smooth HIP trajectories exit the high-index region quickly. Thus the LJ issue is not a bad Hessian: it is a predictable single-mode-GAD globalization failure on a stiff, high-index entry region.

### High-index-gated GAD test

- The batched runner now supports a smooth gate already used conceptually elsewhere in the project:
  `w = sigmoid(k * lambda_2)` and `F_step = F - 2*w*(F dot v1)*v1`.
  It is force descent while `lambda_2 < 0`, then becomes ordinary single-mode GAD when the second mode is positive.
- Corrected LJ7, hydrogen mass, `dt=0.005`, cap `0.005`, 8000-step strict metric (`n_neg=1` and `fmax<0.01`):

| Method | noise 0.10 | noise 0.15 | noise 0.20 |
|---|---:|---:|---:|
| pure GAD | 69.7% | 51.2% | 36.2% |
| hard descent until `n_neg <= 1`, then GAD | 99.0% | 97.2% | 96.5% |
| smooth gate, `k=50` | 100.0% | 100.0% | 99.7% |
| smooth gate, `k=10` or `k=100` | not run | not run | 100.0% |

- At noise `0.20`, smooth-gated trajectories reach strict convergence in a median 257--258 steps, versus 619 for the pure-GAD successes. The `k=10`, `k=50`, and `k=100` runs reached the same final energy to `1e-4` for 285--286 of 287 starts, so this is not a brittle sharpness choice.
- This improves the strict saddle-convergence metric, but it is not honest to call it pure GAD from step 0. It is a minimal globalization/gating extension. It also changes the saddle-family distribution: only 33 of the 104 pure-GAD successes at noise `0.20` finished at the same energy (within `1e-4`) as their smooth-gated counterpart. LJ-specific IRC/topology or basin validation is still required before claiming equal chemistry.

### Noise-unit note

- The direct HIP/LJ sweep scripts pass `0.01, 0.03, ..., 0.20` as Angstrom coordinate noise and label it with `round(noise * 1000)` pm. The repository itself documents `1 A = 100 pm`, so `0.20 A` is physically 20 pm, not 200 pm. The config-driven path in `src/gadplus/orchestration/run.py` uses the correct `pm / 100` conversion.
- This is a label/experimental-reporting inconsistency for the existing HIP paper matrix. For reduced LJ, pm labels should be removed entirely unless a physical `sigma` mapping is specified; report the perturbation as a fraction of `sigma`.

### New artifacts

- Corrected-start recovery sweeps: `/lustre07/scratch/memoozd/gadplus/runs/lj_geometry_recovery_20260709/`
- Smooth-gate high-noise sensitivity: `blend_k10_corrected`, `blend_k50_corrected`, and `blend_k100_corrected` under that root.

## Current Artifacts

- Fixed-cap/ramp summary files:
  - `/lustre07/scratch/memoozd/gadplus/runs/lj_gad_freeze_trace_refine_20260706_211821/summary_trace.csv`
  - `/lustre07/scratch/memoozd/gadplus/runs/lj_gad_pure_ramp_20260706_213810/summary_trace.csv`
- Direction diagnostic smoke output:
  - `/lustre07/scratch/memoozd/gadplus/runs/lj_direction_diag_smoke/`

## Open Follow-Ups

- Add LJ-specific IRC/topology validation if the goal is a strict paper-style comparison rather than TS-quality comparison.
- Continue diagnosing pure GAD around the first large-step event: compare successful and unsuccessful traces by raw step direction, applied step direction, distance to known convergence basin, and post-step Hessian mode changes.
- If retuning continues, prioritize preventing the first overstep cleanly over adding descent prephases; the stated target is pure GAD.
