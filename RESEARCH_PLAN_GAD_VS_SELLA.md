# Research Plan: Why Does GAD Outperform Sella on HIP?

Last updated: 2026-07-17

The canonical current scientific interpretation, literature grounding,
publication assessment, and shortest experimental path are in
[`HIP_GAD_SELLA_SYNTHESIS_2026_07_17.md`](HIP_GAD_SELLA_SYNTHESIS_2026_07_17.md).
The claim-by-claim mathematical and implementation critique is in
[`HIP_GAD_SELLA_THEORY_AUDIT_2026_07_17.md`](HIP_GAD_SELLA_THEORY_AUDIT_2026_07_17.md).
This plan retains the broader program; when priorities differ, the synthesis
document's gated critical path takes precedence.

## Research Question

The umbrella question is:

> Which measurable properties of a molecular force/curvature model cause
> force-reflection GAD to outperform full-Hessian RFO/Sella, and are HIP's
> results caused by GAD robustness, Sella sensitivity, or both?

This includes three possible outcomes without presupposing one:

1. GAD is unusually effective on HIP.
2. Sella is unusually ineffective on HIP.
3. A common property of HIP helps GAD and hurts Sella.

The study is restricted to methods supplied with a full Hessian at every
step. Autoregressive or quasi-Newton Hessian construction is not a primary
research question.

### Investigation Priority

The first objective is to explain the observed *paired optimizer separation*
on HIP, not to explain HIP Hessian non-integrability in isolation:

1. Identify where otherwise matched GAD-only and Sella-only trajectories first
   separate, using spectral, mode, step, force, and reaction-tube diagnostics.
2. Determine whether the separation is primarily GAD's efficient use of a
   persistent low mode, Sella's full-spectrum step/globalization behavior, or
   a common start-distribution effect.
3. Test whether the discriminator predicts the result across controlled start
   types and independent PESs.
4. Treat force-Jacobian work as a secondary implementation diagnostic only.
   It is not the intended HIP Hessian product and does not answer the umbrella
   question.

## Evidence At Start

The current cross-surface evidence is recorded in
[`BENCHMARK_RESULTS_2026_07_16.md`](BENCHMARK_RESULTS_2026_07_16.md).

- HIP/Transition1x: Sella wins at low perturbation, the methods cross near the
  middle of the grid, and plain GAD wins clearly at high perturbation.
- DFTB0: GAD has a small lead in the easiest cell; Sella wins thereafter.
- LJ7: Sella strongly beats pure GAD. A high-index gate makes GAD nearly
  perfect, but changes the method and does not yet have endpoint validation.
- HORM LEFTNet: the four-structure screen is negative but not conclusive,
  because the structures were not HORM-native saddles.
- MACE-OFF23: only exploratory data exist.

The primary background papers are available locally:

- [`references/papers/transition1x_2207.12858.pdf`](references/papers/transition1x_2207.12858.pdf)
- [`references/papers/hip_2509.21624.pdf`](references/papers/hip_2509.21624.pdf)

## Important Corrections And Observations

1. Transition1x transition-state structures are converged CINEB maximum
   images (`Fmax < 0.05 eV/A`) and were not subsequently refined into exact
   stationary points. Each calculator therefore needs PES-native refined
   saddles before perturbation.
2. PaiNN forces are computed as `-grad(E)`, making PaiNN a clean
   conservative-force, autograd-Hessian comparison.
3. The HIP checkpoint used here has `regress_forces=True`. Its forces and
   Hessians are directly predicted. The Hessian is symmetric but need not
   equal either `-dF/dx` or `d2E/dx2`.
4. The HIP paper explicitly discusses Hessian non-integrability. Its loss also
   emphasizes the lowest Hessian subspace, which may favor GAD's use of the
   minimum mode relative to Sella's use of the broader spectrum.
5. HIP reports strong RS-P-RFO performance from GSM-generated guesses.
   Therefore, an explanation based only on RFO being incompatible with HIP is
   unlikely. Starting-distribution and implementation differences matter.
6. The checkpoint was trained on HORM-Transition1x configurations. The paper
   draft must not say that HIP was not trained on Transition1x. Exact overlap
   between the 287 benchmark reactions and training data must be audited.
7. Historical scripts convert Angstrom to picometre labels incorrectly. Units
   must be corrected before publication.
8. A 12-geometry implementation diagnostic confirmed that HIP's supplied
   Hessian, energy Hessian, and direct-force Jacobian are distinct objects.
   This is expected for separately supervised direct heads and is not an
   accuracy ranking. Results are retained as provenance in the synthesis
   document, but Jacobian substitution is not a scientific axis of this plan.
9. The completed `0.15 A` paired analysis does not support first-step size or
   consecutive `v1` overlap as the explanation. GAD-only Sella traces move
   away from the labelled TS and then spend a median 97.9% of stored rows at
   the minimum trust radius.
10. In Sella 2.3.4, `gamma` is an iterative eigensolver tolerance, not a line
    search control, and external full-Hessian refreshes bypass it. Sella
    applies the step before adapting its next trust radius from the
    actual/predicted energy ratio.
11. The historical test Sella trajectory artifact records `delta0=0.1` and
    `gamma=0.4`. Dependency and configuration provenance must come from the
    artifact and job script, not current defaults.
12. Project provenance confirms that the headline Sella benchmark receives
    the current full HIP Hessian at every optimization step. Quasi-Newton
    Hessian mixing is not a candidate explanation for the crossover.

## Competing Hypotheses

| Hypothesis | Expected evidence |
|---|---|
| HIP's low-mode reliability exceeds its finite-step quadratic-model reliability | `v1` remains aligned with a native path while Sella model ratios deteriorate; replacing the stable block rescues Sella |
| HIP's low-mode-focused loss changes optimizer sensitivity | The GAD/Sella ranking changes across matched HIP MAE/MSE/subspace-loss checkpoints |
| Sella enters another basin and then collapses its trust radius | Poor `rho` and path departure precede floor-level trust radii; a better trust/globalization policy or stable block reduces the gap |
| Isotropic Cartesian noise creates the result | The advantage disappears for GSM/NEB guesses or perturbations matched by energy/index rather than Cartesian RMSD |
| Intended-saddle semantics create the result | The gap is much smaller for any valid stationary point than for native IRC endpoint recovery |
| Historical implementation/configuration creates the result | A pinned, frozen rerun removes the crossover or another standard RS-P-RFO implementation does not reproduce it |
| GAD is genuinely better for a class of reactive MLIPs | The advantage repeats on PaiNN and at least one HORM/NewtonNet surface and is predicted by measurable landscape features |

## Phase 0: Lock The Benchmark

- Correct the perturbation units and retain coordinates in Angstrom as the
  source of truth.
- Audit HIP/HORM training overlap using reaction IDs, formulas, hashes, and
  geometry fingerprints.
- Give GAD and Sella identical energies, forces, raw Hessians, masses, Eckart
  treatment, numerical precision, and full-Hessian refresh frequency.
- Declare the GAD metric mathematically. Mass-coordinate GAD and Euclidean
  Cartesian GAD are different dynamics even when both remove rigid modes.
- Pin Sella and HIP versions/checkpoint hashes. Do not rely on `sella>=2.3`.
- Preserve and record the confirmed full-Hessian-at-every-step Sella path.
- Record Sella proposed/applied steps, `rho`, and predicted/actual energy
  changes. Sella 2.3.4 does not reject and roll back poor model steps.
- Recompute every terminal metric outside the optimizer.
- Use `n_neg = 1` plus force-threshold curves as the stationary-point metric.
- Use PES-native two-sided IRC endpoint recovery as the primary chemical
  metric.
- Tune on a validation subset and freeze one configuration per method. Do not
  choose a different best configuration in each test cell.
- Refine every reference TS on its active PES before perturbing it.

## Phase 1: Independent MLIP Benchmarks

Run candidates in this order.

### 1. PaiNN / NeuralNEB

Use the published pretrained Transition1x models, conservative forces, and
autograd full Hessians. This is the cleanest test of a generic reactive
Transition1x MLIP versus HIP-specific direct force/Hessian behavior.

### 2. HORM Matched Pairs

Start with conservative LEFTNet and AlphaNet and compare their matched E-F and
E-F-H checkpoints. This tests whether Hessian supervision itself moves the
GAD/Sella boundary. Evaluate direct-force EquiformerV2 later because it adds
force non-conservativity as another variable.

### 3. NewtonNet

Reuse the authors' unseen reactions and degraded guesses where possible. Its
published Sella-positive result makes it a strong adversarial benchmark.

### 4. MACE-OFF23

Retain MACE as an out-of-domain negative control rather than primary evidence.

For each surface, use at least 50 PES-native saddles for a pilot and target
100-200 for the final benchmark. Relax endpoints, generate paths with NEB or
GSM, refine their maxima, and include structures only when they satisfy tight
stationarity, index-one, and two-sided native IRC requirements. Inclusion must
not depend on whether GAD or Sella succeeds from the test perturbations.

## Phase 2: Starting-Geometry Experiments

Use exactly paired starting coordinates for both methods and compare:

- Isotropic Cartesian perturbations, matching the current study.
- Perturbations along the reaction mode.
- Perturbations only in the stable-mode subspace.
- Perturbations matched by initial `n_neg`.
- Perturbations matched by harmonic displacement energy
  `dx^T abs(H) dx`.
- Real GSM/NEB candidates at multiple path-convergence stages.

This separates geometric distance from chemically meaningful search
difficulty.

## Phase 3: Trajectory Diagnostics

Record the following at every step:

- Full projected spectrum, `n_neg`, `lambda1`, `lambda2`, eigengap, and minimum
  mode overlap with the preceding step.
- Overlap of the minimum mode and optimizer step with the IRC/NEB reaction
  tangent.
- Proposed and applied step norms, maximum atom displacement, displacement-cap
  activity, and Sella trust-radius behavior.
- Actual versus Hessian-predicted force and energy changes, including Sella
  `rho`.
- Distance to the native TS and reaction-path tube.
- Minimum interatomic distance and bond graph.
- Force/Hessian consistency residuals.
- MLIP ensemble disagreement or another out-of-distribution signal where
  available.

Analyze paired outcomes as: both succeed, GAD only, Sella only, and neither.
Align traces at their first large separation or first suspect step, rather
than comparing only final structures.

## Phase 4: Supporting HIP Causal Experiments

On a 20-50 reaction diagnostic subset, prioritize:

- Matched Sella trust radii and GAD step distributions.
- Direct HIP Hessians.
- Direct-HIP `v1` with a neutral, reference, or secant-updated stable block.
- Perturbed/reference `v1` with the HIP stable spectrum retained.
- HIP checkpoints trained with different Hessian losses, if available.

Inject controlled errors separately into the minimum mode, eigengap, low
positive modes, and high-frequency modes. This tests whether GAD's reduced
dependence on the full spectrum explains its robustness.

These interventions follow, rather than replace, the paired trajectory
analysis. Their role is to eliminate or support a mechanism identified in
Phases 2--3; they are not the main outcome of the study.

The force Jacobian and energy Hessian are not polished replacements for HIP's
direct Hessian and should not be central experimental axes.

## Scientific And Mathematical Watch List

- Standard one-vector GAD assumes a gradient force field and its Hessian.
  With HIP direct forces and direct Hessians, the implemented method is a
  GAD-like min-mode flow and may not be the exact mathematical GAD for the
  learned vector field.
- Small `lambda2-lambda1` makes the minimum mode intrinsically sensitive.
  Mode crossings can create discontinuous dynamics even with accurate
  Hessians.
- Compare the dimensionless explicit-Euler stiffness
  `dt * max(abs(lambda))`, not the same raw `dt` across different PESs.
- The reported GAD convention mass-weights for the Eckart/vibrational
  construction and then un-mass-weights the direction back to Cartesian
  coordinates before applying it. Mass choice can still change mode ordering
  and the trajectory. Use physical masses consistently and treat the LJ mass
  as an explicit algorithmic choice.
- Eckart-projected curvature is standard at stationarity, but away from
  stationarity a projected ambient Cartesian Hessian is not automatically the
  exact quotient-manifold Hessian because rotational curvature has
  gradient-dependent terms.
- Linear and nearly linear molecules need special Eckart handling because the
  rotational rank changes.
- High `n_neg` and close repulsive contacts measure globalization difficulty,
  not merely perturbation magnitude.
- The lowest Hessian mode should be checked for chemical character by its
  overlap with bond-change vectors and the NEB/IRC tangent.

## Computational-Chemistry Watch List

- An MLIP-native IRC can be internally correct but disagree with DFT. Recheck
  a stratified subset of saddles and endpoints at the source DFT level.
- Treat proton transfers, conformational changes, symmetry-equivalent atoms,
  fragmentation, and shallow/bifurcating IRCs as separate endpoint classes.
- Track whether trajectories leave the training manifold. An optimizer should
  not receive credit for exploiting an unphysical extrapolation region.
- Use graph matching for chemical identity and RMSD only as a stricter
  geometric secondary metric.
- Compare nominal Cartesian noise with PES-local quantities. The same
  displacement can produce radically different force, index, and close-contact
  distributions on different surfaces.

## Statistics

- Use paired McNemar tests and paired bootstrap confidence intervals for each
  fixed benchmark cell.
- Fit a mixed-effects outcome model using method, initial index, eigengap,
  stiffness, Hessian consistency, reaction-mode overlap, and model identity.
- Evaluate mechanistic predictors by leaving one MLIP out. A retrospective
  correlation on HIP alone is not enough.
- Report all fixed configurations, failure categories, threshold curves, and
  confidence intervals. Do not report post-hoc best-of-grid values as a single
  predefined method.

## Paper Viability Criteria

The broad GAD-over-Sella narrative is supported if either:

1. GAD beats Sella in PES-native IRC recovery on at least two independent
   MLIP architectures at difficult starts, using frozen configurations and
   confidence intervals excluding zero; or
2. A matched HORM or direct-versus-conservative experiment establishes a
   reproducible curvature property that predicts which optimizer wins.

If only HIP remains positive, the broad claim is unsupported. A narrower
paper may remain viable around the useful decoupling provided by directly
predicted, low-mode-focused curvature and how optimizers should consume it.

## Computational Execution

- Follow [`EXPERIMENT_ORGANIZATION.md`](EXPERIMENT_ORGANIZATION.md): use
  project space only for source and small research records, and place
  environments, caches, logs, trajectories, and active runs in scratch.
- Before broad new runs, migrate the project-local `.venv` to scratch, verify
  it through the existing `.venv` path, and request approval to remove the
  project backup. This is expected to recover roughly 55,000 `rrg-aspuru`
  inodes.
- Use separate `uv` environments for incompatible MLIP dependency stacks. The
  current HIP and MACE dependencies conflict on `e3nn`.
- Require an energy/force/Hessian smoke, directional finite-difference check,
  and GAD/Sella adapter agreement test for every backend.
- Run 3-5 structure smokes, then a 20-structure pilot, before full grids.
- Batch GAD by atom count on GPUs.
- Pack multiple independent Sella trajectories per GPU after profiling.
- Use CPU-node process pools for LJ and DFTB0.
- Submit tens of packed Slurm array tasks rather than one task per molecule.
- Record checkpoint hashes, environment locks, seeds, calculator settings,
  and one durable result row per attempt.

## Immediate Order Of Work

1. Verify the stored historical GAD recurrence and pin the exact Sella/HIP
   versions and configurations.
2. Preserve the confirmed benchmark condition that Sella receives the current
   full HIP Hessian at every optimization step.
3. Treat the completed `0.15 A` paired result as descriptive: it identifies
   Sella trust collapse, not its cause.
4. Run the predeclared Taylor-compatibility panel on 12 fixed starts spanning
   both, GAD-only, Sella-only, and neither.
5. Instrument proposed/applied steps, model ratio, trust radius, and
   path/destination diagnostics on those same cases.
6. Test whether Taylor residuals, native-path-aligned `v1`, and poor Sella
   model ratios precede trust collapse and basin separation.
7. Run one tied trust/globalization control, then HIP `v1` versus stable-block
   ablations.
8. Reject the mechanism if the residuals are not predictive or the tied
   intervention does not rescue failures.
9. Compare matched HIP MAE/MSE/subspace-loss checkpoints if available.
10. Compare HIP's index-one and low-mode persistence with the completed
   control surfaces.
11. Only then invest in a second PES-native directly learned Hessian benchmark.
12. Keep environment migration, split/leakage audit, and corrected noise units
   as mandatory publication hygiene, but do not let broad infrastructure work
   delay the first causal HIP result.

Status as of 2026-07-17: project wrapped after documentation. The diagnostic
is implemented, but the HIP/SCINE panel and causal interventions were not run.
The analytic LJ diagnostic smoke passed and is not evidence for the HIP
mechanism.
