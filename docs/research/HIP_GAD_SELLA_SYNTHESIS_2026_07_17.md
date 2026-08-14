# HIP, GAD, and Sella: Research Synthesis and Critical Path

Last updated: 2026-07-17

This is the canonical scientific synthesis for the current project. It
separates measured results from interpretation, states the strongest current
answer to the research question, relates that answer to the literature, and
defines the shortest route to a meaningful paper result.

Supporting repository records:

- [Critical theory and implementation audit](HIP_GAD_SELLA_THEORY_AUDIT_2026_07_17.md)
- [Cross-benchmark results](BENCHMARK_RESULTS_2026_07_16.md)
- [Full research plan](RESEARCH_PLAN_GAD_VS_SELLA.md)
- [Lennard-Jones audit](LJ_FINDINGS_2026_07_09.md)
- [Experiment organization](../EXPERIMENT_ORGANIZATION.md)
- [HIP paper](../../references/papers/hip_2509.21624.pdf)
- [Transition1x paper](../../references/papers/transition1x_2207.12858.pdf)

## Project Status

This investigation was wrapped on 2026-07-17. Pending HIP and SCINE
Taylor-compatibility jobs were cancelled before scientific evaluation. The
diagnostic and its predeclared decision rule remain in
`experiments/2026-07-17-hip-taylor-compatibility.json`; only its analytic LJ
implementation smoke was completed. No cancelled run is counted as evidence.

## Executive Answer

The broad claim "GAD is better than Sella" is not supported. A narrower
mechanism is plausible but not yet established:

> HIP may preserve a useful leftmost direction beyond the region where its
> separately learned energy, direct force, and direct Hessian form a reliable
> finite-step Taylor model. GAD consumes the current force and a rank-one mode
> projector. Sella constructs a model from the direct force and full Hessian,
> then evaluates that model against the learned energy. If the energy gradient
> and direct force disagree to first order, shrinking the trust radius does
> not guarantee that Sella's actual/predicted energy ratio approaches one. In
> stored GAD-only cases, Sella moves away from the labelled saddle and then
> spends almost all remaining iterations at its minimum trust radius. This
> makes Taylor-product compatibility a concrete candidate mechanism, while
> low-mode reliability, distribution shift, basin semantics, spectral
> conditioning, and trust-policy hysteresis remain live alternatives.

This is the best current explanation of the observed crossover:

- Sella clearly wins on HIP at low perturbation.
- GAD and Sella tie around the middle of the perturbation grid.
- Plain GAD wins on HIP at the largest perturbations.
- The same crossover does not occur on exact Lennard-Jones curvature or on
  SCINE DFTB0.

The previous "small GAD steps versus large Sella steps" explanation is not
supported: median first maximum-atom displacements at `0.15 A` are about
`0.088 A` for GAD and `0.100 A` for Sella, with no difference between the
GAD-only and Sella-only groups. The decisive missing evidence is whether
the measured Taylor residuals predict paired outcomes before method
divergence, whether HIP's lowest mode remains chemically useful relative to a
native reaction path, why Sella's model ratio drives trust-radius collapse,
and whether the methods enter different valid saddle basins.

Project provenance confirms that Sella receives the current full HIP Hessian
at every optimization step. Quasi-Newton Hessian mixing is therefore not an
explanation for the headline crossover.

## Collaboration And Framing Principle

The intended paper includes HIP's first author as a coauthor. More
importantly, the positive framing is the more accurate scientific framing.

Do not describe HIP as a bad, defective, or incoherent PES. HIP's direct
Hessian is trained against DFT curvature and is not merely intended to imitate
the derivative of an approximate learned force. Its disagreement with that
force Jacobian can therefore represent additional supervised chemical
information rather than simple error. This observation is useful provenance,
not the scientific explanation pursued here.

The paper should present:

- HIP as a model with a task-specialized, DFT-informed curvature channel.
- GAD as one way to consume the persistent low-mode information.
- Sella as a strong full-Hessian optimizer with a different step construction
  and basin/globalization behavior.
- The observed crossover as an optimizer-interface opportunity, not as an
  indictment of HIP or Sella.

Preferred language includes "task-aligned curvature", "spectral reliability",
"local quadratic-model reliability", and "curvature-consumer interface."
Use "learned reaction tube" only for an operationally defined, measured
path-neighborhood result. Avoid unqualified phrases such as "bad Hessian",
"broken PES", or "Sella cannot use HIP."

### The force Jacobian is not a scientific axis

The symmetric direct-force Jacobian `-sym(dF/dx)` was never intended to be a
polished replacement for HIP's directly inferred Hessian. HIP's motivation is
partly that differentiating a learned energy or force can amplify small
prediction fluctuations into poor second derivatives. The direct Hessian is
the explicitly supervised product.

Keep four objects conceptually separate:

| Object | Meaning in this project | Appropriate use |
|---|---|---|
| HIP direct Hessian | DFT-supervised curvature prediction | Primary curvature supplied to optimizers |
| Direct-force Jacobian | Local derivative of the force head actually followed | Optimizer-interface and local linear-response diagnostic |
| Energy Hessian | Second derivative of the learned scalar energy head | Conservative control, not assumed superior |
| DFT/HORM Hessian label | Reference electronic-structure curvature | Curvature-accuracy ground truth when available |

Disagreement between the HIP Hessian and force Jacobian does not establish
that the HIP Hessian is inaccurate. Indeed, HIP can be closer to the DFT
Hessian precisely because it is not forced to inherit rough second
derivatives from an approximate force model. The `12/12` versus `0/12`
index-one observation should therefore be framed as evidence that direct
Hessian supervision recovers useful TS curvature that force differentiation
misses.

The force Jacobian remains only an implementation diagnostic. It can describe
the local derivative of the force head, but results from that unpolished
object do not answer the main question: why the GAD/Sella ranking changes on
HIP relative to other potentials. It should not appear as a headline result
or central paper figure.

Do not organize the paper or the next experiment loop around the force
Jacobian. Do not spend additional compute on whole-Hessian Jacobian
substitution unless the primary trajectory analysis independently identifies
force-response mismatch as the likely cause. The main scientific axes are
HIP's learned reaction-mode field, index topology, optimizer step geometry,
and basin preservation.

## What The Two Optimizers Ask From Curvature

Let `F(x)` be the force used to move the atoms and let `H_hat(x)` be the
Hessian supplied by the model.

For the simplified single-mode implementation used in this repository, the
GAD position update is proportional to

```text
F_GAD = F - 2 (F . v1) v1
```

up to sign convention, mass/Eckart projection, and timestep. The Hessian
enters principally through the projector `v1 v1^T`, where `v1` is the
leftmost vibrational eigenvector. Plain GAD does not invert the stable
spectrum and its position update is invariant to the magnitudes of the
Hessian eigenvalues when `v1` is fixed.

Sella uses restricted-step partitioned rational-function optimization. Its
step is constructed from the gradient and a model of curvature in both the
ascending and stable subspaces. The trust-region machinery limits damage from
an imperfect model, but the proposed step still depends on substantially more
of the Hessian than GAD does.

This creates a reliability hierarchy:

```text
Useful v1 direction
        |
        | enough for force-reflection dynamics
        v
Accurate low spectrum and index
        |
        | increasingly useful for mode-following/RFO
        v
Accurate full quadratic model: Delta F approximately -H Delta x
        |
        | required for quantitatively trustworthy second-order steps
        v
Globally integrable energy/force/curvature field
```

The central hypothesis is that HIP may preserve the first level farther from
path-local starts than it preserves a quantitatively predictive finite-step
quadratic model. Consecutive mode overlap and index topology do not establish
chemical reaction-direction quality; that requires comparison with a
PES-native path tangent or reference mode.

### Taylor-product compatibility

HIP exposes a learned energy `E_hat`, a direct force `F_hat`, and a directly
DFT-supervised Hessian `H_hat`. With `g_F = -F_hat`, the quadratic energy
model consumed by Sella has the local form

```text
m(s) = E_hat(x) + g_F^T s + 0.5 s^T H_hat s.
```

If `g_E = grad E_hat` and `H_E = Hessian E_hat`, its finite-step error is

```text
E_hat(x+s) - m(s)
  = (g_E - g_F)^T s
    + 0.5 s^T (H_E - H_hat) s
    + O(norm(s)^3).
```

The first term matters especially for trust globalization. Along
`s = delta d`, a nonzero energy/direct-force mismatch generically gives

```text
lim(delta -> 0) rho(delta d) = (g_E^T d) / (g_F^T d),
```

rather than one. Therefore, repeatedly shrinking the radius can leave an
energy-ratio controller at its floor. Separately,
`F_hat(x+s)-F_hat(x)+H_hat s = O(norm(s))` rather than
`O(norm(s)^2)` when the Hessian is not the local derivative of the followed
force field.

This is not an accuracy judgment about HIP's Hessian. A directly supervised
Hessian can be closer to DFT and still be a less compatible finite-step model
of a separately learned force. GAD asks for the weaker interface
`(F_hat, v1 v1^T)` and never compares a predicted energy change with
`E_hat(x+s)`. The detailed derivation, non-gradient GAD caveat, evidence
ledger, and falsification criteria are in the theory audit.

## Why HIP Is A Special Case

### Facts from the HIP model and paper

HIP predicts symmetric Hessians directly rather than obtaining them as exact
derivatives of its energy or force output. This gives the Hessian head direct
access to DFT curvature targets instead of forcing it to inherit amplified
derivative noise from the learned energy or force. The paper explicitly notes
that the predicted Hessian is not guaranteed to be globally integrable. It
also uses a loss that emphasizes the subspace of the eight lowest reference
eigenvectors and eigenvalues. This loss improves first-eigenvector similarity
and second-eigenvalue accuracy modestly in the paper's ablations, while some
full-matrix and eigenvalue errors become slightly worse.

The model is trained on Hessian-labelled reactive structures derived from
HORM/Transition1x. Transition1x itself contains structures sampled on and
around DFT NEB reaction paths, rather than arbitrary isotropic perturbations.
Thus HIP receives unusually direct supervision about reaction-region
curvature and low modes.

The repository uses `hip_v2.ckpt`, the recommended end-to-end HIP checkpoint.
The exact relationship between this checkpoint and every main-text HIP model
must be recorded before publication; it must not be silently assumed that all
paper ablations apply identically to this checkpoint.

The same HIP paper reports strong RFO results from GSM-generated starts and
describes the directly predicted spectrum as useful for full topology. It
therefore contradicts any claim that HIP's full Hessian is intrinsically
unsuitable for RFO. Our hypothesis is conditional on the isotropically
degraded start distribution and the particular optimizer interface.

### Secondary implementation measurements

On 12 fixed Transition1x test structures:

- HIP's predicted Hessian was index one in `12/12`.
- The symmetric direct-force Jacobian was index one in `0/12`.
- The energy Hessian was index one in `0/12`.
- The median absolute overlap between the predicted-Hessian `v1` and
  direct-force-Jacobian `v1` was `0.9903`.
- The median relative full-matrix disagreement between those two curvature
  sources was `0.4735`.
- The median predicted-Hessian versus energy-Hessian `v1` overlap was
  `0.0116`.
- The direct-force Jacobian had median relative antisymmetry `0.2631`.

In a separate directional finite-difference panel, HIP's median relative
`H v` versus `-dF/dx v` residual was `0.0763`, compared with `2.34e-4` for
conservative PaiNN. HIP's energy/force and energy/Hessian differential
agreement was also lower than PaiNN's by construction. These are interface
measurements, not Hessian-accuracy measurements; only comparison with the DFT
Hessian label can answer which curvature is more accurate.

These measurements are retained to document completed make-sures. They do not
establish a scientifically useful cause of the optimizer ranking and are not
part of the proposed paper claim.

## The HIP Crossover

The fixed test set contains 287 noised Transition1x structures. The current
CS² follow-up explicitly loads the HDF5 `test` partition and exact filtered
IDs `0..286`; this is separate from the checkpoint metadata's training LMDB
`ts1x_hess_train_big.lmdb` and validation LMDB `ts1x-val.lmdb`. Its g-xTB-
selected `eta=0.01` and `tau_s=0.01` were frozen before HIP evaluation, so
the HIP test outcomes are not used for per-noise tuning. Under the strict
criterion `n_neg = 1` and `fmax < 0.01`:

| Historical noise label | Plain GAD | Best completed Sella | GAD minus Sella |
|---:|---:|---:|---:|
| 10 pm | 89.2% | 96.5% | -7.3 pp |
| 30 pm | 88.9% | 95.5% | -6.6 pp |
| 50 pm | 85.7% | 92.0% | -6.3 pp |
| 100 pm | 72.8% | 72.8% | 0.0 pp |
| 150 pm | 58.2% | 54.0% | +4.2 pp |
| 200 pm | 44.6% | 27.2% | +17.4 pp |

The completed held-out CS²-GAD cells give `202/287`, `157/287`, and `111/287`
strict successes at `0.10/0.15/0.20 A`, versus plain GAD's `209/287`,
`167/287`, and `128/287`. Intended all-endpoint IRC_TOPO is `218/287`,
`166/287`, and `117/287`, versus plain GAD's `225/287`, `177/287`, and
`128/287`. CS² therefore does not improve HIP local capture or intended
mechanism recovery at any hard-noise cell. See
`HIP_CS2_H100_HANDOFF_2026_08_09.md` for immutable start hashes and the final
failure ledger.

The labels above are historically wrong by a factor of ten: the scripts use
coordinate standard deviations of `0.01` through `0.20 A` but label them
`10` through `200 pm`. Publication tables must report Angstrom values or
correct physical picometres.

At the `0.15 A` cell, paired outcomes are:

| Outcome | Count |
|---|---:|
| Both succeed | 133 |
| GAD only | 34 |
| Sella only | 22 |
| Neither | 98 |

This matters. The result is a statistical shift in robustness, not a rule that
Sella always fails where GAD succeeds.

The completed paired trajectory analysis adds:

- Initial `n_neg`, `fmax`, eigengap, first-step scale, time to index one, and
  consecutive `v1` overlap do not cleanly distinguish GAD-only from
  Sella-only starts.
- The first maximum-atom step is nearly the same in those two outcome groups:
  about `0.088 A` for GAD and exactly the `0.100 A` Sella trust limit.
- In GAD-only cases, Sella's median minimum/final distance to the labelled TS
  is `0.226/0.447 A`, compared with `0.117/0.128 A` in Sella-only cases.
- GAD-only Sella runs spend a median `97.9%` of stored rows at the
  `1e-4 A` trust-radius floor.

This supports a departure-followed-by-trust-collapse failure signature. It
does not identify whether the cause is finite-step quadratic-model mismatch,
an alternate saddle basin, or the trust/update policy.

For intended-saddle validation, the completed HIP IRC-topology rates at
`0.10/0.15/0.20 A` are `78.4/61.7/44.6%` for plain GAD and
`72.5/49.8/23.3%` for the matched Sella baseline. The high-noise advantage is
therefore not solely an `n_neg/fmax` semantic artifact.

## The Best Current Mechanistic Hypotheses

The evidence currently supports a sequence with unresolved branches:

1. Isotropic Cartesian noise moves some structures away from the
   path-generated distribution represented in Transition1x/HORM.
2. GAD repeatedly reflects the current force through the supplied rank-one
   projector. Sella builds an RS-P-RFO step from both curvature partitions.
3. In many Sella failures, one or more applied steps move into a region where
   the energy-model ratio repeatedly shrinks the trust radius to its floor.
4. GAD more often returns toward the labelled saddle on those paired starts.
5. The missing causal question is whether this occurs because HIP `v1`
   remains chemically aligned while the broader finite-step model is
   miscalibrated, because the methods choose different basins, or because the
   historical Sella configuration/globalization policy is unfavorable.

The HIP paper's successful RFO results do not refute this explanation. Its TS
workflow first uses GSM to produce a reaction-path-local guess and then runs
RFO. Our high-noise benchmark deliberately supplies a different and more
off-path starting distribution. This makes the start distribution a central
experimental variable, not proof of a broad HIP reaction tube.

## Why The Other Surfaces Behave Differently

### Lennard-Jones

The LJ7 potential, force, and Hessian implementations were independently
smoked and agree across the GAD and Sella routes. The GAD branch convention
also agrees with the batched implementation after the analytically expected
uniform-mass timestep scaling.

At `0.20 sigma` noise, LJ starts have median initial `n_neg = 8`, median
reduced-unit `fmax` near `2.0e3`, and a severe close-pair repulsive tail.
Single-mode GAD flips one negative mode. An ordinary GAD fixed point is
locally stable only for an index-one saddle, so these starts are outside its
local guarantee. This does not imply universal failure in every transient
high-index region.

The decisive result is that descent while `n_neg > 1`, or a smooth
`lambda_2` gate, raises strict LJ convergence from `36.2%` to approximately
`100%` at the highest noise. Smaller timesteps and displacement caps alone do
not. LJ therefore fails for a different reason: exact curvature faithfully
reports a stiff, high-index entry region that pure one-mode GAD cannot
globalize.

### SCINE DFTB0

DFTB0 does not show a high-noise GAD advantage. Sella wins at every noise
above the smallest cell. The Transition1x labelled structures and endpoint
semantics do not transfer cleanly to the DFTB0 PES, but the strict
`n_neg/fmax` comparison also favors Sella at moderate and high noise.

This is compatible with the hypothesis that a coherent local quadratic model
helps Sella, but it does not establish it. The labelled structures and equal
Cartesian perturbations are not matched in local difficulty on the DFTB0
surface.

### PaiNN, HORM, and MACE

PaiNN is conservative and therefore provides a clean energy-derived Hessian.
Current work has not produced an independent positive GAD-over-Sella result.

The four-structure HORM LEFTNet screen is not a valid broad negative result
because the structures were not HORM-native validated saddles. It nevertheless
did not provide a positive replication.

MACE-OFF23 remains an out-of-domain exploratory control. Transition1x labels
are often not stationary on its PES, so it cannot currently answer the
optimizer question.

## What Is Established, Inferred, And Unknown

### Established by completed measurements

- HIP has a real noise-dependent GAD/Sella crossover.
- The high-noise advantage remains under completed IRC-topology validation.
- HIP energy, direct force, and direct Hessian are not mutually derivable to
  the accuracy of conservative PaiNN; this is expected for independently
  supervised direct heads and is not by itself a defect or an accuracy
  ranking.
- Sella is not generally incompatible with HIP: it is substantially better
  near the TS.
- The historical Sella benchmark supplies the current full HIP Hessian at
  every optimization step.
- Pure GAD's LJ failure is strongly associated with high-index,
  stiff-entry-region globalization; an index-aware gate nearly eliminates
  strict failures.
- A broad claim that GAD beats Sella across PESs is false.
- The paired HIP Sella failures have a reproducible trust-radius-collapse
  signature after departure from the labelled TS.
- First-step magnitude and consecutive `v1` continuity do not explain the
  paired outcome split.

### Working hypotheses

- HIP may supply a useful low mode beyond the region where its full
  finite-step quadratic model is calibrated.
- Sella's model-ratio/trust update or destination-basin choice may drive the
  high-noise failures.
- The crossover depends on the starting distribution, especially isotropic
  noise versus path-generated GSM/NEB guesses.

### Not yet established

- That HIP `v1` is aligned with a native reaction path in GAD-only cases.
- That poor Sella model ratios precede trust collapse and are caused by the
  stable spectrum rather than by basin choice or optimizer configuration.
- That outcome-conditioned Taylor residuals predict the reported high-noise
  deficit before the trajectories diverge.
- That preserving HIP `v1` while changing step scale or stable-spectrum use
  reproduces the optimizer crossover.
- That the same diagnostic predicts optimizer ranking on a second directly
  learned Hessian model.
- That the 287 reactions are fully disjoint from HIP/HORM training data.
- That a frozen, validation-selected optimizer configuration preserves the
  reported per-cell best-of-grid crossover.

## Literature Grounding And Novelty Boundary

The interpretation is grounded in, but must be distinguished from, existing
work:

- [E and Zhou, The Gentlest Ascent Dynamics
  (2011)](https://doi.org/10.1088/0951-7715/24/6/008) establishes index-one
  saddles as stable fixed points of GAD-type dynamics and discusses
  non-gradient systems. It does not provide a general globalization guarantee
  from stiff high-index molecular starts.
- [Banerjee et al., Searches for Stationary Points on Surfaces
  (1985)](https://doi.org/10.1021/j100247a015) formulates stationary-point
  search through a rational local surface approximation using first and
  second derivatives. This is the mathematical reason full-model curvature
  quality matters to RFO.
- [Sella](https://doi.org/10.1021/acs.jctc.2c00395) implements a robust,
  automation-oriented constrained RS-PRFO framework. Our claim cannot be
  "Sella is a poor optimizer"; it is about how a good full-Hessian optimizer
  interfaces with independently supervised force and curvature channels.
- [Transition1x](https://arxiv.org/abs/2207.12858) contains roughly 9.6
  million DFT structures sampled on and around NEB reaction paths. Its authors
  explicitly show that reactive-region training data are necessary and note
  that independently random perturbations can create a different force
  distribution, especially on hydrogen.
- [HIP](https://arxiv.org/abs/2509.21624) directly predicts symmetric
  Hessians, uses a low-subspace-aware loss, reports improved `v1` and
  `lambda_2` accuracy, and shows that force/energy differentiation can amplify
  MLIP fluctuations into poor Hessians. It also reports successful RFO from
  GSM-generated starts, making start distribution an essential part of our
  hypothesis.
- [NewtonNet analytical Hessians for TS optimization
  (2024)](https://doi.org/10.1038/s41467-024-52481-5) shows that Sella with a
  differentiable, energy-consistent ML Hessian is robust on 240 unseen
  reactions and degraded guesses. This is an important negative control for
  any claim that full Hessians or Sella are intrinsically fragile.
- [Machine-Learned Leftmost Hessian Eigenvectors for Robust TS Finding
  (2026)](https://arxiv.org/abs/2603.21323) reports that a learned leftmost
  Hessian eigenvector can match full-Hessian success inside an RS-P-RFO
  framework with a secant-updated stable block. This supports the premise
  that the reaction direction is privileged, means "predict/use only `v1`"
  is not novel by itself, and shows that low-mode learning and RFO are
  complementary rather than opposites.
- [HORM](https://arxiv.org/abs/2505.12447) provides matched reactive
  Hessian-supervised model families and is the best eventual independent test
  of whether Hessian supervision changes optimizer sensitivity.

The possible novelty is therefore not that `v1` is useful. If the causal
tests succeed, it is:

> Directly learned Hessians can have different reliability radii for their
> low eigenspace and their finite-step quadratic model. Measuring that
> spectral reliability gap can determine whether to consume learned
> curvature as a full RS-P-RFO model, a low-mode constraint, or a
> force-reflection field.

An even stronger contribution would turn that observation into a diagnostic
or method-selection rule.

## Publication Assessment

For a demanding general ML venue:

- Current evidence: approximately `3/10` "wow" probability. It is a strong
  post-hoc optimizer crossover around one positive model, with no established
  mechanism.
- Causal HIP basin analysis plus controlled low-mode/stable-spectrum and step
  perturbations: approximately `6/10`.
  This is attainable without another large model integration.
- A second real model, a predictive reliability diagnostic, and a useful
  optimizer-selection/hybrid result: approximately `7-8/10`.

Difficulty estimates:

- Reaching the `6/10` version: about `5/10` difficulty and plausibly one to
  two focused weeks. Estimated success probability is roughly 45-50%,
  conditional on the paired traces supporting a reproducible basin mechanism.
- Reaching the `7-8/10` version: about `8/10` difficulty and several weeks.
  Estimated success probability is roughly 20-30%, because transfer to a
  second model is a scientific uncertainty rather than an engineering task.

These are judgment calls, not measured probabilities. The 2026 LMHE paper
reduces novelty for a generic "low mode is enough" story but increases the
plausibility of the narrower reliability-hierarchy story.

## Recommended Paper Claim

Do not lead with:

> GAD is better than Sella for transition-state optimization.

Lead, if the causal tests succeed, with:

> Directly learned curvature can have distinct reliability radii for its
> lowest mode and its full finite-step quadratic model. A diagnostic of that
> gap predicts when to use full RS-P-RFO, low-mode-constrained RFO, or
> force-reflection dynamics from degraded starts.

HIP is then the central empirical case, GAD and Sella are contrasting
curvature consumers, LJ/PaiNN provide coherent-curvature controls, and
controlled step and low-mode/stable-spectrum interventions establish
causality. This framing extends HIP's utility story by showing that direct
curvature can be exposed and consumed at different levels of granularity.

## Straight-Line Route To A Meaningful Result

This is the shortest path. Do not integrate another large MLIP before these
gates are resolved.

### Gate 0: Lock implementation and artifact identity

Before causal interpretation:

- Verify the historical GAD recurrence from stored first steps rather than
  inferring it from a Git timestamp.
- Pin Sella 2.3.4 and the HIP checkpoint hash, or rerun both methods under an
  explicitly new frozen environment.
- Define Cartesian Euclidean GAD or mass-metric GAD mathematically and keep
  the metric fixed.
- Record that the historical test Sella artifact used `delta0=0.1` and
  `gamma=0.4`; `gamma` is an eigensolver tolerance, not line search.
- Preserve the confirmed full-HIP-Hessian-at-every-step Sella condition.
- Select one GAD and one Sella configuration on validation data rather than
  choosing a per-cell test winner.

Decision:

- Do not attribute the crossover to landscape geometry until the stored
  results reproduce under declared implementations.

### Gate 1: Diagnose existing paired HIP outcomes

Use the existing `0.15 A` and `0.20 A` trajectories. Partition starts into
`both`, `GAD-only`, `Sella-only`, and `neither`.

For each start, measure:

- Initial `n_neg`, `lambda_1`, `lambda_2`, eigengap, and stiffness.
- Initial and early `v1` continuity.
- Force magnitude and distance to the labelled/native TS.
- Time to first enter an index-one region.
- First large step and first departure from the reference reaction tube.

The `0.15 A` summary job `65828024` is complete. It finds that initial
spectrum, first-step scale, and consecutive mode continuity do not explain
the outcome split. Reanalysis of both methods shows Sella departure followed
by near-total trust-floor occupancy in GAD-only cases. Extend this descriptive
analysis to `0.20 A` only after implementation identity is locked.

Decision:

- Continue with model-ratio instrumentation because trust collapse separates
  successful from failed Sella traces but its cause is not logged.
- If initial index or stiffness alone explains the split, prioritize
  globalization rather than the curvature-consumer interface.

Expected time: hours, using existing files and CPU only.

### Gate 2: Rerun a small, fully instrumented paired panel

Select 12 predeclared starts:

- Three `GAD-only`.
- Three `Sella-only`.
- Three `both`.
- Three `neither`.

Run both methods from byte-identical coordinates through the same HIP
calculator path. Record every proposed and applied Sella step and every GAD
step:

- Coordinates, force, energy, projected Hessian, and full spectrum.
- `v1` continuity and overlap with the TS/IRC tangent.
- Proposed and applied step.
- Sella trust radius, predicted/actual energy change, and model ratio `rho`.
- Distance to the TS and reaction-path tube.
- Step overlap with `v1`, the IRC/NEB tangent, and the displacement back
  toward the labelled TS.
- First bond-graph change and first crossing into a different saddle basin.

Align each pair at the first significant trajectory separation. Test whether
HIP `v1` remains aligned with a native path while Sella's model ratio
deteriorates before trust collapse. Consecutive `v1` overlap alone is not
chemical alignment.

Decision:

- H1 receives strong observational support if GAD-only cases show persistent
  `v1`/reaction-tangent alignment and poor Sella model ratios before
  trust-radius collapse and basin separation.
- If both trajectories remain in the same tube, inspect target-mode semantics,
  stopping criteria, and destination-saddle classification instead.

Expected compute: one smoke with four starts, then 24 paired trajectories.
Pack trajectories into a few 3g.20gb MIG jobs rather than one Slurm job per
trajectory.

### Gate 3: Separate mode information from step/globalization behavior

Do not replace the entire HIP Hessian with the force Jacobian. The Jacobian is
not a production-quality alternative and the result does not answer the
research question.

Run two compact intervention families.

First, match globalization:

- Reduce Sella's initial and maximum trust radius across a small predeclared
  validation-selected range.
- Compare applied per-atom displacement distributions with GAD. Existing
  first-step data show only a modest `0.100 A` versus `0.088 A` difference,
  so step scale is a control rather than the leading explanation.
- Compare the Sella model ratio and trust update, not a nonexistent
  accept/reject count.
- Increase GAD timestep only as a converse control.

If conservative Sella steps close the high-noise gap, the answer is mainly
globalization. If they only accelerate trust collapse, test the quadratic
model and basin semantics.

Second, separate low-mode from stable-spectrum information using only polished
HIP curvature or reference DFT/HORM curvature:

```text
H_A = original HIP Hessian

H_B = HIP lambda_1 and v1
      + isotropic/calibrated stable block orthogonal to v1

H_C = perturbed or reference lambda_1 and v1
      + HIP stable eigenvalues in an orthogonal completion

H_D = HIP v1
      + a stable block updated from force-displacement secants
```

For `H_B`, if `U` spans the complement of HIP `v1`, use

```text
H_B = lambda_1 v1 v1^T + U (U^T H_alt U) U^T
```

with the alternate stable block taken preferentially from a DFT/HORM reference
Hessian when available, or from a deliberately neutral isotropic model. This
preserves the mode whose utility is being tested while changing the
information and step anisotropy Sella consumes.

Important control: GAD's trajectory should be unchanged when `v1` is
preserved, apart from stopping criteria that count the altered eigenvalues.
Sella's trajectory is allowed to change. Recompute convergence from the
original HIP diagnostic Hessian as a separate evaluator so altered stopping
semantics do not create a false result.

The most decisive outcomes would be:

- Improving Sella's model ratio or trust policy rescues its intended-basin
  rate.
- Preserving HIP `v1` while simplifying the stable block preserves or
  improves robustness.
- Perturbing only `v1` removes the advantage even when the stable block is
  retained.

Start with four cases. Expand to 20-40 only if the direction of effect is
consistent.

Expected time: two to four days including implementation smokes.

### Gate 4: Test whether HIP's low-mode field is actually distinctive

Compare local landscape descriptors on matched starts from HIP, PaiNN,
DFTB0, and LJ:

1. Fraction of perturbations that remain index one.
2. Spatial continuity of `v1` under equal, dimensionless local displacement.
3. `v1` overlap with the reaction tangent.
4. Eigengap and spectral stiffness.
5. Distance traveled before the first mode crossing or basin change.

Candidate diagnostics:

```text
directional reliability = overlap(v1_hat, reaction tangent or reference v1)

reaction-path directional persistence =
    distance traveled while overlap(v1, native path tangent) remains high

basin-preserving step fraction =
    applied steps that reduce distance to the intended TS/path tube
```

The exact scalar should be chosen on a validation subset and then frozen. A
retrospective metric selected on the test outcomes is not evidence.

The highest-leverage additional test is an HIP checkpoint/loss ablation. If
the first author has MAE-, MSE-, and low-subspace-loss checkpoints from the
HIP paper, run the same fixed starts through them. A change in GAD/Sella
ranking with low-mode supervision would connect architecture/training,
landscape geometry, and optimizer behavior without introducing an unrelated
potential.

This is the fastest route from a HIP observation to a general mechanism and
is likely sufficient for the `6/10` paper version if the interventions are
clean.

## Route From A Solid Paper To A Strong General Paper

Only after Gates 1-4 succeed:

1. Build at least 50 PES-native, IRC-validated saddles for a second direct or
   Hessian-supervised model.
2. Prefer matched HORM E-F versus E-F-H models because they isolate Hessian
   supervision.
3. Use frozen GAD and Sella settings selected without access to the test
   outcomes.
4. Test whether the reliability-gap diagnostic predicts the winning
   optimizer across model families.
5. Implement an actionable method:
   - choose GAD versus RFO from the diagnostic;
   - blend from min-mode dynamics to RFO when quadratic reliability rises; or
   - make Sella trust the learned `v1` but learn/update the stable block from
     force differences.
6. Add a local perturbation analysis showing formally why GAD is insensitive
   to eigenvalue/stable-block error at fixed `v1`, while RFO is not.

That would support a substantially stronger contribution: not merely an
optimizer comparison, but a principled interface between learned curvature
and saddle optimizers.

## Stop Conditions And Alternative Answers

The project should be willing to reject the preferred mechanism.

- If a smaller Sella trust radius or different trust/globalization policy
  removes the crossover, the result is primarily basin globalization. That is
  less novel, but still useful if tied to a measured HIP path-neighborhood
  property.
- If `v1` itself becomes unstable before Sella fails, GAD's advantage may come
  from its step scale or mode-tracking implementation rather than HIP's
  low-mode fidelity.
- If isotropic noise creates close contacts or unphysical OOD geometries,
  while GSM/NEB-stage perturbations remove the advantage, the result is a
  benchmark-start-distribution effect.
- If training/test overlap is substantial, the HIP result cannot support a
  generalization claim until repeated on disjoint reactions.
- If low-mode/stable-block ablations and step matching do not predict the
  ranking, abandon the curvature-consumer narrative.

Any of these outcomes can still produce a useful technical result, but the
paper claim must change accordingly.

## Minimal Statistical Standard

- Use paired outcomes and McNemar tests for fixed benchmark cells.
- Report paired bootstrap confidence intervals for success-rate differences.
- Compare `GAD-only` directly with `Sella-only`; do not pool all failures.
- Cluster or stratify by reaction/formula where repeated structures are used.
- Freeze thresholds and diagnostic definitions on a validation subset.
- Report `fmax` threshold curves rather than one favorable cutoff.
- Keep `n_neg/fmax` and intended IRC/topology outcomes separate.
- Report all runs, including exceptions and evaluator failures.

## Compute And Experiment Layout

Keep project space for code, this synthesis, compact experiment JSON, and
paper-level tables. Keep trajectories, checkpoints, caches, environments, and
logs in `/lustre07/scratch/memoozd/gadplus/`.

For the critical path:

- Use CPU analysis for stored Parquet trajectories.
- Use a four-start test job before every HIP intervention.
- Pack multiple HIP trajectories into each GPU allocation.
- Prefer a small number of 3g.20gb MIG jobs for diagnostic panels.
- Use full A100s only after profiling demonstrates that batching or concurrent
  trajectories use them effectively.
- Do not submit hundreds of one-molecule jobs.
- Use `uv` for any dependency installation and keep incompatible model
  environments in scratch.

Every changed scientific assumption gets a child record under
`experiments/`. Every record declares a decision rule before expansion.

## Straight-Line Resumption Path

The benchmark condition that Sella receives the current full HIP Hessian at
every step is already confirmed. If the project is resumed, do only the
following before broad infrastructure or another model family:

1. Pin Sella 2.3.4, the HIP checkpoint hash, and one
   validation-selected configuration per method.
2. Run the already implemented four-case Taylor-compatibility smoke, then the
   predeclared 12-case paired panel if the smoke is numerically sound.
3. Record proposed/applied Sella steps, model ratio `rho`, predicted and
   actual energy change, trust radius, path distance, and destination saddle.
4. Test one reject-before-apply or force-consistent globalization control and
   one HIP-`v1`/alternate-stable-block control on the same cases.
5. Reject the Taylor-compatibility mechanism if residuals do not predict
   outcomes before divergence or if the tied intervention does not rescue
   failures.
6. Only after a causal HIP result, test matched HIP loss checkpoints and one
   independent PES-native reactive model family.

That is the shortest path from the current descriptive crossover to a
defensible mechanism. The project is closed without claiming those tests
succeeded.
