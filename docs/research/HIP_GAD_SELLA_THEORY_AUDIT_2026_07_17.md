# Theory Audit: HIP, GAD, and Sella

Last updated: 2026-07-17

This document audits the proposed explanation for the HIP GAD/Sella
crossover against:

- the mathematical guarantees of GAD;
- the actual RS-P-RFO implementation used by Sella;
- the HIP, Transition1x, HORM, NewtonNet, and LMHE papers;
- the stored paired HIP trajectories; and
- the implementation and artifact provenance in this repository.

It is deliberately more critical than the research synthesis. Attractive
language such as "learned reaction tube" is treated as a hypothesis unless an
experiment has measured the corresponding object.

## Closure Status

The project was wrapped on 2026-07-17. The Taylor-compatibility diagnostic was
implemented and predeclared, and its analytic LJ numerical smoke passed.
Pending HIP and SCINE jobs were cancelled before scientific evaluation, so
there is no outcome-conditioned Taylor-panel result and no causal claim. The
documents preserve both the strongest current explanation and the conditions
that would falsify it if work resumes.

## Bottom Line

The current data establish a noise-dependent optimizer crossover on one HIP
checkpoint. They do not yet establish why it occurs.

The most defensible working explanation is:

> From isotropically degraded Transition1x starts, HIP may preserve a useful
> leftmost direction even when its full local quadratic model is not
> sufficiently predictive for repeated RS-P-RFO steps. GAD consumes only that
> direction and the force at the current point. Sella consumes a partitioned
> quadratic model, applies the restricted step, and then adapts its trust
> radius from the energy-model ratio. In the stored GAD-only cases, Sella
> moves somewhat toward the reference saddle, then moves away and spends most
> of its run at the minimum trust radius. This makes trust-model mismatch and
> basin selection credible mechanisms. It does not yet prove that HIP has a
> broad reaction tube, that Sella's steps are generally too large, or that
> HIP's stable Hessian spectrum is inaccurate.

There are three serious null explanations that must be eliminated before a
paper claim:

1. The comparison may depend on optimizer version, metric, stopping, update
   frequency, or per-cell hyperparameter selection.
2. Isotropic Cartesian noise may create a HIP-specific out-of-distribution
   benchmark rather than a generally relevant transition-state search regime.
3. The methods may be selecting different valid saddles rather than differing
   in stationary-point robustness.

## Claim Audit

| Claim | Verdict | Correction |
|---|---|---|
| GAD is better than Sella | Rejected as a general claim | Sella wins on HIP near the labelled saddle and on pure LJ, DFTB0, and the current HORM screen. |
| HIP has a broad learned reaction tube | Plausible but unmeasured | Define a tube from a PES-native path and measure distance and tangent overlap. Index-one topology alone is not a reaction tube. |
| HIP's `v1` stays chemically useful far from the saddle | Plausible but unmeasured | Consecutive `v1` overlap is not enough. Compare with a path tangent, reference `v1`, bond-change vector, and destination saddle. |
| GAD uses only `v1`, while Sella uses the full Hessian | Directionally correct but oversimplified | GAD's position field depends on the force and a rank-one projector. Sella uses RS-P-RFO in ascending and stable subspaces with a trust region; the benchmark supplies a current full HIP Hessian at every step. It is not raw Newton inversion. |
| GAD wins because its steps are much smaller | Not supported by the paired data | At `0.15 A`, median first maximum-atom displacements are about `0.088 A` for GAD and `0.100 A` for Sella. The difference is modest. Later step sizes are outcome-dependent. |
| Sella takes one huge step out of the basin | Not established | Stored GAD-only Sella traces show departure followed by trust-radius collapse, not a single universal catastrophic jump. Proposed and applied step geometry still needs instrumentation. |
| Sella rejects bad proposals | False for Sella 2.3.4 | `PES.kick` applies the step, computes actual/predicted energy change, updates the Hessian, and then changes the next trust radius. The current path does not reject and roll back an energetically poor step. |
| Sella `gamma` controls line search | False for Sella 2.3.4 | `gamma` is an iterative eigensolver tolerance. With `hessian_function` and an exact refresh, it is bypassed. The historical `gamma=0.4` result is not a line-search ablation. |
| The headline Sella baseline uses the full HIP Hessian at every optimization step | Confirmed project provenance | Treat this as the benchmark condition. Do not use quasi-Newton mixing as an explanation for the crossover. |
| Direct Hessian supervision favors GAD | Plausible, not demonstrated | The HIP low-subspace loss modestly improves low-mode metrics, but HIP also reports strong RFO performance. A matched HIP loss/checkpoint ablation is required. |
| Full learned Hessians are intrinsically fragile for RFO | Contradicted | HIP's own GSM-to-RFO experiment and NewtonNet's full-Hessian Sella benchmark are strong counterexamples. |
| LJ proves GAD is broken | Rejected | LJ starts at high noise are stiff and high-index. Ordinary one-vector GAD has no global guarantee there; an index-aware descent gate fixes most strict failures. |
| LJ proves ordinary GAD is unstable whenever `n_neg > 1` | Too strong | The local linear stability theorem says an ordinary GAD fixed point is stable only at an index-one saddle. It does not imply every trajectory through every high-index region must fail. |
| The force Jacobian explains the HIP crossover | Unsupported and off-axis | It is not HIP's intended Hessian product. It remains an implementation diagnostic, not the scientific comparator. |

## What GAD Theory Actually Guarantees

For a conservative potential `E(x)`, force `F = -grad E`, and a normalized
leftmost Hessian eigenvector `v`, the instantaneous minimum-mode form is

```text
xdot = F - 2 v (v^T F).
```

The reflection reverses the force component parallel to `v` while retaining
descent in its orthogonal complement.

E and Zhou prove a local fixed-point statement for the coupled continuous
dynamics: stable fixed points correspond to index-one saddles, with `v`
aligned to the unstable direction. This does not provide:

- global convergence from arbitrary molecular geometries;
- robustness in regions with several negative modes;
- robustness to an inaccurate or discontinuous learned mode;
- stability of explicit Euler integration at an arbitrary timestep; or
- intended-saddle recovery.

The original paper explicitly notes that GAD can fail to converge globally
and can exhibit numerical instabilities. It also constructs examples where
the GAD basin is larger than Newton's, but that example is not a general
ordering of basin sizes.

### Metric dependence

GAD depends on the chosen inner product. This matters whenever masses are
introduced.

For Cartesian coordinates `x`, diagonal mass matrix `M`, and mass coordinates
`q = M^(1/2) x`,

```text
g_q = M^(-1/2) g_x
H_q = M^(-1/2) H_x M^(-1/2).
```

Euclidean GAD in `q`, transformed back to `x`, gives

```text
xdot = M^(-1) F - 2 u (u^T F),
u = M^(-1/2) v_q,
u^T M u = 1.
```

This is a mass-metric dynamics. It is not the same trajectory as Euclidean
Cartesian GAD,

```text
xdot = F - 2 v_x (v_x^T F),
H_x v_x = lambda v_x.
```

Neither metric is automatically "the" correct optimizer. It must be declared
and kept fixed. Mass weighting preserves Hessian inertia under a full
positive-definite congruence, but it changes eigenvectors, eigenvalue
ordering, and dynamics.

The reported project convention is to use masses for the Eckart/vibrational
construction, then **remove that weighting** by un-mass-weighting the direction
back to Cartesian coordinates before applying the displacement. The
implementation that returned a still-mass-weighted displacement errored and
produced no reported benchmark data; it is not a candidate explanation for the
HIP, LJ, or SCINE results. A direct first-step replay remains the authoritative
artifact provenance check; Git timestamps alone are not.

### Eckart projection away from stationarity

At a stationary molecular geometry, removing rigid translations and rotations
is the standard way to identify vibrational curvature and count imaginary
modes. Away from stationarity, the separation is less exact:

- rotational tangent directions need not be zero modes of the Cartesian
  Hessian because gradient-dependent terms appear;
- a projected ambient Hessian is not generally identical to the covariant
  Hessian on the rotation/translation quotient manifold; and
- the projected `n_neg` can therefore be an optimizer-dependent local
  descriptor rather than an invariant chemical fact.

This does not invalidate Eckart projection. It means that high-noise
trajectory claims should be checked under a clearly defined metric and, where
possible, against a path-based chemical descriptor.

## What Sella Actually Does

The historical HIP runs used Sella 2.3.4 with Cartesian RS-P-RFO,
`order=1`, an external HIP Hessian, and translation/rotation constraints.

RS-P-RFO:

1. diagonalizes or updates an approximate Hessian;
2. partitions the lowest `order`-dimensional subspace from the stable
   complement;
3. solves separate rational-function problems in the ascent and descent
   subspaces;
4. restricts the Cartesian step by a trust radius;
5. applies the step;
6. compares actual and local-model energy changes; and
7. shrinks or expands the trust radius for the next step.

This is substantially more robust than an unrestricted Newton step. A fair
mechanistic comparison must not describe Sella as simply inverting the full
Hessian.

The benchmark provenance is that Sella receives the current full HIP Hessian
at every optimization step. The scientific comparison must therefore be
interpreted as full-Hessian RS-P-RFO versus lowest-mode GAD, not as a
full-Hessian method versus a quasi-Newton baseline. HIP's paper distinguishes
RFO with predicted Hessians from RFO-BFGS initialized by a predicted Hessian;
the repository result belongs to the former category.

### The important Sella failure signal

At `0.15 A`, the stored paired outcomes are:

| Outcome | Count |
|---|---:|
| Both | 133 |
| GAD only | 34 |
| Sella only | 22 |
| Neither | 98 |

Recomputed from both stored trajectory sets using the same per-atom
displacement definition:

| Median diagnostic | Both | GAD only | Sella only | Neither |
|---|---:|---:|---:|---:|
| GAD first max-atom step, A | 0.090 | 0.088 | 0.088 | 0.085 |
| Sella first max-atom step, A | 0.100 | 0.100 | 0.100 | 0.100 |
| Sella minimum distance to labelled TS in first 100 steps, A | 0.108 | 0.226 | 0.117 | 0.196 |
| Sella final distance to labelled TS, A | 0.108 | 0.447 | 0.128 | 0.328 |
| Sella final trust radius, A | 0.0029 | 0.0001 | 0.0011 | 0.0001 |
| Fraction of Sella rows at trust floor | 0.000 | 0.979 | 0.000 | 0.984 |

The first-step scale does not distinguish GAD-only from Sella-only starts.
The dominant Sella failure signature is that almost the entire failed run is
spent at the minimum trust radius after earlier movement has led elsewhere.

This is consistent with at least three mechanisms:

1. the HIP Hessian/energy local model predicts poor RFO energy changes in
   those regions;
2. Sella enters a different basin, after which shrinking the trust radius
   cannot undo the basin choice; or
3. the Hessian update/trust policy becomes trapped on a nonstationary
   plateau.

Only logging `rho`, predicted/actual changes, applied step projections, and
destination chemistry can separate them.

### Sella details that must be corrected in project records

- The test trajectory baseline actually records `delta0=0.1` and
  `gamma=0.4`, not `delta0=0.048`.
- `gamma` is not line search in Sella 2.3.4. It controls iterative
  eigensolver convergence and has no effect during an external full-Hessian
  refresh.
- Sella's own ASE convergence uses the maximum per-atom force norm. The
  repository's strict evaluator uses the maximum absolute Cartesian force
  component plus projected `n_neg=1`. These criteria are close but not
  identical.
- Sella applies a trial step before adapting the trust radius. Instrument
  "proposed/applied step" and model ratio, not "accepted/rejected step."
- The project dependency is `sella>=2.3`, while the current environment has
  Sella 2.5.0. Reproduction must pin the historical version or intentionally
  rerun both methods under a new frozen environment.
- Low-noise and high-noise tables select different completed Sella
  configurations. A publication claim needs one validation-selected method or
  must label the table as per-cell oracle selection.

## What HIP Training Does And Does Not Imply

The HIP paper supports the following facts:

- HIP directly predicts a symmetric Hessian supervised by DFT Hessians.
- Its MAE+subspace objective includes the eight lowest reference modes,
  covering the six rigid modes and two lowest nontrivial modes.
- Relative to MAE-only training, the reported subspace loss improves the first
  eigenvector cosine and second-eigenvalue error modestly, while slightly
  worsening full-matrix and eigenvalue MAE.
- HIP is trained on approximately 1.73 million HORM-Transition1x geometries.
- The paper reports strong RFO transition-state results from GSM-generated
  guesses.

The paper does not establish:

- that `v1` remains aligned with an IRC/NEB tangent under `0.1-0.2 A`
  isotropic noise;
- that HIP has a larger attraction basin than conservative MLIPs;
- that the stable spectrum is too inaccurate for RFO;
- that direct Hessian supervision favors GAD over RFO; or
- that the repository checkpoint is exactly each reported paper ablation.

The HIP ablation itself is highly relevant. It suggests a possible
intermediate spectral-reliability regime, but it does not prove one.

## Strongest Technical Explanation: Taylor-Jet Compatibility

The most substantive explanation produced by this investigation is not that
HIP is a bad PES, nor that its direct Hessian should be replaced by a force
Jacobian. It is that different optimizers require different compatibility
relationships among the products exposed by a learned model.

Write HIP's shipped products as

```text
E_hat(x)       learned scalar energy
F_hat(x)       learned direct force
H_hat(x)       directly predicted, DFT-supervised Hessian
g_F(x) = -F_hat(x)
```

Sella's local quadratic energy model is effectively

```text
m_x(s) = E_hat(x) + g_F(x)^T s + 0.5 s^T H_hat(x) s.
```

For that model to become accurate as the trust radius shrinks, these three
products must approximate one Taylor jet. Let

```text
g_E = grad E_hat
H_E = Hessian E_hat.
```

Then

```text
E_hat(x+s) - E_hat(x) - g_F^T s - 0.5 s^T H_hat s

  = (g_E - g_F)^T s
    + 0.5 s^T (H_E - H_hat) s
    + O(norm(s)^3).
```

This separates two qualitatively different failure modes.

### First-order energy/force incompatibility

For `s = delta d`,

```text
rho(delta d)
  = [E_hat(x+delta d)-E_hat(x)]
    / [delta g_F^T d + 0.5 delta^2 d^T H_hat d].
```

If `g_E != g_F`, then generically

```text
lim(delta -> 0) rho(delta d) = (g_E^T d) / (g_F^T d),
```

not one. Shrinking the trust radius therefore need not repair the model ratio.
This is a mathematically direct candidate explanation for failed Sella traces
spending almost all of their iterations at the trust-radius floor.

### Second-order force/Hessian incompatibility

Even when the energy and force agree to first order, the Hessian supplied to
RFO may not predict the finite force change:

```text
F_hat(x+s) - F_hat(x) + H_hat(x)s.
```

For a coherent smooth field this residual is `O(norm(s)^2)`. A persistent
`O(norm(s))` term means the full Hessian is not the local derivative of the
force field that the optimizer follows. RFO can then receive a chemically
accurate DFT-informed Hessian that is nevertheless a poorly calibrated
finite-step model of the learned force surface.

These are optimizer-interface statements. They do not rank `H_hat` against a
DFT label and do not imply that an energy Hessian or force Jacobian is a better
scientific product.

## Why GAD Can Exploit A Weaker Contract

Once a unit vector `v1` is supplied, the position-only force reflection is

```text
F_GAD = (I - 2 v1 v1^T) F_hat.
```

It does not use:

- `E_hat`;
- the magnitude of `lambda1`;
- any stable-subspace eigenvalue; or
- a predicted finite energy change.

If the true useful direction is `u1`, the operator error obeys

```text
norm(v1 v1^T - u1 u1^T) = sin(theta),
```

where `theta` is the principal angle between the modes. Consequently,

```text
norm(F_GAD(v1) - F_GAD(u1))
    <= 2 norm(F_hat) sin(theta).
```

This gives a precise sense in which a reliable low-mode projector can remain
useful while eigenvalues and the full quadratic model are less reliable.
HIP's `k=8` low-subspace objective makes this regime structurally plausible,
although it has not been measured on the failed trajectories.

RFO is more demanding. Its stable components are Newton-like and remain
sensitive to soft-mode curvature, schematically `s_i ~ -g_i/lambda_i`, while
its trust update uses the agreement between the learned energy change and the
mixed force/Hessian model. Restricted-step RFO regularizes this sensitivity;
it does not eliminate the need for a locally useful model.

### Formal GAD caveat

The original GAD theory distinguishes gradient and non-gradient systems. For a
non-gradient vector field, the formal method uses the Jacobian and associated
left/right eigenvectors. Reflecting a direct force with the lowest eigenvector
of a separately predicted symmetric Hessian is therefore best described
mathematically as a learned-mode force-reflection flow, not as an exact
instance of the conservative GAD theorem.

At a root `F_hat(x*) = 0`, derivatives of the mode field multiply the zero
force and vanish from the first linearization. Local stability is governed
approximately by

```text
(I - 2 v1 v1^T) J_F(x*).
```

Empirical success consequently implies that HIP's predicted `v1` is aligned
well enough with the relevant unstable structure of the force field. It does
not prove global convergence or identify a unique underlying scalar PES.

## Critical Reading Of The HIP Non-Integrability Argument

HIP Appendix A.5.1 correctly notes that directly predicted Hessians are not
guaranteed to be integrable and argues that, unlike long molecular dynamics,
geometry optimization does not accumulate an unbounded time-integrated error.
That supports the practical use of direct Hessians, but it does not establish:

- agreement of the energy-ratio trust model;
- convergence of RS-P-RFO with independently predicted energy, force, and
  Hessian products;
- preservation of the intended saddle basin; or
- recovery of model agreement merely by shrinking the trust radius.

The comparison with BFGS also needs qualification. A sequence of BFGS matrices
is not a globally integrable Hessian field, but each update is constructed
from displacement/gradient secants of the same gradient field being optimized.
A directly predicted Hessian can be closer to the DFT Hessian while being less
secant-compatible with a separately learned force head. DFT accuracy and
optimizer-local compatibility are different properties.

This is best framed as a useful design feature and interface question:
direct supervision can provide curvature information not recoverable from a
rough learned energy or force derivative, but the optimizer should decide
whether to consume that information as a full Taylor model, a low-mode
projector, or a diagnostic.

## Reaction Mode Is Not Reaction Path

At an index-one stationary point, the mass-weighted imaginary normal mode
gives the local initial IRC direction.

Away from the saddle:

- an IRC tangent follows mass-weighted steepest descent;
- an NEB tangent follows the discrete reaction path;
- the leftmost Hessian eigenvector is a local curvature direction; and
- these three directions need not agree.

Therefore, consecutive overlap `abs(v1(x_k)^T v1(x_{k+1}))` measures field
continuity, not chemical correctness. The stored HIP GAD traces have median
early consecutive overlaps near one for every outcome group, so this metric
does not explain the paired separation.

A defensible "learned reaction tube" requires:

```text
tube distance(x) = min_s aligned_distance(x, path(s))

tangent alignment(x) =
    abs(v1(x)^T tangent(path at nearest s))
```

The path must be native to the evaluated PES. A labelled T1x path can be used
as a DFT reference, but it is not automatically a HIP, DFTB0, or HORM-native
path.

## Why Other PES Results Do Not Yet Prove The HIP Mechanism

### Lennard-Jones

LJ provides the cleanest mathematical negative control:

- exact shared energy, force, and Hessian;
- extremely stiff repulsive starts at high noise;
- median initial index around eight at the hardest cell; and
- large recovery from an index-aware descent or smooth `lambda2` gate.

This supports a high-index globalization explanation for LJ. It does not show
that HIP's GAD advantage comes from low-mode supervision. LJ and HIP differ
simultaneously in chemistry, stiffness, index distribution, metric, start
distribution, and intended-saddle semantics.

### DFTB0

DFTB0's `n_neg/fmax` results favor Sella at moderate and high noise. This is
compatible with a coherent-curvature explanation, but it does not prove it.
The T1x labelled points and intended endpoint identities are not native to the
DFTB0 surface, and equal Cartesian noise does not create equal local
difficulty.

### NewtonNet

NewtonNet is the strongest published counterexample to a generic
"full-spectrum RFO is fragile" claim:

- conservative energy and analytical autograd Hessian;
- Transition1x plus ANI-derived coverage, including compressed bonds;
- Sella supplied with full Hessians; and
- robust performance on degraded transition-state guesses.

Its contrast with HIP suggests concrete hypotheses: conservative differential
consistency, repulsive/compression coverage, and start distribution. None has
yet been isolated in this repository.

### LMHE

The 2026 LMHE work predicts the leftmost eigenvector but still consumes it
inside Sella RS-P-RFO with a secant-updated stable block. This has two
consequences:

1. "Only `v1` is needed" is no longer novel by itself.
2. Low-mode learning and RFO are complementary, not opposing categories.

A strong project contribution would explain when a directly learned full
Hessian should be used as a full quadratic model, as a low-mode constraint,
or only as a diagnostic.

## Why The Cross-PES Pattern Is Consistent With This Explanation

| Surface/model | Derivative relationship | Expected optimizer consequence |
|---|---|---|
| HIP checkpoint | Learned energy and direct force plus a separately DFT-supervised direct Hessian | A useful low mode can coexist with weaker full Taylor-model agreement; GAD may exploit the former while Sella also encounters the latter. |
| Exact Lennard-Jones | Energy, force, and Hessian are exact derivatives of one analytic function | Sella receives a coherent full model. Pure GAD's hard-cell failure instead comes from stiff, high-index globalization outside its local guarantee. |
| SCINE DFTB0 | Energy, gradient, and Hessian come from the same electronic-structure method | Full curvature is internally coherent, so Sella's ordinary second-order advantage is expected. T1x intended-saddle labels remain a separate semantic confound. |
| NewtonNet | Conservative energy with analytical autograd Hessian and broad off-equilibrium training | Published full-Hessian Sella robustness is a counterexample to generic full-Hessian fragility and supports consistency/coverage as candidate variables. |
| HORM autograd models | Hessian is tied to the learned scalar energy | Useful conservative controls, once PES-native saddles exist. |
| HORM direct-force E-F-H models | Hessian constraints improve the derivative of a direct-force architecture but energy/force consistency is not automatic | Best matched family for testing whether the optimizer boundary tracks derivative-channel compatibility. The existing four-case screen is not sufficient. |

HIP's successful published RS-P-RFO result is not contradictory. Those starts
come from a GSM/path-local workflow, whereas the repository crossover appears
under much larger isotropic Cartesian perturbations. Compatibility can be
direction- and distribution-dependent: a model may be fully adequate near a
reaction path but less predictive after compressed-bond or off-path
perturbations.

## Evidence Ledger For The Taylor-Compatibility Hypothesis

### Evidence in support

1. A completed held-out directional diagnostic found that HIP's shipped
   products are observably distinct. Across 15 random directions, the recorded
   medians were:

   ```text
   abs(F.v + dE/ds)                         0.322 eV/A
   norm(H.v + dF/ds) / norm(dF/ds)         0.076
   abs(v.H.v - d2E/ds2)                   27.738 eV/A^2
   ```

   The matched conservative PaiNN diagnostic was orders of magnitude smaller.
   This establishes local product incompatibility on the sampled panel, not
   its causality for the optimizer crossover.
2. In GAD-only HIP cases, Sella moves away and then spends a median `97.9%` of
   stored rows at the minimum trust radius. Persistent model-ratio
   incompatibility predicts exactly this qualitative failure mode.
3. HIP explicitly supervises the lowest Hessian subspace and reports strong
   lowest-mode metrics. It is therefore plausible that directional information
   has a different reliability radius from the complete quadratic model.
4. Exact LJ and internally coherent DFTB0 do not reproduce the high-noise HIP
   crossover. Their negative results are consistent with, although not proof
   of, a derivative-interface explanation.
5. The ICML 2025 study by Bigi, Langer, and Ceriotti independently reports
   ill-defined geometry-optimization convergence as a failure mode of
   non-conservative direct-force models.

### Evidence against or still missing

1. Initial index, `fmax`, eigengap, first-step size, and consecutive `v1`
   overlap do not separate GAD-only from Sella-only HIP starts.
2. GAD and Sella first maximum-atom displacements are similar (`~0.088 A`
   versus `0.100 A`), so a simple small-step/large-step explanation is false.
3. The HIP paper reports strong RFO performance, demonstrating that direct
   Hessian prediction is not intrinsically incompatible with RFO.
4. No completed experiment has yet shown that Taylor residuals are larger on
   the specific trajectories where Sella fails.
5. No intervention has yet shown that replacing the energy-ratio
   globalization, retaining only `v1`, or changing the stable block rescues
   the predeclared failures.
6. Consecutive mode continuity is not chemical correctness. Alignment to a
   PES-native path or destination saddle remains unmeasured.
7. The current HORM set is too small and not sufficiently PES-native to support
   a general direct-force versus conservative conclusion.

## Complete Competing-Hypothesis Set

The current evidence does not justify selecting only one mechanism.

| Hypothesis | Mechanism | Distinguishing prediction |
|---|---|---|
| H1: first-order product incompatibility | `F_hat != -grad E_hat`, so Sella's energy ratio does not approach one as the radius shrinks | `r_E1 = O(delta)` in failures; bad `rho` persists at small `delta` |
| H2: second-order product incompatibility | `H_hat` does not predict finite changes of the force head | `r_F2 = O(delta)` and stable-block replacement improves RFO |
| H3: privileged low-mode reliability | HIP's DFT/subspace-trained `v1` remains useful beyond the full model's reliability radius | Path/reference-mode overlap remains high; perturbing `v1` hurts both GAD and a low-mode RFO control |
| H4: isotropic-noise distribution shift | Large Cartesian noise creates compressed bonds and off-path geometries outside the training distribution | Crossover tracks minimum distance/OOD scores and weakens for GSM/NEB starts |
| H5: basin semantics | Sella reaches other valid saddles while GAD more often returns the labelled saddle | Any-saddle success gap is smaller than intended-IRC gap |
| H6: trust-policy hysteresis | Sella applies a poor step before shrinking and cannot undo the basin departure | A reject-before-apply or alternative globalization rescues matched failures |
| H7: ordinary optimizer tuning | Version, trust parameters, coordinate treatment, or stopping criterion create the crossover | Frozen validation-selected settings or another RS-P-RFO implementation closes the gap |
| H8: HIP-specific spectral shape | Soft stable modes or small gaps make full RFO steps unusually sensitive even when products are locally coherent | Residuals are normal, but stable-spectrum conditioning predicts failure and regularization rescues it |

The mass transform is not a candidate explanation for the recorded results:
masses were used for Eckart construction and removed by un-mass-weighting
before applying the Cartesian GAD displacement. The implementation that kept
the weighting produced no benchmark data.

## Revised Mechanistic Hypotheses

The hypotheses should be predeclared and tested against direct alternatives.

### H1: Low-mode reliability exceeds quadratic-model reliability

HIP preserves a useful `v1`, but the full predicted Hessian is less reliable
for predicting finite force/energy changes under isotropic off-path
displacements. GAD remains useful because its position field depends only on
the force and the rank-one projector at the current point.

Predictions:

- `v1` remains aligned with a native path or reference mode in GAD-only cases;
- Sella model ratios become poor before trust collapse;
- preserving HIP `v1` while replacing the stable block rescues Sella; and
- perturbing `v1` hurts both GAD and the rescued Sella variant.

### H2: Start distribution, not HIP curvature, creates the crossover

Isotropic noise samples compressed bonds or other regions poorly represented
by HORM-T1x. GAD happens to be more robust under that benchmark, while
path-local GSM/NEB starts favor RFO.

Predictions:

- the crossover weakens under real path-stage starts;
- it tracks minimum distance, force, energy, and an OOD score more strongly
  than nominal RMSD;
- adding compression/off-path coverage changes the result; and
- matched-difficulty starts reduce the difference across PESs.

### H3: Basin semantics create apparent optimizer outperformance

Both methods find valid saddles, but GAD more often returns the labelled
T1x saddle from isotropic perturbations.

Predictions:

- stationary-point success differs less than intended IRC recovery;
- Sella failures often terminate at alternate valid saddles; and
- outcome depends strongly on symmetry and reaction branching.

### H4: Sella configuration creates the crossover

The trust parameters, optimizer version, coordinate system, or stopping
semantics are unfavorable for this checkpoint.

Predictions:

- a frozen but properly validated Sella configuration closes the gap;
- the effect transfers poorly to pysisyphus RS-P-RFO used in the HIP paper;
  or
- intermediate trajectories satisfy the common criterion before Sella's own
  stricter stop and later leave it.

## Shortest Decisive Experiment Path

### Predeclared Taylor-compatibility test

The primary mechanism test uses the shipped HIP energy, direct force, and
direct Hessian as products in their own right. It does not substitute a force
Jacobian for HIP's Hessian.

For a geometry `x`, an observed Sella direction `d` normalized so that its
largest atomic displacement is one, and probe length `delta`, define

```text
s = delta d
Delta E = E(x + s) - E(x)
Delta F = F(x + s) - F(x)

r_E1 = abs(Delta E - g_F^T s)
r_E2 = abs(Delta E - g_F^T s - 0.5 s^T H_HIP s)
r_F2 = norm(Delta F + H_HIP s)
```

Here `g_F = -F`. A coherent smooth energy/force/Hessian Taylor jet has
`r_E1 = O(delta^2)`, `r_E2 = O(delta^3)`, and `r_F2 = O(delta^2)`. A persistent
energy/direct-force discrepancy instead gives `r_E1 = O(delta)` and cannot be
repaired by shrinking an energy-ratio trust radius. A force/Hessian
discrepancy with a coherent first derivative gives `r_F2 = O(delta)` and
`r_E2 = O(delta^2)`.

The fixed panel contains three deterministically sampled `0.15 A` test cases
from each paired outcome class (`both`, `GAD only`, `Sella only`, `neither`).
For each case, evaluate directions from the initial, departure, and
trust-floor portions of the stored Sella trajectory at maximum-atom probe
lengths `0.08, 0.04, 0.02, 0.01, 0.005 A`. Report both the raw model Hessian
and the Eckart-cleaned, un-mass-weighted Cartesian Hessian actually supplied
to Sella; the latter is the primary result. Use the same diagnostic on exact
Lennard-Jones and, if practical, SCINE DFTB0 controls.

The hypothesis is supported only if all of the following hold:

1. HIP GAD-only/neither trajectories have materially worse small-step model
   agreement than both/Sella-only trajectories, especially near departure or
   trust collapse.
2. Exact LJ shows the expected Taylor orders and does not show persistent
   small-step ratio failure.
3. The failure class is predictable from residuals before the terminal
   outcome, rather than merely differing after the methods have diverged.
4. A compact intervention that removes or replaces the incompatible
   energy-ratio globalization improves the predeclared failed cases.

If residual scaling does not distinguish paired outcomes and a corresponding
globalization intervention does not rescue Sella, reject this mechanism rather
than broadening the experiment.

Execution status: the analytic LJ implementation smoke reproduced the
expected fitted orders across its sampled phases: approximately `2.00-2.03`
for the first-order energy residual, `2.87-3.05` for the second-order energy
residual, and `1.99-2.05` for the force/Hessian residual. This validates the
diagnostic numerics on an exact potential. It does not test the HIP mechanism.
All HIP and SCINE jobs associated with this panel were cancelled while
pending; see `experiments/2026-07-17-hip-taylor-compatibility.json`.

### 1. Lock implementation identity

- Verify the historical GAD recurrence from stored first steps.
- Pin Sella 2.3.4 and the HIP checkpoint hash.
- Record and preserve the confirmed full-HIP-Hessian-at-every-step Sella
  condition.
- Define either Cartesian Euclidean GAD or mass-metric GAD explicitly.
- Use one common projected diagnostic for terminal scoring.
- Freeze one validation-selected `dt` and one Sella configuration.

This is mandatory before interpreting optimizer mechanics.

### 2. Instrument 12 paired HIP cases

Use three predeclared cases from each outcome class. Log:

- proposed and applied step;
- maximum atom displacement;
- `v1`, full spectrum, and subspace projections of the step;
- Sella `rho`, predicted energy change, actual energy change, and trust
  radius;
- force-change residual `||Delta F + H Delta x||`;
- distance and tangent overlap to a PES-native or carefully labelled path;
- minimum interatomic distance and bond graph; and
- first entry into another stationary-point basin.

The first causal discriminator should be chosen from this panel, not from
terminal outcomes.

### 3. Run two compact interventions

**Trust/model intervention**

- Reduce Sella's initial and maximum trust radius on validation cases.
- Compare with a policy that reacts before applying a poor model step, if
  available in a standard optimizer.
- Compare with pysisyphus RS-P-RFO if its HIP-paper setup can be reproduced.

**Spectral-consumer intervention**

- Keep HIP `v1` and use a neutral or secant-updated stable block.
- Keep HIP's stable block and perturb or replace only `v1`.
- Score using the unmodified HIP Hessian and native IRC evaluator.

The LMHE paper makes `v1 + secant stable block` a particularly well-grounded
control.

### 4. Use the highest-leverage model ablation

Run matched HIP MAE, MSE, and MAE+subspace checkpoints on the same starts if
the first author can provide them.

This directly tests:

```text
low-mode-targeted training
    -> low-mode/path reliability
    -> GAD/Sella crossover
```

It is more informative than replacing HIP with an unrelated force Jacobian.

### 5. Test generality only after the HIP mechanism survives

The best comparator is a conservative reactive model with a full analytical
Hessian and broad off-equilibrium coverage, especially NewtonNet. HORM
E-F/E-F-H pairs become valuable once a native saddle set exists.

Match starts by local difficulty:

- projected index;
- `fmax`;
- harmonic displacement energy;
- minimum pair distance;
- spectral stiffness; and
- model OOD score.

Do not compare equal raw Angstrom noise across PESs as though it were equal
optimizer difficulty.

## Paper-Level Interpretation

The current result is not yet a publishable mechanism. A publishable,
coauthor-compatible result could be:

> Direct Hessian prediction exposes several levels of useful curvature
> information. Under path-local starts, the full HIP Hessian supports
> efficient RFO. Under sufficiently degraded starts, low-mode information may
> remain useful beyond the region where the full finite-step quadratic model
> is well calibrated. Measuring this spectral reliability gap enables a
> better curvature-consumer or hybrid optimizer.

This frames HIP's decoupled curvature channel as a useful feature and design
opportunity. It does not require claiming that HIP is a bad potential or that
Sella is a bad optimizer.

The result becomes substantially stronger if:

- the model-ratio/trust-collapse mechanism is causal rather than descriptive;
- HIP loss ablations move the crossover as predicted;
- a reliability diagnostic predicts the winning consumer; and
- the same rule transfers to one independent reactive MLIP.

### Objective project verdict

The present result is a credible research hypothesis, not a demonstrated
mechanism. The generic statement "non-conservative learned products can cause
optimizer problems" is not novel by itself: direct-force pathology, inexact
trust-region requirements, and HIP's Hessian non-integrability are already in
the literature.

The potentially novel statement is narrower:

> Directly supervised curvature can remain useful as a low-dimensional
> directional signal outside the region where the model's full
> energy/force/Hessian bundle is a reliable finite-step Taylor model. This
> creates a predictable optimizer crossover and motivates a
> reliability-aware curvature consumer.

Without the outcome-conditioned residual test and a causal intervention, this
is approximately a `4/10` paper position. A clean causal result plus one
matched direct-force/conservative model family could plausibly reach `6-7/10`.
The project is wrapped here without claiming that those experiments succeeded.

## Primary Literature

- E and Zhou, [The Gentlest Ascent
  Dynamics](https://doi.org/10.1088/0951-7715/24/6/008), 2011.
- Banerjee et al., [Search for Stationary Points on Surfaces](https://doi.org/10.1021/j100247a015),
  1985.
- Hermes et al., [Sella, an Open-Source Automation-Friendly Molecular Saddle
  Point Optimizer](https://doi.org/10.1021/acs.jctc.2c00395), 2022.
- Schreiner et al., [Transition1x](https://arxiv.org/abs/2207.12858), 2022.
- Yuan et al., [Full analytical Hessians for transition-state optimization
  with neural network potentials](https://doi.org/10.1038/s41467-024-52481-5),
  2024.
- Cui et al., [HORM](https://arxiv.org/abs/2505.12447), 2025.
- Burger et al., [Shoot from the HIP](https://arxiv.org/abs/2509.21624), 2025.
- Wu et al., [Machine-Learned Leftmost Hessian Eigenvectors for Robust
  Transition State Finding](https://arxiv.org/abs/2603.21323), 2026.
- Bigi et al., [The Dark Side of the Forces: Assessing Non-Conservative Force
  Models for Atomistic Machine Learning](https://arxiv.org/abs/2412.11569),
  ICML 2025.
- Carter, [On the Global Convergence of Trust Region Algorithms Using Inexact
  Gradient Information](https://doi.org/10.1137/0728014), 1991.
- Bandeira et al., [Convergence of Trust-Region Methods Based on Probabilistic
  Models](https://doi.org/10.1137/130915984), 2014.
