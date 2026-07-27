# From ordinary GAD to pointwise intrinsic smooth-index GAD

This document explains the sequence of ideas that led from ordinary
one-mode Gentlest Ascent Dynamics (GAD) to the current pointwise intrinsic
optimizer. The progression is:

1. ordinary GAD, which fails on high-index Lennard–Jones starts;
2. a hard descent-to-GAD gate, which fixes the basic failure mechanism;
3. a smooth $\lambda_2$ gate, which removes the hard switch;
4. the current scale-covariant, degeneracy-safe, closed-form update.

The Lennard–Jones tests use an artificial cluster of seven identical particles
(LJ7), not a Transition1x molecule. LJ7 is valuable here because its energy,
gradient, and Hessian are mutually exact, so optimizer behavior can be studied
without ML-potential or electronic-structure error.

## Common geometry and notation

Let $q\in\mathbb R^{3N}$ be Cartesian coordinates, with energy $E(q)$,
gradient $g_q=\nabla E(q)$, force $F_q=-g_q$, Cartesian Hessian
$H_q=\nabla^2E(q)$, and diagonal mass matrix $M$.

Use mass-weighted coordinates and derivatives:

```math
x=M^{1/2}q,
\qquad
g_x=M^{-1/2}g_q,
\qquad
H_x=M^{-1/2}H_qM^{-1/2}.
```

At the current geometry, construct an orthonormal Eckart vibrational basis
$Q(q)$. It removes translations and rotations in mass-weighted space. The
projected Hessian is

```math
K(q)=Q^\top H_xQ.
```

Diagonalize it:

```math
K=V\Lambda V^\top,
\qquad
\Lambda=\mathrm{diag}(\lambda_1,\ldots,\lambda_m),
\qquad
\lambda_1\le\lambda_2\le\cdots\le\lambda_m.
```

Here $m=3N-6$ for a nonlinear cluster and $m=3N-5$ for a linear
molecule. Define the full mass-weighted vibrational modes

```math
U=QV
```

and the gradient coefficients in that instantaneous eigenbasis:

```math
c=U^\top g_x.
```

The numerical projected Morse index is

```math
n_{\mathrm{neg}}^{(\tau_I)}
=
\#\left\{i:\lambda_i<-\tau_I\right\},
```

with $\tau_I=10^{-4}$ in the current LJ experiments. Final TS acceptance is
separate from the optimizer dynamics:

```math
n_{\mathrm{neg}}^{(\tau_I)}=1
\qquad\text{and}\qquad
\lVert F_q\rVert_\infty<f_{\max}.
```

The maintained LJ result uses $f_{\max}=0.01$.

## 1. Ordinary one-mode GAD

Physical force descent has eigenbasis coefficients

```math
d_i^{\mathrm{desc}}=-c_i.
```

Ordinary one-mode GAD reverses only the component along the lowest-curvature
mode:

```math
d_i^{\mathrm{GAD}}
=
\begin{cases}
+c_1, & i=1,\\
-c_i, & i>1.
\end{cases}
```

Equivalently, define a modified gradient

```math
b_i^{\mathrm{GAD}}=(1-2\delta_{i1})c_i,
```

so that $d^{\mathrm{GAD}}=-b^{\mathrm{GAD}}$. In vector form,

```math
d^{\mathrm{GAD}}
=
-c+2(c_1)e_1.
```

The corresponding Cartesian Euler update is

```math
q^+
=
q+\Delta t\,M^{-1/2}U d^{\mathrm{GAD}}.
```

### Why ordinary GAD is locally correct at an index-one saddle

Near a stationary point, let $y_i$ denote displacement along mode $i$.
Then $c_i\approx\lambda_i y_i$. The linearized GAD flow is

```math
\dot y_1=+\lambda_1y_1,
\qquad
\dot y_i=-\lambda_i y_i\quad(i>1).
```

At an index-one saddle,

```math
\lambda_1<0<\lambda_2\le\cdots,
```

so every component contracts:

```math
\dot y_1=-\lvert\lambda_1\rvert y_1,
\qquad
\dot y_i=-\lambda_i y_i\quad(i>1).
```

Thus an index-one saddle is a local attractor of one-mode GAD.

### Why ordinary GAD fails on high-index LJ starts

At an index-$k$ point with $k>1$, the additional negative modes satisfy

```math
\lambda_i<0,
\qquad 2\le i\le k.
```

GAD flips only mode 1. Every additional negative mode evolves as

```math
\dot y_i=-\lambda_i y_i
=
+\lvert\lambda_i\rvert y_i,
```

and is therefore unstable. This is not an implementation bug; it is the
expected local stability structure of one-mode GAD.

Noised LJ7 starts frequently enter stiff, repulsive regions with five to nine
negative vibrational modes. Ordinary fixed-step GAD immediately ascends one
mode while diverging along the remaining negative modes. Reducing the timestep
or clipping the displacement limits overshoot, but does not remove this
high-index instability.

## 2. Simplest fix: hard descent, then GAD

The most direct repair is to suppress saddle ascent while the current geometry
has more than one negative mode. Define the hard gate

```math
h(q)=\mathbf 1\!\left[\lambda_2(q)\ge0\right].
```

Then use

```math
b_i^{\mathrm{hard}}
=
\left(1-2h(q)\delta_{i1}\right)c_i.
```

This has two regimes:

```math
\lambda_2<0
\quad\Longrightarrow\quad
h=0,
\quad
b^{\mathrm{hard}}=c,
\quad
d=-c,
```

which is ordinary vibrational force descent, and

```math
\lambda_2\ge0
\quad\Longrightarrow\quad
h=1,
\quad
b_1^{\mathrm{hard}}=-c_1,
```

which restores one-mode GAD once the local curvature is index-one-like.

The same idea can be written using the numerical index:

```math
d(q)=
\begin{cases}
d^{\mathrm{desc}}(q), & n_{\mathrm{neg}}>1,\\
d^{\mathrm{GAD}}(q), & n_{\mathrm{neg}}\le1.
\end{cases}
```

This simple intervention established the diagnosis: high-index entry, rather
than an incorrect LJ Hessian, caused most ordinary-GAD failures. In the
historical LJ7 study, hard descent until $n_{\mathrm{neg}}\le1$, followed by
GAD, raised strict convergence at noise $0.10/0.15/0.20\,\sigma$ to roughly
$99.0/97.2/96.5\%$.

### Limitation of the hard gate

The instantaneous rule is discontinuous at $\lambda_2=0$. Numerical noise
can make it chatter across the boundary. A one-way “descent, then permanently
lock into GAD” rule prevents chatter, but stores whether the switch has already
occurred and is therefore history-dependent.

The hard gate solves the main dynamical problem, but it is not the cleanest
pointwise vector field.

## 3. Historical smooth λ₂ gate

The next step replaces the hard indicator by a sigmoid:

```math
w_k(q)
=
\sigma\!\left(k\lambda_2(q)\right)
=
\frac{1}{1+\exp[-k\lambda_2(q)]}.
```

The modified gradient becomes

```math
b_i^{(\lambda_2)}
=
\left(1-2w_k(q)\delta_{i1}\right)c_i,
```

and the direction is

```math
d^{(\lambda_2)}=-b^{(\lambda_2)}.
```

Equivalently, in the vibrational mass-weighted space,

```math
F_{\mathrm{gate}}
=
F_{\mathrm{vib}}
-
2w_k(q)
\left(F_{\mathrm{vib}}^\top u_1\right)u_1.
```

Its limiting behavior is exactly the desired hard-gate behavior:

```math
\lambda_2\ll0
\quad\Longrightarrow\quad
w_k\approx0
\quad\Longrightarrow\quad
d^{(\lambda_2)}\approx d^{\mathrm{desc}},
```

and

```math
\lambda_2\gg0
\quad\Longrightarrow\quad
w_k\approx1
\quad\Longrightarrow\quad
d^{(\lambda_2)}\approx d^{\mathrm{GAD}}.
```

At $\lambda_2=0$, $w_k=1/2$, so the lowest-mode component is momentarily
suppressed rather than abruptly reversed.

The historical implementation used

```math
k=50,
\qquad
\Delta t=0.005,
\qquad
d_{\max}=0.005,
```

in reduced LJ units. It achieved strict convergence rates of approximately
$100.0/100.0/99.7\%$ at noise $0.10/0.15/0.20\,\sigma$.

This was the first nearly complete LJ recovery and showed that a smooth local
index gate is sufficient to globalize one-mode GAD on this surface.

### What remained inelegant

The historical method still had four avoidable weaknesses:

1. **Dimensional sharpness.** The product $k\lambda_2$ must be dimensionless,
   so the numerical value $k=50$ is tied to the LJ Hessian scale.
2. **Rank-one ambiguity.** The projector $u_1u_1^\top$ is not uniquely
   defined when the lowest eigenvalue is degenerate.
3. **Fixed Euler scale.** A fixed $\Delta t$ is surface-dependent and becomes
   unsafe near the LJ repulsive wall.
4. **A posteriori clipping.** The displacement is computed first and then
   clipped, adding a piecewise boundary rather than regularizing the step at
   its source.

These limitations motivate the current formulation.

## 4. Current formulation: pointwise intrinsic smooth-index GAD

The current method keeps the successful $\lambda_2$-gating mechanism but
replaces its dimensional, rank-one, and fixed-step components.

### 4.1 Dimensionless local curvature scale

Define the RMS vibrational curvature

```math
s_H(q)
=
\left(
\frac1m\sum_{i=1}^m\lambda_i(q)^2
\right)^{1/2}.
```

All spectral decisions use normalized eigenvalues $\lambda_i/s_H$. A
positive rescaling of the energy therefore does not change the gate or mode
weights.

The exact scale-covariant construction has the natural domain $s_H>0$. At
the completely flat matrix $K=0$, no nonzero homogeneous curvature scale
or distinguished spectral direction exists. The implementation uses a
machine-precision floor solely to return finite numerical diagnostics in that
singular case; the analytic claims below concern $s_H>0$.

### 4.2 Basis-invariant soft lowest-mode operator

Instead of selecting one eigenvector, define the normalized matrix soft-min

```math
\rho_{\tau_s}(K)
=
\frac{
\exp[-K/(\tau_s s_H)]
}{
\mathrm{tr}\exp[-K/(\tau_s s_H)]
}
=
V\mathrm{diag}(p_1,\ldots,p_m)V^\top,
```

where

```math
p_i
=
\frac{
\exp[-\lambda_i/(\tau_s s_H)]
}{
\sum_j\exp[-\lambda_j/(\tau_s s_H)]
}.
```

For finite dimensionless temperature $\tau_s>0$, $\rho_{\tau_s}(K)$ is an
analytic matrix function of $K$. Equal eigenvalues receive equal weights,
so the operator is invariant under arbitrary rotations within a degenerate
eigenspace. When the lowest mode is isolated and $\tau_s\to0$,

```math
\rho_{\tau_s}(K)\longrightarrow u_1u_1^\top.
```

This is the non-arbitrary replacement for a tracked or discontinuously chosen
lowest eigenvector.

### 4.3 Scale-covariant λ₂ gate

Use the dimensionless gate

```math
w(q)
=
\sigma\!\left(
\frac{\lambda_2(q)}{\tau_s s_H(q)}
\right).
```

The same $\tau_s$ controls the resolution of both the soft lowest-mode
operator and the index gate. In the instantaneous eigenbasis, define

```math
b_i(q)
=
\left(1-2w(q)p_i(q)\right)c_i(q).
```

For a separated lowest mode:

```math
\lambda_2\ll0
\quad\Longrightarrow\quad
w\approx0
\quad\Longrightarrow\quad
b\approx c,
```

so the method descends out of a clear high-index region. Conversely,

```math
\lambda_2\gg0,
\quad
p_1\approx1
\quad\Longrightarrow\quad
b_1\approx-c_1,
\qquad
b_i\approx c_i\ (i>1),
```

so it approaches ordinary one-mode GAD near an index-one saddle.

The method does **not** flip every negative mode. Doing so would make a
high-index saddle locally attractive, which is the opposite of the intended
index-one selection.

### 4.4 Geometry-scaled pointwise radius

Define a local length from the current pair distances:

```math
\ell(q)
=
\left[
\frac1{N_p}
\sum_{a<b}r_{ab}(q)^{-2}
\right]^{-1/2},
\qquad
N_p=\frac{N(N-1)}2.
```

This inverse-RMS pair length is permutation-invariant, rigid-motion-invariant,
and homogeneous under coordinate scaling. It also contracts smoothly toward
zero as any pair approaches collision.

Define the mass-weighted RMS step scale

```math
R(q)
=
\eta\,\ell(q)\sqrt{\sum_a m_a},
```

where $\eta$ is a dimensionless locality fraction.

### 4.5 Closed-form regularized step

Set

```math
\mu(q)
=
\frac{\lVert b(q)\rVert_2}{R(q)}.
```

Then compute the step coefficients directly:

```math
a_i(q)
=
-\frac{b_i(q)}{\sqrt{\lambda_i(q)^2+\mu(q)^2}}.
```

If $b(q)=0$, define $a(q)=0$ directly. This is the stationary-point
limit used by the implementation and avoids an indeterminate $0/0$ when a
zero gradient and zero curvature occur simultaneously.

The Cartesian update is

```math
\boxed{
q^+
=
q+M^{-1/2}U(q)a(q)
}
```

with $U=QV$.

This is a regularized eigenvector-following map. It is not fixed-step Euler
integration, an energy-minimizing line search, or a history-dependent adaptive
trust region.

### 4.6 Algebraic step bound

For $b\ne0$, every denominator satisfies

```math
\sqrt{\lambda_i^2+\mu^2}\ge\mu.
```

Therefore

```math
\lVert a\rVert_2^2
=
\sum_i
\frac{b_i^2}{\lambda_i^2+\mu^2}
\le
\frac{\lVert b\rVert_2^2}{\mu^2}
=
R^2.
```

Equivalently, the mass-weighted Cartesian RMS displacement obeys

```math
\boxed{
\frac{
\lVert M^{1/2}(q^+-q)\rVert_2
}{
\sqrt{\sum_a m_a}
}
\le
\eta\,\ell(q)
}.
```

The step is bounded when it is formed; it is never computed and then clipped.

### 4.7 Scale covariance

Under a positive energy rescaling

```math
E\longmapsto\alpha E,
\qquad \alpha>0,
```

the quantities $c$, $\lambda$, $s_H$, and $\mu$ all scale by
$\alpha$, while $p$, $w$, and $a$ remain unchanged.

Under a uniform mass rescaling

```math
M\longmapsto\gamma M,
```

the mass-weighted step scales by $\sqrt\gamma$, while the Cartesian
back-transform scales by $1/\sqrt\gamma$. The Cartesian update is unchanged.

### 4.8 Strictly pointwise and non-conservative

The complete map has the form

```math
q^+
=
\Psi\!\left(q,g_q(q),H_q(q)\right).
```

It uses no previous mode, previous geometry, iteration-dependent switch,
remembered radius, rejected trial, momentum, quasi-Newton history, reaction
path, or global pseudo-potential. Diagnostic history may be recorded, but it
does not enter $\Psi$.

The soft density operator is analytic. The ordered eigenvalue $\lambda_2$ is
continuous but can be nondifferentiable when eigenvalues cross. The resulting
map is therefore continuous and piecewise smooth on its natural domain
$s_H>0$ with no coincident atoms; it does not claim global $C^\infty$
smoothness.

### 4.9 Compatibility audit

The construction satisfies the four requirements for optimization on the
hybrid GAD field:

1. **Pointwise Markov property.** The next geometry is a deterministic
   function of the present $q$, $g_q(q)$, and $H_q(q)$. No trajectory
   state enters the update.
2. **Second-order local evaluability.** It requires no endpoint, reaction
   coordinate, global minimum, or integrated path. Energy is used for
   reporting, not step acceptance.
3. **Tolerance of a non-conservative field.** The map is evaluated directly;
   it never assumes that the reflected field is the gradient of a global
   scalar potential.
4. **Spectral-boundary robustness.** There is no hard branch on the Morse
   index. The density operator is basis-invariant at degeneracies, the
   ordered-$\lambda_2$ gate is continuous, and the step is algebraically
   bounded. Eigenvalue crossings may make derivatives nonsmooth, but do not
   make the update multivalued.

## 5. What the LJ7 experiments show

The present benchmark uses one artificial LJ7 cluster with many independently
noised starting geometries. It is an analytic optimizer-control experiment,
not a Transition1x molecular benchmark.

### Main noise sweep

For noise through $0.50\,\sigma$:

- $2880/2880$ trajectories reached the strict index-one and
  $f_{\max}<0.01$ gate;
- $2870/2880$ also produced two projected-minimum downhill endpoints under
  the current endpoint relaxation;
- 11 permutation-invariant saddle families were resolved;
- at least 8 endpoint-energy-pair families were observed;
- no accepted candidate was near-flat under $\lambda_2/s_H<0.01$;
- no accepted candidate was fragmented under the declared
  $1.5\,\sigma$ connectivity cutoff.

In the targeted high-noise tail cells at $0.60$, $0.75$, and
$1.0\,\sigma$, every tested trajectory reached the strict TS gate and two
valid downhill minima. At $1.5\,\sigma$, strict convergence fell to
$46/48$, and approximately 17–22% of the accepted candidates were near-flat
or fragmented. Thus physical-quality filters, rather than local optimizer
convergence, define the useful high-noise boundary.

### Diversity is not failure

As noise increases, recovery of the original reference saddle decreases, but
strict saddle and endpoint validity remain high. For a diffusion-terminal
application, this is desirable: the optimizer should map dispersed samples to
nearby valid index-one basins, not collapse every sample to one predetermined
saddle.

The appropriate metric hierarchy is therefore:

1. strict index-one and force convergence;
2. two valid downhill endpoint minima;
3. energy/barrier, spectral-gap, and fragmentation filters;
4. duplicate-adjusted saddle and event diversity;
5. recovery of a particular labelled event only when the benchmark supplies
   one.

The current downhill test displaces both signs of the unstable mode and
minimizes each branch. It is an IRC-like endpoint screen, not a discretized
intrinsic reaction-coordinate integration.

### Comparison with the historical smooth gate

On paired LJ7 cells, both the historical $k=50$ Euler gate and the current
pointwise method achieved essentially complete strict convergence. The current
method typically required roughly an order of magnitude fewer pointwise
iterations because its step uses local curvature directly rather than a small
fixed Euler timestep.

The historical gate supplied the essential dynamical idea. The current method
makes that idea dimensionless, degeneracy-safe, algebraically bounded, and
strictly pointwise.

## 6. Scope and next validation

The LJ7 result establishes optimizer mechanics on an exact many-particle
surface. It does not establish chemical performance. In particular, LJ7 does
not test unequal elements, realistic bonding, ML-potential Hessian error,
labelled reactant/product connectivity, or chemical `IRC_TOPO` recovery.

The next chemically meaningful benchmark is the 287-reaction Transition1x
test split:

1. noise the labelled Transition1x transition structures to measure local
   capture basins;
2. test mode-pushed reactants to measure nonlocal search;
3. run on actual diffusion outputs;
4. require the TS gate and two-branch IRC endpoint minima;
5. report labelled `IRC_TOPO` recovery, valid novel events, barriers,
   fragmentation, spectral separation, and duplicate-adjusted diversity.

The implementation is in `src/gadplus/search/intrinsic_gad.py`. Mathematical
and LJ regression tests are in `tests/test_intrinsic_gad.py`, and the complete
LJ noise-study results are in
`docs/research/LJ_INTRINSIC_GAD_NOISE_2026_07_26.md`.
