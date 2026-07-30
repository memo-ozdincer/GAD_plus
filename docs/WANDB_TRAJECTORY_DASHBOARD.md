# W&B trajectory observability design

## Purpose

This document specifies the W&B representation for matched transition-state
searches with competitive pointwise GAD, Sella, and regular GAD. The dashboard
must answer three questions quickly:

1. Did the optimizer produce a locally valid index-1 candidate?
2. Did its two downhill branches recover the intended endpoint topology?
3. What mechanism led to success or failure?

The dashboard is observational only. Logging, post-processing, and W&B state
must never enter an optimizer update. In particular, competitive GAD remains a
strictly pointwise map of the current coordinates, gradient, and Hessian.

## 1. Architecture: exact local record, derived remote view

Use a two-stage pipeline:

```text
optimizer/calculator
        |
        | append-only local observations
        v
trajectory.parquet + coordinates.npz + summary.json
        |
        | post-run enrichment and event-preserving view construction
        v
W&B run history + compact Tables + exact trajectory Artifact
```

The local files are the source of truth. The exporter computes hindsight-only
quantities after the search has ended, such as distance to the final candidate
and endpoint-topology labels. This separation has four advantages:

- no network request occurs inside a numerical Hessian or optimization loop;
- an interrupted Slurm job still leaves a recoverable trajectory;
- the same raw record can be re-exported when a chart definition changes;
- hindsight diagnostics cannot accidentally influence the dynamics.

On Trillium, compute jobs use `WANDB_MODE=offline` and write W&B spool files
under `/scratch/memoozd/gadplus/wandb`. Synchronization is a separate login-node
operation after the packed CPU job completes. API credentials are supplied only
through `WANDB_API_KEY`; they must not appear in code, configuration, Slurm
arguments, logs, or committed files.

## 2. W&B run identity

Create one W&B run for each

$$
(\text{campaign},\ \text{sample},\ \text{noise},\ \text{seed},\
\text{optimizer}).
$$

Use a deterministic run identifier derived from those fields so an interrupted
export can resume without creating a duplicate. Use:

- project: `gadplus-ts-mechanisms`;
- group: the matched comparison campaign;
- job type: `competitive-gad`, `sella`, or `regular-gad`;
- tags: surface, calculator, noise tier, terminal outcome, and code version.

### Immutable run configuration

Record the Git commit, dataset split, sample/reaction identifier, atom formula,
calculator and version, noise in angstrom, random seed, iteration/evaluation
budget, force and index tolerances, Hessian policy, mass convention, optimizer
parameters, Slurm job and task identifiers, and the endpoint-validation policy.

### Terminal summary

Every run exposes the following scalar summary fields:

- `calculator_valid` and any literal calculator error class;
- `local_ts`, `strict_ts`, final projected index, and final force maximum;
- `endpoint_a_is_minimum`, `endpoint_b_is_minimum`, and `native_topology`;
- terminal energy and the first evaluation satisfying each acceptance layer;
- energy, gradient, and Hessian evaluation counts separately;
- wall time, calculator time, optimizer time, and export time;
- termination reason, numerical warning flags, and fragmentation flag.

`native_topology` means the two downhill branches recover the labelled native
endpoint pair under the declared topology classifier. It is not inferred from
noise proximity and is not replaced by local TS convergence.

## 3. Exact per-evaluation schema

The common horizontal coordinate is `trajectory/evaluation`: the number of
coherent local Hessian evaluations. Do not use W&B's implicit logging step as
the scientific x-axis. Define `trajectory/evaluation` explicitly as the step
metric for all `trajectory/*` fields.

### Common fields

| Family | Fields |
|---|---|
| Identity | evaluation, optimizer iteration, wall time, energy/gradient/Hessian call counts |
| Stationarity | energy, energy minus start, force maximum, force RMS, gradient norm |
| Spectrum | projected index, $\lambda_1$ through $\lambda_5$, $s_H$, and $\lambda_i/s_H$ |
| Geometry | closest pair, maximum pair separation, radius of gyration, fragmentation flag |
| Step | Cartesian RMS, maximum atom displacement, mass-weighted RMS, step/$\ell$, step/$R$, cumulative path length |
| Hindsight | Kabsch RMSD to labelled TS, RMSD to recovered terminal saddle, best-so-far distance, raw and clipped progress, path efficiency |

The labelled-TS distance uses atom identity for Transition1x. For identical-atom
LJ clusters, use permutation-invariant alignment or a declared pair-distance
fingerprint. Report the alignment rule in run configuration.

Raw hindsight progress is

$$
P_t=1-\frac{d_t}{d_0}.
$$

It may be negative and must be retained. A clipped $[0,1]$ copy may be used in
a gauge, but never as the sole stored quantity.

### Competitive-GAD fields

Record the $\lambda_2$ gate $w_2$, final competitive weight $w_{\rm comp}$,
soft-mode activity, other-negative-mode activity, their fraction and log ratio,
soft-min weights $p_1,p_2$, spectral entropy, $\mu/s_H$, $R$, and the effective
lowest-mode multiplier

$$
m_1=1-2w_{\rm comp}p_1.
$$

These expose when the method descends, when it begins one-mode ascent, and
whether the local closed-form radius is active.

### Sella fields

Always record the common quantities derivable from accepted coordinates:
accepted displacement, force, energy, Hessian spectrum, projected index, and
evaluation counts. If the installed Sella interface provides stable hooks,
also record trust radius, actual/predicted model ratio, accepted/rejected trial
status, predicted change, and actual change. Optional fields remain missing
when unavailable; do not synthesize them from unrelated values.

### Regular-GAD fields

Record softest-mode gradient fraction $|c_1|^2/\|c\|^2$, signed total work
$g^T\delta q$, selected-mode and orthogonal work, displacement-cap fraction,
and overlap of consecutive instantaneous modes. Mode overlap is a diagnostic
only and must not become mode-tracking state.

## 4. Dashboard layout

Use a 24-column report grid. The default page opens only summary and mechanism
sections; raw records and artifacts remain collapsed. The color palette is
colorblind-safe:

- competitive GAD: `#0072B2`;
- Sella: `#E69F00`;
- regular GAD: `#7F7F7F`;
- $\lambda_1$: `#D55E00`;
- $\lambda_2$: `#CC79A7`;
- $\lambda_3$: `#009E73`;
- thresholds/reference lines: `#222222`, dashed.

### A. Campaign overview

The first row contains six 4-column cards: calculator-valid starts, local-TS
rate, strict-TS rate, native-topology rate, median Hessian evaluations, and
calculator-error rate. Every rate displays its denominator in the subtitle.

The second row contains:

- a 12-column stacked terminal-outcome bar, split into calculator error,
  index 0, index 1 above force tolerance, accepted local TS, and index $>1$;
- a 12-column scatter of Hessian evaluations versus native-topology recovery,
  with optimizer color and noise encoded by point shape.

The third row contains a topology-by-convergence matrix and an empirical CDF of
evaluations to first local-TS candidate. Runs that never achieve the event are
shown as right-censored at their budget, rather than silently dropped.

### B. Trajectory cockpit

Use one full-width custom Vega-Lite composite, not several unrelated W&B
panels. At normal report width it is approximately 1,450 pixels wide and 900
pixels high. A 45-pixel overview navigator sits above six vertically stacked
detail strips with a shared exact x-window:

1. `force_max / force_tolerance`, logarithmic y-axis, with a dashed line at 1;
2. $\lambda_1/s_H$, $\lambda_2/s_H$, and $\lambda_3/s_H$, symmetric-log y-axis
   with linear constant 0.01 and a zero reference;
3. projected index as an integer staircase, with event markers for every index
   change, first $\lambda_2=0$ crossing, first index-1 point, and acceptance;
4. Kabsch RMSD to the labelled TS and recovered terminal saddle, logarithmic
   y-axis;
5. step/$R$, step/$\ell$, and maximum-atom displacement, logarithmic y-axis;
6. energy minus starting energy, symmetric-log y-axis.

The overview shows force ratio and projected-index transitions. Its interval
brush filters the data supplied to every detail strip, rather than merely
changing their x-scale domains. Consequently, each free y-domain is recomputed
inside the selected interval and a very large early repulsive force does not
flatten the late convergence dynamics. Probability panels retain their fixed
$[0,1]$ domains. Dragging the navigator brush zooms all strips; dragging its
center pans the window; double-clicking clears the brush and restores the full
trajectory. A vertical hover rule shared by the detail strips snaps to the
nearest evaluation and reports raw eigenvalues with units, index, force, gate
state, displacement, distance, and event labels. Avoid dual y-axes. The report
also provides a full-screen link.

Recommended strip heights are 120 pixels for force, 165 for the spectrum, 75
for projected index, 130 for distance, 120 for step mechanics, and 120 for
energy. Only the bottom strip draws x tick labels. This leaves enough vertical
resolution for eigenvalue crossings without making the page slow to scan.

The scientific x-axis can be switched between Hessian evaluations and wall
time. Optimizer iteration is tooltip-only because rejected Sella trials and
different Hessian refresh policies can make iteration an unfair comparator.

### C. Competitive-GAD mechanism

Use another full-width linked composite with five strips:

1. $w_2$ and $w_{\rm comp}$ on a fixed $[0,1]$ axis;
2. soft activity fraction on $[0,1]$, with the raw activities and log activity
   ratio in the tooltip;
3. $p_1$, $p_2$, and normalized spectral entropy on $[0,1]$;
4. $\lambda_2/(\tau s_H)$ and $m_1$, with zero reference lines;
5. $\mu/s_H$, step/$R$, and $R$, separated into facets when their dimensions
   differ rather than placed on a dual axis.

This panel makes the causal interpretation immediate: the sign of $m_1$ shows
whether the softest component is currently descended or ascended, while the
activity panel explains why.

It reuses the cockpit's selected evaluation interval. If W&B isolates selection
state between custom panels in the installed version, embed the competitive
strips in the same Vega specification below the common cockpit instead of
pretending the panels are linked.

### D. Sella mechanism

Show accepted displacement, energy-model agreement, trust radius, and
accepted/rejected trials in aligned strips when those internals are available.
The common force/spectrum/index cockpit remains present even when they are not.
A visible `instrumentation_level` badge distinguishes public-coordinate-only
logging from deep Sella instrumentation.

### E. Regular-GAD mechanism

Show softest-mode gradient fraction, signed selected and orthogonal work,
displacement-cap utilization, and instantaneous mode overlap. Mark the final
index-0, high-index, force-limited, or budget-limited failure class directly in
the title.

### F. Population comparison

Never render 287 full trajectories simultaneously. Precompute cohort curves
with median and 10th--90th percentile ribbons. Provide two views:

- absolute Hessian evaluation, for efficiency comparisons;
- event-aligned evaluation $t-t_{\rm first\ index1}$ over a declared window,
  for mechanism comparisons around entry into the index-1 region.

Stratify by optimizer, noise, calculator validity, and final native topology.
Trajectory progress is a secondary alignment option only because it uses
hindsight and can hide inefficient paths.

### G. Molecular keyframes

Use a compact W&B Table with 3D molecule objects at no more than eight search
events: start, maximum observed index, first $\lambda_2$ crossing, largest gate
change, first index-1 point, minimum force, and terminal geometry. Duplicate
events collapse to one row. Put the two downhill endpoints in a separate
two-row endpoint table so their chemical role is visually unambiguous. Every
keyframe includes the same numeric hover fields as the cockpit. The complete
coordinate sequence belongs in the exact trajectory artifact, not in the
interactive table.

### Visual reading order

The default report is arranged as follows:

```text
| valid | local TS | strict TS | topology | median NHE | calc errors |
|------------- terminal outcomes ------------|------ Pareto -------|
|----------- topology/convergence ------------|----- event ECDF ----|
| navigator: complete trajectory with movable selected interval     |
| force / tolerance                                                 |
| normalized lambda_1, lambda_2, lambda_3                            |
| projected index and event markers                                 |
| hindsight distance                                                |
| bounded-step mechanics                                            |
| energy relative to start                                          |
| optimizer-specific mechanism                                      |
|-------------------- molecular keyframes --------------------------|
|------------------- two downhill endpoints ------------------------|
|---------------- population comparison (collapsed) ----------------|
|---------------- exact artifacts and raw tables (collapsed) -------|
```

## 5. Responsiveness and fidelity rules

Scalar history may be logged for every evaluation for trajectories up to 2,000
evaluations. Store chart input in wide form--one row per evaluation--so five
eigenvalues do not multiply the table length by five.

If a trajectory exceeds 5,000 records, construct an event-preserving view of at
most 2,500 rows. It must retain:

- start and final records;
- all projected-index changes and $\lambda_2$ sign crossings;
- first satisfaction of every acceptance condition;
- large gate changes, the largest displacements, and local force minima;
- per-bin minima and maxima for force, energy, and the first three eigenvalues.

The exact unsampled Parquet and coordinate artifact is always uploaded. Native
W&B scalar line panels may use full-fidelity aggregation for exploratory zoom;
the composite chart uses the compact event-preserving table so linked zoom and
hover remain responsive. Population curves are pre-aggregated to at most 200
x-bins per optimizer/outcome stratum.

There is no curve smoothing in a mechanism panel. Axes use normalized
quantities where physical scales differ across molecules, while tooltips retain
the raw value and unit. Eigenvalues and energy differences use symmetric-log
scales; forces and distances use log scales; probabilities and gate weights use
fixed $[0,1]$ scales. This prevents an outlier repulsive geometry from making
the scientifically relevant region visually flat.

For logarithmic charts, plot a documented positive display floor only when the
raw value is exactly zero; keep the raw zero in the tooltip and artifact. Do not
discard or winsorize outliers. The navigator-filtered y-domain, symmetric-log
scales, and normalized quantities supply the required readable zoom without
altering the underlying data.

## 6. Filters and comparison discipline

The persistent workspace filter bar exposes campaign, split, sample, formula,
optimizer, calculator, noise, seed, terminal index, local TS, strict TS,
native topology, fragmentation, and calculator validity. Matched-comparison
views default to the intersection of calculator-valid starts for the displayed
methods, but an adjacent card always reports each method's literal error count.

Rates must label the denominator explicitly. Recommended names are:

- `local TS / calculator-valid starts`;
- `native topology / calculator-valid starts`;
- `native topology / local TS candidates`.

The last quantity describes selectivity conditional on reaching a candidate;
it must not be presented as overall success.

## 7. Implementation contract

Implementation should have four independent pieces:

1. a method-neutral local observer that writes the common schema;
2. small method adapters that add competitive-GAD, Sella, or regular-GAD
   diagnostics without altering their updates;
3. a deterministic post-run enricher/exporter that computes hindsight metrics,
   creates keyframes, and uploads artifacts;
4. a versioned workspace/report builder plus versioned Vega specifications.

The exporter must be idempotent and capable of reconstructing W&B entirely
from local files. Pin a W&B SDK new enough to support current long API keys;
keep `wandb-workspaces` optional because its programmatic workspace API is a
public-preview surface. Tests should compare optimization coordinates with
logging enabled and disabled to prove instrumentation non-interference.

The first production export should occur only after the competitive-GAD
robustness profile has been frozen. Calibration runs and held-out comparison
runs use different W&B groups and tags so parameter selection cannot be
confused with evaluation.

### Frozen g-XTB rollout profile

The predeclared 12-reaction development sweep completed over
$\eta,\tau\in\{0.005,0.01,0.02\}$. Native-topology recovery was unchanged across
all tested $\eta$ values at both lower temperatures: 8/12 at 0.10 angstrom and
5/12 at 0.20 angstrom. At $\tau=0.02$, recovery fell consistently to 7/12 and
4/12. The predeclared central-plateau rule therefore retains

$$
\eta=0.01,\qquad \tau=0.01
$$

for production g-XTB competitive GAD. Export those runs with
`selection_stage=evaluation`; export the nine development cells with
`selection_stage=calibration` and a separate group.

## 8. Implemented resources and operation

The deployed private W&B chart presets are:

- `memo-ozdincer-university-of-toronto/gadplus-trajectory-cockpit-v1`;
- `memo-ozdincer-university-of-toronto/gadplus-competitive-mechanism-v1`.

The saved workspace is
[`GADplus TS mechanisms v1`](https://wandb.ai/memo-ozdincer-university-of-toronto/gadplus-ts-mechanisms?nw=62vj2ut3mwn).
The initial
[`analytic-LJ instrumentation smoke`](https://wandb.ai/memo-ozdincer-university-of-toronto/gadplus-ts-mechanisms/runs/8b8bf77f9b13843df51b)
is an observability check, not a benchmark result.

Implementation entry points are:

- `IntrinsicGADObservation` and the read-only `observer` argument in
  `src/gadplus/search/intrinsic_gad.py`;
- exact local bundles in `src/gadplus/logging/pointwise.py`;
- hindsight enrichment, event-preserving sampling, Artifacts, and W&B replay
  in `src/gadplus/logging/wandb_export.py`;
- versioned Vega specifications in `src/gadplus/logging/vega/`;
- repository-native packed g-xTB worker in
  `scripts/t1x_intrinsic_gxtb_pilot.py`;
- campaign replay in `scripts/export_wandb_campaign.py`;
- chart and workspace setup in `scripts/register_wandb_charts.py` and
  `scripts/create_wandb_workspace.py`.

Compute workers use `--record-trajectories` and write below
`CAMPAIGN_ROOT/trajectories`. They do not source a W&B credential and do not
import W&B. After a campaign completes, export from a login node:

```bash
cd /scratch/memoozd/GAD/GAD_plus
source /scratch/memoozd/GAD/secrets/wandb.env
PYTHONPATH=src /scratch/memoozd/GAD/.venv-wandb/bin/python \
  scripts/export_wandb_campaign.py \
  /scratch/memoozd/gadplus/runs/CAMPAIGN_ROOT \
  --group CAMPAIGN_GROUP \
  --entity memo-ozdincer-university-of-toronto
```

Create the lightweight exporter environment once with
`scripts/setup_wandb_observability.sh`. It is separate from the optimizer
environment because the local HIP package pins an older W&B SDK. The exporter
requires no Torch, HIP, g-xTB, or Transition1x installation; it only replays
the exact local files.

The exporter is deterministic and uses W&B resume mode, so replaying a bundle
does not intentionally create a second scientific run. The exact Parquet,
coordinates, references, and metadata are uploaded as one versioned trajectory
Artifact per run.
