# Multi-size Lennard-Jones benchmark of CS²-GAD

## Status

The frozen benchmark completed on 2026-08-09. Production job `2076554` ran
all 1,344 trajectories under `def-aspuru` in `03:14:36` and exited `0:0`.
All 1,344 records were calculator-valid, with no calculator exceptions, and
the strict post-analysis verified byte-identical paired starts. The immutable
aggregate and derived analysis are under
`/scratch/memoozd/gadplus/runs/lj-multisize-cs2-2076554/`.

The result does **not** support CS²-GAD alone as a universal LJ catchall. It
improves strict recovery over intrinsic `lambda2` on LJ38 and LJ55, is mixed
on LJ13 and LJ31, and loses substantially on LJ75. Sella is retained only as
an external comparator because it is not an available downstream search
method. The operational result is therefore GAD-family specific: intrinsic
`lambda2` is the base method and CS² is a complementary recovery/diversity
channel.

## Question

Can Competitive Soft-Spectral GAD (CS²-GAD) serve as a common Lennard-Jones
saddle optimizer without losing the performance of the established intrinsic
`lambda2` profile, especially on the frustrated or double-funnel LJ31, LJ38,
and LJ75 landscapes?

This is not a global-minimum benchmark. Cluster landscape difficulty supplies
structurally different local environments, but the measured task is local
index-one saddle capture from controlled perturbations of known minima.

## Cluster panel

| Cluster | Reference basin(s) | Landscape role |
|---|---|---|
| LJ13 | global icosahedral minimum | small symmetric single-funnel control |
| LJ31 | global Mackay-overlayer minimum | frustrated competing-overlayer case |
| LJ38 | fcc global and lowest icosahedral minima | archetypal double funnel |
| LJ55 | global Mackay icosahedron | larger single-funnel size control |
| LJ75 | Marks-decahedral global and lowest icosahedral minima | hard double funnel |

Coordinates come from the Cambridge Energy Landscape Database and are stored
with checksums in `references/lj_global_minima/`. The runner independently
recomputes analytic LJ energy, force, and projected index, then tightly
relaxes each finite-precision database geometry with analytic-gradient
L-BFGS-B, restarting from the relaxed endpoint up to three times when needed.
Starts are constructed only from a relaxed projected minimum with `fmax<1e-5`;
both source and relaxed diagnostics are frozen in `protocol.json`.

## Frozen methods

1. **Intrinsic lambda2 GAD**, the established LJ profile:
   `gate_variant=lambda2`, `tau_s=0.01`, `eta=0.05`.
2. **CS²-GAD**, the selected g-xTB profile:
   `gate_variant=competitive_subspace`, `tau_s=0.01`, `eta=0.01`.
3. **Sella**, Cartesian Eckart RS-P-RFO with the current analytic Hessian
   supplied after each kick.

The comparison uses the documented production profiles rather than forcing a
common step fraction. Evaluation count and wall time must therefore accompany
success rates.

## Frozen starts

Every method receives byte-identical starts for each cell.

- `mode_push`: displace a reference minimum along one of its first eight
  projected vibrational modes, normalize to the declared per-atom Cartesian
  RMS displacement, choose signs deterministically, and add `0.01 sigma`
  Gaussian symmetry-breaking jitter.
- `cartesian_noise`: add independent Cartesian Gaussian noise directly to the
  reference minimum and remove center-of-mass translation.
- Levels: `0.10` and `0.20 sigma`.
- Replicates: 16 deterministic starts per method/basin/family/level cell. For
  `mode_push`, this balances both signs of each of the first eight modes.
- Seed base: `20260809`.

The complete production grid contains 1,344 trajectories:

```text
3 methods x 7 reference basins x 2 start families x 2 levels x 16 starts
```

## Gates and reporting

Primary local success requires projected `n_neg=1` at eigenvalue threshold
`-1e-4` and `fmax<0.01` in reduced force units. Every local success receives
a two-branch unstable-mode displacement followed by analytic-gradient
minimization. Endpoint validity requires both branches to have projected
index zero and `fmax<1e-5`.

Report, without changing denominators:

- calculator-valid, strict-TS, and two-minimum counts;
- terminal index-zero, index-one/force-limited, and multi-negative counts;
- median and tail evaluation cost;
- near-flat candidates using `lambda2/s_H<0.01`;
- fragmentation under the declared `1.5 sigma` connectivity graph;
- permutation-invariant saddle and endpoint-energy-pair diversity;
- paired CS²-minus-intrinsic outcomes for every identical start.

The two-branch endpoint screen establishes local downhill minima. It is not a
discretized IRC and does not by itself prove recovery of a particular
database transition or inter-funnel path.

## Compute protocol

The launcher requests one complete 192-core Trillium CPU node under
`def-aspuru`. Each trajectory is an independent exclusive one-core Slurm
step, with up to 192 steps active and a replacement launched whenever one
finishes; there is no nested Python process pool. This dynamically packs the
analytic-Hessian workload across the node rather than oversubscribing one
trajectory. Completed calculator-valid task files are restartable and the
launcher refuses to replace a pre-existing incompatible protocol.

Job `2076554` was submitted from the first launcher revision, which grouped
trajectories into fixed 192-task waves. At 566 completed records, its third
wave exposed a ten-task LJ75 tail and poor full-node utilization. The launcher
was therefore changed on 2026-08-09 to dynamically refilled, one-trajectory
exclusive steps for any continuation or rerun. This scheduling-only change
does not alter the frozen worker or any scientific result, and the running
allocation was retained rather than sent back through the queue.

- Runner: `scripts/lj_multisize_cs2_benchmark.py`
- Launcher: `scripts/run_lj_multisize_cs2_benchmark.slurm`
- Pooled/matched analysis: `scripts/summarize_lj_multisize_results.py`
- Output convention:
  `/scratch/memoozd/gadplus/runs/lj-multisize-cs2-$JOB_ID/`

## Run ledger

| Job | Purpose | Account | State | Counted scientifically? |
|---:|---|---|---|---|
| 2076528 | five-step launcher smoke; LJ38 intrinsic subset only because comma-valued `sbatch --export` was parsed incorrectly | def-aspuru | completed | no |
| 2076537 | preliminary full-matrix smoke; cancelled after the reference-tightening change made successive waves version-mixed | def-aspuru | cancelled | no |
| 2076542 | finalized-runner smoke; stopped in preparation when LJ31 missed the reference force gate by `4.8e-7` | def-aspuru | failed safely | no |
| 2076543 | iterative-tightening five-step full-matrix smoke; 1,008/1,008 calculator-valid task results, zero exceptions | def-aspuru | completed in 6m47s | no |
| 2076554 | frozen 1,344-trajectory production grid, 500-update budget | def-aspuru | completed in 03:14:36, exit 0:0 | yes |

The pooled size tables, terminal-index failure taxonomy, and matched per-start
discordance tables were generated with:

```bash
.venv/bin/python scripts/summarize_lj_multisize_results.py \
  /scratch/memoozd/gadplus/runs/lj-multisize-cs2-2076554 \
  --expected-tasks 1344
```

This writes `analysis.json` and `ANALYSIS.md` beside the immutable raw results.
The run used repository base commit
`6e4ff47a15cae426e8efe8508777ea57ea2ae3b4` plus the benchmark files recorded
in this working tree.

## Production results

| Method | LJ13 | LJ31 | LJ38 | LJ55 | LJ75 | Pooled |
|---|---:|---:|---:|---:|---:|---:|
| intrinsic `lambda2` | 56/64 (87.5%) | 63/64 (98.4%) | 115/128 (89.8%) | 49/64 (76.6%) | 112/128 (87.5%) | 395/448 (88.2%) |
| CS²-GAD | 59/64 (92.2%) | 60/64 (93.8%) | 122/128 (95.3%) | 58/64 (90.6%) | 100/128 (78.1%) | 399/448 (89.1%) |
| Sella | 60/64 (93.8%) | 61/64 (95.3%) | 122/128 (95.3%) | 61/64 (95.3%) | 118/128 (92.2%) | 422/448 (94.2%) |

These are strict local index-one successes; each denominator includes every
predeclared start. Pooled CS² exceeds intrinsic by only four successes, while
the direction changes with cluster size. Pooling therefore hides the main
scientific result.

| LJ | CS² only | Intrinsic only | Net CS² | exact two-sided McNemar p |
|---:|---:|---:|---:|---:|
| 13 | 8 | 5 | +3 | 0.5811 |
| 31 | 1 | 4 | -3 | 0.3750 |
| 38 | 12 | 5 | +7 | 0.1435 |
| 55 | 14 | 5 | +9 | 0.0636 |
| 75 | 12 | 24 | -12 | 0.0652 |

No single size reaches a conventional `p<0.05` threshold on its own, so the
size-dependent signs and failure modes are more informative than a binary
significance claim. CS²'s LJ75 loss is associated with 23 terminal
multi-negative structures, versus zero for intrinsic `lambda2`. Across all
sizes, intrinsic had no terminal multi-negative failures; CS² had 40. Sella
had 11 and combined the best pooled strict rate with much smaller tails.

## Endpoint and diversity screen

| Method | LJ13 | LJ31 | LJ38 | LJ55 | LJ75 |
|---|---:|---:|---:|---:|---:|
| intrinsic: two minima | 52 | 30 | 71 | 14 | 33 |
| CS²: two minima | 53 | 35 | 70 | 19 | 30 |
| Sella: two minima | 53 | 37 | 64 | 15 | 33 |
| intrinsic: saddle families / endpoint pairs | 9/6 | 16/11 | 26/19 | 14/5 | 47/18 |
| CS²: saddle families / endpoint pairs | 13/9 | 35/21 | 63/37 | 19/7 | 64/23 |
| Sella: saddle families / endpoint pairs | 20/10 | 50/30 | 76/41 | 36/12 | 89/24 |

The downhill screen is deliberately stricter than the local TS gate. It
shows that CS² can broaden the recovered saddle/event set, especially on
LJ31, LJ38, and LJ75, but it does not rescue its lower LJ75 strict rate. The
family counts are fingerprint-based and endpoint pairs use rounded energies;
they are diversity diagnostics, not a proof of distinct database pathways.

Median valid evaluation counts for CS²/intrinsic/Sella were respectively
`32.5/15.0/12.0` (LJ13), `33.0/21.0/21.5` (LJ31), `38.0/44.0/25.5`
(LJ38), `32.0/23.0/20.0` (LJ55), and `44.0/30.5/24.0` (LJ75). A value of
`501` in several p95 tails identifies budget-limited GAD-family cases; Sella
generally had the lower cost tail.

## Decision

- Do not replace the maintained intrinsic LJ profile with CS² globally.
- Use CS² as a useful additional LJ search channel, particularly for LJ38 and
  LJ55 or when diversity is valuable.
- Prefer intrinsic `lambda2` over CS² on LJ75 under this protocol.
- Treat Sella's 422/448 only as a robustness comparator, not as a candidate
  for the downstream optimizer step.
- Keep the molecular CS² selection separate. This exact-LJ result does not
  prejudge the HIP benchmark, where unequal masses, learned curvature, and
  labelled chemistry change the problem.

## GAD-only route toward a catchall

Running intrinsic `lambda2` and CS² on the same start and accepting either
strict result gives 442/448 (98.7%) strict recovery. The paired union is
64/64 for LJ13, 64/64 for LJ31, 127/128 for LJ38, 63/64 for LJ55, and
124/128 for LJ75. Thus the two profiles are genuinely complementary and
already reach 100% on the smaller LJ13/LJ31 controls without using Sella.

The six misses are all budget-limited at 500 updates: one LJ38, one LJ55,
and four LJ75 starts. Across the failed pair, intrinsic most often terminates
at an index-zero structure while CS² often remains multi-negative; one LJ75
intrinsic endpoint is index one but force-limited. This predeclares the next
GAD-only recovery target clearly: a separate continuation/restart experiment
on exactly these six starts, preserving them as a fixed denominator and
reporting the additional evaluation cost. It must remain separate from the
frozen 1,344-trajectory test matrix so that attempts to reach 100% do not
rewrite the completed benchmark.

That post-hoc recovery sweep was first launched as Trillium CPU job `2077217`
under `def-aspuru`. A live utilization check showed that this Slurm version
gave the first exclusive step all memory inherited from the allocation's
`--mem=0`, serializing later steps despite their one-core requests. One valid
task was preserved, the allocation was cancelled after 2m31s, and the launcher
was corrected to request 3 GiB per step and skip completed valid files.
Resume job `2077219`, also under `def-aspuru`, then ran 191 Python workers
concurrently alongside the preserved task on the same 192-core node.

The experiment uses exactly one full 192-core node: six fixed misses times a
predeclared 32-profile grid spanning the two GAD gates, four profile-specific
step fractions, two spectral temperatures (`0.005`, `0.01`), and two budgets
(`2,000`, `5,000`). Every trajectory reconstructs and exactly checks its
parent start; the parent protocol and aggregate are locked by SHA256. This is
exploratory rescue evidence, not an unbiased revision of the parent rate.

- Rescue runner: `scripts/lj_gad_rescue_benchmark.py`
- Full-node launcher: `scripts/run_lj_gad_rescue.slurm`
- Output (shared across the restart):
  `/scratch/memoozd/gadplus/runs/lj-gad-rescue-2077217/`

The complete resume job `2077219` finished on `tri0235` after `7:09:59` with
all 192 task IDs present, all 192 calculator-valid, and zero errors. Of those
post-hoc profiles, 89/192 (46.4%) were strict and 36/192 (18.8%) reached two
tight downhill minima. Every one of the six selected parent misses had at
least one strict rescue and at least one two-minimum rescue. Thus a GAD-only
*post-hoc exploratory* strict union reaches 448/448, while adding the six
endpoint-valid starts raises the parent's endpoint union only from 287/448 to
293/448. Neither number replaces the frozen, unbiased 442/448 paired-GAD
strict result because the rescue grid was chosen after observing its misses.

| LJ | Basin | Start | Level | Strict profiles | Endpoint profiles | Best strict evaluations |
|---:|---|---|---:|---:|---:|---:|
| 38 | global | Cartesian noise | 0.20 | 20/32 | 14/32 | 27 |
| 55 | global | mode push | 0.20 | 10/32 | 2/32 | 28 |
| 75 | global | Cartesian noise | 0.10 | 8/32 | 4/32 | 40 |
| 75 | global | mode push | 0.20 | 8/32 | 6/32 | 50 |
| 75 | icosahedral | mode push | 0.10 | 26/32 | 4/32 | 24 |
| 75 | icosahedral | mode push | 0.20 | 17/32 | 6/32 | 23 |

Intrinsic `lambda2` supplied 50/96 strict and 24/96 endpoint-valid rows;
CS² supplied 39/96 strict and 12/96 endpoint-valid rows. The 2,000-update
half of the grid gave 44/96 strict and 18/96 endpoint-valid results, versus
45/96 and 18/96 for the otherwise matched 5,000-update half. The longer
budget therefore added only one strict result and no two-minimum results.
Across the six starts, the best strict rescue cost only 23--50 evaluations,
so profile choice mattered far more than extending every trajectory to 5,000
updates. Terminal indices over all profiles were 58 index-zero, 101 index-one,
and 33 multi-negative. These robustness rates describe a deliberately hard,
failure-selected rescue set and must not be presented as population-level LJ
convergence rates.

The full cell-level paired table, near-flat and fragmentation counts, terminal
taxonomy, and exact aggregate are retained in `ANALYSIS.md`, `analysis.json`,
and `all_results.json` in the run directory.
