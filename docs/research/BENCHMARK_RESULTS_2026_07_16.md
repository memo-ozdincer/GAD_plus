# GAD vs Sella Results Ledger

This is the compact cross-benchmark comparison requested on 2026-07-16.
It uses the strongest completed configuration recorded for each method and
cell, but does not hide whether that configuration is pure GAD, a GAD
extension, or a post-hoc per-cell best choice.

## Common Criterion

Unless stated otherwise, **strict saddle convergence** means the
Eckart-projected vibrational Hessian has `n_neg = 1` and `fmax < 0.01` at
the candidate. This is a transition-state-quality measure only; IRC/topology
is listed separately where it was evaluated.

Reported "best" values are a summary of completed parameter grids, not a
newly tuned configuration for the winning method. In particular, the HIP
table selects the best documented *plain* GAD timestep and the best completed
Sella Hessian-update setting at each noise value.

Important Sella provenance: the benchmark path supplies the current full HIP
Hessian at every optimization step. The Sella results should be interpreted as
full-Hessian RS-P-RFO, not as a quasi-Newton baseline.

## Summary

| Surface | Strict-saddle result | What it supports |
|---|---|---|
| HIP / Transition1x | Sella leads at low noise; plain GAD ties around 100 pm and leads at 150--200 pm. | The original high-noise GAD advantage, on HIP, is real. |
| SCINE DFTB0 | GAD has a small 10 pm lead; Sella wins at every higher noise value. | Not a robustness win for GAD. |
| LJ7 | Sella strongly beats **pure** GAD at high noise. A high-index gate makes GAD nearly perfect, but is not pure GAD and has no endpoint validation yet. | Pure GAD has a stiff/high-index globalization failure; the gated extension is promising. |
| LJ13/31/38/55/75 | CS²-GAD is mixed against intrinsic `lambda2` (399/448 vs 395/448 strict); their paired union is 442/448. Sella's 422/448 is comparator-only. | CS² is not a universal LJ catchall alone, but it is a strong complementary GAD channel; intrinsic is safer on LJ75. |
| HORM LEFTNet | Stabilized GAD 0/4; Sella 1/4. | Negative independent-MLIP screen. |
| MACE-OFF23 | Exploratory probes only; no completed matched pool. | Do not use as comparative evidence. |

## HIP / Transition1x

Starting points are noised T1x test-split transition states, `n = 287`.
Values are strict saddle-convergence percentages. The noise labels are the
repository's historical labels; see the unit note below.

| Noise label | Best plain GAD | Best Sella | Delta, GAD - Sella |
|---:|---:|---:|---:|
| 10 pm | 89.2% (`dt=0.003/0.005/0.007`) | 96.5% (Cartesian Eckart, Hess.Freq.=3) | -7.3 pp |
| 30 pm | 88.9% (`dt=0.007`) | 95.5% (Cartesian Eckart, Hess.Freq.=3) | -6.6 pp |
| 50 pm | 85.7% (`dt=0.005/0.007`) | 92.0% (Cartesian Eckart, Hess.Freq.=3) | -6.3 pp |
| 100 pm | 72.8% (`dt=0.007`) | 72.8% (Cartesian Eckart, Hess.Freq.=3) | 0.0 pp |
| 150 pm | 58.2% (`dt=0.007`) | 54.0% (Cartesian Eckart, Hess.Freq.=1) | +4.2 pp |
| 200 pm | 44.6% (`dt=0.007`) | 27.2% (Cartesian Eckart, Hess.Freq.=1) | +17.4 pp |

For intended-saddle validation, the best completed plain-GAD IRC topology
rates at 100/150/200 are 78.4%/61.7%/44.6%, versus Sella's
72.5%/49.8%/23.3% for the matched Cartesian-Eckart baseline. This is the
strongest positive comparison currently in the repository.

The frozen CS²-GAD follow-up uses the held-out HDF5 `test` partition, exact
filtered IDs `0..286`, rather than the checkpoint's recorded training LMDB
`ts1x_hess_train_big.lmdb` or validation LMDB `ts1x-val.lmdb`. Its optimizer
parameters were inherited from g-xTB and were not selected on these HIP test
outcomes. Completed strict counts are `202/287`, `157/287`, and `111/287` at
`0.10/0.15/0.20 A`, below plain GAD's `209/287`, `167/287`, and `128/287`.
Intended all-endpoint IRC_TOPO is `218/287`, `166/287`, and `117/287`, versus
plain GAD's `225/287`, `177/287`, and `128/287`. The completed held-out
evidence therefore rejects CS² as HIP's catchall. Operational details,
start-table hashes, and retained calculator failures are recorded in
`HIP_CS2_H100_HANDOFF_2026_08_09.md`.

Source: `analysis_2026_04_29/master_2026_05_16.csv`.

## SCINE DFTB0

Starting points are noised T1x test-split transition states, `n = 287`.
GAD uses the completed long-budget DFTB0 recipe; Sella is saturated by its
standard budget, so the comparison is not explained by Sella step starvation.

| Noise | GAD strict | Sella strict | Delta, GAD - Sella |
|---:|---:|---:|---:|
| 10 pm | 89.9% (258/287) | 87.8% (252/287) | +2.1 pp |
| 30 pm | 66.6% | 73.5% | -6.9 pp |
| 50 pm | 43.9% | 56.1% | -12.2 pp |
| 100 pm | 19.5% | 32.4% | -12.9 pp |
| 150 pm | 6.3% | 15.3% | -9.0 pp |
| 200 pm | 1.4% | 3.8% | -2.4 pp |

The 10 pm strict-convergence lead is real but small. It should not be
converted into an endpoint-recovery claim: under common DFTB0-native endpoint
labels, GAD/Sella topology correctness is 42/287 (14.63%) vs 55/287 (19.16%)
at 10 pm, while strict geometry correctness is tied at 21/287.

Source: `SCINE_XTB_FINDINGS_2026_05_15.md`, including the corrected
DFTB0-native validation.

## LJ7

This is an LJ7 reduced-unit, hydrogen-mass benchmark (`n = 287`), not a T1x
chemistry/IRC benchmark. Its strict metric is `n_neg = 1` and `fmax < 0.01`.
The corrected force-balanced D5h reference geometry is used in the later
results below.

| Noise (fraction of sigma) | Best pure GAD | Best Sella | GAD smooth high-index gate |
|---:|---:|---:|---:|
| 0.10 | 69.7% | 95.5% | 100.0% |
| 0.15 | 51.2% | 83.6% | 100.0% |
| 0.20 | 36.2% | 74.9% | 99.7% |

The GAD gate is `w = sigmoid(k * lambda_2)` with the GAD ascent suppressed
while the second vibrational mode remains negative. It is a well-motivated
globalization extension, but it is explicitly **not pure GAD**. LJ has no
completed intended-saddle/IRC classifier, so these numbers cannot yet be used
as chemistry-correct success rates.

Sources: `LJ_FINDINGS_2026_07_09.md` and the completed LJ batch aggregates
under `/lustre07/scratch/memoozd/gadplus/runs/`.

### LJ Implementation and Retuning Record

The LJ evidence was smoke-tested before interpreting optimizer results.  The
local predictor returns a full `3N x 3N` Hessian (`21 x 21` for LJ7), whose
finite-difference check had maximum/RMS errors of about `6.3e-4`/`6.6e-5`.
The batched-GAD and Sella/ASE routes agreed on the same potential: energy
exactly, forces to `4.7e-10` or better, raw Hessians to `3.0e-8`, and the
checked Eckart vibrational spectrum to `9e-8`.  Therefore the LJ difference
is not currently attributable to the methods seeing different energies,
forces, or Hessians.

The original LJ7 pentagonal-bipyramid helper used pair-equilibrium distances,
which was close but not force balanced (`fmax = 1.95618`).  It was replaced by
the relaxed D5h geometry (`fmax = 1.06e-7`, `n_neg = 0`).  This fixes the
zero/low-noise reference but did not change the 0.15-noise pure-GAD rate
(51.2%), so it is not the high-noise explanation.

| Change tested | Observed effect at high noise | Conclusion |
|---|---|---|
| Hydrogen mass | Best tested mass weighting; carbon-like and argon-like masses were worse (for example, at 0.20 noise with a 0.005 cap: roughly 38% H, 28% C, 10% Ar). | Keep hydrogen mass for this reduced-unit benchmark. |
| Fixed displacement cap | Original pure-GAD rates near 0.15/0.20 were about 41.5%/29.3%; the best hydrogen fixed-cap retune reached 51.2%/38.3%. | Modest gain; it limits oversteps but does not globalize GAD. |
| Ramped cap | About 50.5%/39.7% at 0.15/0.20. | Comparable to a fixed cap, with no decisive improvement. |
| Smaller fixed cap | A 0.001 cap held motion closer to the reference in replay, but slowed force/index relaxation; the 0.20 rate was about 34.8%. | A global tiny cap is not a solution. |
| `dt` sweep, 0.002--0.007 | With the cap active, rates changed little: about 49.8% at 0.15 and 36.6--38.0% at 0.20. | Lowering `dt` alone cannot recover the failures. |
| Descent while `n_neg > 1`, then GAD | 99.0%/97.2%/96.5% at noise 0.10/0.15/0.20. | Identifies the high-index region as the mechanism, but is not pure GAD. |
| Smooth `lambda_2` gate | 100.0%/100.0%/99.7% at 0.10/0.15/0.20; 0.20 median strict convergence in 257--258 steps. | Strongest LJ recovery, but an explicit GAD globalization extension. |

The trace/replay diagnostics explain why the cap/dt-only fixes plateau.
At 0.20 noise, starts have median initial `n_neg = 8`, median `fmax =
2.045e3`, and a severe close-pair tail from the LJ repulsive wall.  Successful
fixed-cap traces had median 217 cap hits, failures 1021; 24 failures remained
capped for all 8000 steps, while no success did.  Single-mode GAD is locally
appropriate once there is one negative mode, but it is unstable in the other
negative modes of these high-index entry regions.  Sella's trust-region
behavior is consequently a better globalization mechanism on the pure-LJ
test.  The smooth gate supplies the missing index-aware descent phase, but it
also changes the saddle distribution: at 0.20, only 33/104 pure-GAD successes
ended at the same energy (within `1e-4`) as their gated counterparts.  An
LJ-specific endpoint/IRC-like classifier is required before treating gated
LJ success as intended-saddle recovery.

### Hard multi-size LJ validation (2026-08-09)

A later frozen benchmark tested intrinsic `lambda2`, CS²-GAD, and Sella on
LJ13, LJ31, the two LJ38 funnels, LJ55, and the two LJ75 funnels. It used
1,344 trajectories with exact matched starts, and all records were
calculator-valid. Strict successes pooled over the 448 unique starts per
method were 395 intrinsic, 399 CS², and 422 Sella.

The pooled near-tie between the two GAD profiles conceals strong size
dependence. CS²-minus-intrinsic matched strict counts were `+3`, `-3`, `+7`,
`+9`, and `-12` for LJ13, LJ31, LJ38, LJ55, and LJ75. CS² ended at a
multi-negative structure 40 times, including 23/128 LJ75 starts; intrinsic
had no multi-negative terminal cases. Conversely, CS² recovered more
fingerprint families than intrinsic at every size. The operational conclusion
is therefore to retain intrinsic `lambda2` as the LJ GAD default and add CS²
as a complementary LJ38/LJ55 or diversity channel. Sella is only a comparator
for this downstream use case, not an optimizer option.

The GAD-only paired union reaches 442/448 strict outcomes: 64/64 LJ13, 64/64
LJ31, 127/128 LJ38, 63/64 LJ55, and 124/128 LJ75. The six misses all exhaust
the 500-update budget, providing a fixed target for a separately documented
continuation/restart experiment toward 100% rather than a reason to retune
the completed benchmark post hoc.

That fixed post-hoc rescue experiment subsequently completed 192/192 valid
profiles with zero errors. All six misses were rescued by at least one strict
GAD-family profile, giving an explicitly exploratory 448/448 union, and all
six had at least one two-minimum profile. The sweep produced 89/192 strict and
36/192 endpoint-valid rows; intrinsic `lambda2` contributed 50/96 strict and
24/96 endpoints versus CS²'s 39/96 and 12/96. Increasing the matched budget
from 2,000 to 5,000 updates added only one strict result and no endpoints.
This failure-selected sweep supports a profile-diverse rescue strategy but
does not revise the frozen 442/448 population estimate.

Protocol, per-size strict and endpoint results, paired tests, cost tails, and
limitations are in `LJ_MULTISIZE_CS2_BENCHMARK_2026_08_09.md`; immutable run
artifacts are under
`/scratch/memoozd/gadplus/runs/lj-multisize-cs2-2076554/`.

## HORM LEFTNet

HORM is an independent energy-conservative, Hessian-supervised reactive MLIP.
The calculator path was smoke-tested with a full autograd Hessian and a
directional finite-difference check before optimization. Both optimizers use
the same energy/force/Hessian adapter.

The screen used four predeclared zero-noise formula-block representatives
(`sample_id = 0, 5, 11, 15`), 500 steps, and the strict criterion above.

| Method | Strict saddles | Per-sample outcome |
|---|---:|---|
| GAD with descent until `n_neg <= 1` | 0/4 | 0: `n_neg=5`, 5: `n_neg=3`, 11: `n_neg=2`, 15: `n_neg=2` |
| Sella Cartesian+Eckart | 1/4 | 0 strict; 5 and 15 end at `n_neg=2`; 11 has `n_neg=1` but `fmax=0.0547` |

Pure GAD, the `lambda_2` smooth gate, and the hard descent gate also all
failed on the detailed sample-2 trace. This is a negative screen, not a
claim about all HORM-native saddles. It nevertheless does not provide the
needed independent positive result for a broad GAD-over-Sella claim.

Source: `/lustre07/scratch/memoozd/gadplus/runs/horm_formula_starts_zero_20260711/`.

## Transition1x / g-xTB matched noise grid

This is a separate, current 287-start test-split campaign on the g-xTB PES.
Each method received the same Cartesian-noise realization for each labelled
Transition1x TS.  `local index 1` means projected \(n_{\rm neg}=1\) and
\(f_{\max}<0.03\) eV A\(^{-1}\). `native topology` additionally requires
two downhill, minimized branches to match the labelled Transition1x endpoint
pair. It is an inexpensive two-branch endpoint screen, **not a full IRC**.

| noise (A) | method | updates | valid / 287 | local index-1 / 287 | native topology / 287 | topology / local |
|---:|---|---:|---:|---:|---:|---:|
| 0.10 | regular GAD | 300 | 287 | 35 (12.2%) | 33 (11.5%) | 94.3% |
| 0.10 | competitive GAD | 300 | 287 | 264 (92.0%) | 222 (77.4%) | 84.1% |
| 0.10 | **competitive-subspace GAD** | 300 | 286 | **282 (98.3%)** | **231 (80.5%)** | 81.9% |
| 0.10 | Sella | 300 | 286 | 275 (95.8%) | 210 (73.2%) | 76.4% |
| 0.20 | regular GAD | 300 | 287 | 2 (0.7%) | 2 (0.7%) | 100.0% |
| 0.20 | competitive GAD | 300 | 281 | 236 (82.2%) | 123 (42.9%) | 52.1% |
| 0.20 | **competitive-subspace GAD** | 300 | 276 | **262 (91.3%)** | **124 (43.2%)** | 47.3% |
| 0.20 | Sella | 300 | 276 | 261 (90.9%) | 107 (37.3%) | 41.0% |
| 1.00 | regular GAD | 2000 | 95 | 3 (1.0%) | 0 (0.0%) | 0.0% |
| 1.00 | competitive GAD | 2000 | 117 | 105 (36.6%) | 0 (0.0%) | 0.0% |
| 1.00 | competitive-subspace GAD | 2000 | 94 | 90 (31.4%) | 0 (0.0%) | 0.0% |
| 1.00 | Sella | 2000 | 101 | 87 (30.3%) | 0 (0.0%) | 0.0% |

### Interpretation

At 0.10 and 0.20 A, competitive-subspace GAD has the highest observed
labelled endpoint recovery: 80.5% and 43.2%, respectively. Its advantage over
the rank-one competitive reflection is small at 0.20 A (124 versus 123
starts), so this is a selection based on the complete paired evidence and the
minimum-capture mechanism, not a broad claim of a large effect. It is clearly
stronger than regular one-mode GAD on these starts, and it exceeds Sella on
the labelled two-branch screen at both moderate noise levels.

At 1.00 A, g-xTB calculator failures dominate (170--193 failures depending on
method), and no method recovers the labelled endpoint pair. This does **not**
mean no local saddles exist: competitive GAD, competitive-subspace GAD, and
Sella still reached 105, 90, and 87 local index-one candidates, respectively.
It means that the labelled-event recovery metric is outside its useful regime
at this perturbation under the present PES and endpoint labels.

The selected g-xTB method is therefore competitive-subspace GAD for the
state-based local search. Accepted candidates still require the normal
physical filters: force/index, a clear spectral gap where relevant,
fragmentation screening, and—when chemical connectivity matters—a proper IRC
or equivalently justified endpoint validation. The raw manifests and
reproducible aggregate are
`experiments/t1x_gxtb_matched_noise_grid_manifest.json` and
`/scratch/memoozd/gadplus/analysis/t1x-gxtb-matched-noise-grid/`.

## MACE-OFF23

MACE work is exploratory only. The compatibility screen found that all
labeled reactant/product geometries relaxed on the MACE PES, but only 10/20
labeled transition-state geometries were index-one. Small optimizer probes
favored Sella under the initial settings; a single retuned GAD case converged.
There is no completed, common-pool, matched Sella/GAD result, so MACE is
excluded from the quantitative cross-benchmark comparison.

## Interpretation

The evidence supports a conditional statement, not a universal one:

> On HIP's T1x-trained Hessian landscape, plain GAD is more robust than the
> best completed Sella baseline at high noise. On DFTB0, LJ7 (for pure GAD),
> and the current HORM screen, it is not.

The recurring discriminator is entry into a high-index region. Single-mode
GAD is locally appropriate near an index-one saddle, but it has no general
globalization mechanism when several Hessian modes are negative. Sella's
trust-region steps handle those starts better. LJ demonstrates that an
explicit index-aware descent/gate can fix this failure mechanism, but that
extension requires its own intended-saddle validation before becoming a paper
claim.

## Reporting Cautions

- The direct HIP/LJ scripts historically label `noise * 1000` as pm even
  though the repository conversion is 1 Angstrom = 100 pm. Those labels need
  correction before publication; LJ should be reported in reduced `sigma`
  units.
- Do not compare LJ force magnitudes directly to HIP or DFTB0: LJ force units
  are reduced `epsilon / sigma` units.
- Do not report a method as converged based on Sella's own status alone. The
  strict tables above recompute projected `n_neg` and `fmax` independently.
