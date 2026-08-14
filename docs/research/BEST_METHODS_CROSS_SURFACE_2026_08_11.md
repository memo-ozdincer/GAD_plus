# GADplus results gallery: best methods and all-method comparisons

This is the single GitHub-renderable index for the LJ, g-xTB, and HIP method
decision. It puts selection, success rate, convergence cost, criterion
sensitivity, and the exact role of Sella in one place. Counts always retain
the full planned denominator; a dash means that quantity was not recorded,
not that it was zero.

## Selected implementations

These are intentionally one-update statements of the optimizer maps, not
search programs, command-line tools, calculator adapters, or frameworks:

| Surface | Selected method | Fixed production constants | Reference code |
|---|---|---|---|
| Lennard-Jones | intrinsic `lambda2` GAD | `tau=0.01`, `eta=0.05`, 500 updates | [`examples/best_methods/lj.py`](../../examples/best_methods/lj.py) |
| g-xTB | Competitive Soft-Spectral GAD (CS²-GAD) | `tau=0.01`, `eta=0.01`, 300 updates | [`examples/best_methods/gxtb.py`](../../examples/best_methods/gxtb.py) |
| HIP | plain one-mode GAD | `dt=0.007`, 5,000 updates | [`examples/best_methods/hip.py`](../../examples/best_methods/hip.py) |

Sella is deliberately absent from the implementation set. It remains in the
performance tables because it was an evaluated comparator.

There are no callbacks, convergence loops, flags, logging, I/O, or extension
hooks in these files. Each function takes the mathematical quantities at one
point and returns the next point. In the intrinsic files, if `V` maps mass-weighted vibrational
eigenmode coefficients back to Cartesian displacement and `g` is the
Cartesian gradient, then `c = V.T @ g` are the coefficients used below. The
HIP function takes the equivalent Eckart-clean Cartesian force and normalized
Cartesian image of the lowest mode, matching its historical benchmark map.
This isolates optimizer theory from HIP, g-xTB, LJ, unit, scheduler, and I/O
code.

The selections are not made by taking one pooled percentage blindly:

- **LJ:** CS² has four more pooled strict successes than intrinsic `lambda2`,
  but its sign changes with cluster size and it leaves 40 multi-negative
  terminals versus zero for intrinsic. Intrinsic is safer on LJ75 and was the
  stronger post-hoc rescue branch. It is therefore the maintained single LJ
  default; CS² is a complementary diversity/recovery channel.
- **g-xTB:** CS² has the best native two-branch endpoint recovery at both
  useful matched noise levels and directly addresses the observed
  high-index-to-minimum capture mechanism.
- **HIP:** plain GAD is the best completed non-Sella method. Held-out CS² is
  below plain GAD at all three hard noise levels in both strict recovery and
  intended IRC_TOPO, so CS² is not promoted on HIP.

## The three maps

Plain HIP GAD reflects only the current lowest-mode component of the
Eckart-clean Cartesian force. If `u1` is the lowest mass-weighted vibrational
mode mapped back to Cartesian coordinates and normalized there:

```text
F_GAD = F - 2 (F . u1) u1
x+ = x + 0.007 F_GAD
```

Intrinsic LJ GAD forms a soft lowest-mode density `p`, gates ascent by the
second eigenvalue, and uses a pointwise scale-covariant regularizer:

```text
s = rms(lambda)
p_i = softmax(-lambda_i / (tau s))
w = sigmoid(lambda_2 / (tau s))
b_i = (1 - 2 w p_i) c_i
R = eta * inverse_rms_pair_length(x) * sqrt(total_mass)
mu = ||b|| / R
a_i = -b_i / sqrt(lambda_i^2 + mu^2)
x+ = x + V a
```

CS² keeps the same bounded pointwise step, but gives every near-degenerate
lowest mode a relative reflection weight and competes lowest-mode activity
against activity in other negative directions:

```text
r_i = p_i / max(p)
n_i = sigmoid(-lambda_i / (tau s))
A = sum_i p_i c_i^2
B = sum_i n_i (1-p_i)^2 c_i^2
chi = A / (A+B), with chi=0 when A+B=0
w_cs2 = w + (1-w) chi
b_i = (1 - 2 w_cs2 r_i) c_i
```

No map contains a line search, trust history, rejected trial, mode history,
quasi-Newton state, or post-step displacement clip.

## Every evaluated decision-grid method on every surface

“Every method” here means every optimizer family/configuration in the final
matched decision grids used to choose the three implementations. Small
development panels, threshold-only reanalyses, and unmatched diagnostics are
reported separately below rather than mixed into denominators they did not
share.

This matrix uses the hardest directly informative completed cell for each
surface. The protocols necessarily differ, so compare methods *within a
column*, not percentages across columns. `NE` means the method was not
evaluated under that surface's matched decision protocol.

| Method | LJ13/31/38/55/75 pooled: strict / two minima (448 starts) | g-xTB 0.20 A: local index-1 / native endpoints (287 starts) | HIP 0.20 A: strict / IRC_TOPO (287 starts) |
|---|---:|---:|---:|
| Plain/regular GAD | NE | 2 / 2 | **128 / 128** |
| Intrinsic `lambda2` GAD | **395 / 200** | NE | NE |
| Competitive GAD | NE | 236 / 123 | NE |
| CS²-GAD | 399 / **207** | **262 / 124** | 111 / 117 |
| Sella comparator | **422 / 202** | 261 / 107 | 78 / 67 |

The Sella row does not alter the implementation choice: the requested
downstream methods are GAD-family methods only.

### Exactly where Sella is better

Sella is a strong comparator, but “Sella is better” is only true for local
convergence in particular regimes. The deltas below compare Sella to the
strongest GAD-family entry in the same matched cell. Positive means Sella is
better. The HIP “best Sella” row is explicitly an across-configuration oracle,
so it is favorable to Sella and is not a deployable single configuration.

| Comparison | Sella delta | Interpretation |
|---|---:|---|
| LJ pooled strict, Sella vs CS² | +23/448, **+5.1 pp** | Clear local-convergence advantage |
| LJ pooled two-minimum endpoints, Sella vs CS² | -5/448, **-1.1 pp** | Sella does not preserve the advantage after endpoint validation |
| g-xTB local at 0.10/0.20/0.50/1.00/2.00 A | -7/-1/-19/-18/-4 starts | CS² wins through 0.50 A; competitive GAD wins at 1–2 A |
| g-xTB native endpoints at 0.10/0.20/0.50 A | -21/-17/-5 starts | Sella loses every informative endpoint cell |
| HIP strict, best Sella in each cell vs plain GAD `dt=0.007` | +7.3/+6.6/+6.3/0.0/-4.2/-17.4 pp | Better only at 0.01–0.05 A, tied at 0.10 A, substantially worse at 0.15–0.20 A |
| HIP IRC_TOPO, fixed Sella Eckart A vs plain GAD `dt=0.005` | +0.3/0.0/-1.4/-5.5/-11.9/-21.3 pp | Essentially tied at low noise; GAD increasingly better as starts get harder |

Thus Sella is much more iteration-efficient when it succeeds, and its best
HIP configuration is 6–7 percentage points better near the labelled saddle.
It is not the best catch-all: GAD-family methods win the LJ endpoint measure,
all useful g-xTB endpoint cells, and hard-noise HIP local and IRC recovery.

### Lennard-Jones matched multisize benchmark

Strict means projected `n_neg=1` and reduced-unit `fmax<0.01`. The endpoint
count additionally requires two minimized downhill endpoints.

| Method | LJ13 strict | LJ31 strict | LJ38 strict | LJ55 strict | LJ75 strict | Pooled strict | Two minima |
|---|---:|---:|---:|---:|---:|---:|---:|
| Intrinsic `lambda2` | 56/64 | 63/64 | 115/128 | 49/64 | 112/128 | 395/448 (88.2%) | 200/448 |
| CS²-GAD | 59/64 | 60/64 | 122/128 | 58/64 | 100/128 | 399/448 (89.1%) | 207/448 |
| Sella comparator | 60/64 | 61/64 | 122/128 | 61/64 | 118/128 | 422/448 (94.2%) | 202/448 |

Running intrinsic and CS² as a two-channel GAD portfolio gives a frozen
paired strict union of 442/448 (98.7%). A later explicitly post-hoc rescue
grid finds at least one strict and one two-minimum profile for all six misses,
but its exploratory 448/448 union is not a replacement population estimate.

The earlier matched LJ7 formulation progression covers the methods that
preceded the multisize decision grid:

| LJ7 method | Strict TS | Two valid endpoints | Median strict evaluations |
|---|---:|---:|---:|
| Ordinary GAD | 115/288 (39.9%) | 115/288 | 494 |
| Sella comparator | 169/288 (58.7%) | 167/288 | **13** |
| Hard descent-to-GAD gate | 283/288 (98.3%) | 283/288 | 240 |
| Historical smooth `lambda2` gate | 283/288 (98.3%) | 283/288 | 240 |
| Pointwise intrinsic `lambda2` GAD | **288/288 (100%)** | **288/288** | 17 |

### g-xTB matched Transition1x benchmark

Each entry is `local index-1 / labelled native endpoint pair`, always over
all 287 planned test starts. Local index-1 uses `n_neg=1` and `fmax<0.03`
eV/A; the endpoint screen is not a full IRC.

| Method | 0.10 A | 0.20 A | 0.50 A | 1.00 A | 2.00 A |
|---|---:|---:|---:|---:|---:|
| Regular GAD | 35 / 33 | 2 / 2 | 23 / 3 | 3 / 0 | 1 / 0 |
| Competitive GAD | 264 / 222 | 236 / 123 | 204 / 5 | **105 / 0** | **8 / 0** |
| CS²-GAD | **282 / 231** | **262 / 124** | **221 / 5** | 90 / 0 | 3 / 0 |
| Sella comparator | 275 / 210 | 261 / 107 | 202 / 0 | 87 / 0 | 4 / 0 |

From 1.00 A onward, calculator failures dominate and no method recovers a
labelled endpoint pair. Those cells do not overturn the useful-regime CS²
selection.

### HIP held-out Transition1x benchmark

Strict means projected `n_neg=1` and `fmax<0.01` eV/A. Plain-GAD and Sella
values below are the completed principal grids; CS² uses exact held-out HDF5
`test` IDs 0..286 and parameters frozen before looking at HIP test outcomes.

| Method | 0.01 A | 0.03 A | 0.05 A | 0.10 A | 0.15 A | 0.20 A |
|---|---:|---:|---:|---:|---:|---:|
| Plain GAD, `dt=0.003` | 89.2% | 88.5% | 85.4% | 71.1% | 55.1% | 40.8% |
| Plain GAD, `dt=0.005` | 89.2% | 88.5% | **85.7%** | 71.8% | 57.1% | 43.2% |
| **Plain GAD, `dt=0.007`** | **89.2%** | **88.9%** | **85.7%** | **72.8%** | **58.2%** | **44.6%** |
| CS²-GAD, held-out | NE | NE | NE | 202/287 (70.4%) | 157/287 (54.7%) | 111/287 (38.7%) |
| Sella Cartesian | 92.0% | 91.3% | 87.5% | 65.5% | 42.9% | 18.8% |
| Sella Cartesian+Eckart A | 92.7% | 92.0% | 88.2% | 70.7% | 54.0% | 27.2% |
| Sella internal | 79.1% | 77.4% | 71.8% | 50.9% | 26.8% | 13.9% |
| Sella Cartesian+Eckart B | **96.5%** | **95.5%** | **92.0%** | **72.8%** | 50.5% | 23.3% |
| Hybrid damped Eckart | 85.4% | 85.0% | 81.5% | 66.9% | 50.9% | 33.1% |
| Hybrid undamped Eckart | 84.7% | 84.3% | 81.5% | 65.5% | 49.8% | 31.0% |

For the selected plain GAD at 0.10/0.15/0.20 A, completed intended
IRC_TOPO is 225/177/128. The matched Sella comparator gives 208/143/67 and
held-out CS² gives 218/166/117. Thus CS² trails plain GAD by 7/11/11 intended
mechanisms, although it exceeds Sella by 10/23/50. Across the full
0.01/0.03/0.05/0.10/0.15/0.20 A IRC grid,
plain GAD `dt=0.005` gives 88.9/89.2/88.9/78.0/61.7/44.6%, Sella
Cartesian+Eckart A gives 89.2/89.2/87.5/72.5/49.8/23.3%, and the damped
Eckart hybrid gives 89.2/88.9/88.9/76.7/57.5/38.7%.

The separate 80-start Newton-polish diagnostic at 0.05 A is not part of the
287-start decision grid: loose and strict NR-GAD reached 46.2% and 45.0% at
`fmax<0.01`, versus 85.7% for plain GAD. The strict variant reached 31.2% at
`fmax<0.005`, showing a force-floor tradeoff rather than a better HIP search
method.

## Steps to convergence

These tables report the median only among runs that passed the stated local
gate. A low median is not useful without the success count above: for example,
g-xTB Sella at 2.00 A takes few steps on its four successes but misses 283 of
287 starts. Also, one Sella iteration, one GAD update, and one LJ evaluation
are not equal-cost operations. Sella may evaluate gradients/Hessians inside
an iteration; the tables are optimizer progress counts, not a universal
calculator-cost measure.

### LJ: median evaluations among strict successes

| Method | LJ13 | LJ31 | LJ38 | LJ55 | LJ75 |
|---|---:|---:|---:|---:|---:|
| Intrinsic `lambda2` | 14 | 20 | 40 | 19 | 27.5 |
| CS²-GAD | 31 | 32 | 35 | 31 | 30 |
| Sella comparator | **11.5** | 21 | **24** | **19** | **23** |

Sella is fastest on four cells including ties; intrinsic is faster on LJ31
and is dramatically faster on LJ13. Success coverage still has to be read
from the preceding table.

### g-xTB: median steps among local `fmax<0.03` successes

| Method | 0.10 A | 0.20 A | 0.50 A | 1.00 A | 2.00 A |
|---|---:|---:|---:|---:|---:|
| Regular GAD | 274 | 282.5 | 1,015 | 1,605 | 1,889 |
| Competitive GAD | 14 | **28** | **76** | 108 | 215.5 |
| CS²-GAD | 14 | 29.5 | **76** | 102 | 167 |
| Sella comparator | **6** | **15** | **32** | **45** | **57.5** |

The regular-GAD values were recovered from its preserved per-task Parquet
summaries; competitive/CS² values come from frozen aggregate rows and Sella
values from the original task logs. All success denominators match the main
g-xTB table.

### HIP: median steps among strict successes

| Method | 0.01 A | 0.03 A | 0.05 A | 0.10 A | 0.15 A | 0.20 A |
|---|---:|---:|---:|---:|---:|---:|
| Plain GAD `dt=0.003` | 165.5 | 337.5 | 458 | 757.5 | 993 | 1,196 |
| Plain GAD `dt=0.005` | 99.5 | 203.5 | 278 | 458 | 613.5 | 738 |
| Plain GAD `dt=0.007` | 73 | 145 | 200 | 332 | 437 | 546 |
| CS²-GAD held-out | — | — | — | 17 evals | 26 evals | 5,001 evals |
| Sella Cartesian | 4 | 6 | 7 | 10.5 | 16 | 20 |
| Sella Cartesian+Eckart A | **4** | **6** | **7** | **9** | **11** | **13** |
| Sella internal | 4 | 6 | 9 | 17 | — | — |
| Sella Cartesian+Eckart B | 5 | 8 | 9 | 11 | 14 | — |
| Hybrid damped Eckart | 6 | 12 | 19 | 36.5 | 61 | 95 |
| Hybrid undamped Eckart | 6 | 14 | 21 | 39.5 | 65 | 95 |

The historical HIP grid records steps; the newer CS² runner records energy/
gradient evaluations, hence the explicit `evals` label. Missing Sella medians
are left blank rather than reconstructed from partial runs. Sella needs far
fewer optimizer iterations, but at 0.20 A its best fixed configuration reaches
only 27.2% strict versus 44.6% for plain GAD `dt=0.007`.

## Convergence-criterion sensitivity

### What each gate means

| Gate | Shared local requirement | Additional requirement | What it answers |
|---|---|---|---|
| LJ strict | projected `n_neg=1`, reduced `fmax<0.01` | none | Did the search end at a local first-order saddle? |
| LJ two minima | LJ strict | both downhill minimizations valid | Does the saddle have two usable downhill basins? |
| g-xTB local | projected `n_neg=1`, `fmax<0.03` eV/A | none | Did the search reach the operational local gate? |
| g-xTB strict | projected `n_neg=1`, `fmax<0.01` eV/A | none | Does it survive the tighter force gate? |
| g-xTB native endpoints | g-xTB local | endpoint pair matches labelled native topology | Did local recovery retain the labelled mechanism proxy? |
| HIP strict | projected `n_neg=1`, `fmax<0.01` eV/A | none | Did the optimizer recover a strict local TS? |
| HIP IRC_TOPO | validated TS candidate | full Hessian IRC and endpoint topology | Did it recover the intended reaction mechanism? |

Local convergence is never presented as mechanism recovery. In particular,
the g-xTB endpoint screen is a labelled endpoint proxy, whereas HIP IRC_TOPO
is the full downstream IRC gate.

### g-xTB at 0.20 A under three criteria (of 287)

| Method | Local, `fmax<0.03` | Strict, `fmax<0.01` | Native endpoints |
|---|---:|---:|---:|
| Regular GAD | 2 | 0 | 2 |
| Competitive GAD | 236 | 9 | 123 |
| CS²-GAD | **262** | 10 | **124** |
| Sella comparator | 261 | **104** | 107 |

Sella's apparent near-tie at the operational local gate becomes a large
strict-force advantage, but that advantage does not transfer to native
endpoint recovery. CS² retains 17 more native endpoint pairs.

### HIP at hard 0.20 A under force thresholds (percent of 287)

Every entry still requires projected `n_neg=1`; only the force threshold
changes. `0.023` is included because it was the historical intermediate HIP
diagnostic.

| Method | `<0.050` | `<0.023` | `<0.010` | `<0.005` | `<0.001` |
|---|---:|---:|---:|---:|---:|
| Plain GAD `dt=0.003` | 62.7 | 52.3 | 40.8 | 0.0 | 0.0 |
| Plain GAD `dt=0.005` | 63.1 | 53.7 | 43.2 | 0.0 | 0.0 |
| Plain GAD `dt=0.007` | **68.3** | **56.4** | **44.6** | 0.0 | 0.0 |
| Hybrid damped Eckart | 44.9 | 37.6 | 33.1 | 6.6 | 0.0 |
| Hybrid undamped Eckart | 43.9 | 37.3 | 31.0 | 1.4 | 0.0 |
| Sella Cartesian+Eckart A | **46.0** | **39.0** | **27.2** | **7.0** | 0.0 |
| Sella Cartesian+Eckart B | 39.7 | 34.5 | 23.3 | **7.0** | 0.0 |
| Sella Cartesian | 27.9 | 24.7 | 18.8 | 5.6 | 0.0 |
| Sella internal | 20.9 | 18.1 | 13.9 | 4.2 | 0.0 |

Bold within the GAD and comparator/refinement groups avoids implying that a
post-GAD hybrid or Sella configuration is a candidate for the maintained
plain-GAD implementation. The zeroes at the tightest threshold expose the
plain-GAD force floor; they do not negate its better hard-start recovery at
the registered `0.01` criterion.

### LJ criterion ladder on the multisize pool (of 448)

| Method | Strict local TS | Two valid minima | Retention from strict to endpoints |
|---|---:|---:|---:|
| Intrinsic `lambda2` | 395 | 200 | 50.6% |
| CS²-GAD | 399 | **207** | **51.9%** |
| Sella comparator | **422** | 202 | 47.9% |

This is why “Sella wins LJ” needs qualification: it wins the local strict
gate, but CS² supplies more fully validated endpoint pairs and retains a
larger fraction of its strict successes.

## Evidence

GitHub-resident evidence and detailed ledgers:

- [LJ multisize/CS² ledger](LJ_MULTISIZE_CS2_BENCHMARK_2026_08_09.md)
- [cross-benchmark ledger](BENCHMARK_RESULTS_2026_07_16.md)
- [HIP CS² held-out H100 ledger](HIP_CS2_H100_HANDOFF_2026_08_09.md)
- [HIP threshold-sweep source table](analysis_2026_04_29/threshold_sweep_2026_05_16.csv)
- [HIP convergence-step/IRC source table](analysis_2026_04_29/master_2026_05_16.csv)

Immutable scratch artifacts used to regenerate and audit the displayed
tables:

- LJ final run: `/scratch/memoozd/gadplus/runs/lj-multisize-cs2-2076554/`
- LJ post-hoc rescue: `/scratch/memoozd/gadplus/runs/lj-gad-rescue-2077217/`
- g-xTB aggregate: `/scratch/memoozd/gadplus/analysis/t1x-gxtb-matched-noise-grid/`
- HIP CS²: `/scratch/memoozd/gadplus/runs/hip-cs2-h100-production-20260809-v2/`
