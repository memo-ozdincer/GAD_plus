# Every method on every surface

| Symbol | Meaning |
|---|---|
| **bold** | Best result within the same surface, cell, and criterion |
| — | Not evaluated or not recorded under that matched protocol |
| LJ local | Projected `n_neg=1`, reduced `fmax<0.01` |
| g-xTB local / strict | Projected `n_neg=1`, `fmax<0.03 / 0.01` eV/A |
| HIP strict / tight | Projected `n_neg=1`, `fmax<0.01 / 0.005` eV/A |
| Downstream | LJ two-minimum gate / g-xTB native endpoints / HIP IRC_TOPO |

## Tall hard-cell comparison

| Surface and hard cell | Method | Local | Tighter local | Downstream | Median successful steps/evaluations |
|---|---|---:|---:|---:|---:|
| LJ13/31/38/55/75 pool, 448 starts | Intrinsic `lambda2` GAD | 395 (88.2%) | — | 200 (44.6%) | 25 evals |
| LJ13/31/38/55/75 pool, 448 starts | CS²-GAD | 399 (89.1%) | — | **207 (46.2%)** | 32 evals |
| LJ13/31/38/55/75 pool, 448 starts | Sella | **422 (94.2%)** | — | 202 (45.1%) | **20.5 evals** |
| g-xTB 0.20 A, 287 starts | Regular GAD | 2 (0.7%) | 0 | 2 (0.7%) | 282.5 steps |
| g-xTB 0.20 A, 287 starts | Competitive GAD | 236 (82.2%) | 9 (3.1%) | 123 (42.9%) | 28 steps |
| g-xTB 0.20 A, 287 starts | CS²-GAD | **262 (91.3%)** | 10 (3.5%) | **124 (43.2%)** | 29.5 steps |
| g-xTB 0.20 A, 287 starts | Sella | 261 (90.9%) | **104 (36.2%)** | 107 (37.3%) | **15 steps** |
| HIP 0.20 A, 287 starts | Plain GAD `dt=0.003` | 40.8% | 0.0% | **44.6%** | 1,196 steps |
| HIP 0.20 A, 287 starts | Plain GAD `dt=0.005` | 43.2% | 0.0% | **44.6%** | 738 steps |
| HIP 0.20 A, 287 starts | Plain GAD `dt=0.007` | **44.6%** | 0.0% | 43.9% | 546 steps |
| HIP 0.20 A, 287 starts | CS²-GAD, held-out | 38.7% | 2.8% | 40.8% | 5,001 evals |
| HIP 0.20 A, 287 starts | Sella Cartesian | 18.8% | 5.6% | 17.8% | 20 steps |
| HIP 0.20 A, 287 starts | Sella Cartesian+Eckart A | 27.2% | **7.0%** | 23.3% | **13 steps** |
| HIP 0.20 A, 287 starts | Sella Cartesian+Eckart B | 23.3% | **7.0%** | 22.0% | — |
| HIP 0.20 A, 287 starts | Sella internal | 13.9% | 4.2% | 16.0% | — |
| HIP 0.20 A, 287 starts | Hybrid damped Eckart | 33.1% | 6.6% | 38.7% | 95 steps |
| HIP 0.20 A, 287 starts | Hybrid undamped Eckart | 31.0% | 1.4% | 38.7% | 95 steps |

## Lennard-Jones multisize: strict / two minima

| Method | LJ13 | LJ31 | LJ38 | LJ55 | LJ75 | Pooled strict | Pooled two minima |
|---|---:|---:|---:|---:|---:|---:|---:|
| Intrinsic `lambda2` GAD | 56 / 52 | **63** / 30 | 115 / **71** | 49 / 14 | 112 / **33** | 395 | 200 |
| CS²-GAD | 59 / **53** | 60 / 35 | **122** / 70 | 58 / **19** | 100 / 30 | 399 | **207** |
| Sella | **60** / **53** | 61 / **37** | **122** / 64 | **61** / 15 | **118** / **33** | **422** | 202 |

## Lennard-Jones LJ7 formulation progression

| Method | Strict TS | Two valid endpoints | Median strict evaluations |
|---|---:|---:|---:|
| Ordinary GAD | 115/288 (39.9%) | 115/288 | 494 |
| Hard descent-to-GAD gate | 283/288 (98.3%) | 283/288 | 240 |
| Historical smooth `lambda2` gate | 283/288 (98.3%) | 283/288 | 240 |
| Pointwise intrinsic `lambda2` GAD | **288/288 (100%)** | **288/288** | **17** |

## g-xTB: local / native endpoints, each of 287

| Method | 0.10 A | 0.20 A | 0.50 A | 1.00 A | 2.00 A |
|---|---:|---:|---:|---:|---:|
| Regular GAD | 35 / 33 | 2 / 2 | 23 / 3 | 3 / **0** | 1 / **0** |
| Competitive GAD | 264 / 222 | 236 / 123 | 204 / **5** | **105** / **0** | **8** / **0** |
| CS²-GAD | **282** / **231** | **262** / **124** | **221** / **5** | 90 / **0** | 3 / **0** |
| Sella | 275 / 210 | 261 / 107 | 202 / 0 | 87 / **0** | 4 / **0** |

## g-xTB 0.20 A convergence criteria, each of 287

| Method | Local `fmax<0.03` | Strict `fmax<0.01` | Native endpoints | Median local steps |
|---|---:|---:|---:|---:|
| Regular GAD | 2 | 0 | 2 | 282.5 |
| Competitive GAD | 236 | 9 | 123 | 28 |
| CS²-GAD | **262** | 10 | **124** | 29.5 |
| Sella | 261 | **104** | 107 | **15** |

## HIP strict recovery, each of 287

| Method | 0.01 A | 0.03 A | 0.05 A | 0.10 A | 0.15 A | 0.20 A |
|---|---:|---:|---:|---:|---:|---:|
| Plain GAD `dt=0.003` | 89.2% | 88.5% | 85.4% | 71.1% | 55.1% | 40.8% |
| Plain GAD `dt=0.005` | 89.2% | 88.5% | 85.7% | 71.8% | 57.1% | 43.2% |
| Plain GAD `dt=0.007` | 89.2% | 88.9% | 85.7% | **72.8%** | **58.2%** | **44.6%** |
| CS²-GAD, held-out | — | — | — | 70.4% | 54.7% | 38.7% |
| Sella Cartesian | 92.0% | 91.3% | 87.5% | 65.5% | 42.9% | 18.8% |
| Sella Cartesian+Eckart A | 92.7% | 92.0% | 88.2% | 70.7% | 54.0% | 27.2% |
| Sella Cartesian+Eckart B | **96.5%** | **95.5%** | **92.0%** | **72.8%** | 50.5% | 23.3% |
| Sella internal | 79.1% | 77.4% | 71.8% | 50.9% | 26.8% | 13.9% |
| Hybrid damped Eckart | 85.4% | 85.0% | 81.5% | 66.9% | 50.9% | 33.1% |
| Hybrid undamped Eckart | 84.7% | 84.3% | 81.5% | 65.5% | 49.8% | 31.0% |

## HIP 0.20 A force-threshold sensitivity, percent of 287

| Method | `fmax<0.050` | `fmax<0.023` | `fmax<0.010` | `fmax<0.005` | `fmax<0.001` |
|---|---:|---:|---:|---:|---:|
| Plain GAD `dt=0.003` | 62.7 | 52.3 | 40.8 | 0.0 | **0.0** |
| Plain GAD `dt=0.005` | 63.1 | 53.7 | 43.2 | 0.0 | **0.0** |
| Plain GAD `dt=0.007` | **68.3** | **56.4** | **44.6** | 0.0 | **0.0** |
| CS²-GAD, held-out | 57.8 | 48.4 | 38.7 | 2.8 | **0.0** |
| Sella Cartesian | 27.9 | 24.7 | 18.8 | 5.6 | **0.0** |
| Sella Cartesian+Eckart A | 46.0 | 39.0 | 27.2 | **7.0** | **0.0** |
| Sella Cartesian+Eckart B | 39.7 | 34.5 | 23.3 | **7.0** | **0.0** |
| Sella internal | 20.9 | 18.1 | 13.9 | 4.2 | **0.0** |
| Hybrid damped Eckart | 44.9 | 37.6 | 33.1 | 6.6 | **0.0** |
| Hybrid undamped Eckart | 43.9 | 37.3 | 31.0 | 1.4 | **0.0** |

## HIP intended IRC_TOPO, percent of 287

| Method | 0.10 A | 0.15 A | 0.20 A |
|---|---:|---:|---:|
| Plain GAD `dt=0.003` | **78.4%** | 61.0% | **44.6%** |
| Plain GAD `dt=0.005` | 78.0% | **61.7%** | **44.6%** |
| Plain GAD `dt=0.007` | **78.4%** | **61.7%** | 43.9% |
| CS²-GAD, held-out | 76.0% | 57.8% | 40.8% |
| Sella Cartesian | 70.7% | 46.7% | 17.8% |
| Sella Cartesian+Eckart A | 72.5% | 49.8% | 23.3% |
| Sella Cartesian+Eckart B | — | — | 22.0% |
| Sella internal | 64.5% | 33.1% | 16.0% |
| Hybrid damped Eckart | 76.7% | 57.5% | 38.7% |
| Hybrid undamped Eckart | 77.0% | 57.1% | 38.7% |
