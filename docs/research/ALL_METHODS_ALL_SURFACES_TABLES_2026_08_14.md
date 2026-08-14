# Every method on every surface

| Reading the tables | Meaning |
|---|---|
| **Bold value** | Best positive result in that column under the same protocol; all-zero ties are left unbolded |
| — | Not evaluated or not recorded |
| Local TS | Projected vibrational Hessian has exactly one negative mode and the stated force test passes |
| Validated outcome | LJ: two downhill minima; g-xTB: labelled native endpoint pair; HIP: intended full IRC topology |
| Denominators | LJ multisize: 448; LJ7: 288; g-xTB and HIP: 287 |

## Results at a glance

| Surface and representative hard cell | Best local TS result | Best tighter-force result | Best validated outcome | Selected non-Sella method |
|---|---|---|---|---|
| LJ13/31/38/55/75 pooled | **Sella: 422/448** | — | **CS²-GAD: 207/448 two-minimum pairs** | Intrinsic `lambda2` GAD; CS² as complementary channel |
| g-xTB, 0.20 A noise | **CS²-GAD: 262/287** at `fmax<0.03` | **Sella: 104/287** at `fmax<0.01` | **CS²-GAD: 124/287 native pairs** | CS²-GAD |
| HIP, 0.20 A noise | **Plain GAD `dt=0.007`: 44.6%** at `fmax<0.01` | **Sella Eckart configurations: 7.0%** at `fmax<0.005` | **Plain GAD: 44.6% intended IRC** | Plain GAD `dt=0.007` |

## Lennard-Jones

### Multisize local TS successes

| Method | LJ13 (64) | LJ31 (64) | LJ38 (128) | LJ55 (64) | LJ75 (128) | All sizes (448) |
|---|---:|---:|---:|---:|---:|---:|
| Intrinsic `lambda2` GAD | 56 | **63** | 115 | 49 | 112 | 395 (88.2%) |
| CS²-GAD | 59 | 60 | **122** | 58 | 100 | 399 (89.1%) |
| Sella | **60** | 61 | **122** | **61** | **118** | **422 (94.2%)** |

### Multisize validated two-minimum outcomes

| Method | LJ13 (64) | LJ31 (64) | LJ38 (128) | LJ55 (64) | LJ75 (128) | All sizes (448) |
|---|---:|---:|---:|---:|---:|---:|
| Intrinsic `lambda2` GAD | 52 | 30 | **71** | 14 | **33** | 200 (44.6%) |
| CS²-GAD | **53** | 35 | 70 | **19** | 30 | **207 (46.2%)** |
| Sella | **53** | **37** | 64 | 15 | **33** | 202 (45.1%) |

### Direct LJ comparison: best GAD-family result versus Sella

| LJ comparison | Best GAD-family result | Sella | Sella minus GAD | Better result |
|---|---:|---:|---:|---|
| LJ13 local TS | CS²: 59/64 | **60/64** | +1 | Sella |
| LJ31 local TS | **Intrinsic: 63/64** | 61/64 | -2 | Intrinsic |
| LJ38 local TS | **CS²: 122/128** | **122/128** | 0 | Tie |
| LJ55 local TS | CS²: 58/64 | **61/64** | +3 | Sella |
| LJ75 local TS | Intrinsic: 112/128 | **118/128** | +6 | Sella |
| All-size local TS | CS²: 399/448 | **422/448** | +23 (+5.1 pp) | Sella |
| All-size two-minimum outcome | **CS²: 207/448** | 202/448 | -5 (-1.1 pp) | CS² |
| Median evaluations among pooled local successes; lower is better | Intrinsic: 25 | **20.5** | -4.5 evaluations | Sella |

### LJ7 matched method progression

| Method | Local TS successes | Valid two-minimum outcomes | Median evaluations among local successes |
|---|---:|---:|---:|
| Ordinary GAD | 115/288 (39.9%) | 115/288 (39.9%) | 494 |
| Sella | 169/288 (58.7%) | 167/288 (58.0%) | **13** |
| Hard descent-to-GAD gate | 283/288 (98.3%) | 283/288 (98.3%) | 240 |
| Historical smooth `lambda2` gate | 283/288 (98.3%) | 283/288 (98.3%) | 240 |
| Pointwise intrinsic `lambda2` GAD | **288/288 (100%)** | **288/288 (100%)** | 17 |

## g-xTB / Transition1x

### Local TS successes at `fmax<0.03` eV/A

| Method | 0.10 A noise | 0.20 A | 0.50 A | 1.00 A | 2.00 A |
|---|---:|---:|---:|---:|---:|
| Regular GAD | 35 | 2 | 23 | 3 | 1 |
| Competitive GAD | 264 | 236 | 204 | **105** | **8** |
| CS²-GAD | **282** | **262** | **221** | 90 | 3 |
| Sella | 275 | 261 | 202 | 87 | 4 |

### Validated labelled-native endpoint pairs

| Method | 0.10 A noise | 0.20 A | 0.50 A | 1.00 A | 2.00 A |
|---|---:|---:|---:|---:|---:|
| Regular GAD | 33 | 2 | 3 | 0 | 0 |
| Competitive GAD | 222 | 123 | **5** | 0 | 0 |
| CS²-GAD | **231** | **124** | **5** | 0 | 0 |
| Sella | 210 | 107 | 0 | 0 | 0 |

### What changes when the g-xTB convergence criterion changes at 0.20 A

| Method | Local TS, `fmax<0.03` | Stricter local TS, `fmax<0.01` | Native endpoint pair | Median steps among local successes |
|---|---:|---:|---:|---:|
| Regular GAD | 2 | 0 | 2 | 282.5 |
| Competitive GAD | 236 | 9 | 123 | 28 |
| CS²-GAD | **262** | 10 | **124** | 29.5 |
| Sella | 261 | **104** | 107 | **15** |

## HIP / held-out Transition1x test set

### Local TS success at `fmax<0.01` eV/A

| Method | 0.01 A noise | 0.03 A | 0.05 A | 0.10 A | 0.15 A | 0.20 A |
|---|---:|---:|---:|---:|---:|---:|
| Plain GAD, `dt=0.003` | 89.2% | 88.5% | 85.4% | 71.1% | 55.1% | 40.8% |
| Plain GAD, `dt=0.005` | 89.2% | 88.5% | 85.7% | 71.8% | 57.1% | 43.2% |
| Plain GAD, `dt=0.007` | 89.2% | 88.9% | 85.7% | **72.8%** | **58.2%** | **44.6%** |
| CS²-GAD, held-out | — | — | — | 70.4% | 54.7% | 38.7% |
| Sella, Cartesian | 92.0% | 91.3% | 87.5% | 65.5% | 42.9% | 18.8% |
| Sella, Cartesian + Eckart, Hessian every step | 92.7% | 92.0% | 88.2% | 70.7% | 54.0% | 27.2% |
| Sella, Cartesian + Eckart, Hessian every 3 steps | **96.5%** | **95.5%** | **92.0%** | **72.8%** | 50.5% | 23.3% |
| Sella, internal coordinates | 79.1% | 77.4% | 71.8% | 50.9% | 26.8% | 13.9% |
| GAD + damped Eckart/Newton refinement | 85.4% | 85.0% | 81.5% | 66.9% | 50.9% | 33.1% |
| GAD + undamped Eckart/Newton refinement | 84.7% | 84.3% | 81.5% | 65.5% | 49.8% | 31.0% |

### What changes when the HIP force criterion changes at 0.20 A

| Method | `fmax<0.050` | `fmax<0.023` | Registered `fmax<0.010` | Tight `fmax<0.005` | `fmax<0.001` |
|---|---:|---:|---:|---:|---:|
| Plain GAD, `dt=0.003` | 62.7% | 52.3% | 40.8% | 0.0% | 0.0% |
| Plain GAD, `dt=0.005` | 63.1% | 53.7% | 43.2% | 0.0% | 0.0% |
| Plain GAD, `dt=0.007` | **68.3%** | **56.4%** | **44.6%** | 0.0% | 0.0% |
| CS²-GAD, held-out | 57.8% | 48.4% | 38.7% | 2.8% | 0.0% |
| Sella, Cartesian | 27.9% | 24.7% | 18.8% | 5.6% | 0.0% |
| Sella, Cartesian + Eckart, Hessian every step | 46.0% | 39.0% | 27.2% | **7.0%** | 0.0% |
| Sella, Cartesian + Eckart, Hessian every 3 steps | 39.7% | 34.5% | 23.3% | **7.0%** | 0.0% |
| Sella, internal coordinates | 20.9% | 18.1% | 13.9% | 4.2% | 0.0% |
| GAD + damped Eckart/Newton refinement | 44.9% | 37.6% | 33.1% | 6.6% | 0.0% |
| GAD + undamped Eckart/Newton refinement | 43.9% | 37.3% | 31.0% | 1.4% | 0.0% |

### Intended full-IRC mechanism recovery

| Method | 0.10 A noise | 0.15 A | 0.20 A |
|---|---:|---:|---:|
| Plain GAD, `dt=0.003` | **78.4%** | 61.0% | **44.6%** |
| Plain GAD, `dt=0.005` | 78.0% | **61.7%** | **44.6%** |
| Plain GAD, `dt=0.007` | **78.4%** | **61.7%** | 43.9% |
| CS²-GAD, held-out | 76.0% | 57.8% | 40.8% |
| Sella, Cartesian | 70.7% | 46.7% | 17.8% |
| Sella, Cartesian + Eckart, Hessian every step | 72.5% | 49.8% | 23.3% |
| Sella, Cartesian + Eckart, Hessian every 3 steps | — | — | 22.0% |
| Sella, internal coordinates | 64.5% | 33.1% | 16.0% |
| GAD + damped Eckart/Newton refinement | 76.7% | 57.5% | 38.7% |
| GAD + undamped Eckart/Newton refinement | 77.0% | 57.1% | 38.7% |
