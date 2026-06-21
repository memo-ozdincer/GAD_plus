# Lennard-Jones LJ7 First-Order GAD Notes

These notes intentionally document only the current first-order LJ experiments.
Older hybrid/Newton and pre-Gaussian-start experiments are outdated.

## Setup

Transition-state candidate criterion:

- `final_n_neg == 1`
- `final_force_max < 0.05`

Current LJ runner defaults and conventions:

- Analytical LJ derivatives only.
- `--start-from gaussian_origin`
- `--gaussian-origin-sigma 1.0`
- LJ7 with `epsilon=1`, `sigma=1`
- `--atomic-number 1` for unit-mass Eckart projection
- `--method gad`
- `--high-index-descent gradient`
- `--use-projection`
- `--force-criterion fmax`
- `--force-threshold 0.05`
- `--k-track 0` (mode tracking off; always flip along the lowest Eckart eigenvector)

The dynamics are first order only:

- Use projected Eckart GAD when `n_neg <= 1`.
- Use projected Eckart gradient descent when `n_neg > 1`.
- No Newton or curvature-scaled step is used.

## 1000-Step Sweep

Output root:

`runs/lj_gaussian_origin_projected_firstorder_dt_sweep`

All rows use `n_samples=100`, `n_steps=1000`, `start_from=gaussian_origin`, and
`gaussian_origin_sigma=1.0`.

| Output directory | `dt` | TS candidates | Index-1 finals | Final index pattern | Median converged step | Mean converged step |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| `dt0005` | `0.0005` | 0/100 | 4 | `{1: 4, 2: 10, 3: 9, 4: 23, 5: 23, 6: 16, 7: 11, 8: 4}` | n/a | n/a |
| `dt001` | `0.001` | 5/100 | 17 | `{1: 17, 2: 14, 3: 16, 4: 21, 5: 16, 6: 11, 7: 2, 8: 3}` | 840 | 848 |
| `dt002` | `0.002` | 20/100 | 43 | `{1: 43, 2: 26, 3: 9, 4: 10, 5: 8, 6: 3, 7: 1}` | 590 | 629 |
| `dt003` | `0.003` | 42/100 | 59 | `{1: 59, 2: 26, 3: 5, 4: 8, 5: 1, 7: 1}` | 683 | 631 |
| `dt005` | `0.005` | 61/100 | 77 | `{1: 77, 2: 16, 3: 4, 4: 2, 5: 1}` | 476 | 492 |
| `dt007` | `0.007` | 70/100 | 79 | `{1: 79, 2: 19, 3: 1, 4: 1}` | 360 | 408 |

Runtime for one 100-start, 1000-step CPU job was about 2.5-6 minutes on
`cpu_short`.

## Larger-dt and 2000-Step Follow-Up

Larger time steps at 1000 max steps:

| Output directory | `dt` | `n_steps` | TS candidates | Index-1 finals | Final index pattern |
| --- | ---: | ---: | ---: | ---: | --- |
| `dt010` | `0.01` | 1000 | 0/100 | 79 | `{1: 79, 2: 2, 3: 18, 4: 1}` |
| `dt015` | `0.015` | 1000 | 0/100 | 77 | `{0: 1, 1: 77, 2: 1, 3: 19, 4: 2}` |
| `dt020` | `0.02` | 1000 | 0/100 | 79 | `{0: 1, 1: 79, 2: 1, 3: 17, 4: 2}` |

Longer 2000-step budget:

| Output directory | `dt` | `n_steps` | TS candidates | Index-1 finals | Final index pattern | Median converged step | Mean converged step |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| `dt007_n2000` | `0.007` | 2000 | 73/100 | 87 | `{1: 87, 2: 12, 3: 1}` | 380 | 443 |
| `dt010_n2000` | `0.01` | 2000 | 0/100 | 80 | `{1: 80, 2: 1, 3: 19}` | n/a | n/a |
| `dt015_n2000` | `0.015` | 2000 | 0/100 | 77 | `{0: 1, 1: 77, 2: 1, 3: 19, 4: 2}` | n/a | n/a |
| `dt020_n2000` | `0.02` | 2000 | 0/100 | 79 | `{0: 1, 1: 79, 2: 1, 3: 17, 4: 2}` | n/a | n/a |

Current best setting:

`dt=0.007` with `n_steps=2000`, giving 73/100 TS candidates.

## Interpretation

The step size matters strongly for Gaussian-origin LJ7 starts. Small `dt`
values spend too much of the 1000-step budget in high-index regions. Increasing
`dt` up to `0.007` improves both convergence rate and speed among converged
trajectories. Larger steps, starting at `dt=0.01`, often reach index 1 but keep
large residual forces, so they do not satisfy the TS criterion. Extending the
budget from 1000 to 2000 steps improves `dt=0.007` from 70/100 to 73/100.
