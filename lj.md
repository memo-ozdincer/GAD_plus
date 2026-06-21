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

The dynamics are first order only:

- Use projected Eckart GAD when `n_neg <= 1`.
- Use projected Eckart gradient descent when `n_neg > 1`.
- No Newton or curvature-scaled step is used.

## Completed 1000-Step Sweep

Output root:

`runs/lj_gaussian_origin_projected_firstorder_dt_sweep`

All rows use `n_samples=100`, `n_steps=1000`, `start_from=gaussian_origin`,
and `gaussian_origin_sigma=1.0`.

| Output directory | `dt` | TS candidates | Index-1 finals | Final index pattern | Median converged step | Mean converged step |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| `dt0005` | `0.0005` | 0/100 | 5 | `{1: 5, 2: 9, 3: 9, 4: 23, 5: 23, 6: 16, 7: 11, 8: 4}` | n/a | n/a |
| `dt001` | `0.001` | 3/100 | 15 | `{1: 15, 2: 16, 3: 16, 4: 21, 5: 16, 6: 11, 7: 2, 8: 3}` | 841 | 820 |
| `dt002` | `0.002` | 14/100 | 39 | `{0: 3, 1: 39, 2: 27, 3: 9, 4: 10, 5: 8, 6: 3, 7: 1}` | 807 | 757 |
| `dt003` | `0.003` | 35/100 | 54 | `{0: 2, 1: 54, 2: 29, 3: 6, 4: 7, 5: 1, 7: 1}` | 688 | 674 |
| `dt005` | `0.005` | 62/100 | 75 | `{0: 1, 1: 75, 2: 18, 3: 4, 4: 2}` | 538 | 557 |
| `dt007` | `0.007` | 75/100 | 84 | `{1: 84, 2: 13, 3: 2, 4: 1}` | 420 | 443 |

Current best completed setting:

`dt=0.007`, with 75/100 TS candidates in 1000 steps.

Runtime for one 100-start, 1000-step CPU job was about 2.5-6 minutes on
`cpu_short`.

## Active Follow-Up Jobs

Larger time steps at 1000 max steps:

| Job ID | `dt` | `n_steps` | Output directory | Status at last check |
| --- | ---: | ---: | --- | --- |
| `29337427` | `0.01` | 1000 | `dt010` | running |
| `29337428` | `0.015` | 1000 | `dt015` | running |
| `29337429` | `0.02` | 1000 | `dt020` | running |

Longer 2000-step budget:

| Job ID | `dt` | `n_steps` | Output directory | Status at last check |
| --- | ---: | ---: | --- | --- |
| `29337430` | `0.007` | 2000 | `dt007_n2000` | running |
| `29337431` | `0.01` | 2000 | `dt010_n2000` | running |
| `29337432` | `0.015` | 2000 | `dt015_n2000` | running |
| `29337433` | `0.02` | 2000 | `dt020_n2000` | running |

## Current Interpretation

The step size matters strongly for Gaussian-origin LJ7 starts. Small `dt`
values spend too much of the 1000-step budget in high-index regions. Increasing
`dt` up to `0.007` improves both convergence rate and speed among converged
trajectories. The active larger-`dt` jobs test whether this trend continues or
whether the method becomes unstable.
