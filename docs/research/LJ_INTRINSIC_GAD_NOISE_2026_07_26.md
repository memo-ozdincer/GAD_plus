# Pointwise Intrinsic GAD on Noised LJ7, July 26, 2026

## Executive result

The pointwise intrinsic GAD optimizer is extremely robust as a **local saddle
finder** on analytic LJ7. Across 2,880 trajectories at noise up to
`0.50 sigma`, every trajectory reached the strict gate

\[
n_{\rm neg}=1,\qquad \|F\|_\infty<0.01.
\]

Of these, 2,870/2,880 also produced two projected-minimum downhill endpoints
under the current endpoint procedure. Noise did not mainly cause optimizer
failure; it increased exploration. The main sweep resolved 11 permutation-
invariant saddle families and at least 8 endpoint-energy-pair families.

At `0.75` and `1.0 sigma`, all 96 additional trajectories still passed the
strict TS and two-endpoint gates. At `1.5 sigma`, 46/48 passed both gates, but
physical quality began to break down: 17--22% of accepted candidates were
near-flat and fragmented under the declared filters, and the 95th-percentile
barrier rose above `4 epsilon`. This places the useful exploratory range below
`1.5 sigma` for this LJ7 setup, with `0.1--0.5 sigma` providing substantial
diversity without observed fragmentation or near-flat saddles.

## Scientific interpretation for diffusion-terminal use

For a generative diffusion model, recovery of one predetermined saddle is not
the primary success metric. A useful terminal optimizer should map dispersed
samples into nearby valid index-one basins while preserving diversity. The
metric hierarchy used to interpret these results is therefore:

1. strict index-one and force convergence;
2. two physically relaxed, projected-minimum downhill endpoints;
3. energy/barrier, spectral-separation, and fragmentation filters;
4. diversity across saddle and endpoint-event families;
5. recovery of the original reference event as an optional selectivity
   diagnostic only.

The optimizer performed strongly on levels 1--4. The falling reference-event
rate with noise is evidence of basin exploration, not optimizer failure.

## Method

### Surface and optimizer

- analytic reduced-unit LJ7, `epsilon=sigma=1`;
- uniform assigned hydrogen mass `1.008`;
- pointwise intrinsic GAD with spectral temperature `0.01` and locality
  fraction `0.05`;
- 200 pointwise updates in the main sweep and 300 in the high-noise tail;
- projected numerical index threshold `lambda < -1e-4`;
- strict force threshold `fmax < 0.01`.

The mathematical derivation is in `docs/POINTWISE_INTRINSIC_GAD.md`.

### Panels and sampling

Two panels used the same paired Gaussian perturbations:

- `saddle`: noise added to a converged reference LJ7 saddle;
- `pushed`: noise added to the mode-0-pushed LJ7 minimum that targets that
  saddle.

The main panel used 96 seeds in every cell at noise

```text
0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.125,
0.15, 0.20, 0.25, 0.30, 0.40, 0.50 sigma.
```

The high-noise tail used 24 seeds per panel at `0.60`, `0.75`, `1.0`, and
`1.5 sigma`.

### Downhill endpoint validation

For every strict TS candidate:

1. recompute the projected unstable mode;
2. displace by `0.03 sigma` in both signs;
3. minimize physical LJ energy with analytic gradients and L-BFGS-B;
4. require both endpoints to have projected `n_neg=0` and
   `fmax < 1e-5`.

This is a two-branch downhill endpoint validation. It is the LJ analogue used
here for an `IRC_TOPO`-type screen, but it is **not** a discretized intrinsic
reaction-coordinate integration. LJ also lacks labelled chemical bond
topologies, so endpoint-energy-pair counts are lower bounds on event diversity.

## Main sweep results

### Local convergence and endpoint validity

| Noise range | Strict TS gate | Two valid endpoints |
|---|---:|---:|
| `0--0.50 sigma`, both panels | 2880/2880 (100%) | 2870/2880 (99.65%) |
| `0.60 sigma`, both panels | 48/48 (100%) | 48/48 (100%) |
| `0.75--1.0 sigma`, both panels | 96/96 (100%) | 96/96 (100%) |
| `1.5 sigma`, both panels | 46/48 (95.8%) | 46/48 (95.8%) |

The two `1.5 sigma` failures were the same paired perturbation in the two
panels. Both ended at index one after 300 steps but still had `fmax about 0.389`;
loosening the gate to `0.03` or `0.05` would not rescue them.

Ten main-sweep endpoint validations failed the very tight endpoint-force gate.
In every case both endpoint Hessians had index zero, but one L-BFGS-B branch
stopped with a force of roughly `3.8--8.8`. These are relaxation failures, not
marginal force-threshold disagreements, and should be retried with a more
robust endpoint minimizer before being treated as invalid events.

### Diversity

Permutation-invariant families used energy tolerance `1e-4` and sorted
pair-distance RMS tolerance `1e-3`.

| Panel/noise | Valid candidates | TS families | Effective TS families | Endpoint-pair lower bound | Largest-family share |
|---|---:|---:|---:|---:|---:|
| pushed `0.10` | 96 | 4 | 2.8 | 3 | 63.5% |
| pushed `0.20` | 96 | 7 | 4.6 | 6 | 33.3% |
| pushed `0.40` | 96 | 11 | 8.6 | 8 | 27.1% |
| saddle `0.10` | 96 | 5 | 1.3 | 5 | 94.8% |
| saddle `0.20` | 94 | 7 | 5.4 | 6 | 40.4% |
| saddle `0.40` | 95 | 11 | 8.5 | 8 | 23.2% |

Across the main sweep, all 11 resolved saddle families occurred in both
panels. The pushed panel populated non-reference families at lower noise than
the saddle-centered panel, as expected from its greater initial distance from
the saddle.

### Optional reference-event selectivity

The reference event remained dominant only at modest perturbations:

| Noise | Pushed-start same event | Saddle-centered same event |
|---:|---:|---:|
| `0.02` | 100.0% | 100.0% |
| `0.05` | 85.4% | 100.0% |
| `0.10` | 63.5% | 94.8% |
| `0.20` | 33.3% | 39.6% |
| `0.30` | 10.4% | 15.6% |
| `0.50` | 7.3% | 7.3% |

This table measures selectivity, not validity. At `0.50 sigma`, strict
convergence was still 100% and two-endpoint validation was 99--100%.

## Secondary physical filters

Across the 2,870 main-sweep candidates with two valid endpoints:

- 11 saddle families and at least 8 endpoint-energy-pair families were found;
- 2,719/2,870 (94.7%) had endpoints with different minimum energies;
- no candidate was fragmented using connectivity cutoff `1.5 sigma`;
- no candidate had normalized `lambda_2/s_H < 0.01`;
- `lambda_2/s_H` ranged from `0.0397` to `0.2730`;
- maximum pair distance ranged from `1.916` to `2.697 sigma`;
- energy above the LJ7 global minimum ranged from `1.061` to `1.957 epsilon`;
- the barrier above the higher-energy endpoint had median `0.490 epsilon`,
  95th percentile `0.936 epsilon`, and maximum `1.025 epsilon`.

An absolute thermal-accessibility cap depends on the chosen reduced
temperature `T*=k_B T/epsilon`; the report therefore records barrier
distributions rather than declaring a universal temperature cutoff.

At `1.5 sigma`, the screen changed qualitatively:

| Panel | Near-flat | Fragmented | Barrier q95 |
|---|---:|---:|---:|
| pushed | 21.7% | 21.7% | `4.200 epsilon` |
| saddle | 17.4% | 17.4% | `4.100 epsilon` |

This is the clearest empirical high-noise boundary in the current experiment.

## Historical `lambda_2` Euler comparison

The historical formulation used

\[
w=\sigma(50\lambda_2),\qquad
F_{\rm gate}=F-2w(F^Tu_1)u_1,
\]

with `dt=0.005`, per-atom cap `0.005`, and an 8000-step budget. The comparison
used the first 48 paired seeds at `0.10` and `0.20 sigma`.

| Panel/noise | Method | Strict | Two valid endpoints | Same reference event | Median steps for same event |
|---|---|---:|---:|---:|---:|
| pushed `0.10` | pointwise intrinsic | 100% | 100% | 62.5% | 11.0 |
| pushed `0.10` | historical Euler | 100% | 100% | 41.7% | 158.5 |
| pushed `0.20` | pointwise intrinsic | 100% | 100% | 33.3% | 14.5 |
| pushed `0.20` | historical Euler | 100% | 100% | 27.1% | 201 |
| saddle `0.10` | pointwise intrinsic | 100% | 100% | 93.8% | 9 |
| saddle `0.10` | historical Euler | 100% | 100% | 93.8% | 118 |
| saddle `0.20` | pointwise intrinsic | 100% | 100% | 43.8% | 15 |
| saddle `0.20` | historical Euler | 100% | 97.9% | 39.6% | 189 |

Both methods are strong local saddle finders on these selected cells. The new
closed-form map reaches its terminal basin in roughly an order of magnitude
fewer pointwise iterations. Same-event differences are secondary in a
diversity-oriented application and should not be read as a universal quality
ranking.

## Force-threshold decision

GAD can be slow to reduce `fmax` on other surfaces, so a candidate screen at
`0.03` or `0.05` followed by stringent downstream validation remains a useful
future sensitivity test. It is not needed to explain the present LJ7 result:

- all 2,880 main candidates reached `fmax < 0.01`;
- all candidates through `1.0 sigma` reached `fmax < 0.01`;
- the two failures at `1.5 sigma` remained near `0.389`, far above either
  proposed relaxed threshold.

The maintained `0.01` result is therefore retained, while downstream endpoint
validity and physical filters carry the scientific interpretation.

## Reproducibility

Completed Trillium jobs used `def-aspuru` and packed one full 192-core CPU node
as 192 independently bound one-core Slurm tasks:

- main 2,880-trajectory sweep: job `1940584`;
- `0.60 sigma` tail: job `1940586`;
- `0.75/1.0/1.5 sigma` tail: job `1940591`;
- historical-gate comparison: job `1940587`;
- main diversity analysis: job `1940594`;
- high-tail diversity analysis: job `1940595`.

Primary artifact roots:

```text
/scratch/memoozd/gadplus/runs/lj-intrinsic-noise-1940584/
/scratch/memoozd/gadplus/runs/lj-intrinsic-noise-1940586/
/scratch/memoozd/gadplus/runs/lj-intrinsic-noise-1940591/
/scratch/memoozd/gadplus/runs/lj-lambda2-compare-1940587/
```

The preregistration and exact execution scripts are retained in the repository.

