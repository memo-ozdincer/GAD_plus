# Comprehensive wave — 2026-05-16

SLURM job: **61087603** (9 tasks, all running on Narval MIG slices as of submission).

## What's in flight

| Task | Wave | What | Closes |
|---|---|---|---|
| 0 | R1-a | Hybrid damped Eckart eig tr=0.05 from **reactant** @ 0 pm | Reactant bar chart hybrid bar |
| 1 | R1-b | Hybrid undamped Eckart eig tr=0.05 from **reactant** @ 0 pm | Companion to R1-a |
| 2 | R2   | Sella cart+Eckart untuned **d=3** @ **200 pm** | Open `---` in Table 1 |
| 3 | R3-a | Sella internal tuned d=1 @ **150 pm** | Replaces partial 222/287 |
| 4 | R3-b | Sella internal tuned d=1 @ **200 pm** | Replaces partial 196/287 |
| 5 | R4-a | GAD dt=0.005 × **10000 steps** @ 50 pm | Probe fmax<0.001 reachability |
| 6 | R4-b | Sella cart+Eckart untuned d=1 × **10000 steps** @ 50 pm with target fmax=0.001 | "" |
| 7 | R4-c | Hybrid damped × **10000 steps** @ 50 pm with target fmax=0.001 | "" |
| 8 | R1-c | Sella cart+Eckart untuned d=1 from **midpoint** @ 0 pm | Cross-family reactant/midpoint companion |

## Code changes shipped

- `scripts/hybrid_gad_newton_runner.py`: added `--start-from {ts_noised, reactant, product, midpoint}` flag mirroring `scripts/sella_baseline.py`. Output dir tag includes `start-<name>` for non-default starts to avoid summary collisions.
- `scripts/run_comprehensive_2026_05_16.slurm`: 9-task array.

## Once tasks land

1. **Update reactant bar chart** (`figures_2026_05_16/fig_reactant_bar.pdf`): add hybrid damped + undamped bars at 0 pm.
2. **Refresh Table 1** (`master_2026_05_11.csv` → new master csv): replace `$^p$` partial cells (Sella internal 150/200 pm) and fill d=3 @ 200 pm.
3. **New fmax-reachability section**: 10k-step results at 50 pm — does any method reach fmax<0.001 with more compute, and at what wall cost?
4. Rebuild `BENCHMARK_REPORT_2026-05-16.pdf` from new data.

## Still open

- **R5** Mode-overlap-aware switching for hybrid: deferred. Needs design discussion before code (gate Newton trigger on eigenvector continuity in addition to n_neg=1).
- **Hybrid noise sweep from reactant** (1, 3, 5, 10 pm noise levels around reactant): the current cells are 0 pm only. If the user wants a full 4-axis figure with noise variation on reactant geometry, this is a follow-up sweep.
