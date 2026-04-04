# GAD_plus Experiment Log

## Phase 1: Parameter sweep — 2026-04-03

Config: gad_projected (Level 2, Eckart on), 10 test samples, 100 steps, noise=0.05A (50pm), seed=42
Grid: dt=[0.001, 0.003, 0.005, 0.01, 0.02] x k_track=[0, 4, 8]

| dt | k_track | Conv | Rate | Avg Steps |
|----|---------|------|------|-----------|
| 0.01 | 0 | 4/10 | 40% | 83 |
| 0.01 | 4 | 3/10 | 30% | 82 |
| 0.01 | 8 | 3/10 | 30% | 83 |
| all others | * | 0/10 | 0% | — |

Finding: dt=0.01 is the only value that converges at 50pm noise in 100 steps. 2x the old default (0.005). k_track has negligible effect — projection handles eigenvector consistency, making explicit mode tracking redundant.

Comparison to Level 0 (from RESULTS_2026-04-03.md): Level 0 at dt=0.005 with 300 steps got 0% at 50pm. Level 2 at dt=0.01 with only 100 steps gets 40%. Eckart projection in dynamics + larger dt is a major improvement.

Best params: **dt=0.01, k_track=0**
Next: Phase 2 noise robustness survey with these params.
Data: /lustre07/scratch/memoozd/gadplus/runs/sweep_dt/sweep_dt_results.parquet
Job: 58833650
