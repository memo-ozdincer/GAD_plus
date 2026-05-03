# LIVE_STATUS — GAD+ comprehensive benchmark
**Last updated:** 2026-05-01 (auto-updated by Claude across sessions)
**Owner:** memo.ozdincer@mail.utoronto.ca
**Cluster:** Narval, account `rrg-aspuru`

This file is the source of truth for ongoing experimental work. If you
(future Claude) are restarting after a compaction, read this top to
bottom before doing anything else.

## 🎯 Headline result (2026-05-01, IRC validation landed)

**At 200pm noise, GAD beats best Sella by +21.3pp on IRC TOPO-intended
(44.6% vs 23.3%).** At 150pm: +11.9pp. At 100pm: +5.9pp. Tied below 50pm.

The IRC-validated gap is **bigger than the raw-conv gap** (+17.4pp at
fmax<0.01) because IRC catches Sella's wrong-saddle failures — Sella's
high-noise failures land at higher-order saddles (n_neg≥2) which IRC
correctly rejects, while GAD's failures are plateau-orbit in the right
basin.

Robustness checks done (all confirm headline):
- Sella step budget 2k → 5k → 10k: no change (within ±2pp). Step budget
  is NOT the handicap.
- Sella Hessian-injection cadence d=1, 3, 5, 10, 25: only 3-6pp drop;
  the cliff is between "any HIP H" and "no HIP H".
- Huge noise (300/500/1000/2000pm): GAD lead persists to ~500pm; both
  saturate beyond.

**Read `analysis_2026_04_29/FINDINGS.md` §15-18 for the full numbers.**

## Paper argument (target narrative)

> When Hessians are available (via MLIPs/HIP), a simple gentlest-ascent
> dynamics (GAD) outperforms the popular P-RFO + QN methods (Sella) on
> the chemistry ground truth (IRC TOPO-intended), with the gap growing
> with TS-noise. On Transition1x test split at 200pm, GAD recovers the
> intended R+P bond connectivity 44.6\% of the time vs Sella 23.3\%.
> The gap is +21.3pp (IRC-validated) and +17.4pp (raw conv) — IRC is
> the stronger metric because Sella's high-noise failures land at
> wrong saddles, which IRC catches. Sella's strength entirely depends
> on the injected MLIP Hessian; without it, Sella is at 5\%.

Convergence definition: $n_{\text{neg}}{=}1 \wedge \|F\|{<}\text{thresh}$.
Paper uses **thresh = 1e-4** (tight). We've been benchmarking at
**fmax<0.01** (loose, to match Sella). Need both reported.

Outcomes: **converged** (TS), **intended** (R+P bond connectivity
recovered after IRC), **half-intended** (one of R/P).

## Key constraints (don't violate)

- **All new sweeps use T1x test split** (287 samples), not train.
- **fmax<0.01 + n_neg=1** is the canonical loose criterion.
- **‖F‖<1e-4 + n_neg=1** is the strict (paper) criterion — track both.
- **Eckart projection** must be used for n_neg counting.
- Don't add features without independent benchmarking justification.

## Currently running jobs (Slurm)

Run `squeue -u memoozd` to see live state.

| ID | Sweep | State | Output dir / notes |
|---|---|---|---|
| 60051805 | Sella high-noise recovery | DONE (4 ok, 2 timeout) | `runs/test_set/sella_*` (sella_internal_default 150/200pm missing — partial in `logs/testsellarec_60051805_{4,5}.out`) |
| 60110188 | Sella no-Hessian sweep | TIMEOUT (all 12) | parsed: `analysis_2026_04_29/sella_nohess_partial.csv` |
| 60110465 | IRC validation (broken, split=train) | DONE | `runs/test_irc/*` parquets — superseded by 60146557 |
| 60146557 | IRC validation rerun | RUNNING | `runs/test_irc/*` (will overwrite) — verified by smoke 60145419 |
| 60110201 | From-reactants single-ended | RUNNING (~10h left) | `runs/test_reactant/{gad_dt003_fmax,gad_dt005_fmax,sella_carteck_libdef,sella_internal_default}` |
| 60110297 | Low-dt GAD diagnostic | RUNNING (~10h left) | `runs/test_lowdt/gad_dt{001,0005,0001}_fmax/` — logs `testlowdt_60110297_*.out` |
| 60145419 | IRC smoke test (5 samples) | DONE | `/tmp/irc_smoketest/`, log `testirc_smoke_60145419.out` (1 RMSD-INTENDED, 4 TOPO-INTENDED) |
| 60147671 | Sella Hessian-freq sweep | PENDING | `runs/test_hessfreq/sella_carteck_libdef_d{3,5,10,25}/` (4 freq × 6 noise) |
| 60148863 | Sella trajlog on test (6 cells) | PENDING | `runs/test_sella_trajlog/carteck_libdef/` — fills Sella in dynamics figure across noise |
| 60151717 | NR-GAD polish sweep (12 cells)  | PENDING | `runs/test_nrpolish/nr_gad_polish_dt007_{loose,strict}/` — tests if spectral NR breaks GAD plateau |
| 60153126 | Midpoint single-ended (4 cells) | PENDING | `runs/test_midpoint/{gad_dt003,gad_dt007,sella_carteck_libdef,sella_internal_default}/` — start from R+P midpoint, 0pm noise, 10000 steps |
| 60154004 | Huge-noise probe (16 cells, n=50) | PENDING | `runs/test_huge/*` at 300/500/1000/2000pm. Tests Sella's wrong-saddle mode + Sella-5k vs 2k step budget + GAD/HIP robustness at huge displacement |
| 60154183 | Sella matched-budget (12 cells)   | DONE 8/12 | `runs/test_sella_extended/carteck_libdef_{5k,10k}/` — 5k cells done; 10k high-noise timed out. **Result: step budget is NOT bottleneck (5k=2k within 2pp).** |
| 60314225 | NR-GAD polish (12 cells, n=80)    | PENDING | RELAUNCH of 60151717 — smaller n, longer time (18h, 3000 steps). Tests if NR breaks GAD plateau. |
| 60314226 | Midpoint single-ended (4 cells)   | PENDING | RELAUNCH of 60153126 — 96G mem, 18h. Sella internal at 10k steps OOMd at 48G. |

**Next launches planned (waiting in TODO):**
- IRC validation rerun (after split=test fix verified by smoke job 60145419)
- Adaptive-dt investigation (why does it collapse?) — diagnostic only
- **0 pm** noise added to existing grids — to see baseline behavior
- Tighter convergence threshold (1e-4) re-evaluation on existing data
- Sella recovery rerun for sella_internal_default 150/200pm (12h+ likely needed)

## Critical finding (added 2026-05-01): Sella collapses without HIP Hessian

| method | noise | n done | conv (Sella's gate) | conv (n_neg=1 ∧ F<0.01) |
|---|---|---|---|---|
| carteck_nohess | 10pm | 30 | 10.0% | 10.0% |
| carteck_nohess | 30pm | 29 | 3.4%  | 3.4% |
| carteck_nohess | 50pm | 30 | 10.0% | 10.0% |
| carteck_nohess | 100pm | 30 | 3.3% | 3.3% |
| carteck_nohess | 150pm | 29 | 0.0% | 0.0% |
| carteck_nohess | 200pm | 29 | 3.4% | 0.0% |
| internal_nohess | 10pm | 23 | 4.3% | 8.7% |
| internal_nohess | 30pm | 23 | 0.0% | 4.3% |
| internal_nohess | 50pm | 23 | 4.3% | 4.3% |
| internal_nohess | 100pm | 25 | 0.0% | 0.0% |
| internal_nohess | 150pm | 27 | 0.0% | 0.0% |
| internal_nohess | 200pm | 27 | 0.0% | 0.0% |

vs. with HIP Hessian (full 287, source `runs/test_set/sella_carteck_libdef/summary_*.parquet`):
Sella libdef 95% / 85% / 80% / 70% / 54% / 23%. (See `analysis_2026_04_29/threshold_sweep.csv`,
`threshold=0.01`, `gate=conv_fmax_pct`, `method='Sella libdef'`.)

Drop on the order of **80pp at 10pm** without the analytic Hessian. **Sella's
strength on T1x is entirely from the injected MLIP Hessian.** The QN+BFGS path
without it gets nowhere — median fmax 0.12-1.4 after 2000 steps, ~25 minutes
each. This is the load-bearing comparison for the paper's argument.

Median wall_s per attempt: ~1500s carteck, ~1850s internal. Hence the 12h timeout.

**Caveat:** the "no Hessian" endpoint is one extreme — Sella's library default
is `nsteps_per_diag=3` (recompute every 3 steps), not "never." The full
Hessian-frequency sweep (job 60147671, in progress) fills the curve at
`diag_every ∈ {3, 5, 10, 25}` to show how Sella degrades vs HIP-injection cadence.

**Sources:**
- raw: `logs/testsellanohess_60110188_*.out` (12 cells, all timed out before parquet write)
- parsed: `analysis_2026_04_29/sella_nohess_partial.csv` (via `scripts/parse_nohess_logs.py`)
- with-HIP comparison: `analysis_2026_04_29/threshold_sweep.csv`

## Low-dt diagnostic partial (2026-05-01, 60110297 still running)

| dt (steps) | 10pm | 30pm | 50pm | 100pm | 150pm | 200pm |
|---|---|---|---|---|---|---|
| 1e-3 (20k) | 89.7/282 | 89.3/242 | 86.4/191 | 77.4/124 | 56.6/76 | 50.0/64 |
| 5e-4 (40k) | 89.4/151 | 89.0/127 | 87.0/100 | 79.1/67 | 56.8/37 | 51.5/33 |
| 1e-4 (100k) | 91.9/62 | 90.5/42 | 87.9/33 | 81.0/21 | 50.0/12 | 45.5/11 |

ref: dt=7e-3 (5k) full 287 → 89/89/86/73/58/45.

70× dt drop gives ≤5pp improvement. **Plateau confirmed structural, not numerical.**
Sticking with dt=7e-3 + 5000 steps as the operating point.

**Sources:**
- raw logs: `logs/testlowdt_60110297_{0..17}.out` (tasks 0-5 dt=1e-3, 6-11 dt=5e-4, 12-17 dt=1e-4)
- partial-parquet writes: `runs/test_lowdt/gad_dt{001,0005,0001}_fmax/traj_*.parquet` (per-sample, no summary yet)
- ref dt=0.007: `runs/test_dtgrid/gad_dt007_fmax/summary_*.parquet`
- numbers parsed live with inline script (see chat log); will save to `analysis_2026_04_29/lowdt_partial.csv` once jobs complete

## From-reactant partial (2026-05-01, jobs still running)

Pulled live from slurm logs (jobs 60110201_0..3 still running).

| method | completed | converged (fmax<0.01 ∧ n_neg=1) | rate |
|---|---|---|---|
| GAD dt=0.003 | 86/287 | 44 | 51% |
| GAD dt=0.005 | 105/287 | 62 | 59% |
| Sella libdef (DONE) | 287/287 | 232 | 80.8% |
| Sella internal default | 207/287 | 181 | 87.4% |

**Sources:**
- GAD dt=0.003: `logs/testreact_60110201_0.out` (parsed live)
- GAD dt=0.005: `logs/testreact_60110201_1.out`
- Sella libdef (final): `runs/test_reactant/sella_carteck_libdef/summary_sella_cartesian_eckart_fmax0p01_carteck_libdef_0pm.parquet` (`converged` column, 233/287 = 81.2%)
- Sella internal: `logs/testreact_60110201_3.out`

**Caveat:** these are "any saddle" rates — many samples may climb to the *wrong*
saddle. Without IRC validation we can't tell. The paper's claim is that GAD
finds the *intended* TS more reliably from minima. Need IRC TOPO match to
make that claim.

Wall-time interesting: GAD takes 7000-9000 steps per sample (~600s) when it
converges; Sella averages ~50 steps + 5s. GAD is ~100× more compute-expensive
from reactant. Stops the paper from saying "GAD is cheap from minima" — it
isn't, but it is "single-ended".

## Sella recovery results (60051805)

- carteck_default 150/200pm: DONE → `runs/test_set/sella_carteck_default/summary_*_{150,200}pm.parquet`
- carteck_libdef 200pm: DONE → `runs/test_set/sella_carteck_libdef/summary_*_200pm.parquet`
- internal_default 100pm: DONE → `runs/test_set/sella_internal_default/summary_*_100pm.parquet`
- internal_default 150pm: TIMEOUT — 259 samples, 72 sella=CONV (~28%) → `logs/testsellarec_60051805_4.out`
- internal_default 200pm: TIMEOUT — 236 samples, 35 sella=CONV (~15%) → `logs/testsellarec_60051805_5.out`

Partial-coverage rows for missing internal cells parsed from logs at
job-end. The internal-coords Sella is ~3× slower per sample than carteck;
hence the timeouts.

## Critical bug (added 2026-05-01): IRC validation was running split=train

`scripts/irc_validate.py:218` hardcoded `split="train"`. When running on
test-split TS pools, sample_id 0 in the survey ≠ sample_id 0 in the dataset.
Reactant/product references were chemistry-mismatched. Combined with silent
exception swallowing in `irc_sella_hip.py:207`, every IRC cell returned
0 intended / 0 topo_intended.

Fixed: added `--split {train,test,val}` CLI arg; updated `run_test_irc.slurm`
to pass `--split test`; added traceback print on IRC exceptions. Smoke test
60145419 will verify before re-launching the full 36-cell sweep.

## Completed (data on disk)

| Sweep | Dir | Notes |
|---|---|---|
| Test-set GAD (dt003, dt005, adaptive_dt) at 2000 steps | `runs/test_set/gad_*` | 18/18 cells |
| Test-set Sella (3 configs) at 2000 steps | `runs/test_set/sella_*` | 12/18 cells, 6 timeout/OOM in recovery |
| Train Round 6 canonical | `runs/round2/`, `round3/`, `gad_eckart_fmax/`, `sella_2000/` | Historical reference |
| Sella tuning grid (3 configs × 6 noise × n=100, train) | `runs/sella_tune/{default,libdef,lson}/` | 18/18 cells |
| GAD bigger-dt partial (train, cancelled) | `runs/gad_bigger_dt/` | dt=0.005 OK; dt≥0.010 unstable |

## Critical finding: convergence-threshold sensitivity (added 2026-04-29)

The GAD-vs-Sella ranking inverts depending on the chosen force criterion:

| criterion | who wins | by how much |
|---|---|---|
| `fmax<0.01` (Sella default) | tied at low noise; GAD +2-17pp at 100-200pm | small |
| `force_norm<0.01` (GAD original) | GAD wins big | +24pp at 200pm |
| `fmax<0.005` | Sella wins everywhere | +30pp |
| `force_norm<0.005` | Sella wins everywhere | +60pp |
| `*<0.001` or tighter | nothing converges | — |

Mechanism: GAD plateaus near force_norm ≈ 0.005-0.01 (Euler-step granularity).
Sella's QN+trust-region drives forces to tiny values when it succeeds, hence
the bimodal RMSD distribution (close-or-far). GAD's residuals are uniformly
small but never tiny.

**Takeaway for paper:** at loose criterion + IRC validation, GAD wins at
high noise. The story has to acknowledge Sella's tighter-residual edge.

## Key findings so far

1. **dt=0.007 is the GAD sweet spot** (test, 5000 steps). Beats canonical dt=0.003 by +1.7pp at low noise and +3.8pp at 200pm. Exact cliff at dt∈(0.007, 0.008) — dt=0.008 collapses 17pp at low noise.
2. **GAD dt=0.007 beats Sella libdef** at 100pm (72.8 vs 70.7) and 150pm (58.2 vs 54.0) on test. Sella 200pm pending.
3. **dt=0.005 > dt=0.003** by ~+2pp at every noise AND 30–40% fewer steps when converged. Was the previous best fixed-dt.
4. **dt≥0.008 collapses** at low noise (Euler-stability cliff: dt < 2/λ_max for stiff bond modes).
5. **5000-step budget vs 2000** at 200pm gives +6pp for canonical dt=0.003 (34.8 → 40.8). Step budget mattered a lot at high noise.
6. **adaptive_dt is BAD on test** — drops 15–25 pp below fixed-dt. The eigenvalue clamp is doing the wrong thing. Investigate.
7. **Sella libdef (δ₀=0.1, γ=0.4)** beats default by +1–10 pp AND uses fewer fnevs (~15% cheaper). Tuning generalizes to test.
8. **Test set is HARDER than train** — both methods drop ~10pp at 100pm. Median GAD steps roughly 2× higher on test.
9. **65% of GAD-canonical (dt=0.003, 2000 steps) at 200pm hits the step cap**. Hence the 5000-step rerun.

## Open questions to investigate

1. Why does `gad_adaptive_dt` collapse on test? (eigenvalue clamp logic? overshoots?)
2. Step-count distributions for converged-vs-failed at each noise level.
3. With strict ‖F‖<1e-4 criterion, do the rankings change?
4. From-reactants: does GAD converge? Does Sella? With/without "kick"?
5. Sella w/o Hessians: does QN+BFGS recover, or does the H-injection matter?

## Reports / PDFs (canonical)

| File | Status |
|---|---|
| `IRC_COMPREHENSIVE_2026-04-20.pdf` | Old train-set 5-method | superseded |
| `IRC_COMPREHENSIVE_2026-04-28.pdf` | Train 3-method, partitioned bars, with prose | superseded by reduced |
| `IRC_COMPREHENSIVE_2026-04-28-reduced.pdf` | Reduced (no prose), train numbers | obsolete with test data |
| `IRC_COMPREHENSIVE_2026-04-28v2.pdf` | Used libdef-canonical Sella (n=100 train) | superseded |
| **(pending)** `IRC_TEST_2026-04-29.pdf` | **First test-set canonical report** | TO BE BUILT after dtgrid lands |
| **`IRC_TEST_2026-04-29.pdf`** | **First version built (11 pages, threshold sweep + RMSD distribs + dt grid + saddle-vs-no-saddle)** | **CURRENT** |

## How to resume after compaction

1. `cd /lustre06/project/6033559/memoozd/GAD_plus`
2. `squeue -u memoozd` — what's still running?
3. `cat LIVE_STATUS.md` — this file.
4. `tail -200 EXPERIMENT_LOG.md` — narrative of recent rounds.
5. `ls runs/test_*` and `runs/test_set/*` — what data has landed?
6. **Update this file** after every meaningful action.

## Slurm scripts reference

| Script | What it runs |
|---|---|
| `scripts/run_test_gad.slurm` | 3 GAD methods × 6 noise on test |
| `scripts/run_test_sella.slurm` | 3 Sella configs × 6 noise on test |
| `scripts/run_test_sella_recover.slurm` | High-noise Sella retries (12h, 96G) |
| `scripts/run_test_irc.slurm` | IRC validation on test TS pools |
| `scripts/run_test_gad_dtgrid.slurm` | 6 dt × 6 noise × 5000 steps |
| (TODO) `scripts/run_test_sella_nohess.slurm` | Sella w/o Hessian injection |
| (TODO) `scripts/run_test_reactant.slurm` | From-reactants single-ended |

## Memories saved (under `~/.claude/.../memory/`)

See `MEMORY.md` index. Key recent ones:
- `feedback_test_split_only.md` — all sweeps must use test split
- `feedback_diffusion_compatibility.md` — methods must be differentiable
- `project_eckart_n_neg.md` — Eckart for n_neg counting
- `feedback_narval_slurm.md` — WANDB_DISABLED, etc.
