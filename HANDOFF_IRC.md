# IRC experiment handoff — sella_hip vs rigorous

Context for a fresh Claude Code session on Narval. Read this + `CLAUDE.md` + `EXPERIMENT_LOG.md` and you'll have full context.

## What we're doing

Comparing two IRC integrators on TSs that `gad_dt003` converged to (the current best GAD method, per EXPERIMENT_LOG Round 2/3, dt=0.003, 2000 steps).

Only two methods matter for this experiment — **do not run `sella_baseline`**:

| Method | What it is | Code |
|---|---|---|
| `sella_hip` | Exactly Sella's IRC (same `dx=0.1, eta=1e-4, gamma=0.4`, same trust-region QN inner loop, same Cartesian coords) — the ONLY change is that HIP's analytical Hessian (mass-weighted, Eckart-projected, un-mass-weighted) is injected into `pes.H` after every inner kick, overwriting BFGS updates. | [src/gadplus/search/irc_sella_hip.py](src/gadplus/search/irc_sella_hip.py) |
| `rigorous` | Hratchian-Schlegel EulerPC-inspired predictor-corrector. Mass-weighted Cartesians throughout. Eckart projection of both gradient AND Hessian every step. Initial kick along lowest *vibrational* eigenvector (ignores residual TR modes). Per step: HIP Hessian at current AND predictor points, midpoint-averaged gradient, curvature correction `-0.5·s²·(I-ĝĝᵀ)·H·ĝ/|g|`. Adaptive arc-length clamped against `√λ_max`. K=2 consecutive convergence flags required. | [src/gadplus/search/irc_rigorous.py](src/gadplus/search/irc_rigorous.py) |

Both return the same `IRCResult` via the shared `score_endpoints` helper in [src/gadplus/search/irc_validate.py](src/gadplus/search/irc_validate.py) — so endpoint scoring (Kabsch+Hungarian RMSD <0.3 Å and element-labeled bond-graph isomorphism) is identical across methods. Fair comparison.

## Key parameters (already set)

- `--irc-steps 500` — max steps per direction for both methods.
- `--rmsd-threshold 0.3` — Å, direction-agnostic (forward/reverse can swap).
- **Naive convergence acceptance**: the validator trusts the duckdb `converged=true` label and pulls coords at exactly the `converged_step` recorded in the summary parquet. No re-verification, no refinement, no quality gating. This is deliberate — we're measuring IRC quality, not TS quality.
- Sella's fmax (inner): 0.01 eV/Å (unchanged from baseline).
- Rigorous `grad_tol`: 5e-4, `k_hold`: 2, `alpha_clamp`: 0.3, `s_min`: 0.01, `s_max`: 0.15 (all defaults in [src/gadplus/search/irc_rigorous.py](src/gadplus/search/irc_rigorous.py)).

## Data source — ONLY use gad_dt003 parquets

Do not use `noise_survey_300/` or any other directory.

| Noise | Parquets dir |
|---|---|
| 10, 30, 50 pm | `/lustre07/scratch/memoozd/gadplus/runs/round2/` |
| 100, 150, 200 pm | `/lustre07/scratch/memoozd/gadplus/runs/round3/` |

Pass one of those to `--survey-dir`. If you want a noise level that spans both rounds (not needed for smoke test), you'll have to submit two jobs — the script doesn't currently take multiple survey dirs.

The smoke-test slurm ([scripts/run_irc_smoke.slurm](scripts/run_irc_smoke.slurm)) is already pointed at `round2/` and iterates 10/50 pm × 2 methods × 5 samples.

## Workflow

### Step 1 — git pull

The Mac session wrote these files you need:

- `src/gadplus/search/irc_sella_hip.py` (new)
- `src/gadplus/search/irc_rigorous.py` (new)
- `src/gadplus/search/irc_validate.py` (refactored — `score_endpoints` extracted)
- `scripts/irc_validate.py` (added `--method` flag, bumped `--irc-steps` default to 500, added `method` column to output parquet)
- `scripts/analyze_irc.py` (new — duckdb analyzer)
- `scripts/run_irc_smoke.slurm` (new — smoke-test launcher)

Before doing anything else:

```bash
cd /lustre06/project/6033559/memoozd/GAD_plus
git pull
ls src/gadplus/search/irc_rigorous.py  # should exist
```

If the files aren't present, stop and ping me — the Mac session may not have pushed yet.

### Step 2 — submit smoke test

```bash
sbatch scripts/run_irc_smoke.slurm
# 4 array tasks: 2 methods (sella_hip, rigorous) x 2 noise (10, 50 pm) x N=5
```

Check queue, note the job ID:

```bash
squeue -u $USER -o "%.18i %.20j %.2t %.10M %.6D %R"
```

MIG jobs typically start within minutes. Don't use interactive `salloc` — wait for sbatch.

### Step 3 — monitor

After submission, poll once every ~2-3 minutes. When all 4 tasks finish (STATE disappears from squeue output):

```bash
ls /lustre07/scratch/memoozd/gadplus/runs/irc_sellahip_rigorous/
# expect: irc_validation_sella_hip_10pm.parquet
#         irc_validation_sella_hip_50pm.parquet
#         irc_validation_rigorous_10pm.parquet
#         irc_validation_rigorous_50pm.parquet
```

Check the per-task SLURM logs for crashes:

```bash
tail -30 /lustre07/scratch/memoozd/gadplus/logs/ircsmoke_<JOBID>_0.out
# Repeat for _1, _2, _3
```

### Step 4 — automated analysis

```bash
python scripts/analyze_irc.py /lustre07/scratch/memoozd/gadplus/runs/irc_sellahip_rigorous/
```

Produces three things:

1. **Per-(method, noise) summary table** (stdout + optional `--csv ...`). Columns: `n`, `n_intended`, `n_half`, `n_error`, `intended_pct`, `topo_int_pct`, `error_pct`, `avg_wall_s`.

2. **One stacked bar chart per noise level**, saved as `irc_bars_{noise_pm}pm.png` in the output dir. Each chart shows proportions of **intended / half / unintended / error** stacked per method. Sample count annotated above each bar.

3. **Text diagnostics on unintended runs.** For each (method, noise), prints:
   - How many unintended samples had **both endpoints at valid minima** (n_neg_vib == 0 on forward AND reverse) — those are cases where IRC reached a minimum, just the wrong one. Chemically meaningful failure.
   - How many had one endpoint at a valid minimum.
   - How many had neither (IRC didn't finish; often at a shoulder or saddle).
   - **Top 5 "closest miss" samples** per cell, sorted by min(forward/reverse, reactant/product) RMSD. For each: sample_id, formula, both endpoint n_neg values, both endpoint min vibrational eigenvalues, and all four raw RMSDs.

The endpoint diagnostics come from re-evaluating HIP's analytical Hessian at the forward and reverse coords at the end of each IRC run — stored as `forward_n_neg_vib`, `reverse_n_neg_vib`, `forward_min_vib_eig`, `reverse_min_vib_eig` in the output parquet. Negative `min_vib_eig` means the endpoint is still on a saddle/ridge; exactly zero n_neg and positive min_eig means it's a true minimum.

### Step 5 — report back to me

Paste the analyzer's stdout. Also include:

- Any SLURM task that didn't finish (stderr tail).
- Wall time per direction (look for the `RMSD=...|TOPO=...|...` lines in the stdout, they include `Xs` per sample).
- A 1-sentence interpretation: does `rigorous` beat `sella_hip`? By how much? Are error rates near zero or is something systematically broken?

### Step 6 — wait for my decision

I'll either:
- Ask you to scale to 300 samples at all 6 noise levels (10/30/50/100/150/200 pm) → edit `N_SAMPLES=300` and both survey dirs in the slurm, or submit two sets of jobs (one for round2, one for round3).
- Or change the code myself and ask you to re-run smoke.

Do NOT auto-scale to 300 without confirmation.

## If the smoke test errors out

Likely culprits in order of prior probability:

1. **Import error on cluster** (e.g., `gadplus.search.irc_rigorous` missing) → confirm `git pull` ran and `ls src/gadplus/search/irc_*.py` shows all four files.
2. **`networkx` not installed** (needed for bond-graph isomorphism) → `uv pip install networkx` into the venv. It's an optional dep per [src/gadplus/search/irc_validate.py:22](src/gadplus/search/irc_validate.py#L22).
3. **Sella version mismatch** on the injected `hessian_function` path — if `pes.calculate_hessian()` is undefined on the installed Sella, check `python -c "import sella; print(sella.__version__)"`. Should be >=2.3. The monkey-patch is in [src/gadplus/search/irc_sella_hip.py](src/gadplus/search/irc_sella_hip.py) and assumes `pes.hessian_function` + `pes.calculate_hessian()` exist (they do in 2.3.x).
4. **No converged TSs at the requested noise level** — the duckdb query returns 0 rows. Point at the right survey dir (see table above) or pick a different noise level. Empty → `print("No converged TS to validate.")` + clean exit.
5. **Rigorous integrator NaN / explode** — if the step-size clamp isn't tight enough for stiff modes. Symptoms: `error` column populated, or endpoints way off. First diagnostic: tighten `alpha_clamp` from 0.3 → 0.15 in `run_irc_rigorous(...)` call inside [scripts/irc_validate.py](scripts/irc_validate.py) dispatch, or just temporarily add `alpha_clamp=0.15` to the `_run_irc` kwargs. Report to me before changing.
6. **Sella inner-loop divergence** for sella_hip — the existing code catches exceptions per direction and stores `None` endpoint. Non-fatal for individual samples; high error% is fatal for the experiment. Report the count.

For runtime errors, fix items 1-2 yourself (they're environment). For items 3-6, diagnose and ping me.

## Scaling up (post-confirmation)

When approved, to run 300 samples at all 6 noise levels:

```bash
# Edit scripts/run_irc_smoke.slurm or duplicate it:
#   N_SAMPLES=300
#   NOISE_PMS=(10 30 50)             -> submit with SURVEY_DIR=.../round2
#   NOISE_PMS=(100 150 200)          -> separate job with SURVEY_DIR=.../round3
# Array size = 2 methods * 3 noise = 6 tasks per submission, 2 submissions.
# Bump --time to at least 6:00:00 — rigorous at 200pm takes 5000 HIP Hessian
# calls per direction at s_min which is slow.
```

Alternatively, update [scripts/irc_validate.py:100](scripts/irc_validate.py#L100) to accept comma-separated survey dirs, but I'd rather you just submit twice to avoid code churn.

## What to NOT touch

- Do not tune the algorithm hyperparameters unless explicitly approved. We're measuring, not hyperparameter-sweeping.
- Do not change the scoring criterion (RMSD threshold, topology check). Keep it consistent with the baseline `sella_baseline` runs so numbers are directly comparable.
- Do not re-enable refinement or TS-quality gates in [scripts/irc_validate.py](scripts/irc_validate.py). The current script is deliberately the trust-converged baseline.
- Do not commit without asking. Just push analysis CSVs / logs if helpful.

## Quick reference — file map

```
src/gadplus/search/
  irc_validate.py        # baseline (sella vanilla) + shared score_endpoints()
  irc_sella_hip.py       # variant 1 — Sella + HIP Hessian every step
  irc_rigorous.py        # variant 2 — predictor-corrector + HIP Hessian

scripts/
  irc_validate.py                 # driver — dispatches on --method
  analyze_irc.py                  # duckdb summary over result parquets
  run_irc_smoke.slurm             # 2 methods x 2 noise x 5 samples
  run_irc_validate_three_way.slurm  # 3 methods x 3 noise x 30 samples (ignore for now)

EXPERIMENT_LOG.md         # full context on gad_dt003 method and prior Sella baselines
HANDOFF_IRC.md            # this file
```

## Check-in format I want

When you report back after the smoke test, use this template:

```
Smoke test complete. Job <JOBID>.

Analyzer output:
<paste analyze_irc.py stdout — summary table + unintended text notes>

Bar charts written to:
<list PNG paths>

Key unintended-geometry signal:
- sella_hip @ 10pm: <X>/<N> unintended with both endpoints at n_neg=0 (valid minima)
                    Closest miss: sample <sid>, min RMSD <X.XXX>
- sella_hip @ 50pm: ...
- rigorous  @ 10pm: ...
- rigorous  @ 50pm: ...

Wall time observations (per direction, from stdout `| Xs` marks):
- sella_hip  @ 10pm: avg X s/sample
- rigorous   @ 10pm: avg X s/sample
- (etc)

Errors (if any):
<list>

Interpretation:
<2-3 sentences. Which method wins by intended%? Does rigorous find more
 valid-but-wrong minima (rules out shoulder-minima hypothesis) or more
 true-failures? Any sample systematically broken across both methods?>

Decision needed:
- Scale to 300 samples across all 6 noise levels?
- Fix something first? (flag algorithmic concerns)
```

Don't expand beyond that unless something's on fire.
