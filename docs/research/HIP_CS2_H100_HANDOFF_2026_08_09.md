# H100 handoff: CS²-GAD on HIP / Transition1x

## Objective

Run Competitive Soft-Spectral GAD (CS²-GAD) on the same HIP checkpoint,
Transition1x test ordering, perturbations, and strict terminal criterion used
by the historical plain-GAD/Sella grid. The primary question is whether CS²
improves HIP recovery at `0.10`, `0.15`, and `0.20 A` without sacrificing the
already strong plain-GAD high-noise behavior.

This handoff does not claim that CS² will improve HIP. The g-xTB motivation
was prevention of high-index-to-minimum capture. HIP has not yet shown the
same failure taxonomy, and its lowest mode may be more reliable than its wider
low-curvature subspace.

## Frozen scientific protocol

- Surface: HIP `hip_v2.ckpt`, direct energy/force/Hessian products.
- Dataset: held-out, filtered Transition1x `test` split, first 287 valid
  records. The worker passes `split="test"` to `Transition1xDataset`, which
  passes `datasplit="test"` to the Transition1x HDF5 loader. The immutable
  live protocol records `split: "test"` and selected IDs `0..286`.
- Starts: labelled transition structures plus Cartesian Gaussian noise.
- Noise cells: `0.10`, `0.15`, and `0.20 A` for the first campaign.
- Noise seed: `42`.
- Noise construction: seed the CPU generator once per noise cell, then call
  `torch.randn_like(sample.pos)` sequentially in filtered test-set order.
  Every shard reconstructs the entire start table before selecting samples.
- Verified full start-table SHA256 values (concatenated float32 coordinate
  bytes in sample order):
  `c4817f12c171a4deadc4907c11d71b8c3465ab599ec219560d1576f9e9c44a9a`
  at `0.10 A`,
  `846ab8ed7d4fa47c86bfbd50209916bb14217c21414960921a5aed7aeefea161`
  at `0.15 A`, and
  `fdb17613f0d30c8ef6db96def7a06b812762fccaa9615c93907a1c3e393ba156`
  at `0.20 A`. The worker recomputes and enforces these digests before
  loading HIP.
- Optimizer: CS²-GAD, internal `gate_variant="competitive_subspace"`.
- Spectral temperature: `tau_s=0.01`.
- Step fraction: `eta=0.01`.
- Budget: 5,000 updates plus the terminal evaluation, matching the ceiling of
  the historical best plain-GAD reference used in the comparison table.
- Strict success: projected `n_neg == 1` at eigenvalue threshold `-1e-4` and
  Cartesian-component `fmax < 0.01 eV/A`.
- Primary denominator: all 287 planned starts. Also report the
  calculator-valid denominator separately.

The checkpoint itself records training data
`ts1x_hess_train_big.lmdb` and validation data `ts1x-val.lmdb` in its
`hyper_parameters.training_config`; neither is the HDF5 `test` partition used
here. CS²'s `eta=0.01` and `tau_s=0.01` were selected on the earlier g-xTB
work and frozen before this HIP test run. No HIP test outcome is used for
method selection or per-noise tuning. The three per-noise start-table hashes
below additionally prove that every shard used the precomputed held-out
starts rather than regenerating a different subset.

Do not tune `eta`, `tau_s`, or the budget independently in each noise cell.
If later calibration is desired, use a separate validation split and freeze
one configuration before returning to test.

## Files and paths

- Worker and aggregator: `scripts/hip_cs2_benchmark.py`
- Matched all-endpoint HIP IRC_TOPO validator: `scripts/hip_cs2_irc_topo.py`
- Checked cross-method report: `scripts/summarize_hip_cs2_results.py`
- H100 Slurm template: `scripts/run_hip_cs2_h100.slurm`
- Persistent interactive reservation: `scripts/reserve_hip_h100.slurm`
- Reservation payload: `scripts/keep_gpu_allocation.py`
- Reserved-node campaign: `scripts/run_hip_cs2_on_reserved_h100.sh`
- Allocation-lifetime supervisor: `scripts/supervise_reserved_hip_campaign.sh`
- Missing-only production continuation: `scripts/resume_hip_missing_on_reserved_h100.sh`
- Reproducible PyG overlay setup: `scripts/setup_hip_h100_pyg_overlay.sh`
- HIP checkpoint: `/scratch/memoozd/GAD/models/hip_v2.ckpt`
- Transition1x HDF5: `/scratch/memoozd/GAD/data/transition1x.h5`
- Default output: `/scratch/memoozd/gadplus/runs/hip-cs2-$ARRAY_JOB_ID/`

The launcher requests one H100 and eight CPU cores for each of 36 array
elements: three noise cells times twelve shards. Trillium assigns 186 GiB of
host memory per GPU and rejects explicit `--mem` requests. If another cluster
spells the H100 GRES differently, change only the Slurm resource line. Keep
the scientific arguments frozen. Trillium's submission wrapper selects the
partition from the GPU request; do not add an explicit partition directive.

The active Trillium workflow uses one persistent interactive H100 allocation,
not repeated short allocations. Submit `scripts/reserve_hip_h100.slurm`, wait
for it to run, obtain its node with `squeue -j JOB_ID`, and SSH to that node.
The batch payload sleeps indefinitely in Python until explicitly cancelled.
It records the allocated hostname and GPU visibility in
`/scratch/memoozd/gadplus/allocations/hip_h100_JOB_ID.json`.
Run all smoke tests and campaign commands from a separate SSH shell on that
node, then use `scancel JOB_ID` only after the campaign and aggregation are
complete. The default account is `def-aspuru`; consider `rrg-aspuru` only if a
measured default-account queue wait prevents the run. For this campaign,
`def-aspuru` job `733698` remained pending for more than three hours and its
estimated start moved later. Fallback `rrg-aspuru` reservation `733972`
started first on `trig0041`; the still-pending default-account duplicate was
then cancelled. The fallback reservation was retained for the whole campaign
rather than repeatedly returning the GPU to the queue.

The production search later exposed strong deterministic shard imbalance:
two modulo-four workers reached the end of the `0.10 A` cell while another
still had 29 long-budget sample IDs. Because the remaining search plus the
predeclared all-endpoint IRC could approach the reservation's 24-hour limit,
continuation reservation `734620` was submitted under the default
`def-aspuru` account with `Dependency=afterany:733972`. It remains dependency
pending and therefore cannot overlap the active H100. If job `733972` finishes
the complete campaign, cancel `734620`; otherwise, let it start once, SSH to
its recorded node, and resume the same immutable run root. Do not create a new
scientific run or change the frozen cell parameters.

## Binary dependency overlay

PyG 2.8.0 changed `radius_graph` to require `pyg-lib>=0.6.0`, but no matching
`pyg-lib>=0.6.0` wheel is published for the frozen PyTorch 2.7/CUDA 12.6
stack. HIP's own installation instructions use `torch-cluster`. To avoid
changing the shared 7 GiB `.venv` while LJ production is active, the HIP run
uses this small, isolated overlay:

```text
/scratch/memoozd/gadplus/envs/hip-pyg27-overlay/lib/python3.11/site-packages
```

It contains `torch-geometric==2.7.0`,
`torch-cluster==1.6.3+pt27cu126`, and
`torch-scatter==2.1.2+pt27cu126`. Put it first in `PYTHONPATH`, ahead of the
shared environment and `$WORK_ROOT/src`. A CPU functional preflight confirms
that this combination executes `radius_graph` and `segment_coo`. The
reserved-node preflight additionally executes `segment_coo` on CUDA before
loading any scientific samples.

Rebuild and validate it with:

```bash
bash scripts/setup_hip_h100_pyg_overlay.sh
```

## Frozen asset provenance

- GADplus code base before the uncommitted benchmark additions:
  `6e4ff47a15cae426e8efe8508777ea57ea2ae3b4` (`main`).
- `hip_v2.ckpt`: 232,915,328 bytes; SHA256
  `154d658f9c5d0b082a9c4893f3978038494d2499794a5ac647448fe397f2d1cb`.
- `transition1x.h5`: 6,623,929,000 bytes; SHA256
  `6a20f8a3f49c50d462270d10d4c44ca102e788072e2096a91d70b5a0f598b629`.

## Preflight and smoke

From `/scratch/memoozd/GAD/GAD_plus`:

```bash
source .venv/bin/activate
export PYG_OVERLAY=/scratch/memoozd/gadplus/envs/hip-pyg27-overlay/lib/python3.11/site-packages
export PYTHONPATH="$PYG_OVERLAY:$PWD/src"
python -m py_compile scripts/hip_cs2_benchmark.py
test -r /scratch/memoozd/GAD/models/hip_v2.ckpt
test -r /scratch/memoozd/GAD/data/transition1x.h5
```

For a manual diagnostic on any future reservation, SSH to its recorded
hostname, preserve the recorded `CUDA_VISIBLE_DEVICES`, and run a
single-sample H100 smoke directly on the reserved node:

```bash
cd /scratch/memoozd/GAD/GAD_plus
source .venv/bin/activate
export PYG_OVERLAY=/scratch/memoozd/gadplus/envs/hip-pyg27-overlay/lib/python3.11/site-packages
export PYTHONPATH="$PYG_OVERLAY:$PWD/src"
python scripts/hip_cs2_benchmark.py worker \
  --output-root /scratch/memoozd/gadplus/runs/hip-cs2-h100-smoke-pyg27 \
  --h5 /scratch/memoozd/GAD/data/transition1x.h5 \
  --checkpoint /scratch/memoozd/GAD/models/hip_v2.ckpt \
  --noise 0.15 --sample-ids 2 --n-samples 287 --seed 42 \
  --max-steps 2 --n-shards 1 --shard-id 0 --device cuda
```

Confirm that the sample JSON has `calculator_valid=true`, finite E/F/H
diagnostics, and either two completed updates or earlier strict convergence.

Environment preflight from `trig-login01` passed on 2026-08-09: the H100
nodes are `x86_64`; the shared environment reports PyTorch `2.7.0+cu126`,
CUDA `12.6`, PyG `2.8.0`, and imports HIP from
`/scratch/memoozd/GAD/hip/hip`. The CS² configuration instantiated with
`competitive_subspace`, `tau_s=0.01`, and `eta=0.01`.

### Run ledger

| Job | Purpose | Account/partition | State | Scientific evidence? |
|---:|---|---|---|---|
| 733685 | one-sample, two-update H100 runtime smoke at `0.15 A` | def-aspuru/debug | completed scheduler-side; calculator invalid because PyG 2.8 required unavailable `pyg-lib>=0.6.0` | no |
| 733698 | persistent one-H100 interactive reservation; default account assessed for more than three hours | def-aspuru/automatic | cancelled only after fallback began running | no |
| 733972 | persistent one-H100 reservation on `trig0041`, GPU 0, CPU affinity 88--95 | rrg-aspuru/automatic | ended before the campaign; four detached workers survived until detected and stopped | allocation only |
| 734620 | non-overlapping continuation reservation, `afterany:733972` | def-aspuru/automatic | no longer present when checked; it did not provide scientific output | allocation only |
| 736194 | replacement persistent one-H100 reservation on `trig0015` | def-aspuru/automatic | ran 3:41, then cancelled; automatic SSH launch failed before HIP because its non-login shell had no Lmod `module` function | allocation only |
| 750387 | second replacement persistent one-H100 reservation | def-aspuru/automatic | cancelled after a measured priority wait remained `StartTime=Unknown`; never allocated | allocation only |
| 750415 | fallback persistent one-H100 reservation | rrg-aspuru/automatic | disappeared from the active Trillium queue before allocation; no new sample file or IRC row was produced | allocation only |
| 750731 | replacement fallback persistent one-H100 reservation | rrg-aspuru/automatic | ran on `trig0042` for 24 hours; completed missing search rows, all-endpoint IRC, and checked aggregation before timeout | final production allocation |

The first reserved-node preflight root,
`hip-cs2-h100-production-20260809`, passed both large-asset checks but exposed
a missing `torch-scatter` binary operation: all four smoke samples failed
with `segment_sum_coo` unavailable. The smoke gate stopped the launcher before
production. No scientific trajectory from that root is counted. After adding
the exact PyTorch 2.7/CUDA 12.6 `torch-scatter` wheel and a CPU/CUDA operation
test, the source manifest necessarily changed; the failed root was preserved
and the campaign restarted in the fresh `-v2` root.

The `-v2` four-worker smoke passed `4/4` calculator-valid with zero errors.
It used the assigned H100 and reached high concurrent GPU utilization. The
production search therefore began under the frozen protocol in
`/scratch/memoozd/gadplus/runs/hip-cs2-h100-production-20260809-v2`.

At `119/287` atomic sample files in the held-out `0.20 A` cell, an allocation
audit found that jobs `733972` and `734620` were no longer present in Slurm
although the SSH-launched campaign still had four live GPU workers on
`trig0041`. Those workers and their detached launcher were terminated
explicitly; already atomically written sample JSON was retained, while any
in-flight sample without a completed JSON remains eligible for normal
`--resume` recomputation. Replacement keeper `736194` was submitted under
the default `def-aspuru` account. This operational interruption does not
change the held-out IDs, start-table hash, optimizer parameters, or primary
denominator. A fresh campaign must be launched only after the replacement
keeper is confirmed RUNNING. Use the outer allocation-lifetime supervisor
added after the audit; it owns the campaign session and terminates its full
process group when that exact keeper leaves RUNNING state. This supervisor is
operational only and is intentionally outside the already frozen scientific
code manifest.

Keeper `736194` subsequently started on `trig0015`, but the first supervised
resume did not enter scientific code: Trillium's non-login compute-node SSH
shell did not define Lmod's `module` function, and the frozen launcher stopped
at its first `module purge`. The campaign therefore remained at `119/287` in
the `0.20 A` search and zero IRC rows. The operational supervisor now invokes
the unchanged, manifest-locked launcher with `bash -l`; `bash -n`, a login-
shell `type module` check, and `git diff --check` passed before replacement
keeper `750387` was submitted under `def-aspuru`. This failed allocation
produced no scientific evidence and changes no denominator.

The restart audit also found that the original worker's `--resume` condition
reuses only calculator-valid JSON. Restarting the full frozen launcher would
therefore retry the already retained calculator-error rows in the completed
`0.10/0.15 A` cells and the partial `0.20 A` cell. Those failures are valid
production outcomes, not missing work. The operational missing-only driver
preserves every existing atomic sample JSON regardless of outcome and runs
the unchanged, hash-locked worker only for absent sample IDs. It assigns one
sample per logical shard (`287` shard manifests) and keeps four independent
processes resident concurrently, within the keeper's eight-core affinity.
The scientific protocol does not include execution shard count; each process
still reconstructs and checks the same complete held-out start table before
selecting its one assigned ID. IRC continuation uses the same missing-file
rule, so an allocation boundary cannot silently retry or replace a completed
IRC failure either. Aggregation still requires the exact `0..286`
partition and common protocol digest.

Job `750387` remained priority-pending with no estimated start during the
measured default-account assessment and was cancelled before allocation.
Following the predeclared account policy, exactly one fallback keeper,
`750415`, was then submitted under `rrg-aspuru`; Slurm supplied a projected
start of `2026-08-11 20:58 EDT`. The automatic waiter targets only this job
and the same immutable `-v2` root, so the account switch cannot create an
overlapping reservation or a second scientific campaign.

On the next audit, `750415` was absent from the active Trillium queue and had
not produced a keeper state file, additional `0.20 A` sample, or IRC row.
Because the default-account wait had already been measured, replacement
fallback keeper `750731` was submitted from `trig-login01` under
`rrg-aspuru`. It is the only active reservation. Its automatic waiter resumes
only absent atomic sample IDs in the same immutable `-v2` root, then performs
the already-declared aggregation and all-endpoint IRC campaign. The job uses
the persistent `sleep infinity` keeper and is cancelled only after those
outputs complete.

Keeper `750731` started on `trig0042` and completed the campaign before its
24-hour timeout. Final held-out strict counts at `0.10/0.15/0.20 A` are
`202/157/111` of 287; intended all-endpoint IRC_TOPO counts are
`218/166/117`. Plain GAD gives `209/167/128` strict and `225/177/128`
IRC_TOPO on the same cells. The completed result therefore rejects CS² as the
HIP default. Final checked outputs are `analysis.json` and `ANALYSIS.md` in
the immutable `-v2` run root.

The completed held-out `0.10 A` cell contains 11 calculator-invalid rows in
two formula/reaction classes: `C3H5NO2` samples 123, 128, and 153
(`rxn3104`, `rxn3109`, `rxn4499`), and `C5H5NO` samples 212, 218, 239, 248,
252, 267, 269, and 280 (`rxn6192`, `rxn6198`, `rxn7942`, `rxn7951`,
`rxn7955`, `rxn8833`, `rxn8835`, `rxn8885`). The HIP model logged
shrinking `edge_vec_0_distance` values down to exactly zero and then raised an
internal assertion. Each planned start had passed the common start-table
digest; the failures occurred during optimization and are retained in the
287-start denominator as atomic-collapse optimizer failures, not discarded as
missing data. Their clustering by formula and nearby reaction IDs makes the
pattern chemistry/geometry-specific rather than a random CUDA fault. Rerun
each exact sample in an isolated diagnostic root only after it cannot compete
with production; diagnostics must not replace the original production
outcomes.

## Run and aggregate on the reserved node

```bash
cd /scratch/memoozd/GAD/GAD_plus
nohup env KEEPER_JOB_ID=JOB_ID \
  RUN_ROOT=/scratch/memoozd/gadplus/runs/hip-cs2-h100-production-20260809-v2 \
  CAMPAIGN_LOG=/scratch/memoozd/gadplus/logs/hip_cs2_reserved_JOB_ID_v2.out \
  bash scripts/supervise_reserved_hip_campaign.sh </dev/null \
  > /scratch/memoozd/gadplus/logs/hip_cs2_supervisor_JOB_ID_v2.out 2>&1 &
echo $!
```

The launcher first reads
`/scratch/memoozd/gadplus/allocations/hip_h100_KEEPER_JOB_ID.json`, refuses a
hostname mismatch, and exports the allocation's recorded
`CUDA_VISIBLE_DEVICES`; this is required because a separate SSH shell does
not inherit the Slurm job environment. It also restricts the SSH-launched
process group to the keeper's recorded CPU affinity. The outer supervisor
starts that launcher in a new `setsid` process group and independently
terminates the complete group if the keeper leaves RUNNING state, so work
cannot survive into another user's allocation. The persistent-node campaign uses four
disjoint workers per noise cell, with two CPU threads and one 233 MB HIP
checkpoint copy each. This stays within the keeper's eight allocated CPU
cores while improving H100 occupancy; workers never share a trajectory or
sample file. Each sample JSON is an independent restart boundary. `--resume`
reuses only calculator-valid sample files with the exact matching protocol
digest; failed, incompatible, or interrupted samples are recomputed. The
separate array template retains twelve shards per noise cell for independently
scheduled jobs.

Before production, the launcher automatically runs samples `0,1,2,3` for two
updates through all four concurrent workers, aggregates the smoke cell, and
requires `4/4` calculator-valid outcomes with zero errors. The preserved
artifacts live under `preflight_four_worker/` and never enter production
denominators.

To aggregate explicitly if needed:

```bash
RUN_ROOT=/scratch/memoozd/gadplus/runs/hip-cs2-ARRAY_JOB_ID
N_SHARDS=4  # use 12 only for output produced by the array template
for noise in 0.10 0.15 0.20; do
  PYG_OVERLAY=/scratch/memoozd/gadplus/envs/hip-pyg27-overlay/lib/python3.11/site-packages
  PYTHONPATH=$PYG_OVERLAY:$PWD/src .venv/bin/python scripts/hip_cs2_benchmark.py aggregate \
    --output-root "$RUN_ROOT" \
    --h5 /scratch/memoozd/GAD/data/transition1x.h5 \
    --checkpoint /scratch/memoozd/GAD/models/hip_v2.ckpt \
    --noise "$noise" --seed 42 --n-samples 287 --max-steps 5000 \
    --step-fraction 0.01 --spectral-temperature 0.01 \
    --n-shards "$N_SHARDS" --expected-shards "$N_SHARDS" --device cuda
done
```

Each noise directory contains immutable per-sample JSON, shard manifests,
`protocol.json`, `results.csv`, and `summary.json`. The protocol records the
full selected sample set, asset paths/sizes/expected SHA256 values, optimizer
parameters, verified start-table digest, and runner SHA256. Before loading
HIP, the launcher checks both large assets byte-for-byte and locks hashes of
all scientific Python sources in top-level `code_manifest.json`; a resumed
run refuses source drift. Per-keeper host/GPU/time records are stored under
`allocations/`. Workers reuse a completed sample only when its protocol digest
matches; aggregation rejects mixed protocols. The summary reports terminal
index classes and median evaluations, steps, and wall time. Preserve failed
samples and their error messages; never silently shrink the denominator.

## Required comparisons

Compare strict recovery directly with the completed historical values:

| Noise (A) | Best plain GAD | Best completed Sella |
|---:|---:|---:|
| 0.10 | 72.8% (`dt=0.007`) | 72.8% |
| 0.15 | 58.2% (`dt=0.007`) | 54.0% |
| 0.20 | 44.6% (`dt=0.007`) | 27.2% |

The first two frozen CS² cells are complete and checked:

| Noise (A) | Planned | Valid | CS² strict / planned | Strict / valid | Median evaluations | Terminal classes |
|---:|---:|---:|---:|---:|---:|---|
| 0.10 | 287 | 276 | 202/287 (70.4%) | 202/276 (73.2%) | 17 | 202 strict, 54 index-one/force-limited, 19 multi-negative, 1 index-zero, 11 calculator errors |
| 0.15 | 287 | 262 | 157/287 (54.7%) | 157/262 (59.9%) | 26 | 157 strict, 70 index-one/force-limited, 34 multi-negative, 1 index-zero, 25 calculator errors |

The primary planned-denominator result is seven recoveries below both
historical plain GAD and Sella (`209/287`, 72.8%), a difference of `-2.4`
percentage points. Therefore CS² does not improve local strict recovery at
`0.10 A`; its valid-only percentage must not be used to hide the 11 optimizer
collapse failures. The four shard manifests cover `72+72+72+71=287` rows,
all sample protocol digests equal
`a219020aff6c6119058698dcde779a8b0d8cfa38b2f41f20001f2f193a4a69a3`,
and the selected held-out IDs are exactly `0..286`. The launcher then moved
without intervention to the frozen `0.15 A` cell.

The completed held-out `0.15 A` cell is ten strict recoveries below the best
plain-GAD result (`167/287`, 58.2%), a difference of `-3.5` percentage
points, and two above the Sella comparator (`155/287`, 54.0%), a difference
of `+0.7` percentage points. Thus CS² again does not improve on plain GAD;
its small count advantage over Sella does not make Sella an operational
optimizer option. The four shard manifests cover `72+72+72+71=287` rows,
all row protocol digests equal
`b2401fd3649415cb2d5f7afbc41657813f2f5a91f560810dda8df29daca7bc40`,
and the selected test IDs are exactly `0..286`.

The 25 retained `0.15 A` atomic-collapse rows are sample IDs `0`, `4`, `8`,
`42`, `62`, `107`, `110`, `123`, `128`, `153`, `198`, `212`, `218`, `226`,
`227`, `239`, `248`, `252`, `257`, `267`, `269`, `277`, `279`, `280`, and
`283`. They span the same clustered formula families seen at `0.10 A`, plus
the lowest-ID `C2H3N3O2`/`C2H5NO2` cases. They remain optimizer failures in
the planned denominator. After the checked aggregate was written, the
launcher moved automatically to the frozen held-out `0.20 A` cell with start
hash `fdb17613f0d30c8ef6db96def7a06b812762fccaa9615c93907a1c3e393ba156`.

The CS² run uses a different pointwise update and evaluation cost, so report
both success and median HIP evaluations. Do not describe a gain as an
ordinary-GAD timestep improvement.

Before interpreting a change mechanistically, classify CS² and plain-GAD
failures by terminal `n_neg`: index zero, index one/force-limited, or
multi-negative. CS² has specific support only if it reduces minimum capture
or improves recovery in nearly degenerate soft subspaces without trading that
gain for fragmentation, flat saddles, or loss of intended connectivity.

The reserved-node campaign then runs the maintained HIP-Hessian Sella IRC for
500 steps per direction from all 287 terminal geometries, matching the
historical all-endpoint validation design. Report intended `IRC_TOPO` over all
planned starts and also within the locally strict subset. Keep local strict
recovery and chemical endpoint recovery as separate tables; an IRC success
does not retroactively change the local `n_neg/fmax` count.

After all three search and IRC cells aggregate, generate the checked report:

```bash
.venv/bin/python scripts/summarize_hip_cs2_results.py \
  /scratch/memoozd/gadplus/runs/hip-cs2-h100-production-20260809-v2
```

This refuses incomplete 287-start denominators or mixed protocol digests and
writes `analysis.json` plus `ANALYSIS.md` with historical count-level
comparisons, Wilson intervals, terminal-index taxonomy, and evaluation tails.

## Documentation after completion

Add the job IDs, cluster/GPU type, code commit, checkpoint checksum, HDF5
checksum, exact valid denominators, strict table, terminal-index taxonomy,
evaluation-cost table, and IRC results to this document. Then update
`HIP_GAD_SELLA_SYNTHESIS_2026_07_17.md`, `BENCHMARK_RESULTS_2026_07_16.md`,
and the CS² section of `POINTWISE_INTRINSIC_GAD.md`. A negative or tied result
is still complete and should be recorded without retuning the test cells.
