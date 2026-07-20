# Experiment Organization And Storage

Last updated: 2026-07-16

## Default Policy

Work almost exclusively in scratch.

Project space is reserved for:

- Source code and Git history.
- Research plans and one small JSON record per experiment.
- Compact, paper-critical summary tables.
- Small final figures and manuscripts.

Scratch is the default location for:

- Python environments.
- Package, model, and dataset caches.
- Downloaded model checkpoints and working datasets.
- Slurm logs and temporary files.
- Active experiment outputs, trajectories, and intermediate analyses.
- Cloud-upload staging.

Recommended paths are:

```text
/lustre07/scratch/memoozd/gadplus/envs/
/lustre07/scratch/memoozd/gadplus/models/
/lustre07/scratch/memoozd/gadplus/cache/uv/
/lustre07/scratch/memoozd/gadplus/spool/<experiment-id>/
/lustre07/scratch/memoozd/gadplus/runs/<experiment-id>/
```

Inside Slurm jobs, use `$SLURM_TMPDIR` for high-churn temporary files and copy
only required outputs back to the scratch experiment directory.

Scratch is not durable storage. A completed or important experiment must be
uploaded to cloud object storage before it is considered safely retained.

## Minimal Experiment Record

Each experiment has one Git-tracked file:

```text
experiments/<experiment-id>.json
```

It records only the information needed to make the experiment understandable
and actionable:

- Question and hypothesis.
- Difference from its parent experiment.
- Design and fixed decision rule.
- Pilot or full-run budget.
- Status and Slurm job IDs.
- Automatically captured code, environment, model, and dataset provenance.
- Headline, caveats, and next action after analysis.

Large files never enter Git. They are staged in scratch and uploaded under:

```text
s3://gadplus-research/experiments/<experiment-id>/
```

The cloud directory contains the resolved configuration, compact summary
Parquet, task-level trajectory Parquets, logs, figures, provenance hashes, and
a final `COMPLETE` marker.

## Agent Loop

Every agent loop should follow the same short sequence:

1. Read active experiment records and select the highest-value `next_action`.
2. Create a child experiment record when changing a scientific assumption.
3. Run implementation and energy/force/Hessian smokes.
4. Run a small pilot and apply the written decision rule.
5. Expand, stop, or invalidate the experiment.
6. Upload results from the login node and verify checksums.
7. Record the headline, caveats, and next action.

Agents should not reconstruct experiment state from chat history, directory
names, or Slurm output.

## File-Count Discipline

The `rrg-aspuru` project is constrained by inode count, not bytes. Therefore:

- Do not create environments or package caches in project space.
- Do not create local MLflow directory trees in project space.
- Do not write one cloud-staging file per molecular sample when one Parquet per
  Slurm task can contain the same data.
- Each Slurm task writes its own files; parallel tasks never append to a shared
  file.
- Compact task outputs at experiment completion.
- Keep nearline archives as a small number of compressed bundles, not expanded
  trees.

## Initial Project Cleanup Audit

Before broad new experiments, recover enough `rrg-aspuru` inodes to work
comfortably. The read-only audit on 2026-07-16 found:

| Path | Inodes | Size | Action |
|---|---:|---:|---|
| `GAD_plus/.venv` | 55,243 | 7.3 GB | Migrate to scratch, validate, then delete project copy |
| `GAD_plus/.git` | 2,316 | 175 MB | Keep |
| `GAD_plus/figures` | 275 | 28 MB | Keep; too few files to matter |
| Remaining source/results directories | approximately 1,000 | less than 100 MB excluding papers | Keep |

The complete user directory occupies about 59,179 inodes. The project-local
virtual environment is therefore the only high-leverage cleanup target found
under this user's project directory. Removing its project copy should recover
roughly 55,000 shared-project inodes.

### Safe Migration Sequence

No deletion occurs during the audit. Use this sequence:

1. Capture `uv pip freeze`, Python/CUDA details, and key package versions from
   the current environment.
2. Copy the exact `.venv` tree to a versioned scratch path.
3. Compare file counts and sizes between source and destination.
4. Temporarily expose the scratch environment through the existing `.venv`
   path using a symlink, preserving script shebangs and current commands.
5. Run core imports, HIP/PaiNN calculator loading, and the standard smoke tests.
6. Keep the project copy as a renamed backup until those checks pass.
7. Present the exact backup path as the deletion manifest.
8. Delete it only after explicit approval, then rerun `diskusage_report`.

There is currently no `uv.lock`, and the combined HIP/MACE dependencies do not
resolve cleanly. The existing project environment must not be deleted until
the scratch copy has passed these checks. Longer term, each incompatible MLIP
stack should have its own reproducible `uv` environment in scratch.

## Cloud Synchronization

Narval compute nodes cannot access the internet. Jobs write only to scratch.
After jobs finish, a login-node command uploads new files, verifies checksums,
and writes the cloud `COMPLETE` marker. Cloud credentials remain off compute
nodes.

The current command surface stays small:

```text
python scripts/gadlog.py list
python scripts/gadlog.py validate <id>
python scripts/gadlog.py attach-job <id> <slurm-id>
python scripts/gadlog.py status <id>
```

Creation, closure, and cloud upload remain explicit JSON and login-node steps
until an actual object-store bucket and credentials are configured. No database
or tracking server is required. Git is the research index; cloud object
storage is the durable artifact store; scratch is the working area.
