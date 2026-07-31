# Trillium migration checklist

This repository is being migrated from Narval to Trillium.  The migration is
complete only after the checks below pass on Trillium; copying the repository
alone is not a validated migration.

## What to transfer

Transfer Git-tracked source plus the required external assets.  Do not copy
Narval virtual environments, `outputs/`, `runs/`, or a compiled g-xTB binary.

| Asset | Target convention |
| --- | --- |
| Repository | `/project/rrg-aspuru/memoozd/GAD_plus` |
| HIP checkout/checkpoint | `/project/rrg-aspuru/memoozd/hip`, `models/` |
| Transition1x checkout/HDF5 | `/project/rrg-aspuru/memoozd/transition1x`, `data/` |
| g-xTB source/release binary | `/project/rrg-aspuru/memoozd/third_party/g-xtb` |
| Environments and outputs | `/scratch/memoozd/gadplus/` |

Keep the g-xTB clone outside the Git worktree and export its executable path
through `GADPLUS_GXTB_EXE`.

## First login

The current Narval account does not authenticate to Trillium with its loaded
SSH key.  Configure the Alliance SSH key/account first, then verify:

```bash
ssh trillium.alliancecan.ca hostname
```

On Trillium:

```bash
cd /project/rrg-aspuru/memoozd/GAD_plus
CLUSTER=trillium bash scripts/setup_env.sh
```

The environment must be created on Trillium, not copied from Narval.  Verify
the exact install with the pre-flight checks in [OPERATIONS.md](OPERATIONS.md).

## Transfer helper

From an authenticated Narval or workstation login, run:

```bash
./scripts/migrate_to_trillium.sh
```

It copies only source/provenance and deliberately skips environments,
generated runs, figures, and external scientific assets.  To copy the HIP,
Transition1x, and g-xTB assets as well, set their source paths and use
`MIGRATE_EXTERNAL=1`; the script is non-destructive and never uses
`rsync --delete`.

## CPU g-xTB jobs

g-xTB numerical/full Hessian work is CPU parallel.  H100 GPUs are not the
accelerator for this path.  Start with a 32-core CPU allocation on Trillium
and use `GADPLUS_GXTB_PARALLEL=32`; increase only after a one-Hessian scaling
test.  One trajectory per allocation avoids nested parallel oversubscription.

```bash
export GADPLUS_T1X_H5=/project/rrg-aspuru/memoozd/data/transition1x.h5
export GADPLUS_GXTB_EXE=/project/rrg-aspuru/memoozd/third_party/g-xtb/xtb-6.7.1/bin/xtb
export GADPLUS_GXTB_PARALLEL=32
```

Copy the three g-xTB Slurm scripts to a Trillium-specific campaign directory
or override their paths/resources at submission.  Before a large campaign,
run a two-step GAD and Sella pilot, validate the n_neg/fmax summaries, then
run IRC_TOPO only for gate-passing candidates.

## GPU/HIP jobs

Hydra cluster configuration is portable:

```bash
python -m gadplus.orchestration.run cluster=trillium search=gad_projected max_samples=1
```

The `cluster=trillium` configuration supplies Trillium paths and module
versions.  It is appropriate for HIP/MACE/HORM GPU calculations.  Do not
request an H100 for a g-xTB-only campaign unless that allocation is required
for a separate GPU surface.

## Migration acceptance record

Record the following in the first Trillium campaign README:

- `hostname`, Slurm job ID, allocation, module list;
- Python, Torch, Sella, and g-xTB versions;
- direct g-xTB energy/gradient/Hessian smoke result;
- one GAD and one Sella two-step result, including Hessian-call count;
- projected `n_neg`/`fmax` gate count;
- IRC_TOPO result for every TS-gate-passing row.
