# GADplus

GADplus runs transition-state searches with HIP Hessian predictions and the
Transition1x dataset.

## Installation With uv

This checkout is managed by `uv`. The lockfile is the source of truth: use
`uv lock` after dependency/source changes. GADplus uses the project virtual
environment at `.venv`; run commands through `uv run` from the repo root so the
Slurm wrapper, local commands, and dependency lock all use the same environment.

Create local asset directories:

```bash
cd /lustre/fs12/portfolios/nvr/projects/nvr_qualg_lmbm/users/anburger/GAD_plus && mkdir -p external models data
```

Clone the local path dependencies used by `pyproject.toml`:

```bash
cd /lustre/fs12/portfolios/nvr/projects/nvr_qualg_lmbm/users/anburger/GAD_plus && test -d external/hip || git clone https://github.com/BurgerAndreas/hip.git external/hip && test -d external/Transition1x || git clone https://gitlab.com/matschreiner/Transition1x.git external/Transition1x
```

Sync the project environment. Do not use plain `pip`; keep dependencies in
`pyproject.toml` and `uv.lock`.

```bash
cd /lustre/fs12/portfolios/nvr/projects/nvr_qualg_lmbm/users/anburger/GAD_plus && uv lock && uv sync --extra analysis
```

Download the HIP v3 checkpoint and config:

```bash
cd /lustre/fs12/portfolios/nvr/projects/nvr_qualg_lmbm/users/anburger/GAD_plus && wget -O models/hip_v3.ckpt https://huggingface.co/andreasburger/hip/resolve/main/ckpt/hip_v3.ckpt && wget -O models/hip_v3.yaml https://huggingface.co/andreasburger/hip/resolve/main/ckpt/hip_v3.yaml
```

Download Transition1x and place it at `data/transition1x.h5`. Use the direct `ndownloader.figshare.com` host, not `figshare.com/ndownloader`, because the latter may return an AWS WAF challenge and save a zero-byte file on this cluster.

```bash
cd /lustre/fs12/portfolios/nvr/projects/nvr_qualg_lmbm/users/anburger/GAD_plus && uv run python scripts/download_transition1x.py --output data/transition1x.h5
```

Verify the uv environment, local packages, and HIP checkpoint:

```bash
cd /lustre/fs12/portfolios/nvr/projects/nvr_qualg_lmbm/users/anburger/GAD_plus && uv lock --check && source scripts/setup_cuda.sh && uv run python -c "import gadplus, hip, transition1x, torch; from gadplus.calculator.hip import load_hip_calculator; from gadplus.paths import hip_checkpoint_path; c=load_hip_calculator(str(hip_checkpoint_path()), device='cpu'); print(torch.__version__, type(c).__name__)"
```

Paths can be overridden with `GADPLUS_PROJECT_DIR`, `GADPLUS_SCRATCH_DIR`, `GADPLUS_HIP_CHECKPOINT`, and `GADPLUS_T1X_H5`.

## Runners

Regular GAD: `scripts/method_single.py`. Recommended method:
`gad_dt003_fmax` for canonical Eckart GAD with `fmax < 0.01`.

```bash
cd /lustre/fs12/portfolios/nvr/projects/nvr_qualg_lmbm/users/anburger/GAD_plus && sbatch scripts/run_batch_singlenode_uv.sbatch python -u scripts/method_single.py --method gad_dt003_fmax --noise 0.01 --n-samples 287 --n-steps 5000 --split test --output-dir runs/test_dtgrid/gad_dt003_fmax
```

Hybrid GAD + Newton: `scripts/hybrid_gad_newton_runner.py`. Recommended method:
`hybrid_damped_eckart` with force-based switching unless you are explicitly
testing eigenvalue-based switching.

```bash
cd /lustre/fs12/portfolios/nvr/projects/nvr_qualg_lmbm/users/anburger/GAD_plus && sbatch scripts/run_batch_singlenode_uv.sbatch python -u scripts/hybrid_gad_newton_runner.py --method hybrid_damped_eckart --switch-by-eig false --gad-dt 5e-3 --trust-radius 0.01 --noise 0.01 --n-samples 287 --n-steps 1000 --split test --output-dir runs/hybrid_gad_newton/hybrid_damped_eckart_swFORCE_dt5e-3_tr0.01
```
