"""Path helpers for cluster-local data and checkpoints."""
from __future__ import annotations

import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def project_dir() -> Path:
    """Return the directory containing local models/data for this checkout."""
    return Path(os.environ.get("GADPLUS_PROJECT_DIR", REPO_ROOT)).expanduser().resolve()


def scratch_dir() -> Path:
    """Return the writable run-output directory root."""
    return Path(os.environ.get("GADPLUS_SCRATCH_DIR", project_dir())).expanduser().resolve()


def _first_existing(candidates: list[Path], label: str) -> Path:
    for path in candidates:
        if path.is_file() and path.stat().st_size > 0:
            return path
    checked = "\n  ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"{label} not found. Checked:\n  {checked}")


def hip_checkpoint_path() -> Path:
    """Resolve the HIP checkpoint, preferring the current public HIP v3 model."""
    env_path = os.environ.get("GADPLUS_HIP_CHECKPOINT")
    if env_path:
        return Path(env_path).expanduser().resolve()

    root = project_dir()
    return _first_existing(
        [
            root / "models" / "hip_v3.ckpt",
            root / "models" / "hip_v2.ckpt",
            Path("/lustre06/project/6033559/memoozd/models/hip_v3.ckpt"),
            Path("/lustre06/project/6033559/memoozd/models/hip_v2.ckpt"),
            Path("/project/rrg-aspuru/memoozd/models/hip_v3.ckpt"),
            Path("/project/rrg-aspuru/memoozd/models/hip_v2.ckpt"),
        ],
        "HIP checkpoint",
    )


def transition1x_h5_path() -> Path:
    """Resolve the Transition1x HDF5 file."""
    env_path = os.environ.get("GADPLUS_T1X_H5")
    if env_path:
        return Path(env_path).expanduser().resolve()

    root = project_dir()
    return _first_existing(
        [
            root / "data" / "transition1x.h5",
            root / "data" / "Transition1x.h5",
            Path("/lustre06/project/6033559/memoozd/data/transition1x.h5"),
            Path("/project/rrg-aspuru/memoozd/data/transition1x.h5"),
        ],
        "Transition1x HDF5",
    )
