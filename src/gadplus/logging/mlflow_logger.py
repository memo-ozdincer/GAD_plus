"""Thin wrapper for offline MLflow tracking.

All data is stored locally under ``output_dir/mlruns/`` so no server is
required.  Import and call ``setup_mlflow`` at the start of a run, then use
the helper functions to log parameters, metrics, and artifacts.

Usage::

    from gadplus.logging.mlflow_logger import setup_mlflow, log_run_params, log_run_metrics, log_artifact

    setup_mlflow("/scratch/runs/my_experiment")
    with mlflow.start_run(run_name="sample_42"):
        log_run_params({"dt": 0.01, "method": "gad"})
        for step in range(max_steps):
            ...
            log_run_metrics({"energy": E, "n_neg": n}, step=step)
        log_artifact("traj_abc_42.parquet", artifact_path="trajectories")
"""
from __future__ import annotations

import os

import mlflow


def setup_mlflow(
    output_dir: str,
    experiment_name: str = "gadplus",
) -> str:
    """Configure MLflow for offline file-based tracking.

    Parameters
    ----------
    output_dir : str
        Root directory for this experiment.  An ``mlruns/`` subdirectory is
        created automatically.
    experiment_name : str
        MLflow experiment name.

    Returns
    -------
    str
        The tracking URI that was set.
    """
    uri = f"file://{os.path.abspath(os.path.join(output_dir, 'mlruns'))}"
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment_name)
    return uri


def log_run_params(params: dict) -> None:
    """Log a dict of parameters to the active MLflow run.

    Long string values are truncated to 500 chars (MLflow limit).
    """
    truncated = {
        k: (str(v)[:500] if isinstance(v, str) and len(str(v)) > 500 else v)
        for k, v in params.items()
    }
    mlflow.log_params(truncated)


def log_run_metrics(metrics: dict, step: int | None = None) -> None:
    """Log a dict of numeric metrics to the active MLflow run.

    Non-finite values (NaN, inf) are silently skipped.
    """
    import math

    clean = {
        k: v for k, v in metrics.items()
        if isinstance(v, (int, float)) and math.isfinite(v)
    }
    mlflow.log_metrics(clean, step=step)


def log_artifact(path: str, artifact_path: str | None = None) -> None:
    """Log a local file as an MLflow artifact."""
    mlflow.log_artifact(path, artifact_path=artifact_path)
