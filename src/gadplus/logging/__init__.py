"""Trajectory logging, MLflow integration, failure autopsy."""

from gadplus.logging.schema import TRAJECTORY_SCHEMA, SUMMARY_SCHEMA
from gadplus.logging.trajectory import TrajectoryLogger
from gadplus.logging.autopsy import FailureType, classify_failure

# MLflow imports are lazy — mlflow may not be installed
try:
    from gadplus.logging.mlflow_logger import (
        setup_mlflow,
        log_run_params,
        log_run_metrics,
        log_artifact,
    )
except ImportError:
    pass

__all__ = [
    "TRAJECTORY_SCHEMA",
    "SUMMARY_SCHEMA",
    "TrajectoryLogger",
    "FailureType",
    "classify_failure",
]
