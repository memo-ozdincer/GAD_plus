"""Trajectory logging with lazy optional integrations."""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "FailureType": ("gadplus.logging.autopsy", "FailureType"),
    "IntrinsicTrajectoryRecorder": (
        "gadplus.logging.pointwise",
        "IntrinsicTrajectoryRecorder",
    ),
    "SUMMARY_SCHEMA": ("gadplus.logging.schema", "SUMMARY_SCHEMA"),
    "TRAJECTORY_SCHEMA": ("gadplus.logging.schema", "TRAJECTORY_SCHEMA"),
    "TrajectoryLogger": ("gadplus.logging.trajectory", "TrajectoryLogger"),
    "classify_failure": ("gadplus.logging.autopsy", "classify_failure"),
}

__all__ = [
    "SUMMARY_SCHEMA",
    "TRAJECTORY_SCHEMA",
    "FailureType",
    "IntrinsicTrajectoryRecorder",
    "TrajectoryLogger",
    "classify_failure",
]


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module_name, attribute = _EXPORTS[name]
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value
