"""Eigenvector mode tracking across optimization steps.

When eigenvalues are nearly degenerate, torch.linalg.eigh can swap the
ordering of eigenvectors between steps. Mode tracking maintains continuity
by selecting the eigenvector with maximum overlap to the previous step's
tracked mode, and enforcing consistent sign.
"""
from __future__ import annotations

import torch


def pick_tracked_mode(
    evecs: torch.Tensor,
    v_prev: torch.Tensor | None,
    *,
    k: int = 8,
) -> tuple[torch.Tensor, int, float]:
    """Pick the eigenvector most aligned with the previous tracked mode.

    Among the lowest `k` eigenvectors, selects the one with maximum
    |dot(v_prev, v_i)| and enforces sign continuity.

    Args:
        evecs: (D, M) eigenvectors from torch.linalg.eigh (columns).
        v_prev: (D,) previous tracked mode, or None for first step.
        k: Number of candidate eigenvectors to consider.

    Returns:
        v: (D,) normalized tracked eigenvector.
        j: Index of the selected candidate (0..k-1).
        overlap: |dot(v_prev, v)| before sign correction (1.0 if v_prev is None).
    """
    if v_prev is None or k == 0:
        v0 = evecs[:, 0]
        if v_prev is not None and torch.dot(v0, v_prev.reshape(-1)) < 0:
            v0 = -v0
        v0 = v0 / (v0.norm() + 1e-12)
        overlap = float(torch.abs(torch.dot(v0, v_prev.reshape(-1))).item()) if v_prev is not None else 1.0
        return v0, 0, overlap

    k_eff = int(min(int(k), int(evecs.shape[1])))
    V = evecs[:, :k_eff]

    v_prev = v_prev.reshape(-1)
    overlaps = torch.abs(V.transpose(0, 1) @ v_prev)
    j = int(torch.argmax(overlaps).item())

    v = V[:, j]
    overlap = float(overlaps[j].item())

    if torch.dot(v, v_prev) < 0:
        v = -v

    v = v / (v.norm() + 1e-12)
    return v, j, overlap
