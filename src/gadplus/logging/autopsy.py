"""Post-mortem failure classification for non-converged trajectories.

Given a list of per-step dicts (the same rows stored by TrajectoryLogger),
``classify_failure`` returns one of six ``FailureType`` categories.  The
classification is checked in priority order: the first matching condition wins.

Usage::

    from gadplus.logging.autopsy import classify_failure
    failure = classify_failure(logger.rows)
    print(failure)  # FailureType.GHOST_MODES
"""
from __future__ import annotations

from enum import Enum
from typing import List


class FailureType(Enum):
    """Mutually-exclusive failure categories for non-converged runs.

    GHOST_MODES      -- All negative eigenvalues fall in [-1e-4, 0); they are
                        numerical ghosts from imperfect Eckart projection, not
                        real saddle character.
    ALMOST_CONVERGED -- Very close to TS: n_neg <= 2 and forces are small.
                        Likely needs a few more NR steps or tighter dt.
    OSCILLATING      -- The eigenvalue spectrum flips back and forth without
                        monotonic progress (n_neg oscillates).
    ENERGY_PLATEAU   -- Energy is stagnant: the optimizer is stuck on a flat
                        region of the PES.
    GENUINELY_STUCK  -- n_neg is frozen at a non-1 value for the majority of
                        the trajectory.
    DRIFTING         -- Metrics are slowly improving but the run timed out
                        before convergence.
    """
    GHOST_MODES = "ghost_modes"
    ALMOST_CONVERGED = "almost_converged"
    OSCILLATING = "oscillating"
    ENERGY_PLATEAU = "energy_plateau"
    GENUINELY_STUCK = "genuinely_stuck"
    DRIFTING = "drifting"


def classify_failure(trajectory: List[dict]) -> FailureType:
    """Classify why a trajectory failed to converge.

    Parameters
    ----------
    trajectory : list[dict]
        Per-step rows, each containing at minimum the keys ``n_neg``,
        ``eig0``, ``force_norm``, and ``energy``.

    Returns
    -------
    FailureType
        The most specific failure category that matches.
    """
    if not trajectory:
        return FailureType.GENUINELY_STUCK

    # Work with the tail of the trajectory (last 100 steps or all if shorter)
    tail = trajectory[-100:]
    n = len(tail)

    n_negs = [row["n_neg"] for row in tail]
    eig0s = [row["eig0"] for row in tail]
    force_norms = [row["force_norm"] for row in tail]
    energies = [row["energy"] for row in tail]

    last = tail[-1]

    # ── 1. Ghost modes ───────────────────────────────────────────────
    # If the final step has n_neg >= 1 but ALL negative eigenvalues are
    # in the ghost band [-1e-4, 0), the negatives are numerical noise.
    if last["n_neg"] >= 1 and last["eig0"] > -1e-4:
        # Check that no negative eigenvalue is actually significant
        # by looking at the cascade: n_neg at threshold 1e-4 should be 0
        n_neg_1e4 = last.get("n_neg_1e4", None)
        if n_neg_1e4 is not None and n_neg_1e4 == 0:
            return FailureType.GHOST_MODES
        # Fallback: if cascade data isn't available, use eig0 alone
        if n_neg_1e4 is None and last["eig0"] > -1e-4:
            return FailureType.GHOST_MODES

    # ── 2. Almost converged ──────────────────────────────────────────
    # n_neg is 1 or 2 and forces are close to threshold
    if last["n_neg"] <= 2 and last["force_norm"] < 0.05:
        return FailureType.ALMOST_CONVERGED

    # ── 3. Oscillating ───────────────────────────────────────────────
    # Count sign changes in n_neg over the tail.  If n_neg bounces
    # frequently (> 30% of steps change from previous), it's oscillating.
    if n >= 10:
        changes = sum(
            1 for i in range(1, len(n_negs)) if n_negs[i] != n_negs[i - 1]
        )
        change_rate = changes / (len(n_negs) - 1)
        if change_rate > 0.30:
            return FailureType.OSCILLATING

    # ── 4. Genuinely stuck ───────────────────────────────────────────
    # The most common n_neg value accounts for >50% of the tail AND
    # that value is not 1 (otherwise it would have converged or be
    # almost_converged).  Checked before energy plateau because a frozen
    # wrong n_neg is more informative than flat energy.
    if n >= 10:
        from collections import Counter
        counter = Counter(n_negs)
        most_common_n_neg, most_common_count = counter.most_common(1)[0]
        if most_common_count / n > 0.50 and most_common_n_neg != 1:
            return FailureType.GENUINELY_STUCK

    # ── 5. Energy plateau ────────────────────────────────────────────
    # Energy range in the tail is tiny relative to the absolute energy
    if n >= 10:
        e_min = min(energies)
        e_max = max(energies)
        e_range = abs(e_max - e_min)
        # Plateau: energy changes by less than 1e-6 eV over the tail
        if e_range < 1e-6:
            return FailureType.ENERGY_PLATEAU

    # ── 6. Drifting ──────────────────────────────────────────────────
    # Default: some progress is being made but convergence wasn't reached.
    return FailureType.DRIFTING
