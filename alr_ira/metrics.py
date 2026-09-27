"""How the paper aggregates acceptance length and speedup (Section 4 and Appendix H)."""

from __future__ import annotations

import math
from statistics import mean, stdev


def mat(committed, rounds) -> float:
    """Mean acceptance length tau, pooled over verification rounds.

    `committed` counts accepted draft tokens plus the one target token of each round.
    """
    total = sum(rounds)
    if not total:
        raise ValueError("no verification rounds")
    return sum(committed) / total


def geomean(values) -> float:
    values = list(values)
    if not values or any(v <= 0 for v in values):
        raise ValueError("a geometric mean needs positive values")
    return math.exp(sum(math.log(v) for v in values) / len(values))


def aggregate(per_seed):
    """Geometric mean across tasks within each training seed, then mean and sample SD across seeds.

    `per_seed` is a list with one list of task values per seed.
    Returns (mean, sd), with sd None for a single seed.
    """
    values = [geomean(tasks) for tasks in per_seed]
    return mean(values), (stdev(values) if len(values) > 1 else None)


def speedup(ar_ms, draft_ms) -> float:
    """Speedup of one domain, the mean AR milliseconds per token over the drafter's."""
    return mean(ar_ms) / mean(draft_ms)


def gain(new: float, reference: float) -> float:
    """Relative difference in percent."""
    return (new / reference - 1.0) * 100.0
