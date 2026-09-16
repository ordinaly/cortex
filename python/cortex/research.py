"""Small research helpers; experiment orchestration deliberately stays in Python."""
from __future__ import annotations
from dataclasses import dataclass
from time import perf_counter_ns
from typing import Iterable, Sequence

from . import Cortex

@dataclass(frozen=True)
class RunSummary:
    steps: int
    mean_us_per_step: float
    brier: float
    stored: int


def replay(model: Cortex, xs: Iterable[Sequence[float]], ys: Iterable[float]) -> RunSummary:
    n = 0
    se = 0.0
    t0 = perf_counter_ns()
    for x, y in zip(xs, ys):
        r = model.step(list(map(float, x)), float(y))
        se += (float(y) - r.prediction) ** 2
        n += 1
    elapsed_us = (perf_counter_ns() - t0) / 1000.0
    return RunSummary(n, elapsed_us / max(1, n), se / max(1, n), model.stored)
