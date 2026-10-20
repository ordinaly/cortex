"""Research-only learner for predictive distinguishability operators.

The learner sees only opaque contexts, declared action probes and observed output
words.  It never imports the finite SL-Bench oracle or hidden transition state.
The exact oracle belongs to evaluation only.

This first version learns the response of a fixed family of trace probes.  Probe
selection/topology discovery are later stages.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from math import isfinite, log, sqrt
from typing import Hashable, Iterable

from .five_operator_loop import DistinctionBank, DistinctionNeuron
from .trace_memory import TracePredictiveMemory


EPS = 1.0e-12


@dataclass(frozen=True, slots=True)
class LearnedDistinctionPolicy:
    epsilon: float = 0.25
    min_samples: int = 64
    delta: float = 0.05
    confidence: bool = True

    def __post_init__(self) -> None:
        if not isfinite(float(self.epsilon)) or not 0.0 <= self.epsilon <= 1.0:
            raise ValueError("epsilon must be finite and lie in [0,1]")
        if type(self.min_samples) is not int or self.min_samples < 1:
            raise ValueError("min_samples must be a positive integer")
        if not isfinite(float(self.delta)) or not 0.0 < self.delta < 1.0:
            raise ValueError("delta must lie in (0,1)")


class EmpiricalTraceDistinguishability:
    """Learn a finite bank of predictive distinction operators from traces.

    The operational population distance is the supremum over declared probes,
    matching the Sigma-Lambda accessible-question distinguishability convention.
    A partition proposal uses complete-link compatibility and is conservative:
    a context with insufficient support for any probe will not be folded with a
    different context.
    """

    def __init__(
        self,
        *,
        actions=(0, 1),
        symbols=(0, 1),
        horizon: int = 2,
        max_contexts: int = 32,
        window: int = 1024,
        policy: LearnedDistinctionPolicy | None = None,
    ) -> None:
        self.policy = policy or LearnedDistinctionPolicy()
        self.memory = TracePredictiveMemory(
            actions=actions,
            symbols=symbols,
            horizon=horizon,
            max_contexts=max_contexts,
            window=window,
            action_conditioned=True,
        )
        self._support: dict[Hashable, dict[tuple, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        self.max_contexts = int(max_contexts)

    @property
    def probes(self) -> tuple[tuple, ...]:
        return self.memory.probes

    @property
    def contexts(self) -> tuple[Hashable, ...]:
        return tuple(self.memory.rows)

    def observe(self, context: Hashable, probe: Iterable, outcomes: Iterable) -> None:
        probe = tuple(probe)
        self.memory.observe(context, probe, tuple(outcomes))
        # The trace memory can evict a context. Keep the support ledger aligned.
        for stale in tuple(self._support):
            if stale not in self.memory.rows:
                del self._support[stale]
        self._support[context][probe] += 1

    def support(self, context: Hashable, probe: Iterable) -> int:
        return self._support.get(context, {}).get(tuple(probe), 0)

    def _radius(self, count: int) -> float:
        if count < self.policy.min_samples:
            return 1.0
        if not self.policy.confidence:
            return 0.0
        tests = max(1, self.max_contexts**2 * len(self.probes))
        alpha = self.policy.delta / tests
        return min(1.0, sqrt(log(2.0 / alpha) / (2.0 * count)))

    def probe_distance(
        self,
        left: Hashable,
        right: Hashable,
        probe: Iterable,
    ) -> float:
        """Empirical total-variation distance for one learned probe response."""
        probe = tuple(probe)
        if probe not in self.probes:
            raise ValueError("probe lies outside the declared trace domain")
        if left == right:
            return 0.0
        if min(self.support(left, probe), self.support(right, probe)) < self.policy.min_samples:
            # Conservative unresolved response: do not collapse unsupported states.
            return 1.0
        p = self.memory.predict(left, probe)
        q = self.memory.predict(right, probe)
        return 0.5 * sum(abs(a - b) for a, b in zip(p, q))

    def population_distance(self, left: Hashable, right: Hashable) -> float:
        """Operational distinguishability: supremum over accessible probes."""
        return max(
            (self.probe_distance(left, right, probe) for probe in self.probes),
            default=0.0,
        )

    def compatible(self, left: Hashable, right: Hashable) -> bool:
        if left == right:
            return True
        for probe in self.probes:
            nl = self.support(left, probe)
            nr = self.support(right, probe)
            if min(nl, nr) < self.policy.min_samples:
                return False
            empirical = self.probe_distance(left, right, probe)
            if empirical + self._radius(nl) + self._radius(nr) > self.policy.epsilon:
                return False
        return True

    def propose_partition(
        self,
        contexts: Iterable[Hashable] | None = None,
    ) -> tuple[tuple[Hashable, ...], ...]:
        """Complete-link empirical quotient proposal over the requested contexts."""
        rows = tuple(self.contexts if contexts is None else contexts)
        if len(rows) != len(set(rows)):
            raise ValueError("contexts must be unique")
        cells: list[list[Hashable]] = []
        for context in rows:
            for cell in cells:
                if all(self.compatible(context, other) for other in cell):
                    cell.append(context)
                    break
            else:
                cells.append([context])
        return tuple(tuple(cell) for cell in cells)

    def build_bank(self, *, channel: str = "predictive-trace") -> DistinctionBank:
        """Expose learned probe responses through the five-operator neuron API."""
        neurons = []
        for index, probe in enumerate(self.probes):
            neurons.append(
                DistinctionNeuron(
                    neuron_id=f"trace-probe-{index}",
                    channel=channel,
                    operator=lambda left, right, probe=probe: self.probe_distance(
                        left, right, probe
                    ),
                )
            )
        return DistinctionBank(tuple(neurons))

    @property
    def snapshot(self) -> dict:
        return {
            "contexts": len(self.contexts),
            "probes": len(self.probes),
            "observations": sum(
                count for rows in self._support.values() for count in rows.values()
            ),
            "epsilon": self.policy.epsilon,
            "min_samples": self.policy.min_samples,
            "confidence": self.policy.confidence,
        }
