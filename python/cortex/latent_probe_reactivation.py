"""Sentinel-driven reactivation support for latent distinguishability probes.

This module closes one loop between Probe Sampling v0.4 and Probe Economy v0.3:

latent question -> sparse sentinel evidence -> detected response drift ->
temporary query escalation -> ordinary learner refresh -> ordinary economy
reactivation if the distinction is still needed.

The monitor never activates a probe by itself. It controls observation priority
only. Representation authority remains with ``HystereticProbePopulation`` after
new evidence has passed through the existing distinguishability learner.
"""
from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from math import isfinite
from typing import Hashable, Iterable

from .learned_distinguishability import EmpiricalTraceDistinguishability
from .probe_economy import HystereticProbePopulation, Probe, ProbeEconomy


@dataclass(frozen=True, slots=True)
class LatentReactivationPolicy:
    sentinel_window: int = 4
    min_sentinels: int = 3
    trigger_threshold: float = 0.50
    escalation_samples: int = 16
    max_priority_probes: int = 1

    def __post_init__(self) -> None:
        if any(
            type(value) is not int or value < 1
            for value in (
                self.sentinel_window,
                self.min_sentinels,
                self.escalation_samples,
                self.max_priority_probes,
            )
        ):
            raise ValueError("reactivation counts must be positive integers")
        if self.min_sentinels > self.sentinel_window:
            raise ValueError("min_sentinels must not exceed sentinel_window")
        if not isfinite(float(self.trigger_threshold)) or not 0.0 <= self.trigger_threshold <= 1.0:
            raise ValueError("trigger_threshold must be finite and lie in [0,1]")


class LatentProbeReactivationMonitor:
    """Detect drift in sparse latent-probe answers and request query escalation.

    Baselines are frozen when a probe enters latency. Recent response words are
    maintained separately from the main learner, so the detector can compare a
    short current sentinel window with the representation that justified
    latency. Triggering grants only temporary sampling priority.
    """

    def __init__(
        self,
        learner: EmpiricalTraceDistinguishability,
        economy: ProbeEconomy,
        population: HystereticProbePopulation,
        *,
        policy: LatentReactivationPolicy | None = None,
    ) -> None:
        if economy.learner is not learner or population.economy is not economy:
            raise ValueError("learner, economy and population must share one state")
        self.learner = learner
        self.economy = economy
        self.population = population
        self.policy = policy or LatentReactivationPolicy()
        self._baseline: dict[tuple[Probe, Hashable], tuple[float, ...]] = {}
        self._recent: dict[tuple[Probe, Hashable], deque[tuple]] = {}
        self._status: dict[Probe, bool] = {}
        self._escalation: dict[Probe, int] = {}
        self._scores: dict[Probe, float] = {}
        self.triggers = 0
        self.completed_escalations = 0
        self.reactivations = 0
        self.sync_population(initial=True)

    def _words(self, probe: Probe) -> tuple[tuple, ...]:
        return self.learner.memory.words[len(probe)]

    def _capture_baseline(self, probe: Probe) -> None:
        for context in self.economy.contexts:
            self._baseline[(probe, context)] = tuple(
                self.learner.memory.predict(context, probe)
            )
            self._recent[(probe, context)] = deque(maxlen=self.policy.sentinel_window)
        self._scores[probe] = 0.0

    def sync_population(self, *, initial: bool = False) -> None:
        """Track active/latent transitions and freeze new latency baselines."""
        current_active = set(self.population.active)
        for probe in self.economy.probes:
            is_active = probe in current_active
            was_active = self._status.get(probe)
            if not is_active and (initial or was_active is None or was_active):
                self._capture_baseline(probe)
                self._escalation.pop(probe, None)
            elif is_active and was_active is False:
                if self._escalation.pop(probe, None) is not None:
                    self.reactivations += 1
                self._scores.pop(probe, None)
                for context in self.economy.contexts:
                    self._recent.pop((probe, context), None)
            self._status[probe] = is_active

    def _empirical(self, probe: Probe, context: Hashable) -> tuple[float, ...] | None:
        row = self._recent.get((probe, context))
        if row is None or len(row) < self.policy.min_sentinels:
            return None
        counts = Counter(row)
        words = self._words(probe)
        return tuple(counts.get(word, 0) / len(row) for word in words)

    @staticmethod
    def _tv(left: Iterable[float], right: Iterable[float]) -> float:
        return 0.5 * sum(abs(a - b) for a, b in zip(left, right))

    def drift_score(self, probe: Iterable) -> float:
        probe = self.economy._check_probe(probe)
        scores = []
        for context in self.economy.contexts:
            recent = self._empirical(probe, context)
            if recent is None:
                continue
            baseline = self._baseline.get((probe, context))
            if baseline is not None:
                scores.append(self._tv(recent, baseline))
        return max(scores, default=0.0)

    def observe(self, probe: Iterable, context: Hashable, outcomes: Iterable) -> bool:
        """Record one sampled response; return True when it newly triggers drift.

        The caller should separately feed the same observation to the ordinary
        distinguishability learner. This monitor deliberately owns no predictive
        representation.
        """
        probe = self.economy._check_probe(probe)
        outcomes = tuple(outcomes)
        if context not in self.economy.contexts:
            raise ValueError("context lies outside the monitored economy carrier")
        if outcomes not in self._words(probe):
            raise ValueError("outcome word lies outside the declared probe domain")
        if probe in self.population.active:
            return False

        row = self._recent.setdefault(
            (probe, context),
            deque(maxlen=self.policy.sentinel_window),
        )
        row.append(outcomes)
        score = self.drift_score(probe)
        self._scores[probe] = score
        if score >= self.policy.trigger_threshold and self._escalation.get(probe, 0) <= 0:
            self._escalation[probe] = self.policy.escalation_samples
            self.triggers += 1
            return True
        return False

    def priority_probes(self) -> tuple[Probe, ...]:
        """Latent probes currently entitled to temporary dense sampling."""
        self.sync_population()
        rows = [
            probe
            for probe in self.economy.probes
            if probe not in self.population.active and self._escalation.get(probe, 0) > 0
        ]
        rows.sort(
            key=lambda probe: (
                -self._scores.get(probe, 0.0),
                self.economy._index[probe],
            )
        )
        return tuple(rows[: self.policy.max_priority_probes])

    def after_tick(self, selected: Iterable[Iterable]) -> None:
        """Consume escalation sample budgets after one scheduler tick."""
        selected_set = {self.economy._check_probe(probe) for probe in selected}
        for probe in tuple(self._escalation):
            if probe in self.population.active:
                continue
            if probe in selected_set and self._escalation[probe] > 0:
                self._escalation[probe] -= 1
            if self._escalation[probe] <= 0:
                del self._escalation[probe]
                self.completed_escalations += 1
                # If the refreshed evidence still did not earn activation,
                # accept it as the new latent baseline before looking for a
                # later change. This prevents immediate repeated bursts.
                self._capture_baseline(probe)
        self.sync_population()

    @property
    def snapshot(self) -> dict:
        priority = self.priority_probes()
        return {
            "triggers": self.triggers,
            "reactivations": self.reactivations,
            "completed_escalations": self.completed_escalations,
            "priority_probe_indices": tuple(self.economy._index[p] for p in priority),
            "max_drift_score": max(self._scores.values(), default=0.0),
        }
