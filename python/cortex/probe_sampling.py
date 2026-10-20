"""Budgeted sampling policy for active and latent distinguishability probes.

Probe Economy v0.3 decides which learned questions are currently needed to
preserve the empirical distinction relation. This module addresses the next
resource boundary: do not sample every declared question at every tick.

The v0.4 scheduler is deliberately simple and auditable:

- every active probe is sampled every tick;
- latent probes receive bounded-gap sentinel samples;
- the per-tick query budget is a hard cap;
- if the requested active/latent service rates are infeasible, the scheduler
  rejects the policy instead of silently starving latent questions.

This does not yet infer new probes or interpret sentinel observations. It only
provides the sampling substrate needed for later latent-probe rediscovery tests.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .probe_economy import HystereticProbePopulation, Probe, ProbeEconomy


@dataclass(frozen=True, slots=True)
class ProbeSamplingPolicy:
    budget_per_tick: int = 3
    latent_max_gap: int = 12

    def __post_init__(self) -> None:
        if type(self.budget_per_tick) is not int or self.budget_per_tick < 1:
            raise ValueError("budget_per_tick must be a positive integer")
        if type(self.latent_max_gap) is not int or self.latent_max_gap < 1:
            raise ValueError("latent_max_gap must be a positive integer")


class ProbeSamplingScheduler:
    """Deadline scheduler over an active/latent probe population.

    Active probes have period one. Latent probes have period
    ``latent_max_gap``. The scheduler samples only due latent probes, so the
    budget is a maximum rather than a target that must be filled.
    """

    def __init__(
        self,
        economy: ProbeEconomy,
        population: HystereticProbePopulation,
        *,
        policy: ProbeSamplingPolicy | None = None,
    ) -> None:
        if population.economy is not economy:
            raise ValueError("population and scheduler must share one ProbeEconomy")
        self.economy = economy
        self.population = population
        self.policy = policy or ProbeSamplingPolicy()
        self.tick = 0
        self.total_queries = 0
        self.active_queries = 0
        self.latent_queries = 0
        self._due: dict[Probe, int] = {}
        self._last_sample: dict[Probe, int] = {}
        self._status: dict[Probe, bool] = {}
        self._max_gap: dict[Probe, int] = {probe: 0 for probe in economy.probes}
        self._sync_status(initial=True)

    def _ordered_active(self) -> tuple[Probe, ...]:
        return tuple(probe for probe in self.economy.probes if probe in self.population.active)

    def _ordered_latent(self) -> tuple[Probe, ...]:
        return tuple(probe for probe in self.economy.probes if probe not in self.population.active)

    def _latent_capacity(self, active_count: int) -> int:
        return self.policy.budget_per_tick - active_count

    def _validate_feasibility(self, active: tuple[Probe, ...], latent: tuple[Probe, ...]) -> int:
        if len(active) > self.policy.budget_per_tick:
            raise ValueError("sampling budget cannot service every active probe each tick")
        capacity = self._latent_capacity(len(active))
        if latent and capacity <= 0:
            raise ValueError("sampling budget leaves no capacity for latent sentinels")
        if latent and len(latent) > capacity * self.policy.latent_max_gap:
            raise ValueError("latent_max_gap is infeasible for the declared budget/population")
        return capacity

    def _sync_status(self, *, initial: bool = False) -> None:
        active = self._ordered_active()
        latent = self._ordered_latent()
        capacity = self._validate_feasibility(active, latent)

        for probe in active:
            self._due[probe] = self.tick

        if latent:
            # Stagger only newly-latent probes. A probe that remains latent keeps
            # its existing next sentinel deadline; otherwise repeated status
            # synchronization would accidentally pull it forward every tick.
            for index, probe in enumerate(latent):
                was_latent = probe in self._status and not self._status[probe]
                if was_latent and probe in self._due:
                    continue
                self._due[probe] = self.tick + 1 + index // max(1, capacity)

        current_active = set(active)
        self._status = {probe: probe in current_active for probe in self.economy.probes}

    def step(self) -> tuple[Probe, ...]:
        """Return the probes to query at the current tick and advance time."""
        self._sync_status()
        active = self._ordered_active()
        latent = self._ordered_latent()
        capacity = self._validate_feasibility(active, latent)

        selected: list[Probe] = list(active)
        due_latent = sorted(
            (probe for probe in latent if self._due.get(probe, self.tick + 1) <= self.tick),
            key=lambda probe: (self._due[probe], self.economy._index[probe]),
        )
        selected.extend(due_latent[:capacity])

        for probe in selected:
            if probe in self._last_sample:
                gap = self.tick - self._last_sample[probe]
                self._max_gap[probe] = max(self._max_gap[probe], gap)
            self._last_sample[probe] = self.tick
            if probe in self.population.active:
                self.active_queries += 1
                self._due[probe] = self.tick + 1
            else:
                self.latent_queries += 1
                self._due[probe] = self.tick + self.policy.latent_max_gap

        self.total_queries += len(selected)
        self.tick += 1
        return tuple(selected)

    def max_observed_gap(self, probe: Iterable) -> int:
        probe = self.economy._check_probe(probe)
        return self._max_gap[probe]

    @property
    def uniform_query_baseline(self) -> int:
        return self.tick * len(self.economy.probes)

    @property
    def relative_query_saving(self) -> float:
        baseline = self.uniform_query_baseline
        if baseline <= 0:
            return 0.0
        return 1.0 - self.total_queries / baseline

    @property
    def snapshot(self) -> dict:
        latent = self._ordered_latent()
        return {
            "ticks": self.tick,
            "probes": len(self.economy.probes),
            "active": len(self._ordered_active()),
            "latent": len(latent),
            "budget_per_tick": self.policy.budget_per_tick,
            "latent_max_gap": self.policy.latent_max_gap,
            "total_queries": self.total_queries,
            "active_queries": self.active_queries,
            "latent_queries": self.latent_queries,
            "uniform_query_baseline": self.uniform_query_baseline,
            "relative_query_saving": self.relative_query_saving,
            "max_latent_observed_gap": max(
                (self._max_gap[probe] for probe in latent),
                default=0,
            ),
        }
