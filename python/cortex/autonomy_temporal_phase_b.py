"""Autonomy Robustness & Temporal Plasticity v1 — Phase B mechanisms.

Phase B addresses the two failures measured by Phase A without changing durable
history:

1. source-aware current evidence, using bounded reliability transforms;
2. an execution-freshness gate over explicit missing observations.

The underlying RelationalRegimeTracker remains authoritative for accepted
regimes and raw historical counts. Reliability affects only the fast/current
channel. Freshness affects only permission to execute, never stored evidence.

Research-only: this module does not modify spec-v0.9.5.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Iterable

from .articulation_controller import ArticulationState
from .autonomy_robustness import ContextKey, TemporalObservation
from .regime_plasticity import (
    RegimePlasticityPolicy,
    RegimeSnapshot,
    RelationalRegimeTracker,
)


@dataclass(frozen=True, slots=True)
class TemporalRobustnessPolicy:
    """Phase-B source/freshness policy.

    source_weighting:
      - "unweighted": Phase-A control;
      - "linear": reliability / full_trust_reliability, clipped to one;
      - "odds": bounded odds-relative weighting.

    The odds transform is intentionally bounded. A reliability equal to the
    full-trust reference contributes weight 1.0; weaker sources contribute less
    but are never converted into negative evidence.
    """

    source_weighting: str = "odds"
    full_trust_reliability: float = 0.95
    freshness_missing_grace: int = 1

    def __post_init__(self) -> None:
        if self.source_weighting not in {"unweighted", "linear", "odds"}:
            raise ValueError(
                "source_weighting must be unweighted, linear, or odds"
            )
        if (
            not isfinite(float(self.full_trust_reliability))
            or not 0.5 < float(self.full_trust_reliability) < 1.0
        ):
            raise ValueError(
                "full_trust_reliability must lie strictly between 0.5 and 1"
            )
        if self.freshness_missing_grace < 0:
            raise ValueError("freshness_missing_grace must be non-negative")

    def current_weight(self, reliability: float) -> float:
        reliability = float(reliability)
        if not isfinite(reliability) or not 0.0 <= reliability <= 1.0:
            raise ValueError("reliability must lie in [0,1]")
        if self.source_weighting == "unweighted":
            return 1.0

        reference = float(self.full_trust_reliability)
        if reliability >= reference:
            return 1.0
        if reliability <= 0.0:
            return 0.0
        if self.source_weighting == "linear":
            return reliability / reference

        # Bounded odds-relative evidence. This uses reliability only as a
        # positive attenuation factor; low-reliability observations are not
        # inverted into evidence for another relation.
        odds = reliability / (1.0 - reliability)
        reference_odds = reference / (1.0 - reference)
        return min(1.0, odds / reference_odds)


@dataclass(frozen=True, slots=True)
class PhaseBTemporalRead:
    """Read emitted after one Phase-B wall event."""

    wall_step: int
    observed_step: int
    context: ContextKey
    missing: bool
    missing_streak: int
    source: str
    source_reliability: float
    current_weight: float
    accepted_relation: str | None
    memory_state: ArticulationState
    state: ArticulationState
    suspected_change: bool
    challenger: str | None
    freshness_suspended: bool
    active_cells: int
    context_count: int
    unsafe_execute: bool | None
    safe_execute: bool | None

    @property
    def executable(self) -> bool:
        return self.state is ArticulationState.EXECUTE

    @property
    def execution_fresh(self) -> bool:
        return not self.freshness_suspended


class PhaseBTemporalMemory:
    """Source-aware regime bank with a read-only execution-freshness boundary."""

    def __init__(
        self,
        policy: TemporalRobustnessPolicy | None = None,
        *,
        regime_policy: RegimePlasticityPolicy | None = None,
    ) -> None:
        self.policy = policy or TemporalRobustnessPolicy()
        self.regime_policy = regime_policy or RegimePlasticityPolicy()
        self._trackers: dict[ContextKey, RelationalRegimeTracker] = {}
        self._missing_streaks: dict[ContextKey, int] = {}
        self._wall_steps = 0
        self._observed_steps = 0

    @property
    def wall_steps(self) -> int:
        return self._wall_steps

    @property
    def observed_steps(self) -> int:
        return self._observed_steps

    @property
    def context_count(self) -> int:
        return len(self._trackers)

    @property
    def active_cells(self) -> int:
        return sum(
            tracker.snapshot.active_cells
            for tracker in self._trackers.values()
        )

    def snapshot(self, context: ContextKey) -> RegimeSnapshot | None:
        tracker = self._trackers.get(tuple(context))
        return tracker.snapshot if tracker is not None else None

    def missing_streak(self, context: ContextKey) -> int:
        return self._missing_streaks.get(tuple(context), 0)

    def observe(self, observation: TemporalObservation) -> PhaseBTemporalRead:
        self._wall_steps += 1
        context = tuple(str(item) for item in observation.context)
        tracker = self._trackers.get(context)
        missing = observation.relation is None

        if missing:
            streak = self._missing_streaks.get(context, 0) + 1
            self._missing_streaks[context] = streak
            snap = tracker.snapshot if tracker is not None else None
            current_weight = 0.0
        else:
            self._missing_streaks[context] = 0
            streak = 0
            if tracker is None:
                tracker = RelationalRegimeTracker(self.regime_policy)
                self._trackers[context] = tracker
            current_weight = self.policy.current_weight(
                observation.source_reliability
            )
            snap = tracker.observe(
                str(observation.relation),
                bootstrap=observation.bootstrap,
                current_weight=current_weight,
            )
            self._observed_steps += 1

        if snap is None:
            accepted = None
            memory_state = ArticulationState.EXPLORE
            suspected = False
            challenger = None
        else:
            accepted = snap.accepted_relation
            memory_state = snap.state
            suspected = snap.suspected_change
            challenger = snap.challenger

        freshness_suspended = (
            missing
            and memory_state is ArticulationState.EXECUTE
            and streak > self.policy.freshness_missing_grace
        )
        state = (
            ArticulationState.EXPLORE
            if freshness_suspended
            else memory_state
        )

        unsafe: bool | None = None
        safe: bool | None = None
        if observation.true_relation is not None:
            unsafe = (
                state is ArticulationState.EXECUTE
                and accepted != observation.true_relation
            )
            safe = (
                state is ArticulationState.EXECUTE
                and accepted == observation.true_relation
            )

        return PhaseBTemporalRead(
            wall_step=self._wall_steps,
            observed_step=self._observed_steps,
            context=context,
            missing=missing,
            missing_streak=streak,
            source=observation.source,
            source_reliability=float(observation.source_reliability),
            current_weight=current_weight,
            accepted_relation=accepted,
            memory_state=memory_state,
            state=state,
            suspected_change=suspected,
            challenger=challenger,
            freshness_suspended=freshness_suspended,
            active_cells=self.active_cells,
            context_count=self.context_count,
            unsafe_execute=unsafe,
            safe_execute=safe,
        )

    def consume(
        self,
        observations: Iterable[TemporalObservation],
    ) -> tuple[PhaseBTemporalRead, ...]:
        return tuple(self.observe(row) for row in observations)
