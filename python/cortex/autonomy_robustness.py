"""Autonomy robustness / temporal plasticity research adapters.

Phase A deliberately wraps the frozen Relational Regime Plasticity v0.1 policy
without changing its update semantics. The adapter adds:

- opaque relation-signature contexts;
- explicit missing-observation events;
- source/provenance metadata for diagnostics;
- downstream execution-safety accounting;
- aggregate fixed-vocabulary state accounting.

Source reliability is recorded but intentionally NOT consumed in Phase A. That
blind spot is a preregistered diagnostic for the next mechanism rather than a
hidden tuning knob.

Research-only: this module does not modify spec-v0.9.5.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Iterable

from .articulation_controller import ArticulationState
from .regime_plasticity import (
    RegimePlasticityPolicy,
    RegimeSnapshot,
    RelationalRegimeTracker,
)


ContextKey = tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TemporalObservation:
    """One chronological observation for an opaque relational context."""

    context: ContextKey
    relation: str | None
    source: str = "unspecified"
    source_reliability: float = 1.0
    true_relation: str | None = None
    bootstrap: bool = False

    def __post_init__(self) -> None:
        if not self.context or any(not str(item) for item in self.context):
            raise ValueError("context must contain at least one non-empty symbol")
        if self.relation is not None and not str(self.relation):
            raise ValueError("relation must be non-empty when present")
        if not self.source:
            raise ValueError("source must be non-empty")
        if (
            not isfinite(float(self.source_reliability))
            or not 0.0 <= float(self.source_reliability) <= 1.0
        ):
            raise ValueError("source_reliability must lie in [0,1]")
        if self.true_relation is not None and not str(self.true_relation):
            raise ValueError("true_relation must be non-empty when present")


@dataclass(frozen=True, slots=True)
class TemporalRead:
    """Read emitted after one wall-clock event."""

    wall_step: int
    observed_step: int
    context: ContextKey
    missing: bool
    source: str
    source_reliability: float
    accepted_relation: str | None
    state: ArticulationState
    suspected_change: bool
    challenger: str | None
    active_cells: int
    context_count: int
    unsafe_execute: bool | None
    safe_execute: bool | None

    @property
    def executable(self) -> bool:
        return self.state is ArticulationState.EXECUTE


class AutonomyTemporalMemory:
    """Bank of independent local regime trackers keyed by opaque signatures.

    The bank is intentionally minimal. It tests whether the already calibrated
    local plasticity mechanism composes across multiple concurrent relational
    structures before introducing any new reliability weighting or budget
    policy.
    """

    def __init__(
        self,
        policy: RegimePlasticityPolicy | None = None,
    ) -> None:
        self.policy = policy or RegimePlasticityPolicy()
        self._trackers: dict[ContextKey, RelationalRegimeTracker] = {}
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
        return sum(tracker.snapshot.active_cells for tracker in self._trackers.values())

    def snapshot(self, context: ContextKey) -> RegimeSnapshot | None:
        tracker = self._trackers.get(tuple(context))
        return tracker.snapshot if tracker is not None else None

    def observe(self, observation: TemporalObservation) -> TemporalRead:
        self._wall_steps += 1
        context = tuple(str(item) for item in observation.context)
        tracker = self._trackers.get(context)
        missing = observation.relation is None

        if not missing:
            if tracker is None:
                tracker = RelationalRegimeTracker(self.policy)
                self._trackers[context] = tracker
            snap = tracker.observe(
                str(observation.relation),
                bootstrap=observation.bootstrap,
            )
            self._observed_steps += 1
        elif tracker is not None:
            # Phase A uses event-time evidence semantics: missing observations
            # do not decay, contradict, confirm, or otherwise rewrite a regime.
            snap = tracker.snapshot
        else:
            snap = None

        if snap is None:
            accepted = None
            state = ArticulationState.EXPLORE
            suspected = False
            challenger = None
        else:
            accepted = snap.accepted_relation
            state = snap.state
            suspected = snap.suspected_change
            challenger = snap.challenger

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

        return TemporalRead(
            wall_step=self._wall_steps,
            observed_step=self._observed_steps,
            context=context,
            missing=missing,
            source=observation.source,
            source_reliability=float(observation.source_reliability),
            accepted_relation=accepted,
            state=state,
            suspected_change=suspected,
            challenger=challenger,
            active_cells=self.active_cells,
            context_count=self.context_count,
            unsafe_execute=unsafe,
            safe_execute=safe,
        )

    def consume(
        self,
        observations: Iterable[TemporalObservation],
    ) -> tuple[TemporalRead, ...]:
        return tuple(self.observe(row) for row in observations)
