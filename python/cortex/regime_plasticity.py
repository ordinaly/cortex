"""Dual-timescale relational regime plasticity for Cortex research.

This module keeps durable historical evidence separate from a fast
recency-sensitive articulation channel. A bounded hysteretic pressure signal
controls when Cortex stops trusting an accepted regime, while the frozen v0.2
controller surfaces remain the execution-risk oracle for replacement evidence.

Research-only: this is not part of the frozen executable specification.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Mapping

from .articulation_controller import ArticulationState
from .controller_calibration import (
    CalibrationBundle,
    epistemic_conflict_features,
    epistemic_execute_features,
    epistemic_structural_features,
    frozen_calibration_v0_2,
)


EPS = 1.0e-12


@dataclass(frozen=True, slots=True)
class RegimePlasticityPolicy:
    """Parameters for dual-timescale regime articulation."""

    fast_decay: float = 0.50
    pressure_decay: float = 0.90
    suspicion_ratio: float = 0.05
    confirmation_steps: int = 5
    maturity_cap: int = 64
    mean_support: float = 2.0 / 3.0
    source_support: float = 1.0
    source_entropy: float = 0.0

    def __post_init__(self) -> None:
        for name, value in (
            ("fast_decay", self.fast_decay),
            ("pressure_decay", self.pressure_decay),
            ("suspicion_ratio", self.suspicion_ratio),
            ("mean_support", self.mean_support),
            ("source_support", self.source_support),
            ("source_entropy", self.source_entropy),
        ):
            if not isfinite(value):
                raise ValueError(f"{name} must be finite")
        if not 0.0 <= self.fast_decay < 1.0:
            raise ValueError("fast_decay must lie in [0,1)")
        if not 0.0 <= self.pressure_decay < 1.0:
            raise ValueError("pressure_decay must lie in [0,1)")
        if not 0.0 < self.suspicion_ratio <= 1.0:
            raise ValueError("suspicion_ratio must lie in (0,1]")
        if self.confirmation_steps < 1:
            raise ValueError("confirmation_steps must be positive")
        if self.maturity_cap < 1:
            raise ValueError("maturity_cap must be positive")
        for name, value in (
            ("mean_support", self.mean_support),
            ("source_support", self.source_support),
            ("source_entropy", self.source_entropy),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0,1]")


@dataclass(frozen=True, slots=True)
class RegimeSnapshot:
    accepted_relation: str | None
    state: ArticulationState
    suspected_change: bool
    challenger: str | None
    confirmation_streak: int
    pressure: float
    suspicion_threshold: float
    fast_evidence: Mapping[str, float]
    historical_evidence: Mapping[str, int]

    @property
    def active_cells(self) -> int:
        """Count relation-indexed cells; constant in time for fixed vocabulary."""
        return len(self.fast_evidence) + len(self.historical_evidence)


@dataclass(frozen=True, slots=True)
class RawRegimeAssessment:
    relation: str
    state: ArticulationState
    confidence: float
    competitor_confidence: float
    margin: float
    execute_probability: float
    conflict_probability: float
    structural_probability: float


def _clamp01(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def assess_fast_evidence(
    relation: str,
    fast_evidence: Mapping[str, float],
    *,
    calibration: CalibrationBundle | None = None,
    mean_support: float = 2.0 / 3.0,
    source_support: float = 1.0,
    source_entropy: float = 0.0,
) -> RawRegimeAssessment:
    """Evaluate recency-weighted evidence with the frozen controller surfaces.

    Geometry profiling is intentionally excluded here: this stage only decides
    whether a relation is epistemically executable. Exact path execution remains
    the responsibility of the existing articulation controller/LPU boundary.
    """
    calibration = calibration or frozen_calibration_v0_2()
    weights = {
        key: max(0.0, float(value))
        for key, value in fast_evidence.items()
        if float(value) > EPS
    }
    total = sum(weights.values())
    if total <= EPS or relation not in weights:
        return RawRegimeAssessment(
            relation=relation,
            state=ArticulationState.EXPLORE,
            confidence=0.0,
            competitor_confidence=0.0,
            margin=0.0,
            execute_probability=0.0,
            conflict_probability=0.0,
            structural_probability=0.0,
        )

    confidence = weights[relation] / total
    competitor = max(
        (value for key, value in weights.items() if key != relation),
        default=0.0,
    ) / total
    margin = confidence - competitor
    effective_examples = max(1, int(round(total)))

    execute_probability = calibration.execute_surface.probability(
        epistemic_execute_features(
            total_examples=effective_examples,
            confidence=confidence,
            margin=margin,
            mean_support=_clamp01(mean_support),
            source_support=_clamp01(source_support),
            source_entropy=_clamp01(source_entropy),
        )
    )
    conflict_probability = calibration.conflict_surface.probability(
        epistemic_conflict_features(
            total_examples=effective_examples,
            confidence=confidence,
            competitor_confidence=competitor,
            margin=margin,
            source_entropy=_clamp01(source_entropy),
        )
    )
    structural_probability = calibration.structural_surface.probability(
        epistemic_structural_features(
            total_examples=effective_examples,
            mean_support=_clamp01(mean_support),
            source_entropy=_clamp01(source_entropy),
        )
    )

    if execute_probability + EPS >= calibration.execute_surface.threshold:
        state = ArticulationState.EXECUTE
    elif (
        conflict_probability + EPS >= calibration.conflict_surface.threshold
        or structural_probability + EPS
        >= calibration.structural_surface.threshold
    ):
        state = ArticulationState.UNRESOLVED
    else:
        state = ArticulationState.EXPLORE

    return RawRegimeAssessment(
        relation=relation,
        state=state,
        confidence=confidence,
        competitor_confidence=competitor,
        margin=margin,
        execute_probability=execute_probability,
        conflict_probability=conflict_probability,
        structural_probability=structural_probability,
    )


class RelationalRegimeTracker:
    """Local dual-timescale regime tracker with explicit hysteresis.

    Historical evidence is never decayed. Fast evidence is exponentially
    discounted and is used only for current-regime articulation. A bounded
    contradiction-pressure signal determines when the accepted articulation is
    suspended. Replacement execution still requires the frozen v0.2 execution
    surface, optionally for several consecutive observations.
    """

    def __init__(
        self,
        policy: RegimePlasticityPolicy | None = None,
        *,
        calibration: CalibrationBundle | None = None,
    ) -> None:
        self.policy = policy or RegimePlasticityPolicy()
        self.calibration = calibration or frozen_calibration_v0_2()
        self._fast: dict[str, float] = {}
        self._history: dict[str, int] = {}
        self._accepted: str | None = None
        self._state = ArticulationState.EXPLORE
        self._suspected = False
        self._challenger: str | None = None
        self._confirmation_streak = 0
        self._pressure = 0.0

    @property
    def accepted_relation(self) -> str | None:
        return self._accepted

    @property
    def state(self) -> ArticulationState:
        return self._state

    @property
    def historical_evidence(self) -> Mapping[str, int]:
        return dict(self._history)

    @property
    def fast_evidence(self) -> Mapping[str, float]:
        return dict(self._fast)

    def _threshold(self) -> float:
        if self._accepted is None:
            return 1.0
        maturity = min(
            self.policy.maturity_cap,
            max(1, self._history.get(self._accepted, 0)),
        )
        return max(1.0, self.policy.suspicion_ratio * maturity)

    def _assessment(self, relation: str) -> RawRegimeAssessment:
        return assess_fast_evidence(
            relation,
            self._fast,
            calibration=self.calibration,
            mean_support=self.policy.mean_support,
            source_support=self.policy.source_support,
            source_entropy=self.policy.source_entropy,
        )

    def _snapshot(self) -> RegimeSnapshot:
        return RegimeSnapshot(
            accepted_relation=self._accepted,
            state=self._state,
            suspected_change=self._suspected,
            challenger=self._challenger,
            confirmation_streak=self._confirmation_streak,
            pressure=self._pressure,
            suspicion_threshold=self._threshold(),
            fast_evidence=dict(sorted(self._fast.items())),
            historical_evidence=dict(sorted(self._history.items())),
        )

    def observe(
        self,
        relation: str,
        *,
        bootstrap: bool = False,
    ) -> RegimeSnapshot:
        """Consume one solved relation outcome.

        bootstrap=True is reserved for establishing an already trusted initial
        regime in calibration fixtures. Normal online updates should leave it
        false.
        """
        relation = str(relation)
        if not relation:
            raise ValueError("relation must be non-empty")

        for key in tuple(self._fast):
            self._fast[key] *= self.policy.fast_decay
            if self._fast[key] <= EPS:
                self._fast[key] = 0.0
        self._fast[relation] = self._fast.get(relation, 0.0) + 1.0
        self._history[relation] = self._history.get(relation, 0) + 1

        if self._accepted is None or bootstrap:
            self._accepted = relation
            self._state = ArticulationState.EXECUTE
            self._suspected = False
            self._challenger = None
            self._confirmation_streak = 0
            self._pressure = 0.0
            return self._snapshot()

        if relation == self._accepted:
            self._pressure *= self.policy.pressure_decay
        else:
            self._pressure = (
                self._pressure * self.policy.pressure_decay + 1.0
            )

        if (
            not self._suspected
            and self._pressure + EPS >= self._threshold()
        ):
            self._suspected = True
            self._challenger = None
            self._confirmation_streak = 0

        if not self._suspected:
            self._state = ArticulationState.EXECUTE
            return self._snapshot()

        accepted = self._accepted
        accepted_assessment = self._assessment(accepted)
        if (
            relation == accepted
            and accepted_assessment.state is ArticulationState.EXECUTE
        ):
            self._suspected = False
            self._challenger = None
            self._confirmation_streak = 0
            self._pressure = 0.0
            self._state = ArticulationState.EXECUTE
            return self._snapshot()

        challengers = []
        for candidate in self._fast:
            if candidate == accepted:
                continue
            assessment = self._assessment(candidate)
            if assessment.state is ArticulationState.EXECUTE:
                challengers.append(assessment)

        if challengers:
            strongest = max(
                challengers,
                key=lambda row: (
                    row.execute_probability,
                    row.margin,
                    row.confidence,
                    row.relation,
                ),
            )
            if strongest.relation == self._challenger:
                self._confirmation_streak += 1
            else:
                self._challenger = strongest.relation
                self._confirmation_streak = 1

            if self._confirmation_streak >= self.policy.confirmation_steps:
                self._accepted = strongest.relation
                self._state = ArticulationState.EXECUTE
                self._suspected = False
                self._challenger = None
                self._confirmation_streak = 0
                self._pressure = 0.0
                return self._snapshot()
        else:
            self._challenger = None
            self._confirmation_streak = 0

        self._state = ArticulationState.UNRESOLVED
        return self._snapshot()
