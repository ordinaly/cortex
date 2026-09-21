"""Dual-timescale relational regime plasticity for Cortex research.

Historical evidence is durable. A separate exponentially weighted recent
channel detects local regime changes and controls when the current articulation
must be challenged, withdrawn, or replaced. The mechanism is relation-label
agnostic and operates on opaque composition signatures.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Mapping, Protocol, Sequence

from .articulation_controller import ArticulationState
from .hypergraph_inference import CompositionRule, RelationStep


EPS = 1.0e-12


class RegimePhase(str, Enum):
    BOOTSTRAP = "bootstrap"
    STABLE = "stable"
    CHALLENGED = "challenged"
    UNRESOLVED = "unresolved"


@dataclass(frozen=True, slots=True)
class RegimeConfig:
    recent_decay: float = 0.90
    challenge_threshold: float = 0.40
    challenge_hysteresis: float = 0.05
    challenge_patience: int = 2
    acceptance_threshold: float = 0.90
    acceptance_patience: int = 2
    minimum_initial_observations: int = 4
    minimum_candidate_observations: int = 8

    def __post_init__(self) -> None:
        if not 0.0 < self.recent_decay < 1.0:
            raise ValueError("recent_decay must lie in (0,1)")
        if not 0.0 < self.challenge_threshold < 1.0:
            raise ValueError("challenge_threshold must lie in (0,1)")
        if not 0.0 <= self.challenge_hysteresis < self.challenge_threshold:
            raise ValueError(
                "challenge_hysteresis must be non-negative and smaller "
                "than challenge_threshold"
            )
        if not self.challenge_threshold < self.acceptance_threshold <= 1.0:
            raise ValueError(
                "acceptance_threshold must exceed challenge_threshold"
            )
        for name, value in (
            ("challenge_patience", self.challenge_patience),
            ("acceptance_patience", self.acceptance_patience),
            ("minimum_initial_observations", self.minimum_initial_observations),
            ("minimum_candidate_observations", self.minimum_candidate_observations),
        ):
            if value < 1:
                raise ValueError(f"{name} must be positive")


@dataclass(frozen=True, slots=True)
class RegimeArchive:
    relation: str
    started_at: int
    ended_at: int
    active_observations: int
    historical_observations_at_close: int


@dataclass(frozen=True, slots=True)
class RegimeRead:
    time: int
    phase: RegimePhase
    accepted_relation: str | None
    candidate_relation: str | None
    recent_posterior: tuple[tuple[str, float], ...]
    historical_counts: tuple[tuple[str, int], ...]
    active_observations: int
    candidate_observations: int
    challenge_streak: int
    acceptance_streak: int
    regime_index: int
    archives: tuple[RegimeArchive, ...]

    @property
    def recent_top(self) -> tuple[str, float] | None:
        return self.recent_posterior[0] if self.recent_posterior else None


class _ArticulationControllerLike(Protocol):
    def decide(
        self,
        source: str,
        relation: str,
        rules: Sequence[CompositionRule],
        *,
        source_support: float = 1.0,
        source_entropy: float = 0.0,
        exact_witness: bool = False,
    ): ...


@dataclass(frozen=True, slots=True)
class RegimeArticulationDecision:
    state: ArticulationState
    phase: RegimePhase
    relation: str
    accepted_relation: str | None
    candidate_relation: str | None
    reasons: tuple[str, ...]
    controller_decision: object | None


class RelationalRegimeMemory:
    """Track durable history and a fast local regime articulation.

    Historical counts never decay. Only the recent detection channel decays.
    A confirmed regime switch archives the previous active articulation rather
    than deleting it.
    """

    def __init__(
        self,
        signature: Sequence[RelationStep],
        *,
        config: RegimeConfig | None = None,
        mean_support: float = 2.0 / 3.0,
    ) -> None:
        signature = tuple(signature)
        if not signature:
            raise ValueError("signature must be non-empty")
        if not 0.0 <= mean_support <= 1.0 or not math.isfinite(mean_support):
            raise ValueError("mean_support must be finite and in [0,1]")
        self.signature = signature
        self.config = config or RegimeConfig()
        self.mean_support = float(mean_support)

        self.time = 0
        self.phase = RegimePhase.BOOTSTRAP
        self.accepted_relation: str | None = None
        self.candidate_relation: str | None = None
        self.regime_index = 0
        self.regime_started_at = 0
        self.active_observations = 0
        self.candidate_observations = 0
        self.challenge_streak = 0
        self.acceptance_streak = 0

        self._historical_counts: dict[str, int] = {}
        self._recent_weights: dict[str, float] = {}
        self._alternate_run_relation: str | None = None
        self._alternate_run_count = 0
        self._archives: list[RegimeArchive] = []

    @property
    def historical_counts(self) -> Mapping[str, int]:
        return dict(self._historical_counts)

    @property
    def archives(self) -> tuple[RegimeArchive, ...]:
        return tuple(self._archives)

    def _posterior(self) -> tuple[tuple[str, float], ...]:
        total = sum(self._recent_weights.values())
        if total <= EPS:
            return ()
        rows = [
            (relation, weight / total)
            for relation, weight in self._recent_weights.items()
            if weight > EPS
        ]
        rows.sort(key=lambda row: (-row[1], row[0]))
        return tuple(rows)

    def _read(self) -> RegimeRead:
        return RegimeRead(
            time=self.time,
            phase=self.phase,
            accepted_relation=self.accepted_relation,
            candidate_relation=self.candidate_relation,
            recent_posterior=self._posterior(),
            historical_counts=tuple(sorted(self._historical_counts.items())),
            active_observations=self.active_observations,
            candidate_observations=self.candidate_observations,
            challenge_streak=self.challenge_streak,
            acceptance_streak=self.acceptance_streak,
            regime_index=self.regime_index,
            archives=tuple(self._archives),
        )

    def read(self) -> RegimeRead:
        return self._read()

    def _reset_candidate(self) -> None:
        self.candidate_relation = None
        self.candidate_observations = 0
        self.challenge_streak = 0
        self.acceptance_streak = 0

    def _accept_initial(self, relation: str) -> None:
        self.accepted_relation = relation
        self.phase = RegimePhase.STABLE
        self.active_observations = self._historical_counts[relation]
        self.regime_started_at = max(
            1,
            self.time - self.active_observations + 1,
        )
        self._reset_candidate()

    def _accept_replacement(self, relation: str) -> None:
        incumbent = self.accepted_relation
        if incumbent is not None:
            self._archives.append(
                RegimeArchive(
                    relation=incumbent,
                    started_at=self.regime_started_at,
                    ended_at=self.time,
                    active_observations=self.active_observations,
                    historical_observations_at_close=(
                        self._historical_counts.get(incumbent, 0)
                    ),
                )
            )
        observations = max(
            self.config.minimum_candidate_observations,
            self.candidate_observations,
        )
        self.accepted_relation = relation
        self.phase = RegimePhase.STABLE
        self.regime_index += 1
        self.regime_started_at = max(1, self.time - observations + 1)
        self.active_observations = observations
        self._alternate_run_relation = None
        self._alternate_run_count = 0
        self._reset_candidate()

    def observe(
        self,
        relation: str,
        *,
        reliability: float = 1.0,
    ) -> RegimeRead:
        if not relation:
            raise ValueError("relation must be non-empty")
        if not math.isfinite(reliability) or not 0.0 < reliability <= 1.0:
            raise ValueError("reliability must be finite and in (0,1]")

        self.time += 1
        for key in tuple(self._recent_weights):
            self._recent_weights[key] *= self.config.recent_decay
            if self._recent_weights[key] <= EPS:
                del self._recent_weights[key]
        self._recent_weights[relation] = (
            self._recent_weights.get(relation, 0.0) + reliability
        )
        self._historical_counts[relation] = (
            self._historical_counts.get(relation, 0) + 1
        )

        posterior = self._posterior()
        dominant, dominant_support = posterior[0]

        if self.accepted_relation is None:
            if (
                self._historical_counts[dominant]
                >= self.config.minimum_initial_observations
                and dominant_support + EPS
                >= self.config.acceptance_threshold
            ):
                self._accept_initial(dominant)
            return self._read()

        incumbent = self.accepted_relation
        if relation == incumbent:
            self._alternate_run_relation = None
            self._alternate_run_count = 0
            if self.phase is RegimePhase.STABLE:
                self.active_observations += 1
        else:
            if self._alternate_run_relation == relation:
                self._alternate_run_count += 1
            else:
                self._alternate_run_relation = relation
                self._alternate_run_count = 1

        clear_threshold = (
            self.config.challenge_threshold
            - self.config.challenge_hysteresis
        )
        incumbent_support = dict(posterior).get(incumbent, 0.0)

        if (
            dominant == incumbent
            or dominant_support + EPS < clear_threshold
        ):
            self.phase = RegimePhase.STABLE
            self._reset_candidate()
            return self._read()

        if (
            dominant != incumbent
            and dominant_support + EPS
            >= self.config.challenge_threshold
        ):
            if self.candidate_relation != dominant:
                self.candidate_relation = dominant
                self.candidate_observations = (
                    self._alternate_run_count
                    if self._alternate_run_relation == dominant
                    else int(relation == dominant)
                )
                self.challenge_streak = 1
                self.acceptance_streak = 0
            else:
                self.challenge_streak += 1
                if relation == dominant:
                    self.candidate_observations += 1

            if self.phase is RegimePhase.STABLE:
                self.phase = RegimePhase.CHALLENGED
            if (
                self.challenge_streak
                >= self.config.challenge_patience
            ):
                self.phase = RegimePhase.UNRESOLVED

            if self.phase is RegimePhase.UNRESOLVED:
                if (
                    dominant_support + EPS
                    >= self.config.acceptance_threshold
                    and self.candidate_observations
                    >= self.config.minimum_candidate_observations
                ):
                    self.acceptance_streak += 1
                else:
                    self.acceptance_streak = 0
                if (
                    self.acceptance_streak
                    >= self.config.acceptance_patience
                ):
                    self._accept_replacement(dominant)
            return self._read()

        # Between the clear and challenge thresholds, preserve the current
        # challenge state rather than oscillating on a single marginal sample.
        if self.phase is RegimePhase.STABLE:
            self.phase = RegimePhase.CHALLENGED
        if incumbent_support > dominant_support:
            self.phase = RegimePhase.STABLE
            self._reset_candidate()
        return self._read()

    def active_rules(self) -> tuple[CompositionRule, ...]:
        if self.accepted_relation is None:
            return ()
        return (
            CompositionRule(
                signature=self.signature,
                target_relation=self.accepted_relation,
                examples=max(1, self.active_observations),
                confidence=1.0,
                mean_support=self.mean_support,
            ),
        )

    def gate_state(self) -> ArticulationState | None:
        if self.phase is RegimePhase.STABLE:
            return None
        if self.phase is RegimePhase.UNRESOLVED:
            return ArticulationState.UNRESOLVED
        return ArticulationState.EXPLORE


def decide_with_regime(
    controller: _ArticulationControllerLike,
    regime: RelationalRegimeMemory,
    source: str,
    relation: str,
    *,
    source_support: float = 1.0,
    source_entropy: float = 0.0,
) -> RegimeArticulationDecision:
    """Gate exact articulation through the current temporal regime."""

    gate = regime.gate_state()
    read = regime.read()
    if gate is not None:
        return RegimeArticulationDecision(
            state=gate,
            phase=read.phase,
            relation=relation,
            accepted_relation=read.accepted_relation,
            candidate_relation=read.candidate_relation,
            reasons=(f"regime-{read.phase.value}",),
            controller_decision=None,
        )

    decision = controller.decide(
        source,
        relation,
        regime.active_rules(),
        source_support=source_support,
        source_entropy=source_entropy,
    )
    return RegimeArticulationDecision(
        state=decision.state,
        phase=read.phase,
        relation=relation,
        accepted_relation=read.accepted_relation,
        candidate_relation=read.candidate_relation,
        reasons=tuple(decision.reasons),
        controller_decision=decision,
    )
