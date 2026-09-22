"""Epistemic articulation controller for Cortex exact execution.

The controller decides whether a fuzzy relational articulation is sufficiently
stable to execute exactly. It never invents relations or changes evidence.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from typing import Iterable

from .hypergraph_geometry import HypergraphGeometry
from .hypergraph_inference import CompositionRule, RelationStep
from .hypergraph_lpu import (
    EnumeratingRuleExecutor,
    LPURuleExecutor,
    RuleExecutionRead,
    _matching_edges,
)


EPS = 1.0e-12


class ArticulationState(str, Enum):
    EXPLORE = "explore"
    EXECUTE = "execute"
    UNRESOLVED = "unresolved"


class ExecutionBackend(str, Enum):
    CORTEX = "cortex"
    LPU = "lpu"


@dataclass(frozen=True, slots=True)
class ControllerPolicy:
    minimum_examples: int = 4
    minimum_rule_confidence: float = 0.75
    minimum_rule_margin: float = 0.25
    conflict_evidence_floor: int = 4
    minimum_source_support: float = 0.70
    maximum_source_entropy: float = 0.35
    minimum_resolution: float = 0.50
    resolution_levels: tuple[float, ...] = (
        0.80,
        0.75,
        0.70,
        0.65,
        0.60,
        0.55,
        0.50,
    )
    lpu_min_paths: int = 32
    lpu_min_compression_ratio: float = 2.0

    def __post_init__(self) -> None:
        if self.minimum_examples < 1:
            raise ValueError("minimum_examples must be positive")
        if self.conflict_evidence_floor < 1:
            raise ValueError("conflict_evidence_floor must be positive")
        for name, value in (
            ("minimum_rule_confidence", self.minimum_rule_confidence),
            ("minimum_rule_margin", self.minimum_rule_margin),
            ("minimum_source_support", self.minimum_source_support),
            ("maximum_source_entropy", self.maximum_source_entropy),
            ("minimum_resolution", self.minimum_resolution),
        ):
            if not isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and in [0,1]")
        if self.lpu_min_paths < 1:
            raise ValueError("lpu_min_paths must be positive")
        if (
            not isfinite(self.lpu_min_compression_ratio)
            or self.lpu_min_compression_ratio < 0.0
        ):
            raise ValueError(
                "lpu_min_compression_ratio must be finite and non-negative"
            )
        if not self.resolution_levels:
            raise ValueError("resolution_levels must be non-empty")
        if any(
            not isfinite(value) or not 0.0 <= value <= 1.0
            for value in self.resolution_levels
        ):
            raise ValueError("resolution_levels must lie in [0,1]")


@dataclass(frozen=True, slots=True)
class RuleAssessment:
    rule: CompositionRule
    competitor_confidence: float
    confidence_margin: float
    admissible: bool


@dataclass(frozen=True, slots=True)
class ArticulationProfile:
    resolution: float
    path_multiplicity: int
    materialized_edges: int
    layered_states: int
    reachable_targets: tuple[str, ...]

    @property
    def compression_ratio(self) -> float:
        return self.path_multiplicity / max(1, self.materialized_edges)


@dataclass(frozen=True, slots=True)
class ArticulationDecision:
    state: ArticulationState
    backend: ExecutionBackend | None
    relation: str
    source: str
    resolution: float | None
    selected_rules: tuple[CompositionRule, ...]
    assessments: tuple[RuleAssessment, ...]
    source_support: float
    source_entropy: float
    path_multiplicity: int
    materialized_edges: int
    compression_ratio: float
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ControlledExecution:
    decision: ArticulationDecision
    execution: RuleExecutionRead | None



def profile_rules(
    geometry: HypergraphGeometry,
    source: str,
    rules: Iterable[CompositionRule],
    *,
    resolution: float,
) -> ArticulationProfile:
    """Profile exact execution without enumerating complete paths."""

    total_paths = 0
    layered_edges: set[
        tuple[tuple[int, str], tuple[int, str], str]
    ] = set()
    layered_states: set[tuple[int, str]] = {(0, source)}
    targets: set[str] = set()

    for rule_index, rule in enumerate(rules):
        current: dict[str, int] = {source: 1}
        for depth, step in enumerate(rule.signature):
            nxt: dict[str, int] = {}
            for node, count in current.items():
                for target, _support, edge_id in _matching_edges(
                    geometry,
                    node,
                    step,
                    resolution=resolution,
                ):
                    nxt[target] = nxt.get(target, 0) + count
                    layered_states.add((depth + 1, target))
                    layered_edges.add(
                        (
                            (depth, node),
                            (depth + 1, target),
                            f"{rule_index}:{edge_id}",
                        )
                    )
            current = nxt
            if not current:
                break

        total_paths += sum(current.values())
        targets.update(current)

    return ArticulationProfile(
        resolution=resolution,
        path_multiplicity=total_paths,
        materialized_edges=len(layered_edges),
        layered_states=len(layered_states),
        reachable_targets=tuple(sorted(targets)),
    )


class ArticulationController:
    """Choose exploration, exact execution, or unresolved abstention."""

    def __init__(
        self,
        geometry: HypergraphGeometry,
        *,
        policy: ControllerPolicy | None = None,
    ) -> None:
        self.geometry = geometry
        self.policy = policy or ControllerPolicy()

    @staticmethod
    def _competitor_confidence(
        rule: CompositionRule,
        rules: tuple[CompositionRule, ...],
    ) -> float:
        return max(
            (
                other.confidence
                for other in rules
                if (
                    other.signature == rule.signature
                    and other.target_relation != rule.target_relation
                )
            ),
            default=0.0,
        )

    def _assess(
        self,
        relation: str,
        rules: tuple[CompositionRule, ...],
    ) -> tuple[RuleAssessment, ...]:
        rows: list[RuleAssessment] = []
        for rule in rules:
            if rule.target_relation != relation:
                continue
            competitor = self._competitor_confidence(rule, rules)
            margin = rule.confidence - competitor
            admissible = (
                rule.examples >= self.policy.minimum_examples
                and rule.confidence + EPS
                >= self.policy.minimum_rule_confidence
                and margin + EPS >= self.policy.minimum_rule_margin
                and rule.mean_support + EPS
                >= self.policy.minimum_resolution
            )
            rows.append(
                RuleAssessment(
                    rule=rule,
                    competitor_confidence=competitor,
                    confidence_margin=margin,
                    admissible=admissible,
                )
            )
        rows.sort(
            key=lambda row: (
                -row.rule.confidence,
                -row.confidence_margin,
                -row.rule.examples,
                -row.rule.mean_support,
                tuple(
                    (step.relation, step.orientation)
                    for step in row.rule.signature
                ),
            )
        )
        return tuple(rows)

    def decide(
        self,
        source: str,
        relation: str,
        rules: Iterable[CompositionRule],
        *,
        source_support: float = 1.0,
        source_entropy: float = 0.0,
        exact_witness: bool = False,
    ) -> ArticulationDecision:
        if not isfinite(source_support) or not 0.0 <= source_support <= 1.0:
            raise ValueError("source_support must be finite and in [0,1]")
        if not isfinite(source_entropy) or not 0.0 <= source_entropy <= 1.0:
            raise ValueError("source_entropy must be finite and in [0,1]")

        rules = tuple(rules)
        assessments = self._assess(relation, rules)
        reasons: list[str] = []

        if source_support + EPS < self.policy.minimum_source_support:
            reasons.append("source-support-below-threshold")
        if source_entropy > self.policy.maximum_source_entropy + EPS:
            reasons.append("source-entropy-too-high")
        if reasons:
            return ArticulationDecision(
                state=ArticulationState.EXPLORE,
                backend=None,
                relation=relation,
                source=source,
                resolution=None,
                selected_rules=(),
                assessments=assessments,
                source_support=source_support,
                source_entropy=source_entropy,
                path_multiplicity=0,
                materialized_edges=0,
                compression_ratio=0.0,
                reasons=tuple(reasons),
            )

        if not assessments:
            return ArticulationDecision(
                state=ArticulationState.EXPLORE,
                backend=None,
                relation=relation,
                source=source,
                resolution=None,
                selected_rules=(),
                assessments=(),
                source_support=source_support,
                source_entropy=source_entropy,
                path_multiplicity=0,
                materialized_edges=0,
                compression_ratio=0.0,
                reasons=("no-composition-evidence",),
            )

        admissible = tuple(
            row.rule for row in assessments if row.admissible
        )
        if not admissible:
            strongest = assessments[0]
            enough_evidence = (
                strongest.rule.examples
                >= self.policy.conflict_evidence_floor
            )
            conflicting = (
                strongest.rule.confidence + EPS
                < self.policy.minimum_rule_confidence
                or strongest.confidence_margin + EPS
                < self.policy.minimum_rule_margin
            )
            state = (
                ArticulationState.UNRESOLVED
                if enough_evidence and conflicting
                else ArticulationState.EXPLORE
            )
            reason = (
                "conflicting-composition-evidence"
                if state is ArticulationState.UNRESOLVED
                else "insufficient-composition-evidence"
            )
            return ArticulationDecision(
                state=state,
                backend=None,
                relation=relation,
                source=source,
                resolution=None,
                selected_rules=(),
                assessments=assessments,
                source_support=source_support,
                source_entropy=source_entropy,
                path_multiplicity=0,
                materialized_edges=0,
                compression_ratio=0.0,
                reasons=(reason,),
            )

        levels = tuple(
            sorted(
                {
                    value
                    for value in self.policy.resolution_levels
                    if value + EPS >= self.policy.minimum_resolution
                },
                reverse=True,
            )
        )
        selected_profile: ArticulationProfile | None = None
        for resolution in levels:
            profile = profile_rules(
                self.geometry,
                source,
                admissible,
                resolution=resolution,
            )
            if profile.path_multiplicity > 0:
                selected_profile = profile
                break

        if selected_profile is None:
            return ArticulationDecision(
                state=ArticulationState.UNRESOLVED,
                backend=None,
                relation=relation,
                source=source,
                resolution=None,
                selected_rules=admissible,
                assessments=assessments,
                source_support=source_support,
                source_entropy=source_entropy,
                path_multiplicity=0,
                materialized_edges=0,
                compression_ratio=0.0,
                reasons=("no-consistent-path-at-admissible-resolution",),
            )

        compression = selected_profile.compression_ratio
        use_lpu = (
            exact_witness
            or (
                selected_profile.path_multiplicity
                >= self.policy.lpu_min_paths
                and compression + EPS
                >= self.policy.lpu_min_compression_ratio
            )
        )
        backend = (
            ExecutionBackend.LPU if use_lpu else ExecutionBackend.CORTEX
        )
        if exact_witness:
            reasons.append("exact-witness-requested")
        if backend is ExecutionBackend.LPU:
            reasons.append("path-multiplicity-benefits-from-layering")
        else:
            reasons.append("native-execution-is-sufficient")

        return ArticulationDecision(
            state=ArticulationState.EXECUTE,
            backend=backend,
            relation=relation,
            source=source,
            resolution=selected_profile.resolution,
            selected_rules=admissible,
            assessments=assessments,
            source_support=source_support,
            source_entropy=source_entropy,
            path_multiplicity=selected_profile.path_multiplicity,
            materialized_edges=selected_profile.materialized_edges,
            compression_ratio=compression,
            reasons=tuple(reasons),
        )

    def execute(
        self,
        source: str,
        relation: str,
        rules: Iterable[CompositionRule],
        *,
        source_support: float = 1.0,
        source_entropy: float = 0.0,
        exact_witness: bool = False,
    ) -> ControlledExecution:
        decision = self.decide(
            source,
            relation,
            rules,
            source_support=source_support,
            source_entropy=source_entropy,
            exact_witness=exact_witness,
        )
        if (
            decision.state is not ArticulationState.EXECUTE
            or decision.backend is None
            or decision.resolution is None
        ):
            return ControlledExecution(decision, None)

        if decision.backend is ExecutionBackend.LPU:
            executor = LPURuleExecutor(
                self.geometry,
                resolution=decision.resolution,
            )
        else:
            executor = EnumeratingRuleExecutor(
                self.geometry,
                resolution=decision.resolution,
            )
        execution = executor.infer(
            source,
            relation,
            decision.selected_rules,
        )
        return ControlledExecution(decision, execution)
