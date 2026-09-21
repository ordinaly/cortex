"""Learned calibration surfaces for the Cortex articulation controller.

The learned surfaces are deliberately small and interpretable. They operate on
structural/epistemic features only; relation names never enter calibration.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
import math
from typing import Iterable, Mapping, Sequence

from .articulation_controller import (
    ArticulationState,
    ControllerPolicy,
    ExecutionBackend,
    profile_rules,
)
from .hypergraph_geometry import HypergraphGeometry
from .hypergraph_inference import CompositionRule
from .hypergraph_lpu import (
    EnumeratingRuleExecutor,
    LPURuleExecutor,
    RuleExecutionRead,
)


EPS = 1.0e-12


def _clamp01(value: float) -> float:
    if not math.isfinite(value):
        raise ValueError("surface feature must be finite")
    return min(1.0, max(0.0, float(value)))


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


@dataclass(frozen=True, slots=True)
class SurfaceObservation:
    features: tuple[float, ...]
    label: bool
    weight: float = 1.0

    def __post_init__(self) -> None:
        if not self.features:
            raise ValueError("surface observation must contain features")
        if not math.isfinite(self.weight) or self.weight <= 0.0:
            raise ValueError("observation weight must be finite and positive")
        if any(not math.isfinite(value) for value in self.features):
            raise ValueError("observation features must be finite")


@dataclass(frozen=True, slots=True)
class LinearProbabilitySurface:
    feature_names: tuple[str, ...]
    weights: tuple[float, ...]
    intercept: float
    threshold: float = 0.5
    monotone_positive: bool = True

    def __post_init__(self) -> None:
        if not self.feature_names:
            raise ValueError("surface must contain at least one feature")
        if len(self.feature_names) != len(self.weights):
            raise ValueError("feature and weight dimensions differ")
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("threshold must lie in [0,1]")
        if self.monotone_positive and any(weight < -EPS for weight in self.weights):
            raise ValueError("monotone-positive surface has a negative weight")

    def score(self, features: Sequence[float]) -> float:
        if len(features) != len(self.weights):
            raise ValueError("feature dimension mismatch")
        return self.intercept + sum(
            weight * float(value)
            for weight, value in zip(self.weights, features)
        )

    def probability(self, features: Sequence[float]) -> float:
        return _sigmoid(self.score(features))

    def decide(self, features: Sequence[float]) -> bool:
        return self.probability(features) + EPS >= self.threshold

    def with_threshold(self, threshold: float) -> "LinearProbabilitySurface":
        return replace(self, threshold=_clamp01(threshold))


@dataclass(frozen=True, slots=True)
class CalibrationBundle:
    version: str
    execute_surface: LinearProbabilitySurface
    unresolved_surface: LinearProbabilitySurface
    backend_surface: LinearProbabilitySurface
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
    risk_contract: str = "future-oracle-v0.1"

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, payload: str) -> "CalibrationBundle":
        raw = json.loads(payload)
        return cls(
            version=str(raw["version"]),
            execute_surface=LinearProbabilitySurface(
                feature_names=tuple(
                    raw["execute_surface"]["feature_names"]
                ),
                weights=tuple(raw["execute_surface"]["weights"]),
                intercept=float(raw["execute_surface"]["intercept"]),
                threshold=float(raw["execute_surface"]["threshold"]),
                monotone_positive=bool(
                    raw["execute_surface"]["monotone_positive"]
                ),
            ),
            unresolved_surface=LinearProbabilitySurface(
                feature_names=tuple(
                    raw["unresolved_surface"]["feature_names"]
                ),
                weights=tuple(raw["unresolved_surface"]["weights"]),
                intercept=float(raw["unresolved_surface"]["intercept"]),
                threshold=float(raw["unresolved_surface"]["threshold"]),
                monotone_positive=bool(
                    raw["unresolved_surface"]["monotone_positive"]
                ),
            ),
            backend_surface=LinearProbabilitySurface(
                feature_names=tuple(
                    raw["backend_surface"]["feature_names"]
                ),
                weights=tuple(raw["backend_surface"]["weights"]),
                intercept=float(raw["backend_surface"]["intercept"]),
                threshold=float(raw["backend_surface"]["threshold"]),
                monotone_positive=bool(
                    raw["backend_surface"]["monotone_positive"]
                ),
            ),
            minimum_resolution=float(raw["minimum_resolution"]),
            resolution_levels=tuple(raw["resolution_levels"]),
            risk_contract=str(raw["risk_contract"]),
        )


@dataclass(frozen=True, slots=True)
class CalibratedRuleAssessment:
    rule: CompositionRule
    competitor_confidence: float
    competitor_examples: int
    total_examples: int
    confidence_margin: float
    execute_probability: float
    unresolved_probability: float


@dataclass(frozen=True, slots=True)
class CalibratedDecision:
    state: ArticulationState
    backend: ExecutionBackend | None
    relation: str
    source: str
    resolution: float | None
    selected_rules: tuple[CompositionRule, ...]
    assessments: tuple[CalibratedRuleAssessment, ...]
    source_support: float
    source_entropy: float
    path_multiplicity: int
    materialized_edges: int
    compression_ratio: float
    backend_probability: float
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CalibratedExecution:
    decision: CalibratedDecision
    execution: RuleExecutionRead | None


def fit_logistic_surface(
    observations: Sequence[SurfaceObservation],
    feature_names: Sequence[str],
    *,
    epochs: int = 1800,
    learning_rate: float = 0.16,
    l2: float = 1.0e-3,
    monotone_positive: bool = True,
) -> LinearProbabilitySurface:
    """Fit a tiny deterministic logistic surface with projected gradients."""

    if not observations:
        raise ValueError("cannot fit an empty calibration surface")
    names = tuple(feature_names)
    dimension = len(names)
    if dimension == 0:
        raise ValueError("feature_names must be non-empty")
    if any(len(row.features) != dimension for row in observations):
        raise ValueError("observation feature dimension mismatch")

    positives = sum(row.weight for row in observations if row.label)
    negatives = sum(row.weight for row in observations if not row.label)
    total = positives + negatives
    if positives <= 0.0 or negatives <= 0.0:
        raise ValueError("surface fitting requires both classes")

    positive_scale = total / (2.0 * positives)
    negative_scale = total / (2.0 * negatives)

    weights = [0.0] * dimension
    base_rate = positives / total
    intercept = math.log(base_rate / max(EPS, 1.0 - base_rate))

    for epoch in range(epochs):
        grad_w = [0.0] * dimension
        grad_b = 0.0
        normalizer = 0.0
        for row in observations:
            class_scale = positive_scale if row.label else negative_scale
            row_weight = row.weight * class_scale
            probability = _sigmoid(
                intercept
                + sum(
                    weight * value
                    for weight, value in zip(weights, row.features)
                )
            )
            error = probability - float(row.label)
            grad_b += row_weight * error
            for index, value in enumerate(row.features):
                grad_w[index] += row_weight * error * value
            normalizer += row_weight

        normalizer = max(EPS, normalizer)
        rate = learning_rate / math.sqrt(1.0 + epoch / 250.0)
        intercept -= rate * grad_b / normalizer
        for index in range(dimension):
            gradient = grad_w[index] / normalizer + l2 * weights[index]
            weights[index] -= rate * gradient
            if monotone_positive and weights[index] < 0.0:
                weights[index] = 0.0

    return LinearProbabilitySurface(
        feature_names=names,
        weights=tuple(weights),
        intercept=intercept,
        threshold=0.5,
        monotone_positive=monotone_positive,
    )


def select_f1_threshold(
    surface: LinearProbabilitySurface,
    observations: Sequence[SurfaceObservation],
) -> LinearProbabilitySurface:
    """Choose a held-out threshold that maximizes weighted F1."""

    scored = tuple(
        (
            surface.probability(row.features),
            bool(row.label),
            row.weight,
        )
        for row in observations
    )
    candidates = sorted(
        {0.0, 0.5, 1.0, *(score for score, _label, _weight in scored)}
    )
    best: tuple[float, float, float] | None = None
    best_threshold = 0.5
    for threshold in candidates:
        tp = fp = fn = tn = 0.0
        for score, label, weight in scored:
            prediction = score + EPS >= threshold
            if label and prediction:
                tp += weight
            elif label:
                fn += weight
            elif prediction:
                fp += weight
            else:
                tn += weight
        precision = tp / max(EPS, tp + fp)
        recall = tp / max(EPS, tp + fn)
        f1 = (
            2.0 * precision * recall / max(EPS, precision + recall)
            if tp > 0.0
            else 0.0
        )
        accuracy = (tp + tn) / max(EPS, tp + tn + fp + fn)
        key = (f1, accuracy, -threshold)
        if best is None or key > best:
            best = key
            best_threshold = threshold
    return surface.with_threshold(best_threshold)


def select_precision_threshold(
    surface: LinearProbabilitySurface,
    observations: Sequence[SurfaceObservation],
    *,
    minimum_precision: float,
) -> LinearProbabilitySurface:
    """Choose the most permissive threshold satisfying held-out precision."""

    minimum_precision = _clamp01(minimum_precision)
    scored = sorted(
        (
            surface.probability(row.features),
            bool(row.label),
            row.weight,
        )
        for row in observations
    )
    candidates = sorted(
        {0.0, 0.5, 1.0, *(score for score, _label, _weight in scored)}
    )

    best: tuple[float, float] | None = None
    # maximize weighted recall, then choose the lower threshold.
    positive_total = sum(weight for _score, label, weight in scored if label)
    for threshold in candidates:
        tp = fp = 0.0
        for score, label, weight in scored:
            if score + EPS < threshold:
                continue
            if label:
                tp += weight
            else:
                fp += weight
        if tp <= 0.0:
            continue
        precision = tp / max(EPS, tp + fp)
        if precision + EPS < minimum_precision:
            continue
        recall = tp / max(EPS, positive_total)
        candidate = (recall, -threshold)
        if best is None or candidate > best:
            best = candidate

    threshold = -best[1] if best is not None else 1.0
    return surface.with_threshold(threshold)


def _evidence_scale(total_examples: int) -> float:
    return _clamp01(math.log1p(max(0, total_examples)) / math.log(33.0))


def epistemic_execute_features(
    *,
    total_examples: int,
    confidence: float,
    margin: float,
    mean_support: float,
    source_support: float,
    source_entropy: float,
) -> tuple[float, ...]:
    return (
        _evidence_scale(total_examples),
        _clamp01(confidence),
        _clamp01(max(0.0, margin)),
        _clamp01(mean_support),
        _clamp01(source_support),
        _clamp01(1.0 - source_entropy),
    )


EXECUTE_FEATURE_NAMES = (
    "evidence",
    "rule_confidence",
    "rule_margin",
    "mean_support",
    "source_support",
    "source_certainty",
)


def epistemic_unresolved_features(
    *,
    total_examples: int,
    confidence: float,
    competitor_confidence: float,
    margin: float,
    mean_support: float,
    source_entropy: float,
) -> tuple[float, ...]:
    return (
        _evidence_scale(total_examples),
        _clamp01(competitor_confidence),
        _clamp01(1.0 - max(0.0, margin)),
        _clamp01(1.0 - confidence),
        _clamp01(1.0 - mean_support),
        _clamp01(1.0 - source_entropy),
    )


UNRESOLVED_FEATURE_NAMES = (
    "evidence",
    "competitor_confidence",
    "conflict",
    "low_target_confidence",
    "low_structural_support",
    "source_certainty",
)


def backend_features(
    *,
    path_multiplicity: int,
    materialized_edges: int,
) -> tuple[float, ...]:
    paths = max(0, int(path_multiplicity))
    edges = max(0, int(materialized_edges))
    ratio = paths / max(1, edges)
    return (
        _clamp01(math.log1p(paths) / math.log(1_000_001.0)),
        _clamp01(ratio / (1.0 + ratio)),
    )


BACKEND_FEATURE_NAMES = (
    "path_scale",
    "compression",
)


class CalibratedArticulationController:
    """Controller whose decision surfaces are learned from calibration data."""

    def __init__(
        self,
        geometry: HypergraphGeometry,
        calibration: CalibrationBundle,
    ) -> None:
        self.geometry = geometry
        self.calibration = calibration

    @staticmethod
    def _competitor(
        rule: CompositionRule,
        rules: tuple[CompositionRule, ...],
    ) -> CompositionRule | None:
        candidates = [
            other
            for other in rules
            if (
                other.signature == rule.signature
                and other.target_relation != rule.target_relation
            )
        ]
        if not candidates:
            return None
        return max(
            candidates,
            key=lambda row: (
                row.confidence,
                row.examples,
                row.mean_support,
                row.target_relation,
            ),
        )

    def _assess(
        self,
        relation: str,
        rules: tuple[CompositionRule, ...],
        *,
        source_support: float,
        source_entropy: float,
    ) -> tuple[CalibratedRuleAssessment, ...]:
        out: list[CalibratedRuleAssessment] = []
        for rule in rules:
            if rule.target_relation != relation:
                continue
            competitor = self._competitor(rule, rules)
            competitor_confidence = (
                competitor.confidence if competitor is not None else 0.0
            )
            competitor_examples = (
                competitor.examples if competitor is not None else 0
            )
            total_examples = rule.examples + competitor_examples
            margin = rule.confidence - competitor_confidence

            execute_features = epistemic_execute_features(
                total_examples=total_examples,
                confidence=rule.confidence,
                margin=margin,
                mean_support=rule.mean_support,
                source_support=source_support,
                source_entropy=source_entropy,
            )
            unresolved_features = epistemic_unresolved_features(
                total_examples=total_examples,
                confidence=rule.confidence,
                competitor_confidence=competitor_confidence,
                margin=margin,
                mean_support=rule.mean_support,
                source_entropy=source_entropy,
            )
            out.append(
                CalibratedRuleAssessment(
                    rule=rule,
                    competitor_confidence=competitor_confidence,
                    competitor_examples=competitor_examples,
                    total_examples=total_examples,
                    confidence_margin=margin,
                    execute_probability=(
                        self.calibration.execute_surface.probability(
                            execute_features
                        )
                    ),
                    unresolved_probability=(
                        self.calibration.unresolved_surface.probability(
                            unresolved_features
                        )
                    ),
                )
            )
        out.sort(
            key=lambda row: (
                -row.execute_probability,
                -row.rule.confidence,
                -row.total_examples,
                tuple(
                    (step.relation, step.orientation)
                    for step in row.rule.signature
                ),
            )
        )
        return tuple(out)

    def decide(
        self,
        source: str,
        relation: str,
        rules: Iterable[CompositionRule],
        *,
        source_support: float = 1.0,
        source_entropy: float = 0.0,
        exact_witness: bool = False,
    ) -> CalibratedDecision:
        source_support = _clamp01(source_support)
        source_entropy = _clamp01(source_entropy)
        rules = tuple(rules)
        assessments = self._assess(
            relation,
            rules,
            source_support=source_support,
            source_entropy=source_entropy,
        )

        if not assessments:
            return CalibratedDecision(
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
                backend_probability=0.0,
                reasons=("no-composition-evidence",),
            )

        selected = tuple(
            row.rule
            for row in assessments
            if (
                row.execute_probability + EPS
                >= self.calibration.execute_surface.threshold
            )
        )

        if not selected:
            strongest_unresolved = max(
                row.unresolved_probability for row in assessments
            )
            if (
                strongest_unresolved + EPS
                >= self.calibration.unresolved_surface.threshold
            ):
                state = ArticulationState.UNRESOLVED
                reason = "learned-conflict-surface"
            else:
                state = ArticulationState.EXPLORE
                reason = "learned-exploration-surface"
            return CalibratedDecision(
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
                backend_probability=0.0,
                reasons=(reason,),
            )

        levels = tuple(
            sorted(
                {
                    float(value)
                    for value in self.calibration.resolution_levels
                    if value + EPS >= self.calibration.minimum_resolution
                },
                reverse=True,
            )
        )
        profile = None
        for resolution in levels:
            candidate = profile_rules(
                self.geometry,
                source,
                selected,
                resolution=resolution,
            )
            if candidate.path_multiplicity > 0:
                profile = candidate
                break

        if profile is None:
            return CalibratedDecision(
                state=ArticulationState.UNRESOLVED,
                backend=None,
                relation=relation,
                source=source,
                resolution=None,
                selected_rules=selected,
                assessments=assessments,
                source_support=source_support,
                source_entropy=source_entropy,
                path_multiplicity=0,
                materialized_edges=0,
                compression_ratio=0.0,
                backend_probability=0.0,
                reasons=("no-consistent-path-at-calibrated-resolution",),
            )

        bfeatures = backend_features(
            path_multiplicity=profile.path_multiplicity,
            materialized_edges=profile.materialized_edges,
        )
        backend_probability = (
            self.calibration.backend_surface.probability(bfeatures)
        )
        use_lpu = (
            exact_witness
            or backend_probability + EPS
            >= self.calibration.backend_surface.threshold
        )
        backend = (
            ExecutionBackend.LPU if use_lpu else ExecutionBackend.CORTEX
        )
        reasons = []
        if exact_witness:
            reasons.append("exact-witness-requested")
        reasons.append(
            "learned-backend-surface:lpu"
            if use_lpu
            else "learned-backend-surface:cortex"
        )

        return CalibratedDecision(
            state=ArticulationState.EXECUTE,
            backend=backend,
            relation=relation,
            source=source,
            resolution=profile.resolution,
            selected_rules=selected,
            assessments=assessments,
            source_support=source_support,
            source_entropy=source_entropy,
            path_multiplicity=profile.path_multiplicity,
            materialized_edges=profile.materialized_edges,
            compression_ratio=profile.compression_ratio,
            backend_probability=backend_probability,
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
    ) -> CalibratedExecution:
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
            return CalibratedExecution(decision, None)

        executor = (
            LPURuleExecutor(
                self.geometry,
                resolution=decision.resolution,
            )
            if decision.backend is ExecutionBackend.LPU
            else EnumeratingRuleExecutor(
                self.geometry,
                resolution=decision.resolution,
            )
        )
        return CalibratedExecution(
            decision=decision,
            execution=executor.infer(
                source,
                relation,
                decision.selected_rules,
            ),
        )
