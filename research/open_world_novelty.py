"""Open-world novelty attribution by predictive-description model selection.

Research-only Cortex Reasoning Map prototype.

The attributor receives a finite structured world model and a bounded evidence
window. It scores competing explanations for residual evidence:

- noise / no structural edit;
- bind previously unseen instances to known entity types;
- introduce a new entity type and bind unseen instances to it;
- introduce one previously absent relation schema;
- introduce a new predictive regime over known entity types.

The winning explanation is accepted only when it has a declared score margin.
Near-ties remain unresolved and produce no model mutation.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
from json import dumps
from math import isfinite, log
from typing import Mapping, Sequence

OutcomeCounts = tuple[int, ...]
ProbabilityVector = tuple[float, ...]
RelationKey = tuple[str, str, str]
RelationCounts = tuple[int, int]

CATEGORIES = (
    "noise",
    "new_instance",
    "new_entity_type",
    "new_relation",
    "new_regime",
)


def _normalized(
    values: Sequence[float],
) -> ProbabilityVector:
    numbers = tuple(float(value) for value in values)
    if not numbers:
        raise ValueError("probability vector must not be empty")
    if any(
        value < 0.0 or not isfinite(value)
        for value in numbers
    ):
        raise ValueError(
            "probabilities must be finite and non-negative"
        )
    total = sum(numbers)
    if total <= 0.0:
        raise ValueError(
            "probabilities must have positive mass"
        )
    return tuple(value / total for value in numbers)


def _multinomial_nll(
    counts: OutcomeCounts,
    probabilities: ProbabilityVector,
) -> float:
    if len(counts) != len(probabilities):
        raise ValueError(
            "count/probability dimensions differ"
        )
    if any(count < 0 for count in counts):
        raise ValueError(
            "counts must be non-negative"
        )
    return -sum(
        float(count) * log(max(probability, 1.0e-15))
        for count, probability
        in zip(counts, probabilities)
    )


def _smoothed_mle(
    counts: Sequence[int],
    *,
    alpha: float,
) -> ProbabilityVector:
    if alpha <= 0.0 or not isfinite(alpha):
        raise ValueError("alpha must be positive")
    values = tuple(int(count) for count in counts)
    if not values or any(count < 0 for count in values):
        raise ValueError(
            "counts must be a non-empty non-negative vector"
        )
    denominator = (
        float(sum(values))
        + alpha * len(values)
    )
    return tuple(
        (count + alpha) / denominator
        for count in values
    )


def _bernoulli_nll(
    counts: RelationCounts,
    probability: float,
) -> float:
    successes, total = counts
    if total < 0 or successes < 0 or successes > total:
        raise ValueError("invalid Bernoulli counts")
    p = min(
        max(float(probability), 1.0e-15),
        1.0 - 1.0e-15,
    )
    return -(
        successes * log(p)
        + (total - successes) * log(1.0 - p)
    )


def _bernoulli_mle(
    counts: RelationCounts,
    *,
    alpha: float,
) -> float:
    successes, total = counts
    return (
        successes + alpha
    ) / (
        total + 2.0 * alpha
    )


def _digest(payload: object) -> str:
    return sha256(
        dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()[:16]


@dataclass(frozen=True)
class WorldModel:
    type_probabilities: Mapping[
        str,
        ProbabilityVector,
    ]
    instances: Mapping[str, str]
    relation_probabilities: Mapping[
        RelationKey,
        float,
    ]
    regime_profiles: Mapping[
        str,
        Mapping[str, ProbabilityVector],
    ] = None
    active_regime: str | None = None
    default_relation_probability: float = 0.05

    def __post_init__(self) -> None:
        if not self.type_probabilities:
            raise ValueError(
                "world requires at least one entity type"
            )
        dimensions = {
            len(probabilities)
            for probabilities
            in self.type_probabilities.values()
        }
        if len(dimensions) != 1:
            raise ValueError(
                "all type distributions must share dimension"
            )
        for probabilities in self.type_probabilities.values():
            _normalized(probabilities)
        for instance, type_id in self.instances.items():
            if not instance:
                raise ValueError(
                    "instance IDs must be non-empty"
                )
            if type_id not in self.type_probabilities:
                raise ValueError(
                    f"unknown type for instance: {type_id}"
                )
        if not (
            0.0
            < self.default_relation_probability
            < 1.0
        ):
            raise ValueError(
                "default relation probability must lie in (0,1)"
            )

        profiles = (
            {}
            if self.regime_profiles is None
            else dict(self.regime_profiles)
        )
        for profile in profiles.values():
            for type_id, probabilities in profile.items():
                if type_id not in self.type_probabilities:
                    raise ValueError(
                        "regime profile references unknown type"
                    )
                if (
                    len(probabilities)
                    != self.outcomes
                ):
                    raise ValueError(
                        "regime distribution dimension mismatch"
                    )
                _normalized(probabilities)

        if (
            self.active_regime is not None
            and self.active_regime not in profiles
        ):
            raise ValueError(
                "active regime is not in regime library"
            )

        object.__setattr__(
            self,
            "regime_profiles",
            profiles,
        )

    @property
    def outcomes(self) -> int:
        return len(
            next(
                iter(
                    self.type_probabilities.values()
                )
            )
        )

    def probabilities_for_type(
        self,
        type_id: str,
    ) -> ProbabilityVector:
        if type_id not in self.type_probabilities:
            raise KeyError(
                f"unknown type: {type_id}"
            )
        if self.active_regime is not None:
            profile = self.regime_profiles[
                self.active_regime
            ]
            if type_id in profile:
                return tuple(
                    profile[type_id]
                )
        return tuple(
            self.type_probabilities[type_id]
        )

    def probabilities_for_instance(
        self,
        instance_id: str,
    ) -> ProbabilityVector:
        try:
            type_id = self.instances[instance_id]
        except KeyError as exc:
            raise KeyError(
                f"unknown instance: {instance_id}"
            ) from exc
        return self.probabilities_for_type(
            type_id
        )

    def global_mixture(
        self,
    ) -> ProbabilityVector:
        distributions = [
            self.probabilities_for_type(type_id)
            for type_id
            in sorted(self.type_probabilities)
        ]
        return tuple(
            sum(
                distribution[index]
                for distribution
                in distributions
            ) / len(distributions)
            for index in range(self.outcomes)
        )

    def relation_probability(
        self,
        key: RelationKey,
    ) -> float:
        return float(
            self.relation_probabilities.get(
                key,
                self.default_relation_probability,
            )
        )

    def component_fingerprints(
        self,
    ) -> dict[str, str]:
        return {
            "types": _digest(
                sorted(
                    (
                        type_id,
                        tuple(probabilities),
                    )
                    for type_id, probabilities
                    in self.type_probabilities.items()
                )
            ),
            "instances": _digest(
                sorted(self.instances.items())
            ),
            "relations": _digest(
                sorted(
                    (
                        tuple(key),
                        float(value),
                    )
                    for key, value
                    in self.relation_probabilities.items()
                )
            ),
            "regimes": _digest(
                sorted(
                    (
                        regime_id,
                        sorted(
                            (
                                type_id,
                                tuple(probabilities),
                            )
                            for type_id, probabilities
                            in profile.items()
                        ),
                    )
                    for regime_id, profile
                    in self.regime_profiles.items()
                )
            ),
            "active_regime": _digest(
                self.active_regime
            ),
        }

    def fingerprint(self) -> str:
        return _digest(
            self.component_fingerprints()
        )


@dataclass(frozen=True)
class EvidenceWindow:
    outcome_counts: Mapping[
        str,
        OutcomeCounts,
    ]
    relation_counts: Mapping[
        tuple[str, str, str],
        RelationCounts,
    ]

    def __post_init__(self) -> None:
        for counts in self.outcome_counts.values():
            if not counts:
                raise ValueError(
                    "outcome counts must not be empty"
                )
            if any(count < 0 for count in counts):
                raise ValueError(
                    "outcome counts must be non-negative"
                )
        for counts in self.relation_counts.values():
            successes, total = counts
            if (
                successes < 0
                or total < 0
                or successes > total
            ):
                raise ValueError(
                    "invalid relation counts"
                )


@dataclass(frozen=True)
class ModelEdit:
    category: str
    payload: tuple


@dataclass(frozen=True)
class Candidate:
    category: str
    score: float
    data_nll: float
    complexity: float
    edit: ModelEdit | None


@dataclass(frozen=True)
class AttributionRead:
    category: str
    resolved: bool
    best_score: float
    second_score: float
    margin: float
    scores: tuple[tuple[str, float], ...]
    proposed_edit: ModelEdit | None

    @property
    def structural_change(self) -> bool:
        return (
            self.resolved
            and self.category
            not in ("noise", "unresolved")
        )


class CortexNoveltyAttributor:
    """Compare finite structural explanations by predictive description."""

    def __init__(
        self,
        model: WorldModel,
        *,
        alpha: float = 0.5,
        instance_penalty: float = 2.0,
        entity_type_penalty: float = 5.0,
        relation_penalty: float = 4.0,
        regime_penalty: float = 8.0,
        regime_type_penalty: float = 2.0,
        resolution_margin: float = 0.5,
        regime_min_instances: int = 3,
    ) -> None:
        if alpha <= 0.0:
            raise ValueError("alpha must be positive")
        if resolution_margin < 0.0:
            raise ValueError(
                "resolution_margin must be non-negative"
            )
        if regime_min_instances < 1:
            raise ValueError(
                "regime_min_instances must be positive"
            )
        self.model = model
        self.alpha = float(alpha)
        self.instance_penalty = float(
            instance_penalty
        )
        self.entity_type_penalty = float(
            entity_type_penalty
        )
        self.relation_penalty = float(
            relation_penalty
        )
        self.regime_penalty = float(
            regime_penalty
        )
        self.regime_type_penalty = float(
            regime_type_penalty
        )
        self.resolution_margin = float(
            resolution_margin
        )
        self.regime_min_instances = int(
            regime_min_instances
        )

    def _validate_window(
        self,
        evidence: EvidenceWindow,
    ) -> None:
        for counts in evidence.outcome_counts.values():
            if len(counts) != self.model.outcomes:
                raise ValueError(
                    "outcome count dimension mismatch"
                )

    def _outcome_nll_current(
        self,
        evidence: EvidenceWindow,
    ) -> float:
        mixture = self.model.global_mixture()
        total = 0.0
        for instance_id, counts in evidence.outcome_counts.items():
            probabilities = (
                self.model.probabilities_for_instance(
                    instance_id
                )
                if instance_id
                in self.model.instances
                else mixture
            )
            total += _multinomial_nll(
                counts,
                probabilities,
            )
        return total

    def _relation_nll_current(
        self,
        evidence: EvidenceWindow,
    ) -> float:
        total = 0.0
        for (
            source,
            target,
            relation,
        ), counts in evidence.relation_counts.items():
            if (
                source not in self.model.instances
                or target not in self.model.instances
            ):
                total += _bernoulli_nll(
                    counts,
                    self.model.default_relation_probability,
                )
                continue
            key = (
                self.model.instances[source],
                self.model.instances[target],
                relation,
            )
            total += _bernoulli_nll(
                counts,
                self.model.relation_probability(key),
            )
        return total

    def _noise(
        self,
        evidence: EvidenceWindow,
    ) -> Candidate:
        data_nll = (
            self._outcome_nll_current(evidence)
            + self._relation_nll_current(evidence)
        )
        return Candidate(
            category="noise",
            score=data_nll,
            data_nll=data_nll,
            complexity=0.0,
            edit=None,
        )

    def _new_instance(
        self,
        evidence: EvidenceWindow,
    ) -> Candidate | None:
        unknown = sorted(
            instance_id
            for instance_id
            in evidence.outcome_counts
            if instance_id
            not in self.model.instances
        )
        if not unknown:
            return None

        relation_nll = self._relation_nll_current(
            evidence
        )
        outcome_nll = 0.0
        bindings: list[tuple[str, str]] = []

        for instance_id, counts in evidence.outcome_counts.items():
            if instance_id in self.model.instances:
                outcome_nll += _multinomial_nll(
                    counts,
                    self.model.probabilities_for_instance(
                        instance_id
                    ),
                )
                continue

            options = [
                (
                    _multinomial_nll(
                        counts,
                        self.model.probabilities_for_type(
                            type_id
                        ),
                    ),
                    type_id,
                )
                for type_id
                in sorted(
                    self.model.type_probabilities
                )
            ]
            score, type_id = min(options)
            outcome_nll += score
            bindings.append(
                (instance_id, type_id)
            )

        complexity = (
            self.instance_penalty
            * len(bindings)
        )
        data_nll = outcome_nll + relation_nll
        return Candidate(
            category="new_instance",
            score=data_nll + complexity,
            data_nll=data_nll,
            complexity=complexity,
            edit=ModelEdit(
                category="new_instance",
                payload=tuple(bindings),
            ),
        )

    def _new_entity_type(
        self,
        evidence: EvidenceWindow,
    ) -> Candidate | None:
        unknown = sorted(
            instance_id
            for instance_id
            in evidence.outcome_counts
            if instance_id
            not in self.model.instances
        )
        if not unknown:
            return None

        pooled = [
            0
            for _ in range(self.model.outcomes)
        ]
        for instance_id in unknown:
            for index, count in enumerate(
                evidence.outcome_counts[
                    instance_id
                ]
            ):
                pooled[index] += int(count)

        probabilities = _smoothed_mle(
            pooled,
            alpha=self.alpha,
        )
        outcome_nll = 0.0
        for instance_id, counts in evidence.outcome_counts.items():
            if instance_id in self.model.instances:
                outcome_nll += _multinomial_nll(
                    counts,
                    self.model.probabilities_for_instance(
                        instance_id
                    ),
                )
            else:
                outcome_nll += _multinomial_nll(
                    counts,
                    probabilities,
                )

        relation_nll = self._relation_nll_current(
            evidence
        )
        complexity = (
            self.entity_type_penalty
            + self.instance_penalty
            * len(unknown)
        )
        type_id = (
            "novel-type-"
            + _digest(
                (
                    tuple(unknown),
                    probabilities,
                )
            )
        )
        data_nll = outcome_nll + relation_nll
        return Candidate(
            category="new_entity_type",
            score=data_nll + complexity,
            data_nll=data_nll,
            complexity=complexity,
            edit=ModelEdit(
                category="new_entity_type",
                payload=(
                    type_id,
                    probabilities,
                    tuple(unknown),
                ),
            ),
        )

    def _new_relation(
        self,
        evidence: EvidenceWindow,
    ) -> Candidate | None:
        candidates = []
        baseline_outcome_nll = (
            self._outcome_nll_current(evidence)
        )

        for (
            source,
            target,
            relation,
        ), counts in evidence.relation_counts.items():
            if (
                source not in self.model.instances
                or target not in self.model.instances
            ):
                continue
            key = (
                self.model.instances[source],
                self.model.instances[target],
                relation,
            )
            if key in self.model.relation_probabilities:
                continue

            fitted = _bernoulli_mle(
                counts,
                alpha=self.alpha,
            )
            relation_nll = 0.0
            for raw_key, raw_counts in (
                evidence.relation_counts.items()
            ):
                raw_source, raw_target, raw_relation = raw_key
                if (
                    raw_source in self.model.instances
                    and raw_target in self.model.instances
                ):
                    typed_key = (
                        self.model.instances[
                            raw_source
                        ],
                        self.model.instances[
                            raw_target
                        ],
                        raw_relation,
                    )
                else:
                    typed_key = None

                probability = (
                    fitted
                    if typed_key == key
                    else (
                        self.model.relation_probability(
                            typed_key
                        )
                        if typed_key is not None
                        else self.model.default_relation_probability
                    )
                )
                relation_nll += _bernoulli_nll(
                    raw_counts,
                    probability,
                )

            data_nll = (
                baseline_outcome_nll
                + relation_nll
            )
            candidates.append(
                Candidate(
                    category="new_relation",
                    score=(
                        data_nll
                        + self.relation_penalty
                    ),
                    data_nll=data_nll,
                    complexity=self.relation_penalty,
                    edit=ModelEdit(
                        category="new_relation",
                        payload=(
                            key,
                            fitted,
                        ),
                    ),
                )
            )

        if not candidates:
            return None
        return min(
            candidates,
            key=lambda candidate: (
                candidate.score,
                candidate.edit.payload,
            ),
        )

    def _new_regime(
        self,
        evidence: EvidenceWindow,
    ) -> Candidate | None:
        known = [
            instance_id
            for instance_id
            in evidence.outcome_counts
            if instance_id
            in self.model.instances
        ]
        if len(known) < self.regime_min_instances:
            return None

        pooled_by_type: dict[str, list[int]] = {}
        for instance_id in known:
            type_id = self.model.instances[
                instance_id
            ]
            pooled = pooled_by_type.setdefault(
                type_id,
                [
                    0
                    for _ in range(
                        self.model.outcomes
                    )
                ],
            )
            for index, count in enumerate(
                evidence.outcome_counts[
                    instance_id
                ]
            ):
                pooled[index] += int(count)

        profile = {
            type_id: _smoothed_mle(
                counts,
                alpha=self.alpha,
            )
            for type_id, counts
            in pooled_by_type.items()
        }

        mixture = self.model.global_mixture()
        outcome_nll = 0.0
        for instance_id, counts in evidence.outcome_counts.items():
            if instance_id in self.model.instances:
                type_id = self.model.instances[
                    instance_id
                ]
                probabilities = profile[
                    type_id
                ]
            else:
                probabilities = mixture
            outcome_nll += _multinomial_nll(
                counts,
                probabilities,
            )

        relation_nll = self._relation_nll_current(
            evidence
        )
        complexity = (
            self.regime_penalty
            + self.regime_type_penalty
            * len(profile)
        )
        regime_id = (
            "novel-regime-"
            + _digest(
                sorted(
                    (
                        type_id,
                        probabilities,
                    )
                    for type_id, probabilities
                    in profile.items()
                )
            )
        )
        data_nll = outcome_nll + relation_nll
        return Candidate(
            category="new_regime",
            score=data_nll + complexity,
            data_nll=data_nll,
            complexity=complexity,
            edit=ModelEdit(
                category="new_regime",
                payload=(
                    regime_id,
                    tuple(
                        sorted(
                            profile.items()
                        )
                    ),
                ),
            ),
        )

    def candidates(
        self,
        evidence: EvidenceWindow,
    ) -> tuple[Candidate, ...]:
        self._validate_window(evidence)
        candidates = [
            self._noise(evidence),
        ]
        for candidate in (
            self._new_instance(evidence),
            self._new_entity_type(evidence),
            self._new_relation(evidence),
            self._new_regime(evidence),
        ):
            if candidate is not None:
                candidates.append(candidate)
        return tuple(candidates)

    def evaluate(
        self,
        evidence: EvidenceWindow,
    ) -> AttributionRead:
        candidates = sorted(
            self.candidates(evidence),
            key=lambda candidate: (
                candidate.score,
                CATEGORIES.index(
                    candidate.category
                ),
            ),
        )
        best = candidates[0]
        second_score = (
            candidates[1].score
            if len(candidates) >= 2
            else float("inf")
        )
        margin = second_score - best.score

        resolved = (
            margin >= self.resolution_margin
        )
        category = (
            best.category
            if resolved
            else "unresolved"
        )
        edit = (
            best.edit
            if resolved
            and best.category != "noise"
            else None
        )
        return AttributionRead(
            category=category,
            resolved=resolved,
            best_score=best.score,
            second_score=second_score,
            margin=margin,
            scores=tuple(
                (
                    candidate.category,
                    candidate.score,
                )
                for candidate in candidates
            ),
            proposed_edit=edit,
        )


def apply_attribution(
    model: WorldModel,
    read: AttributionRead,
) -> WorldModel:
    """Apply one resolved structural proposal; unresolved/noise is immutable."""

    if (
        not read.resolved
        or read.category == "noise"
        or read.proposed_edit is None
    ):
        return model

    edit = read.proposed_edit
    if edit.category != read.category:
        raise ValueError(
            "attribution/edit category mismatch"
        )

    if edit.category == "new_instance":
        instances = dict(model.instances)
        for instance_id, type_id in edit.payload:
            instances[instance_id] = type_id
        return replace(
            model,
            instances=instances,
        )

    if edit.category == "new_entity_type":
        (
            type_id,
            probabilities,
            instance_ids,
        ) = edit.payload
        types = dict(model.type_probabilities)
        types[type_id] = tuple(probabilities)
        instances = dict(model.instances)
        for instance_id in instance_ids:
            instances[instance_id] = type_id
        return replace(
            model,
            type_probabilities=types,
            instances=instances,
        )

    if edit.category == "new_relation":
        key, probability = edit.payload
        relations = dict(
            model.relation_probabilities
        )
        relations[tuple(key)] = float(
            probability
        )
        return replace(
            model,
            relation_probabilities=relations,
        )

    if edit.category == "new_regime":
        regime_id, profile_items = edit.payload
        regimes = {
            key: dict(value)
            for key, value
            in model.regime_profiles.items()
        }
        regimes[regime_id] = {
            type_id: tuple(probabilities)
            for type_id, probabilities
            in profile_items
        }
        return replace(
            model,
            regime_profiles=regimes,
            active_regime=regime_id,
        )

    raise ValueError(
        f"unsupported edit category: {edit.category}"
    )


def changed_components(
    before: WorldModel,
    after: WorldModel,
) -> tuple[str, ...]:
    left = before.component_fingerprints()
    right = after.component_fingerprints()
    return tuple(
        key
        for key in sorted(left)
        if left[key] != right[key]
    )


class NearestKnownTypeBaseline:
    """Always assimilate unseen instances; otherwise call broad drift a regime."""

    def predict(
        self,
        model: WorldModel,
        evidence: EvidenceWindow,
    ) -> str:
        if any(
            instance_id not in model.instances
            for instance_id
            in evidence.outcome_counts
        ):
            return "new_instance"

        for (
            source,
            target,
            relation,
        ), counts in evidence.relation_counts.items():
            if (
                source in model.instances
                and target in model.instances
            ):
                key = (
                    model.instances[source],
                    model.instances[target],
                    relation,
                )
                if (
                    key not in model.relation_probabilities
                    and counts[1] >= 8
                    and counts[0] / max(
                        counts[1],
                        1,
                    ) >= 0.5
                ):
                    return "noise"

        known_evidence = sum(
            1
            for instance_id
            in evidence.outcome_counts
            if instance_id in model.instances
        )
        return (
            "new_regime"
            if known_evidence >= 3
            else "noise"
        )


class GlobalResetBaseline:
    """Treat any sufficiently repeated residual as a global regime change."""

    def predict(
        self,
        model: WorldModel,
        evidence: EvidenceWindow,
    ) -> str:
        sample_mass = sum(
            sum(counts)
            for counts in evidence.outcome_counts.values()
        ) + sum(
            total
            for _, total
            in evidence.relation_counts.values()
        )
        return (
            "new_regime"
            if sample_mass >= 20
            else "noise"
        )
