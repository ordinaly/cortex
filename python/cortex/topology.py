"""Read-only local articulation over Cortex's informational topology.

This module does not create a second memory. It projects existing Cortex
articulation, graph, and evidence state into a sparse local fiber around an
entity or arbitrary observation vector.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import Mapping, Sequence


EPS = 1.0e-12


@dataclass(frozen=True, slots=True)
class EntitySupport:
    entity_id: int
    label: str | None
    support: float
    distance: float
    mean_reliability: float


@dataclass(frozen=True, slots=True)
class FeatureCoordinate:
    index: int
    value: float
    prototype_value: float
    reliability: float
    noise_state: int


@dataclass(frozen=True, slots=True)
class RelationCoordinate:
    source_id: int
    target_id: int
    source_label: str | None
    target_label: str | None
    posterior_mean: float
    evidence_margin: float
    exposure: int
    state: int
    last_updated: int


@dataclass(frozen=True, slots=True)
class CausalCoordinate:
    source_id: int
    target_id: int
    source_label: str | None
    target_label: str | None
    p_do: float
    p_control: float
    effect: float
    do_n: int
    control_n: int
    state: int
    last_updated: int


@dataclass(frozen=True, slots=True)
class LocalArticulation:
    anchor_entity_id: int | None
    anchor_label: str | None
    entities: tuple[EntitySupport, ...]
    dimensions: tuple[FeatureCoordinate, ...]
    relations: tuple[RelationCoordinate, ...]
    causal_relations: tuple[CausalCoordinate, ...]
    entity_entropy: float
    residual_entity_support: float

    @property
    def top_entity(self) -> EntitySupport | None:
        return self.entities[0] if self.entities else None


def _shift_vector(vector: Sequence[float], shift: int) -> tuple[float, ...]:
    size = len(vector)
    if size == 0:
        return ()
    shift %= size
    return tuple(vector[(i + size - shift) % size] for i in range(size))


def _weighted_mse(
    left: Sequence[float],
    right: Sequence[float],
    weights: Sequence[float],
) -> float:
    numerator = sum(
        weight * (a - b) ** 2
        for a, b, weight in zip(left, right, weights)
    )
    denominator = max(EPS, sum(weights))
    return numerator / denominator


class TopologyView:
    """Sparse local readout of a Cortex runtime snapshot."""

    def __init__(
        self,
        snapshot_json: str,
        config_json: str,
        labels: Mapping[int, str] | None = None,
    ):
        self.snapshot = json.loads(snapshot_json)
        self.config = json.loads(config_json)
        self.labels = {int(k): str(v) for k, v in (labels or {}).items()}

        articulation = self.snapshot["articulation"]
        self.prototypes = tuple(
            tuple(float(value) for value in row)
            for row in articulation["prototypes"]
        )
        self.reliability = tuple(
            tuple(float(value) for value in row)
            for row in articulation["reliability"]
        )
        self.noise_state = tuple(
            tuple(int(value) for value in row)
            for row in articulation["noise_state"]
        )
        self.subgroup = tuple(int(value) for value in articulation["subgroup"])
        self.subgroup_margin = float(articulation["subgroup_margin"])
        self.beta = float(self.config["articulation"]["beta"])
        self.feature_dim = int(self.config["articulation"]["feature_dim"])
        fixed = self.config["articulation"].get("fixed_group")
        self.fixed_group = (
            tuple(int(value) for value in fixed)
            if fixed is not None
            else None
        )

    @classmethod
    def from_runtime(
        cls,
        runtime,
        labels: Mapping[int, str] | None = None,
    ) -> "TopologyView":
        return cls(runtime.snapshot_json(), runtime.config_json(), labels)

    def _active_match_group(self) -> tuple[int, ...]:
        if self.fixed_group is not None:
            return self.fixed_group
        if self.subgroup_margin > 0.01:
            return self.subgroup
        return tuple(range(self.feature_dim))

    def _distance(self, vector: Sequence[float], entity_id: int) -> float:
        prototype = self.prototypes[entity_id]
        reliability = self.reliability[entity_id]
        return min(
            _weighted_mse(
                vector,
                _shift_vector(prototype, shift),
                reliability,
            )
            for shift in self._active_match_group()
        )

    def at_entity(
        self,
        entity_id: int,
        *,
        top_k: int = 8,
        min_support: float = 1.0e-6,
        max_relations: int = 64,
    ) -> LocalArticulation:
        if not 0 <= int(entity_id) < len(self.prototypes):
            raise ValueError("entity_id is outside the Cortex articulation")
        return self.at_vector(
            self.prototypes[int(entity_id)],
            anchor_entity_id=int(entity_id),
            top_k=top_k,
            min_support=min_support,
            max_relations=max_relations,
        )

    def at_vector(
        self,
        vector: Sequence[float],
        *,
        anchor_entity_id: int | None = None,
        top_k: int = 8,
        min_support: float = 1.0e-6,
        max_relations: int = 64,
    ) -> LocalArticulation:
        if len(vector) != self.feature_dim:
            raise ValueError(
                f"expected point dimension {self.feature_dim}, got {len(vector)}"
            )
        values = tuple(float(value) for value in vector)
        if any(not math.isfinite(value) for value in values):
            raise ValueError("point contains non-finite values")
        if top_k < 1:
            raise ValueError("top_k must be positive")

        distances = [
            self._distance(values, entity_id)
            for entity_id in range(len(self.prototypes))
        ]
        if distances:
            logits = [-self.beta * distance for distance in distances]
            max_logit = max(logits)
            raw = [math.exp(logit - max_logit) for logit in logits]
            total = max(EPS, sum(raw))
            supports = [value / total for value in raw]
        else:
            supports = []

        ranked = sorted(
            range(len(supports)),
            key=lambda entity_id: (-supports[entity_id], entity_id),
        )
        selected = [
            entity_id
            for entity_id in ranked[:top_k]
            if supports[entity_id] >= min_support
        ]
        entities = tuple(
            EntitySupport(
                entity_id=entity_id,
                label=self.labels.get(entity_id),
                support=supports[entity_id],
                distance=distances[entity_id],
                mean_reliability=(
                    sum(self.reliability[entity_id])
                    / max(1, len(self.reliability[entity_id]))
                ),
            )
            for entity_id in selected
        )

        if supports:
            entropy = -sum(
                support * math.log(max(EPS, support))
                for support in supports
                if support > 0.0
            )
            entropy /= math.log(len(supports)) if len(supports) > 1 else 1.0
        else:
            entropy = 0.0

        selected_support = sum(supports[entity_id] for entity_id in selected)
        residual = max(0.0, 1.0 - selected_support)

        top_id = selected[0] if selected else anchor_entity_id
        dimensions: tuple[FeatureCoordinate, ...] = ()
        if top_id is not None and 0 <= top_id < len(self.prototypes):
            dimensions = tuple(
                FeatureCoordinate(
                    index=index,
                    value=values[index],
                    prototype_value=self.prototypes[top_id][index],
                    reliability=self.reliability[top_id][index],
                    noise_state=self.noise_state[top_id][index],
                )
                for index in range(self.feature_dim)
            )

        local_ids = set(selected)
        relations = self._relation_coordinates(local_ids, max_relations)
        causal = self._causal_coordinates(local_ids, max_relations)

        return LocalArticulation(
            anchor_entity_id=anchor_entity_id,
            anchor_label=(
                self.labels.get(anchor_entity_id)
                if anchor_entity_id is not None
                else None
            ),
            entities=entities,
            dimensions=dimensions,
            relations=relations,
            causal_relations=causal,
            entity_entropy=entropy,
            residual_entity_support=residual,
        )

    def _relation_coordinates(
        self,
        local_ids: set[int],
        limit: int,
    ) -> tuple[RelationCoordinate, ...]:
        coordinates: list[RelationCoordinate] = []
        for entry in self.snapshot["graph"]["relations"]:
            a = int(entry["a"])
            b = int(entry["b"])
            if a not in local_ids and b not in local_ids:
                continue
            evidence = entry["evidence"]
            state = int(evidence["state"])
            if state == -1:
                continue
            alpha = float(evidence["a"])
            beta = float(evidence["b"])
            posterior = alpha / max(EPS, alpha + beta)
            coordinates.append(
                RelationCoordinate(
                    source_id=a,
                    target_id=b,
                    source_label=self.labels.get(a),
                    target_label=self.labels.get(b),
                    posterior_mean=posterior,
                    evidence_margin=2.0 * abs(posterior - 0.5),
                    exposure=int(evidence["exposure"]),
                    state=state,
                    last_updated=int(evidence["last_updated"]),
                )
            )
        coordinates.sort(
            key=lambda row: (
                -int(row.state == 1),
                -row.evidence_margin,
                row.source_id,
                row.target_id,
            )
        )
        return tuple(coordinates[:limit])

    def _causal_coordinates(
        self,
        local_ids: set[int],
        limit: int,
    ) -> tuple[CausalCoordinate, ...]:
        coordinates: list[CausalCoordinate] = []
        for entry in self.snapshot["graph"]["causal"]:
            source = int(entry["source"])
            target = int(entry["target"])
            if source not in local_ids and target not in local_ids:
                continue
            evidence = entry["evidence"]
            do_a = float(evidence["do_a"])
            do_b = float(evidence["do_b"])
            ctrl_a = float(evidence["ctrl_a"])
            ctrl_b = float(evidence["ctrl_b"])
            p_do = do_a / max(EPS, do_a + do_b)
            p_control = ctrl_a / max(EPS, ctrl_a + ctrl_b)
            coordinates.append(
                CausalCoordinate(
                    source_id=source,
                    target_id=target,
                    source_label=self.labels.get(source),
                    target_label=self.labels.get(target),
                    p_do=p_do,
                    p_control=p_control,
                    effect=p_do - p_control,
                    do_n=int(evidence["do_n"]),
                    control_n=int(evidence["ctrl_n"]),
                    state=int(evidence["state"]),
                    last_updated=int(evidence["last_updated"]),
                )
            )
        coordinates.sort(
            key=lambda row: (
                -int(row.state == 1),
                -abs(row.effect),
                row.source_id,
                row.target_id,
            )
        )
        return tuple(coordinates[:limit])
