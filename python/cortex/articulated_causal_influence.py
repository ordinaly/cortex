"""Research-only causal influence over integrated Cortex articulations.

C is estimated from controlled intervention/control samples, not from temporal
succession or passive correlation.  Each outcome is a SimplicialIntegrationRead.
The effect is the distinguishability between the two empirical post-condition
articulation distributions, measured across protected marginal dimensions.

Causal interpretation requires that the intervention/control assignment is
exogenous (or otherwise validly controlled).  This module does not infer do()
semantics from passive observations.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from itertools import combinations
from math import isfinite
from typing import Hashable, Mapping

from .probe_economy import Probe
from .simplicial_integration import SimplicialIntegrationRead


Distribution = dict[tuple[Hashable, ...], float]


def _tv(left: Mapping[tuple, float], right: Mapping[tuple, float]) -> float:
    keys = set(left) | set(right)
    return 0.5 * sum(abs(left.get(key, 0.0) - right.get(key, 0.0)) for key in keys)


@dataclass(frozen=True, slots=True)
class CausalArticulationObservation:
    source: str
    target: str
    intervened: bool
    outcome: SimplicialIntegrationRead
    weight: float = 1.0

    def __post_init__(self) -> None:
        if not self.source or not self.target:
            raise ValueError("causal source and target must be non-empty")
        if type(self.intervened) is not bool:
            raise ValueError("intervened must be boolean")
        if not isfinite(float(self.weight)) or float(self.weight) <= 0.0:
            raise ValueError("causal observation weight must be finite and positive")


@dataclass(frozen=True, slots=True)
class ArticulatedCausalPolicy:
    max_dimension: int = 2
    dimension_weights: tuple[float, ...] = (0.15, 0.25, 0.60)
    min_group_observations: int = 8
    window: int = 128

    def __post_init__(self) -> None:
        if type(self.max_dimension) is not int or self.max_dimension < 0:
            raise ValueError("max_dimension must be a non-negative integer")
        if len(self.dimension_weights) <= self.max_dimension:
            raise ValueError("dimension_weights must cover max_dimension")
        for value in self.dimension_weights:
            if not isfinite(float(value)) or float(value) < 0.0:
                raise ValueError("dimension weights must be finite and non-negative")
        if sum(self.dimension_weights[: self.max_dimension + 1]) <= 0.0:
            raise ValueError("at least one used dimension weight must be positive")
        if type(self.min_group_observations) is not int or self.min_group_observations < 1:
            raise ValueError("min_group_observations must be positive")
        if type(self.window) is not int or self.window < self.min_group_observations:
            raise ValueError("window must cover min_group_observations")


@dataclass(frozen=True, slots=True)
class CausalInfluenceRead:
    influence: float
    supported: bool
    by_dimension: tuple[tuple[int, float], ...]
    full_joint_distance: float
    shared_probes: tuple[Probe, ...]
    intervention_observations: int
    control_observations: int


class ArticulatedCausalInfluence:
    """Controlled distributional causal effect over Cortex articulations."""

    def __init__(self, *, policy: ArticulatedCausalPolicy | None = None) -> None:
        self.policy = policy or ArticulatedCausalPolicy()
        self._intervention: dict[tuple[str, str], deque[CausalArticulationObservation]] = {}
        self._control: dict[tuple[str, str], deque[CausalArticulationObservation]] = {}

    def observe(self, observation: CausalArticulationObservation) -> None:
        key = (observation.source, observation.target)
        table = self._intervention if observation.intervened else self._control
        row = table.setdefault(key, deque(maxlen=self.policy.window))
        row.append(observation)

    @staticmethod
    def _marginal(read: SimplicialIntegrationRead, support: tuple[Probe, ...]) -> Distribution:
        index = {probe: i for i, probe in enumerate(read.probes)}
        if any(probe not in index for probe in support):
            raise ValueError("marginal support is absent from articulation")
        positions = tuple(index[probe] for probe in support)
        result: Distribution = {}
        for state, probability in read.distribution:
            key = tuple(state[position] for position in positions)
            result[key] = result.get(key, 0.0) + probability
        return result

    @staticmethod
    def _shared_probes(rows: tuple[CausalArticulationObservation, ...]) -> tuple[Probe, ...]:
        if not rows:
            return ()
        shared = set(rows[0].outcome.probes)
        for row in rows[1:]:
            shared.intersection_update(row.outcome.probes)
        return tuple(probe for probe in rows[0].outcome.probes if probe in shared)

    def _mixture(
        self,
        rows: tuple[CausalArticulationObservation, ...],
        support: tuple[Probe, ...],
    ) -> Distribution:
        result: Distribution = {}
        total = sum(float(row.weight) for row in rows)
        for row in rows:
            fraction = float(row.weight) / total
            for key, probability in self._marginal(row.outcome, support).items():
                result[key] = result.get(key, 0.0) + fraction * probability
        return result

    def effect(self, source: str, target: str) -> CausalInfluenceRead:
        key = (source, target)
        intervention = tuple(self._intervention.get(key, ()))
        control = tuple(self._control.get(key, ()))
        supported = (
            len(intervention) >= self.policy.min_group_observations
            and len(control) >= self.policy.min_group_observations
        )
        if not supported:
            return CausalInfluenceRead(
                influence=0.0,
                supported=False,
                by_dimension=(),
                full_joint_distance=0.0,
                shared_probes=(),
                intervention_observations=len(intervention),
                control_observations=len(control),
            )

        all_rows = intervention + control
        shared = self._shared_probes(all_rows)
        if not shared:
            return CausalInfluenceRead(
                influence=0.0,
                supported=False,
                by_dimension=(),
                full_joint_distance=0.0,
                shared_probes=(),
                intervention_observations=len(intervention),
                control_observations=len(control),
            )

        weighted = 0.0
        weight_sum = 0.0
        by_dimension: list[tuple[int, float]] = []
        upper = min(self.policy.max_dimension, len(shared) - 1)
        for dimension in range(upper + 1):
            distances = []
            for support in combinations(shared, dimension + 1):
                do_mix = self._mixture(intervention, support)
                control_mix = self._mixture(control, support)
                distances.append(_tv(do_mix, control_mix))
            value = sum(distances) / len(distances)
            by_dimension.append((dimension, value))
            weight = self.policy.dimension_weights[dimension]
            weighted += weight * value
            weight_sum += weight

        influence = weighted / weight_sum
        full_joint_distance = _tv(
            self._mixture(intervention, shared),
            self._mixture(control, shared),
        )
        return CausalInfluenceRead(
            influence=influence,
            supported=True,
            by_dimension=tuple(by_dimension),
            full_joint_distance=full_joint_distance,
            shared_probes=shared,
            intervention_observations=len(intervention),
            control_observations=len(control),
        )

    def influence(self, source: str, target: str) -> float:
        return self.effect(source, target).influence
