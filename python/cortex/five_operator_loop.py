"""Five-operator articulation loop for Cortex research.

This module implements a research-only shared articulation substrate for:

- Delta: local distinguishability operators;
- K: temporal transport between articulated states;
- C: intervention-sensitive causal influence;
- V: orientation / valuation;
- I: local-to-global integration.

The implementation is deliberately independent of the frozen native runtime and
spec-v0.9.5.  It provides an auditable reference mechanism that can later be
benchmarked before any promotion into the native update loop.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from itertools import combinations
from math import isfinite, sqrt
from typing import Callable, Iterable, Mapping


EPS = 1.0e-12
Simplex = frozenset[str]


def _unit(value: float, *, name: str) -> float:
    value = float(value)
    if not isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be finite and lie in [0,1]")
    return value


def _all_nonempty_faces(simplex: Simplex) -> tuple[Simplex, ...]:
    items = tuple(sorted(simplex))
    return tuple(
        frozenset(face)
        for size in range(1, len(items) + 1)
        for face in combinations(items, size)
    )


@dataclass(frozen=True, slots=True)
class DistinctionNeuron:
    """One local distinguishability operator.

    ``operator`` must return a normalized pseudodistance-like response in [0,1].
    Cortex does not require a single neuron to separate all states.
    """

    neuron_id: str
    channel: str
    operator: Callable[[object, object], float]
    weight: float = 1.0
    value_axes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.neuron_id:
            raise ValueError("neuron_id must be non-empty")
        if not self.channel:
            raise ValueError("channel must be non-empty")
        if not isfinite(float(self.weight)) or self.weight <= 0.0:
            raise ValueError("weight must be finite and positive")


@dataclass(frozen=True, slots=True)
class DistinctionProfile:
    by_neuron: tuple[tuple[str, float], ...]
    by_channel: tuple[tuple[str, float], ...]
    aggregate: float

    def neuron(self, neuron_id: str) -> float:
        return dict(self.by_neuron).get(neuron_id, 0.0)

    def channel(self, channel: str) -> float:
        return dict(self.by_channel).get(channel, 0.0)


class DistinctionBank:
    """Population of typed local distinguishability operators (Delta)."""

    def __init__(self, neurons: Iterable[DistinctionNeuron]) -> None:
        rows = tuple(neurons)
        ids = [row.neuron_id for row in rows]
        if len(ids) != len(set(ids)):
            raise ValueError("neuron_id values must be unique")
        if not rows:
            raise ValueError("at least one distinction neuron is required")
        self.neurons = rows
        self._by_id = {row.neuron_id: row for row in rows}

    @property
    def neuron_ids(self) -> tuple[str, ...]:
        return tuple(row.neuron_id for row in self.neurons)

    def compare(self, left: object, right: object) -> DistinctionProfile:
        per_neuron: list[tuple[str, float]] = []
        channel_num: dict[str, float] = defaultdict(float)
        channel_den: dict[str, float] = defaultdict(float)
        total_num = 0.0
        total_den = 0.0

        for neuron in self.neurons:
            response = _unit(
                neuron.operator(left, right),
                name=f"distinction response for {neuron.neuron_id}",
            )
            per_neuron.append((neuron.neuron_id, response))
            channel_num[neuron.channel] += neuron.weight * response
            channel_den[neuron.channel] += neuron.weight
            total_num += neuron.weight * response
            total_den += neuron.weight

        channels = tuple(
            sorted(
                (
                    channel,
                    channel_num[channel] / max(EPS, channel_den[channel]),
                )
                for channel in channel_num
            )
        )
        return DistinctionProfile(
            by_neuron=tuple(sorted(per_neuron)),
            by_channel=channels,
            aggregate=total_num / max(EPS, total_den),
        )


@dataclass(slots=True)
class _TransportRow:
    counts: dict[str, float] = field(default_factory=dict)
    total: float = 0.0


class TemporalTransport:
    """Empirical temporal transport kernel K(y | x).

    The kernel records continuity/succession evidence without requiring states to
    remain indistinguishable over time.  A transported identity may therefore
    change substantially in Delta while remaining temporally coherent.
    """

    def __init__(self, *, decay: float = 1.0, prior: float = 0.0) -> None:
        if not isfinite(decay) or not 0.0 < decay <= 1.0:
            raise ValueError("decay must lie in (0,1]")
        if not isfinite(prior) or prior < 0.0:
            raise ValueError("prior must be finite and non-negative")
        self.decay = float(decay)
        self.prior = float(prior)
        self._rows: dict[str, _TransportRow] = {}

    def observe(self, source: str, target: str, *, weight: float = 1.0) -> None:
        if not source or not target:
            raise ValueError("transport states must be non-empty")
        if not isfinite(weight) or weight <= 0.0:
            raise ValueError("transport weight must be finite and positive")
        row = self._rows.setdefault(source, _TransportRow())
        if self.decay < 1.0 and row.total > 0.0:
            row.counts = {key: value * self.decay for key, value in row.counts.items()}
            row.total *= self.decay
        row.counts[target] = row.counts.get(target, 0.0) + weight
        row.total += weight

    def probability(self, source: str, target: str) -> float:
        row = self._rows.get(source)
        if row is None:
            return 0.0
        support = set(row.counts)
        support.add(target)
        denominator = row.total + self.prior * len(support)
        if denominator <= EPS:
            return 0.0
        return (row.counts.get(target, 0.0) + self.prior) / denominator

    def surprise(self, source: str, target: str) -> float:
        return 1.0 - self.probability(source, target)

    def best_successor(self, source: str) -> tuple[str, float] | None:
        row = self._rows.get(source)
        if row is None or not row.counts:
            return None
        target = min(
            row.counts,
            key=lambda key: (-row.counts[key], key),
        )
        return target, self.probability(source, target)


@dataclass(frozen=True, slots=True)
class CausalObservation:
    source: str
    target: str
    outcome: float
    intervened: bool
    weight: float = 1.0
    neuron_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.source or not self.target:
            raise ValueError("causal source and target must be non-empty")
        _unit(self.outcome, name="causal outcome")
        if not isfinite(self.weight) or self.weight <= 0.0:
            raise ValueError("causal weight must be finite and positive")


@dataclass(slots=True)
class _CausalAccumulator:
    do_sum: float = 0.0
    do_weight: float = 0.0
    control_sum: float = 0.0
    control_weight: float = 0.0


class CausalModel:
    """Intervention-sensitive directional influence model (C)."""

    def __init__(self) -> None:
        self._effects: dict[tuple[str, str], _CausalAccumulator] = {}

    def observe(self, event: CausalObservation) -> None:
        row = self._effects.setdefault((event.source, event.target), _CausalAccumulator())
        if event.intervened:
            row.do_sum += event.weight * float(event.outcome)
            row.do_weight += event.weight
        else:
            row.control_sum += event.weight * float(event.outcome)
            row.control_weight += event.weight

    def signed_effect(self, source: str, target: str) -> float:
        row = self._effects.get((source, target))
        if row is None or row.do_weight <= EPS or row.control_weight <= EPS:
            return 0.0
        p_do = row.do_sum / row.do_weight
        p_control = row.control_sum / row.control_weight
        return max(-1.0, min(1.0, p_do - p_control))

    def influence(self, source: str, target: str) -> float:
        return abs(self.signed_effect(source, target))


class OrientationField:
    """Vector-valued orientation/valuation field (V)."""

    def __init__(self, axis_weights: Mapping[str, float] | None = None) -> None:
        self.axis_weights: dict[str, float] = {}
        for axis, weight in (axis_weights or {}).items():
            self.set_weight(axis, weight)

    def set_weight(self, axis: str, weight: float) -> None:
        if not axis:
            raise ValueError("orientation axis must be non-empty")
        if not isfinite(float(weight)):
            raise ValueError("orientation weight must be finite")
        self.axis_weights[str(axis)] = float(weight)

    def vector(self, features: Mapping[str, float]) -> tuple[tuple[str, float], ...]:
        rows: list[tuple[str, float]] = []
        for axis in sorted(set(self.axis_weights) | set(features)):
            value = float(features.get(axis, 0.0))
            if not isfinite(value):
                raise ValueError("orientation feature values must be finite")
            rows.append((axis, value))
        return tuple(rows)

    def utility(self, features: Mapping[str, float]) -> float:
        values = dict(self.vector(features))
        return sum(
            weight * values.get(axis, 0.0)
            for axis, weight in self.axis_weights.items()
        )

    def salience(self, axes: Iterable[str], features: Mapping[str, float]) -> float:
        axes = tuple(axes)
        if not axes:
            return 0.0
        numerator = 0.0
        denominator = 0.0
        for axis in axes:
            weight = abs(self.axis_weights.get(axis, 0.0))
            value = abs(float(features.get(axis, 0.0)))
            if not isfinite(value):
                raise ValueError("orientation feature values must be finite")
            numerator += weight * value
            denominator += weight
        if denominator <= EPS:
            return 0.0
        return min(1.0, numerator / denominator)


@dataclass(frozen=True, slots=True)
class LocalView:
    """One local articulation contributed by an active neuronal simplex."""

    support: Simplex
    values: Mapping[str, float]
    confidence: float = 1.0

    def __post_init__(self) -> None:
        if not self.support:
            raise ValueError("local view support must be non-empty")
        _unit(self.confidence, name="local view confidence")
        for coordinate, value in self.values.items():
            if not coordinate or not isfinite(float(value)):
                raise ValueError("local view coordinates must be non-empty and finite")


@dataclass(frozen=True, slots=True)
class IntegrationRead:
    articulation: tuple[tuple[str, float], ...]
    residual: float
    conflicts: tuple[str, ...]
    used_views: int


class Integrator:
    """Local-to-global integration/composition operator (I).

    Compatible local views are combined by confidence-weighted least squares.
    Contradictory coordinates are reported explicitly rather than silently
    erased by averaging.
    """

    def __init__(self, *, conflict_tolerance: float = 0.25) -> None:
        self.conflict_tolerance = _unit(
            conflict_tolerance,
            name="conflict_tolerance",
        )

    def integrate(
        self,
        views: Iterable[LocalView],
        *,
        active_complex: Iterable[Simplex] | None = None,
    ) -> IntegrationRead:
        active = set(active_complex) if active_complex is not None else None
        selected = tuple(
            view
            for view in views
            if active is None or view.support in active
        )
        if not selected:
            return IntegrationRead((), 0.0, (), 0)

        sums: dict[str, float] = defaultdict(float)
        weights: dict[str, float] = defaultdict(float)
        samples: dict[str, list[tuple[float, float]]] = defaultdict(list)
        for view in selected:
            weight = float(view.confidence)
            if weight <= EPS:
                continue
            for coordinate, raw_value in view.values.items():
                value = float(raw_value)
                sums[coordinate] += weight * value
                weights[coordinate] += weight
                samples[coordinate].append((value, weight))

        means = {
            coordinate: sums[coordinate] / max(EPS, weights[coordinate])
            for coordinate in sums
        }
        squared = 0.0
        total_weight = 0.0
        conflicts: list[str] = []
        for coordinate, rows in samples.items():
            mean = means[coordinate]
            values = [value for value, _ in rows]
            if values and max(values) - min(values) > self.conflict_tolerance:
                conflicts.append(coordinate)
            for value, weight in rows:
                squared += weight * (value - mean) ** 2
                total_weight += weight

        residual = sqrt(squared / max(EPS, total_weight)) if total_weight else 0.0
        return IntegrationRead(
            articulation=tuple(sorted(means.items())),
            residual=residual,
            conflicts=tuple(sorted(conflicts)),
            used_views=len(selected),
        )


@dataclass(frozen=True, slots=True)
class DynamicComplexPolicy:
    activation_threshold: float = 0.65
    deactivation_threshold: float = 0.45
    novelty_weight: float = 0.35
    temporal_weight: float = 0.15
    causal_weight: float = 0.20
    value_weight: float = 0.20
    base_weight: float = 0.10
    synergy_gain: float = 0.25

    def __post_init__(self) -> None:
        on = _unit(self.activation_threshold, name="activation_threshold")
        off = _unit(self.deactivation_threshold, name="deactivation_threshold")
        if off >= on:
            raise ValueError("deactivation_threshold must be lower than activation_threshold")
        for name in (
            "novelty_weight",
            "temporal_weight",
            "causal_weight",
            "value_weight",
            "base_weight",
            "synergy_gain",
        ):
            value = float(getattr(self, name))
            if not isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class ComplexRead:
    active: tuple[Simplex, ...]
    scores: tuple[tuple[tuple[str, ...], float], ...]
    activated: tuple[Simplex, ...]
    deactivated: tuple[Simplex, ...]


class DynamicDistinguishabilityComplex:
    """Context-sensitive simplicial topology over a persistent neuron set."""

    def __init__(
        self,
        neuron_ids: Iterable[str],
        *,
        potential_simplices: Iterable[Iterable[str]] = (),
        policy: DynamicComplexPolicy | None = None,
    ) -> None:
        vertices = tuple(sorted(set(neuron_ids)))
        if not vertices:
            raise ValueError("the dynamic complex requires at least one neuron")
        self.policy = policy or DynamicComplexPolicy()
        self._vertices = frozenset(vertices)
        potential: set[Simplex] = {frozenset((vertex,)) for vertex in vertices}
        for raw in potential_simplices:
            simplex = frozenset(str(item) for item in raw)
            if not simplex:
                continue
            if not simplex <= self._vertices:
                raise ValueError("potential simplex contains an unknown neuron")
            potential.update(_all_nonempty_faces(simplex))
        self.potential = frozenset(potential)
        self._active: set[Simplex] = set()

    @property
    def vertices(self) -> tuple[str, ...]:
        return tuple(sorted(self._vertices))

    @property
    def active(self) -> tuple[Simplex, ...]:
        return tuple(sorted(self._active, key=lambda item: (len(item), tuple(sorted(item)))))

    def _raw_score(
        self,
        simplex: Simplex,
        relevance: Mapping[str, float],
        synergy: Mapping[Simplex, float],
    ) -> float:
        base = sum(_unit(relevance.get(v, 0.0), name=f"relevance[{v}]") for v in simplex)
        base /= len(simplex)
        bonus = float(synergy.get(simplex, 0.0))
        if not isfinite(bonus):
            raise ValueError("synergy values must be finite")
        return max(0.0, min(1.0, base + self.policy.synergy_gain * bonus))

    def update(
        self,
        relevance: Mapping[str, float],
        *,
        synergy: Mapping[Simplex, float] | None = None,
    ) -> ComplexRead:
        synergy = synergy or {}
        raw = {
            simplex: self._raw_score(simplex, relevance, synergy)
            for simplex in self.potential
        }
        scores = {
            simplex: min(raw[face] for face in _all_nonempty_faces(simplex))
            for simplex in self.potential
        }
        previous = set(self._active)
        nxt: set[Simplex] = set()
        for simplex in sorted(self.potential, key=lambda item: (len(item), tuple(sorted(item)))):
            score = scores[simplex]
            if score >= self.policy.activation_threshold:
                nxt.add(simplex)
            elif score > self.policy.deactivation_threshold and simplex in previous:
                nxt.add(simplex)

        for simplex in tuple(nxt):
            missing = set(_all_nonempty_faces(simplex)) - nxt
            if missing:
                raise AssertionError("dynamic complex update violated simplicial closure")

        self._active = nxt
        activated = nxt - previous
        deactivated = previous - nxt
        return ComplexRead(
            active=self.active,
            scores=tuple(
                (tuple(sorted(simplex)), scores[simplex])
                for simplex in sorted(scores, key=lambda item: (len(item), tuple(sorted(item))))
            ),
            activated=tuple(sorted(activated, key=lambda item: (len(item), tuple(sorted(item))))),
            deactivated=tuple(sorted(deactivated, key=lambda item: (len(item), tuple(sorted(item))))),
        )


@dataclass(frozen=True, slots=True)
class ArticulationInput:
    state_id: str
    observation: object
    orientation: Mapping[str, float] = field(default_factory=dict)
    local_views: tuple[LocalView, ...] = ()
    causal_updates: tuple[CausalObservation, ...] = ()
    simplex_synergy: Mapping[Simplex, float] = field(default_factory=dict)
    transport_weight: float = 1.0

    def __post_init__(self) -> None:
        if not self.state_id:
            raise ValueError("state_id must be non-empty")
        if not isfinite(self.transport_weight) or self.transport_weight <= 0.0:
            raise ValueError("transport_weight must be finite and positive")


@dataclass(frozen=True, slots=True)
class FiveOperatorFrame:
    state_id: str
    observation: object
    distinction: DistinctionProfile
    temporal_surprise: float
    causal_relevance: tuple[tuple[str, float], ...]
    orientation_vector: tuple[tuple[str, float], ...]
    orientation_utility: float
    neuron_relevance: tuple[tuple[str, float], ...]
    complex: ComplexRead
    integration: IntegrationRead


class FiveOperatorArticulationLoop:
    """Coordinator over Delta, K, C, V and I on one evolving topology."""

    def __init__(
        self,
        distinction: DistinctionBank,
        transport: TemporalTransport,
        causal: CausalModel,
        orientation: OrientationField,
        integrator: Integrator,
        complex_: DynamicDistinguishabilityComplex,
    ) -> None:
        if set(distinction.neuron_ids) != set(complex_.vertices):
            raise ValueError("distinction bank and dynamic complex must share the same neurons")
        self.distinction = distinction
        self.transport = transport
        self.causal = causal
        self.orientation = orientation
        self.integrator = integrator
        self.complex = complex_
        self.policy = complex_.policy
        self._previous: FiveOperatorFrame | None = None

    @property
    def previous(self) -> FiveOperatorFrame | None:
        return self._previous

    def step(self, item: ArticulationInput) -> FiveOperatorFrame:
        if self._previous is None:
            distinction = DistinctionProfile(
                by_neuron=tuple((neuron_id, 0.0) for neuron_id in self.distinction.neuron_ids),
                by_channel=tuple(
                    (channel, 0.0)
                    for channel in sorted({neuron.channel for neuron in self.distinction.neurons})
                ),
                aggregate=0.0,
            )
            temporal_surprise = 0.0
        else:
            distinction = self.distinction.compare(
                self._previous.observation,
                item.observation,
            )
            predicted = self.transport.probability(self._previous.state_id, item.state_id)
            temporal_surprise = 1.0 - predicted
            self.transport.observe(
                self._previous.state_id,
                item.state_id,
                weight=item.transport_weight,
            )

        causal_relevance = {neuron_id: 0.0 for neuron_id in self.distinction.neuron_ids}
        for update in item.causal_updates:
            self.causal.observe(update)
            effect = self.causal.influence(update.source, update.target)
            for neuron_id in update.neuron_ids:
                if neuron_id not in causal_relevance:
                    raise ValueError(f"causal update references unknown neuron {neuron_id!r}")
                causal_relevance[neuron_id] = max(causal_relevance[neuron_id], effect)

        neuron_relevance: dict[str, float] = {}
        for neuron in self.distinction.neurons:
            novelty = distinction.neuron(neuron.neuron_id)
            value = self.orientation.salience(neuron.value_axes, item.orientation)
            causal = causal_relevance[neuron.neuron_id]
            numerator = (
                self.policy.base_weight
                + self.policy.novelty_weight * novelty
                + self.policy.temporal_weight * temporal_surprise
                + self.policy.causal_weight * causal
                + self.policy.value_weight * value
            )
            denominator = (
                self.policy.base_weight
                + self.policy.novelty_weight
                + self.policy.temporal_weight
                + self.policy.causal_weight
                + self.policy.value_weight
            )
            neuron_relevance[neuron.neuron_id] = min(1.0, numerator / max(EPS, denominator))

        complex_read = self.complex.update(
            neuron_relevance,
            synergy=item.simplex_synergy,
        )
        integration = self.integrator.integrate(
            item.local_views,
            active_complex=complex_read.active,
        )
        frame = FiveOperatorFrame(
            state_id=item.state_id,
            observation=item.observation,
            distinction=distinction,
            temporal_surprise=temporal_surprise,
            causal_relevance=tuple(sorted(causal_relevance.items())),
            orientation_vector=self.orientation.vector(item.orientation),
            orientation_utility=self.orientation.utility(item.orientation),
            neuron_relevance=tuple(sorted(neuron_relevance.items())),
            complex=complex_read,
            integration=integration,
        )
        self._previous = frame
        return frame
