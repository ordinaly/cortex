"""Exact finite possibility dynamics; opt-in oracle, not a learned world model.

Rows are J(successor, observation | predecessor, action). Probabilities use
Fraction; pass strings or Fractions for intended decimal/rational values.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as F
from itertools import combinations
from types import MappingProxyType
from typing import Mapping


def probability(value) -> F:
    value = F(value)
    if not 0 <= value <= 1:
        raise ValueError("probability outside [0, 1]")
    return value


def distribution(values, keys):
    if set(values) != set(keys):
        raise ValueError("distribution must cover exactly the declared candidates")
    result = {key: probability(values[key]) for key in keys}
    if sum(result.values()) != 1:
        raise ValueError("distribution must sum exactly to one")
    return MappingProxyType(result)


@dataclass(frozen=True)
class Candidate:
    """Mutually exclusive situated alternative referencing caller-owned memory."""
    name: str
    memory_refs: tuple[str, ...] = ()
    assumptions: tuple[str, ...] = ()

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("candidate needs a nonempty string name")
        for field in ("memory_refs", "assumptions"):
            values = tuple(getattr(self, field))
            if any(not isinstance(value, str) for value in values):
                raise ValueError("references and assumptions must be strings")
            object.__setattr__(self, field, values)


@dataclass(frozen=True)
class Probe:
    """Yes/no event on traces under a fixed, open-loop action sequence."""
    actions: tuple[str, ...]
    accepted: frozenset[tuple[str, ...]]

    def __post_init__(self):
        object.__setattr__(self, "actions", tuple(self.actions))
        object.__setattr__(self, "accepted", frozenset(tuple(x) for x in self.accepted))
        if any(len(trace) != len(self.actions) for trace in self.accepted):
            raise ValueError("probe traces must match its horizon")


@dataclass(frozen=True)
class FiniteModel:
    version: str
    candidates: tuple[Candidate, ...]
    actions: tuple[str, ...]
    observations: tuple[str, ...]
    rows: Mapping[tuple[str, str], Mapping[tuple[str, str], F]]

    def __post_init__(self):
        for field in ("candidates", "actions", "observations"):
            object.__setattr__(self, field, tuple(getattr(self, field)))
        for values in (self.names, self.actions, self.observations):
            if not values or len(set(values)) != len(values):
                raise ValueError("alphabets must be nonempty and unique")
        expected = {(v, a) for v in self.names for a in self.actions}
        if set(self.rows) != expected:
            raise ValueError("joint kernel needs every candidate/action row")
        copied = {}
        for key, row in self.rows.items():
            if any(v not in self.names or o not in self.observations for v, o in row):
                raise ValueError("unknown successor or observation")
            converted = {edge: probability(p) for edge, p in row.items()}
            if sum(converted.values()) != 1:
                raise ValueError("joint kernel row must sum exactly to one")
            copied[key] = MappingProxyType(converted)
        object.__setattr__(self, "rows", MappingProxyType(copied))

    @property
    def names(self):
        return tuple(v.name for v in self.candidates)

    def trace_distribution(self, belief, actions):
        """Exact dynamic programming over observable traces and hidden states."""
        belief = distribution(belief, self.names)
        mass = {(v, ()): p for v, p in belief.items() if p}
        for action in actions:
            if action not in self.actions:
                raise ValueError("unknown action")
            following = {}
            for (v, trace), p in mass.items():
                for (u, o), weight in self.rows[v, action].items():
                    key = (u, trace + (o,))
                    following[key] = following.get(key, F(0)) + p * weight
            mass = following
        result = {}
        for (_, trace), p in mass.items():
            result[trace] = result.get(trace, F(0)) + p
        return MappingProxyType(result)

    def probe_profile(self, candidate, probes):
        if candidate not in self.names:
            raise ValueError("unknown candidate")
        belief = {v: F(v == candidate) for v in self.names}
        answers = []
        for probe in probes:
            if any(o not in self.observations for trace in probe.accepted for o in trace):
                raise ValueError("unknown probe observation")
            traces = self.trace_distribution(belief, probe.actions)
            answers.append(sum((traces.get(t, F(0)) for t in probe.accepted), F(0)))
        return tuple(answers)

    def geometry(self, probes, resolution=0):
        """Flag complex stored by its edges, avoiding exponential enumeration."""
        probes = tuple(probes)
        resolution = F(resolution)
        if not probes or resolution < 0:
            raise ValueError("nonempty probes and nonnegative resolution required")
        profiles = {v: self.probe_profile(v, probes) for v in self.names}
        distances = {(v, w): max(abs(x-y) for x, y in zip(profiles[v], profiles[w]))
                     for v, w in combinations(self.names, 2)}
        edges = frozenset(frozenset(pair) for pair, d in distances.items() if d <= resolution)
        return Geometry(self.names, MappingProxyType(profiles), MappingProxyType(distances), edges)

    def is_exact_partition(self, cells):
        """Joint observation/successor-cell closure, stronger than probe equality.

        No merging is performed. A true result certifies this finite kernel only.
        """
        cells = tuple(tuple(cell) for cell in cells)
        flat = [v for cell in cells for v in cell]
        if any(not cell for cell in cells) or len(flat) != len(set(flat)) or set(flat) != set(self.names):
            raise ValueError("cells must partition candidates")
        def signature(v):
            return tuple(sum((self.rows[v, a].get((u, o), F(0)) for u in target), F(0))
                         for a in self.actions for o in self.observations for target in cells)
        return all(all(signature(v) == signature(cell[0]) for v in cell) for cell in cells)


@dataclass(frozen=True)
class Geometry:
    vertices: tuple[str, ...]
    profiles: Mapping
    distances: Mapping
    edges: frozenset[frozenset[str]]

    def contains(self, simplex):
        simplex = frozenset(simplex)
        return simplex <= set(self.vertices) and all(
            frozenset(pair) in self.edges for pair in combinations(simplex, 2))


@dataclass(frozen=True)
class Prediction:
    model_version: str
    action: str
    prior: Mapping
    successor: Mapping
    outcomes: Mapping
    candidate_outcomes: Mapping
    joint: Mapping


@dataclass(frozen=True)
class Assimilation:
    status: str
    observation: str
    evidence: F
    prediction: Prediction
    posterior: Mapping | None


class PossibilityFilter:
    """One pending forecast; failure blocks further steps pending explicit revision.

    Failed updates retain the last valid belief for diagnosis, not as a posterior
    at the new time. Construct a new filter/version after external revision.
    Records are immutable; history retention is the caller's responsibility.
    """
    def __init__(self, model: FiniteModel, belief):
        self._model = model
        self._belief = distribution(belief, model.names)
        self._pending = None
        self._failure = None

    @property
    def belief(self):
        return self._belief

    @property
    def failure(self):
        return self._failure

    def predict(self, action):
        if self._failure is not None:
            raise RuntimeError("model failure requires explicit external revision")
        if self._pending is not None:
            raise RuntimeError("assimilate pending prediction first")
        if action not in self._model.actions:
            raise ValueError("unknown action")
        joint = {}
        conditional = {}
        for v in self._model.names:
            outcomes = {o: F(0) for o in self._model.observations}
            for (u, o), weight in self._model.rows[v, action].items():
                joint[u, o] = joint.get((u, o), F(0)) + self._belief[v] * weight
                outcomes[o] += weight
            conditional[v] = MappingProxyType(outcomes)
        successor = {u: sum((p for (v, _), p in joint.items() if v == u), F(0))
                     for u in self._model.names}
        outcomes = {o: sum((p for (_, obs), p in joint.items() if obs == o), F(0))
                    for o in self._model.observations}
        self._pending = Prediction(self._model.version, action, self._belief,
                                   MappingProxyType(successor), MappingProxyType(outcomes),
                                   MappingProxyType(conditional), MappingProxyType(joint))
        return self._pending

    def assimilate(self, action, observation):
        prediction = self._pending
        if prediction is None or prediction.action != action:
            raise ValueError("matching pre-observation prediction required")
        evidence = prediction.outcomes.get(observation, F(0))
        posterior = None
        if evidence:
            posterior = distribution({v: prediction.joint.get((v, observation), F(0)) / evidence
                                      for v in self._model.names}, self._model.names)
            self._belief = posterior
        result = Assimilation("updated" if evidence else "model_failure", observation,
                              evidence, prediction, posterior)
        self._pending = None
        if not evidence:
            self._failure = result
        return result
