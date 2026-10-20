"""Research-only local-to-global integration over a learned simplicial complex.

The v0.9 distinction complex can tell a triangle boundary from a genuinely
irreducible filled 2-simplex.  This module makes that topology computationally
meaningful for the integration operator I.

For a selected probe population and one context, every active simplex supplies
an empirical marginal distribution (a local section).  The integrator builds
the maximum-entropy global distribution consistent with those local sections
using iterative proportional fitting (IPF):

- vertices constrain one-probe marginals;
- edges constrain pairwise marginals;
- filled triangles constrain synchronized three-probe marginals.

Consequently a K3 boundary does not silently imply a three-way joint model.  If
all pairwise marginals are identical across two contexts but an active 2-simplex
carries different joint evidence, the integrated articulations can differ only
when that coface is present.

This is a bounded research reference for finite categorical responses, not yet
a general sheaf solver.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import isfinite, log
from typing import Hashable, Iterable, Mapping

from .higher_order_irreducibility import HystereticDistinctionComplex, JointDistinctionMemory
from .probe_economy import Probe


Distribution = dict[tuple[Hashable, ...], float]


def _tv(left: Mapping[tuple, float], right: Mapping[tuple, float]) -> float:
    keys = set(left) | set(right)
    return 0.5 * sum(abs(left.get(key, 0.0) - right.get(key, 0.0)) for key in keys)


@dataclass(frozen=True, slots=True)
class SimplicialIntegrationPolicy:
    """Numerical/resource controls for finite maximum-entropy composition."""

    max_states: int = 4096
    max_iterations: int = 256
    tolerance: float = 1.0e-9
    epsilon: float = 1.0e-12

    def __post_init__(self) -> None:
        if type(self.max_states) is not int or self.max_states < 1:
            raise ValueError("max_states must be a positive integer")
        if type(self.max_iterations) is not int or self.max_iterations < 1:
            raise ValueError("max_iterations must be a positive integer")
        for name, value in (("tolerance", self.tolerance), ("epsilon", self.epsilon)):
            if not isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True, slots=True)
class SimplicialIntegrationRead:
    probes: tuple[Probe, ...]
    distribution: tuple[tuple[tuple[Hashable, ...], float], ...]
    residual: float
    entropy: float
    used_constraints: tuple[tuple[int, ...], ...]
    max_dimension: int
    iterations: int

    def as_distribution(self) -> Distribution:
        return dict(self.distribution)


class SimplicialIntegrator:
    """Compose active local distinction sections into one global articulation."""

    def __init__(
        self,
        complex_: HystereticDistinctionComplex,
        memory: JointDistinctionMemory,
        *,
        policy: SimplicialIntegrationPolicy | None = None,
    ) -> None:
        if complex_.memory is not memory:
            raise ValueError("simplicial complex and integrator must share one joint memory")
        self.complex = complex_
        self.memory = memory
        self.economy = memory.economy
        self.policy = policy or SimplicialIntegrationPolicy()

    def _ordered_probes(self, probes: Iterable[Iterable] | None) -> tuple[Probe, ...]:
        if probes is None:
            rows = tuple(self.complex.graph.vertices)
        else:
            rows = tuple(self.economy._check_probe(probe) for probe in probes)
        if not rows:
            raise ValueError("integration requires at least one active probe")
        if len(rows) != len(set(rows)):
            raise ValueError("integration probes must be unique")
        return tuple(sorted(rows, key=self.economy._index.__getitem__))

    def _alphabet(self, probe: Probe) -> tuple[Hashable, ...]:
        values: set[Hashable] = set()
        for context in self.economy.contexts:
            for key in self.memory.distribution(context, (probe,)):
                if len(key) != 1:
                    raise AssertionError("singleton probe distribution has non-singleton key")
                values.add(key[0])
        if not values:
            raise ValueError("probe has no sufficiently supported joint observations")
        return tuple(sorted(values, key=repr))

    def _supports(self, probes: tuple[Probe, ...], max_dimension: int) -> tuple[tuple[Probe, ...], ...]:
        if type(max_dimension) is not int or not 0 <= max_dimension <= 2:
            raise ValueError("max_dimension must be one of 0, 1, 2")
        selected = set(probes)
        active_vertices = set(self.complex.graph.vertices)
        supports: set[tuple[Probe, ...]] = {
            (probe,) for probe in probes if probe in active_vertices
        }
        if max_dimension >= 1:
            for edge in self.complex.graph.edges:
                if set(edge) <= selected:
                    supports.add(tuple(sorted(edge, key=self.economy._index.__getitem__)))
        if max_dimension >= 2:
            for triangle in self.complex.triangles:
                if set(triangle) <= selected:
                    supports.add(tuple(sorted(triangle, key=self.economy._index.__getitem__)))
        return tuple(
            sorted(
                supports,
                key=lambda support: (
                    len(support),
                    tuple(self.economy._index[probe] for probe in support),
                ),
            )
        )

    @staticmethod
    def _marginal(
        distribution: Mapping[tuple[Hashable, ...], float],
        positions: tuple[int, ...],
    ) -> Distribution:
        result: Distribution = {}
        for state, probability in distribution.items():
            key = tuple(state[position] for position in positions)
            result[key] = result.get(key, 0.0) + probability
        return result

    def _constraint_residual(
        self,
        distribution: Mapping[tuple[Hashable, ...], float],
        constraints: tuple[tuple[tuple[int, ...], Distribution], ...],
    ) -> float:
        return max(
            (_tv(self._marginal(distribution, positions), target) for positions, target in constraints),
            default=0.0,
        )

    def integrate(
        self,
        context: Hashable,
        *,
        probes: Iterable[Iterable] | None = None,
        max_dimension: int = 2,
    ) -> SimplicialIntegrationRead:
        probes = self._ordered_probes(probes)
        if context not in self.economy.contexts:
            raise ValueError("context lies outside the economy context set")

        alphabets = tuple(self._alphabet(probe) for probe in probes)
        state_count = 1
        for alphabet in alphabets:
            state_count *= len(alphabet)
        if state_count > self.policy.max_states:
            raise ValueError("global response space exceeds simplicial integration resource cap")

        states = tuple(product(*alphabets))
        probability = 1.0 / len(states)
        distribution: Distribution = {state: probability for state in states}
        local_index = {probe: index for index, probe in enumerate(probes)}

        supports = self._supports(probes, max_dimension)
        constraints: list[tuple[tuple[int, ...], Distribution]] = []
        for support in supports:
            target = self.memory.distribution(context, support)
            if not target:
                raise ValueError("active simplex lacks sufficient synchronized support")
            positions = tuple(local_index[probe] for probe in support)
            constraints.append((positions, target))
        frozen_constraints = tuple(constraints)

        residual = self._constraint_residual(distribution, frozen_constraints)
        iterations = 0
        for iteration in range(1, self.policy.max_iterations + 1):
            iterations = iteration
            for positions, target in frozen_constraints:
                current = self._marginal(distribution, positions)
                updated: Distribution = {}
                for state, old_probability in distribution.items():
                    key = tuple(state[position] for position in positions)
                    desired = target.get(key, 0.0)
                    observed = current.get(key, 0.0)
                    if desired <= self.policy.epsilon:
                        updated[state] = 0.0
                    elif observed <= self.policy.epsilon:
                        raise ValueError("local sections are mutually inconsistent")
                    else:
                        updated[state] = old_probability * desired / observed
                normalizer = sum(updated.values())
                if normalizer <= self.policy.epsilon:
                    raise ValueError("local sections admit no global articulation")
                distribution = {state: value / normalizer for state, value in updated.items()}

            residual = self._constraint_residual(distribution, frozen_constraints)
            if residual <= self.policy.tolerance:
                break

        entropy = -sum(
            probability * log(probability)
            for probability in distribution.values()
            if probability > self.policy.epsilon
        )
        used_constraints = tuple(
            tuple(local_index[probe] for probe in support)
            for support in supports
        )
        return SimplicialIntegrationRead(
            probes=probes,
            distribution=tuple(sorted(distribution.items(), key=lambda row: repr(row[0]))),
            residual=residual,
            entropy=entropy,
            used_constraints=used_constraints,
            max_dimension=max((len(support) - 1 for support in supports), default=0),
            iterations=iterations,
        )

    def distance(
        self,
        left: Hashable,
        right: Hashable,
        *,
        probes: Iterable[Iterable] | None = None,
        max_dimension: int = 2,
    ) -> float:
        left_read = self.integrate(left, probes=probes, max_dimension=max_dimension)
        right_read = self.integrate(right, probes=probes, max_dimension=max_dimension)
        return _tv(left_read.as_distribution(), right_read.as_distribution())

    def composition_gain(
        self,
        left: Hashable,
        right: Hashable,
        *,
        probes: Iterable[Iterable] | None = None,
    ) -> float:
        boundary = self.distance(left, right, probes=probes, max_dimension=1)
        filled = self.distance(left, right, probes=probes, max_dimension=2)
        return max(0.0, filled - boundary)
