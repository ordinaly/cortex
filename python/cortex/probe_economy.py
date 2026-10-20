"""Research-only economy over learned distinguishability probes.

This module operates only on learner-visible empirical trace responses. Hidden
world state and the finite Sigma-Lambda oracle remain evaluator-only.

For a fixed context set, each probe induces a set of *merge-veto edges*: context
pairs that the current confidence rule does not permit us to fold together. The
full distinction population induces the union of those edges. Selecting the
smallest probe population that preserves the same merge-veto relation is thus a
finite minimum set-cover problem.

The exact solver here is intentionally bounded and evaluator-like with respect
to *empirical* probe redundancy only. It does not use ground-truth dynamics.
The hysteretic controller prefers already-active probes among equally small
covers, so representations do not churn when several equivalent probe bases
exist.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import isfinite
from typing import Hashable, Iterable

from .learned_distinguishability import EmpiricalTraceDistinguishability


Probe = tuple
Edge = tuple[int, int]


@dataclass(frozen=True, slots=True)
class ProbeEconomyPolicy:
    """Finite probe-economy controls.

    ``probe_cost`` is a normalized per-probe complexity penalty used only in the
    reported marginal utility. The hard sufficiency constraint always wins: a
    required merge-veto edge is never dropped merely to save cost.

    ``activation_threshold`` is the minimum fractional population-size saving
    required before the controller performs a global basis swap while its
    current population is already sufficient. Missing coverage always forces
    activation regardless of this threshold.

    ``retirement_threshold`` bounds the fraction of currently required edges a
    probe may uniquely support when considering a local retirement. The default
    zero therefore retires only genuinely redundant active probes.
    """

    max_exact_probes: int = 18
    probe_cost: float = 0.01
    activation_threshold: float = 0.05
    retirement_threshold: float = 0.0

    def __post_init__(self) -> None:
        if type(self.max_exact_probes) is not int or self.max_exact_probes < 1:
            raise ValueError("max_exact_probes must be a positive integer")
        for name, value in (
            ("probe_cost", self.probe_cost),
            ("activation_threshold", self.activation_threshold),
            ("retirement_threshold", self.retirement_threshold),
        ):
            if not isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be finite and lie in [0,1]")
        if self.retirement_threshold > self.activation_threshold:
            raise ValueError("retirement_threshold must not exceed activation_threshold")


class ProbeEconomy:
    """Measure and minimize empirical distinguishability-probe redundancy."""

    def __init__(
        self,
        learner: EmpiricalTraceDistinguishability,
        contexts: Iterable[Hashable] | None = None,
        *,
        policy: ProbeEconomyPolicy | None = None,
    ) -> None:
        self.learner = learner
        self.policy = policy or ProbeEconomyPolicy()
        self.contexts = tuple(learner.contexts if contexts is None else contexts)
        if len(self.contexts) != len(set(self.contexts)):
            raise ValueError("contexts must be unique")
        if any(context not in learner.contexts for context in self.contexts):
            raise ValueError("all economy contexts must be present in the learner")
        self.probes = tuple(learner.probes)
        if len(self.probes) > self.policy.max_exact_probes:
            raise ValueError("probe family exceeds exact economy resource cap")
        self._index = {probe: i for i, probe in enumerate(self.probes)}

    @property
    def pairs(self) -> tuple[Edge, ...]:
        return tuple(combinations(range(len(self.contexts)), 2))

    def _check_probe(self, probe: Iterable) -> Probe:
        probe = tuple(probe)
        if probe not in self._index:
            raise ValueError("probe lies outside the declared trace domain")
        return probe

    def fully_supported(self, probe: Iterable) -> bool:
        probe = self._check_probe(probe)
        minimum = self.learner.policy.min_samples
        return all(self.learner.support(context, probe) >= minimum for context in self.contexts)

    def merge_upper_bound(self, edge: Edge, probe: Iterable) -> float:
        """Empirical merge-risk upper bound used by the existing learner.

        An unsupported pair returns 1.0, preserving the learner's rule that lack
        of evidence is not evidence of equivalence.
        """
        probe = self._check_probe(probe)
        left_i, right_i = edge
        if not (0 <= left_i < right_i < len(self.contexts)):
            raise ValueError("edge must index an ordered context pair")
        left, right = self.contexts[left_i], self.contexts[right_i]
        nl = self.learner.support(left, probe)
        nr = self.learner.support(right, probe)
        if min(nl, nr) < self.learner.policy.min_samples:
            return 1.0
        return min(
            1.0,
            self.learner.probe_distance(left, right, probe)
            + self.learner._radius(nl)
            + self.learner._radius(nr),
        )

    def veto_edges(self, probe: Iterable) -> frozenset[Edge]:
        probe = self._check_probe(probe)
        epsilon = self.learner.policy.epsilon
        return frozenset(edge for edge in self.pairs if self.merge_upper_bound(edge, probe) > epsilon)

    @property
    def required_edges(self) -> frozenset[Edge]:
        edges: set[Edge] = set()
        for probe in self.probes:
            edges.update(self.veto_edges(probe))
        return frozenset(edges)

    def covered_edges(self, probes: Iterable[Iterable]) -> frozenset[Edge]:
        covered: set[Edge] = set()
        for probe in probes:
            covered.update(self.veto_edges(probe))
        return frozenset(covered)

    def coverage(self, probes: Iterable[Iterable]) -> float:
        required = self.required_edges
        if not required:
            return 1.0
        return len(self.covered_edges(probes) & required) / len(required)

    def addition_gain(self, probe: Iterable, active: Iterable[Iterable]) -> float:
        probe = self._check_probe(probe)
        required = self.required_edges
        if not required:
            return 0.0
        active_set = tuple(tuple(p) for p in active if tuple(p) != probe)
        before = self.covered_edges(active_set)
        gained = (self.veto_edges(probe) & required) - before
        return len(gained) / len(required)

    def removal_loss(self, probe: Iterable, active: Iterable[Iterable]) -> float:
        probe = self._check_probe(probe)
        required = self.required_edges
        if not required:
            return 0.0
        active_set = tuple(tuple(p) for p in active)
        if probe not in active_set:
            return 0.0
        before = self.covered_edges(active_set)
        after = self.covered_edges(p for p in active_set if p != probe)
        lost = (before & required) - after
        return len(lost) / len(required)

    def marginal_utility(self, probe: Iterable, active: Iterable[Iterable]) -> float:
        """Normalized distinction gain/loss minus a constant representation cost."""
        probe = self._check_probe(probe)
        active_set = tuple(tuple(p) for p in active)
        gain = (
            self.removal_loss(probe, active_set)
            if probe in active_set
            else self.addition_gain(probe, active_set)
        )
        return gain - self.policy.probe_cost

    def compatible(self, left_i: int, right_i: int, probes: Iterable[Iterable]) -> bool:
        selected = tuple(self._check_probe(probe) for probe in probes)
        if left_i == right_i:
            return True
        edge = (min(left_i, right_i), max(left_i, right_i))
        return all(edge not in self.veto_edges(probe) for probe in selected)

    def propose_partition(self, probes: Iterable[Iterable]) -> tuple[tuple[Hashable, ...], ...]:
        """Complete-link quotient induced by only the selected probe population."""
        selected = tuple(self._check_probe(probe) for probe in probes)
        cells: list[list[int]] = []
        for index in range(len(self.contexts)):
            for cell in cells:
                if all(self.compatible(index, other, selected) for other in cell):
                    cell.append(index)
                    break
            else:
                cells.append([index])
        return tuple(tuple(self.contexts[index] for index in cell) for cell in cells)

    def exact_minimal_cover(self, *, prefer: Iterable[Iterable] = ()) -> tuple[Probe, ...]:
        """Minimum-cardinality probe cover of the current full merge-veto relation.

        Ties maximize overlap with ``prefer`` and then use declared probe order.
        This state preference is the first hysteresis mechanism: equally good
        bases do not churn merely because another minimum cover also exists.
        """
        required = self.required_edges
        if not required:
            return ()
        preferred = {self._check_probe(probe) for probe in prefer}
        rows = tuple(self.probes)
        for size in range(1, len(rows) + 1):
            candidates = []
            for indices in combinations(range(len(rows)), size):
                probes = tuple(rows[index] for index in indices)
                if self.covered_edges(probes) >= required:
                    overlap = sum(probe in preferred for probe in probes)
                    candidates.append((-overlap, indices, probes))
            if candidates:
                candidates.sort(key=lambda row: (row[0], row[1]))
                return candidates[0][2]
        raise RuntimeError("full probe family failed to cover its own required edge set")

    @property
    def snapshot(self) -> dict:
        minimal = self.exact_minimal_cover()
        return {
            "contexts": len(self.contexts),
            "probes": len(self.probes),
            "required_edges": len(self.required_edges),
            "minimum_probes": len(minimal),
            "minimum_probe_indices": tuple(self._index[probe] for probe in minimal),
        }


class HystereticProbePopulation:
    """Stateful active/latent probe population with hard sufficiency protection."""

    def __init__(
        self,
        economy: ProbeEconomy,
        *,
        active: Iterable[Iterable] | None = None,
    ) -> None:
        self.economy = economy
        if active is None:
            self.active = set(economy.probes)
        else:
            self.active = {economy._check_probe(probe) for probe in active}
        self.activations = 0
        self.retirements = 0
        self.steps = 0

    @property
    def latent(self) -> tuple[Probe, ...]:
        return tuple(probe for probe in self.economy.probes if probe not in self.active)

    def step(self) -> tuple[Probe, ...]:
        """Move toward the exact empirical minimum while preferring persistence.

        If the current population no longer covers the full learned distinction
        relation, missing target probes are activated immediately. If it is
        already sufficient, a global basis change is attempted only when it
        reduces population size by at least ``activation_threshold``.

        Retirement is conservative: a probe leaves only when the resulting
        population still covers every currently required merge-veto edge.
        """
        self.steps += 1
        policy = self.economy.policy
        current = tuple(probe for probe in self.economy.probes if probe in self.active)
        target = self.economy.exact_minimal_cover(prefer=current)
        current_coverage = self.economy.coverage(current)
        saving = max(0.0, (len(current) - len(target)) / max(1, len(current)))

        allow_swap = current_coverage < 1.0 or saving >= policy.activation_threshold
        if allow_swap:
            for probe in target:
                if probe not in self.active:
                    self.active.add(probe)
                    self.activations += 1

        # Retire locally redundant probes first; when a global size-reducing
        # target was admitted, non-target probes are eligible as long as their
        # removal keeps the hard coverage invariant intact.
        for probe in tuple(probe for probe in self.economy.probes if probe in self.active):
            remaining = tuple(p for p in self.active if p != probe)
            loss = self.economy.removal_loss(probe, self.active)
            target_swap = allow_swap and probe not in target
            if (
                self.economy.coverage(remaining) >= 1.0
                and (loss <= policy.retirement_threshold or target_swap)
            ):
                self.active.remove(probe)
                self.retirements += 1

        # Hard repair: representation cost never overrides empirical sufficiency.
        if self.economy.coverage(self.active) < 1.0:
            target = self.economy.exact_minimal_cover(prefer=self.active)
            for probe in target:
                if probe not in self.active:
                    self.active.add(probe)
                    self.activations += 1

        return tuple(probe for probe in self.economy.probes if probe in self.active)

    @property
    def snapshot(self) -> dict:
        active = tuple(probe for probe in self.economy.probes if probe in self.active)
        return {
            "active": len(active),
            "latent": len(self.economy.probes) - len(active),
            "coverage": self.economy.coverage(active),
            "active_probe_indices": tuple(self.economy._index[probe] for probe in active),
            "activations": self.activations,
            "retirements": self.retirements,
            "steps": self.steps,
        }
