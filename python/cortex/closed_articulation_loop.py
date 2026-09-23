"""Closed articulation loop for Cortex research.

This module implements the functional loop associated with downward articulation,
resolution-conditioned memory, guarded bubbling, and upward re-articulation
without changing the frozen native runtime or spec-v0.9.5.

The mechanism is intentionally generic. It does not know about OWR-A/B/C/D,
source reputation, named novelty classes, or a Cortex self-model. It operates
on opaque contexts, values, provenance, event time, and memory pressure.
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from math import isfinite, log1p
from typing import Iterable

from .articulation_controller import ArticulationState
from .regime_plasticity import RegimePlasticityPolicy, RelationalRegimeTracker


ContextKey = tuple[str, ...]
Fingerprint = tuple[ContextKey, int]


@dataclass(frozen=True, slots=True)
class ClosedLoopPolicy:
    """Research policy for the closed articulation loop."""

    active_budget: int = 64
    latent_budget: int = 192
    utility_decay: float = 0.997
    dependence_min_overlap: int = 6
    dependence_agreement: float = 0.95
    recurrence_min_history: float = 5.0
    recurrence_confirmations: int = 3
    residual_window: int = 6
    compound_min_events: int = 3
    late_event_tolerance: int = 0

    def __post_init__(self) -> None:
        if self.active_budget < 1:
            raise ValueError("active_budget must be positive")
        if self.latent_budget < 0:
            raise ValueError("latent_budget must be non-negative")
        if not isfinite(self.utility_decay) or not 0.0 < self.utility_decay <= 1.0:
            raise ValueError("utility_decay must lie in (0,1]")
        if self.dependence_min_overlap < 1:
            raise ValueError("dependence_min_overlap must be positive")
        if (
            not isfinite(self.dependence_agreement)
            or not 0.0 <= self.dependence_agreement <= 1.0
        ):
            raise ValueError("dependence_agreement must lie in [0,1]")
        if (
            not isfinite(self.recurrence_min_history)
            or self.recurrence_min_history <= 0.0
        ):
            raise ValueError("recurrence_min_history must be positive")
        if self.recurrence_confirmations < 1:
            raise ValueError("recurrence_confirmations must be positive")
        if self.residual_window < 1:
            raise ValueError("residual_window must be positive")
        if self.compound_min_events < 1:
            raise ValueError("compound_min_events must be positive")
        if self.late_event_tolerance < 0:
            raise ValueError("late_event_tolerance must be non-negative")

    @property
    def total_budget(self) -> int:
        return self.active_budget + self.latent_budget


@dataclass(frozen=True, slots=True)
class EvidenceAtom:
    """One learner-visible piece of evidence."""

    context: ContextKey
    relation: str | None
    event_time: int
    arrival_time: int
    provenance: str = "unspecified"
    confidence: float = 1.0
    bootstrap: bool = False

    def __post_init__(self) -> None:
        if not self.context or any(not str(item) for item in self.context):
            raise ValueError("context must contain non-empty symbols")
        if self.relation is not None and not str(self.relation):
            raise ValueError("relation must be non-empty when present")
        if self.event_time < 0 or self.arrival_time < 0:
            raise ValueError("event and arrival times must be non-negative")
        if not self.provenance:
            raise ValueError("provenance must be non-empty")
        if (
            not isfinite(float(self.confidence))
            or not 0.0 <= float(self.confidence) <= 1.0
        ):
            raise ValueError("confidence must lie in [0,1]")


@dataclass(slots=True)
class MemoryBasin:
    basin_id: int
    context: ContextKey
    relation: str
    effective_mass: float
    raw_mass: int
    utility: float
    first_event_time: int
    last_event_time: int
    last_arrival_time: int
    provenances: set[str] = field(default_factory=set)
    active: bool = True
    activations: int = 0


@dataclass(frozen=True, slots=True)
class MemoryActivation:
    basin_id: int
    context: ContextKey
    relation: str
    score: float
    effective_mass: float
    active: bool


@dataclass(frozen=True, slots=True)
class StructuralChange:
    """Generic residual distinction discovered by downward articulation."""

    subject: ContextKey
    axis: str
    before: str | None
    after: str


@dataclass(frozen=True, slots=True)
class BackgroundCandidate:
    changes: tuple[StructuralChange, ...]
    support_events: int
    provenance_count: int
    bubble_eligible: bool


@dataclass(frozen=True, slots=True)
class ClosedLoopRead:
    context: ContextKey
    relation: str | None
    accepted_relation: str | None
    state: ArticulationState
    historical_only: bool
    source_weight: float
    reactivated: bool
    active_memory: tuple[MemoryActivation, ...]
    background_candidate: BackgroundCandidate | None
    novel_coordinates: tuple[str, ...]
    stored_basins: int
    active_basins: int
    latent_basins: int
    coarsened_basins: int
    coarsened_mass: float


@dataclass(frozen=True, slots=True)
class ClosedLoopSnapshot:
    stored_basins: int
    active_basins: int
    latent_basins: int
    coarsened_basins: int
    coarsened_mass: float
    contexts: int
    provenance_groups: tuple[tuple[str, ...], ...]


@dataclass(frozen=True, slots=True)
class _ResidualEvent:
    arrival_time: int
    provenance: str
    changes: tuple[StructuralChange, ...]


class ClosedArticulationLoop:
    """Research-only foreground/downward-memory/upward loop."""

    def __init__(
        self,
        policy: ClosedLoopPolicy | None = None,
        *,
        regime_policy: RegimePlasticityPolicy | None = None,
    ) -> None:
        self.policy = policy or ClosedLoopPolicy()
        self.regime_policy = regime_policy or RegimePlasticityPolicy()
        self._trackers: dict[ContextKey, RelationalRegimeTracker] = {}
        self._basins: dict[tuple[ContextKey, str], MemoryBasin] = {}
        self._next_basin_id = 0
        self._event_frontier: dict[ContextKey, int] = {}
        self._claims: dict[str, dict[Fingerprint, str]] = defaultdict(dict)
        self._recurrence_streak: dict[tuple[ContextKey, str], int] = {}
        self._residuals: deque[_ResidualEvent] = deque()
        self._coordinate_values: dict[int, set[str]] = defaultdict(set)
        self._relation_values: set[str] = set()
        self._coarsened_basins = 0
        self._coarsened_mass = 0.0
        self._step = 0

    @property
    def snapshot(self) -> ClosedLoopSnapshot:
        active = sum(int(basin.active) for basin in self._basins.values())
        groups = tuple(
            sorted(
                (tuple(sorted(group)) for group in self.provenance_groups()),
                key=lambda row: row,
            )
        )
        return ClosedLoopSnapshot(
            stored_basins=len(self._basins),
            active_basins=active,
            latent_basins=len(self._basins) - active,
            coarsened_basins=self._coarsened_basins,
            coarsened_mass=self._coarsened_mass,
            contexts=len(self._trackers),
            provenance_groups=groups,
        )

    def provenance_groups(self) -> tuple[frozenset[str], ...]:
        """Infer dependence classes from repeated public claim agreement."""

        names = sorted(self._claims)
        if not names:
            return ()

        parent = {name: name for name in names}

        def find(name: str) -> str:
            root = name
            while parent[root] != root:
                root = parent[root]
            while parent[name] != name:
                nxt = parent[name]
                parent[name] = root
                name = nxt
            return root

        def union(left: str, right: str) -> None:
            a = find(left)
            b = find(right)
            if a != b:
                parent[b] = a

        for i, left in enumerate(names):
            left_claims = self._claims[left]
            for right in names[i + 1 :]:
                right_claims = self._claims[right]
                overlap = set(left_claims) & set(right_claims)
                if len(overlap) < self.policy.dependence_min_overlap:
                    continue
                agreement = sum(
                    left_claims[key] == right_claims[key] for key in overlap
                ) / len(overlap)
                if agreement >= self.policy.dependence_agreement:
                    union(left, right)

        groups: dict[str, set[str]] = defaultdict(set)
        for name in names:
            groups[find(name)].add(name)
        return tuple(
            frozenset(group)
            for group in sorted(groups.values(), key=lambda value: sorted(value))
        )

    def _provenance_weight(self, provenance: str) -> float:
        for group in self.provenance_groups():
            if provenance in group:
                return 1.0 / len(group)
        return 1.0

    def _decay_utility(self) -> None:
        for basin in self._basins.values():
            basin.utility *= self.policy.utility_decay

    def _novel_coordinates(self, atom: EvidenceAtom) -> tuple[str, ...]:
        novel = []
        for index, value in enumerate(atom.context):
            if value not in self._coordinate_values[index]:
                novel.append(f"context:{index}")
        if atom.relation is not None and atom.relation not in self._relation_values:
            novel.append("relation")
        return tuple(novel)

    def _record_vocabulary(self, atom: EvidenceAtom) -> None:
        for index, value in enumerate(atom.context):
            self._coordinate_values[index].add(value)
        if atom.relation is not None:
            self._relation_values.add(atom.relation)

    def _update_memory(
        self,
        atom: EvidenceAtom,
        source_weight: float,
    ) -> tuple[MemoryBasin | None, float]:
        if atom.relation is None:
            return None, 0.0
        key = (atom.context, atom.relation)
        basin = self._basins.get(key)
        prior_mass = basin.effective_mass if basin is not None else 0.0
        if basin is None:
            basin = MemoryBasin(
                basin_id=self._next_basin_id,
                context=atom.context,
                relation=atom.relation,
                effective_mass=0.0,
                raw_mass=0,
                utility=0.0,
                first_event_time=atom.event_time,
                last_event_time=atom.event_time,
                last_arrival_time=atom.arrival_time,
            )
            self._next_basin_id += 1
            self._basins[key] = basin

        contribution = float(atom.confidence) * source_weight
        basin.effective_mass += contribution
        basin.raw_mass += 1
        basin.utility += contribution
        basin.last_event_time = max(basin.last_event_time, atom.event_time)
        basin.last_arrival_time = max(basin.last_arrival_time, atom.arrival_time)
        basin.provenances.add(atom.provenance)
        basin.active = True
        self._enforce_budget(protect=key)
        return basin, prior_mass

    def _enforce_budget(self, *, protect: tuple[ContextKey, str]) -> None:
        active = [basin for basin in self._basins.values() if basin.active]
        if len(active) > self.policy.active_budget:
            demotable = [
                basin
                for basin in active
                if (basin.context, basin.relation) != protect
            ]
            if demotable:
                victim = min(
                    demotable,
                    key=lambda row: (
                        row.utility * log1p(row.effective_mass),
                        row.last_arrival_time,
                        row.basin_id,
                    ),
                )
                victim.active = False

        while len(self._basins) > self.policy.total_budget:
            latent = [
                basin
                for basin in self._basins.values()
                if not basin.active
                and (basin.context, basin.relation) != protect
            ]
            pool = latent or [
                basin
                for basin in self._basins.values()
                if (basin.context, basin.relation) != protect
            ]
            if not pool:
                return
            victim = min(
                pool,
                key=lambda row: (
                    row.utility * log1p(row.effective_mass),
                    row.last_arrival_time,
                    row.basin_id,
                ),
            )
            self._coarsened_basins += 1
            self._coarsened_mass += victim.effective_mass
            del self._basins[(victim.context, victim.relation)]

    def condition(
        self,
        context: ContextKey,
        *,
        limit: int = 4,
    ) -> tuple[MemoryActivation, ...]:
        """Select memory relevant to one current context."""

        context = tuple(str(item) for item in context)
        rows = []
        for basin in self._basins.values():
            if basin.context != context:
                continue
            score = (
                basin.effective_mass / (1.0 + basin.effective_mass)
            ) * (
                basin.utility / (1.0 + basin.utility)
            )
            rows.append(
                MemoryActivation(
                    basin_id=basin.basin_id,
                    context=basin.context,
                    relation=basin.relation,
                    score=score,
                    effective_mass=basin.effective_mass,
                    active=basin.active,
                )
            )
        rows.sort(
            key=lambda row: (
                row.score,
                row.effective_mass,
                row.relation,
            ),
            reverse=True,
        )
        selected = rows[: max(1, int(limit))]
        selected_ids = {row.basin_id for row in selected}
        for basin in self._basins.values():
            if basin.basin_id in selected_ids:
                basin.activations += 1
                basin.utility += 1.0
                basin.active = True
        if selected:
            self._enforce_budget(
                protect=(selected[0].context, selected[0].relation)
            )
        return tuple(selected)

    def _append_residual(
        self,
        atom: EvidenceAtom,
        *,
        previous_relation: str | None,
        novel_coordinates: tuple[str, ...],
        historical_only: bool,
    ) -> BackgroundCandidate | None:
        if historical_only or atom.relation is None:
            return self._background_candidate(atom.arrival_time)

        changes = []
        for axis in novel_coordinates:
            if axis == "relation":
                after = atom.relation
            else:
                index = int(axis.split(":", 1)[1])
                after = atom.context[index]
            changes.append(
                StructuralChange(
                    subject=atom.context,
                    axis=axis,
                    before=None,
                    after=after,
                )
            )
        if (
            previous_relation is not None
            and previous_relation != atom.relation
        ):
            changes.append(
                StructuralChange(
                    subject=atom.context,
                    axis="current-relation",
                    before=previous_relation,
                    after=atom.relation,
                )
            )
        if changes:
            self._residuals.append(
                _ResidualEvent(
                    arrival_time=atom.arrival_time,
                    provenance=atom.provenance,
                    changes=tuple(changes),
                )
            )
        return self._background_candidate(atom.arrival_time)

    def _background_candidate(
        self,
        arrival_time: int,
    ) -> BackgroundCandidate | None:
        cutoff = arrival_time - self.policy.residual_window
        while self._residuals and self._residuals[0].arrival_time < cutoff:
            self._residuals.popleft()
        if not self._residuals:
            return None

        unique: dict[
            tuple[ContextKey, str, str | None, str],
            StructuralChange,
        ] = {}
        provenances = set()
        events = 0
        for event in self._residuals:
            if not event.changes:
                continue
            events += 1
            provenances.add(event.provenance)
            for change in event.changes:
                unique[
                    (
                        change.subject,
                        change.axis,
                        change.before,
                        change.after,
                    )
                ] = change

        if not unique:
            return None
        changes = tuple(
            sorted(
                unique.values(),
                key=lambda row: (
                    row.axis,
                    row.subject,
                    row.before or "",
                    row.after,
                ),
            )
        )
        distinct_subjects = {change.subject for change in changes}
        compound = len(changes) >= 2 and len(distinct_subjects) >= 2
        eligible = compound and events >= self.policy.compound_min_events
        return BackgroundCandidate(
            changes=changes,
            support_events=events,
            provenance_count=len(provenances),
            bubble_eligible=eligible,
        )

    def _try_reactivate(
        self,
        *,
        atom: EvidenceAtom,
        tracker: RelationalRegimeTracker,
        prior_mass: float,
        historical_only: bool,
        source_weight: float,
    ) -> bool:
        if (
            historical_only
            or atom.relation is None
            or source_weight <= 0.0
            or tracker.accepted_relation in (None, atom.relation)
            or prior_mass + 1.0e-12 < self.policy.recurrence_min_history
        ):
            return False

        key = (atom.context, atom.relation)
        self._recurrence_streak[key] = self._recurrence_streak.get(key, 0) + 1
        for other in tuple(self._recurrence_streak):
            if other[0] == atom.context and other != key:
                self._recurrence_streak[other] = 0

        if self._recurrence_streak[key] < self.policy.recurrence_confirmations:
            return False

        tracker.reactivate_from_memory(atom.relation)
        self._recurrence_streak[key] = 0
        for basin in self._basins.values():
            if basin.context == atom.context:
                basin.active = basin.relation == atom.relation
        return True

    def observe(self, atom: EvidenceAtom) -> ClosedLoopRead:
        """Consume one observation through the closed articulation loop."""

        self._step += 1
        self._decay_utility()
        context = tuple(str(item) for item in atom.context)
        if context != atom.context:
            atom = EvidenceAtom(
                context=context,
                relation=atom.relation,
                event_time=atom.event_time,
                arrival_time=atom.arrival_time,
                provenance=atom.provenance,
                confidence=atom.confidence,
                bootstrap=atom.bootstrap,
            )

        novel_coordinates = self._novel_coordinates(atom)

        if atom.relation is not None:
            self._claims[atom.provenance][
                (atom.context, atom.event_time)
            ] = atom.relation
        source_weight = self._provenance_weight(atom.provenance)

        frontier = self._event_frontier.get(atom.context, -1)
        historical_only = (
            frontier >= 0
            and atom.event_time + self.policy.late_event_tolerance < frontier
        )
        self._event_frontier[atom.context] = max(frontier, atom.event_time)

        tracker = self._trackers.get(atom.context)
        previous_relation = (
            tracker.accepted_relation if tracker is not None else None
        )

        _basin, prior_mass = self._update_memory(atom, source_weight)

        reactivated = False
        if atom.relation is not None and not historical_only:
            if tracker is None:
                tracker = RelationalRegimeTracker(self.regime_policy)
                self._trackers[atom.context] = tracker
            snap = tracker.observe(
                atom.relation,
                bootstrap=atom.bootstrap or tracker.accepted_relation is None,
                current_weight=float(atom.confidence) * source_weight,
            )
            reactivated = self._try_reactivate(
                atom=atom,
                tracker=tracker,
                prior_mass=prior_mass,
                historical_only=historical_only,
                source_weight=source_weight,
            )
            if reactivated:
                snap = tracker.snapshot
        elif tracker is not None:
            snap = tracker.snapshot
        else:
            snap = None

        candidate = self._append_residual(
            atom,
            previous_relation=previous_relation,
            novel_coordinates=novel_coordinates,
            historical_only=historical_only,
        )
        self._record_vocabulary(atom)

        active_memory = self.condition(atom.context)
        if snap is None:
            accepted = None
            state = ArticulationState.EXPLORE
        else:
            accepted = snap.accepted_relation
            state = snap.state

        state_counts = self.snapshot
        return ClosedLoopRead(
            context=atom.context,
            relation=atom.relation,
            accepted_relation=accepted,
            state=state,
            historical_only=historical_only,
            source_weight=source_weight,
            reactivated=reactivated,
            active_memory=active_memory,
            background_candidate=candidate,
            novel_coordinates=novel_coordinates,
            stored_basins=state_counts.stored_basins,
            active_basins=state_counts.active_basins,
            latent_basins=state_counts.latent_basins,
            coarsened_basins=state_counts.coarsened_basins,
            coarsened_mass=state_counts.coarsened_mass,
        )

    def consume(
        self,
        atoms: Iterable[EvidenceAtom],
    ) -> tuple[ClosedLoopRead, ...]:
        return tuple(self.observe(atom) for atom in atoms)
