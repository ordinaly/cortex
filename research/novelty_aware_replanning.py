
"""Novelty-aware integration of structural attribution and plan repair.

Research-only Cortex Reasoning Map integration prototype.

A novelty attribution is first resolved against the current structural model.
Only resolved structural novelty may mutate that model. The resulting edit is
mapped onto explicit planning dependencies. Unaffected plans are preserved;
localized dependency changes invoke dependency-indexed repair; regime changes
invoke global replanning; unresolved novelty defers structural revision.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Mapping

from open_world_novelty import (
    AttributionRead,
    CortexNoveltyAttributor,
    EvidenceWindow,
    WorldModel,
    apply_attribution,
    changed_components,
)
from planning_replanning import (
    Plan,
    PlanningWorld,
    RepairResult,
    execute_actions,
    plan_valid,
    repair_plan_from_index,
    uniform_cost_plan,
)


@dataclass(frozen=True)
class DependencyRef:
    kind: str
    key: Hashable

    def matches(self, other: "DependencyRef") -> bool:
        if self.kind != other.kind:
            return False
        return (
            self.key == "*"
            or other.key == "*"
            or self.key == other.key
        )


@dataclass(frozen=True)
class PlanningDependencyIndex:
    action_dependencies: Mapping[
        str,
        frozenset[DependencyRef],
    ]
    opportunity_indices: Mapping[
        DependencyRef,
        int,
    ]

    def impact_index(
        self,
        plan: Plan,
        changed: tuple[DependencyRef, ...],
    ) -> int | None:
        indices: list[int] = []

        for index, action_id in enumerate(
            plan.action_ids
        ):
            dependencies = self.action_dependencies.get(
                action_id,
                frozenset(),
            )
            if any(
                dependency.matches(change)
                for dependency in dependencies
                for change in changed
            ):
                indices.append(index)

        for dependency, index in (
            self.opportunity_indices.items()
        ):
            if any(
                dependency.matches(change)
                for change in changed
            ):
                indices.append(int(index))

        if not indices:
            return None
        return min(indices)


@dataclass(frozen=True)
class NoveltyAwareResult:
    attribution: AttributionRead
    decision: str
    plan: Plan | None
    model: WorldModel
    model_changed_components: tuple[str, ...]
    impact_index: int | None
    preserved_prefix_actions: int
    preserved_suffix_actions: int
    local_expanded_states: int
    global_expanded_states: int
    safe_to_execute: bool

    @property
    def total_expanded_states(self) -> int:
        return (
            self.local_expanded_states
            + self.global_expanded_states
        )


def dependencies_from_attribution(
    read: AttributionRead,
) -> tuple[DependencyRef, ...]:
    """Translate one resolved structural edit into planning dependency keys."""

    if (
        not read.resolved
        or read.proposed_edit is None
    ):
        return tuple()

    edit = read.proposed_edit

    if edit.category == "new_instance":
        return tuple(
            DependencyRef(
                "instance",
                instance_id,
            )
            for instance_id, _type_id
            in edit.payload
        )

    if edit.category == "new_entity_type":
        (
            type_id,
            _probabilities,
            instance_ids,
        ) = edit.payload
        return (
            DependencyRef(
                "type",
                type_id,
            ),
            *(
                DependencyRef(
                    "instance",
                    instance_id,
                )
                for instance_id
                in instance_ids
            ),
        )

    if edit.category == "new_relation":
        key, _probability = edit.payload
        return (
            DependencyRef(
                "relation",
                tuple(key),
            ),
        )

    if edit.category == "new_regime":
        return (
            DependencyRef(
                "regime",
                "*",
            ),
        )

    raise ValueError(
        f"unsupported novelty edit category: {edit.category}"
    )


class CortexNoveltyAwarePlanner:
    """Use novelty attribution to choose preservation, local repair, or reset."""

    def __init__(
        self,
        model: WorldModel,
        *,
        attributor: CortexNoveltyAttributor | None = None,
    ) -> None:
        self.model = model
        self.attributor = (
            attributor
            if attributor is not None
            else CortexNoveltyAttributor(model)
        )

    @staticmethod
    def _preserved_plan(
        world: PlanningWorld,
        original_plan: Plan,
    ) -> Plan:
        replay = execute_actions(
            world,
            original_plan.action_ids,
        )
        if not plan_valid(
            world,
            replay,
        ):
            raise ValueError(
                "preserved plan is invalid in selected world"
            )
        return replay

    def respond(
        self,
        evidence: EvidenceWindow,
        *,
        original_world: PlanningWorld,
        updated_world: PlanningWorld,
        original_plan: Plan,
        dependencies: PlanningDependencyIndex,
    ) -> NoveltyAwareResult:
        if not plan_valid(
            original_world,
            original_plan,
        ):
            raise ValueError(
                "original plan must be valid"
            )

        attribution = self.attributor.evaluate(
            evidence
        )

        if not attribution.resolved:
            return NoveltyAwareResult(
                attribution=attribution,
                decision="deferred",
                plan=original_plan,
                model=self.model,
                model_changed_components=tuple(),
                impact_index=None,
                preserved_prefix_actions=len(
                    original_plan.action_ids
                ),
                preserved_suffix_actions=0,
                local_expanded_states=0,
                global_expanded_states=0,
                safe_to_execute=False,
            )

        if attribution.category == "noise":
            preserved = self._preserved_plan(
                original_world,
                original_plan,
            )
            return NoveltyAwareResult(
                attribution=attribution,
                decision="preserve",
                plan=preserved,
                model=self.model,
                model_changed_components=tuple(),
                impact_index=None,
                preserved_prefix_actions=len(
                    original_plan.action_ids
                ),
                preserved_suffix_actions=0,
                local_expanded_states=0,
                global_expanded_states=0,
                safe_to_execute=True,
            )

        updated_model = apply_attribution(
            self.model,
            attribution,
        )
        model_components = changed_components(
            self.model,
            updated_model,
        )
        changed_dependencies = (
            dependencies_from_attribution(
                attribution
            )
        )

        if attribution.category == "new_regime":
            global_plan = uniform_cost_plan(
                updated_world
            )
            if global_plan is None:
                return NoveltyAwareResult(
                    attribution=attribution,
                    decision="infeasible",
                    plan=None,
                    model=updated_model,
                    model_changed_components=(
                        model_components
                    ),
                    impact_index=0,
                    preserved_prefix_actions=0,
                    preserved_suffix_actions=0,
                    local_expanded_states=0,
                    global_expanded_states=0,
                    safe_to_execute=False,
                )
            return NoveltyAwareResult(
                attribution=attribution,
                decision="global-replan",
                plan=global_plan,
                model=updated_model,
                model_changed_components=(
                    model_components
                ),
                impact_index=0,
                preserved_prefix_actions=0,
                preserved_suffix_actions=0,
                local_expanded_states=0,
                global_expanded_states=(
                    global_plan.expanded_states
                ),
                safe_to_execute=True,
            )

        impact_index = dependencies.impact_index(
            original_plan,
            changed_dependencies,
        )

        if impact_index is None:
            preserved = self._preserved_plan(
                updated_world,
                original_plan,
            )
            return NoveltyAwareResult(
                attribution=attribution,
                decision="preserve",
                plan=preserved,
                model=updated_model,
                model_changed_components=(
                    model_components
                ),
                impact_index=None,
                preserved_prefix_actions=len(
                    original_plan.action_ids
                ),
                preserved_suffix_actions=0,
                local_expanded_states=0,
                global_expanded_states=0,
                safe_to_execute=True,
            )

        repair = repair_plan_from_index(
            original_world,
            updated_world,
            original_plan,
            impact_index,
        )
        return self._from_repair(
            attribution=attribution,
            updated_model=updated_model,
            model_components=model_components,
            repair=repair,
        )

    @staticmethod
    def _from_repair(
        *,
        attribution: AttributionRead,
        updated_model: WorldModel,
        model_components: tuple[str, ...],
        repair: RepairResult,
    ) -> NoveltyAwareResult:
        if repair.mode == "local-repair":
            decision = "local-repair"
        elif repair.mode == "full-replan":
            decision = "global-replan"
        elif repair.mode == "infeasible":
            decision = "infeasible"
        else:
            decision = repair.mode

        return NoveltyAwareResult(
            attribution=attribution,
            decision=decision,
            plan=repair.plan,
            model=updated_model,
            model_changed_components=(
                model_components
            ),
            impact_index=repair.impact_index,
            preserved_prefix_actions=(
                repair.preserved_prefix_actions
            ),
            preserved_suffix_actions=(
                repair.preserved_suffix_actions
            ),
            local_expanded_states=(
                repair.local_expanded_states
            ),
            global_expanded_states=(
                repair.fallback_expanded_states
            ),
            safe_to_execute=(
                repair.plan is not None
            ),
        )


class AlwaysReplanBaseline:
    """Apply resolved edits and restart planning after every non-noise signal."""

    def respond(
        self,
        read: AttributionRead,
        *,
        model: WorldModel,
        original_world: PlanningWorld,
        updated_world: PlanningWorld,
        original_plan: Plan,
    ) -> tuple[str, Plan | None, int]:
        if not read.resolved:
            return (
                "deferred",
                original_plan,
                0,
            )
        if read.category == "noise":
            return (
                "preserve",
                original_plan,
                0,
            )

        _ = apply_attribution(
            model,
            read,
        )
        plan = uniform_cost_plan(
            updated_world
        )
        return (
            "global-replan"
            if plan is not None
            else "infeasible",
            plan,
            (
                0
                if plan is None
                else plan.expanded_states
            ),
        )


class IgnoreNoveltyBaseline:
    """Never update the structural model and always keep the original plan."""

    def respond(
        self,
        *,
        updated_world: PlanningWorld,
        original_plan: Plan,
    ) -> tuple[str, bool]:
        try:
            replay = execute_actions(
                updated_world,
                original_plan.action_ids,
            )
        except (KeyError, ValueError):
            return ("preserve", False)
        return (
            "preserve",
            plan_valid(
                updated_world,
                replay,
            ),
        )
