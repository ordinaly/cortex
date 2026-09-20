"""Finite planning and local replanning prototype for Cortex Reasoning Map v1.

The model uses deterministic actions with positive preconditions, add/delete
effects, and integer costs. Planning is exact uniform-cost search over finite
states. Replanning first preserves the longest unchanged executable prefix,
then searches for the earliest reachable state on the old trajectory whose
remaining suffix is still valid. If no such local rejoin exists, the planner
falls back to a full replan.

Each returned plan can be annotated with explicit temporal support provenance:
every action precondition and final goal fact records the initial fact or prior
action that currently supports it.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from heapq import heappop, heappush
from json import dumps
from typing import Iterable, Mapping

State = frozenset[str]


@dataclass(frozen=True)
class Action:
    action_id: str
    preconditions: frozenset[str]
    add_effects: frozenset[str]
    delete_effects: frozenset[str]
    cost: int = 1

    def __post_init__(self) -> None:
        if not self.action_id:
            raise ValueError("action_id must be non-empty")
        if self.cost <= 0:
            raise ValueError("action cost must be positive")
        if self.add_effects & self.delete_effects:
            raise ValueError(
                "add and delete effects must be disjoint"
            )

    def applicable(self, state: State) -> bool:
        return self.preconditions.issubset(state)

    def apply(self, state: State) -> State:
        if not self.applicable(state):
            raise ValueError(
                f"action {self.action_id} is not applicable"
            )
        return frozenset(
            (state - self.delete_effects)
            | self.add_effects
        )


@dataclass(frozen=True)
class PlanningWorld:
    initial_state: State
    goal: frozenset[str]
    actions: Mapping[str, Action]

    def __post_init__(self) -> None:
        if not self.goal:
            raise ValueError("goal must be non-empty")
        if not self.actions:
            raise ValueError("at least one action is required")
        if any(
            action_id != action.action_id
            for action_id, action in self.actions.items()
        ):
            raise ValueError(
                "action mapping keys must match action IDs"
            )

    def action(self, action_id: str) -> Action:
        try:
            return self.actions[action_id]
        except KeyError as exc:
            raise KeyError(
                f"unknown action: {action_id}"
            ) from exc


@dataclass(frozen=True)
class Plan:
    action_ids: tuple[str, ...]
    states: tuple[State, ...]
    total_cost: int
    expanded_states: int

    def __post_init__(self) -> None:
        if len(self.states) != len(self.action_ids) + 1:
            raise ValueError(
                "plan states must include initial and every result"
            )

    def fingerprint(self) -> str:
        payload = {
            "actions": self.action_ids,
            "states": [
                sorted(state)
                for state in self.states
            ],
            "cost": self.total_cost,
        }
        return sha256(
            dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()


@dataclass(frozen=True)
class SupportRef:
    atom: str
    source_kind: str
    source_id: str
    source_step: int


@dataclass(frozen=True)
class StepSupport:
    step_index: int
    action_id: str
    preconditions: tuple[SupportRef, ...]


@dataclass(frozen=True)
class PlanSupport:
    steps: tuple[StepSupport, ...]
    goals: tuple[SupportRef, ...]

    def fingerprint(self) -> str:
        payload = {
            "steps": [
                {
                    "step": step.step_index,
                    "action": step.action_id,
                    "preconditions": [
                        (
                            ref.atom,
                            ref.source_kind,
                            ref.source_id,
                            ref.source_step,
                        )
                        for ref in step.preconditions
                    ],
                }
                for step in self.steps
            ],
            "goals": [
                (
                    ref.atom,
                    ref.source_kind,
                    ref.source_id,
                    ref.source_step,
                )
                for ref in self.goals
            ],
        }
        return sha256(
            dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()


@dataclass(frozen=True)
class RepairResult:
    mode: str
    plan: Plan | None
    impact_index: int | None
    rejoin_index: int | None
    preserved_prefix_actions: int
    preserved_suffix_actions: int
    local_expanded_states: int
    fallback_expanded_states: int

    @property
    def total_expanded_states(self) -> int:
        return (
            self.local_expanded_states
            + self.fallback_expanded_states
        )


def _state_key(state: State) -> tuple[str, ...]:
    return tuple(sorted(state))


def _goal_reached(
    state: State,
    target: frozenset[str],
    *,
    exact: bool,
) -> bool:
    if exact:
        return state == target
    return target.issubset(state)


def _reconstruct_plan(
    world: PlanningWorld,
    *,
    start_state: State,
    goal_state: State,
    parents: Mapping[
        State,
        tuple[State, str],
    ],
    expanded_states: int,
) -> Plan:
    actions_reversed: list[str] = []
    states_reversed: list[State] = [goal_state]
    cursor = goal_state

    while cursor != start_state:
        previous, action_id = parents[cursor]
        actions_reversed.append(action_id)
        cursor = previous
        states_reversed.append(cursor)

    action_ids = tuple(reversed(actions_reversed))
    states = tuple(reversed(states_reversed))
    total_cost = sum(
        world.action(action_id).cost
        for action_id in action_ids
    )
    return Plan(
        action_ids=action_ids,
        states=states,
        total_cost=total_cost,
        expanded_states=expanded_states,
    )


def uniform_cost_plan(
    world: PlanningWorld,
    *,
    start_state: State | None = None,
    target: frozenset[str] | None = None,
    exact_target: bool = False,
    max_expansions: int = 50_000,
) -> Plan | None:
    """Return an exact minimum-cost plan under deterministic tie-breaking."""

    start = (
        world.initial_state
        if start_state is None
        else frozenset(start_state)
    )
    goal = (
        world.goal
        if target is None
        else frozenset(target)
    )

    if _goal_reached(
        start,
        goal,
        exact=exact_target,
    ):
        return Plan(
            action_ids=tuple(),
            states=(start,),
            total_cost=0,
            expanded_states=0,
        )

    frontier: list[
        tuple[
            int,
            int,
            tuple[str, ...],
            State,
        ]
    ] = []
    heappush(
        frontier,
        (0, 0, _state_key(start), start),
    )
    best: dict[State, tuple[int, int]] = {
        start: (0, 0)
    }
    parents: dict[State, tuple[State, str]] = {}
    expanded = 0
    actions = tuple(
        world.actions[action_id]
        for action_id in sorted(world.actions)
    )

    while frontier:
        cost, steps, _, state = heappop(frontier)
        if best.get(state) != (cost, steps):
            continue

        if _goal_reached(
            state,
            goal,
            exact=exact_target,
        ):
            return _reconstruct_plan(
                world,
                start_state=start,
                goal_state=state,
                parents=parents,
                expanded_states=expanded,
            )

        expanded += 1
        if expanded > max_expansions:
            raise RuntimeError(
                "planning expansion budget exceeded"
            )

        for action in actions:
            if not action.applicable(state):
                continue
            nxt = action.apply(state)
            candidate = (
                cost + action.cost,
                steps + 1,
            )
            incumbent = best.get(nxt)
            if (
                incumbent is not None
                and candidate >= incumbent
            ):
                continue
            best[nxt] = candidate
            parents[nxt] = (
                state,
                action.action_id,
            )
            heappush(
                frontier,
                (
                    candidate[0],
                    candidate[1],
                    _state_key(nxt),
                    nxt,
                ),
            )

    return None


def execute_actions(
    world: PlanningWorld,
    action_ids: Iterable[str],
    *,
    start_state: State | None = None,
) -> Plan:
    state = (
        world.initial_state
        if start_state is None
        else frozenset(start_state)
    )
    states = [state]
    total_cost = 0
    used: list[str] = []

    for action_id in action_ids:
        action = world.action(action_id)
        state = action.apply(state)
        used.append(action_id)
        states.append(state)
        total_cost += action.cost

    return Plan(
        action_ids=tuple(used),
        states=tuple(states),
        total_cost=total_cost,
        expanded_states=0,
    )


def plan_valid(
    world: PlanningWorld,
    plan: Plan,
    *,
    require_goal: bool = True,
) -> bool:
    if not plan.states:
        return False
    if plan.states[0] != world.initial_state:
        return False

    try:
        replay = execute_actions(
            world,
            plan.action_ids,
        )
    except (KeyError, ValueError):
        return False

    if replay.states != plan.states:
        return False
    if replay.total_cost != plan.total_cost:
        return False
    if (
        require_goal
        and not world.goal.issubset(
            replay.states[-1]
        )
    ):
        return False
    return True


def plan_support(
    world: PlanningWorld,
    plan: Plan,
) -> PlanSupport:
    """Attach explicit temporal support provenance to a valid plan."""

    if not plan_valid(world, plan):
        raise ValueError(
            "support requires a valid goal-reaching plan"
        )

    provenance: dict[str, SupportRef] = {
        atom: SupportRef(
            atom=atom,
            source_kind="initial",
            source_id=atom,
            source_step=-1,
        )
        for atom in world.initial_state
    }
    state = world.initial_state
    steps: list[StepSupport] = []

    for index, action_id in enumerate(
        plan.action_ids
    ):
        action = world.action(action_id)
        if not action.applicable(state):
            raise RuntimeError(
                "plan became invalid during support build"
            )

        refs: list[SupportRef] = []
        for atom in sorted(action.preconditions):
            ref = provenance.get(atom)
            if ref is None:
                raise RuntimeError(
                    "active precondition lacks provenance"
                )
            refs.append(ref)

        steps.append(
            StepSupport(
                step_index=index,
                action_id=action_id,
                preconditions=tuple(refs),
            )
        )

        for atom in action.delete_effects:
            provenance.pop(atom, None)
        for atom in action.add_effects:
            provenance[atom] = SupportRef(
                atom=atom,
                source_kind="action",
                source_id=action_id,
                source_step=index,
            )
        state = action.apply(state)

    goals: list[SupportRef] = []
    for atom in sorted(world.goal):
        ref = provenance.get(atom)
        if ref is None:
            raise RuntimeError(
                "goal fact lacks provenance"
            )
        goals.append(ref)

    return PlanSupport(
        steps=tuple(steps),
        goals=tuple(goals),
    )


def support_is_valid(
    world: PlanningWorld,
    plan: Plan,
    support: PlanSupport,
) -> bool:
    try:
        rebuilt = plan_support(world, plan)
    except (KeyError, ValueError, RuntimeError):
        return False
    return rebuilt == support


def _longest_unchanged_prefix(
    original_world: PlanningWorld,
    updated_world: PlanningWorld,
    original_plan: Plan,
) -> tuple[int, State]:
    if (
        original_plan.states[0]
        != original_world.initial_state
    ):
        raise ValueError(
            "original plan does not start at original state"
        )

    state = updated_world.initial_state

    for index, action_id in enumerate(
        original_plan.action_ids
    ):
        if state != original_plan.states[index]:
            return index, state

        original_action = (
            original_world.actions.get(action_id)
        )
        updated_action = (
            updated_world.actions.get(action_id)
        )
        if (
            original_action is None
            or updated_action is None
            or updated_action != original_action
            or not updated_action.applicable(state)
        ):
            return index, state

        nxt = updated_action.apply(state)
        if nxt != original_plan.states[index + 1]:
            return index, state
        state = nxt

    return len(original_plan.action_ids), state


def _suffix_unchanged_and_valid(
    original_world: PlanningWorld,
    updated_world: PlanningWorld,
    original_plan: Plan,
    rejoin_index: int,
) -> bool:
    state = original_plan.states[rejoin_index]

    for index in range(
        rejoin_index,
        len(original_plan.action_ids),
    ):
        action_id = original_plan.action_ids[index]
        original_action = (
            original_world.actions.get(action_id)
        )
        updated_action = (
            updated_world.actions.get(action_id)
        )
        if (
            original_action is None
            or updated_action is None
            or updated_action != original_action
            or not updated_action.applicable(state)
        ):
            return False
        state = updated_action.apply(state)
        if state != original_plan.states[index + 1]:
            return False

    return updated_world.goal.issubset(state)


def _combine_repair(
    updated_world: PlanningWorld,
    original_plan: Plan,
    *,
    impact_index: int,
    rejoin_index: int,
    repair_segment: Plan,
) -> Plan:
    action_ids = (
        original_plan.action_ids[:impact_index]
        + repair_segment.action_ids
        + original_plan.action_ids[rejoin_index:]
    )
    replay = execute_actions(
        updated_world,
        action_ids,
    )
    return Plan(
        action_ids=replay.action_ids,
        states=replay.states,
        total_cost=replay.total_cost,
        expanded_states=repair_segment.expanded_states,
    )


def repair_plan(
    original_world: PlanningWorld,
    updated_world: PlanningWorld,
    original_plan: Plan,
    *,
    max_expansions: int = 50_000,
) -> RepairResult:
    """Repair by earliest valid trajectory rejoin, else fall back globally."""

    if not plan_valid(
        original_world,
        original_plan,
    ):
        raise ValueError(
            "original plan must be valid in original world"
        )

    impact_index, prefix_state = (
        _longest_unchanged_prefix(
            original_world,
            updated_world,
            original_plan,
        )
    )

    if impact_index == len(
        original_plan.action_ids
    ):
        replay = execute_actions(
            updated_world,
            original_plan.action_ids,
        )
        if updated_world.goal.issubset(
            replay.states[-1]
        ):
            return RepairResult(
                mode="unchanged",
                plan=replay,
                impact_index=None,
                rejoin_index=None,
                preserved_prefix_actions=len(
                    original_plan.action_ids
                ),
                preserved_suffix_actions=0,
                local_expanded_states=0,
                fallback_expanded_states=0,
            )

    local_expanded = 0

    for rejoin_index in range(
        impact_index + 1,
        len(original_plan.states),
    ):
        if not _suffix_unchanged_and_valid(
            original_world,
            updated_world,
            original_plan,
            rejoin_index,
        ):
            continue

        target_state = original_plan.states[
            rejoin_index
        ]
        segment = uniform_cost_plan(
            updated_world,
            start_state=prefix_state,
            target=target_state,
            exact_target=True,
            max_expansions=max_expansions,
        )
        if segment is None:
            continue
        local_expanded += segment.expanded_states

        candidate = _combine_repair(
            updated_world,
            original_plan,
            impact_index=impact_index,
            rejoin_index=rejoin_index,
            repair_segment=segment,
        )
        if not plan_valid(
            updated_world,
            candidate,
        ):
            continue

        return RepairResult(
            mode="local-repair",
            plan=candidate,
            impact_index=impact_index,
            rejoin_index=rejoin_index,
            preserved_prefix_actions=impact_index,
            preserved_suffix_actions=(
                len(original_plan.action_ids)
                - rejoin_index
            ),
            local_expanded_states=local_expanded,
            fallback_expanded_states=0,
        )

    fallback = uniform_cost_plan(
        updated_world,
        max_expansions=max_expansions,
    )
    if fallback is None:
        return RepairResult(
            mode="infeasible",
            plan=None,
            impact_index=impact_index,
            rejoin_index=None,
            preserved_prefix_actions=0,
            preserved_suffix_actions=0,
            local_expanded_states=local_expanded,
            fallback_expanded_states=0,
        )

    return RepairResult(
        mode="full-replan",
        plan=fallback,
        impact_index=impact_index,
        rejoin_index=None,
        preserved_prefix_actions=0,
        preserved_suffix_actions=0,
        local_expanded_states=local_expanded,
        fallback_expanded_states=(
            fallback.expanded_states
        ),
    )


def action_sequence_cost(
    world: PlanningWorld,
    action_ids: Iterable[str],
) -> int | None:
    try:
        replay = execute_actions(
            world,
            action_ids,
        )
    except (KeyError, ValueError):
        return None
    return replay.total_cost

def repair_plan_from_index(
    original_world: PlanningWorld,
    updated_world: PlanningWorld,
    original_plan: Plan,
    impact_index: int,
    *,
    max_expansions: int = 50_000,
) -> RepairResult:
    """Repair from a declared dependency-impact point.

    This variant is used when an external structural change does not alter an
    action already present in the original plan, but does change the set of
    alternatives available from a known trajectory point. The prefix before
    impact_index must remain exactly valid. Cortex then searches for the
    earliest old trajectory state that can be rejoined while preserving the
    unchanged suffix. If no such rejoin exists, it falls back to a full replan.
    """

    if not plan_valid(
        original_world,
        original_plan,
    ):
        raise ValueError(
            "original plan must be valid in original world"
        )
    if not (
        0 <= impact_index
        < len(original_plan.action_ids)
    ):
        raise ValueError(
            "impact_index must identify an original action"
        )

    prefix_ids = original_plan.action_ids[
        :impact_index
    ]
    try:
        prefix = execute_actions(
            updated_world,
            prefix_ids,
        )
    except (KeyError, ValueError) as exc:
        raise ValueError(
            "declared unaffected prefix is not valid"
        ) from exc

    expected_prefix_states = original_plan.states[
        : impact_index + 1
    ]
    if prefix.states != expected_prefix_states:
        raise ValueError(
            "declared unaffected prefix changed state"
        )

    prefix_state = original_plan.states[
        impact_index
    ]
    local_expanded = 0

    for rejoin_index in range(
        impact_index + 1,
        len(original_plan.states),
    ):
        if not _suffix_unchanged_and_valid(
            original_world,
            updated_world,
            original_plan,
            rejoin_index,
        ):
            continue

        segment = uniform_cost_plan(
            updated_world,
            start_state=prefix_state,
            target=original_plan.states[
                rejoin_index
            ],
            exact_target=True,
            max_expansions=max_expansions,
        )
        if segment is None:
            continue
        local_expanded += segment.expanded_states

        candidate = _combine_repair(
            updated_world,
            original_plan,
            impact_index=impact_index,
            rejoin_index=rejoin_index,
            repair_segment=segment,
        )
        if not plan_valid(
            updated_world,
            candidate,
        ):
            continue

        return RepairResult(
            mode="local-repair",
            plan=candidate,
            impact_index=impact_index,
            rejoin_index=rejoin_index,
            preserved_prefix_actions=impact_index,
            preserved_suffix_actions=(
                len(original_plan.action_ids)
                - rejoin_index
            ),
            local_expanded_states=local_expanded,
            fallback_expanded_states=0,
        )

    fallback = uniform_cost_plan(
        updated_world,
        max_expansions=max_expansions,
    )
    if fallback is None:
        return RepairResult(
            mode="infeasible",
            plan=None,
            impact_index=impact_index,
            rejoin_index=None,
            preserved_prefix_actions=0,
            preserved_suffix_actions=0,
            local_expanded_states=local_expanded,
            fallback_expanded_states=0,
        )

    return RepairResult(
        mode="full-replan",
        plan=fallback,
        impact_index=impact_index,
        rejoin_index=None,
        preserved_prefix_actions=0,
        preserved_suffix_actions=0,
        local_expanded_states=local_expanded,
        fallback_expanded_states=(
            fallback.expanded_states
        ),
    )
