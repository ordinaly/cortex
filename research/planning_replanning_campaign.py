"""Campaign for Cortex planning-replanning-v1."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, replace
from pathlib import Path
from random import Random

from planning_replanning import (
    Action,
    Plan,
    PlanningWorld,
    action_sequence_cost,
    plan_support,
    plan_valid,
    repair_plan,
    support_is_valid,
    uniform_cost_plan,
)

PROTOCOL = "planning-replanning-v1"
FAMILIES = (
    "action-failure",
    "cost-shift",
    "transition-rule",
    "reversible-detour",
    "irrelevant-change",
    "global-reroute",
    "infeasible",
)
LOCAL_FAMILIES = (
    "action-failure",
    "cost-shift",
    "transition-rule",
    "reversible-detour",
)
DEVELOPMENT_SEEDS = tuple(range(10))
OFFICIAL_SEEDS = tuple(range(1400, 1420))


@dataclass(frozen=True)
class PlanningCase:
    family: str
    seed: int
    original: PlanningWorld
    updated: PlanningWorld
    action_roles: dict[str, str]
    atom_roles: dict[str, str]
    expected_mode: str
    expected_impact_index: int | None


def _opaque_names(
    rng: Random,
    prefix: str,
    roles: list[str],
) -> dict[str, str]:
    values = [
        f"{prefix}{index:02d}_{rng.randrange(1_000_000):06d}"
        for index in range(len(roles))
    ]
    rng.shuffle(values)
    return {
        role: value
        for role, value in zip(roles, values)
    }


def _move(
    action_id: str,
    source: str,
    target: str,
    *,
    cost: int,
    require: tuple[str, ...] = (),
    add: tuple[str, ...] = (),
    delete: tuple[str, ...] = (),
) -> Action:
    return Action(
        action_id=action_id,
        preconditions=frozenset(
            (source, *require)
        ),
        add_effects=frozenset(
            (target, *add)
        ),
        delete_effects=frozenset(
            (source, *delete)
        ),
        cost=cost,
    )


def build_case(family: str, seed: int) -> PlanningCase:
    if family not in FAMILIES:
        raise ValueError(f"unknown family: {family}")

    rng = Random(
        seed * 20_011
        + sum(ord(ch) for ch in family)
    )

    atom_roles = [
        *(f"p{index}" for index in range(7)),
        "detour1",
        "side",
        "pass",
        "marker",
        "key",
        "global1",
        "global2",
        *(
            f"dead_{branch}_{depth}"
            for branch in range(5)
            for depth in range(1, 5)
        ),
    ]
    atoms = _opaque_names(
        rng,
        "S",
        list(atom_roles),
    )

    action_roles = [
        *(f"main{index}" for index in range(6)),
        "detour-a",
        "detour-b",
        "enter-side",
        "return-pass",
        "pass-exit",
        "global-a",
        "global-b",
        "global-c",
        *(
            f"dead-{branch}-{depth}"
            for branch in range(5)
            for depth in range(1, 5)
        ),
    ]
    actions_by_role = _opaque_names(
        rng,
        "A",
        list(action_roles),
    )

    actions: dict[str, Action] = {}

    for index in range(6):
        role = f"main{index}"
        action = _move(
            actions_by_role[role],
            atoms[f"p{index}"],
            atoms[f"p{index + 1}"],
            cost=2,
        )
        actions[action.action_id] = action

    for branch_index in range(5):
        previous = atoms["p0"]
        for depth in range(1, 5):
            role = f"dead-{branch_index}-{depth}"
            target = atoms[
                f"dead_{branch_index}_{depth}"
            ]
            action = _move(
                actions_by_role[role],
                previous,
                target,
                cost=1,
            )
            actions[action.action_id] = action
            previous = target

    initial_atoms = {atoms["p0"]}

    if family == "action-failure":
        detour_a = _move(
            actions_by_role["detour-a"],
            atoms["p3"],
            atoms["detour1"],
            cost=1,
        )
        detour_b = _move(
            actions_by_role["detour-b"],
            atoms["detour1"],
            atoms["p4"],
            cost=1,
        )
        actions[detour_a.action_id] = detour_a
        actions[detour_b.action_id] = detour_b
        expected_mode = "local-repair"
        expected_impact = 3

    elif family == "cost-shift":
        detour_a = _move(
            actions_by_role["detour-a"],
            atoms["p2"],
            atoms["detour1"],
            cost=1,
        )
        detour_b = _move(
            actions_by_role["detour-b"],
            atoms["detour1"],
            atoms["p3"],
            cost=2,
        )
        actions[detour_a.action_id] = detour_a
        actions[detour_b.action_id] = detour_b
        expected_mode = "local-repair"
        expected_impact = 2

    elif family == "transition-rule":
        detour_a = _move(
            actions_by_role["detour-a"],
            atoms["p4"],
            atoms["detour1"],
            cost=1,
        )
        detour_b = _move(
            actions_by_role["detour-b"],
            atoms["detour1"],
            atoms["p5"],
            cost=1,
        )
        actions[detour_a.action_id] = detour_a
        actions[detour_b.action_id] = detour_b
        expected_mode = "local-repair"
        expected_impact = 4

    elif family == "reversible-detour":
        initial_atoms.add(atoms["marker"])
        enter = _move(
            actions_by_role["enter-side"],
            atoms["p2"],
            atoms["side"],
            cost=1,
            require=(atoms["marker"],),
            delete=(atoms["marker"],),
        )
        returning = _move(
            actions_by_role["return-pass"],
            atoms["side"],
            atoms["p2"],
            cost=1,
            add=(
                atoms["marker"],
                atoms["pass"],
            ),
        )
        exit_action = _move(
            actions_by_role["pass-exit"],
            atoms["p2"],
            atoms["p3"],
            cost=1,
            require=(
                atoms["marker"],
                atoms["pass"],
            ),
            delete=(atoms["pass"],),
        )
        for action in (
            enter,
            returning,
            exit_action,
        ):
            actions[action.action_id] = action
        expected_mode = "local-repair"
        expected_impact = 2

    elif family == "irrelevant-change":
        expected_mode = "unchanged"
        expected_impact = None

    elif family == "global-reroute":
        global_a = _move(
            actions_by_role["global-a"],
            atoms["p1"],
            atoms["global1"],
            cost=4,
        )
        global_b = _move(
            actions_by_role["global-b"],
            atoms["global1"],
            atoms["global2"],
            cost=4,
        )
        global_c = _move(
            actions_by_role["global-c"],
            atoms["global2"],
            atoms["p6"],
            cost=4,
        )
        for action in (
            global_a,
            global_b,
            global_c,
        ):
            actions[action.action_id] = action
        expected_mode = "full-replan"
        expected_impact = 2

    else:
        expected_mode = "infeasible"
        expected_impact = 3

    original = PlanningWorld(
        initial_state=frozenset(initial_atoms),
        goal=frozenset((atoms["p6"],)),
        actions=dict(actions),
    )

    updated_actions = dict(actions)

    if family == "action-failure":
        updated_actions.pop(
            actions_by_role["main3"]
        )

    elif family == "cost-shift":
        action_id = actions_by_role["main2"]
        updated_actions[action_id] = replace(
            updated_actions[action_id],
            cost=5,
        )

    elif family == "transition-rule":
        action_id = actions_by_role["main4"]
        action = updated_actions[action_id]
        updated_actions[action_id] = replace(
            action,
            preconditions=frozenset(
                (
                    *action.preconditions,
                    atoms["key"],
                )
            ),
        )

    elif family == "reversible-detour":
        updated_actions.pop(
            actions_by_role["main2"]
        )

    elif family == "irrelevant-change":
        distractor_id = actions_by_role[
            "dead-0-1"
        ]
        updated_actions[distractor_id] = (
            replace(
                updated_actions[distractor_id],
                cost=9,
            )
        )

    elif family == "global-reroute":
        updated_actions.pop(
            actions_by_role["main2"]
        )

    else:
        updated_actions.pop(
            actions_by_role["main3"]
        )

    updated = PlanningWorld(
        initial_state=original.initial_state,
        goal=original.goal,
        actions=updated_actions,
    )

    return PlanningCase(
        family=family,
        seed=seed,
        original=original,
        updated=updated,
        action_roles=actions_by_role,
        atom_roles=atoms,
        expected_mode=expected_mode,
        expected_impact_index=expected_impact,
    )


def _main_ids(case: PlanningCase) -> tuple[str, ...]:
    return tuple(
        case.action_roles[f"main{index}"]
        for index in range(6)
    )


def _contains_dead_end_action(
    case: PlanningCase,
    plan: Plan | None,
) -> bool:
    if plan is None:
        return False
    dead_ids = {
        action_id
        for role, action_id
        in case.action_roles.items()
        if role.startswith("dead-")
    }
    return any(
        action_id in dead_ids
        for action_id in plan.action_ids
    )


def _local_role_check(
    case: PlanningCase,
    plan: Plan | None,
) -> bool:
    if plan is None:
        return False
    ids = set(plan.action_ids)

    if case.family in (
        "action-failure",
        "transition-rule",
    ):
        return {
            case.action_roles["detour-a"],
            case.action_roles["detour-b"],
        }.issubset(ids)

    if case.family == "cost-shift":
        return (
            {
                case.action_roles["detour-a"],
                case.action_roles["detour-b"],
            }.issubset(ids)
            and case.action_roles["main2"]
            not in ids
        )

    if case.family == "reversible-detour":
        return {
            case.action_roles["enter-side"],
            case.action_roles["return-pass"],
            case.action_roles["pass-exit"],
        }.issubset(ids)

    return True


def run_case(family: str, seed: int) -> dict:
    case = build_case(family, seed)

    base_plan = uniform_cost_plan(
        case.original
    )
    if base_plan is None:
        raise RuntimeError(
            "base world unexpectedly infeasible"
        )

    base_support = plan_support(
        case.original,
        base_plan,
    )
    main_ids = _main_ids(case)
    base_is_main = (
        base_plan.action_ids == main_ids
    )

    result = repair_plan(
        case.original,
        case.updated,
        base_plan,
    )
    full = uniform_cost_plan(
        case.updated
    )

    repaired_valid = (
        result.plan is not None
        and plan_valid(
            case.updated,
            result.plan,
        )
    )
    repaired_support_valid = False
    repaired_support_fingerprint = None
    if result.plan is not None:
        support = plan_support(
            case.updated,
            result.plan,
        )
        repaired_support_valid = (
            support_is_valid(
                case.updated,
                result.plan,
                support,
            )
        )
        repaired_support_fingerprint = (
            support.fingerprint()
        )

    feasible_agreement = (
        (result.plan is None)
        == (full is None)
    )
    optimal_cost = (
        result.plan is None
        and full is None
    ) or (
        result.plan is not None
        and full is not None
        and result.plan.total_cost
        == full.total_cost
    )

    original_updated_cost = (
        action_sequence_cost(
            case.updated,
            base_plan.action_ids,
        )
    )

    local_search_saving = None
    if family in LOCAL_FAMILIES:
        local_search_saving = (
            full is not None
            and result.local_expanded_states
            < full.expanded_states
        )

    irrelevant_exact = None
    support_unchanged = None
    if family == "irrelevant-change":
        irrelevant_exact = (
            result.plan is not None
            and result.plan.fingerprint()
            == base_plan.fingerprint()
        )
        support_unchanged = (
            repaired_support_fingerprint
            == base_support.fingerprint()
        )

    global_roles_present = None
    if family == "global-reroute":
        global_roles_present = (
            result.plan is not None
            and {
                case.action_roles["global-a"],
                case.action_roles["global-b"],
                case.action_roles["global-c"],
            }.issubset(
                set(result.plan.action_ids)
            )
        )

    reversible_state_restored = None
    if family == "reversible-detour":
        marker = case.atom_roles["marker"]
        pass_atom = case.atom_roles["pass"]
        reversible_state_restored = (
            result.plan is not None
            and marker in result.plan.states[-1]
            and pass_atom
            not in result.plan.states[-1]
        )

    cost_improvement = None
    if family == "cost-shift":
        cost_improvement = (
            result.plan is not None
            and original_updated_cost is not None
            and result.plan.total_cost
            < original_updated_cost
        )

    return {
        "protocol": PROTOCOL,
        "family": family,
        "seed": seed,
        "expected_mode": case.expected_mode,
        "expected_impact_index": (
            case.expected_impact_index
        ),
        "base": {
            "valid": plan_valid(
                case.original,
                base_plan,
            ),
            "is_main_chain": base_is_main,
            "cost": base_plan.total_cost,
            "steps": len(
                base_plan.action_ids
            ),
            "expanded_states": (
                base_plan.expanded_states
            ),
            "support_valid": support_is_valid(
                case.original,
                base_plan,
                base_support,
            ),
            "dead_end_action_used": (
                _contains_dead_end_action(
                    case,
                    base_plan,
                )
            ),
        },
        "repair": {
            "mode": result.mode,
            "impact_index": result.impact_index,
            "rejoin_index": result.rejoin_index,
            "preserved_prefix_actions": (
                result.preserved_prefix_actions
            ),
            "preserved_suffix_actions": (
                result.preserved_suffix_actions
            ),
            "local_expanded_states": (
                result.local_expanded_states
            ),
            "fallback_expanded_states": (
                result.fallback_expanded_states
            ),
            "valid": repaired_valid,
            "support_valid": (
                repaired_support_valid
            ),
            "cost": (
                None
                if result.plan is None
                else result.plan.total_cost
            ),
            "steps": (
                None
                if result.plan is None
                else len(result.plan.action_ids)
            ),
            "dead_end_action_used": (
                _contains_dead_end_action(
                    case,
                    result.plan,
                )
            ),
        },
        "full_replan": {
            "feasible": full is not None,
            "cost": (
                None
                if full is None
                else full.total_cost
            ),
            "expanded_states": (
                None
                if full is None
                else full.expanded_states
            ),
            "dead_end_action_used": (
                _contains_dead_end_action(
                    case,
                    full,
                )
            ),
        },
        "checks": {
            "mode_correct": (
                result.mode
                == case.expected_mode
            ),
            "impact_correct": (
                result.impact_index
                == case.expected_impact_index
            ),
            "feasible_agreement": (
                feasible_agreement
            ),
            "optimal_cost": optimal_cost,
            "local_role_check": (
                _local_role_check(
                    case,
                    result.plan,
                )
            ),
            "local_search_saving": (
                local_search_saving
            ),
            "irrelevant_exact": (
                irrelevant_exact
            ),
            "support_unchanged": (
                support_unchanged
            ),
            "global_roles_present": (
                global_roles_present
            ),
            "reversible_state_restored": (
                reversible_state_restored
            ),
            "cost_improvement": (
                cost_improvement
            ),
        },
        "original_updated_cost": (
            original_updated_cost
        ),
    }


def campaign(
    seeds: tuple[int, ...],
    output: Path,
) -> dict:
    rows = [
        run_case(family, seed)
        for seed in seeds
        for family in FAMILIES
    ]
    output.write_text(
        "".join(
            json.dumps(
                row,
                sort_keys=True,
            )
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )

    local_rows = [
        row
        for row in rows
        if row["family"] in LOCAL_FAMILIES
    ]
    summary = {
        "protocol": PROTOCOL,
        "seeds": list(seeds),
        "families": list(FAMILIES),
        "rows": len(rows),
        "local_cases": len(local_rows),
        "mean_local_expansions": (
            sum(
                row["repair"][
                    "local_expanded_states"
                ]
                for row in local_rows
            )
            / max(1, len(local_rows))
        ),
        "mean_full_expansions_local_cases": (
            sum(
                row["full_replan"][
                    "expanded_states"
                ]
                for row in local_rows
            )
            / max(1, len(local_rows))
        ),
    }
    print(
        json.dumps(
            summary,
            sort_keys=True,
        )
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--development",
        action="store_true",
    )
    group.add_argument(
        "--official",
        action="store_true",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "planning-replanning-v1.jsonl"
        ),
    )
    args = parser.parse_args()

    seeds = (
        OFFICIAL_SEEDS
        if args.official
        else DEVELOPMENT_SEEDS
    )
    campaign(seeds, args.output)


if __name__ == "__main__":
    main()
