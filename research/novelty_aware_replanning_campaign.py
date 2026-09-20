
"""Campaign for Cortex novelty-aware-replanning-v1."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from random import Random

from novelty_aware_replanning import (
    AlwaysReplanBaseline,
    CortexNoveltyAwarePlanner,
    DependencyRef,
    IgnoreNoveltyBaseline,
    PlanningDependencyIndex,
)
from open_world_novelty import (
    CortexNoveltyAttributor,
)
from open_world_novelty_campaign import (
    build_case as build_novelty_case,
)
from planning_replanning import (
    Action,
    PlanningWorld,
    plan_valid,
    uniform_cost_plan,
)

PROTOCOL = "novelty-aware-replanning-v1"

FAMILIES = (
    "noise-preserve",
    "new-instance-irrelevant",
    "new-instance-local",
    "new-relation-local",
    "new-relation-irrelevant",
    "new-type-local",
    "new-regime-global",
    "unresolved-defer",
)

NOVELTY_FAMILY = {
    "noise-preserve": "noise-spike",
    "new-instance-irrelevant": "new-instance",
    "new-instance-local": "new-instance",
    "new-relation-local": "new-relation",
    "new-relation-irrelevant": "new-relation",
    "new-type-local": "new-entity-type",
    "new-regime-global": "new-regime",
    "unresolved-defer": "ambiguous-instance-type",
}

EXPECTED_ATTRIBUTION = {
    "noise-preserve": "noise",
    "new-instance-irrelevant": "new_instance",
    "new-instance-local": "new_instance",
    "new-relation-local": "new_relation",
    "new-relation-irrelevant": "new_relation",
    "new-type-local": "new_entity_type",
    "new-regime-global": "new_regime",
    "unresolved-defer": "unresolved",
}

EXPECTED_DECISION = {
    "noise-preserve": "preserve",
    "new-instance-irrelevant": "preserve",
    "new-instance-local": "local-repair",
    "new-relation-local": "local-repair",
    "new-relation-irrelevant": "preserve",
    "new-type-local": "local-repair",
    "new-regime-global": "global-replan",
    "unresolved-defer": "deferred",
}

EXPECTED_IMPACT = {
    "new-instance-local": 2,
    "new-relation-local": 3,
    "new-type-local": 1,
}

DEVELOPMENT_SEEDS = tuple(range(10))
OFFICIAL_SEEDS = tuple(range(1600, 1620))

LOCAL_FAMILIES = (
    "new-instance-local",
    "new-relation-local",
    "new-type-local",
)

PRESERVE_FAMILIES = (
    "noise-preserve",
    "new-instance-irrelevant",
    "new-relation-irrelevant",
)


@dataclass(frozen=True)
class IntegratedCase:
    family: str
    seed: int
    novelty_model: object
    evidence: object
    original_world: PlanningWorld
    updated_world: PlanningWorld
    dependencies: PlanningDependencyIndex
    action_roles: dict[str, str]


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
        for role, value in zip(
            roles,
            values,
        )
    }


def _move(
    action_id: str,
    source: str,
    target: str,
    *,
    cost: int,
) -> Action:
    return Action(
        action_id=action_id,
        preconditions=frozenset((source,)),
        add_effects=frozenset((target,)),
        delete_effects=frozenset((source,)),
        cost=cost,
    )


def build_case(
    family: str,
    seed: int,
) -> IntegratedCase:
    if family not in FAMILIES:
        raise ValueError(
            f"unknown family: {family}"
        )

    novelty = build_novelty_case(
        NOVELTY_FAMILY[family],
        seed,
    )
    rng = Random(
        seed * 40_009
        + sum(ord(ch) for ch in family)
    )

    state_roles = [
        *(f"p{index}" for index in range(7)),
        "shortcut",
        "global1",
        *(
            f"dead_{branch}_{depth}"
            for branch in range(5)
            for depth in range(1, 5)
        ),
    ]
    states = _opaque_names(
        rng,
        "P",
        list(state_roles),
    )

    action_roles = [
        *(f"main{index}" for index in range(6)),
        "shortcut",
        "global-a",
        "global-b",
        *(
            f"dead-{branch}-{depth}"
            for branch in range(5)
            for depth in range(1, 5)
        ),
    ]
    actions_by_role = _opaque_names(
        rng,
        "Q",
        list(action_roles),
    )

    actions: dict[str, Action] = {}
    for index in range(6):
        role = f"main{index}"
        action = _move(
            actions_by_role[role],
            states[f"p{index}"],
            states[f"p{index + 1}"],
            cost=3,
        )
        actions[action.action_id] = action

    for branch_index in range(5):
        previous = states["p0"]
        for depth in range(1, 5):
            role = f"dead-{branch_index}-{depth}"
            target = states[
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

    original = PlanningWorld(
        initial_state=frozenset(
            (states["p0"],)
        ),
        goal=frozenset(
            (states["p6"],)
        ),
        actions=dict(actions),
    )
    updated_actions = dict(actions)
    opportunity_indices: dict[
        DependencyRef,
        int,
    ] = {}

    if family == "new-instance-local":
        shortcut = _move(
            actions_by_role["shortcut"],
            states["p2"],
            states["p3"],
            cost=1,
        )
        updated_actions[
            shortcut.action_id
        ] = shortcut
        unknown = novelty.roles[
            "instance:unknown-1"
        ]
        opportunity_indices[
            DependencyRef(
                "instance",
                unknown,
            )
        ] = 2

    elif family == "new-relation-local":
        shortcut = _move(
            actions_by_role["shortcut"],
            states["p3"],
            states["p4"],
            cost=1,
        )
        updated_actions[
            shortcut.action_id
        ] = shortcut
        type_a = novelty.roles[
            "type:type-a"
        ]
        type_b = novelty.roles[
            "type:type-b"
        ]
        relation = novelty.roles[
            "relation:novel-relation"
        ]
        opportunity_indices[
            DependencyRef(
                "relation",
                (
                    type_a,
                    type_b,
                    relation,
                ),
            )
        ] = 3

    elif family == "new-type-local":
        shortcut = _move(
            actions_by_role["shortcut"],
            states["p1"],
            states["p2"],
            cost=1,
        )
        updated_actions[
            shortcut.action_id
        ] = shortcut
        opportunity_indices[
            DependencyRef(
                "type",
                "*",
            )
        ] = 1

    elif family == "new-regime-global":
        updated_actions.pop(
            actions_by_role["main1"]
        )
        global_a = _move(
            actions_by_role["global-a"],
            states["p0"],
            states["global1"],
            cost=4,
        )
        global_b = _move(
            actions_by_role["global-b"],
            states["global1"],
            states["p6"],
            cost=4,
        )
        updated_actions[
            global_a.action_id
        ] = global_a
        updated_actions[
            global_b.action_id
        ] = global_b

    elif family == "unresolved-defer":
        shortcut = _move(
            actions_by_role["shortcut"],
            states["p1"],
            states["p2"],
            cost=1,
        )
        updated_actions[
            shortcut.action_id
        ] = shortcut
        opportunity_indices[
            DependencyRef(
                "type",
                "*",
            )
        ] = 1

    updated = PlanningWorld(
        initial_state=original.initial_state,
        goal=original.goal,
        actions=updated_actions,
    )

    return IntegratedCase(
        family=family,
        seed=seed,
        novelty_model=novelty.model,
        evidence=novelty.evidence,
        original_world=original,
        updated_world=updated,
        dependencies=PlanningDependencyIndex(
            action_dependencies={},
            opportunity_indices=(
                opportunity_indices
            ),
        ),
        action_roles=actions_by_role,
    )


def _rate(rows, predicate) -> float:
    rows = list(rows)
    if not rows:
        return 0.0
    return sum(
        int(predicate(row))
        for row in rows
    ) / len(rows)


def run_case(
    family: str,
    seed: int,
) -> dict:
    case = build_case(
        family,
        seed,
    )
    original_plan = uniform_cost_plan(
        case.original_world
    )
    if original_plan is None:
        raise RuntimeError(
            "base planning world unexpectedly infeasible"
        )

    controller = CortexNoveltyAwarePlanner(
        case.novelty_model
    )
    result = controller.respond(
        case.evidence,
        original_world=case.original_world,
        updated_world=case.updated_world,
        original_plan=original_plan,
        dependencies=case.dependencies,
    )

    attribution = result.attribution
    full = uniform_cost_plan(
        case.updated_world
    )
    always = AlwaysReplanBaseline()
    always_decision, always_plan, always_expanded = (
        always.respond(
            attribution,
            model=case.novelty_model,
            original_world=case.original_world,
            updated_world=case.updated_world,
            original_plan=original_plan,
        )
    )
    ignore = IgnoreNoveltyBaseline()
    ignore_decision, ignore_valid = (
        ignore.respond(
            updated_world=case.updated_world,
            original_plan=original_plan,
        )
    )

    expected_attribution = (
        EXPECTED_ATTRIBUTION[family]
    )
    expected_decision = (
        EXPECTED_DECISION[family]
    )

    result_valid = (
        result.plan is not None
        and (
            plan_valid(
                case.updated_world,
                result.plan,
            )
            if result.decision
            != "deferred"
            else plan_valid(
                case.original_world,
                result.plan,
            )
        )
    )

    optimal_cost = None
    if result.decision in (
        "local-repair",
        "global-replan",
    ):
        optimal_cost = (
            result.plan is not None
            and full is not None
            and result.plan.total_cost
            == full.total_cost
        )

    local_search_saving = None
    local_expansion_ratio = None
    if family in LOCAL_FAMILIES:
        local_search_saving = (
            full is not None
            and full.expanded_states > 0
            and result.local_expanded_states
            < full.expanded_states
        )
        if (
            full is not None
            and full.expanded_states > 0
        ):
            local_expansion_ratio = (
                result.local_expanded_states
                / full.expanded_states
            )

    exact_plan_preserved = None
    if family in PRESERVE_FAMILIES:
        exact_plan_preserved = (
            result.plan is not None
            and result.plan.fingerprint()
            == original_plan.fingerprint()
        )

    always_correct = (
        always_decision
        == expected_decision
    )
    if (
        expected_decision
        == "local-repair"
        and always_decision
        == "global-replan"
    ):
        always_correct = False

    ignore_optimal = None
    if family in LOCAL_FAMILIES:
        ignore_optimal = (
            ignore_valid
            and full is not None
            and original_plan.total_cost
            == full.total_cost
        )
    elif family == "new-regime-global":
        ignore_optimal = False

    return {
        "protocol": PROTOCOL,
        "family": family,
        "seed": seed,
        "expected_attribution": (
            expected_attribution
        ),
        "expected_decision": (
            expected_decision
        ),
        "attribution": {
            "category": attribution.category,
            "resolved": attribution.resolved,
            "margin": attribution.margin,
        },
        "result": {
            "decision": result.decision,
            "impact_index": result.impact_index,
            "safe_to_execute": (
                result.safe_to_execute
            ),
            "valid": result_valid,
            "cost": (
                None
                if result.plan is None
                else result.plan.total_cost
            ),
            "preserved_prefix_actions": (
                result.preserved_prefix_actions
            ),
            "preserved_suffix_actions": (
                result.preserved_suffix_actions
            ),
            "local_expanded_states": (
                result.local_expanded_states
            ),
            "global_expanded_states": (
                result.global_expanded_states
            ),
            "total_expanded_states": (
                result.total_expanded_states
            ),
            "model_changed_components": (
                result.model_changed_components
            ),
        },
        "full_replan": {
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
        },
        "baselines": {
            "always_replan_decision": (
                always_decision
            ),
            "always_replan_expanded_states": (
                always_expanded
            ),
            "always_replan_correct": (
                always_correct
            ),
            "ignore_decision": (
                ignore_decision
            ),
            "ignore_valid": (
                ignore_valid
            ),
            "ignore_optimal": (
                ignore_optimal
            ),
        },
        "checks": {
            "attribution_correct": (
                attribution.category
                == expected_attribution
            ),
            "decision_correct": (
                result.decision
                == expected_decision
            ),
            "result_valid": result_valid,
            "optimal_cost": optimal_cost,
            "impact_correct": (
                (
                    result.impact_index
                    == EXPECTED_IMPACT[
                        family
                    ]
                )
                if family
                in EXPECTED_IMPACT
                else (
                    result.impact_index
                    in (
                        None,
                        0,
                    )
                )
            ),
            "preserve_exact": (
                exact_plan_preserved
            ),
            "no_unnecessary_search": (
                (
                    result.total_expanded_states
                    == 0
                )
                if family
                in PRESERVE_FAMILIES
                else True
            ),
            "local_search_saving": (
                local_search_saving
            ),
            "local_expansion_ratio": (
                local_expansion_ratio
            ),
            "local_prefix_preserved": (
                (
                    result.preserved_prefix_actions
                    == EXPECTED_IMPACT[
                        family
                    ]
                )
                if family
                in LOCAL_FAMILIES
                else True
            ),
            "local_suffix_preserved": (
                (
                    result.preserved_suffix_actions
                    >= 1
                )
                if family
                in LOCAL_FAMILIES
                else True
            ),
            "regime_global": (
                (
                    result.decision
                    == "global-replan"
                    and result.impact_index
                    == 0
                    and result.global_expanded_states
                    > 0
                )
                if family
                == "new-regime-global"
                else True
            ),
            "unresolved_safe": (
                (
                    result.decision
                    == "deferred"
                    and not result.safe_to_execute
                    and result.model_changed_components
                    == tuple()
                    and result.total_expanded_states
                    == 0
                )
                if family
                == "unresolved-defer"
                else True
            ),
        },
    }


def campaign(
    seeds: tuple[int, ...],
    output: Path,
) -> dict:
    rows = [
        run_case(
            family,
            seed,
        )
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
        if row["family"]
        in LOCAL_FAMILIES
    ]
    ratios = [
        row["checks"][
            "local_expansion_ratio"
        ]
        for row in local_rows
        if row["checks"][
            "local_expansion_ratio"
        ] is not None
    ]

    summary = {
        "protocol": PROTOCOL,
        "seeds": list(seeds),
        "families": list(FAMILIES),
        "rows": len(rows),
        "attribution_accuracy": _rate(
            rows,
            lambda row: row["checks"][
                "attribution_correct"
            ],
        ),
        "decision_accuracy": _rate(
            rows,
            lambda row: row["checks"][
                "decision_correct"
            ],
        ),
        "mean_local_expansion_ratio": (
            sum(ratios) / len(ratios)
            if ratios
            else None
        ),
        "always_replan_decision_accuracy": _rate(
            rows,
            lambda row: row["baselines"][
                "always_replan_correct"
            ],
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
    group = (
        parser.add_mutually_exclusive_group()
    )
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
            "novelty-aware-replanning-v1.jsonl"
        ),
    )
    args = parser.parse_args()

    seeds = (
        OFFICIAL_SEEDS
        if args.official
        else DEVELOPMENT_SEEDS
    )
    campaign(
        seeds,
        args.output,
    )


if __name__ == "__main__":
    main()
