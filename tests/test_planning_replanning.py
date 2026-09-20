from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))

from planning_replanning import (
    Action,
    PlanningWorld,
    plan_support,
    plan_valid,
    repair_plan,
    support_is_valid,
    uniform_cost_plan,
)
from planning_replanning_campaign import (
    FAMILIES,
    LOCAL_FAMILIES,
    build_case,
    run_case,
)


def test_uniform_cost_prefers_lower_cost_over_shorter_plan():
    world = PlanningWorld(
        initial_state=frozenset(("s",)),
        goal=frozenset(("g",)),
        actions={
            "direct": Action(
                "direct",
                frozenset(("s",)),
                frozenset(("g",)),
                frozenset(("s",)),
                5,
            ),
            "step1": Action(
                "step1",
                frozenset(("s",)),
                frozenset(("m",)),
                frozenset(("s",)),
                1,
            ),
            "step2": Action(
                "step2",
                frozenset(("m",)),
                frozenset(("g",)),
                frozenset(("m",)),
                1,
            ),
        },
    )

    plan = uniform_cost_plan(world)
    assert plan is not None
    assert plan.action_ids == (
        "step1",
        "step2",
    )
    assert plan.total_cost == 2
    assert plan_valid(world, plan)


def test_temporal_support_tracks_latest_producer():
    world = PlanningWorld(
        initial_state=frozenset(
            ("at-a", "token")
        ),
        goal=frozenset(("at-c", "token")),
        actions={
            "leave": Action(
                "leave",
                frozenset(("at-a", "token")),
                frozenset(("at-b",)),
                frozenset(("at-a", "token")),
                1,
            ),
            "restore": Action(
                "restore",
                frozenset(("at-b",)),
                frozenset(("at-c", "token")),
                frozenset(("at-b",)),
                1,
            ),
        },
    )
    plan = uniform_cost_plan(world)
    assert plan is not None

    support = plan_support(world, plan)
    assert support_is_valid(
        world,
        plan,
        support,
    )

    token_goal = next(
        ref
        for ref in support.goals
        if ref.atom == "token"
    )
    assert token_goal.source_kind == "action"
    assert token_goal.source_id == "restore"
    assert token_goal.source_step == 1


def test_action_failure_repairs_locally_and_preserves_suffix():
    case = build_case("action-failure", 3)
    base = uniform_cost_plan(case.original)
    assert base is not None

    repaired = repair_plan(
        case.original,
        case.updated,
        base,
    )
    full = uniform_cost_plan(case.updated)

    assert repaired.mode == "local-repair"
    assert repaired.impact_index == 3
    assert repaired.rejoin_index == 4
    assert repaired.preserved_prefix_actions == 3
    assert repaired.preserved_suffix_actions == 2
    assert repaired.plan is not None
    assert full is not None
    assert repaired.plan.total_cost == full.total_cost
    assert (
        repaired.local_expanded_states
        < full.expanded_states
    )


def test_cost_change_causes_cheaper_local_reroute():
    case = build_case("cost-shift", 4)
    base = uniform_cost_plan(case.original)
    assert base is not None

    repaired = repair_plan(
        case.original,
        case.updated,
        base,
    )
    full = uniform_cost_plan(case.updated)

    assert repaired.mode == "local-repair"
    assert repaired.plan is not None
    assert full is not None
    assert repaired.plan.total_cost == full.total_cost
    assert (
        case.action_roles["main2"]
        not in repaired.plan.action_ids
    )
    assert (
        case.action_roles["detour-a"]
        in repaired.plan.action_ids
    )
    assert (
        case.action_roles["detour-b"]
        in repaired.plan.action_ids
    )


def test_reversible_side_trip_restores_state_before_rejoin():
    case = build_case("reversible-detour", 5)
    base = uniform_cost_plan(case.original)
    assert base is not None

    repaired = repair_plan(
        case.original,
        case.updated,
        base,
    )
    assert repaired.mode == "local-repair"
    assert repaired.plan is not None

    ids = set(repaired.plan.action_ids)
    assert case.action_roles["enter-side"] in ids
    assert case.action_roles["return-pass"] in ids
    assert case.action_roles["pass-exit"] in ids

    marker = case.atom_roles["marker"]
    pass_atom = case.atom_roles["pass"]
    assert marker in repaired.plan.states[-1]
    assert pass_atom not in repaired.plan.states[-1]


def test_irrelevant_change_keeps_plan_and_support_exact():
    case = build_case("irrelevant-change", 6)
    base = uniform_cost_plan(case.original)
    assert base is not None
    base_support = plan_support(
        case.original,
        base,
    )

    repaired = repair_plan(
        case.original,
        case.updated,
        base,
    )
    assert repaired.mode == "unchanged"
    assert repaired.plan is not None
    assert (
        repaired.plan.fingerprint()
        == base.fingerprint()
    )

    repaired_support = plan_support(
        case.updated,
        repaired.plan,
    )
    assert (
        repaired_support.fingerprint()
        == base_support.fingerprint()
    )


def test_global_reroute_falls_back_when_old_trajectory_cannot_be_rejoined():
    case = build_case("global-reroute", 7)
    base = uniform_cost_plan(case.original)
    assert base is not None

    repaired = repair_plan(
        case.original,
        case.updated,
        base,
    )
    full = uniform_cost_plan(case.updated)

    assert repaired.mode == "full-replan"
    assert repaired.plan is not None
    assert full is not None
    assert repaired.plan.total_cost == full.total_cost
    ids = set(repaired.plan.action_ids)
    assert {
        case.action_roles["global-a"],
        case.action_roles["global-b"],
        case.action_roles["global-c"],
    }.issubset(ids)


def test_true_infeasibility_is_reported():
    case = build_case("infeasible", 8)
    base = uniform_cost_plan(case.original)
    assert base is not None

    repaired = repair_plan(
        case.original,
        case.updated,
        base,
    )
    full = uniform_cost_plan(case.updated)

    assert repaired.mode == "infeasible"
    assert repaired.plan is None
    assert full is None


def test_development_family_characterization():
    for family in FAMILIES:
        row = run_case(family, 2)
        assert row["base"]["valid"]
        assert row["base"]["support_valid"]
        assert row["base"]["is_main_chain"]
        assert not row["base"][
            "dead_end_action_used"
        ]
        assert row["checks"]["mode_correct"]
        assert row["checks"]["impact_correct"]
        assert row["checks"][
            "feasible_agreement"
        ]
        assert row["checks"]["optimal_cost"]

        if family != "infeasible":
            assert row["repair"]["valid"]
            assert row["repair"]["support_valid"]
            assert not row["repair"][
                "dead_end_action_used"
            ]

        if family in LOCAL_FAMILIES:
            assert (
                row["repair"]["mode"]
                == "local-repair"
            )
            assert (
                row["repair"][
                    "preserved_prefix_actions"
                ]
                == row[
                    "expected_impact_index"
                ]
            )
            assert (
                row["repair"][
                    "preserved_suffix_actions"
                ]
                >= 1
            )
            assert row["checks"][
                "local_search_saving"
            ]
            assert row["checks"][
                "local_role_check"
            ]
