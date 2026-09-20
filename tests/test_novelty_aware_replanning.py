
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))

from novelty_aware_replanning import (
    CortexNoveltyAwarePlanner,
)
from novelty_aware_replanning_campaign import (
    FAMILIES,
    LOCAL_FAMILIES,
    build_case,
    run_case,
)
from planning_replanning import (
    plan_valid,
    repair_plan_from_index,
    uniform_cost_plan,
)


def test_dependency_indexed_repair_can_use_new_cheaper_action():
    case = build_case(
        "new-instance-local",
        2,
    )
    original = uniform_cost_plan(
        case.original_world
    )
    assert original is not None

    repaired = repair_plan_from_index(
        case.original_world,
        case.updated_world,
        original,
        2,
    )
    full = uniform_cost_plan(
        case.updated_world
    )

    assert repaired.mode == "local-repair"
    assert repaired.plan is not None
    assert full is not None
    assert repaired.impact_index == 2
    assert repaired.rejoin_index == 3
    assert repaired.plan.total_cost == full.total_cost
    assert repaired.plan.total_cost < original.total_cost
    assert repaired.local_expanded_states < full.expanded_states


def test_noise_preserves_exact_plan_without_search():
    case = build_case(
        "noise-preserve",
        3,
    )
    original = uniform_cost_plan(
        case.original_world
    )
    assert original is not None

    result = CortexNoveltyAwarePlanner(
        case.novelty_model
    ).respond(
        case.evidence,
        original_world=case.original_world,
        updated_world=case.updated_world,
        original_plan=original,
        dependencies=case.dependencies,
    )

    assert result.attribution.category == "noise"
    assert result.decision == "preserve"
    assert result.plan is not None
    assert (
        result.plan.fingerprint()
        == original.fingerprint()
    )
    assert result.total_expanded_states == 0
    assert result.model_changed_components == tuple()
    assert result.safe_to_execute


def test_resolved_but_irrelevant_structural_novelty_preserves_plan():
    for family in (
        "new-instance-irrelevant",
        "new-relation-irrelevant",
    ):
        case = build_case(
            family,
            4,
        )
        original = uniform_cost_plan(
            case.original_world
        )
        assert original is not None

        result = CortexNoveltyAwarePlanner(
            case.novelty_model
        ).respond(
            case.evidence,
            original_world=case.original_world,
            updated_world=case.updated_world,
            original_plan=original,
            dependencies=case.dependencies,
        )

        assert result.attribution.resolved
        assert result.attribution.category != "noise"
        assert result.decision == "preserve"
        assert result.plan is not None
        assert (
            result.plan.fingerprint()
            == original.fingerprint()
        )
        assert result.total_expanded_states == 0
        assert result.model_changed_components


def test_local_novelty_repairs_only_from_dependency_point():
    for family in LOCAL_FAMILIES:
        case = build_case(
            family,
            5,
        )
        original = uniform_cost_plan(
            case.original_world
        )
        assert original is not None

        result = CortexNoveltyAwarePlanner(
            case.novelty_model
        ).respond(
            case.evidence,
            original_world=case.original_world,
            updated_world=case.updated_world,
            original_plan=original,
            dependencies=case.dependencies,
        )
        full = uniform_cost_plan(
            case.updated_world
        )

        assert result.decision == "local-repair"
        assert result.plan is not None
        assert full is not None
        assert plan_valid(
            case.updated_world,
            result.plan,
        )
        assert result.plan.total_cost == full.total_cost
        assert (
            result.local_expanded_states
            < full.expanded_states
        )
        assert result.preserved_suffix_actions >= 1


def test_regime_change_forces_global_replan():
    case = build_case(
        "new-regime-global",
        6,
    )
    original = uniform_cost_plan(
        case.original_world
    )
    assert original is not None

    result = CortexNoveltyAwarePlanner(
        case.novelty_model
    ).respond(
        case.evidence,
        original_world=case.original_world,
        updated_world=case.updated_world,
        original_plan=original,
        dependencies=case.dependencies,
    )
    full = uniform_cost_plan(
        case.updated_world
    )

    assert result.attribution.category == "new_regime"
    assert result.decision == "global-replan"
    assert result.impact_index == 0
    assert result.plan is not None
    assert full is not None
    assert result.plan.total_cost == full.total_cost
    assert result.global_expanded_states > 0


def test_unresolved_novelty_defers_without_model_or_plan_rewrite():
    case = build_case(
        "unresolved-defer",
        7,
    )
    original = uniform_cost_plan(
        case.original_world
    )
    assert original is not None

    result = CortexNoveltyAwarePlanner(
        case.novelty_model
    ).respond(
        case.evidence,
        original_world=case.original_world,
        updated_world=case.updated_world,
        original_plan=original,
        dependencies=case.dependencies,
    )

    assert result.attribution.category == "unresolved"
    assert result.decision == "deferred"
    assert not result.safe_to_execute
    assert result.model_changed_components == tuple()
    assert result.total_expanded_states == 0
    assert result.plan is original


def test_development_family_characterization():
    for family in FAMILIES:
        row = run_case(
            family,
            1,
        )
        assert row["checks"][
            "attribution_correct"
        ]
        assert row["checks"][
            "decision_correct"
        ]
        assert row["checks"][
            "result_valid"
        ]

        if family in LOCAL_FAMILIES:
            assert row["checks"][
                "optimal_cost"
            ]
            assert row["checks"][
                "local_search_saving"
            ]
            assert row["checks"][
                "local_prefix_preserved"
            ]
            assert row["checks"][
                "local_suffix_preserved"
            ]
