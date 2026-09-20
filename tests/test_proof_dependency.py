from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))

from proof_dependency import (
    Literal,
    ProofDependencyModel,
    Rule,
    SingleProofBaseline,
)
from proof_dependency_campaign import (
    FAMILIES,
    build_world,
    run_case,
)


def test_cycle_is_rejected():
    facts = {
        "f": Literal("a"),
    }
    rules = (
        Rule(
            "r1",
            (Literal("a"),),
            Literal("b"),
        ),
        Rule(
            "r2",
            (Literal("b"),),
            Literal("a"),
        ),
    )
    with pytest.raises(ValueError, match="acyclic"):
        ProofDependencyModel(facts, rules)


def test_redundant_proof_survives_selected_branch_retraction():
    world = build_world("redundant-branches", 3)
    model = ProofDependencyModel(
        world.facts,
        world.rules,
    )
    initial = model.explain(world.target_atom)

    assert initial.state == 1
    assert len(initial.positive_proof_ids) == 2
    assert model.validate_active_dag()

    selected_id = initial.selected_positive_id
    assert selected_id is not None
    selected = model.proof(selected_id)
    alternate = next(
        proof_id
        for proof_id in initial.positive_proof_ids
        if proof_id != selected_id
        and model.independent(
            selected_id,
            proof_id,
        )
    )
    baseline = SingleProofBaseline(
        model,
        world.target_atom,
    )

    removal = selected.fact_dependencies[0]
    model.retract_fact(removal)
    after = model.explain(world.target_atom)

    assert after.state == 1
    assert alternate in after.positive_proof_ids
    assert selected_id not in after.positive_proof_ids
    assert baseline.state(model) == 0


def test_shortest_proof_can_be_replaced_by_longer_valid_proof():
    world = build_world("short-vs-long", 4)
    model = ProofDependencyModel(
        world.facts,
        world.rules,
    )
    initial = model.explain(world.target_atom)

    selected_id = initial.selected_positive_id
    assert selected_id is not None
    selected_depth = model.proof(selected_id).depth
    assert selected_depth == 1
    assert any(
        model.proof(proof_id).depth > selected_depth
        for proof_id in initial.positive_proof_ids
    )

    removal = model.proof(
        selected_id
    ).fact_dependencies[0]
    model.retract_fact(removal)
    after = model.explain(world.target_atom)

    assert after.state == 1
    assert after.selected_positive_id is not None
    assert (
        model.proof(
            after.selected_positive_id
        ).depth
        > selected_depth
    )


def test_shared_dependency_invalidates_all_apparent_branches():
    world = build_world("shared-root", 5)
    model = ProofDependencyModel(
        world.facts,
        world.rules,
    )
    initial = model.explain(world.target_atom)

    assert initial.state == 1
    assert len(initial.positive_proof_ids) == 2

    shared = world.fact_roles["shared"]
    for proof_id in initial.positive_proof_ids:
        assert (
            shared
            in model.proof(
                proof_id
            ).fact_dependencies
        )

    model.retract_fact(shared)
    after = model.explain(world.target_atom)
    assert after.state == 0
    assert after.reason == "unproven"
    assert not after.positive_proof_ids


def test_conflicting_proofs_are_unresolved_until_one_side_is_removed():
    world = build_world("contradictory", 6)
    model = ProofDependencyModel(
        world.facts,
        world.rules,
    )

    initial = model.explain(world.target_atom)
    assert initial.state == 0
    assert initial.reason == "conflict"
    assert len(initial.positive_proof_ids) == 1
    assert len(initial.negative_proof_ids) == 1

    model.retract_fact(
        world.fact_roles["negative-c"]
    )
    supported = model.explain(world.target_atom)
    assert supported.state == 1
    assert supported.reason == "supported"

    model.restore_fact(
        world.fact_roles["negative-c"]
    )
    model.retract_fact(
        world.fact_roles["positive-a"]
    )
    refuted = model.explain(world.target_atom)
    assert refuted.state == -1
    assert refuted.reason == "refuted"


def test_irrelevant_fact_retraction_does_not_change_target_explanation():
    world = build_world("irrelevant-premise", 7)
    model = ProofDependencyModel(
        world.facts,
        world.rules,
    )

    before = model.target_fingerprint(
        world.target_atom
    )
    model.retract_fact(
        world.fact_roles["irrelevant"]
    )
    after = model.target_fingerprint(
        world.target_atom
    )

    assert before == after
    assert model.explain(world.target_atom).state == 1


def test_development_families_meet_structural_expectations():
    for family in FAMILIES:
        row = run_case(family, 2)
        assert row["dag_valid"]
        assert row["provenance_valid"]
        assert row["selected_is_shortest"]
        assert row["restoration_exact"]
        assert row["restored_dag_valid"]

    redundant = run_case(
        "redundant-branches",
        2,
    )
    assert redundant["alternate_survives"]
    assert redundant["after_removal_state"] == 1
    assert redundant["single_proof_baseline_state"] == 0

    shortest = run_case(
        "short-vs-long",
        2,
    )
    assert shortest["deeper_takeover"]
    assert shortest["after_removal_state"] == 1

    conflict = run_case(
        "contradictory",
        2,
    )
    assert conflict["initial_reason"] == "conflict"
    assert conflict["without_negative_state"] == 1
    assert conflict["without_positive_state"] == -1
