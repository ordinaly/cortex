from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))

from belief_revision import (
    apply_ambiguous_contradiction,
    apply_causal_batch,
    apply_strong_contradiction,
    build_model,
    build_world,
    target_cell,
    unrelated_fingerprint,
)
from belief_revision_campaign import run_case


def test_single_contradiction_does_not_immediately_retract():
    world = build_world(3)
    model = build_model(world, prior_batches=1)

    assert target_cell(model, world).state == 1
    assert model.query(*world.downstream_query).state == 1

    apply_strong_contradiction(model, world)

    assert target_cell(model, world).state == 1
    assert model.query(*world.downstream_query).state == 1


def test_strong_contradiction_moves_through_unresolved_before_null():
    world = build_world(4)
    model = build_model(world, prior_batches=1)

    states = []
    for _ in range(10):
        apply_strong_contradiction(model, world)
        states.append(target_cell(model, world).state)
        if states[-1] == -1:
            break

    assert states[0] == 1
    assert 0 in states
    assert states[-1] == -1
    assert states.index(0) < states.index(-1)


def test_downstream_reasoning_tracks_revision_state():
    world = build_world(5)
    model = build_model(world, prior_batches=1)

    assert model.query(*world.downstream_query).state == 1

    unresolved_seen = False
    for _ in range(20):
        apply_strong_contradiction(model, world)
        state = target_cell(model, world).state
        query = model.query(*world.downstream_query)

        if state == 0:
            unresolved_seen = True
            assert query.state == 0

        if state == -1:
            assert unresolved_seen
            assert query.state == -1
            break
    else:
        raise AssertionError("target belief never retracted")


def test_ambiguous_contradiction_remains_unresolved():
    world = build_world(6)
    for prior in (1, 2, 3):
        model = build_model(world, prior_batches=prior)
        for _ in range(20):
            apply_ambiguous_contradiction(model, world)

        assert target_cell(model, world).state == 0
        assert model.query(*world.downstream_query).state == 0


def test_recovery_restores_active_belief_and_downstream_reasoning():
    world = build_world(7)
    model = build_model(world, prior_batches=1)

    for _ in range(40):
        apply_strong_contradiction(model, world)
        if target_cell(model, world).state == -1:
            break
    else:
        raise AssertionError("target belief never retracted")

    unresolved_seen = False
    for _ in range(20):
        apply_causal_batch(model, world)
        state = target_cell(model, world).state
        if state == 0:
            unresolved_seen = True
        if state == 1:
            assert unresolved_seen
            assert model.query(*world.downstream_query).state == 1
            break
    else:
        raise AssertionError("target belief never recovered")


def test_unrelated_structure_is_unchanged():
    world = build_world(8)
    model = build_model(world, prior_batches=2)
    before = unrelated_fingerprint(model, world)

    for _ in range(14):
        apply_strong_contradiction(model, world)

    assert unrelated_fingerprint(model, world) == before

    for _ in range(5):
        apply_causal_batch(model, world)

    assert unrelated_fingerprint(model, world) == before


def test_development_case_orders_latency_by_prior_strength():
    rows = [
        run_case(2, prior)
        for prior in (1, 2, 3)
    ]

    retract = [
        row["strong_contradiction"]["first_null_batch"]
        for row in rows
    ]
    recover = [
        row["recovery"]["active_batch"]
        for row in rows
    ]

    assert retract[0] < retract[1] < retract[2]
    assert recover[0] < recover[1] < recover[2]

    for row in rows:
        assert row["initial"]["state"] == 1
        assert row["single_batch"]["retained_active"]
        assert row["transition_order_valid"]
        assert row["ambiguous"]["unresolved"]
        assert not row["ambiguous"]["false_retraction"]
        assert row["recovery"]["success"]
        assert row["downstream_alignment"]
        assert row["unrelated_structure_preserved"]
        assert row["strong_contradiction"]["margin_monotone"]
        assert row["recovery"]["margin_monotone"]
