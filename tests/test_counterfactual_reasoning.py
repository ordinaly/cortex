from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))

from counterfactual_reasoning import (
    CorrelationBaseline,
    CortexStructuralCounterfactual,
    ground_truth_reachable,
)
from counterfactual_reasoning_campaign import (
    build_world,
    populate_model,
    run_world,
)


def test_native_aligned_causal_states():
    model = CortexStructuralCounterfactual(("A", "B", "C", "D"))

    model.observe_counts(
        "A",
        "B",
        do_positive=18,
        do_negative=2,
        ctrl_positive=2,
        ctrl_negative=18,
    )
    assert model.direct_state("A", "B") == 1

    model.observe_counts(
        "A",
        "C",
        do_positive=2,
        do_negative=18,
        ctrl_positive=2,
        ctrl_negative=18,
    )
    assert model.direct_state("A", "C") == -1

    model.observe_counts(
        "A",
        "D",
        do_positive=16,
        do_negative=4,
        ctrl_positive=16,
        ctrl_negative=4,
    )
    assert model.direct_state("A", "D") == -1

    missing = CortexStructuralCounterfactual(("A", "B"))
    missing.observe_counts(
        "A",
        "B",
        do_positive=4,
        do_negative=0,
        ctrl_positive=0,
        ctrl_negative=4,
    )
    assert missing.direct_state("A", "B") == 0


def test_chain_requires_multi_hop_composition():
    world = build_world("chain", 3)
    model = populate_model(world)
    source = world.integer_to_label[0]
    target = world.integer_to_label[5]

    assert (source, target) not in world.edges
    assert ground_truth_reachable(
        source,
        target,
        world.edges,
    )
    result = model.query(source, target)
    assert result.state == 1
    assert len(result.path) == 6


def test_diamond_edge_removal_preserves_alternate_path():
    world = build_world("diamond", 4)
    model = populate_model(world)
    source = world.integer_to_label[0]
    target = world.integer_to_label[5]
    removed = (
        world.integer_to_label[0],
        world.integer_to_label[1],
    )

    before = model.fingerprint()
    result = model.query(
        source,
        target,
        remove_edges=(removed,),
    )
    after = model.fingerprint()

    assert result.state == 1
    assert removed not in set(zip(result.path, result.path[1:]))
    assert before == after


def test_bridge_removal_breaks_downstream_influence():
    world = build_world("bridge", 5)
    model = populate_model(world)
    assert world.special_bridge is not None

    source = world.integer_to_label[0]
    target = world.integer_to_label[6]
    result = model.query(
        source,
        target,
        remove_edges=(world.special_bridge,),
    )
    assert result.state == -1


def test_missing_bridge_is_unresolved_not_false_negative():
    world = build_world("bridge", 6)
    assert world.special_bridge is not None
    model = populate_model(
        world,
        unresolved_edge=world.special_bridge,
    )

    source = world.integer_to_label[0]
    target = world.integer_to_label[6]
    result = model.query(source, target)

    assert result.state == 0
    assert not result.confirmed_reachable
    assert result.possible_reachable


def test_correlation_trap_is_rejected_by_intervention_difference():
    world = build_world("fork", 7)
    model = populate_model(world)
    baseline = CorrelationBaseline(model)

    for source, target in world.traps:
        assert model.direct_state(source, target) == -1
        assert baseline.direct_state(source, target) == 1


def test_development_campaign_characterization():
    for family in (
        "chain",
        "fork",
        "collider",
        "diamond",
        "bridge",
        "disconnected",
        "random-dag",
    ):
        row = run_world(family, 2)
        assert row["direct"]["resolved_precision"] == 1.0
        assert row["direct"]["resolved_coverage"] == 1.0
        assert row["negative"]["accuracy"] == 1.0
        assert row["counterfactual"]["resolved_accuracy"] == 1.0
        assert row["counterfactual"]["restoration_rate"] == 1.0
        if row["multi_hop"]["counts"]["total"] > 0:
            assert row["multi_hop"]["resolved_accuracy"] == 1.0
            assert row["multi_hop"]["coverage"] == 1.0

    bridge = run_world("bridge", 2)
    assert bridge["missing_bridge"] is not None
    assert bridge["missing_bridge"]["unresolved_rate"] == 1.0
