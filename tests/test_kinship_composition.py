from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))

from kinship_composition import (
    DOWN,
    UP,
    CortexRelationAlgebra,
    ExactSequenceMemory,
    canonical_relation,
    extract_relation_path,
    score,
    test_examples,
    train_examples,
)


def test_kinship_taxonomy_is_stable_under_longer_depth():
    assert canonical_relation((DOWN,)) == "parent"
    assert canonical_relation((DOWN,) * 8) == "ancestor"
    assert canonical_relation((UP,)) == "child"
    assert canonical_relation((UP,) * 8) == "descendant"
    assert canonical_relation((UP, DOWN)) == "sibling"
    assert canonical_relation((UP, DOWN, DOWN, DOWN)) == "aunt_uncle"
    assert canonical_relation((UP, UP, UP, DOWN)) == "niece_nephew"
    assert canonical_relation((UP, UP, UP, DOWN, DOWN, DOWN)) == "cousin"


def test_graph_path_ignores_irrelevant_facts():
    train, primitive_alias, _answer_alias = train_examples(
        seed=3,
        max_length=2,
        repetitions=1,
    )
    example = train[-1]
    path = extract_relation_path(example)
    assert len(path) == example.path_length
    assert set(path).issubset(set(primitive_alias.values()))


def test_cortex_composes_beyond_training_depth():
    train, primitive_alias, answer_alias = train_examples(seed=9)
    model = CortexRelationAlgebra()
    memory = ExactSequenceMemory()
    for example in train:
        model.observe_solved(example)
        memory.observe_solved(example)

    test = test_examples(
        seed=9,
        primitive_alias=primitive_alias,
        answer_alias=answer_alias,
        min_length=6,
        max_length=10,
        repetitions=1,
    )
    cortex = score(model, test)
    lookup = score(memory, test)

    assert cortex["accuracy"] == 1.0
    assert cortex["coverage"] == 1.0
    assert lookup["coverage"] == 0.0


def test_unseen_entities_are_disjoint_by_construction():
    train, primitive_alias, answer_alias = train_examples(seed=11)
    test = test_examples(
        seed=11,
        primitive_alias=primitive_alias,
        answer_alias=answer_alias,
    )

    train_ids = {
        node
        for example in train
        for source, _relation, target in example.facts
        for node in (source, target)
    }
    test_ids = {
        node
        for example in test
        for source, _relation, target in example.facts
        for node in (source, target)
    }
    assert train_ids.isdisjoint(test_ids)


def test_missing_rule_becomes_unresolved_not_hallucinated():
    train, primitive_alias, answer_alias = train_examples(
        seed=13,
        omitted_sequence=(UP, DOWN, DOWN),
    )
    model = CortexRelationAlgebra()
    for example in train:
        model.observe_solved(example)

    test = test_examples(
        seed=13,
        primitive_alias=primitive_alias,
        answer_alias=answer_alias,
        min_length=6,
        max_length=8,
        repetitions=1,
        require_prefix=(UP, DOWN, DOWN),
    )
    result = score(model, test)
    assert result["coverage"] == 0.0
    assert result["accuracy"] == 0.0


def test_alias_permutation_does_not_change_logic():
    for seed in range(5):
        train, primitive_alias, answer_alias = train_examples(seed=100 + seed)
        model = CortexRelationAlgebra()
        for example in train:
            model.observe_solved(example)
        test = test_examples(
            seed=100 + seed,
            primitive_alias=primitive_alias,
            answer_alias=answer_alias,
            min_length=7,
            max_length=7,
            repetitions=1,
        )
        assert score(model, test)["accuracy"] == 1.0
