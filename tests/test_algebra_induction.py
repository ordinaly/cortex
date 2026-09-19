from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))

from algebra_induction import (
    CortexAlgebraInducer,
    anonymize,
    cyclic_group,
    dihedral_group,
    partial_observation,
    score_heldout,
    subtraction_magma,
)


def make_case(algebra, seed: int):
    elements, table, identity = anonymize(algebra, seed=seed * 101 + 17)
    observed, heldout = partial_observation(
        table,
        seed=seed * 1009 + 23,
        holdout_fraction=0.40,
    )
    return elements, table, identity, observed, heldout


def test_cyclic_group_laws_are_discovered_from_partial_table():
    algebra = cyclic_group(7)
    elements, _table, identity, observed, _heldout = make_case(algebra, 2)
    model = CortexAlgebraInducer(elements, observed)
    laws = model.law_summary()

    assert laws["associativity"]["active"]
    assert laws["commutativity"]["active"]
    assert laws["identity"]["active"]
    assert model.identity == identity


def test_dihedral_group_rejects_commutativity():
    algebra = dihedral_group(4)
    elements, _table, identity, observed, _heldout = make_case(algebra, 4)
    model = CortexAlgebraInducer(elements, observed)
    laws = model.law_summary()

    assert laws["associativity"]["active"]
    assert not laws["commutativity"]["active"]
    assert laws["identity"]["active"]
    assert model.identity == identity


def test_non_associative_control_does_not_force_group_laws():
    algebra = subtraction_magma(7)
    elements, _table, _identity, observed, _heldout = make_case(algebra, 5)
    model = CortexAlgebraInducer(elements, observed)
    laws = model.law_summary()

    assert not laws["associativity"]["active"]
    assert not laws["commutativity"]["active"]
    assert not laws["identity"]["active"]


def test_cortex_derives_unseen_cyclic_products():
    algebra = cyclic_group(7)
    elements, table, _identity, observed, heldout = make_case(algebra, 7)
    model = CortexAlgebraInducer(elements, observed)
    metrics = score_heldout(model, table, heldout)

    assert metrics["coverage"] >= 0.95
    assert metrics["resolved_accuracy"] == 1.0
    assert any(pair in model.provenance for pair in heldout)


def test_cortex_derives_unseen_noncommutative_products():
    algebra = dihedral_group(4)
    elements, table, _identity, observed, heldout = make_case(algebra, 8)
    model = CortexAlgebraInducer(elements, observed)
    metrics = score_heldout(model, table, heldout)

    assert metrics["coverage"] == 1.0
    assert metrics["resolved_accuracy"] == 1.0
    assert model.conflicts == 0


def test_non_associative_control_remains_unresolved():
    algebra = subtraction_magma(7)
    elements, table, _identity, observed, heldout = make_case(algebra, 9)
    model = CortexAlgebraInducer(elements, observed)
    metrics = score_heldout(model, table, heldout)

    assert metrics["coverage"] == 0.0
    assert metrics["accuracy"] == 0.0
    assert model.conflicts == 0
