from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))

from algebra_form_induction import (
    CortexFuzzyLawInducer,
    canonical_equation,
    equation_forms,
    op,
    score_heldout,
    var,
)
from algebra_form_induction_campaign import (
    left_zero,
    semilattice_min,
)
from algebra_induction import (
    anonymize,
    cyclic_group,
    dihedral_group,
    partial_observation,
    subtraction_magma,
)


def make_case(algebra, seed: int):
    elements, table, _identity = anonymize(
        algebra,
        seed=seed * 101 + 37,
    )
    observed, heldout = partial_observation(
        table,
        seed=seed * 1009 + 41,
        holdout_fraction=0.40,
    )
    return elements, table, observed, heldout


def target_keys():
    x, y, z = var("x"), var("y"), var("z")
    return {
        "rebracket": canonical_equation(
            op(op(x, y), z),
            op(x, op(y, z)),
        ),
        "swap": canonical_equation(
            op(x, y),
            op(y, x),
        ),
        "repeat": canonical_equation(
            op(x, x),
            x,
        ),
        "left_projection": canonical_equation(
            op(x, y),
            x,
        ),
    }


def test_grammar_contains_target_forms_without_named_law_library():
    keys = {form.key for form in equation_forms(2)}
    for key in target_keys().values():
        assert key in keys

    source = (ROOT / "research" / "algebra_form_induction.py").read_text()
    assert "associativ" not in source.lower()
    assert "commutativ" not in source.lower()
    assert "idempot" not in source.lower()


def test_cyclic_structure_discovers_rebracketing_and_swap_forms():
    elements, table, observed, heldout = make_case(cyclic_group(7), 3)
    model = CortexFuzzyLawInducer(elements, observed)
    active = model.active_keys()
    keys = target_keys()

    assert keys["rebracket"] in active
    assert keys["swap"] in active
    metrics = score_heldout(model, table, heldout)
    assert metrics["resolved_accuracy"] == 1.0
    assert metrics["coverage"] >= 0.90


def test_noncommutative_structure_rejects_swap_form():
    elements, table, observed, heldout = make_case(dihedral_group(4), 4)
    model = CortexFuzzyLawInducer(elements, observed)
    active = model.active_keys()
    keys = target_keys()

    assert keys["rebracket"] in active
    assert keys["swap"] not in active
    metrics = score_heldout(model, table, heldout)
    assert metrics["resolved_accuracy"] == 1.0
    assert metrics["coverage"] >= 0.75


def test_semilattice_discovers_repeat_form():
    elements, table, observed, heldout = make_case(semilattice_min(6), 5)
    model = CortexFuzzyLawInducer(elements, observed)
    active = model.active_keys()
    keys = target_keys()

    # The official semilattice gate is the repetition identity. Other
    # equivalent three-variable laws may outrank a specific syntactic
    # rebracketing representative under sparse predictive validation.
    assert keys["repeat"] in active
    metrics = score_heldout(model, table, heldout)
    assert metrics["resolved_accuracy"] == 1.0


def test_left_zero_discovers_projection_form():
    elements, table, observed, heldout = make_case(left_zero(6), 6)
    model = CortexFuzzyLawInducer(elements, observed)
    active = model.active_keys()

    assert target_keys()["left_projection"] in active
    metrics = score_heldout(model, table, heldout)
    assert metrics["coverage"] == 1.0
    assert metrics["resolved_accuracy"] == 1.0


def test_subtraction_does_not_invent_rebracketing():
    elements, table, observed, heldout = make_case(subtraction_magma(7), 7)
    model = CortexFuzzyLawInducer(elements, observed)

    assert target_keys()["rebracket"] not in model.active_keys()
    metrics = score_heldout(model, table, heldout)
    assert metrics["resolved_accuracy"] == 1.0
    # Non-associative does not mean unstructured: the grammar can discover
    # other exact identities and may legitimately recover held-out products.
    assert model.active_keys()



def test_dihedral_applicability_scope_blocks_sparse_local_overreach():
    # These seeds exposed v1.4's failure mode: identities that were valid on
    # well-observed involutions were extrapolated to unsupported rotations.
    for seed in (6, 13):
        elements, table, observed, heldout = make_case(
            dihedral_group(4),
            seed,
        )
        model = CortexFuzzyLawInducer(elements, observed)
        metrics = score_heldout(model, table, heldout)
        assert metrics["resolved_accuracy"] == 1.0
        assert metrics["coverage"] >= 0.75
