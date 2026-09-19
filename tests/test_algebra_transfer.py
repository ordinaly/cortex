from __future__ import annotations

import sys
from dataclasses import fields
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))

from algebra_form_induction import CortexFuzzyLawInducer
from algebra_form_induction_campaign import pattern_keys, random_magma
from algebra_induction import (
    anonymize,
    cyclic_group,
    partial_observation,
    subtraction_magma,
)
from algebra_transfer import (
    CortexTransferredLawInducer,
    TransferPrior,
    build_transfer_library,
    score_transfer_heldout,
)
from algebra_transfer_campaign import namespace_case


def source_model(order: int, seed: int) -> CortexFuzzyLawInducer:
    algebra = cyclic_group(order)
    raw_elements, raw_table, _identity = anonymize(
        algebra,
        seed=101 * order + seed,
    )
    elements, table = namespace_case(
        raw_elements,
        raw_table,
        f"source:test:{order}:{seed}",
    )
    observed, _heldout = partial_observation(
        table,
        seed=1009 * seed + order,
        holdout_fraction=0.40,
    )
    return CortexFuzzyLawInducer(elements, observed)


def cyclic_library():
    models = [
        source_model(order, seed)
        for order in (5, 7)
        for seed in range(2)
    ]
    return build_transfer_library(
        models,
        minimum_source_support=0.50,
        minimum_source_membership=0.98,
        maximum_forms=24,
    )


def target_case(algebra, seed: int, observed_fraction: float):
    raw_elements, raw_table, _identity = anonymize(
        algebra,
        seed=7001 + seed,
    )
    elements, table = namespace_case(
        raw_elements,
        raw_table,
        f"target:test:{len(algebra.elements)}:{seed}",
    )
    observed, heldout = partial_observation(
        table,
        seed=9001 + seed,
        holdout_fraction=1.0 - observed_fraction,
    )
    return elements, table, observed, heldout


def test_transfer_prior_contains_no_source_scope_or_labels():
    names = {field.name for field in fields(TransferPrior)}
    assert "scope" not in names
    assert "elements" not in names
    assert "table" not in names

    library = cyclic_library()
    assert library
    for prior in library:
        rendered = repr(prior)
        assert "source:test:" not in rendered


def test_source_library_contains_reusable_generated_forms():
    library = cyclic_library()
    keys = {prior.form.key for prior in library}
    target = pattern_keys()

    assert target["rebracket"] in keys
    assert target["swap"] in keys


def test_target_labels_are_disjoint_from_source_labels():
    source = source_model(5, 0)
    target_elements, _table, _observed, _heldout = target_case(
        cyclic_group(8),
        seed=0,
        observed_fraction=0.30,
    )
    assert set(source.elements).isdisjoint(target_elements)


def test_transfer_regrounds_on_new_cyclic_world_without_false_resolution():
    library = cyclic_library()
    elements, table, observed, heldout = target_case(
        cyclic_group(8),
        seed=3,
        observed_fraction=0.30,
    )
    model = CortexTransferredLawInducer(
        elements,
        observed,
        library,
    )
    metrics = score_transfer_heldout(model, table, heldout)

    assert metrics["wrong_resolved"] == 0
    assert metrics["resolved"] > 0, {
        "metrics": metrics,
        "library": [prior.form.key for prior in library],
        "structurally_supported": sorted(model.structurally_supported_keys()),
        "joint_validation": {
            "positive": model.joint_validation.positive,
            "negative": model.joint_validation.negative,
            "membership": model.joint_validation.membership,
            "conflicts": model.joint_validation_conflicts,
        },
        "selected": sorted(model.selected_keys()),
    }
    assert model.selected_keys()


def test_incompatible_subtraction_rejects_rebracketing_form():
    library = cyclic_library()
    elements, table, observed, heldout = target_case(
        subtraction_magma(7),
        seed=5,
        observed_fraction=0.30,
    )
    model = CortexTransferredLawInducer(
        elements,
        observed,
        library,
    )
    metrics = score_transfer_heldout(model, table, heldout)

    assert metrics["wrong_resolved"] == 0
    assert pattern_keys()["rebracket"] not in model.selected_keys()


def test_random_magma_does_not_receive_forced_transfer_structure():
    library = cyclic_library()
    elements, table, observed, heldout = target_case(
        random_magma(6, 12345),
        seed=7,
        observed_fraction=0.30,
    )
    model = CortexTransferredLawInducer(
        elements,
        observed,
        library,
    )
    metrics = score_transfer_heldout(model, table, heldout)

    assert metrics["wrong_resolved"] == 0
    assert metrics["resolved"] == 0


def test_transfer_module_does_not_enumerate_target_grammar():
    source = (ROOT / "research" / "algebra_transfer.py").read_text()
    assert "equation_forms(" not in source
