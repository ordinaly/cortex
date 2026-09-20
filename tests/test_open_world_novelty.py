from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))

from open_world_novelty import (
    CortexNoveltyAttributor,
    apply_attribution,
    changed_components,
)
from open_world_novelty_campaign import (
    EXPECTED_CATEGORY,
    EXPECTED_COMPONENTS,
    FAMILIES,
    build_case,
    run_case,
)


def test_all_frozen_families_have_expected_attribution():
    for family in FAMILIES:
        case = build_case(family, 2)
        read = CortexNoveltyAttributor(
            case.model,
        ).evaluate(case.evidence)

        assert (
            read.category
            == EXPECTED_CATEGORY[family]
        )
        assert (
            read.resolved
            == (
                EXPECTED_CATEGORY[family]
                != "unresolved"
            )
        )


def test_new_instance_binds_without_creating_type():
    case = build_case("new-instance", 3)
    read = CortexNoveltyAttributor(
        case.model,
    ).evaluate(case.evidence)

    assert read.category == "new_instance"
    updated = apply_attribution(
        case.model,
        read,
    )
    assert changed_components(
        case.model,
        updated,
    ) == ("instances",)

    post = CortexNoveltyAttributor(
        updated,
    ).evaluate(case.evidence)
    assert post.category == "noise"
    assert post.resolved


def test_new_entity_type_changes_only_type_and_binding():
    case = build_case("new-entity-type", 4)
    read = CortexNoveltyAttributor(
        case.model,
    ).evaluate(case.evidence)

    assert (
        read.category
        == "new_entity_type"
    )
    updated = apply_attribution(
        case.model,
        read,
    )
    assert changed_components(
        case.model,
        updated,
    ) == (
        "instances",
        "types",
    )

    post = CortexNoveltyAttributor(
        updated,
    ).evaluate(case.evidence)
    assert post.category == "noise"


def test_new_relation_does_not_rewrite_entity_ontology():
    case = build_case("new-relation", 5)
    read = CortexNoveltyAttributor(
        case.model,
    ).evaluate(case.evidence)

    assert read.category == "new_relation"
    updated = apply_attribution(
        case.model,
        read,
    )
    assert changed_components(
        case.model,
        updated,
    ) == ("relations",)


def test_new_regime_preserves_type_and_instance_identity():
    case = build_case("new-regime", 6)
    read = CortexNoveltyAttributor(
        case.model,
    ).evaluate(case.evidence)

    assert read.category == "new_regime"
    updated = apply_attribution(
        case.model,
        read,
    )
    assert changed_components(
        case.model,
        updated,
    ) == (
        "active_regime",
        "regimes",
    )


def test_noise_and_transient_evidence_do_not_mutate_model():
    for family in (
        "noise-spike",
        "incoherent-noise",
        "transient-excursion",
    ):
        case = build_case(family, 7)
        before = case.model.fingerprint()
        read = CortexNoveltyAttributor(
            case.model,
        ).evaluate(case.evidence)
        updated = apply_attribution(
            case.model,
            read,
        )

        assert read.category == "noise"
        assert updated.fingerprint() == before


def test_ambiguous_instance_type_remains_unresolved_without_edit():
    case = build_case(
        "ambiguous-instance-type",
        8,
    )
    attributor = CortexNoveltyAttributor(
        case.model,
    )
    read = attributor.evaluate(
        case.evidence
    )

    assert read.category == "unresolved"
    assert not read.resolved
    assert read.proposed_edit is None
    assert read.margin < 0.5
    assert {
        category
        for category, _score
        in sorted(
            read.scores,
            key=lambda item: item[1],
        )[:2]
    } == {
        "new_instance",
        "new_entity_type",
    }
    assert apply_attribution(
        case.model,
        read,
    ).fingerprint() == case.model.fingerprint()


def test_transient_recovery_returns_to_no_edit_explanation():
    case = build_case(
        "transient-excursion",
        9,
    )
    assert case.recovery_evidence is not None
    attributor = CortexNoveltyAttributor(
        case.model,
    )

    excursion = attributor.evaluate(
        case.evidence
    )
    recovery = attributor.evaluate(
        case.recovery_evidence
    )

    assert excursion.category == "noise"
    assert recovery.category == "noise"
    assert recovery.resolved
    assert apply_attribution(
        case.model,
        recovery,
    ).fingerprint() == case.model.fingerprint()


def test_campaign_rows_preserve_exact_edit_locality():
    for family in FAMILIES:
        row = run_case(family, 1)
        assert row["checks"][
            "category_correct"
        ]
        assert row["checks"][
            "resolved_correct"
        ]
        assert row["model"][
            "pure_evaluation"
        ]
        assert row["model"][
            "edit_locality"
        ]
        assert tuple(
            row["model"][
                "changed_components"
            ]
        ) == EXPECTED_COMPONENTS[family]
