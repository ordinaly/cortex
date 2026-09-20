"""Campaign for Cortex open-world-novelty-attribution-v1."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from random import Random

from open_world_novelty import (
    CortexNoveltyAttributor,
    EvidenceWindow,
    GlobalResetBaseline,
    NearestKnownTypeBaseline,
    WorldModel,
    apply_attribution,
    changed_components,
)

PROTOCOL = "open-world-novelty-attribution-v1"
FAMILIES = (
    "noise-spike",
    "incoherent-noise",
    "new-instance",
    "new-entity-type",
    "new-relation",
    "new-regime",
    "transient-excursion",
    "ambiguous-instance-type",
)
DEVELOPMENT_SEEDS = tuple(range(10))
OFFICIAL_SEEDS = tuple(range(1500, 1520))

EXPECTED_CATEGORY = {
    "noise-spike": "noise",
    "incoherent-noise": "noise",
    "new-instance": "new_instance",
    "new-entity-type": "new_entity_type",
    "new-relation": "new_relation",
    "new-regime": "new_regime",
    "transient-excursion": "noise",
    "ambiguous-instance-type": "unresolved",
}

EXPECTED_COMPONENTS = {
    "noise-spike": tuple(),
    "incoherent-noise": tuple(),
    "new-instance": ("instances",),
    "new-entity-type": ("instances", "types"),
    "new-relation": ("relations",),
    "new-regime": (
        "active_regime",
        "regimes",
    ),
    "transient-excursion": tuple(),
    "ambiguous-instance-type": tuple(),
}


@dataclass(frozen=True)
class NoveltyCase:
    family: str
    seed: int
    model: WorldModel
    evidence: EvidenceWindow
    recovery_evidence: EvidenceWindow | None
    roles: dict[str, str]


def _opaque_names(
    rng: Random,
    prefix: str,
    roles: list[str],
) -> dict[str, str]:
    values = [
        f"{prefix}{index:02d}_{rng.randrange(1_000_000):06d}"
        for index in range(len(roles))
    ]
    rng.shuffle(values)
    return {
        role: value
        for role, value in zip(
            roles,
            values,
        )
    }


def build_case(
    family: str,
    seed: int,
) -> NoveltyCase:
    if family not in FAMILIES:
        raise ValueError(
            f"unknown family: {family}"
        )

    rng = Random(
        seed * 30_013
        + sum(ord(ch) for ch in family)
    )

    type_ids = _opaque_names(
        rng,
        "T",
        ["type-a", "type-b"],
    )
    instance_ids = _opaque_names(
        rng,
        "E",
        [
            "a1",
            "a2",
            "a3",
            "b1",
            "b2",
            "b3",
            "unknown-1",
            "unknown-2",
        ],
    )
    relation_ids = _opaque_names(
        rng,
        "R",
        ["known-relation", "novel-relation"],
    )

    type_a = type_ids["type-a"]
    type_b = type_ids["type-b"]
    model = WorldModel(
        type_probabilities={
            type_a: (0.85, 0.10, 0.05),
            type_b: (0.10, 0.80, 0.10),
        },
        instances={
            instance_ids["a1"]: type_a,
            instance_ids["a2"]: type_a,
            instance_ids["a3"]: type_a,
            instance_ids["b1"]: type_b,
            instance_ids["b2"]: type_b,
            instance_ids["b3"]: type_b,
        },
        relation_probabilities={
            (
                type_a,
                type_b,
                relation_ids["known-relation"],
            ): 0.90,
        },
        regime_profiles={},
        active_regime=None,
        default_relation_probability=0.05,
    )

    a1 = instance_ids["a1"]
    a2 = instance_ids["a2"]
    a3 = instance_ids["a3"]
    b1 = instance_ids["b1"]
    b2 = instance_ids["b2"]
    b3 = instance_ids["b3"]
    u1 = instance_ids["unknown-1"]
    u2 = instance_ids["unknown-2"]
    novel_relation = relation_ids[
        "novel-relation"
    ]

    recovery = None

    if family == "noise-spike":
        evidence = EvidenceWindow(
            outcome_counts={
                a1: (17, 2, 1),
            },
            relation_counts={},
        )

    elif family == "incoherent-noise":
        evidence = EvidenceWindow(
            outcome_counts={
                a1: (16, 3, 1),
                a2: (18, 1, 1),
                a3: (17, 1, 2),
                b1: (2, 16, 2),
                b2: (3, 15, 2),
                b3: (1, 17, 2),
            },
            relation_counts={},
        )

    elif family == "new-instance":
        evidence = EvidenceWindow(
            outcome_counts={
                u1: (34, 4, 2),
            },
            relation_counts={},
        )

    elif family == "new-entity-type":
        evidence = EvidenceWindow(
            outcome_counts={
                u1: (2, 4, 34),
                u2: (3, 3, 34),
            },
            relation_counts={},
        )

    elif family == "new-relation":
        evidence = EvidenceWindow(
            outcome_counts={},
            relation_counts={
                (
                    a1,
                    b1,
                    novel_relation,
                ): (18, 20),
            },
        )

    elif family == "new-regime":
        evidence = EvidenceWindow(
            outcome_counts={
                a1: (4, 24, 2),
                a2: (5, 23, 2),
                a3: (3, 25, 2),
                b1: (21, 6, 3),
                b2: (22, 5, 3),
                b3: (20, 7, 3),
            },
            relation_counts={},
        )

    elif family == "transient-excursion":
        evidence = EvidenceWindow(
            outcome_counts={
                a1: (18, 2, 1),
                a2: (17, 3, 1),
                a3: (18, 1, 2),
                b1: (3, 16, 2),
                b2: (2, 17, 2),
                b3: (2, 16, 3),
            },
            relation_counts={},
        )
        recovery = EvidenceWindow(
            outcome_counts={
                a1: (18, 1, 1),
                a2: (17, 2, 1),
                a3: (18, 1, 1),
                b1: (2, 17, 1),
                b2: (2, 16, 2),
                b3: (1, 17, 2),
            },
            relation_counts={},
        )

    else:
        evidence = EvidenceWindow(
            outcome_counts={
                u1: (29, 0, 5),
            },
            relation_counts={},
        )

    roles = {
        **{
            f"type:{key}": value
            for key, value
            in type_ids.items()
        },
        **{
            f"instance:{key}": value
            for key, value
            in instance_ids.items()
        },
        **{
            f"relation:{key}": value
            for key, value
            in relation_ids.items()
        },
    }
    return NoveltyCase(
        family=family,
        seed=seed,
        model=model,
        evidence=evidence,
        recovery_evidence=recovery,
        roles=roles,
    )


def _score_map(read) -> dict[str, float]:
    return dict(read.scores)


def _edit_improvement(
    read,
) -> float | None:
    scores = _score_map(read)
    if (
        not read.resolved
        or read.category == "noise"
        or "noise" not in scores
    ):
        return None
    return (
        scores["noise"]
        - read.best_score
    )


def run_case(
    family: str,
    seed: int,
) -> dict:
    case = build_case(family, seed)
    attributor = CortexNoveltyAttributor(
        case.model,
    )
    baseline_nearest = (
        NearestKnownTypeBaseline()
    )
    baseline_reset = GlobalResetBaseline()

    before_fingerprint = (
        case.model.fingerprint()
    )
    read = attributor.evaluate(
        case.evidence
    )
    after_evaluation_fingerprint = (
        case.model.fingerprint()
    )

    applied = apply_attribution(
        case.model,
        read,
    )
    components = changed_components(
        case.model,
        applied,
    )
    post_read = (
        CortexNoveltyAttributor(
            applied,
        ).evaluate(
            case.evidence
        )
    )

    expected = EXPECTED_CATEGORY[family]
    expected_components = (
        EXPECTED_COMPONENTS[family]
    )

    recovery_category = None
    recovery_resolved = None
    recovery_no_edit = None
    if case.recovery_evidence is not None:
        recovery_read = attributor.evaluate(
            case.recovery_evidence
        )
        recovery_category = (
            recovery_read.category
        )
        recovery_resolved = (
            recovery_read.resolved
        )
        recovery_applied = apply_attribution(
            case.model,
            recovery_read,
        )
        recovery_no_edit = (
            recovery_applied.fingerprint()
            == case.model.fingerprint()
        )

    scores = _score_map(read)
    top_two = sorted(
        scores.items(),
        key=lambda item: (
            item[1],
            item[0],
        ),
    )[:2]

    nearest_prediction = (
        baseline_nearest.predict(
            case.model,
            case.evidence,
        )
    )
    reset_prediction = (
        baseline_reset.predict(
            case.model,
            case.evidence,
        )
    )

    persistent_expected = expected in (
        "new_instance",
        "new_entity_type",
        "new_relation",
        "new_regime",
    )

    return {
        "protocol": PROTOCOL,
        "family": family,
        "seed": seed,
        "expected_category": expected,
        "read": {
            "category": read.category,
            "resolved": read.resolved,
            "margin": read.margin,
            "best_score": read.best_score,
            "second_score": (
                read.second_score
            ),
            "scores": list(read.scores),
            "top_two": top_two,
            "structural_change": (
                read.structural_change
            ),
            "edit_improvement": (
                _edit_improvement(read)
            ),
        },
        "model": {
            "pure_evaluation": (
                before_fingerprint
                == after_evaluation_fingerprint
            ),
            "changed_components": (
                components
            ),
            "expected_components": (
                expected_components
            ),
            "edit_locality": (
                components
                == expected_components
            ),
            "persistent_edit_expected": (
                persistent_expected
            ),
            "persistent_edit_observed": (
                applied.fingerprint()
                != case.model.fingerprint()
            ),
            "post_edit_category": (
                post_read.category
            ),
            "post_edit_resolved": (
                post_read.resolved
            ),
        },
        "recovery": {
            "category": recovery_category,
            "resolved": recovery_resolved,
            "no_edit": recovery_no_edit,
        },
        "baselines": {
            "nearest_known_type": (
                nearest_prediction
            ),
            "global_reset": (
                reset_prediction
            ),
            "nearest_correct": (
                nearest_prediction
                == expected
            ),
            "reset_correct": (
                reset_prediction
                == expected
            ),
        },
        "checks": {
            "category_correct": (
                read.category == expected
            ),
            "resolved_correct": (
                read.resolved
                == (
                    expected
                    != "unresolved"
                )
            ),
            "no_false_edit": (
                (
                    applied.fingerprint()
                    == case.model.fingerprint()
                )
                if not persistent_expected
                else True
            ),
            "required_edit_present": (
                (
                    applied.fingerprint()
                    != case.model.fingerprint()
                )
                if persistent_expected
                else True
            ),
            "post_edit_stable": (
                (
                    post_read.category
                    == "noise"
                    and post_read.resolved
                )
                if persistent_expected
                else True
            ),
            "ambiguous_no_edit": (
                (
                    not read.resolved
                    and read.proposed_edit
                    is None
                    and read.margin
                    < attributor.resolution_margin
                )
                if family
                == "ambiguous-instance-type"
                else True
            ),
            "ambiguous_competitors": (
                (
                    {
                        item[0]
                        for item in top_two
                    }
                    == {
                        "new_instance",
                        "new_entity_type",
                    }
                )
                if family
                == "ambiguous-instance-type"
                else True
            ),
            "transient_recovery": (
                (
                    recovery_category
                    == "noise"
                    and recovery_resolved
                    and recovery_no_edit
                )
                if family
                == "transient-excursion"
                else True
            ),
        },
    }


def _rate(rows, predicate) -> float:
    rows = list(rows)
    if not rows:
        return 0.0
    return sum(
        int(predicate(row))
        for row in rows
    ) / len(rows)


def campaign(
    seeds: tuple[int, ...],
    output: Path,
) -> dict:
    rows = [
        run_case(family, seed)
        for seed in seeds
        for family in FAMILIES
    ]
    output.write_text(
        "".join(
            json.dumps(
                row,
                sort_keys=True,
            )
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )

    summary = {
        "protocol": PROTOCOL,
        "seeds": list(seeds),
        "families": list(FAMILIES),
        "rows": len(rows),
        "attribution_accuracy": _rate(
            rows,
            lambda row: row["checks"][
                "category_correct"
            ],
        ),
        "nearest_baseline_accuracy": _rate(
            rows,
            lambda row: row["baselines"][
                "nearest_correct"
            ],
        ),
        "reset_baseline_accuracy": _rate(
            rows,
            lambda row: row["baselines"][
                "reset_correct"
            ],
        ),
        "edit_locality_rate": _rate(
            rows,
            lambda row: row["model"][
                "edit_locality"
            ],
        ),
    }
    print(
        json.dumps(
            summary,
            sort_keys=True,
        )
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--development",
        action="store_true",
    )
    group.add_argument(
        "--official",
        action="store_true",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "open-world-novelty-attribution-v1.jsonl"
        ),
    )
    args = parser.parse_args()
    seeds = (
        OFFICIAL_SEEDS
        if args.official
        else DEVELOPMENT_SEEDS
    )
    campaign(seeds, args.output)


if __name__ == "__main__":
    main()
