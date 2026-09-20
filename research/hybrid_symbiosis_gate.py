"""Gates and characterization summary for Hybrid Symbiosis v0.1."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from hybrid_symbiosis import (
    FAMILIES,
    PROTOCOL,
)


def rate(
    rows,
    predicate,
) -> float:
    rows = list(
        rows
    )
    if not rows:
        return 0.0
    return (
        sum(
            int(
                predicate(
                    row
                )
            )
            for row
            in rows
        )
        / len(
            rows
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "results",
        type=Path,
    )
    args = parser.parse_args()

    rows = [
        json.loads(
            line
        )
        for line
        in args.results.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    if not rows:
        raise SystemExit(
            "no campaign rows"
        )

    by_family = {
        family: [
            row
            for row in rows
            if row[
                "family"
            ]
            == family
        ]
        for family in FAMILIES
    }
    seeds = sorted(
        {
            int(
                row[
                    "seed"
                ]
            )
            for row in rows
        }
    )

    def check_rate(
        family: str,
        check: str,
    ) -> float:
        return rate(
            by_family[
                family
            ],
            lambda row: bool(
                row[
                    "checks"
                ][
                    check
                ]
            ),
        )

    normal_families = {
        "neural-prior",
        "independent-support",
        "context-guided",
        "bootstrap-aligned",
        "bootstrap-divergence",
        "bootstrap-promoted",
        "parser-failure",
        "alternative-object-divergence",
        "cold-vs-bootstrap",
    }
    normal_rows = [
        row
        for row in rows
        if row[
            "family"
        ]
        in normal_families
    ]
    conflict_rows = [
        row
        for row in rows
        if row[
            "family"
        ]
        in {
            "conflict-revision",
            "conflict-stubborn",
        }
    ]

    metrics = {
        "neural_prior_passthrough_rate": (
            check_rate(
                "neural-prior",
                "neural_prior",
            )
        ),
        "independent_support_rate": (
            check_rate(
                "independent-support",
                "supported",
            )
        ),
        "context_guided_rate": (
            check_rate(
                "context-guided",
                "context_changed_neural_candidate",
            )
        ),
        "conflict_revision_success_rate": (
            check_rate(
                "conflict-revision",
                "revision_resolved",
            )
        ),
        "stubborn_conflict_preservation_rate": (
            check_rate(
                "conflict-stubborn",
                "conflict_preserved",
            )
        ),
        "bootstrap_no_circular_support_rate": (
            check_rate(
                "bootstrap-aligned",
                "not_independent_support",
            )
        ),
        "bootstrap_divergence_no_revision_rate": (
            check_rate(
                "bootstrap-divergence",
                "no_forced_revision",
            )
        ),
        "bootstrap_promotion_rate": (
            check_rate(
                "bootstrap-promoted",
                "promoted_to_support",
            )
        ),
        "parser_failure_answer_survival_rate": (
            check_rate(
                "parser-failure",
                "answer_survived",
            )
        ),
        "alternative_object_conflict_detection_rate": (
            check_rate(
                "alternative-object-divergence",
                "alternative_object_conflict_detected",
            )
        ),
        "bootstrap_resolution_rate": (
            check_rate(
                "cold-vs-bootstrap",
                "bootstrap_resolved",
            )
        ),
        "normal_two_call_rate": rate(
            normal_rows,
            lambda row: (
                row[
                    "neural_calls"
                ]
                == 2
            ),
        ),
        "conflict_three_call_rate": rate(
            conflict_rows,
            lambda row: (
                row[
                    "neural_calls"
                ]
                == 3
            ),
        ),
        "nonempty_answer_rate": rate(
            rows,
            lambda row: bool(
                row[
                    "answer_nonempty"
                ]
            ),
        ),
    }

    checks = {
        "protocol_exact": all(
            row[
                "protocol"
            ]
            == PROTOCOL
            for row in rows
        ),
        "family_coverage": (
            sorted(
                by_family
            )
            == sorted(
                FAMILIES
            )
            and all(
                by_family[
                    family
                ]
                for family in FAMILIES
            )
        ),
        "family_balance": all(
            len(
                by_family[
                    family
                ]
            )
            == len(
                seeds
            )
            for family in FAMILIES
        ),
        "neural_prior_passthrough": (
            metrics[
                "neural_prior_passthrough_rate"
            ]
            == 1.0
        ),
        "independent_support": (
            metrics[
                "independent_support_rate"
            ]
            == 1.0
        ),
        "context_guidance": (
            metrics[
                "context_guided_rate"
            ]
            == 1.0
        ),
        "conflict_revision": (
            metrics[
                "conflict_revision_success_rate"
            ]
            == 1.0
        ),
        "stubborn_conflict_bounded": (
            metrics[
                "stubborn_conflict_preservation_rate"
            ]
            == 1.0
        ),
        "bootstrap_not_circular": (
            metrics[
                "bootstrap_no_circular_support_rate"
            ]
            == 1.0
        ),
        "bootstrap_divergence_not_forced": (
            metrics[
                "bootstrap_divergence_no_revision_rate"
            ]
            == 1.0
        ),
        "bootstrap_promotion": (
            metrics[
                "bootstrap_promotion_rate"
            ]
            == 1.0
        ),
        "parser_failure_survival": (
            metrics[
                "parser_failure_answer_survival_rate"
            ]
            == 1.0
        ),
        "bootstrap_cold_start": (
            metrics[
                "bootstrap_resolution_rate"
            ]
            == 1.0
        ),
        "normal_two_calls": (
            metrics[
                "normal_two_call_rate"
            ]
            == 1.0
        ),
        "conflict_three_calls": (
            metrics[
                "conflict_three_call_rate"
            ]
            == 1.0
        ),
        "nonempty_answers": (
            metrics[
                "nonempty_answer_rate"
            ]
            == 1.0
        ),
    }

    report = {
        "protocol": PROTOCOL,
        "rows": len(
            rows
        ),
        "seeds": seeds,
        "metrics": metrics,
        "checks": checks,
        "characterization": {
            "alternative_object_conflict_detection_rate": (
                metrics[
                    "alternative_object_conflict_detection_rate"
                ]
            ),
            "note": (
                "Alternative-object divergence is characterized but is "
                "not a v0.1 gate because generic relations do not yet "
                "declare functional/exclusive cardinality."
            ),
        },
    }
    print(
        json.dumps(
            report,
            sort_keys=True,
        )
    )

    failed = [
        key
        for key, value
        in checks.items()
        if not value
    ]
    if failed:
        raise SystemExit(
            "failed gates: "
            + ", ".join(
                failed
            )
        )


if __name__ == "__main__":
    main()
