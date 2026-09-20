"""Campaign runner for Hybrid Symbiosis v0.1."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from hybrid_symbiosis import (
    DEVELOPMENT_SEEDS,
    FAMILIES,
    PROTOCOL,
    run_case,
)


def campaign(
    *,
    seeds: tuple[int, ...],
    output: Path,
) -> dict:
    rows = [
        run_case(
            family,
            seed,
        )
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

    def rate(
        family: str,
        check: str,
    ) -> float:
        selected = [
            row
            for row in rows
            if row[
                "family"
            ]
            == family
        ]
        return (
            sum(
                int(
                    row[
                        "checks"
                    ][
                        check
                    ]
                )
                for row
                in selected
            )
            / len(
                selected
            )
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

    summary = {
        "protocol": PROTOCOL,
        "seeds": list(
            seeds
        ),
        "families": list(
            FAMILIES
        ),
        "rows": len(
            rows
        ),
        "metrics": {
            "neural_prior_passthrough_rate": (
                rate(
                    "neural-prior",
                    "neural_prior",
                )
            ),
            "independent_support_rate": (
                rate(
                    "independent-support",
                    "supported",
                )
            ),
            "context_guided_rate": (
                rate(
                    "context-guided",
                    "context_changed_neural_candidate",
                )
            ),
            "context_guided_no_revision_rate": (
                rate(
                    "context-guided",
                    "supported_without_revision",
                )
            ),
            "conflict_revision_trigger_rate": (
                rate(
                    "conflict-revision",
                    "revision_triggered",
                )
            ),
            "conflict_revision_success_rate": (
                rate(
                    "conflict-revision",
                    "revision_resolved",
                )
            ),
            "stubborn_conflict_bounded_rate": (
                rate(
                    "conflict-stubborn",
                    "bounded_one_revision",
                )
            ),
            "stubborn_conflict_preservation_rate": (
                rate(
                    "conflict-stubborn",
                    "conflict_preserved",
                )
            ),
            "bootstrap_alignment_rate": (
                rate(
                    "bootstrap-aligned",
                    "bootstrap_aligned",
                )
            ),
            "bootstrap_no_circular_support_rate": (
                rate(
                    "bootstrap-aligned",
                    "not_independent_support",
                )
            ),
            "bootstrap_divergence_rate": (
                rate(
                    "bootstrap-divergence",
                    "bootstrap_divergence",
                )
            ),
            "bootstrap_divergence_no_revision_rate": (
                rate(
                    "bootstrap-divergence",
                    "no_forced_revision",
                )
            ),
            "bootstrap_promotion_rate": (
                rate(
                    "bootstrap-promoted",
                    "promoted_to_support",
                )
            ),
            "parser_failure_answer_survival_rate": (
                rate(
                    "parser-failure",
                    "answer_survived",
                )
            ),
            "parser_failure_visibility_rate": (
                rate(
                    "parser-failure",
                    "analysis_error_exposed",
                )
            ),
            "alternative_object_conflict_detection_rate": (
                rate(
                    "alternative-object-divergence",
                    "alternative_object_conflict_detected",
                )
            ),
            "alternative_object_survival_rate": (
                rate(
                    "alternative-object-divergence",
                    "characterized_without_crash",
                )
            ),
            "cold_unknown_rate": (
                rate(
                    "cold-vs-bootstrap",
                    "cold_not_resolved",
                )
            ),
            "bootstrap_resolution_rate": (
                rate(
                    "cold-vs-bootstrap",
                    "bootstrap_resolved",
                )
            ),
            "normal_two_call_rate": (
                sum(
                    int(
                        row[
                            "neural_calls"
                        ]
                        == 2
                    )
                    for row
                    in normal_rows
                )
                / len(
                    normal_rows
                )
            ),
            "conflict_three_call_rate": (
                sum(
                    int(
                        row[
                            "neural_calls"
                        ]
                        == 3
                    )
                    for row
                    in conflict_rows
                )
                / len(
                    conflict_rows
                )
            ),
            "nonempty_answer_rate": (
                sum(
                    int(
                        row[
                            "answer_nonempty"
                        ]
                    )
                    for row
                    in rows
                )
                / len(
                    rows
                )
            ),
        },
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
    parser.add_argument(
        "--seeds",
        type=int,
        default=len(
            DEVELOPMENT_SEEDS
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "hybrid-symbiosis-v0.1.jsonl"
        ),
    )
    args = parser.parse_args()
    seeds = tuple(
        range(
            max(
                1,
                int(
                    args.seeds
                ),
            )
        )
    )
    campaign(
        seeds=seeds,
        output=args.output,
    )


if __name__ == "__main__":
    main()
