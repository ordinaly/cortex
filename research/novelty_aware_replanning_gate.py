
"""Frozen gates for Cortex novelty-aware-replanning-v1."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from novelty_aware_replanning_campaign import (
    DEVELOPMENT_SEEDS,
    FAMILIES,
    LOCAL_FAMILIES,
    OFFICIAL_SEEDS,
    PRESERVE_FAMILIES,
    PROTOCOL,
)


EXPECTED_COMPONENTS = {
    "noise-preserve": tuple(),
    "new-instance-irrelevant": (
        "instances",
    ),
    "new-instance-local": (
        "instances",
    ),
    "new-relation-local": (
        "relations",
    ),
    "new-relation-irrelevant": (
        "relations",
    ),
    "new-type-local": (
        "instances",
        "types",
    ),
    "new-regime-global": (
        "active_regime",
        "regimes",
    ),
    "unresolved-defer": tuple(),
}


def rate(rows, predicate) -> float:
    rows = list(rows)
    if not rows:
        return 0.0
    return sum(
        int(predicate(row))
        for row in rows
    ) / len(rows)


def mean(values) -> float:
    values = list(values)
    if not values:
        return 0.0
    return sum(values) / len(values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "results",
        type=Path,
    )
    parser.add_argument(
        "--official",
        action="store_true",
    )
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in args.results.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    if not rows:
        raise SystemExit(
            "no campaign rows"
        )

    seeds = (
        OFFICIAL_SEEDS
        if args.official
        else DEVELOPMENT_SEEDS
    )
    by_family = {
        family: [
            row
            for row in rows
            if row["family"]
            == family
        ]
        for family in FAMILIES
    }
    local_rows = [
        row
        for row in rows
        if row["family"]
        in LOCAL_FAMILIES
    ]
    preserve_rows = [
        row
        for row in rows
        if row["family"]
        in PRESERVE_FAMILIES
    ]
    active_repair_rows = [
        row
        for row in rows
        if row["family"]
        in (
            *LOCAL_FAMILIES,
            "new-regime-global",
        )
    ]

    ratios = [
        row["checks"][
            "local_expansion_ratio"
        ]
        for row in local_rows
        if row["checks"][
            "local_expansion_ratio"
        ] is not None
    ]

    ignore_impacted = [
        row
        for row in rows
        if row["family"]
        in (
            *LOCAL_FAMILIES,
            "new-regime-global",
        )
    ]

    metrics = {
        "attribution_accuracy": rate(
            rows,
            lambda row: row["checks"][
                "attribution_correct"
            ],
        ),
        "decision_accuracy": rate(
            rows,
            lambda row: row["checks"][
                "decision_correct"
            ],
        ),
        "result_validity": rate(
            rows,
            lambda row: row["checks"][
                "result_valid"
            ],
        ),
        "model_component_locality": rate(
            rows,
            lambda row: tuple(
                row["result"][
                    "model_changed_components"
                ]
            )
            == EXPECTED_COMPONENTS[
                row["family"]
            ],
        ),
        "repair_optimality": rate(
            active_repair_rows,
            lambda row: row["checks"][
                "optimal_cost"
            ],
        ),
        "preserve_exactness": rate(
            preserve_rows,
            lambda row: row["checks"][
                "preserve_exact"
            ],
        ),
        "preserve_zero_search": rate(
            preserve_rows,
            lambda row: row["checks"][
                "no_unnecessary_search"
            ],
        ),
        "local_impact_accuracy": rate(
            local_rows,
            lambda row: row["checks"][
                "impact_correct"
            ],
        ),
        "local_prefix_preservation": rate(
            local_rows,
            lambda row: row["checks"][
                "local_prefix_preserved"
            ],
        ),
        "local_suffix_preservation": rate(
            local_rows,
            lambda row: row["checks"][
                "local_suffix_preserved"
            ],
        ),
        "local_search_saving_rate": rate(
            local_rows,
            lambda row: row["checks"][
                "local_search_saving"
            ],
        ),
        "mean_local_expansion_ratio": (
            mean(ratios)
        ),
        "regime_global_rate": rate(
            by_family[
                "new-regime-global"
            ],
            lambda row: row["checks"][
                "regime_global"
            ],
        ),
        "unresolved_safety_rate": rate(
            by_family[
                "unresolved-defer"
            ],
            lambda row: row["checks"][
                "unresolved_safe"
            ],
        ),
        "always_replan_decision_accuracy": rate(
            rows,
            lambda row: row["baselines"][
                "always_replan_correct"
            ],
        ),
        "ignore_impacted_optimality": rate(
            ignore_impacted,
            lambda row: bool(
                row["baselines"][
                    "ignore_optimal"
                ]
            ),
        ),
    }

    checks = {
        "protocol_exact": all(
            row["protocol"] == PROTOCOL
            for row in rows
        ),
        "row_count": (
            len(rows)
            == len(FAMILIES) * len(seeds)
        ),
        "families_exact": sorted(
            {
                row["family"]
                for row in rows
            }
        ) == sorted(FAMILIES),
        "seeds_exact": sorted(
            {
                int(row["seed"])
                for row in rows
            }
        ) == list(seeds),
        "family_balance": all(
            len(selected)
            == len(seeds)
            for selected
            in by_family.values()
        ),
        "attribution_accuracy": (
            metrics[
                "attribution_accuracy"
            ] == 1.0
        ),
        "decision_accuracy": (
            metrics[
                "decision_accuracy"
            ] == 1.0
        ),
        "result_validity": (
            metrics[
                "result_validity"
            ] == 1.0
        ),
        "model_component_locality": (
            metrics[
                "model_component_locality"
            ] == 1.0
        ),
        "repair_optimality": (
            metrics[
                "repair_optimality"
            ] == 1.0
        ),
        "preserve_exactness": (
            metrics[
                "preserve_exactness"
            ] == 1.0
        ),
        "preserve_zero_search": (
            metrics[
                "preserve_zero_search"
            ] == 1.0
        ),
        "local_impact_accuracy": (
            metrics[
                "local_impact_accuracy"
            ] == 1.0
        ),
        "local_prefix_preservation": (
            metrics[
                "local_prefix_preservation"
            ] == 1.0
        ),
        "local_suffix_preservation": (
            metrics[
                "local_suffix_preservation"
            ] == 1.0
        ),
        "local_search_saving": (
            metrics[
                "local_search_saving_rate"
            ] == 1.0
        ),
        "mean_local_expansion_ratio": (
            metrics[
                "mean_local_expansion_ratio"
            ] <= 0.50
        ),
        "regime_global": (
            metrics[
                "regime_global_rate"
            ] == 1.0
        ),
        "unresolved_safety": (
            metrics[
                "unresolved_safety_rate"
            ] == 1.0
        ),
        "always_replan_discriminated": (
            metrics[
                "always_replan_decision_accuracy"
            ] <= 0.50
        ),
        "ignore_novelty_discriminated": (
            metrics[
                "ignore_impacted_optimality"
            ] <= 0.25
        ),
    }

    payload = {
        "protocol": PROTOCOL,
        "mode": (
            "official"
            if args.official
            else "development"
        ),
        "rows": len(rows),
        "metrics": metrics,
        "checks": checks,
        "passed": all(
            checks.values()
        ),
    }
    print(
        json.dumps(
            payload,
            sort_keys=True,
        )
    )

    if not payload["passed"]:
        raise SystemExit(
            "failed gates: "
            + ", ".join(
                name
                for name, passed
                in checks.items()
                if not passed
            )
        )


if __name__ == "__main__":
    main()
