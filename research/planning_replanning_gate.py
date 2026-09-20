"""Frozen gates for Cortex planning-replanning-v1."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from planning_replanning_campaign import (
    DEVELOPMENT_SEEDS,
    FAMILIES,
    LOCAL_FAMILIES,
    OFFICIAL_SEEDS,
    PROTOCOL,
)


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
        raise SystemExit("no campaign rows")

    seeds = (
        OFFICIAL_SEEDS
        if args.official
        else DEVELOPMENT_SEEDS
    )
    by_family = {
        family: [
            row
            for row in rows
            if row["family"] == family
        ]
        for family in FAMILIES
    }
    local_rows = [
        row
        for row in rows
        if row["family"] in LOCAL_FAMILIES
    ]
    feasible_rows = [
        row
        for row in rows
        if row["family"] != "infeasible"
    ]

    local_ratios = [
        (
            row["repair"][
                "local_expanded_states"
            ]
            / row["full_replan"][
                "expanded_states"
            ]
        )
        for row in local_rows
        if (
            row["full_replan"][
                "expanded_states"
            ]
            and row["full_replan"][
                "expanded_states"
            ] > 0
        )
    ]

    metrics = {
        "base_valid_rate": rate(
            rows,
            lambda row: (
                row["base"]["valid"]
                and row["base"][
                    "support_valid"
                ]
                and row["base"][
                    "is_main_chain"
                ]
                and not row["base"][
                    "dead_end_action_used"
                ]
            ),
        ),
        "mode_accuracy": rate(
            rows,
            lambda row: row["checks"][
                "mode_correct"
            ],
        ),
        "impact_accuracy": rate(
            rows,
            lambda row: row["checks"][
                "impact_correct"
            ],
        ),
        "feasibility_agreement": rate(
            rows,
            lambda row: row["checks"][
                "feasible_agreement"
            ],
        ),
        "optimal_cost_rate": rate(
            rows,
            lambda row: row["checks"][
                "optimal_cost"
            ],
        ),
        "feasible_plan_validity": rate(
            feasible_rows,
            lambda row: (
                row["repair"]["valid"]
                and row["repair"][
                    "support_valid"
                ]
                and not row["repair"][
                    "dead_end_action_used"
                ]
            ),
        ),
        "full_baseline_dead_end_avoidance": rate(
            feasible_rows,
            lambda row: (
                not row["full_replan"][
                    "dead_end_action_used"
                ]
            ),
        ),
        "local_repair_rate": rate(
            local_rows,
            lambda row: (
                row["repair"]["mode"]
                == "local-repair"
            ),
        ),
        "local_rejoin_next_state_rate": rate(
            local_rows,
            lambda row: (
                row["repair"]["rejoin_index"]
                == row[
                    "expected_impact_index"
                ]
                + 1
            ),
        ),
        "local_prefix_preservation_rate": rate(
            local_rows,
            lambda row: (
                row["repair"][
                    "preserved_prefix_actions"
                ]
                == row[
                    "expected_impact_index"
                ]
                and row["repair"][
                    "preserved_prefix_actions"
                ]
                >= 2
            ),
        ),
        "local_suffix_preservation_rate": rate(
            local_rows,
            lambda row: (
                row["repair"][
                    "preserved_suffix_actions"
                ]
                >= 1
            ),
        ),
        "local_role_correctness": rate(
            local_rows,
            lambda row: row["checks"][
                "local_role_check"
            ],
        ),
        "local_search_saving_rate": rate(
            local_rows,
            lambda row: row["checks"][
                "local_search_saving"
            ],
        ),
        "mean_local_vs_full_expansion_ratio": (
            mean(local_ratios)
        ),
        "cost_adaptation_rate": rate(
            by_family["cost-shift"],
            lambda row: row["checks"][
                "cost_improvement"
            ],
        ),
        "reversible_repair_rate": rate(
            by_family[
                "reversible-detour"
            ],
            lambda row: (
                row["checks"][
                    "reversible_state_restored"
                ]
                and row["checks"][
                    "local_role_check"
                ]
            ),
        ),
        "irrelevant_change_stability": rate(
            by_family["irrelevant-change"],
            lambda row: (
                row["checks"][
                    "irrelevant_exact"
                ]
                and row["checks"][
                    "support_unchanged"
                ]
            ),
        ),
        "global_fallback_correctness": rate(
            by_family["global-reroute"],
            lambda row: (
                row["repair"]["mode"]
                == "full-replan"
                and row["checks"][
                    "global_roles_present"
                ]
                and row["repair"]["valid"]
                and row["checks"][
                    "optimal_cost"
                ]
            ),
        ),
        "infeasible_detection_rate": rate(
            by_family["infeasible"],
            lambda row: (
                row["repair"]["mode"]
                == "infeasible"
                and not row["full_replan"][
                    "feasible"
                ]
                and row["repair"]["cost"]
                is None
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
            {row["family"] for row in rows}
        ) == sorted(FAMILIES),
        "seeds_exact": sorted(
            {int(row["seed"]) for row in rows}
        ) == list(seeds),
        "family_balance": all(
            len(selected) == len(seeds)
            for selected in by_family.values()
        ),
        "base_valid": (
            metrics["base_valid_rate"] == 1.0
        ),
        "mode_accuracy": (
            metrics["mode_accuracy"] == 1.0
        ),
        "impact_accuracy": (
            metrics["impact_accuracy"] == 1.0
        ),
        "feasibility_agreement": (
            metrics[
                "feasibility_agreement"
            ] == 1.0
        ),
        "optimal_cost": (
            metrics["optimal_cost_rate"]
            == 1.0
        ),
        "feasible_plan_validity": (
            metrics[
                "feasible_plan_validity"
            ] == 1.0
        ),
        "full_dead_end_avoidance": (
            metrics[
                "full_baseline_dead_end_avoidance"
            ] == 1.0
        ),
        "local_repair": (
            metrics["local_repair_rate"]
            == 1.0
        ),
        "local_rejoin": (
            metrics[
                "local_rejoin_next_state_rate"
            ] == 1.0
        ),
        "local_prefix_preservation": (
            metrics[
                "local_prefix_preservation_rate"
            ] == 1.0
        ),
        "local_suffix_preservation": (
            metrics[
                "local_suffix_preservation_rate"
            ] == 1.0
        ),
        "local_role_correctness": (
            metrics[
                "local_role_correctness"
            ] == 1.0
        ),
        "local_search_saving": (
            metrics[
                "local_search_saving_rate"
            ] == 1.0
        ),
        "mean_search_reduction": (
            metrics[
                "mean_local_vs_full_expansion_ratio"
            ] <= 0.50
        ),
        "cost_adaptation": (
            metrics[
                "cost_adaptation_rate"
            ] == 1.0
        ),
        "reversible_repair": (
            metrics[
                "reversible_repair_rate"
            ] == 1.0
        ),
        "irrelevant_stability": (
            metrics[
                "irrelevant_change_stability"
            ] == 1.0
        ),
        "global_fallback": (
            metrics[
                "global_fallback_correctness"
            ] == 1.0
        ),
        "infeasible_detection": (
            metrics[
                "infeasible_detection_rate"
            ] == 1.0
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
        "passed": all(checks.values()),
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
