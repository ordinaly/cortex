"""Development/frozen gates for latent-concept-induction-v1."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

from latent_concept_induction_campaign import (
    DEVELOPMENT_SEEDS,
    OFFICIAL_SEEDS,
    PROTOCOL,
)


def summarize(rows: list[dict], *, official: bool) -> dict:
    seeds = OFFICIAL_SEEDS if official else DEVELOPMENT_SEEDS

    two = [
        row for row in rows
        if row["family"] == "two-concept"
    ]
    three = [
        row for row in rows
        if row["family"] == "three-concept"
    ]
    singleton = [
        row for row in rows
        if row["case"] == "singleton-control"
    ]
    ambiguity = [
        row for row in rows
        if row["case"] == "ambiguity"
    ]
    lifecycle = [
        row for row in rows
        if row["case"] == "lifecycle"
    ]

    metrics = {
        "two_min_f1": min(row["partition_f1"] for row in two),
        "three_min_f1": min(
            row["partition_f1"] for row in three
        ),
        "two_mean_flat_gain": mean(
            row["flat_gain"] for row in two
        ),
        "three_mean_flat_gain": mean(
            row["flat_gain"] for row in three
        ),
        "min_global_gain": min(
            row["global_gain"]
            for row in two + three
        ),
        "max_decoy_f1": max(
            row["decoy_partition_f1"]
            for row in two + three
        ),
        "singleton_correct_rate": mean(
            int(
                row["concept_count"] == 6
                and row["resolved_latent_blocks"] == 0
            )
            for row in singleton
        ),
        "ambiguity_unresolved_rate": mean(
            int(
                not row["resolved_partition"]
                and not row["ambiguous_entity_resolved"]
                and row["version_space"] >= 2
            )
            for row in ambiguity
        ),
        "lifecycle_exact_rate": mean(
            int(
                row["coarse_f1"] == 1.0
                and row["split_f1"] == 1.0
                and row["merge_f1"] == 1.0
                and row["retract_f1"] == 1.0
            )
            for row in lifecycle
        ),
        "lifecycle_event_rate": mean(
            int(
                row["created_seen"]
                and row["split_seen"]
                and row["merge_seen"]
                and row["retracted_seen"]
            )
            for row in lifecycle
        ),
        "lifecycle_empty_rate": mean(
            int(
                row["final_active_concepts"] == 0
                and row["persistent_empty_state"]
            )
            for row in lifecycle
        ),
        "candidate_count_min": min(
            row["candidate_count"] for row in rows
        ),
        "candidate_count_max": max(
            row["candidate_count"] for row in rows
        ),
    }

    # Frozen gates shared by development verification and the untouched
    # official campaign. Do not weaken after official evaluation begins.
    expected_rows = len(seeds) * 5
    checks = {
        "protocol_exact": all(
            row["protocol"] == PROTOCOL
            for row in rows
        ),
        "row_count": len(rows) == expected_rows,
        "seeds_exact": sorted(
            {int(row["seed"]) for row in rows}
        ) == list(seeds),
        "two_concept_recovery": (
            metrics["two_min_f1"] == 1.0
        ),
        "three_concept_recovery": (
            metrics["three_min_f1"] == 1.0
        ),
        "two_predictive_gain": (
            metrics["two_mean_flat_gain"] >= 0.03
        ),
        "three_predictive_gain": (
            metrics["three_mean_flat_gain"] >= 0.015
        ),
        "beats_global": (
            metrics["min_global_gain"] > 0.0
        ),
        "decoy_rejected": (
            metrics["max_decoy_f1"] <= 1.0 / 3.0 + 1.0e-12
        ),
        "singleton_control": (
            metrics["singleton_correct_rate"] == 1.0
        ),
        "ambiguity_preserved": (
            metrics["ambiguity_unresolved_rate"] == 1.0
        ),
        "lifecycle_partitions": (
            metrics["lifecycle_exact_rate"] == 1.0
        ),
        "lifecycle_events": (
            metrics["lifecycle_event_rate"] == 1.0
        ),
        "lifecycle_retraction": (
            metrics["lifecycle_empty_rate"] == 1.0
        ),
        "grammar_exact": (
            metrics["candidate_count_min"] == 203
            and metrics["candidate_count_max"] == 203
        ),
    }

    return {
        "protocol": PROTOCOL,
        "mode": "official" if official else "development",
        "rows": len(rows),
        "metrics": metrics,
        "checks": checks,
        "passed": all(checks.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--official", action="store_true")
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in args.input.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    payload = summarize(rows, official=args.official)
    print(json.dumps(payload, sort_keys=True))

    if not payload["passed"]:
        raise SystemExit(
            "failed gates: "
            + ", ".join(
                name
                for name, passed in payload["checks"].items()
                if not passed
            )
        )


if __name__ == "__main__":
    main()
