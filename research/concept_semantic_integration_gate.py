"""Development/frozen gates for concept-semantic-integration-v1."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

from concept_semantic_integration_campaign import (
    DEVELOPMENT_SEEDS,
    OFFICIAL_SEEDS,
    PROTOCOL,
)


def summarize(rows: list[dict], *, official: bool) -> dict:
    seeds = OFFICIAL_SEEDS if official else DEVELOPMENT_SEEDS

    adversarial = [
        row for row in rows
        if row["case"] == "adversarial-grounding"
    ]
    isolation = [
        row for row in rows
        if row["case"] == "label-isolation"
    ]
    unnamed = [
        row for row in rows
        if row["case"] == "unnamed-concept"
    ]
    lifecycle = [
        row for row in rows
        if row["case"] == "semantic-lifecycle"
    ]

    metrics = {
        "min_adversarial_concept_f1": min(
            row["concept_f1"] for row in adversarial
        ),
        "mean_raw_neural_accuracy": mean(
            row["raw_neural_accuracy"]
            for row in adversarial
        ),
        "mean_semantic_coverage": mean(
            row["semantic_coverage"]
            for row in adversarial
        ),
        "mean_semantic_accuracy": mean(
            row["semantic_accuracy"]
            for row in adversarial
        ),
        "mean_effective_semantic_accuracy": mean(
            row["effective_semantic_accuracy"]
            for row in adversarial
        ),
        "distractor_resolution_rate": mean(
            int(row["distractor_resolved"] > 0)
            for row in adversarial
        ),
        "label_isolation_rate": mean(
            int(
                row["members_equal"]
                and row["concept_ids_equal"]
                and row["concept_f1"] == 1.0
            )
            for row in isolation
        ),
        "min_unnamed_concept_f1": min(
            row["concept_f1"] for row in unnamed
        ),
        "mean_known_semantic_accuracy": mean(
            row["known_semantic_accuracy"]
            for row in unnamed
        ),
        "unnamed_unresolved_rate": mean(
            int(not row["unnamed_resolved"])
            for row in unnamed
        ),
        "max_unnamed_confidence": max(
            row["unnamed_confidence"]
            for row in unnamed
        ),
        "split_semantic_lineage_rate": mean(
            int(row["split_semantics_correct"])
            for row in lifecycle
        ),
        "merge_semantic_lineage_rate": mean(
            int(row["merge_semantics_correct"])
            for row in lifecycle
        ),
        "retraction_invalidation_rate": mean(
            int(row["retraction_invalidates_semantics"])
            for row in lifecycle
        ),
        "identity_lineage_rate": mean(
            int(row["identity_lineage_correct"])
            for row in lifecycle
        ),
    }

    checks = {
        "protocol_exact": all(
            row["protocol"] == PROTOCOL
            for row in rows
        ),
        "row_count": len(rows) == len(seeds) * 4,
        "seeds_exact": sorted(
            {int(row["seed"]) for row in rows}
        ) == list(seeds),
        "concept_recovery": (
            metrics["min_adversarial_concept_f1"] == 1.0
            and metrics["min_unnamed_concept_f1"] == 1.0
        ),
        "semantic_correction": (
            metrics["mean_effective_semantic_accuracy"] >= 0.95
            and metrics["mean_semantic_accuracy"] >= 0.99
            and metrics["mean_semantic_coverage"] >= 0.95
            and metrics["mean_effective_semantic_accuracy"]
            >= metrics["mean_raw_neural_accuracy"] + 0.50
        ),
        "distractor_rejection": (
            metrics["distractor_resolution_rate"] == 0.0
        ),
        "label_causal_isolation": (
            metrics["label_isolation_rate"] == 1.0
        ),
        "known_naming": (
            metrics["mean_known_semantic_accuracy"] == 1.0
        ),
        "unnamed_preserved": (
            metrics["unnamed_unresolved_rate"] == 1.0
            and metrics["max_unnamed_confidence"] <= 0.50
        ),
        "split_semantic_lineage": (
            metrics["split_semantic_lineage_rate"] == 1.0
        ),
        "merge_semantic_lineage": (
            metrics["merge_semantic_lineage_rate"] == 1.0
        ),
        "retraction_invalidates_semantics": (
            metrics["retraction_invalidation_rate"] == 1.0
        ),
        "identity_lineage": (
            metrics["identity_lineage_rate"] == 1.0
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
