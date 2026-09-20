"""Preflight/official gates for hybrid-semantic-grounding-v1."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

from semantic_grounding_campaign import (
    CORRUPTED_PAIRS,
    DEVELOPMENT_SEEDS,
    FAMILY_EDGES,
    OFFICIAL_SEEDS,
    PROTOCOL,
    STRUCTURE_BUDGETS,
)


def _mean(rows: list[dict], key: str) -> float:
    return mean(float(row[key]) for row in rows)


def summarize(rows: list[dict], *, official: bool) -> dict:
    normal = [row for row in rows if row["case"] == "normal"]
    ambiguity = [row for row in rows if row["case"] == "ambiguity"]
    seeds = OFFICIAL_SEEDS if official else DEVELOPMENT_SEEDS

    def select(pairs: int, budget: int) -> list[dict]:
        return [
            row for row in normal
            if row["corrupted_pairs"] == pairs
            and row["structure_budget"] == budget
        ]

    clean = select(0, 12)
    one_pair_0 = select(1, 0)
    one_pair_8 = select(1, 8)
    two_pair_0 = select(2, 0)
    two_pair_12 = select(2, 12)

    metrics = {
        "clean_accuracy": _mean(clean, "cortex_resolved_accuracy"),
        "clean_coverage": _mean(clean, "cortex_coverage"),
        "clean_false_reinspect_rate": _mean(clean, "false_reinspect_rate"),
        "one_pair_neural_accuracy": _mean(one_pair_0, "neural_accuracy"),
        "one_pair_budget8_accuracy": _mean(
            one_pair_8, "cortex_resolved_accuracy"
        ),
        "one_pair_budget8_coverage": _mean(
            one_pair_8, "cortex_coverage"
        ),
        "one_pair_budget8_correction_rate": _mean(
            one_pair_8, "correction_rate"
        ),
        "one_pair_budget8_reinspect_precision": _mean(
            one_pair_8, "reinspect_precision"
        ),
        "one_pair_false_reinspect_rate": _mean(
            one_pair_8, "false_reinspect_rate"
        ),
        "two_pair_neural_accuracy": _mean(two_pair_0, "neural_accuracy"),
        "two_pair_budget12_accuracy": _mean(
            two_pair_12, "cortex_resolved_accuracy"
        ),
        "two_pair_budget12_coverage": _mean(
            two_pair_12, "cortex_coverage"
        ),
        "two_pair_budget12_effective_correct": _mean(
            two_pair_12, "effective_correct_fraction"
        ),
        "two_pair_budget12_correction_rate": _mean(
            two_pair_12, "correction_rate"
        ),
        "two_pair_false_reinspect_rate": _mean(
            two_pair_12, "false_reinspect_rate"
        ),
        "ambiguity_unresolved_fraction": _mean(
            ambiguity, "twin_unresolved_fraction"
        ),
        "ambiguity_non_twin_accuracy": _mean(
            ambiguity, "non_twin_resolved_accuracy"
        ),
        "ambiguity_mean_confidence": _mean(
            ambiguity, "twin_mean_confidence"
        ),
        "label_overlap_max": max(
            int(row["label_overlap"])
            for row in rows
        ),
    }

    expected_normal = (
        len(seeds)
        * len(FAMILY_EDGES)
        * len(CORRUPTED_PAIRS)
        * len(STRUCTURE_BUDGETS)
    )
    checks = {
        "protocol_exact": all(
            row.get("protocol") == PROTOCOL
            for row in rows
        ),
        "normal_row_count": len(normal) == expected_normal,
        "ambiguity_row_count": len(ambiguity) == len(seeds),
        "seeds_exact": sorted(
            {int(row["seed"]) for row in rows}
        ) == list(seeds),
        "label_namespaces_disjoint": metrics["label_overlap_max"] == 0,
        "clean_accuracy": metrics["clean_accuracy"] == 1.0,
        "clean_coverage": metrics["clean_coverage"] == 1.0,
        "clean_no_false_reinspection": (
            metrics["clean_false_reinspect_rate"] == 0.0
        ),
        "one_pair_neural_fixture": (
            metrics["one_pair_neural_accuracy"] == 0.6
        ),
        "one_pair_accuracy": (
            metrics["one_pair_budget8_accuracy"] >= 0.98
        ),
        "one_pair_coverage": (
            metrics["one_pair_budget8_coverage"] >= 0.95
        ),
        "one_pair_correction": (
            metrics["one_pair_budget8_correction_rate"] >= 0.95
        ),
        "one_pair_feedback_precision": (
            metrics["one_pair_budget8_reinspect_precision"] >= 0.95
        ),
        "one_pair_no_false_reinspection": (
            metrics["one_pair_false_reinspect_rate"] == 0.0
        ),
        "two_pair_neural_fixture": (
            metrics["two_pair_neural_accuracy"] == 0.2
        ),
        "two_pair_resolved_accuracy": (
            metrics["two_pair_budget12_accuracy"] >= 0.90
        ),
        "two_pair_coverage": (
            metrics["two_pair_budget12_coverage"] >= 0.90
        ),
        "two_pair_effective_correct": (
            metrics["two_pair_budget12_effective_correct"] >= 0.85
        ),
        "two_pair_correction": (
            metrics["two_pair_budget12_correction_rate"] >= 0.85
        ),
        "two_pair_no_false_reinspection": (
            metrics["two_pair_false_reinspect_rate"] == 0.0
        ),
        "symmetry_stays_unresolved": (
            metrics["ambiguity_unresolved_fraction"] == 1.0
        ),
        "non_symmetric_labels_resolve": (
            metrics["ambiguity_non_twin_accuracy"] == 1.0
        ),
        "symmetric_support_stays_half": (
            abs(metrics["ambiguity_mean_confidence"] - 0.5) <= 1.0e-9
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
        for line in args.input.read_text(encoding="utf-8").splitlines()
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
