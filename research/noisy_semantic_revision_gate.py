"""Development/frozen gate helper for noisy-semantic-revision-v1."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean

from noisy_semantic_revision_campaign import (
    DEVELOPMENT_SEEDS,
    FAMILY_EDGES,
    OFFICIAL_SEEDS,
    PROTOCOL,
)


def _q95(values: list[float]) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return float(ordered[index])


def summarize(rows: list[dict], *, official: bool) -> dict:
    seeds = OFFICIAL_SEEDS if official else DEVELOPMENT_SEEDS
    metrics = {
        "initial_correct_rate": mean(int(r["initial_correct"]) for r in rows),
        "mean_neural_top_accuracy": mean(r["neural_top_accuracy"] for r in rows),
        "mean_cortex_top_accuracy": mean(r["cortex_top_accuracy"] for r in rows),
        "mean_neural_brier": mean(r["neural_brier"] for r in rows),
        "mean_cortex_brier": mean(r["cortex_brier"] for r in rows),
        "mean_posterior_ece": mean(r["posterior_ece"] for r in rows),
        "mean_trusted_ece": mean(r["trusted_ece"] for r in rows),
        "mean_trusted_coverage": mean(r["trusted_coverage"] for r in rows),
        "mean_trusted_accuracy": mean(r["trusted_accuracy"] for r in rows),
        "mean_false_certainty_rate": mean(r["false_certainty_rate"] for r in rows),
        "mean_uncertainty_interception_rate": mean(
            r["uncertainty_interception_rate"] for r in rows
        ),
        "cases_without_false_revision": mean(
            int(r["false_revision_events"] == 0) for r in rows
        ),
        "shift_success_rate": mean(int(r["shift_success"]) for r in rows),
        "shift_latency_p95": _q95([r["shift_latency"] for r in rows]),
        "recovery_success_rate": mean(int(r["recovery_success"]) for r in rows),
        "recovery_latency_p95": _q95([r["recovery_latency"] for r in rows]),
        "unaffected_preservation_rate": mean(
            int(r["unaffected_revision_events"] == 0) for r in rows
        ),
        "bounded_neural_history_rate": mean(
            int(r["bounded_neural_history"]) for r in rows
        ),
        "bounded_structural_history_rate": mean(
            int(r["bounded_structural_history"]) for r in rows
        ),
    }

    # Development gates are intentionally conservative. They become frozen
    # official gates only after the development campaign is reviewed.
    checks = {
        "protocol_exact": all(r["protocol"] == PROTOCOL for r in rows),
        "row_count": len(rows) == len(seeds) * len(FAMILY_EDGES),
        "seeds_exact": sorted({r["seed"] for r in rows}) == list(seeds),
        "initial_correct_rate": metrics["initial_correct_rate"] >= 0.95,
        "cortex_top_accuracy": (
            metrics["mean_cortex_top_accuracy"]
            >= metrics["mean_neural_top_accuracy"] + 0.03
        ),
        "cortex_brier": (
            metrics["mean_cortex_brier"]
            <= metrics["mean_neural_brier"]
        ),
        "posterior_ece": metrics["mean_posterior_ece"] <= 0.20,
        "trusted_accuracy": metrics["mean_trusted_accuracy"] >= 0.98,
        "trusted_coverage": metrics["mean_trusted_coverage"] >= 0.60,
        "false_certainty": metrics["mean_false_certainty_rate"] <= 0.02,
        "uncertainty_interception": (
            metrics["mean_uncertainty_interception_rate"] >= 0.95
        ),
        "false_revision_resistance": (
            metrics["cases_without_false_revision"] >= 0.95
        ),
        "shift_success": metrics["shift_success_rate"] >= 0.95,
        "shift_latency": metrics["shift_latency_p95"] <= 10,
        "recovery_success": metrics["recovery_success_rate"] >= 0.95,
        "recovery_latency": metrics["recovery_latency_p95"] <= 10,
        "locality": metrics["unaffected_preservation_rate"] >= 0.98,
        "bounded_neural_history": (
            metrics["bounded_neural_history_rate"] == 1.0
        ),
        "bounded_structural_history": (
            metrics["bounded_structural_history_rate"] == 1.0
        ),
    }

    diagnostics = {
        "locality_failures": [
            {
                "seed": r["seed"],
                "family": r["family"],
                "events": r["unaffected_revision_events"],
            }
            for r in rows
            if r["unaffected_revision_events"] != 0
        ],
        "false_revision_cases": [
            {
                "seed": r["seed"],
                "family": r["family"],
                "events": r["false_revision_events"],
            }
            for r in rows
            if r["false_revision_events"] != 0
        ],
    }

    return {
        "protocol": PROTOCOL,
        "mode": "official" if official else "development",
        "rows": len(rows),
        "metrics": metrics,
        "checks": checks,
        "diagnostics": diagnostics,
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
                name for name, passed in payload["checks"].items()
                if not passed
            )
        )


if __name__ == "__main__":
    main()
