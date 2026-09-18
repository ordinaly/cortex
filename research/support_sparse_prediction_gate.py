"""Validation and performance hypotheses for support-sparse-prediction-v1."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def evaluate(rows: list[dict]) -> dict:
    by_anchor: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        by_anchor[int(row["anchors"])].append(row)

    def median(anchors: int, key: str) -> float:
        return float(
            np.median(
                [row[key] for row in by_anchor[anchors]]
            )
        )

    fused_parity = all(
        row["fused_max_candidate_error"] <= 1.0e-10
        and row["fused_max_local_error"] <= 1.0e-10
        and row["fused_resolution_disagreement"] == 0.0
        for row in rows
    )

    support_mass_bound = all(
        row["support_max_dropped_mass"]
        <= (1.0 - row["retained_mass_target"])
        + 1.0e-12
        for row in rows
    )

    support_fidelity = all(
        median(anchors, "support_mean_local_tv") <= 0.01
        and median(anchors, "support_max_local_tv") <= 0.05
        and median(anchors, "support_mean_candidate_tv") <= 0.01
        and median(anchors, "support_max_candidate_tv") <= 0.05
        and median(
            anchors,
            "support_resolution_disagreement",
        )
        <= 0.05
        for anchors in by_anchor
    )

    predictive_fidelity = all(
        median(anchors, "support_divergent_nll")
        <= median(anchors, "fused_divergent_nll")
        + 0.02
        for anchors in by_anchor
    )

    recoarsening = all(
        median(
            anchors,
            "support_recoarsened_fraction",
        )
        >= 0.95
        for anchors in by_anchor
    )

    validity = {
        "fused_exact_parity": fused_parity,
        "support_mass_bound": support_mass_bound,
        "support_prediction_fidelity": support_fidelity,
        "support_predictive_nll": predictive_fidelity,
        "support_recoarsening": recoarsening,
    }

    hypotheses = {
        "support_fraction_144": (
            median(
                144,
                "support_mean_support_fraction",
            )
            <= 0.50
        ),
        "support_fraction_288": (
            median(
                288,
                "support_mean_support_fraction",
            )
            <= 0.40
        ),
        "support_speedup_vs_fused_144": (
            median(
                144,
                "support_speedup_vs_fused",
            )
            >= 1.15
        ),
        "support_speedup_vs_fused_288": (
            median(
                288,
                "support_speedup_vs_fused",
            )
            >= 1.25
        ),
        "overall_speedup_vs_legacy_144": (
            median(
                144,
                "support_speedup_vs_legacy",
            )
            >= 1.25
        ),
        "overall_speedup_vs_legacy_288": (
            median(
                288,
                "support_speedup_vs_legacy",
            )
            >= 1.40
        ),
    }

    return {
        "validity": validity,
        "hypotheses": hypotheses,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
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

    evaluation = evaluate(rows)
    valid = all(
        evaluation["validity"].values()
    )
    hypotheses_passed = all(
        evaluation["hypotheses"].values()
    )

    payload = {
        "protocol": "support-sparse-prediction-v1",
        "rows": len(rows),
        **evaluation,
        "valid": valid,
        "all_performance_hypotheses_passed": (
            hypotheses_passed
        ),
        "failed_hypotheses": [
            name
            for name, passed
            in evaluation["hypotheses"].items()
            if not passed
        ],
    }
    print(json.dumps(payload, sort_keys=True))

    if not valid:
        failed = [
            name
            for name, passed
            in evaluation["validity"].items()
            if not passed
        ]
        raise SystemExit(
            "invalid support-sparse campaign: "
            + ", ".join(failed)
        )


if __name__ == "__main__":
    main()
