"""Frozen validity gates for counterfactual-reasoning-v1."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


OFFICIAL_SEEDS = tuple(range(100, 120))
FAMILIES = (
    "chain",
    "fork",
    "collider",
    "diamond",
    "bridge",
    "disconnected",
    "random-dag",
)


def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def ratio(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in args.results.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise SystemExit("no campaign rows")

    proof_valid = sum(
        row["multi_hop"]["proof_valid"]
        + row["counterfactual"]["proof_valid"]
        for row in rows
    )
    proof_total = sum(
        row["multi_hop"]["proof_total"]
        + row["counterfactual"]["proof_total"]
        for row in rows
    )

    redundant = [
        row["counterfactual"]["redundant_path_rate"]
        for row in rows
        if row["counterfactual"]["redundant_path_rate"] is not None
    ]
    bridge_cut = [
        row["counterfactual"]["bridge_cut_rate"]
        for row in rows
        if row["counterfactual"]["bridge_cut_rate"] is not None
    ]
    missing = [
        row["missing_bridge"]["unresolved_rate"]
        for row in rows
        if row["missing_bridge"] is not None
    ]

    metrics = {
        "direct_resolved_precision": mean(
            row["direct"]["resolved_precision"]
            for row in rows
        ),
        "direct_resolved_coverage": mean(
            row["direct"]["resolved_coverage"]
            for row in rows
        ),
        "causal_edge_recall": mean(
            row["direct"]["causal_recall"]
            for row in rows
        ),
        "null_edge_recall": mean(
            row["direct"]["null_recall"]
            for row in rows
        ),
        "correlation_trap_rejection": mean(
            row["direct"]["trap_rejection"]
            for row in rows
        ),
        "multi_hop_accuracy": mean(
            row["multi_hop"]["accuracy"]
            for row in rows
            if row["multi_hop"]["counts"]["total"] > 0
        ),
        "multi_hop_coverage": mean(
            row["multi_hop"]["coverage"]
            for row in rows
            if row["multi_hop"]["counts"]["total"] > 0
        ),
        "multi_hop_resolved_accuracy": mean(
            row["multi_hop"]["resolved_accuracy"]
            for row in rows
            if row["multi_hop"]["counts"]["total"] > 0
        ),
        "direct_baseline_multi_hop_accuracy": mean(
            row["multi_hop"]["direct_edge_baseline_accuracy"]
            for row in rows
            if row["multi_hop"]["counts"]["total"] > 0
        ),
        "negative_query_accuracy": mean(
            row["negative"]["accuracy"]
            for row in rows
        ),
        "counterfactual_accuracy": mean(
            row["counterfactual"]["accuracy"]
            for row in rows
            if row["counterfactual"]["counts"]["total"] > 0
        ),
        "counterfactual_coverage": mean(
            row["counterfactual"]["coverage"]
            for row in rows
            if row["counterfactual"]["counts"]["total"] > 0
        ),
        "counterfactual_resolved_accuracy": mean(
            row["counterfactual"]["resolved_accuracy"]
            for row in rows
            if row["counterfactual"]["counts"]["total"] > 0
        ),
        "redundant_path_preservation": mean(redundant),
        "bridge_cut_success": mean(bridge_cut),
        "missing_bridge_unresolved": mean(missing),
        "proof_valid_rate": ratio(proof_valid, proof_total),
        "state_restoration_rate": mean(
            row["counterfactual"]["restoration_rate"]
            for row in rows
        ),
        "correlation_baseline_trap_accuracy": mean(
            row["direct"]["correlation_baseline_trap_accuracy"]
            for row in rows
        ),
    }

    checks = {
        "protocol_exact": all(
            row["protocol"] == "counterfactual-reasoning-v1"
            for row in rows
        ),
        "row_count": len(rows) == len(FAMILIES) * len(OFFICIAL_SEEDS),
        "official_seeds_exact": sorted(
            {row["seed"] for row in rows}
        ) == list(OFFICIAL_SEEDS),
        "families_exact": sorted(
            {row["family"] for row in rows}
        ) == sorted(FAMILIES),
        "direct_resolved_precision": (
            metrics["direct_resolved_precision"] >= 0.995
        ),
        "direct_resolved_coverage": (
            metrics["direct_resolved_coverage"] >= 0.98
        ),
        "causal_edge_recall": (
            metrics["causal_edge_recall"] >= 0.98
        ),
        "null_edge_recall": (
            metrics["null_edge_recall"] >= 0.98
        ),
        "correlation_trap_rejection": (
            metrics["correlation_trap_rejection"] >= 0.99
        ),
        "multi_hop_resolved_accuracy": (
            metrics["multi_hop_resolved_accuracy"] >= 0.995
        ),
        "multi_hop_coverage": (
            metrics["multi_hop_coverage"] >= 0.98
        ),
        "multi_hop_gain_over_direct": (
            metrics["multi_hop_accuracy"]
            - metrics["direct_baseline_multi_hop_accuracy"]
            >= 0.25
        ),
        "negative_query_accuracy": (
            metrics["negative_query_accuracy"] >= 0.995
        ),
        "counterfactual_resolved_accuracy": (
            metrics["counterfactual_resolved_accuracy"] >= 0.995
        ),
        "counterfactual_coverage": (
            metrics["counterfactual_coverage"] >= 0.98
        ),
        "redundant_path_preservation": (
            metrics["redundant_path_preservation"] >= 0.99
        ),
        "bridge_cut_success": (
            metrics["bridge_cut_success"] >= 0.99
        ),
        "missing_bridge_unresolved": (
            metrics["missing_bridge_unresolved"] >= 0.99
        ),
        "proof_valid_rate": (
            metrics["proof_valid_rate"] == 1.0
        ),
        "state_restoration_rate": (
            metrics["state_restoration_rate"] == 1.0
        ),
        "correlation_baseline_trap_accuracy": (
            metrics["correlation_baseline_trap_accuracy"] <= 0.10
        ),
    }

    payload = {
        "protocol": "counterfactual-reasoning-v1",
        "rows": len(rows),
        "metrics": metrics,
        "checks": checks,
        "passed": all(checks.values()),
    }
    print(json.dumps(payload, sort_keys=True))

    if not all(checks.values()):
        raise SystemExit(
            "failed gates: "
            + ", ".join(
                name
                for name, passed in checks.items()
                if not passed
            )
        )


if __name__ == "__main__":
    main()
