"""Validation gates for kinship-composition-v1."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


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

    mean = lambda fn: sum(fn(row) for row in rows) / len(rows)

    checks = {
        "unseen_entities": max(row["entity_overlap"] for row in rows) == 0,
        "depth_is_ood": all(
            row["test_min_path_length"] > row["train_max_path_length"]
            for row in rows
        ),
        "cortex_accuracy": mean(
            lambda row: row["metrics"]["cortex"]["accuracy"]
        ) >= 0.98,
        "cortex_coverage": mean(
            lambda row: row["metrics"]["cortex"]["coverage"]
        ) >= 0.98,
        "exact_memory_does_not_generalize": mean(
            lambda row: row["metrics"]["sequence-memory"]["coverage"]
        ) <= 0.01,
        "missing_rule_is_unresolved": mean(
            lambda row: row["missing_rule"]["coverage"]
        ) <= 0.05,
    }
    payload = {
        "protocol": "kinship-composition-v1",
        "rows": len(rows),
        "checks": checks,
        "passed": all(checks.values()),
    }
    print(json.dumps(payload, sort_keys=True))
    if not all(checks.values()):
        raise SystemExit(
            "failed gates: "
            + ", ".join(name for name, ok in checks.items() if not ok)
        )


if __name__ == "__main__":
    main()
