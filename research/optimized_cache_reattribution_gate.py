"""Validity gates for Cortex v1.17-R optimized cache re-attribution."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from optimized_cache_reattribution_campaign import (
    ANCHOR_CASES,
    ATTRIBUTION_GROUPS,
    DEVELOPMENT_SEEDS,
    OFFICIAL_SEEDS,
    PROTOCOL,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("--official", action="store_true")
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

    exact = all(
        float(row["max_candidate_error"]) <= 1.0e-12
        and float(row["max_local_error"]) <= 1.0e-12
        and float(row["max_outcome_count_error"]) <= 1.0e-12
        and float(row["max_feature_error"]) <= 1.0e-12
        and float(row["max_probability_error"]) <= 1.0e-12
        and float(row["max_distance_error"]) <= 1.0e-12
        and float(row["max_kernel_error"]) <= 1.0e-12
        and int(row["dirty_disagreements"]) == 0
        and int(row["structural_disagreements"]) == 0
        for row in rows
    )
    finite = all(
        np.isfinite(float(value))
        for row in rows
        for key, value in row.items()
        if (
            isinstance(value, (int, float))
            and key not in {"seed", "anchors", "steps"}
        )
    )
    accounting = all(
        0.80 <= float(row["accounted_fraction"]) <= 1.0
        for row in rows
    )
    shares = all(
        0.0 <= float(row[f"group_{group}_share"]) <= 1.0
        for row in rows
        for group in ATTRIBUTION_GROUPS
    )
    share_sum = all(
        sum(
            float(row[f"group_{group}_share"])
            for group in ATTRIBUTION_GROUPS
        ) <= 1.0 + 1.0e-9
        for row in rows
    )
    dirty = all(
        0.0 <= float(row["dirty_event_fraction"]) <= 1.0
        and 0.0 <= float(row["mean_dirty_row_fraction"]) <= 1.0
        for row in rows
    )

    checks = {
        "protocol_exact": all(
            row["protocol"] == PROTOCOL
            for row in rows
        ),
        "row_count": (
            len(rows)
            == len(ANCHOR_CASES) * len(seeds)
        ),
        "seed_set": sorted(
            {int(row["seed"]) for row in rows}
        ) == sorted(seeds),
        "anchor_cases": sorted(
            {int(row["anchors"]) for row in rows}
        ) == sorted(ANCHOR_CASES),
        "exact_cache_parity": exact,
        "finite_timings": finite,
        "accounting_coverage": accounting,
        "valid_group_shares": shares,
        "group_share_sum": share_sum,
        "valid_dirty_metrics": dirty,
    }

    payload = {
        "protocol": PROTOCOL,
        "mode": "official" if args.official else "development",
        "rows": len(rows),
        "checks": checks,
        "passed": all(checks.values()),
    }
    print(json.dumps(payload, sort_keys=True))

    if not payload["passed"]:
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
