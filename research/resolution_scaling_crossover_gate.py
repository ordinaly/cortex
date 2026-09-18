"""Acceptance gates for resolution-scaling-crossover-v1."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def evaluate(rows: list[dict]) -> dict[str, bool]:
    by_anchor: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        by_anchor[int(row["anchors"])].append(row)

    def median(anchors: int, key: str) -> float:
        return float(
            np.median(
                [
                    row[key]
                    for row in by_anchor[anchors]
                    if row[key] is not None
                ]
            )
        )

    sparse_fidelity = all(
        median(anchors, "sparse_mean_tv") <= 0.02
        and median(anchors, "sparse_max_tv") <= 0.10
        and median(
            anchors,
            "sparse_resolution_disagreement",
        )
        <= 0.10
        for anchors in by_anchor
    )

    exact_parity = all(
        row["dense_exact_max_prediction_error"]
        is None
        or (
            row["dense_exact_max_prediction_error"]
            <= 1.0e-10
            and row[
                "dense_exact_resolution_disagreement"
            ]
            == 0.0
        )
        for row in rows
    )

    return {
        "dense_sampled_exact_parity": exact_parity,
        "sparse_fidelity_all_scales": sparse_fidelity,
        "sampled_svd_benefit_36": (
            median(
                36,
                "sampled_speedup_vs_exact_svd",
            )
            >= 1.30
        ),
        "sparse_large_scale_speedup": (
            median(
                144,
                "sparse_speedup_vs_dense_sampled",
            )
            >= 1.15
        ),
        "sparse_large_scale_kernel_duty": (
            median(
                144,
                "sparse_kernel_row_refresh_fraction",
            )
            <= 0.35
        ),
        "sparse_large_scale_scan_duty": (
            median(
                144,
                "sparse_anchor_scan_fraction",
            )
            <= 0.60
        ),
        "sparse_large_scale_svd_duty": (
            median(
                144,
                "sparse_svd_fraction",
            )
            <= 0.10
        ),
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

    gates = evaluate(rows)
    payload = {
        "protocol": "resolution-scaling-crossover-v1",
        "rows": len(rows),
        "gates": gates,
        "passed": all(gates.values()),
    }
    print(json.dumps(payload, sort_keys=True))

    if not all(gates.values()):
        failed = [
            name
            for name, passed in gates.items()
            if not passed
        ]
        raise SystemExit(
            "failed scaling gates: "
            + ", ".join(failed)
        )


if __name__ == "__main__":
    main()
