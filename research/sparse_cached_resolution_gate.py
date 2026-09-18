"""Acceptance gates for Cortex sparse-cached-resolution-v1."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def evaluate(rows: list[dict]) -> dict[str, bool]:
    def mean(key: str) -> float:
        return float(
            np.mean([row[key] for row in rows])
        )

    def maximum(key: str) -> float:
        return float(
            np.max([row[key] for row in rows])
        )

    return {
        "exact_prediction_parity": (
            maximum("exact_max_prediction_error")
            <= 1.0e-10
        ),
        "exact_resolution_parity": (
            maximum(
                "exact_resolution_disagreement"
            )
            == 0.0
        ),
        "exact_complexity_parity": (
            maximum("exact_max_complexity_error")
            <= 1.0e-9
        ),
        "exact_cache_speedup": (
            mean("cached_speedup_vs_reference")
            >= 1.20
        ),
        "sparse_prediction_fidelity": (
            mean("sparse_mean_tv") <= 0.01
            and maximum("sparse_max_tv") <= 0.05
        ),
        "sparse_resolution_fidelity": (
            mean(
                "sparse_resolution_disagreement"
            )
            <= 0.10
        ),
        "sparse_predictive_fidelity": (
            mean("sparse_divergent_nll")
            <= mean("cached_divergent_nll") + 0.03
        ),
        "sparse_recoarsening": (
            mean("sparse_recoarsened_fraction")
            >= 0.95
        ),
        "sparse_cache_activity": (
            mean(
                "sparse_kernel_row_refresh_fraction"
            )
            <= 0.50
        ),
        "sparse_svd_activity": (
            mean("sparse_svd_fraction")
            <= 0.10
        ),
        "sparse_scan_activity": (
            mean("sparse_anchor_scan_fraction")
            <= 0.60
        ),
        "sparse_speedup": (
            mean("sparse_speedup_vs_reference")
            >= 1.50
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
        "protocol": "sparse-cached-resolution-v1",
        "seeds": len(rows),
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
            "failed optimization gates: "
            + ", ".join(failed)
        )


if __name__ == "__main__":
    main()
