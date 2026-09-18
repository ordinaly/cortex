"""Frozen success gates for hysteretic-local-resolution-v1."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def evaluate(rows: list[dict]) -> dict[str, bool]:
    def mean(phase: str, model: str, metric: str) -> float:
        return float(
            np.mean(
                [
                    row[phase][model][metric]
                    for row in rows
                ]
            )
        )

    reactive_switches = mean(
        "divergent",
        "reactive",
        "switches",
    )
    hysteretic_switches = mean(
        "divergent",
        "hysteretic",
        "switches",
    )
    reactive_nll = mean(
        "divergent",
        "reactive",
        "nll",
    )
    hysteretic_nll = mean(
        "divergent",
        "hysteretic",
        "nll",
    )
    local_nll = mean(
        "divergent",
        "local",
        "nll",
    )
    hysteretic_complexity = mean(
        "divergent",
        "hysteretic",
        "complexity_tail",
    )
    local_complexity = mean(
        "divergent",
        "local",
        "complexity_tail",
    )
    local_refined_fraction = mean(
        "divergent",
        "local",
        "refined_anchor_fraction_tail",
    )
    local_reconverged_coarse = mean(
        "reconverged",
        "local",
        "coarsest_anchor_fraction_tail",
    )

    return {
        "hysteresis_switch_reduction": (
            hysteretic_switches
            <= 0.50 * reactive_switches
        ),
        "hysteresis_predictive_fidelity": (
            hysteretic_nll <= reactive_nll + 0.02
        ),
        "local_predictive_fidelity": (
            local_nll <= hysteretic_nll + 0.04
        ),
        "local_complexity_reduction": (
            local_complexity
            <= 0.75 * hysteretic_complexity
        ),
        "local_refinement_sparsity": (
            local_refined_fraction <= 0.35
        ),
        "local_recoarsening": (
            local_reconverged_coarse >= 0.95
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
    print(
        json.dumps(
            {
                "protocol": (
                    "hysteretic-local-resolution-v1"
                ),
                "seeds": len(rows),
                "gates": gates,
                "passed": all(gates.values()),
            },
            sort_keys=True,
        )
    )
    if not all(gates.values()):
        failed = [
            name
            for name, passed in gates.items()
            if not passed
        ]
        raise SystemExit(
            "failed frozen gates: " + ", ".join(failed)
        )


if __name__ == "__main__":
    main()
