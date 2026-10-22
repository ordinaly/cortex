#!/usr/bin/env python3
"""State-dependent temporal guard tuning for Closed Articulation Loop v0.2.

A static foreground-policy sweep failed the cross-stage confirmation gate:
safer B transitions cost too much D coverage. This campaign keeps the stable
v0.1 foreground regime policy and tunes only an execution guard conditioned on
learner-visible out-of-order evidence, together with the independently useful
recurrence threshold found in the confirmation panel.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from itertools import product
import json
from math import sqrt
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))

from cortex.closed_articulation_loop import ClosedLoopPolicy
from cortex.regime_plasticity import RegimePlasticityPolicy
from closed_articulation_loop_tuning import (
    _eval_b,
    _evaluate_all,
    _stage_worlds,
)

PROTOCOL = "closed-articulation-loop-v0.2-temporal-guard"


def _candidate(label, policy, regime):
    return {"label": label, "policy": policy, "regime": regime}


def _serial(candidate):
    return {
        "label": candidate["label"],
        "policy": asdict(candidate["policy"]),
        "regime": asdict(candidate["regime"]),
    }


def _dominates(left, right):
    a = left["metrics"]["transition"]
    b = right["metrics"]["transition"]
    no_worse = (
        a["wrong_rate"] <= b["wrong_rate"]
        and a["correct_rate"] >= b["correct_rate"]
        and left["metrics"]["all"]["accuracy"]
        >= right["metrics"]["all"]["accuracy"]
    )
    strict = (
        a["wrong_rate"] < b["wrong_rate"]
        or a["correct_rate"] > b["correct_rate"]
        or left["metrics"]["all"]["accuracy"]
        > right["metrics"]["all"]["accuracy"]
    )
    return no_worse and strict


def _frontier(rows):
    return [
        row for row in rows
        if not any(_dominates(other, row) for other in rows if other is not row)
    ]


def _distance(frontier, row):
    specs = (
        ("wrong_rate", "min"),
        ("correct_rate", "max"),
    )
    total = 0.0
    for key, direction in specs:
        values = [
            float(item["metrics"]["transition"][key])
            for item in frontier
        ]
        lo, hi = min(values), max(values)
        span = hi - lo
        if span <= 1.0e-12:
            d = 0.0
        elif direction == "max":
            d = (hi - float(row["metrics"]["transition"][key])) / span
        else:
            d = (float(row["metrics"]["transition"][key]) - lo) / span
        total += d * d
    return sqrt(total / len(specs))


def _promotion_feasible(metrics, baseline):
    a = metrics["A"]
    b = metrics["B"]
    c = metrics["C"]
    d = metrics["D"]
    return (
        a["target_trap"]["selective_risk"] <= 1.0e-12
        and a["target_trap"]["coverage"]
        >= baseline["A"]["target_trap"]["coverage"] - 0.02
        and a["target"]["selective_risk"] <= 0.01
        and b["transition"]["wrong_rate"]
        <= 0.75 * baseline["B"]["transition"]["wrong_rate"]
        and b["transition"]["correct_rate"]
        >= baseline["B"]["transition"]["correct_rate"]
        and b["settled_b"]["selective_risk"] <= 0.03
        and b["settled_c"]["selective_risk"] <= 0.03
        and b["settled_recurrence"]["selective_risk"] <= 0.03
        and c["local_relation"]["coverage"] >= 0.999999
        and c["local_relation"]["resolved_accuracy"] >= 0.999999
        and c["worlds_with_multi_subject_candidate"] == c["worlds"]
        and d["hard_failure_rate"] <= 1.0e-12
        and d["query"]["selective_risk"] <= 1.0e-12
        and d["query"]["coverage"] >= baseline["D"]["query"]["coverage"] - 0.02
    )


def run(tuning_worlds: int, validation_worlds: int):
    base_policy = ClosedLoopPolicy()
    regime = RegimePlasticityPolicy()
    recurrence_policy = replace(
        base_policy,
        recurrence_min_history=3.0,
        recurrence_confirmations=3,
    )

    tuning_b = _stage_worlds(
        "closed-loop-v0.2-guard-tuning",
        tuning_worlds,
    )["B"]

    rows = []
    for decay, threshold in product(
        (0.25, 0.50, 0.65, 0.75, 0.85, 0.90),
        (0.50, 0.75, 1.00, 1.25, 1.50, 2.00),
    ):
        policy = replace(
            recurrence_policy,
            temporal_disorder_decay=decay,
            temporal_guard_threshold=threshold,
        )
        metrics = _eval_b(tuning_b, policy, regime)
        rows.append({
            "candidate": _candidate(
                f"guard-d{decay:.2f}-t{threshold:.2f}",
                policy,
                regime,
            ),
            "metrics": metrics,
        })

    frontier = _frontier(rows)
    for row in frontier:
        row["ideal_distance"] = _distance(frontier, row)
    frontier.sort(key=lambda row: row["ideal_distance"])
    finalists = frontier[:3]

    validation = _stage_worlds(
        "closed-loop-v0.2-guard-validation",
        validation_worlds,
    )
    baseline_candidate = _candidate(
        "v0.1-baseline",
        base_policy,
        regime,
    )
    recurrence_candidate = _candidate(
        "recurrence-only",
        recurrence_policy,
        regime,
    )
    baseline_metrics = _evaluate_all(validation, baseline_candidate)

    panel = []
    for candidate in (
        baseline_candidate,
        recurrence_candidate,
        *(row["candidate"] for row in finalists),
    ):
        metrics = _evaluate_all(validation, candidate)
        feasible = (
            candidate["label"] != "v0.1-baseline"
            and _promotion_feasible(metrics, baseline_metrics)
        )
        panel.append({
            "candidate": candidate,
            "metrics": metrics,
            "feasible": feasible,
            "delta": {
                "a_trap_coverage": (
                    metrics["A"]["target_trap"]["coverage"]
                    - baseline_metrics["A"]["target_trap"]["coverage"]
                ),
                "b_transition_wrong_rate": (
                    metrics["B"]["transition"]["wrong_rate"]
                    - baseline_metrics["B"]["transition"]["wrong_rate"]
                ),
                "b_transition_correct_rate": (
                    metrics["B"]["transition"]["correct_rate"]
                    - baseline_metrics["B"]["transition"]["correct_rate"]
                ),
                "d_coverage": (
                    metrics["D"]["query"]["coverage"]
                    - baseline_metrics["D"]["query"]["coverage"]
                ),
            },
        })

    feasible = [row for row in panel if row["feasible"]]
    if feasible:
        feasible.sort(
            key=lambda row: (
                row["metrics"]["B"]["transition"]["wrong_rate"],
                -row["metrics"]["B"]["transition"]["correct_rate"],
                -row["metrics"]["A"]["target_trap"]["coverage"],
                -row["metrics"]["D"]["query"]["coverage"],
            )
        )
        selected = feasible[0]
    else:
        selected = panel[0]

    return {
        "protocol": PROTOCOL,
        "mode": "state-dependent-guard-tuning-and-validation",
        "tuning_worlds": tuning_worlds,
        "validation_worlds": validation_worlds,
        "tuning_candidates": len(rows),
        "pareto_candidates": len(frontier),
        "tuning_finalists": [
            {
                "candidate": _serial(row["candidate"]),
                "ideal_distance": row["ideal_distance"],
                "transition": row["metrics"]["transition"],
                "all": row["metrics"]["all"],
            }
            for row in finalists
        ],
        "validation_panel": [
            {
                "candidate": _serial(row["candidate"]),
                "feasible": row["feasible"],
                "delta": row["delta"],
                "metrics": row["metrics"],
            }
            for row in panel
        ],
        "selected": {
            "candidate": _serial(selected["candidate"]),
            "feasible": selected["feasible"],
            "delta": selected["delta"],
            "metrics": selected["metrics"],
        },
        "official_seeds_used": False,
        "owr_e_used": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tuning-worlds", type=int, default=8)
    parser.add_argument("--validation-worlds", type=int, default=20)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run(args.tuning_worlds, args.validation_worlds)
    rendered = json.dumps(report, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
