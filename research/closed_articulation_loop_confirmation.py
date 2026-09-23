#!/usr/bin/env python3
"""Confirmation panel after v0.2 factor-sweep holdout.

The first v0.2 holdout showed that aggressive A-specific and memory-budget
choices did not generalize cleanly, while the temporal candidate did. This
script tests a small frozen ablation panel on a third development material
before any default is promoted.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))

from cortex.closed_articulation_loop import ClosedLoopPolicy
from cortex.regime_plasticity import RegimePlasticityPolicy
from closed_articulation_loop_tuning import (
    _evaluate_all,
    _objectives,
    _stage_worlds,
)

PROTOCOL = "closed-articulation-loop-v0.2-confirmation"


def _candidate(label, policy, regime):
    return {"label": label, "policy": policy, "regime": regime}


def _serial(candidate):
    return {
        "label": candidate["label"],
        "policy": asdict(candidate["policy"]),
        "regime": asdict(candidate["regime"]),
    }


def _panel():
    base_policy = ClosedLoopPolicy()
    base_regime = RegimePlasticityPolicy()
    temporal_t3 = replace(
        base_regime,
        fast_decay=0.65,
        pressure_decay=0.85,
        suspicion_ratio=0.03,
        confirmation_steps=3,
    )
    temporal_t5 = replace(temporal_t3, confirmation_steps=5)
    recurrence = replace(
        base_policy,
        recurrence_min_history=3.0,
        recurrence_confirmations=3,
    )
    full_policy = ClosedLoopPolicy(
        active_budget=32,
        latent_budget=96,
        utility_decay=0.995,
        dependence_min_overlap=4,
        dependence_agreement=0.85,
        episode_min_completed=2,
        episode_separation_ratio=1.5,
        recurrence_min_history=3.0,
        recurrence_confirmations=3,
        residual_window=6,
        compound_min_events=3,
        late_event_tolerance=0,
    )
    return (
        _candidate("v0.1-baseline", base_policy, base_regime),
        _candidate("temporal-core-t3", base_policy, temporal_t3),
        _candidate("recurrence-only", recurrence, base_regime),
        _candidate("temporal-plus-recurrence-t3", recurrence, temporal_t3),
        _candidate("temporal-plus-recurrence-t5", recurrence, temporal_t5),
        _candidate("aggressive-full-v0.2-first-pass", full_policy, temporal_t3),
    )


def _confirmation_feasible(metrics, baseline):
    a = metrics["A"]
    b = metrics["B"]
    c = metrics["C"]
    d = metrics["D"]
    bb = baseline["B"]
    return (
        a["target_trap"]["selective_risk"] <= 1.0e-12
        and a["target_trap"]["coverage"]
        >= baseline["A"]["target_trap"]["coverage"] - 0.05
        and a["target"]["selective_risk"] <= 0.01
        and b["transition"]["wrong_rate"]
        <= 0.75 * bb["transition"]["wrong_rate"]
        and b["transition"]["correct_rate"]
        >= bb["transition"]["correct_rate"]
        and b["settled_b"]["selective_risk"] <= 0.03
        and b["settled_c"]["selective_risk"] <= 0.03
        and b["settled_recurrence"]["selective_risk"] <= 0.03
        and c["local_relation"]["coverage"] >= 0.999999
        and c["local_relation"]["resolved_accuracy"] >= 0.999999
        and c["worlds_with_multi_subject_candidate"] == c["worlds"]
        and d["hard_failure_rate"] <= 1.0e-12
        and d["query"]["selective_risk"] <= 1.0e-12
        and d["query"]["coverage"] >= baseline["D"]["query"]["coverage"] - 0.03
    )


def run(worlds_per_stage: int):
    worlds = _stage_worlds(
        "closed-loop-v0.2-confirmation",
        worlds_per_stage,
    )
    panel = _panel()
    rows = []
    for candidate in panel:
        metrics = _evaluate_all(worlds, candidate)
        rows.append({
            "candidate": candidate,
            "metrics": metrics,
            "objectives": _objectives(metrics),
        })

    baseline = rows[0]["metrics"]
    for row in rows:
        row["feasible"] = (
            row["candidate"]["label"] != "v0.1-baseline"
            and _confirmation_feasible(row["metrics"], baseline)
        )
        row["delta"] = {
            "a_trap_coverage": (
                row["metrics"]["A"]["target_trap"]["coverage"]
                - baseline["A"]["target_trap"]["coverage"]
            ),
            "b_transition_wrong_rate": (
                row["metrics"]["B"]["transition"]["wrong_rate"]
                - baseline["B"]["transition"]["wrong_rate"]
            ),
            "b_transition_correct_rate": (
                row["metrics"]["B"]["transition"]["correct_rate"]
                - baseline["B"]["transition"]["correct_rate"]
            ),
            "d_coverage": (
                row["metrics"]["D"]["query"]["coverage"]
                - baseline["D"]["query"]["coverage"]
            ),
        }

    feasible = [row for row in rows if row["feasible"]]
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
        selected = rows[0]

    return {
        "protocol": PROTOCOL,
        "mode": "third-development-material-confirmation",
        "worlds_per_stage": worlds_per_stage,
        "selection_rule": {
            "max_a_trap_coverage_regression": 0.05,
            "max_b_transition_wrong_ratio": 0.75,
            "min_b_transition_correct_ratio": 1.0,
            "max_d_coverage_regression": 0.03,
            "settled_b_risk_ceiling": 0.03,
        },
        "rows": [
            {
                "candidate": _serial(row["candidate"]),
                "feasible": row["feasible"],
                "objectives": row["objectives"],
                "delta": row["delta"],
                "metrics": row["metrics"],
            }
            for row in rows
        ],
        "selected": {
            "candidate": _serial(selected["candidate"]),
            "feasible": selected["feasible"],
            "objectives": selected["objectives"],
            "delta": selected["delta"],
            "metrics": selected["metrics"],
        },
        "official_seeds_used": False,
        "owr_e_used": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worlds", type=int, default=20)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run(args.worlds)
    rendered = json.dumps(report, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
