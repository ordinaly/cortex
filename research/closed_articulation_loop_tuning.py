#!/usr/bin/env python3
"""Closed Articulation Loop v0.2 development tuning campaign.

The campaign deliberately separates tuning and validation materials. It tunes
mechanism groups independently, recombines only the best local candidates, and
selects a Pareto knee on fresh development worlds. OWR-E and official seed
material are never used.
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
from closed_articulation_loop_campaign import (
    _run_a,
    _run_b,
    _run_c,
    _run_d,
)
from open_world_robustness_world import build_suite

PROTOCOL = "closed-articulation-loop-v0.2-tuning"


def _pool_metric(results, path):
    selected = []
    for result in results:
        value = result
        for key in path:
            value = value[key]
        selected.append(value)
    queries = sum(row["queries"] for row in selected)
    resolved = sum(row["resolved"] for row in selected)
    correct = sum(row["correct"] for row in selected)
    wrong = sum(row["wrong"] for row in selected)
    return {
        "queries": queries,
        "resolved": resolved,
        "correct": correct,
        "wrong": wrong,
        "coverage": resolved / max(1, queries),
        "resolved_accuracy": correct / max(1, resolved),
        "accuracy": correct / max(1, queries),
        "selective_risk": wrong / max(1, resolved),
        "wrong_rate": wrong / max(1, queries),
    }


def _stage_worlds(material: str, count: int):
    return {
        stage: build_suite(
            material=f"{material}-{stage}",
            count=count,
            stages=(stage,),
            scale="development",
        )
        for stage in ("A", "B", "C", "D")
    }


def _eval_a(worlds, policy, regime):
    rows = [_run_a(world, policy, regime) for world in worlds]
    return {
        part: _pool_metric(rows, ("metrics", part))
        for part in (
            "all",
            "calibration",
            "target",
            "target_trap",
            "target_control",
        )
    }


def _eval_b(worlds, policy, regime):
    rows = [_run_b(world, policy, regime) for world in worlds]
    metrics = {
        part: _pool_metric(rows, ("metrics", part))
        for part in (
            "all",
            "pre_shift",
            "near_b",
            "settled_b",
            "near_c",
            "settled_c",
            "near_recurrence",
            "settled_recurrence",
        )
    }
    near = [
        metrics["near_b"],
        metrics["near_c"],
        metrics["near_recurrence"],
    ]
    q = sum(row["queries"] for row in near)
    wrong = sum(row["wrong"] for row in near)
    correct = sum(row["correct"] for row in near)
    metrics["transition"] = {
        "queries": q,
        "wrong": wrong,
        "correct": correct,
        "wrong_rate": wrong / max(1, q),
        "correct_rate": correct / max(1, q),
        "coverage": (wrong + correct) / max(1, q),
    }
    return metrics


def _eval_c(worlds, policy, regime):
    rows = [_run_c(world, policy, regime) for world in worlds]
    return {
        "local_relation": _pool_metric(rows, ("local_relation",)),
        "worlds": len(rows),
        "worlds_with_compound_candidate": sum(
            row["compound_candidates"] > 0 for row in rows
        ),
        "worlds_with_multi_subject_candidate": sum(
            row["multi_subject_candidates"] > 0 for row in rows
        ),
        "compound_candidates": sum(row["compound_candidates"] for row in rows),
        "multi_subject_candidates": sum(
            row["multi_subject_candidates"] for row in rows
        ),
        "bubble_candidates": sum(row["bubble_candidates"] for row in rows),
    }


def _eval_d(worlds, policy, regime):
    rows = [_run_d(world, policy, regime) for world in worlds]
    metric = _pool_metric(rows, ("metrics",))
    return {
        "query": metric,
        "hard_failure_rate": sum(row["hard_failure"] for row in rows)
        / max(1, len(rows)),
        "mean_stored_basins": sum(row["stored_basins"] for row in rows)
        / max(1, len(rows)),
        "mean_coarsened_basins": sum(row["coarsened_basins"] for row in rows)
        / max(1, len(rows)),
        "budget": policy.total_budget,
    }


def _candidate(policy, regime, *, label):
    return {
        "label": label,
        "policy": policy,
        "regime": regime,
    }


def _serial(candidate):
    return {
        "label": candidate["label"],
        "policy": asdict(candidate["policy"]),
        "regime": asdict(candidate["regime"]),
    }


def _top_evidence(worlds, baseline_policy, baseline_regime, keep=2):
    scored = []
    for overlap, agreement, completed, separation in product(
        (4, 6, 8),
        (0.85, 0.875, 0.90),
        (2, 3),
        (1.5, 2.0),
    ):
        policy = replace(
            baseline_policy,
            dependence_min_overlap=overlap,
            dependence_agreement=agreement,
            episode_min_completed=completed,
            episode_separation_ratio=separation,
        )
        metrics = _eval_a(worlds, policy, baseline_regime)
        trap = metrics["target_trap"]
        control = metrics["target_control"]
        calibration = metrics["calibration"]
        feasible = (
            trap["selective_risk"] <= 1.0e-12
            and control["selective_risk"] <= 1.0e-12
            and calibration["selective_risk"] <= 0.05
        )
        scored.append({
            "candidate": _candidate(policy, baseline_regime, label="evidence"),
            "metrics": metrics,
            "feasible": feasible,
            "rank": (
                int(feasible),
                trap["coverage"],
                control["coverage"],
                -calibration["selective_risk"],
                calibration["coverage"],
            ),
        })
    scored.sort(key=lambda row: row["rank"], reverse=True)
    return scored[:keep], scored


def _top_temporal_core(worlds, baseline_policy, baseline_regime, keep=3):
    scored = []
    for fast, pressure, suspicion, confirmation in product(
        (0.35, 0.50, 0.65),
        (0.85, 0.90),
        (0.03, 0.05, 0.08),
        (3, 5),
    ):
        regime = replace(
            baseline_regime,
            fast_decay=fast,
            pressure_decay=pressure,
            suspicion_ratio=suspicion,
            confirmation_steps=confirmation,
        )
        metrics = _eval_b(worlds, baseline_policy, regime)
        transition = metrics["transition"]
        settled_risk = max(
            metrics["settled_b"]["selective_risk"],
            metrics["settled_c"]["selective_risk"],
            metrics["settled_recurrence"]["selective_risk"],
        )
        feasible = (
            settled_risk <= 0.03
            and metrics["pre_shift"]["selective_risk"] <= 0.03
            and metrics["all"]["coverage"] >= 0.30
        )
        scored.append({
            "candidate": _candidate(baseline_policy, regime, label="temporal-core"),
            "metrics": metrics,
            "feasible": feasible,
            "rank": (
                int(feasible),
                -transition["wrong_rate"],
                transition["correct_rate"],
                metrics["all"]["accuracy"],
                metrics["all"]["coverage"],
            ),
        })
    scored.sort(key=lambda row: row["rank"], reverse=True)
    return scored[:keep], scored


def _top_recurrence(worlds, baseline_policy, temporal_rows, keep=2):
    scored = []
    for temporal in temporal_rows:
        regime = temporal["candidate"]["regime"]
        for mass, confirmations in product((3.0, 5.0, 8.0), (2, 3, 4)):
            policy = replace(
                baseline_policy,
                recurrence_min_history=mass,
                recurrence_confirmations=confirmations,
            )
            metrics = _eval_b(worlds, policy, regime)
            transition = metrics["transition"]
            settled_risk = max(
                metrics["settled_b"]["selective_risk"],
                metrics["settled_c"]["selective_risk"],
                metrics["settled_recurrence"]["selective_risk"],
            )
            feasible = settled_risk <= 0.03 and metrics["all"]["coverage"] >= 0.30
            scored.append({
                "candidate": _candidate(policy, regime, label="temporal"),
                "metrics": metrics,
                "feasible": feasible,
                "rank": (
                    int(feasible),
                    -transition["wrong_rate"],
                    transition["correct_rate"],
                    metrics["near_recurrence"]["accuracy"],
                    metrics["all"]["accuracy"],
                ),
            })
    scored.sort(key=lambda row: row["rank"], reverse=True)
    return scored[:keep], scored


def _top_residual(worlds, baseline_policy, baseline_regime, keep=2):
    scored = []
    for window, min_events in product((3, 4, 6, 8, 10), (2, 3, 4, 5)):
        policy = replace(
            baseline_policy,
            residual_window=window,
            compound_min_events=min_events,
        )
        metrics = _eval_c(worlds, policy, baseline_regime)
        local = metrics["local_relation"]
        full_candidates = (
            metrics["worlds_with_multi_subject_candidate"] == metrics["worlds"]
        )
        feasible = (
            local["coverage"] >= 0.999999
            and local["resolved_accuracy"] >= 0.999999
            and full_candidates
        )
        excess = metrics["multi_subject_candidates"] - metrics["worlds"]
        scored.append({
            "candidate": _candidate(policy, baseline_regime, label="residual"),
            "metrics": metrics,
            "feasible": feasible,
            "rank": (
                int(feasible),
                -abs(excess),
                -metrics["bubble_candidates"],
                -window,
            ),
        })
    scored.sort(key=lambda row: row["rank"], reverse=True)
    return scored[:keep], scored


def _pareto_memory(rows):
    frontier = []
    for row in rows:
        a = row["metrics"]
        dominated = False
        for other in rows:
            if other is row:
                continue
            b = other["metrics"]
            no_worse = (
                b["query"]["accuracy"] >= a["query"]["accuracy"]
                and b["query"]["coverage"] >= a["query"]["coverage"]
                and b["budget"] <= a["budget"]
            )
            better = (
                b["query"]["accuracy"] > a["query"]["accuracy"]
                or b["query"]["coverage"] > a["query"]["coverage"]
                or b["budget"] < a["budget"]
            )
            if no_worse and better:
                dominated = True
                break
        if not dominated:
            frontier.append(row)
    return frontier


def _top_memory(worlds, baseline_policy, baseline_regime, keep=2):
    scored = []
    for total, active_fraction, decay in product(
        (128, 192, 256, 384),
        (0.25, 0.50),
        (0.995, 0.997, 0.999),
    ):
        active = max(16, int(round(total * active_fraction)))
        latent = total - active
        policy = replace(
            baseline_policy,
            active_budget=active,
            latent_budget=latent,
            utility_decay=decay,
        )
        metrics = _eval_d(worlds, policy, baseline_regime)
        feasible = (
            metrics["hard_failure_rate"] <= 1.0e-12
            and metrics["query"]["selective_risk"] <= 1.0e-12
        )
        efficiency = metrics["query"]["coverage"] / sqrt(total)
        scored.append({
            "candidate": _candidate(policy, baseline_regime, label="memory"),
            "metrics": metrics,
            "feasible": feasible,
            "efficiency": efficiency,
        })
    feasible = [row for row in scored if row["feasible"]]
    frontier = _pareto_memory(feasible)
    frontier.sort(
        key=lambda row: (
            row["metrics"]["query"]["coverage"],
            row["efficiency"],
            -row["metrics"]["budget"],
        ),
        reverse=True,
    )
    if not frontier:
        scored.sort(key=lambda row: row["efficiency"], reverse=True)
        return scored[:keep], scored
    highest_coverage = frontier[0]
    highest_efficiency = max(frontier, key=lambda row: row["efficiency"])
    selected = [highest_coverage]
    if highest_efficiency is not highest_coverage:
        selected.append(highest_efficiency)
    for row in frontier:
        if row not in selected:
            selected.append(row)
        if len(selected) >= keep:
            break
    return selected[:keep], scored


def _merge_candidates(evidence, temporal, residual, memory, index):
    ep = evidence["candidate"]["policy"]
    tp = temporal["candidate"]["policy"]
    rp = residual["candidate"]["policy"]
    mp = memory["candidate"]["policy"]
    policy = ClosedLoopPolicy(
        active_budget=mp.active_budget,
        latent_budget=mp.latent_budget,
        utility_decay=mp.utility_decay,
        dependence_min_overlap=ep.dependence_min_overlap,
        dependence_agreement=ep.dependence_agreement,
        episode_min_completed=ep.episode_min_completed,
        episode_separation_ratio=ep.episode_separation_ratio,
        recurrence_min_history=tp.recurrence_min_history,
        recurrence_confirmations=tp.recurrence_confirmations,
        residual_window=rp.residual_window,
        compound_min_events=rp.compound_min_events,
        late_event_tolerance=0,
    )
    return _candidate(
        policy,
        temporal["candidate"]["regime"],
        label=f"combined-{index}",
    )


def _evaluate_all(worlds, candidate):
    p = candidate["policy"]
    r = candidate["regime"]
    return {
        "A": _eval_a(worlds["A"], p, r),
        "B": _eval_b(worlds["B"], p, r),
        "C": _eval_c(worlds["C"], p, r),
        "D": _eval_d(worlds["D"], p, r),
    }


def _objectives(metrics):
    return {
        "a_trap_coverage": metrics["A"]["target_trap"]["coverage"],
        "b_transition_wrong": metrics["B"]["transition"]["wrong_rate"],
        "b_transition_correct": metrics["B"]["transition"]["correct_rate"],
        "d_coverage": metrics["D"]["query"]["coverage"],
        "budget": metrics["D"]["budget"],
    }


def _feasible(metrics):
    return (
        metrics["A"]["target_trap"]["selective_risk"] <= 1.0e-12
        and metrics["A"]["target"]["selective_risk"] <= 0.01
        and metrics["C"]["local_relation"]["coverage"] >= 0.999999
        and metrics["C"]["local_relation"]["resolved_accuracy"] >= 0.999999
        and metrics["C"]["worlds_with_multi_subject_candidate"]
        == metrics["C"]["worlds"]
        and metrics["D"]["hard_failure_rate"] <= 1.0e-12
        and metrics["D"]["query"]["selective_risk"] <= 1.0e-12
        and metrics["B"]["settled_b"]["selective_risk"] <= 0.03
        and metrics["B"]["settled_c"]["selective_risk"] <= 0.03
        and metrics["B"]["settled_recurrence"]["selective_risk"] <= 0.03
    )


def _dominates(left, right):
    a = left["objectives"]
    b = right["objectives"]
    no_worse = (
        a["a_trap_coverage"] >= b["a_trap_coverage"]
        and a["b_transition_wrong"] <= b["b_transition_wrong"]
        and a["b_transition_correct"] >= b["b_transition_correct"]
        and a["d_coverage"] >= b["d_coverage"]
        and a["budget"] <= b["budget"]
    )
    strictly = any((
        a["a_trap_coverage"] > b["a_trap_coverage"],
        a["b_transition_wrong"] < b["b_transition_wrong"],
        a["b_transition_correct"] > b["b_transition_correct"],
        a["d_coverage"] > b["d_coverage"],
        a["budget"] < b["budget"],
    ))
    return no_worse and strictly


def _frontier(rows):
    feasible = [row for row in rows if row["feasible"]]
    return [
        row
        for row in feasible
        if not any(_dominates(other, row) for other in feasible if other is not row)
    ]


def _ideal_distance(rows, row):
    objectives = [item["objectives"] for item in rows]
    specs = {
        "a_trap_coverage": "max",
        "b_transition_wrong": "min",
        "b_transition_correct": "max",
        "d_coverage": "max",
        "budget": "min",
    }
    total = 0.0
    for key, direction in specs.items():
        values = [float(item[key]) for item in objectives]
        low = min(values)
        high = max(values)
        span = high - low
        if span <= 1.0e-12:
            normalized = 0.0
        elif direction == "max":
            normalized = (high - float(row["objectives"][key])) / span
        else:
            normalized = (float(row["objectives"][key]) - low) / span
        total += normalized * normalized
    return sqrt(total / len(specs))


def _summarize_sweep(rows, limit=8):
    selected = []
    for row in rows[:limit]:
        selected.append({
            "candidate": _serial(row["candidate"]),
            "feasible": row.get("feasible"),
            "metrics": row["metrics"],
        })
    return selected


def run(tuning_worlds: int, validation_worlds: int):
    baseline_policy = ClosedLoopPolicy()
    baseline_regime = RegimePlasticityPolicy()
    tuning = _stage_worlds("closed-loop-v0.2-tuning", tuning_worlds)

    evidence_top, evidence_all = _top_evidence(
        tuning["A"], baseline_policy, baseline_regime
    )
    temporal_core_top, temporal_core_all = _top_temporal_core(
        tuning["B"], baseline_policy, baseline_regime
    )
    temporal_top, temporal_all = _top_recurrence(
        tuning["B"], baseline_policy, temporal_core_top
    )
    residual_top, residual_all = _top_residual(
        tuning["C"], baseline_policy, baseline_regime
    )
    memory_top, memory_all = _top_memory(
        tuning["D"], baseline_policy, baseline_regime
    )

    combined = []
    index = 0
    for e, t, c, d in product(
        evidence_top,
        temporal_top,
        residual_top,
        memory_top,
    ):
        candidate = _merge_candidates(e, t, c, d, index)
        metrics = _evaluate_all(tuning, candidate)
        combined.append({
            "candidate": candidate,
            "metrics": metrics,
            "feasible": _feasible(metrics),
            "objectives": _objectives(metrics),
        })
        index += 1

    frontier = _frontier(combined)
    reference = frontier or combined
    for row in reference:
        row["ideal_distance"] = _ideal_distance(reference, row)
    reference.sort(key=lambda row: row["ideal_distance"])
    finalists = reference[:3]

    validation = _stage_worlds(
        "closed-loop-v0.2-validation",
        validation_worlds,
    )
    validated = []
    for finalist in finalists:
        metrics = _evaluate_all(validation, finalist["candidate"])
        validated.append({
            "candidate": finalist["candidate"],
            "tuning_metrics": finalist["metrics"],
            "metrics": metrics,
            "feasible": _feasible(metrics),
            "objectives": _objectives(metrics),
        })

    valid_reference = [row for row in validated if row["feasible"]] or validated
    for row in valid_reference:
        row["ideal_distance"] = _ideal_distance(valid_reference, row)
    valid_reference.sort(key=lambda row: row["ideal_distance"])
    selected = valid_reference[0]

    baseline_metrics = _evaluate_all(validation, _candidate(
        baseline_policy,
        baseline_regime,
        label="v0.1-baseline",
    ))

    return {
        "protocol": PROTOCOL,
        "mode": "development-tuning-and-holdout-validation",
        "tuning_worlds_per_stage": tuning_worlds,
        "validation_worlds_per_stage": validation_worlds,
        "baseline": {
            "candidate": _serial(_candidate(
                baseline_policy,
                baseline_regime,
                label="v0.1-baseline",
            )),
            "metrics": baseline_metrics,
            "objectives": _objectives(baseline_metrics),
        },
        "factor_sweeps": {
            "evidence_top": _summarize_sweep(evidence_top),
            "temporal_core_top": _summarize_sweep(temporal_core_top),
            "temporal_top": _summarize_sweep(temporal_top),
            "residual_top": _summarize_sweep(residual_top),
            "memory_top": _summarize_sweep(memory_top),
            "counts": {
                "evidence": len(evidence_all),
                "temporal_core": len(temporal_core_all),
                "temporal_recurrence": len(temporal_all),
                "residual": len(residual_all),
                "memory": len(memory_all),
                "combined": len(combined),
                "pareto": len(frontier),
            },
        },
        "finalists": [
            {
                "candidate": _serial(row["candidate"]),
                "feasible": row["feasible"],
                "objectives": row["objectives"],
                "ideal_distance": row.get("ideal_distance"),
                "metrics": row["metrics"],
                "tuning_metrics": row["tuning_metrics"],
            }
            for row in validated
        ],
        "selected": {
            "candidate": _serial(selected["candidate"]),
            "feasible": selected["feasible"],
            "objectives": selected["objectives"],
            "ideal_distance": selected.get("ideal_distance"),
            "metrics": selected["metrics"],
        },
        "official_seeds_used": False,
        "owr_e_used": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tuning-worlds", type=int, default=4)
    parser.add_argument("--validation-worlds", type=int, default=12)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run(args.tuning_worlds, args.validation_worlds)
    rendered = json.dumps(report, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
