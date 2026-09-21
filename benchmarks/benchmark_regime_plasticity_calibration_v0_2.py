#!/usr/bin/env python3
"""Calibrate the Cortex relational regime-plasticity policy.

The search optimizes a declared stability/plasticity objective rather than a
single switch-lag metric. Candidate parameters are screened on deterministic
regime shifts and burst controls, ranked on stochastic training streams, then
validated unchanged on held-out seeds.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from itertools import product
import json
from pathlib import Path
from random import Random
import statistics

from cortex.hypergraph_inference import RelationStep
from cortex.regime_plasticity import (
    RegimeConfig,
    RegimePhase,
    RelationalRegimeMemory,
)


PROTOCOL = "regime-plasticity-calibration-v0.2"
SIGNATURE = (
    RelationStep("R0", "forward"),
    RelationStep("R1", "forward"),
)
PRIOR_MASSES = (8, 32, 128, 256)


def _stable(config: RegimeConfig, relation: str, count: int):
    regime = RelationalRegimeMemory(SIGNATURE, config=config)
    for _ in range(count):
        regime.observe(relation)
    return regime


def _switch_lag(config: RegimeConfig, prior: int) -> tuple[int | None, int | None]:
    regime = _stable(config, "OLD", prior)
    old_exit = None
    new_accept = None
    for step in range(1, 80):
        read = regime.observe("NEW")
        if old_exit is None and read.phase is not RegimePhase.STABLE:
            old_exit = step
        if read.accepted_relation == "NEW":
            new_accept = step
            break
    return old_exit, new_accept


def _recurrence(config: RegimeConfig, prior: int = 64):
    regime = _stable(config, "OLD", prior)
    to_new = None
    for step in range(1, 80):
        if regime.observe("NEW").accepted_relation == "NEW":
            to_new = step
            break
    if to_new is None:
        return None, None, False

    to_old = None
    for step in range(1, 80):
        if regime.observe("OLD").accepted_relation == "OLD":
            to_old = step
            break
    archived = [row.relation for row in regime.archives]
    return to_new, to_old, archived[:2] == ["OLD", "NEW"]


def _burst(config: RegimeConfig, burst: int, prior: int = 32):
    regime = _stable(config, "OLD", prior)
    switched = False
    deepest = RegimePhase.STABLE
    order = {
        RegimePhase.BOOTSTRAP: 0,
        RegimePhase.STABLE: 1,
        RegimePhase.CHALLENGED: 2,
        RegimePhase.UNRESOLVED: 3,
    }
    for _ in range(burst):
        read = regime.observe("NEW")
        switched |= read.accepted_relation == "NEW"
        if order[read.phase] > order[deepest]:
            deepest = read.phase

    recovery = 0
    while (
        regime.read().accepted_relation != "OLD"
        or regime.read().phase is not RegimePhase.STABLE
    ):
        regime.observe("OLD")
        recovery += 1
        if recovery > 96:
            break
    return switched, deepest.value, recovery


def _stationary(
    config: RegimeConfig,
    *,
    seed: int,
    noise_rate: float,
    steps: int = 700,
    noise_reliability: float = 1.0,
):
    rng = Random(seed)
    regime = _stable(config, "OLD", 32)
    false_switches = 0
    nonstable = 0
    unresolved = 0

    for _ in range(steps):
        noisy = rng.random() < noise_rate
        relation = "NEW" if noisy else "OLD"
        reliability = noise_reliability if noisy else 1.0
        before = regime.read().accepted_relation
        read = regime.observe(relation, reliability=reliability)
        false_switches += int(
            before == "OLD" and read.accepted_relation == "NEW"
        )
        nonstable += int(read.phase is not RegimePhase.STABLE)
        unresolved += int(read.phase is RegimePhase.UNRESOLVED)

    return {
        "false_switches": false_switches,
        "nonstable_fraction": nonstable / steps,
        "unresolved_fraction": unresolved / steps,
        "end_old": regime.read().accepted_relation == "OLD",
    }


def _candidate_grid():
    for (
        decay,
        challenge,
        patience,
        acceptance,
        acceptance_patience,
        minimum_candidate,
    ) in product(
        (0.82, 0.86, 0.90, 0.94),
        (0.35, 0.40, 0.45),
        (2, 3),
        (0.85, 0.90, 0.93),
        (1, 2),
        (6, 8, 10),
    ):
        hysteresis = 0.05
        if hysteresis >= challenge:
            continue
        yield RegimeConfig(
            recent_decay=decay,
            challenge_threshold=challenge,
            challenge_hysteresis=hysteresis,
            challenge_patience=patience,
            acceptance_threshold=acceptance,
            acceptance_patience=acceptance_patience,
            minimum_initial_observations=4,
            minimum_candidate_observations=minimum_candidate,
        )


def _deterministic_screen(config: RegimeConfig):
    lags = []
    exits = []
    for prior in PRIOR_MASSES:
        old_exit, new_accept = _switch_lag(config, prior)
        if old_exit is None or new_accept is None:
            return None
        lags.append(new_accept)
        exits.append(old_exit)

    recurrence_new, recurrence_old, archives = _recurrence(config)
    if (
        recurrence_new is None
        or recurrence_old is None
        or not archives
    ):
        return None

    burst_rows = {
        burst: _burst(config, burst)
        for burst in (1, 2, 4, 8)
    }
    if any(burst_rows[b][0] for b in (1, 2, 4, 8)):
        return None

    # A useful candidate should notice sustained change quickly but must not
    # immediately jump to a replacement.
    if max(exits) > 12:
        return None
    if min(lags) < 8 or max(lags) > 32:
        return None

    return {
        "lags": lags,
        "old_exit": exits,
        "recurrence_new": recurrence_new,
        "recurrence_old": recurrence_old,
        "burst_recovery": {
            str(b): burst_rows[b][2] for b in burst_rows
        },
    }


def _stochastic_score(
    config: RegimeConfig,
    deterministic,
    *,
    seeds: tuple[int, ...],
):
    noise_rows = []
    for seed in seeds:
        for rate in (0.10, 0.20, 0.30):
            noise_rows.append(
                (
                    rate,
                    _stationary(
                        config,
                        seed=seed * 1009 + int(rate * 1000),
                        noise_rate=rate,
                    ),
                )
            )

    false_switches = sum(
        row["false_switches"] for _rate, row in noise_rows
    )
    if false_switches:
        return None
    if not all(row["end_old"] for _rate, row in noise_rows):
        return None

    nonstable_20 = statistics.mean(
        row["nonstable_fraction"]
        for rate, row in noise_rows
        if rate == 0.20
    )
    nonstable_30 = statistics.mean(
        row["nonstable_fraction"]
        for rate, row in noise_rows
        if rate == 0.30
    )
    unresolved_30 = statistics.mean(
        row["unresolved_fraction"]
        for rate, row in noise_rows
        if rate == 0.30
    )

    mean_lag = statistics.mean(deterministic["lags"])
    lag_span = max(deterministic["lags"]) - min(deterministic["lags"])
    mean_exit = statistics.mean(deterministic["old_exit"])
    recurrence = (
        deterministic["recurrence_new"]
        + deterministic["recurrence_old"]
    ) / 2.0
    recovery_penalty = statistics.mean(
        deterministic["burst_recovery"].values()
    )

    # Lower is better. Safety is handled as a hard constraint above. The
    # remaining score balances adaptation speed, history invariance, early
    # change detection, recurrence, transient-noise disturbance, and recovery.
    score = (
        1.00 * mean_lag
        + 0.30 * lag_span
        + 0.20 * mean_exit
        + 0.35 * recurrence
        + 0.10 * recovery_penalty
        + 30.0 * nonstable_20
        + 20.0 * nonstable_30
        + 20.0 * unresolved_30
    )
    return {
        "score": score,
        "mean_lag": mean_lag,
        "lag_span": lag_span,
        "mean_old_exit": mean_exit,
        "mean_recurrence": recurrence,
        "mean_burst_recovery": recovery_penalty,
        "noise20_nonstable": nonstable_20,
        "noise30_nonstable": nonstable_30,
        "noise30_unresolved": unresolved_30,
        "false_switches": false_switches,
    }


def _validate(config: RegimeConfig, *, seeds: tuple[int, ...]):
    switch_rows = []
    for seed in seeds:
        # seed is retained for future stochastic relation streams; the
        # persistent-shift sequence itself is deterministic by construction.
        for prior in (4, 8, 16, 32, 64, 128, 256, 512):
            old_exit, new_accept = _switch_lag(config, prior)
            switch_rows.append(
                {
                    "seed": seed,
                    "prior": prior,
                    "old_exit": old_exit,
                    "new_accept": new_accept,
                }
            )

    noise = []
    for seed in seeds:
        for rate in (0.05, 0.10, 0.20, 0.30, 0.35):
            noise.append(
                {
                    "seed": seed,
                    "rate": rate,
                    **_stationary(
                        config,
                        seed=99173 + seed * 7919 + int(rate * 1000),
                        noise_rate=rate,
                        steps=1200,
                    ),
                }
            )
        # A low-reliability contradictory source should be even less able to
        # move the active regime.
        noise.append(
            {
                "seed": seed,
                "rate": 0.40,
                "low_reliability": True,
                **_stationary(
                    config,
                    seed=199933 + seed * 3571,
                    noise_rate=0.40,
                    steps=1200,
                    noise_reliability=0.25,
                ),
            }
        )

    bursts = []
    for seed in seeds:
        for burst in (1, 2, 4, 8, 12, 16, 24):
            switched, deepest, recovery = _burst(config, burst)
            bursts.append(
                {
                    "seed": seed,
                    "burst": burst,
                    "switched": switched,
                    "deepest": deepest,
                    "recovery": recovery,
                }
            )

    recurrence = []
    for seed in seeds:
        to_new, to_old, archives = _recurrence(config)
        recurrence.append(
            {
                "seed": seed,
                "to_new": to_new,
                "to_old": to_old,
                "archives": archives,
            }
        )

    lags = [row["new_accept"] for row in switch_rows]
    exits = [row["old_exit"] for row in switch_rows]

    hard_noise = [
        row
        for row in noise
        if row["rate"] <= 0.30 and not row.get("low_reliability", False)
    ]
    low_reliability = [
        row for row in noise if row.get("low_reliability", False)
    ]

    metrics = {
        "switch_lag_min": min(lags),
        "switch_lag_max": max(lags),
        "switch_lag_mean": statistics.mean(lags),
        "switch_lag_span": max(lags) - min(lags),
        "old_exit_mean": statistics.mean(exits),
        "false_switches_through_30pct": sum(
            row["false_switches"] for row in hard_noise
        ),
        "all_noise_end_old": all(row["end_old"] for row in hard_noise),
        "mean_nonstable_30pct": statistics.mean(
            row["nonstable_fraction"]
            for row in hard_noise
            if row["rate"] == 0.30
        ),
        "low_reliability_false_switches": sum(
            row["false_switches"] for row in low_reliability
        ),
        "short_burst_switches": sum(
            row["switched"] for row in bursts if row["burst"] <= 8
        ),
        "recurrence_max": max(
            max(row["to_new"], row["to_old"]) for row in recurrence
        ),
        "recurrence_archives": all(row["archives"] for row in recurrence),
    }

    gates = {
        "bounded_lag": metrics["switch_lag_max"] <= 24,
        "history_invariant": metrics["switch_lag_span"] <= 8,
        "stationary_noise_safe": (
            metrics["false_switches_through_30pct"] == 0
            and metrics["all_noise_end_old"]
        ),
        "low_reliability_safe": (
            metrics["low_reliability_false_switches"] == 0
        ),
        "short_bursts_safe": metrics["short_burst_switches"] == 0,
        "recurrence_bounded": metrics["recurrence_max"] <= 24,
        "history_archived": metrics["recurrence_archives"],
    }
    return {
        "metrics": metrics,
        "gates": gates,
        "passed": all(gates.values()),
        "switches": switch_rows,
        "noise": noise,
        "bursts": bursts,
        "recurrence": recurrence,
    }


def campaign():
    train_seeds = tuple(range(12))
    validation_seeds = tuple(range(100, 124))

    screened = []
    for config in _candidate_grid():
        deterministic = _deterministic_screen(config)
        if deterministic is None:
            continue
        score = _stochastic_score(
            config,
            deterministic,
            seeds=train_seeds,
        )
        if score is None:
            continue
        screened.append((score["score"], config, deterministic, score))

    if not screened:
        raise RuntimeError("no plasticity configuration satisfied safety screen")

    screened.sort(
        key=lambda row: (
            row[0],
            row[1].recent_decay,
            row[1].acceptance_threshold,
            row[1].minimum_candidate_observations,
        )
    )
    best_score, best_config, deterministic, train_score = screened[0]

    baseline_config = RegimeConfig()
    baseline_deterministic = _deterministic_screen(baseline_config)
    baseline_score = _stochastic_score(
        baseline_config,
        baseline_deterministic,
        seeds=train_seeds,
    )

    validation = _validate(best_config, seeds=validation_seeds)
    baseline_validation = _validate(
        baseline_config,
        seeds=validation_seeds,
    )

    # Robustness: the selected point should not be a razor-thin optimum. Report
    # the top safe neighborhood and require several distinct safe candidates.
    top = [
        {
            "rank": index + 1,
            "score": score,
            "config": asdict(config),
            "mean_lag": stats["mean_lag"],
            "lag_span": stats["lag_span"],
            "noise30_nonstable": stats["noise30_nonstable"],
        }
        for index, (score, config, _det, stats)
        in enumerate(screened[:12])
    ]

    gates = {
        "validation_passed": validation["passed"],
        "beats_hand_tuned_mean_lag": (
            validation["metrics"]["switch_lag_mean"]
            <= baseline_validation["metrics"]["switch_lag_mean"]
        ),
        "does_not_worsen_noise_safety": (
            validation["metrics"]["false_switches_through_30pct"]
            <= baseline_validation["metrics"]["false_switches_through_30pct"]
        ),
        "safe_neighborhood_exists": len(screened) >= 8,
    }

    return {
        "protocol": PROTOCOL,
        "candidate_count": 432,
        "safe_candidate_count": len(screened),
        "selected_config": asdict(best_config),
        "selected_training_score": best_score,
        "training_metrics": train_score,
        "validation": validation,
        "baseline_config": asdict(baseline_config),
        "baseline_training_metrics": baseline_score,
        "baseline_validation": baseline_validation,
        "top_safe_candidates": top,
        "gates": gates,
        "passed": all(gates.values()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = campaign()
    payload = json.dumps(report, indent=2, sort_keys=True)
    print(payload)
    if args.output:
        args.output.write_text(payload + "\n", encoding="utf-8")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
