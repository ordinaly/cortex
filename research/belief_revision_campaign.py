"""Official campaign for belief-revision-v1."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from belief_revision import (
    AMBIGUOUS_CONTRADICTION_BATCH,
    CAUSAL_BATCH,
    STRONG_CONTRADICTION_BATCH,
    apply_ambiguous_contradiction,
    apply_causal_batch,
    apply_strong_contradiction,
    build_model,
    build_world,
    checkpoint,
    latest_batch_state,
    monotone_nondecreasing,
    monotone_nonincreasing,
    target_cell,
    unrelated_fingerprint,
)

OFFICIAL_SEEDS = tuple(range(200, 220))
PRIOR_STRENGTHS = (1, 2, 3)


def run_case(seed: int, prior_batches: int) -> dict:
    world = build_world(seed)

    # Main revision trajectory.
    model = build_model(
        world,
        prior_batches=prior_batches,
    )
    initial = checkpoint(model, world, 0)
    unrelated_initial = unrelated_fingerprint(model, world)

    contradiction_effects = [initial.effect]
    first_unresolved = None
    first_null = None
    unresolved_downstream = None
    null_downstream = None

    for batch in range(1, 41):
        apply_strong_contradiction(model, world)
        current = checkpoint(model, world, batch)
        contradiction_effects.append(current.effect)

        if current.direct_state == 0 and first_unresolved is None:
            first_unresolved = batch
            unresolved_downstream = current.downstream_state

        if current.direct_state == -1:
            first_null = batch
            null_downstream = current.downstream_state
            break

    unrelated_after_retraction = unrelated_fingerprint(
        model,
        world,
    )

    recovery_effects = [target_cell(model, world).effect]
    recovery_first_unresolved = None
    recovery_active = None
    recovered_downstream = None

    for batch in range(1, 21):
        apply_causal_batch(model, world)
        current = checkpoint(model, world, batch)
        recovery_effects.append(current.effect)

        if (
            current.direct_state == 0
            and recovery_first_unresolved is None
        ):
            recovery_first_unresolved = batch

        if current.direct_state == 1:
            recovery_active = batch
            recovered_downstream = current.downstream_state
            break

    unrelated_after_recovery = unrelated_fingerprint(
        model,
        world,
    )

    # One contradictory batch should not erase accumulated support.
    single = build_model(
        world,
        prior_batches=prior_batches,
    )
    apply_strong_contradiction(single, world)
    single_batch_state = target_cell(single, world).state

    # Ambiguous evidence should weaken confidence without forcing null.
    ambiguous = build_model(
        world,
        prior_batches=prior_batches,
    )
    ambiguous_initial_fp = unrelated_fingerprint(
        ambiguous,
        world,
    )
    ambiguous_effects = [
        target_cell(ambiguous, world).effect
    ]
    for _ in range(20):
        apply_ambiguous_contradiction(
            ambiguous,
            world,
        )
        ambiguous_effects.append(
            target_cell(ambiguous, world).effect
        )
    ambiguous_final = checkpoint(
        ambiguous,
        world,
        20,
    )
    ambiguous_final_fp = unrelated_fingerprint(
        ambiguous,
        world,
    )

    downstream_alignment = (
        initial.downstream_state == 1
        and unresolved_downstream == 0
        and null_downstream == -1
        and recovered_downstream == 1
    )

    transition_order = (
        first_unresolved is not None
        and first_null is not None
        and first_unresolved < first_null
        and recovery_first_unresolved is not None
        and recovery_active is not None
        and recovery_first_unresolved < recovery_active
    )

    unrelated_preserved = (
        unrelated_initial == unrelated_after_retraction
        == unrelated_after_recovery
        and ambiguous_initial_fp == ambiguous_final_fp
    )

    return {
        "protocol": "belief-revision-v1",
        "seed": seed,
        "prior_batches": prior_batches,
        "initial": {
            "state": initial.direct_state,
            "effect": initial.effect,
            "downstream_state": initial.downstream_state,
        },
        "single_batch": {
            "state": single_batch_state,
            "retained_active": single_batch_state == 1,
        },
        "strong_contradiction": {
            "first_unresolved_batch": first_unresolved,
            "first_null_batch": first_null,
            "retraction_success": first_null is not None,
            "margin_monotone": monotone_nonincreasing(
                contradiction_effects
            ),
            "effects": contradiction_effects,
            "unresolved_downstream_state": unresolved_downstream,
            "null_downstream_state": null_downstream,
        },
        "recovery": {
            "first_unresolved_batch": recovery_first_unresolved,
            "active_batch": recovery_active,
            "success": recovery_active is not None,
            "margin_monotone": monotone_nondecreasing(
                recovery_effects
            ),
            "effects": recovery_effects,
            "downstream_state": recovered_downstream,
        },
        "ambiguous": {
            "final_state": ambiguous_final.direct_state,
            "final_effect": ambiguous_final.effect,
            "downstream_state": ambiguous_final.downstream_state,
            "unresolved": ambiguous_final.direct_state == 0,
            "false_retraction": ambiguous_final.direct_state == -1,
            "margin_monotone": monotone_nonincreasing(
                ambiguous_effects
            ),
        },
        "transition_order_valid": transition_order,
        "downstream_alignment": downstream_alignment,
        "unrelated_structure_preserved": unrelated_preserved,
        "sticky_baseline_retraction_success": False,
        "latest_batch_premature_retraction": (
            latest_batch_state(
                STRONG_CONTRADICTION_BATCH
            )
            == -1
        ),
    }


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def campaign(
    output: Path,
    *,
    seeds: Iterable[int] = OFFICIAL_SEEDS,
) -> dict:
    seeds = tuple(seeds)
    rows = [
        run_case(seed, prior)
        for seed in seeds
        for prior in PRIOR_STRENGTHS
    ]

    output.write_text(
        "".join(
            json.dumps(row, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )

    ordering_by_seed = {}
    for seed in seeds:
        seed_rows = sorted(
            (
                row
                for row in rows
                if row["seed"] == seed
            ),
            key=lambda row: row["prior_batches"],
        )
        retract = [
            row["strong_contradiction"]["first_null_batch"]
            for row in seed_rows
        ]
        recover = [
            row["recovery"]["active_batch"]
            for row in seed_rows
        ]
        ordering_by_seed[str(seed)] = {
            "retraction_order_valid": (
                None not in retract
                and retract[0] < retract[1] < retract[2]
            ),
            "recovery_order_valid": (
                None not in recover
                and recover[0] < recover[1] < recover[2]
            ),
        }

    retraction_latencies = [
        row["strong_contradiction"]["first_null_batch"]
        for row in rows
        if row["strong_contradiction"]["first_null_batch"]
        is not None
    ]
    recovery_latencies = [
        row["recovery"]["active_batch"]
        for row in rows
        if row["recovery"]["active_batch"]
        is not None
    ]

    summary = {
        "protocol": "belief-revision-v1",
        "rows": len(rows),
        "seeds": list(seeds),
        "prior_strengths": list(PRIOR_STRENGTHS),
        "metrics": {
            "initial_active_rate": _mean(
                int(row["initial"]["state"] == 1)
                for row in rows
            ),
            "single_batch_retention_rate": _mean(
                int(row["single_batch"]["retained_active"])
                for row in rows
            ),
            "retraction_success_rate": _mean(
                int(
                    row["strong_contradiction"][
                        "retraction_success"
                    ]
                )
                for row in rows
            ),
            "transition_order_valid_rate": _mean(
                int(row["transition_order_valid"])
                for row in rows
            ),
            "max_retraction_latency": max(
                retraction_latencies,
                default=0,
            ),
            "mean_retraction_latency": _mean(
                retraction_latencies
            ),
            "retraction_prior_order_rate": _mean(
                int(
                    value["retraction_order_valid"]
                )
                for value in ordering_by_seed.values()
            ),
            "ambiguous_unresolved_rate": _mean(
                int(row["ambiguous"]["unresolved"])
                for row in rows
            ),
            "ambiguous_false_retraction_rate": _mean(
                int(row["ambiguous"]["false_retraction"])
                for row in rows
            ),
            "recovery_success_rate": _mean(
                int(row["recovery"]["success"])
                for row in rows
            ),
            "max_recovery_latency": max(
                recovery_latencies,
                default=0,
            ),
            "mean_recovery_latency": _mean(
                recovery_latencies
            ),
            "recovery_prior_order_rate": _mean(
                int(
                    value["recovery_order_valid"]
                )
                for value in ordering_by_seed.values()
            ),
            "downstream_alignment_rate": _mean(
                int(row["downstream_alignment"])
                for row in rows
            ),
            "contradiction_margin_monotone_rate": _mean(
                int(
                    row["strong_contradiction"][
                        "margin_monotone"
                    ]
                )
                for row in rows
            ),
            "recovery_margin_monotone_rate": _mean(
                int(row["recovery"]["margin_monotone"])
                for row in rows
            ),
            "unrelated_structure_preservation_rate": _mean(
                int(row["unrelated_structure_preserved"])
                for row in rows
            ),
            "sticky_baseline_retraction_success": _mean(
                int(
                    row["sticky_baseline_retraction_success"]
                )
                for row in rows
            ),
            "latest_batch_premature_retraction_rate": _mean(
                int(
                    row["latest_batch_premature_retraction"]
                )
                for row in rows
            ),
        },
        "latency_by_prior": {
            str(prior): {
                "mean_retraction": _mean(
                    row["strong_contradiction"][
                        "first_null_batch"
                    ]
                    for row in rows
                    if row["prior_batches"] == prior
                    and row["strong_contradiction"][
                        "first_null_batch"
                    ]
                    is not None
                ),
                "mean_recovery": _mean(
                    row["recovery"]["active_batch"]
                    for row in rows
                    if row["prior_batches"] == prior
                    and row["recovery"]["active_batch"]
                    is not None
                ),
            }
            for prior in PRIOR_STRENGTHS
        },
        "ordering_by_seed": ordering_by_seed,
    }

    print(json.dumps(summary, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("belief-revision-v1.jsonl"),
    )
    parser.add_argument(
        "--development-seeds",
        type=int,
        default=0,
        help=(
            "Use development seeds 0..N-1 instead of official "
            "seeds 200..219."
        ),
    )
    args = parser.parse_args()

    seeds = (
        tuple(range(args.development_seeds))
        if args.development_seeds > 0
        else OFFICIAL_SEEDS
    )
    campaign(args.output, seeds=seeds)


if __name__ == "__main__":
    main()
