"""Campaign for latent-concept-induction-v1."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from latent_concept_induction import (
    CortexLatentConceptInducer,
    CortexLatentConceptLifecycle,
    flat_probabilities,
    global_probabilities,
    heldout_nll,
    pairwise_partition_f1,
)

PROTOCOL = "latent-concept-induction-v1"
DEVELOPMENT_SEEDS = tuple(range(40))
OFFICIAL_SEEDS = tuple(range(700, 720))

TWO_TRUTH = (0, 0, 0, 1, 1, 1)
THREE_TRUTH = (0, 0, 1, 1, 2, 2)
SINGLETON_TRUTH = (0, 1, 2, 3, 4, 5)
DECOY_PARTITION = (0, 1, 0, 1, 0, 1)

TWO_PROBABILITIES = np.asarray(
    [
        [0.65, 0.15, 0.10, 0.05, 0.03, 0.02],
        [0.05, 0.05, 0.10, 0.20, 0.25, 0.35],
    ],
    dtype=float,
)

THREE_PROBABILITIES = np.asarray(
    [
        [0.55, 0.25, 0.10, 0.05, 0.03, 0.02],
        [0.05, 0.10, 0.55, 0.20, 0.05, 0.05],
        [0.03, 0.03, 0.04, 0.15, 0.30, 0.45],
    ],
    dtype=float,
)


def _family_code(name: str) -> int:
    return sum(
        (index + 1) * ord(char)
        for index, char in enumerate(name)
    )


def _rng(seed: int, family: str) -> np.random.Generator:
    return np.random.default_rng(
        seed * 104729 + _family_code(family)
    )


def _anonymize(
    counts: np.ndarray,
    heldout: np.ndarray,
    truth: tuple[int, ...],
    *,
    rng: np.random.Generator,
) -> tuple[
    np.ndarray,
    np.ndarray,
    tuple[int, ...],
    tuple[int, ...],
]:
    entity_order = rng.permutation(len(truth))
    outcome_order = rng.permutation(counts.shape[1])

    observed_counts = counts[entity_order][:, outcome_order]
    observed_heldout = heldout[entity_order][:, outcome_order]
    observed_truth = tuple(
        truth[int(index)]
        for index in entity_order
    )
    observed_decoy = tuple(
        DECOY_PARTITION[int(index)]
        for index in entity_order
    )
    return (
        observed_counts,
        observed_heldout,
        observed_truth,
        observed_decoy,
    )


def grouped_fixture(
    family: str,
    seed: int,
) -> dict:
    if family == "two-concept":
        probabilities = TWO_PROBABILITIES
        truth = TWO_TRUTH
        train_per_entity = 20
    elif family == "three-concept":
        probabilities = THREE_PROBABILITIES
        truth = THREE_TRUTH
        train_per_entity = 25
    else:
        raise ValueError(f"unknown grouped family: {family}")

    rng = _rng(seed, family)
    counts = np.asarray(
        [
            rng.multinomial(
                train_per_entity,
                probabilities[truth[entity]],
            )
            for entity in range(6)
        ],
        dtype=float,
    )
    heldout = np.asarray(
        [
            rng.multinomial(
                300,
                probabilities[truth[entity]],
            )
            for entity in range(6)
        ],
        dtype=float,
    )

    (
        counts,
        heldout,
        observed_truth,
        observed_decoy,
    ) = _anonymize(
        counts,
        heldout,
        truth,
        rng=rng,
    )

    model = CortexLatentConceptInducer(counts)
    prediction = model.representative_probabilities()

    return {
        "protocol": PROTOCOL,
        "case": "grouped",
        "family": family,
        "seed": seed,
        "candidate_count": model.fit.candidate_count,
        "version_space": len(model.fit.best_partitions),
        "resolved": model.fit.resolved,
        "concept_count": model.fit.concept_count,
        "partition_f1": pairwise_partition_f1(
            model.fit.best_partition,
            observed_truth,
        ),
        "decoy_partition_f1": pairwise_partition_f1(
            observed_decoy,
            observed_truth,
        ),
        "latent_nll": heldout_nll(prediction, heldout),
        "flat_nll": heldout_nll(
            flat_probabilities(counts),
            heldout,
        ),
        "global_nll": heldout_nll(
            global_probabilities(counts),
            heldout,
        ),
        "flat_gain": (
            heldout_nll(
                flat_probabilities(counts),
                heldout,
            )
            - heldout_nll(prediction, heldout)
        ),
        "global_gain": (
            heldout_nll(
                global_probabilities(counts),
                heldout,
            )
            - heldout_nll(prediction, heldout)
        ),
    }


def singleton_control(seed: int) -> dict:
    family = "no-shared-concept"
    rng = _rng(seed, family)

    probabilities = []
    for entity in range(6):
        vector = np.full(6, 0.04, dtype=float)
        vector[entity] = 0.80
        vector /= vector.sum()
        probabilities.append(vector)
    probabilities = np.asarray(probabilities)

    counts = np.asarray(
        [
            rng.multinomial(35, probabilities[entity])
            for entity in range(6)
        ],
        dtype=float,
    )
    heldout = np.asarray(
        [
            rng.multinomial(100, probabilities[entity])
            for entity in range(6)
        ],
        dtype=float,
    )
    counts, heldout, truth, _decoy = _anonymize(
        counts,
        heldout,
        SINGLETON_TRUTH,
        rng=rng,
    )

    model = CortexLatentConceptInducer(counts)

    return {
        "protocol": PROTOCOL,
        "case": "singleton-control",
        "family": family,
        "seed": seed,
        "candidate_count": model.fit.candidate_count,
        "resolved": model.fit.resolved,
        "concept_count": model.fit.concept_count,
        "partition_f1": pairwise_partition_f1(
            model.fit.best_partition,
            truth,
        ),
        "resolved_latent_blocks": len(
            model.resolved_blocks()
        ),
    }


def ambiguous_member(seed: int) -> dict:
    family = "ambiguous-member"
    rng = _rng(seed, family)

    first = np.asarray([24, 4, 1, 1, 0, 0], dtype=float)
    second = np.asarray([1, 2, 4, 20, 2, 1], dtype=float)
    counts = np.vstack(
        [
            first,
            first,
            first,
            second,
            second,
            np.zeros(6, dtype=float),
        ]
    )
    heldout = np.zeros_like(counts)

    entity_order = rng.permutation(6)
    outcome_order = rng.permutation(6)
    counts = counts[entity_order][:, outcome_order]

    ambiguous_index = int(
        np.where(entity_order == 5)[0][0]
    )
    model = CortexLatentConceptInducer(counts)
    prediction = model.predict_entity(ambiguous_index)

    return {
        "protocol": PROTOCOL,
        "case": "ambiguity",
        "family": family,
        "seed": seed,
        "candidate_count": model.fit.candidate_count,
        "version_space": len(model.fit.best_partitions),
        "resolved_partition": model.fit.resolved,
        "ambiguous_entity_resolved": prediction.resolved,
        "ambiguous_entity_version_space": (
            prediction.version_space
        ),
        "ambiguous_comembership_spread": max(
            model.co_membership(
                ambiguous_index,
                other,
            )
            for other in range(6)
            if other != ambiguous_index
        ) - min(
            model.co_membership(
                ambiguous_index,
                other,
            )
            for other in range(6)
            if other != ambiguous_index
        ),
    }


def lifecycle_case(seed: int) -> dict:
    family = "lifecycle"
    rng = _rng(seed, family)
    entity_order = rng.permutation(6)
    outcome_order = rng.permutation(6)

    coarse = np.tile(
        np.asarray([20, 10, 5, 3, 1, 1], dtype=float),
        (6, 1),
    )
    split = np.vstack(
        [
            np.asarray([28, 5, 3, 2, 1, 1], dtype=float)
            for _ in range(3)
        ]
        + [
            np.asarray([1, 1, 2, 3, 5, 28], dtype=float)
            for _ in range(3)
        ]
    )
    unique = []
    for entity in range(6):
        vector = np.full(6, 2.0)
        vector[entity] = 30.0
        unique.append(vector)
    unique = np.asarray(unique)

    snapshots = (
        coarse,
        split,
        coarse,
        unique,
    )
    truths = (
        (0, 0, 0, 0, 0, 0),
        TWO_TRUTH,
        (0, 0, 0, 0, 0, 0),
        SINGLETON_TRUTH,
    )

    model = CortexLatentConceptLifecycle(6)
    f1s = []
    event_kinds = []

    for epoch, (snapshot, truth) in enumerate(
        zip(snapshots, truths)
    ):
        observed = snapshot[entity_order][:, outcome_order]
        observed_truth = tuple(
            truth[int(index)]
            for index in entity_order
        )
        read = model.advance(epoch, observed)
        f1s.append(
            pairwise_partition_f1(
                read.fit.best_partition,
                observed_truth,
            )
        )
        event_kinds.extend(
            event.kind for event in read.events
        )

    final_read = model.advance(
        4,
        unique[entity_order][:, outcome_order],
    )

    return {
        "protocol": PROTOCOL,
        "case": "lifecycle",
        "family": family,
        "seed": seed,
        "candidate_count": final_read.fit.candidate_count,
        "coarse_f1": f1s[0],
        "split_f1": f1s[1],
        "merge_f1": f1s[2],
        "retract_f1": f1s[3],
        "created_seen": "created" in event_kinds,
        "split_seen": "split" in event_kinds,
        "merge_seen": "merge" in event_kinds,
        "retracted_seen": "retracted" in event_kinds,
        "final_active_concepts": len(
            final_read.active_concepts
        ),
        "persistent_empty_state": (
            len(final_read.events) == 0
        ),
    }


def campaign(seeds: tuple[int, ...]) -> list[dict]:
    rows = []
    for seed in seeds:
        rows.append(grouped_fixture("two-concept", seed))
        rows.append(grouped_fixture("three-concept", seed))
        rows.append(singleton_control(seed))
        rows.append(ambiguous_member(seed))
        rows.append(lifecycle_case(seed))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--development", action="store_true")
    group.add_argument("--official", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    seeds = (
        OFFICIAL_SEEDS
        if args.official
        else DEVELOPMENT_SEEDS
    )
    rows = campaign(seeds)

    args.output.write_text(
        "".join(
            json.dumps(row, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
