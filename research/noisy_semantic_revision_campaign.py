"""Noisy confidence/revision campaign for Cortex semantic grounding."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from random import Random
from statistics import mean
from typing import Mapping

from semantic_grounding import NeuralLabelEvidence, StructuralSemanticObservation
from semantic_grounding_campaign import FAMILY_EDGES, RELATIONS, build_world
from semantic_revision import CortexSemanticRevision

PROTOCOL = "noisy-semantic-revision-v1"
DEVELOPMENT_SEEDS = tuple(range(24))
OFFICIAL_SEEDS = tuple(range(500, 520))

ESTABLISH_EPOCHS = 6
STEADY_EPOCHS = 10
AMBIGUOUS_EPOCHS = 6
SHIFT_EPOCHS = 14
RECOVERY_EPOCHS = 14

NEURAL_MISSING = 0.10
NEURAL_ERROR = 0.18
STRUCTURAL_OBSERVATION = 0.35
STRUCTURAL_FLIP = 0.08


def _family_code(family: str) -> int:
    return sum((index + 1) * ord(ch) for index, ch in enumerate(family))


def _swapped_mapping(
    truth: Mapping[str, str],
    concepts: list[str],
    seed: int,
    family: str,
) -> tuple[dict[str, str], tuple[str, str]]:
    rng = Random(seed * 3571 + _family_code(family) * 17 + 41)
    order = list(concepts)
    rng.shuffle(order)
    left, right = order[:2]
    shifted = dict(truth)
    shifted[left], shifted[right] = shifted[right], shifted[left]
    return shifted, (left, right)


def _distribution(
    labels: list[str],
    intended: str,
    *,
    wrong: bool,
    rng: Random,
) -> tuple[dict[str, float], bool]:
    candidates = [label for label in labels if label != intended]
    top = rng.choice(candidates) if wrong else intended
    confidence = rng.uniform(0.58, 0.90)
    remaining = (1.0 - confidence) / (len(labels) - 1)
    probabilities = {label: remaining for label in labels}
    probabilities[top] = confidence
    return probabilities, top == intended


def noisy_neural(
    concepts: list[str],
    labels: list[str],
    evidence_mapping: Mapping[str, str],
    *,
    seed: int,
    family: str,
    epoch: int,
    error_rate: float = NEURAL_ERROR,
    missing_rate: float = NEURAL_MISSING,
) -> tuple[list[NeuralLabelEvidence], dict[str, dict[str, float]]]:
    rng = Random(
        seed * 100003
        + _family_code(family) * 1009
        + epoch * 7919
        + 3
    )
    evidence: list[NeuralLabelEvidence] = []
    raw: dict[str, dict[str, float]] = {}
    for concept in concepts:
        if rng.random() < missing_rate:
            continue
        wrong = rng.random() < error_rate
        probabilities, correct_top = _distribution(
            labels,
            evidence_mapping[concept],
            wrong=wrong,
            rng=rng,
        )
        reliability = rng.uniform(0.55, 1.0)
        if not correct_top:
            reliability *= rng.uniform(0.55, 0.82)
        item = NeuralLabelEvidence(
            concept=concept,
            probabilities=probabilities,
            reliability=reliability,
        )
        evidence.append(item)
        raw[concept] = probabilities
    return evidence, raw


def noisy_structure(
    concepts: list[str],
    evidence_mapping: Mapping[str, str],
    semantic_edges,
    *,
    seed: int,
    family: str,
    epoch: int,
    observation_rate: float = STRUCTURAL_OBSERVATION,
    flip_rate: float = STRUCTURAL_FLIP,
    reliability_scale: float = 1.0,
) -> list[StructuralSemanticObservation]:
    rng = Random(
        seed * 200003
        + _family_code(family) * 2017
        + epoch * 6151
        + 11
    )
    observations: list[StructuralSemanticObservation] = []
    for source in concepts:
        for target in concepts:
            if source == target:
                continue
            for relation in RELATIONS:
                if rng.random() >= observation_rate:
                    continue
                present = (
                    evidence_mapping[source],
                    relation,
                    evidence_mapping[target],
                ) in semantic_edges
                flipped = rng.random() < flip_rate
                if flipped:
                    present = not present
                reliability = (
                    rng.uniform(0.55, 1.0)
                    * reliability_scale
                    * (rng.uniform(0.50, 0.78) if flipped else 1.0)
                )
                observations.append(
                    StructuralSemanticObservation(
                        source=source,
                        relation=relation,
                        target=target,
                        present=present,
                        reliability=max(0.05, reliability),
                    )
                )
    return observations


def _multiclass_brier(
    probabilities: Mapping[str, float],
    truth: str,
) -> float:
    return sum(
        (probability - float(label == truth)) ** 2
        for label, probability in probabilities.items()
    )


def _ece(samples: list[tuple[float, bool]], bins: int = 10) -> float:
    if not samples:
        return 0.0
    total = len(samples)
    error = 0.0
    for index in range(bins):
        lo = index / bins
        hi = (index + 1) / bins
        bucket = [
            (confidence, correct)
            for confidence, correct in samples
            if (
                lo <= confidence < hi
                or (index == bins - 1 and confidence == 1.0)
            )
        ]
        if not bucket:
            continue
        confidence = mean(value for value, _ in bucket)
        accuracy = mean(int(correct) for _, correct in bucket)
        error += (len(bucket) / total) * abs(accuracy - confidence)
    return error


def _phase_mapping(
    phase: str,
    base: Mapping[str, str],
    shifted: Mapping[str, str],
) -> Mapping[str, str]:
    return shifted if phase == "shift" else base


def run_case(family: str, seed: int) -> dict:
    concepts, labels, _concept_edges, semantic_edges, truth = build_world(
        family,
        seed,
    )
    concepts = list(concepts)
    labels = list(labels)
    shifted, changed = _swapped_mapping(truth, concepts, seed, family)
    unchanged = [concept for concept in concepts if concept not in changed]

    model = CortexSemanticRevision(
        concepts,
        labels,
        RELATIONS,
        semantic_edges,
    )

    neural_briers: list[float] = []
    cortex_briers: list[float] = []
    neural_correct: list[int] = []
    cortex_top_correct: list[int] = []
    posterior_calibration: list[tuple[float, bool]] = []
    trusted_calibration: list[tuple[float, bool]] = []
    trusted_total = 0
    trusted_correct = 0
    wrong_accepted_total = 0
    wrong_accepted_untrusted = 0

    phase_epochs = {
        "establish": ESTABLISH_EPOCHS,
        "steady": STEADY_EPOCHS,
        "ambiguous": AMBIGUOUS_EPOCHS,
        "shift": SHIFT_EPOCHS,
        "recovery": RECOVERY_EPOCHS,
    }

    epoch = 0
    shift_start = None
    recovery_start = None
    shift_resolved_epoch = None
    recovery_resolved_epoch = None
    stable_snapshot = None

    for phase, count in phase_epochs.items():
        if phase == "shift":
            shift_start = epoch
        if phase == "recovery":
            recovery_start = epoch

        for offset in range(count):
            truth_mapping = _phase_mapping(phase, truth, shifted)

            if phase == "ambiguous" and offset % 2 == 0:
                evidence_mapping = shifted
                structural_mapping = truth
                structural_scale = 0.55
            else:
                evidence_mapping = truth_mapping
                structural_mapping = truth_mapping
                structural_scale = 1.0

            neural, raw_neural = noisy_neural(
                concepts,
                labels,
                evidence_mapping,
                seed=seed,
                family=family,
                epoch=epoch,
            )
            structural = noisy_structure(
                concepts,
                structural_mapping,
                semantic_edges,
                seed=seed,
                family=family,
                epoch=epoch,
                reliability_scale=structural_scale,
            )

            reads = model.advance(
                epoch,
                neural=neural,
                structural=structural,
            )
            reads_by_concept = {read.concept: read for read in reads}

            if phase == "establish" and offset == count - 1:
                stable_snapshot = {
                    concept: reads_by_concept[concept].accepted_label
                    for concept in concepts
                }

            if phase != "establish":
                for concept in concepts:
                    expected = truth_mapping[concept]
                    read = reads_by_concept[concept]
                    posterior = dict(read.grounding.posterior)
                    top_label, top_confidence = read.grounding.posterior[0]
                    top_is_correct = top_label == expected
                    cortex_briers.append(
                        _multiclass_brier(posterior, expected)
                    )
                    cortex_top_correct.append(int(top_is_correct))
                    posterior_calibration.append(
                        (top_confidence, top_is_correct)
                    )

                    if concept in raw_neural:
                        neural_distribution = raw_neural[concept]
                        neural_top = max(
                            neural_distribution.items(),
                            key=lambda item: item[1],
                        )[0]
                        neural_briers.append(
                            _multiclass_brier(
                                neural_distribution,
                                expected,
                            )
                        )
                        neural_correct.append(
                            int(neural_top == expected)
                        )

                    if read.trusted:
                        trusted_total += 1
                        correct = read.accepted_label == expected
                        trusted_correct += int(correct)
                        trusted_calibration.append(
                            (read.accepted_label_support, correct)
                        )

                    if (
                        read.accepted_label is not None
                        and read.accepted_label != expected
                    ):
                        wrong_accepted_total += 1
                        wrong_accepted_untrusted += int(not read.trusted)

            if (
                phase == "shift"
                and shift_resolved_epoch is None
                and all(
                    reads_by_concept[concept].accepted_label
                    == shifted[concept]
                    and reads_by_concept[concept].trusted
                    for concept in changed
                )
            ):
                shift_resolved_epoch = epoch

            if (
                phase == "recovery"
                and recovery_resolved_epoch is None
                and all(
                    reads_by_concept[concept].accepted_label
                    == truth[concept]
                    and reads_by_concept[concept].trusted
                    for concept in changed
                )
            ):
                recovery_resolved_epoch = epoch

            epoch += 1

    pre_shift_boundary = shift_start if shift_start is not None else 0
    false_revision_events = [
        event for event in model.events
        if event.epoch < pre_shift_boundary
        and event.kind in {"withdrawn", "revised"}
    ]
    unaffected_revision_events = [
        event for event in model.events
        if event.concept in unchanged
        and event.kind in {"withdrawn", "revised"}
    ]

    shift_success = shift_resolved_epoch is not None
    recovery_success = recovery_resolved_epoch is not None
    shift_latency = (
        shift_resolved_epoch - shift_start + 1
        if shift_success and shift_start is not None
        else SHIFT_EPOCHS + 1
    )
    recovery_latency = (
        recovery_resolved_epoch - recovery_start + 1
        if recovery_success and recovery_start is not None
        else RECOVERY_EPOCHS + 1
    )

    initial_correct = (
        stable_snapshot is not None
        and all(
            stable_snapshot[concept] == truth[concept]
            for concept in concepts
        )
    )

    return {
        "protocol": PROTOCOL,
        "family": family,
        "seed": seed,
        "initial_correct": initial_correct,
        "neural_top_accuracy": mean(neural_correct) if neural_correct else 0.0,
        "cortex_top_accuracy": mean(cortex_top_correct),
        "neural_brier": mean(neural_briers) if neural_briers else 0.0,
        "cortex_brier": mean(cortex_briers),
        "posterior_ece": _ece(posterior_calibration),
        "trusted_ece": _ece(trusted_calibration),
        "trusted_coverage": trusted_total / (
            (STEADY_EPOCHS + AMBIGUOUS_EPOCHS + SHIFT_EPOCHS + RECOVERY_EPOCHS)
            * len(concepts)
        ),
        "trusted_accuracy": trusted_correct / max(1, trusted_total),
        "false_certainty_rate": 1.0 - (
            trusted_correct / max(1, trusted_total)
        ),
        "uncertainty_interception_rate": (
            wrong_accepted_untrusted / max(1, wrong_accepted_total)
        ),
        "false_revision_events": len(false_revision_events),
        "shift_success": shift_success,
        "shift_latency": shift_latency,
        "recovery_success": recovery_success,
        "recovery_latency": recovery_latency,
        "unaffected_revision_events": len(unaffected_revision_events),
        "bounded_neural_history": (
            model.neural_history_size
            <= model.evidence_horizon * len(concepts)
        ),
        "bounded_structural_history": (
            model.structural_history_size
            <= model.evidence_horizon
            * len(concepts)
            * (len(concepts) - 1)
            * len(RELATIONS)
        ),
    }


def campaign(seeds) -> list[dict]:
    return [
        run_case(family, seed)
        for seed in seeds
        for family in FAMILY_EDGES
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--development", action="store_true")
    group.add_argument("--official", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    seeds = OFFICIAL_SEEDS if args.official else DEVELOPMENT_SEEDS
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
