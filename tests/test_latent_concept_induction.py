from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))

from latent_concept_induction import (
    CortexLatentConceptInducer,
    CortexLatentConceptLifecycle,
    canonical_partitions,
    pairwise_partition_f1,
)


def test_six_entities_have_203_candidate_partitions():
    assert len(canonical_partitions(6)) == 203


def test_predictive_compression_recovers_two_groups():
    counts = np.vstack(
        [
            np.asarray([28, 5, 3, 2, 1, 1], dtype=float)
            for _ in range(3)
        ]
        + [
            np.asarray([1, 1, 2, 3, 5, 28], dtype=float)
            for _ in range(3)
        ]
    )
    model = CortexLatentConceptInducer(counts)
    assert model.fit.resolved
    assert pairwise_partition_f1(
        model.fit.best_partition,
        (0, 0, 0, 1, 1, 1),
    ) == 1.0
    assert len(model.resolved_blocks()) == 2


def test_zero_evidence_member_stays_membership_unresolved():
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
    model = CortexLatentConceptInducer(counts)
    assert not model.fit.resolved
    assert len(model.fit.best_partitions) == 2
    assert not model.predict_entity(5).resolved


def test_singleton_world_invents_no_nontrivial_concept():
    counts = []
    for entity in range(6):
        vector = np.full(6, 2.0)
        vector[entity] = 30.0
        counts.append(vector)
    model = CortexLatentConceptInducer(
        np.asarray(counts)
    )
    assert model.fit.best_partition == (0, 1, 2, 3, 4, 5)
    assert model.resolved_blocks() == tuple()


def test_lifecycle_create_split_merge_retract():
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

    model = CortexLatentConceptLifecycle(6)
    reads = [
        model.advance(0, coarse),
        model.advance(1, split),
        model.advance(2, coarse),
        model.advance(3, unique),
    ]
    kinds = [
        event.kind
        for read in reads
        for event in read.events
    ]
    assert "created" in kinds
    assert "split" in kinds
    assert "merge" in kinds
    assert "retracted" in kinds
    assert reads[-1].active_concepts == tuple()


def test_unchanged_block_preserves_concept_identity():
    counts = np.vstack(
        [
            np.asarray([28, 5, 3, 2, 1, 1], dtype=float)
            for _ in range(3)
        ]
        + [
            np.asarray([1, 1, 2, 3, 5, 28], dtype=float)
            for _ in range(3)
        ]
    )
    model = CortexLatentConceptLifecycle(6)
    first = model.advance(0, counts)
    second = model.advance(1, counts)
    assert first.active_concepts == second.active_concepts
    assert second.events == tuple()
