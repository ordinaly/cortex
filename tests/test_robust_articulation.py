from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))

from robust_articulation import RobustArticulator
from robust_articulation_campaign import run_case


def test_single_observation_does_not_create_permanent_identity():
    articulator = RobustArticulator(commit_support=3)
    token = articulator.observe_frame([[1.0, 0.0, 0.0]])[0]
    assert token.provisional is True
    assert articulator.committed_entities == 0


def test_repeated_tracklet_commits_and_resolves_prior_tokens():
    articulator = RobustArticulator(commit_support=3)
    tokens = [
        articulator.observe_frame([[1.0, 0.01 * i, 0.0]])[0]
        for i in range(3)
    ]
    assert articulator.committed_entities == 1
    assert all(articulator.resolve(token) == 0 for token in tokens)


def test_ambiguous_tracklet_can_reconcile_to_existing_identity():
    articulator = RobustArticulator(
        accept_distance=0.02,
        provisional_distance=0.95,
        reconcile_distance=0.7,
        commit_support=3,
    )
    for _ in range(3):
        articulator.observe_frame([[1.0, 0.0, 0.0]])
    assert articulator.committed_entities == 1

    tokens = [
        articulator.observe_frame([[1.0, 0.35, (-1) ** i * 0.25]])[0]
        for i in range(3)
    ]
    assert articulator.committed_entities == 1
    assert articulator.reconciliations >= 1
    assert all(articulator.resolve(token) == 0 for token in tokens)


def test_robust_layer_improves_high_noise_identity_characterization():
    rows = [run_case(seed, 0.25, 0.0) for seed in range(4)]
    cortex_purity = np.mean([row["cortex"]["purity"] for row in rows])
    robust_purity = np.mean([row["robust"]["purity"] for row in rows])
    cortex_fragmentation = np.mean(
        [row["cortex"]["fragmentation"] for row in rows]
    )
    robust_fragmentation = np.mean(
        [row["robust"]["fragmentation"] for row in rows]
    )
    assert robust_purity > cortex_purity + 0.25
    assert robust_fragmentation < cortex_fragmentation
