from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))

from learned_world_model import (
    heldout_true_pairs,
    probe_distribution,
    run_case,
)


def test_probe_distributions_are_learnable_but_not_deterministic():
    for entity in range(24):
        q = probe_distribution(entity)
        assert abs(float(q.sum()) - 1.0) < 1.0e-12
        assert float(np.max(q)) < 0.90
        assert float(np.max(q)) > 0.60


def test_heldout_pairs_cover_all_semantic_conjunctions():
    held = heldout_true_pairs()
    assert len(held) == 16
    truths = [truth for _s, _t, truth in held]
    assert [truths.count(relation) for relation in range(2, 6)] == [4, 4, 4, 4]


def test_learned_semantics_support_prospective_prediction():
    rows = [
        run_case(
            seed,
            perception_noise=0.03,
            probes_per_entity=40,
            interactions_per_pair=8,
        )
        for seed in range(4)
    ]
    purity = np.mean([row["identity"]["purity"] for row in rows])
    separation = np.mean(
        [row["semantic"]["separation_ratio"] for row in rows]
    )
    hit = np.mean([row["final_full_hit_at_1"] for row in rows])
    interaction_only = np.mean(
        [row["final_interaction_only_hit_at_1"] for row in rows]
    )
    assert purity > 0.98
    assert separation > 2.0
    # The finite smoke test should establish a real predictive advantage,
    # while the larger campaign measures the effect size. Chance among the
    # four rare types is 0.25.
    assert hit > 0.60
    assert hit > interaction_only + 0.20
    assert interaction_only < 0.50


def test_destroying_semantic_alignment_hurts_prediction():
    rows = [
        run_case(
            seed,
            perception_noise=0.03,
            probes_per_entity=40,
            interactions_per_pair=8,
        )
        for seed in range(5)
    ]
    full = np.mean([row["final_full_hit_at_1"] for row in rows])
    shuffled = np.mean([row["final_shuffled_hit_at_1"] for row in rows])
    assert full > shuffled + 0.25


def test_harder_perception_or_sparse_semantic_evidence_can_degrade():
    easy = np.mean(
        [
            run_case(
                seed,
                perception_noise=0.03,
                probes_per_entity=80,
                interactions_per_pair=6,
            )["final_full_hit_at_1"]
            for seed in range(3)
        ]
    )
    hard = np.mean(
        [
            run_case(
                seed,
                perception_noise=0.25,
                probes_per_entity=5,
                interactions_per_pair=6,
            )["final_full_hit_at_1"]
            for seed in range(3)
        ]
    )
    assert easy > hard


def test_oracle_identity_localizes_high_noise_failure_to_articulation():
    rows = [
        run_case(
            seed,
            perception_noise=0.25,
            probes_per_entity=40,
            interactions_per_pair=6,
        )
        for seed in range(4)
    ]
    cortex_bound = np.mean([row["final_full_hit_at_1"] for row in rows])
    oracle_bound = np.mean(
        [row["final_oracle_identity_hit_at_1"] for row in rows]
    )
    assert oracle_bound > 0.65
    assert oracle_bound > cortex_bound + 0.20
