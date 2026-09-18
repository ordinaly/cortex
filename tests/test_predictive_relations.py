from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))

from predictive_relations import (
    PairEvidence,
    heat_kernel,
    learned_affinity,
    run_seed,
)


def test_unexposed_pair_has_no_direct_affinity():
    adjacency = learned_affinity(
        3,
        {(0, 1): PairEvidence(successes=9, exposures=10)},
    )
    assert adjacency[0, 2] == 0.0


def test_heat_kernel_is_zero_across_disconnected_components():
    adjacency = np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
    )
    kernel = heat_kernel(adjacency, 1.0)
    assert abs(kernel[0, 2]) < 1.0e-12
    assert abs(kernel[1, 3]) < 1.0e-12


def test_campaign_has_no_direct_evidence_leakage():
    row = run_seed(0)
    assert row["all_evaluated_pairs_unexposed"] is True


def test_deep_relation_signal_beats_two_hop_common_neighbors():
    # The missing actor/receiver relation is supported by odd three-step paths
    # in the bipartite training graph. Two-hop common neighbors should carry no
    # discriminating signal, while the diffusion potential should.
    rows = [run_seed(seed) for seed in range(12)]
    heat_auc = np.mean([r["metrics"]["heat-kernel"]["auc"] for r in rows])
    common_auc = np.mean(
        [r["metrics"]["common-neighbors"]["auc"] for r in rows]
    )
    direct_auc = np.mean([r["metrics"]["direct-prior"]["auc"] for r in rows])
    assert direct_auc == 0.5
    assert common_auc == 0.5
    assert heat_auc > 0.95


def test_heat_kernel_predicts_heldout_future_pairs_across_seeds():
    rows = [run_seed(seed) for seed in range(20)]
    auc = np.mean([r["metrics"]["heat-kernel"]["auc"] for r in rows])
    ap = np.mean(
        [r["metrics"]["heat-kernel"]["average_precision"] for r in rows]
    )
    assert auc > 0.97
    assert ap > 0.90
