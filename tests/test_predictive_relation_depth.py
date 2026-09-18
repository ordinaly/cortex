from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))

from predictive_relation_depth import (
    heat_scores,
    make_query_graph,
    path_scores,
    run_seed,
)


def test_first_fixed_path_signal_moves_with_relation_depth():
    for depth in (3, 5, 7):
        adjacency, source, targets = make_query_graph(depth, seed=depth)
        for order in (3, 5, 7):
            scores = path_scores(adjacency, source, targets, order)
            if order < depth:
                assert all(abs(score) < 1.0e-12 for score in scores)
            elif order == depth:
                assert scores[0] > 0.0


def test_heat_kernel_exposes_all_tested_depths_without_changing_operator():
    for depth in (3, 5, 7):
        adjacency, source, targets = make_query_graph(depth, seed=100 + depth)
        scores = heat_scores(adjacency, source, targets, rho=1.0)
        assert scores[0] > max(scores[1:])


def test_variable_depth_campaign_prefers_future_positive():
    rows = [run_seed(seed) for seed in range(10)]
    for depth in ("3", "5", "7"):
        heat_hit1 = np.mean(
            [row["depths"][depth]["heat-kernel"]["hit_at_1"] for row in rows]
        )
        assert heat_hit1 > 0.95


def test_path3_fails_when_first_support_is_deeper_than_three():
    rows = [run_seed(seed) for seed in range(5)]
    for depth in ("5", "7"):
        hit1 = np.mean(
            [row["depths"][depth]["path3"]["hit_at_1"] for row in rows]
        )
        assert abs(hit1 - 0.25) < 1.0e-12
