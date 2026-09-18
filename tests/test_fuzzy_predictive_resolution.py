from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))

from fuzzy_predictive_field_campaign import run_case


def test_predictive_sufficiency_has_a_coarsening_frontier():
    resolutions = (0.02, 0.10, 0.20, 0.30, 0.50)
    summaries = []
    for resolution in resolutions:
        rows = [
            run_case(seed, semantic_resolution=resolution)
            for seed in range(5)
        ]
        summaries.append(
            (
                resolution,
                np.mean(
                    [row["fuzzy_future_hit_at_1"] for row in rows]
                ),
                np.mean(
                    [row["predictive_effective_rank"] for row in rows]
                ),
            )
        )

    best = max(hit for _resolution, hit, _rank in summaries)
    sufficient = [
        row for row in summaries if row[1] >= best - 0.04
    ]
    selected = max(sufficient, key=lambda row: row[0])

    # The field should compress substantially before forecast quality breaks.
    assert selected[0] >= 0.10
    assert selected[2] < 10.0
    assert selected[1] > 0.80

    # Over-coarsening should eventually destroy predictive distinctions.
    coarse = next(row for row in summaries if row[0] == 0.50)
    assert coarse[1] < selected[1] - 0.30
