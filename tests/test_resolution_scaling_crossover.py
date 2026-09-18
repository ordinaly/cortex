from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))

from resolution_scaling_crossover_campaign import run_case


def test_small_scaling_case_preserves_dense_exact_behavior():
    row = run_case(0, 9)
    assert (
        row["dense_exact_max_prediction_error"]
        <= 1.0e-10
    )
    assert (
        row["dense_exact_resolution_disagreement"]
        == 0.0
    )
    assert row["sparse_mean_tv"] <= 0.02
    assert (
        row["sparse_resolution_disagreement"]
        <= 0.10
    )


def test_medium_scaling_case_reports_optimization_duty():
    row = run_case(0, 18)
    assert row["dense_sampled_us_per_step"] > 0
    assert row["sparse_us_per_step"] > 0
    assert (
        0.0
        <= row["sparse_kernel_row_refresh_fraction"]
        <= 1.0
    )
    assert (
        0.0
        <= row["sparse_anchor_scan_fraction"]
        <= 1.0
    )
    assert 0.0 <= row["sparse_svd_fraction"] <= 1.0
