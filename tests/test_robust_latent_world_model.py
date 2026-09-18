from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))

from robust_latent_world_model import run_case


def test_combined_latent_event_channel_is_recovered():
    rows = [
        run_case(
            seed,
            perception_noise=0.12,
            probes_per_entity=16,
            events_per_pair=5,
        )
        for seed in range(2)
    ]
    assert np.mean(
        [row["latent_event"]["pair_f1"] for row in rows]
    ) > 0.90


def test_robust_identity_improves_downstream_latent_event_prediction():
    rows = [
        run_case(
            seed,
            perception_noise=0.25,
            probes_per_entity=20,
            events_per_pair=6,
        )
        for seed in range(3)
    ]
    native_purity = np.mean(
        [row["native_identity"]["purity"] for row in rows]
    )
    robust_purity = np.mean(
        [row["robust_identity"]["purity"] for row in rows]
    )
    native_hit = np.mean(
        [row["native_prediction"]["full_hit_at_1"] for row in rows]
    )
    robust_hit = np.mean(
        [row["robust_prediction"]["full_hit_at_1"] for row in rows]
    )
    oracle_hit = np.mean(
        [row["oracle_prediction"]["full_hit_at_1"] for row in rows]
    )

    assert robust_purity > native_purity + 0.40
    assert robust_hit > native_hit + 0.15
    assert oracle_hit >= robust_hit - 0.20
