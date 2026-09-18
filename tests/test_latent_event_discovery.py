from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))

from latent_event_discovery import LatentEventDiscoverer, event_center
from latent_event_discovery_campaign import run_case


def test_event_discoverer_uses_no_label_api():
    discoverer = LatentEventDiscoverer()
    cluster = discoverer.observe(event_center(0))
    assert cluster == 0
    assert discoverer.observe(event_center(0) + 0.01) == 0


def test_clean_event_dynamics_are_discovered_with_high_purity():
    rows = [run_case(seed, 0.08, events=400) for seed in range(5)]
    assert np.mean([row["purity"] for row in rows]) > 0.95
    assert np.mean([row["pair_f1"] for row in rows]) > 0.90


def test_rare_discovered_event_has_finer_specificity_than_common_event():
    rows = [run_case(seed, 0.08, events=600) for seed in range(5)]
    assert np.mean(
        [float(row["rare_more_specific"]) for row in rows]
    ) > 0.80


def test_event_discovery_has_a_noise_failure_regime():
    easy = np.mean([run_case(seed, 0.08, 400)["pair_f1"] for seed in range(4)])
    hard = np.mean([run_case(seed, 0.30, 400)["pair_f1"] for seed in range(4)])
    assert easy > hard
