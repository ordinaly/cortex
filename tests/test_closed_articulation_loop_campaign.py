from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))

from closed_articulation_loop_campaign import campaign


def test_closed_loop_owr_probe_uses_one_policy_and_preserves_stage_boundaries():
    report = campaign(
        worlds_per_stage=1,
        material="closed-loop-test-fixture",
    )

    assert report["mode"] == "development-probe"
    assert set(report["aggregate"]) == {"A", "B", "C", "D"}
    assert report["aggregate"]["D"]["hard_failure_rate"] == 0.0
    assert report["aggregate"]["D"]["mean_stored_basins"] <= (
        report["policy"]["active_budget"] + report["policy"]["latent_budget"]
    )
    assert 0.0 <= report["aggregate"]["A"]["target"]["coverage"] <= 1.0
    assert 0.0 <= report["aggregate"]["B"]["all"]["selective_risk"] <= 1.0
    assert 0.0 <= report["aggregate"]["C"]["local_relation"]["coverage"] <= 1.0
