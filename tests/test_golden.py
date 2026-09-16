from __future__ import annotations
import gzip, json, math
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / "tests" / "fixtures"

try:
    from cortex import Cortex, NATIVE_AVAILABLE
except Exception:
    NATIVE_AVAILABLE = False

pytestmark = pytest.mark.skipif(not NATIVE_AVAILABLE, reason="native Rust extension not built")


def close(a, b, atol=2e-8):
    return math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=atol)

@pytest.mark.parametrize("name", ["update_seed100", "split_seed100", "novel_seed100", "merge_seed300"])
def test_native_replays_frozen_python_trace(name):
    meta = json.loads((FIX / f"{name}.final.json").read_text())
    cfg = json.dumps(meta["config"])
    model = Cortex(dim=meta["config"]["dim"], budget=meta["config"]["budget"], config_json=cfg)
    with gzip.open(FIX / f"{name}.jsonl.gz", "rt", encoding="utf8") as f:
        for step, line in enumerate(f, 1):
            exp = json.loads(line)
            got = model.step(exp["x"], float(exp["y"]))
            assert close(got.prediction, exp["prediction"], 3e-8), (name, step, got.prediction, exp["prediction"])
            assert got.current_id == exp["current_id"], (name, step, "current")
            assert got.stored == exp["stored"], (name, step, "stored")
            assert got.revision == exp["revision"], (name, step, "revision")
            assert got.reactivated == exp["reactivated"], (name, step, "reactivated")
            assert got.discovered == exp["discovered"], (name, step, "discovered")
            assert got.unresolved == exp["unresolved"], (name, step, "unresolved")
            assert got.budget_pressure == exp["budget_pressure"], (name, step, "budget")
            assert close(got.nearest_dist, exp["nearest_dist"], 3e-8), (name, step, "distance")

    snap = json.loads(model.snapshot_json())
    exp = meta["final"]
    for key in ["t","current_id","pending_kind","pending_id","pending_count","revision_count","reactivation_count",
                "discovery_count","unresolved_count","budget_pressure_count","recompression_count","split_proposals",
                "split_rejections","split_promotions","merge_promotions","rank_checks","curvature_updates"]:
        assert snap[key] == exp[key], (name, key, snap[key], exp[key])
    for key in ["pending_score","sigma2","hazard"]:
        assert close(snap[key], exp[key], 5e-8), (name, key)
    assert len(snap["prototypes"]) == len(exp["prototypes"])
    for a, b in zip(snap["prototypes"], exp["prototypes"]):
        for key in ["count","successes","failures","utility","recent_mean"]:
            assert close(a[key], b[key], 5e-8), (name, key)
        for key in ["last_used","created_t","lineage_id","split_created_t","recent_obs","refinement_depth"]:
            assert a[key] == b[key], (name, key)
        assert len(a["centroid"]) == len(b["centroid"])
        assert all(close(x,y,5e-8) for x,y in zip(a["centroid"],b["centroid"])), (name,"centroid")
