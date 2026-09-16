from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from cortex import CortexRuntime

REF_ROOT = Path(__file__).resolve().parents[1] / "reference" / "v0_9_5"
if str(REF_ROOT) not in sys.path:
    sys.path.insert(0, str(REF_ROOT))

from cortex_v07 import SyntheticIntegratedWorld  # noqa: E402
from cortex_v08 import CortexV08Runtime  # noqa: E402
from cortex_v095_compute_optimized import ComputeOptimizedRefinementReasoner  # noqa: E402


def _reference_continual(budget: int = 24) -> ComputeOptimizedRefinementReasoner:
    """Frozen v0.9.5 with the canonical RC1 native configuration.

    Only parameters whose historical Python defaults differ from the frozen RC1
    Rust configuration need to be overridden here. All remaining v0.9.4/v0.9.5
    defaults are already identical to `cortex_core::Config::default()`.
    """
    return ComputeOptimizedRefinementReasoner(
        dim=12,
        budget=budget,
        evidence_decay=0.85,
        hazard_lr=0.35,
        recurrence_threshold=0.75,
        initial_burst=24,
        post_ready_burst=8,
        refresh_interval=144,
        refresh_burst=4,
        degradation_ratio=1.0,
        degradation_patience=6,
        curvature_stride=2,
    )


def _relation_pairs(state, changed):
    return [tuple(map(int, state.pidx.pairs[int(pid)])) for pid in changed]


def _causal_pairs(state, changed):
    return [tuple(map(int, state.pidx.ordered[int(oid)])) for oid in changed]


def test_full_native_runtime_matches_frozen_composed_stack():
    feature_dim = 8
    max_entities = 16
    budget = 24

    world = SyntheticIntegratedWorld(
        n_entities=8,
        feature_dim=feature_dim,
        visible=4,
        seed=731,
        noise_sigma=0.055,
        corrupt_p=0.025,
        intervention_p=0.40,
    )
    world.schedule_local_drift(180, entity=2)

    py08 = CortexV08Runtime(
        memory_dim=12,
        memory_budget=budget,
        feature_dim=feature_dim,
        max_entities=max_entities,
    )
    py095 = _reference_continual(budget)
    rs = CortexRuntime(
        feature_dim=feature_dim,
        max_entities=max_entities,
        budget=budget,
    )

    for step in range(1, 421):
        frame = world.frame()
        py_art, py_mem = py08.step(frame)
        vector = py08.articulation_vector()
        # The historical full-stack wrapper leaves the downstream task label to
        # the caller. Use a deterministic observed target from this frame for both
        # implementations; this exercises the complete predictive/plastic path.
        outcome = float(frame.outcomes[0][1]) if frame.outcomes else None
        py_cont = py095.step(vector, outcome)

        rs_read = rs.step(
            [d.x.tolist() for d in frame.detections],
            [tuple(map(int, r)) for r in frame.relation_obs],
            frame.intervention_src_det,
            [tuple(map(int, r)) for r in frame.outcomes],
            outcome,
        )

        assert list(rs_read.bindings) == list(py_art.bindings), (step, "bindings")
        assert list(rs_read.subgroup) == list(py_art.subgroup), (step, "subgroup")
        assert rs_read.subgroup_margin == pytest.approx(py_art.subgroup_margin, abs=2e-11)
        assert list(rs_read.relation_changed) == _relation_pairs(py08.state, py_art.relation_changed)
        assert list(rs_read.causal_changed) == _causal_pairs(py08.state, py_art.causal_changed)
        assert np.asarray(rs_read.articulation_vector) == pytest.approx(vector, abs=2e-11)

        assert rs_read.memory_stored == py_mem.stored, (step, "memory stored")
        assert rs_read.memory_unresolved == py_mem.unresolved, (step, "memory unresolved")

        assert rs_read.prediction == pytest.approx(py_cont.prediction, abs=6e-8)
        assert np.asarray(rs_read.memberships) == pytest.approx(py_cont.memberships, abs=6e-8)
        assert rs_read.current_id == py_cont.current_id
        assert rs_read.nearest_id == py_cont.nearest_id
        assert rs_read.nearest_dist == pytest.approx(py_cont.nearest_dist, abs=6e-8)
        assert rs_read.revision == py_cont.revision
        assert rs_read.reactivated == py_cont.reactivated
        assert rs_read.discovered == py_cont.discovered
        assert rs_read.unresolved == py_cont.unresolved
        assert rs_read.budget_pressure == py_cont.budget_pressure
        assert rs_read.stored == py_cont.stored

    snap = json.loads(rs.snapshot_json())
    assert np.asarray(snap["articulation_vector"]) == pytest.approx(
        py08.articulation_vector(), abs=2e-11
    )

    mem = snap["memory"]
    assert np.asarray(mem["prototypes"]) == pytest.approx(
        np.asarray(py08.memory.prototypes), abs=2e-11
    )
    assert mem["counts"] == pytest.approx(py08.memory.counts, abs=2e-11)
    assert mem["unresolved_count"] == py08.memory.unresolved_count
    assert mem["recompression_count"] == py08.memory.recompression_count

    cont = snap["continual"]
    assert cont["t"] == py095.t
    assert cont["current_id"] == py095.current_id
    assert cont["revision_count"] == py095.revision_count
    assert cont["reactivation_count"] == py095.reactivation_count
    assert cont["discovery_count"] == py095.discovery_count
    assert cont["split_promotions"] == py095.split_promotions
    assert cont["merge_promotions"] == py095.merge_promotions
    assert len(cont["prototypes"]) == len(py095.prototypes)
    for got, exp in zip(cont["prototypes"], py095.prototypes):
        assert np.asarray(got["centroid"]) == pytest.approx(exp.centroid, abs=6e-8)
        assert got["count"] == pytest.approx(exp.count, abs=6e-8)
        assert got["successes"] == pytest.approx(exp.successes, abs=6e-8)
        assert got["failures"] == pytest.approx(exp.failures, abs=6e-8)
