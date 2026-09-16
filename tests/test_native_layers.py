from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from cortex import Articulation, FuzzyMemory

REF_ROOT = Path(__file__).resolve().parents[1] / "reference" / "v0_9_5"
if str(REF_ROOT) not in sys.path:
    sys.path.insert(0, str(REF_ROOT))

from cortex_v07 import DependencyIntegratedRuntime, Detection, Frame  # noqa: E402
from cortex_v08 import FuzzyAccordionMemory as ReferenceMemory  # noqa: E402


def _frame(vectors: list[np.ndarray]) -> Frame:
    return Frame(
        detections=tuple(Detection(np.asarray(x, dtype=float)) for x in vectors),
        relation_obs=tuple(),
        intervention_src_det=None,
        outcomes=tuple(),
    )


def test_native_articulation_matches_frozen_entity_invariance_path():
    dim = 8
    py = DependencyIntegratedRuntime(feature_dim=dim, max_entities=8)
    rs = Articulation(feature_dim=dim, max_entities=8)

    a = np.array([0.0, 0.2, 1.1, -0.4, 0.7, -0.9, 0.35, 0.05])
    b = np.array([1.3, -0.2, 0.45, 0.95, -0.65, 0.1, -0.85, 0.55])
    schedule = [(0, 0), (2, 4), (4, 6), (6, 2), (0, 4), (4, 0), (2, 6), (6, 4),
                (0, 2), (2, 0), (4, 2), (6, 6), (0, 0), (2, 4), (4, 6), (6, 2)]

    for ga, gb in schedule:
        obs = [np.roll(a, ga), np.roll(b, gb)]
        py_read = py.step(_frame(obs))
        rs_read = rs.step([x.tolist() for x in obs])

        assert list(rs_read.bindings) == list(py_read.bindings)
        assert list(rs_read.subgroup) == list(py_read.subgroup)
        assert rs_read.subgroup_margin == pytest.approx(py_read.subgroup_margin, abs=1e-11)
        assert rs_read.active_entities == len(py.state.entities)

        snap = json.loads(rs.snapshot_json())
        assert snap["active_entities"] == len(py.state.entities)
        assert snap["subgroup"] == list(py.state.subgroup)
        assert snap["subgroup_margin"] == pytest.approx(py.state.subgroup_margin, abs=1e-11)
        assert np.asarray(snap["prototypes"]) == pytest.approx(np.asarray(py.state.prototypes), abs=1e-12)
        assert np.asarray(snap["reliability"]) == pytest.approx(np.asarray(py.state.reliability), abs=1e-12)
        assert np.array_equal(np.asarray(snap["noise_state"], dtype=np.int8), np.asarray(py.state.noise_state))


def test_native_fuzzy_memory_matches_frozen_v08_reference():
    kwargs = dict(
        dim=3,
        budget=3,
        tau=0.20,
        alpha=0.10,
        fit_tolerance=0.06,
        distortion_budget=0.12,
        redundancy_tolerance=0.10,
        decay=0.99,
        recompress_interval=1,
    )
    py = ReferenceMemory(**kwargs)
    rs = FuzzyMemory(**kwargs)

    stream = [
        [0.00, 0.00, 0.00],
        [0.03, 0.00, 0.00],
        [0.09, 0.00, 0.00],
        [1.00, 1.00, 1.00],
        [1.03, 1.00, 1.00],
        [-1.00, -1.00, -1.00],
        [0.14, 0.00, 0.00],
        [2.50, -2.00, 1.75],
        [1.06, 1.02, 1.00],
        [-1.03, -1.00, -1.00],
    ]

    for raw in stream:
        x = np.asarray(raw, dtype=float)
        py_read = py.step(x)
        rs_read = rs.step(raw)

        assert np.asarray(rs_read.reconstruction) == pytest.approx(py_read.reconstruction, abs=1e-12)
        assert np.asarray(rs_read.memberships) == pytest.approx(py_read.memberships, abs=1e-12)
        assert list(rs_read.active_ids) == list(py_read.active_ids)
        assert rs_read.nearest_id == py_read.nearest_id
        if math.isinf(py_read.nearest_dist):
            assert math.isinf(rs_read.nearest_dist)
        else:
            assert rs_read.nearest_dist == pytest.approx(py_read.nearest_dist, abs=1e-12)
        assert rs_read.stored == py_read.stored
        assert rs_read.unresolved == py_read.unresolved
        assert rs_read.compressed == py_read.compressed
        assert rs_read.structural_change == py_read.structural_change

    snap = json.loads(rs.snapshot_json())
    assert np.asarray(snap["prototypes"]) == pytest.approx(np.asarray(py.prototypes), abs=1e-12)
    assert snap["counts"] == pytest.approx(py.counts, abs=1e-12)
    assert snap["utility"] == pytest.approx(py.utility, abs=1e-12)
    assert snap["last_used"] == py.last_used
    assert snap["unresolved_count"] == py.unresolved_count
    assert snap["recompression_count"] == py.recompression_count
    assert snap["spawn_count"] == py.spawn_count
    assert snap["prototype_updates"] == py.prototype_updates
