from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from cortex import GraphEvidence

REF_ROOT = Path(__file__).resolve().parents[1] / "reference" / "v0_9_5"
if str(REF_ROOT) not in sys.path:
    sys.path.insert(0, str(REF_ROOT))

from cortex_v07 import DependencyIntegratedRuntime, SyntheticIntegratedWorld  # noqa: E402


def test_sparse_graph_evidence_matches_frozen_v07_statistics():
    """Replay v0.7 frames while allocating only pairs that receive evidence."""
    world = SyntheticIntegratedWorld(
        n_entities=6,
        feature_dim=8,
        visible=3,
        seed=2026,
        noise_sigma=0.03,
        corrupt_p=0.01,
        intervention_p=0.55,
    )
    py = DependencyIntegratedRuntime(feature_dim=8, max_entities=12)
    rs = GraphEvidence()

    for _ in range(320):
        frame = world.frame()
        py_read = py.step(frame)
        rs_read = rs.step(
            list(py_read.bindings),
            list(frame.relation_obs),
            frame.intervention_src_det,
            list(frame.outcomes),
        )

        py_rel_changed = sorted(tuple(py.state.pidx.pairs[pid]) for pid in py_read.relation_changed)
        py_causal_changed = sorted(tuple(py.state.pidx.ordered[oid]) for oid in py_read.causal_changed)
        assert sorted(tuple(x) for x in rs_read.relation_changed) == py_rel_changed
        assert sorted(tuple(x) for x in rs_read.causal_changed) == py_causal_changed

        assert rs_read.represented_relations == int((py.state.rel_exposure > 0).sum())
        represented_causal = int(((py.state.c_do_n + py.state.c_ctrl_n) > 0).sum())
        assert rs_read.represented_causal == represented_causal

    snap = json.loads(rs.snapshot_json())

    for entry in snap["relations"]:
        a, b = int(entry["a"]), int(entry["b"])
        pid = py.state.pidx.pair_id(a, b)
        ev = entry["evidence"]
        assert ev["a"] == pytest.approx(float(py.state.rel_a[pid]), abs=1e-12)
        assert ev["b"] == pytest.approx(float(py.state.rel_b[pid]), abs=1e-12)
        assert ev["exposure"] == int(py.state.rel_exposure[pid])
        assert ev["state"] == int(py.state.relation_state[pid])

    for entry in snap["causal"]:
        source, target = int(entry["source"]), int(entry["target"])
        oid = py.state.pidx.ordered_id(source, target)
        ev = entry["evidence"]
        assert ev["do_a"] == pytest.approx(float(py.state.c_do_a[oid]), abs=1e-12)
        assert ev["do_b"] == pytest.approx(float(py.state.c_do_b[oid]), abs=1e-12)
        assert ev["ctrl_a"] == pytest.approx(float(py.state.c_ctrl_a[oid]), abs=1e-12)
        assert ev["ctrl_b"] == pytest.approx(float(py.state.c_ctrl_b[oid]), abs=1e-12)
        assert ev["do_n"] == int(py.state.c_do_n[oid])
        assert ev["ctrl_n"] == int(py.state.c_ctrl_n[oid])
        assert ev["state"] == int(py.state.causal_state[oid])


def test_sparse_graph_does_not_allocate_unexposed_pairs():
    rs = GraphEvidence()
    rs.step([0, 1], [(0, 1, 1)], None, [])
    assert rs.represented_relations == 1
    assert rs.represented_causal == 0

    # Merely having large entity IDs does not allocate the Cartesian carrier.
    rs.step([1000, 2000], [], None, [])
    assert rs.represented_relations == 2
    assert rs.represented_causal == 0
