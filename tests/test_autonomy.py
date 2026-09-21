from __future__ import annotations

import json

import pytest

from cortex.autonomy import ObservationFrame, train_dataset


class FakeRead:
    def __init__(self, unresolved=False):
        self.unresolved = unresolved


class FakeRuntime:
    def __init__(self):
        self.calls = []

    def step(
        self,
        detections,
        relation_obs,
        intervention_src_det,
        outcomes,
        outcome,
    ):
        self.calls.append(
            {
                "detections": detections,
                "relation_obs": relation_obs,
                "intervention_src_det": intervention_src_det,
                "outcomes": outcomes,
                "outcome": outcome,
            }
        )
        return FakeRead(unresolved=len(self.calls) == 2)

    def snapshot_json(self):
        return json.dumps({"frames": len(self.calls)}, sort_keys=True)


def test_dataset_training_preserves_chronological_order_and_snapshots():
    runtime = FakeRuntime()
    frames = (
        ObservationFrame(
            detections=((1.0, 0.0), (0.0, 1.0)),
            relation_observations=((0, 1, 1),),
            target_outcomes=((1, 1),),
            provenance="dataset:train:0",
        ),
        ObservationFrame(
            detections=((2.0, 0.0),),
            outcome=0.75,
            provenance="dataset:train:1",
        ),
    )

    report = train_dataset(runtime, frames)

    assert report.frames == 2
    assert report.detections == 3
    assert report.relation_observations == 1
    assert report.unresolved_reads == 1
    assert json.loads(report.snapshot_json) == {"frames": 2}
    assert runtime.calls[0]["relation_obs"] == [(0, 1, 1)]
    assert runtime.calls[0]["outcomes"] == [(1, 1)]
    assert runtime.calls[1]["outcomes"] == []
    assert runtime.calls[1]["outcome"] == 0.75


def test_frame_rejects_out_of_range_relations():
    with pytest.raises(ValueError, match="outside the frame"):
        ObservationFrame(
            detections=((1.0,),),
            relation_observations=((0, 1, 1),),
        )


def test_frame_rejects_mixed_dimensions():
    with pytest.raises(ValueError, match="share one dimension"):
        ObservationFrame(
            detections=((1.0,), (1.0, 2.0)),
        )
