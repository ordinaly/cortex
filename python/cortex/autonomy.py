"""Structured dataset ingestion for autonomous Cortex learning.

This module intentionally contains no neural model dependency. It presents
chronological structured observations directly to a Cortex-compatible runtime
and preserves stream order.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, Protocol, Sequence


class CortexStepRuntime(Protocol):
    def step(
        self,
        detections: Sequence[Sequence[float]],
        relation_obs: Sequence[tuple[int, int, int]],
        intervention_src_det: int | None,
        outcomes: Sequence[tuple[int, int]],
        outcome: float | None,
    ) -> Any: ...

    def snapshot_json(self) -> str: ...


@dataclass(frozen=True, slots=True)
class ObservationFrame:
    detections: tuple[tuple[float, ...], ...]
    relation_observations: tuple[tuple[int, int, int], ...] = ()
    intervention_source: int | None = None
    target_outcomes: tuple[tuple[int, int], ...] = ()
    outcome: float | None = None
    provenance: str | None = None

    def __post_init__(self) -> None:
        if not self.detections:
            raise ValueError("an observation frame requires at least one detection")
        dim = len(self.detections[0])
        if dim < 1:
            raise ValueError("detection vectors must be non-empty")
        for vector in self.detections:
            if len(vector) != dim:
                raise ValueError("all detections in a frame must share one dimension")
            if any(not math.isfinite(float(value)) for value in vector):
                raise ValueError("detection vectors must contain only finite values")
        count = len(self.detections)
        for source, target, present in self.relation_observations:
            if not 0 <= int(source) < count or not 0 <= int(target) < count:
                raise ValueError("relation observation index is outside the frame")
            if int(present) not in (0, 1):
                raise ValueError("relation observation state must be 0 or 1")
        if self.intervention_source is not None and not (
            0 <= int(self.intervention_source) < count
        ):
            raise ValueError("intervention source index is outside the frame")
        for target, value in self.target_outcomes:
            if not 0 <= int(target) < count:
                raise ValueError("target outcome index is outside the frame")
            if int(value) not in (0, 1):
                raise ValueError("target outcome state must be 0 or 1")
        if self.outcome is not None and not math.isfinite(float(self.outcome)):
            raise ValueError("outcome must be finite")


@dataclass(frozen=True, slots=True)
class TrainingReport:
    frames: int
    detections: int
    relation_observations: int
    unresolved_reads: int
    snapshot_json: str


def train_dataset(
    runtime: CortexStepRuntime,
    frames: Iterable[ObservationFrame],
) -> TrainingReport:
    """Present a chronological dataset directly to Cortex memory."""

    frame_count = 0
    detection_count = 0
    relation_count = 0
    unresolved = 0

    for frame in frames:
        read = runtime.step(
            [list(vector) for vector in frame.detections],
            list(frame.relation_observations),
            frame.intervention_source,
            list(frame.target_outcomes),
            frame.outcome,
        )
        frame_count += 1
        detection_count += len(frame.detections)
        relation_count += len(frame.relation_observations)
        unresolved += int(bool(getattr(read, "unresolved", False)))

    return TrainingReport(
        frames=frame_count,
        detections=detection_count,
        relation_observations=relation_count,
        unresolved_reads=unresolved,
        snapshot_json=runtime.snapshot_json(),
    )
