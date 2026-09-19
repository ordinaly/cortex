"""Cortex v1.15-R cache-refresh attribution campaign."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter_ns

import numpy as np

from fused_stage_attribution_campaign import (
    ANCHOR_CASES,
    RESOLUTIONS,
    make_trace,
)
from fuzzy_predictive_field import FuzzyPredictiveField
from hysteretic_local_resolution_campaign import make_anchors
from support_sparse_prediction import (
    FusedSupportSparseLocalResolutionController,
    SupportSparsePredictiveCache,
)

OUTCOMES = 3
SUBSTAGES = (
    "features",
    "probabilities",
    "dirty_detection",
    "feature_write",
    "distance_compute",
    "distance_write",
    "kernel_compute",
    "kernel_write",
    "bookkeeping",
)


class ProfiledSupportSparsePredictiveCache(
    SupportSparsePredictiveCache
):
    """Exact SupportSparsePredictiveCache with refresh substage timing."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.refresh_total_ns = 0
        self.substage_ns = {
            name: 0
            for name in SUBSTAGES
        }
        self.profiled_events = 0
        self.dirty_events = 0
        self.dirty_rows = 0
        self.dirty_event_substage_ns = {
            name: 0
            for name in SUBSTAGES
        }
        self.dirty_refresh_total_ns = 0

    @staticmethod
    def _timed(fn):
        start = perf_counter_ns()
        value = fn()
        return value, perf_counter_ns() - start

    def _record(
        self,
        name: str,
        elapsed: int,
        *,
        dirty_event: bool,
    ) -> None:
        self.substage_ns[name] += elapsed
        if dirty_event:
            self.dirty_event_substage_ns[name] += elapsed

    def refresh(self) -> np.ndarray:
        refresh_start = perf_counter_ns()
        event = {
            name: 0
            for name in SUBSTAGES
        }

        current_features, elapsed = self._timed(
            self.field.features
        )
        event["features"] += elapsed

        probabilities, elapsed = self._timed(
            self.field.outcome_probabilities
        )
        event["probabilities"] += elapsed
        self.probabilities = probabilities

        anchors = len(self.field.anchors)

        start = perf_counter_ns()
        self.refresh_events += 1
        event["bookkeeping"] += perf_counter_ns() - start

        if self.feature_tolerance == 0.0:
            start = perf_counter_ns()
            self.features = current_features
            event["feature_write"] += perf_counter_ns() - start

            distances, elapsed = self._timed(
                lambda: np.linalg.norm(
                    self.features[:, None, :]
                    - self.features[None, :, :],
                    axis=2,
                )
                / np.sqrt(2.0)
            )
            event["distance_compute"] += elapsed

            start = perf_counter_ns()
            self.distances = distances
            event["distance_write"] += perf_counter_ns() - start

            kernels, elapsed = self._timed(
                lambda: self._kernels_from_distances(
                    self.distances
                )
            )
            event["kernel_compute"] += elapsed

            start = perf_counter_ns()
            self.kernels = kernels
            event["kernel_write"] += perf_counter_ns() - start

            start = perf_counter_ns()
            self.last_dirty_mask = np.ones(
                anchors,
                dtype=bool,
            )
            self.kernel_revision += 1
            self.full_rebuilds += 1
            self.rows_refreshed += anchors
            event["bookkeeping"] += perf_counter_ns() - start
            dirty = self.last_dirty_mask.copy()
        else:
            def detect():
                drift = np.linalg.norm(
                    current_features - self.features,
                    axis=1,
                )
                return drift >= self.feature_tolerance

            dirty, elapsed = self._timed(detect)
            event["dirty_detection"] += elapsed

            start = perf_counter_ns()
            self.last_dirty_mask = dirty
            event["bookkeeping"] += perf_counter_ns() - start

            if np.any(dirty):
                start = perf_counter_ns()
                self.features[dirty] = current_features[dirty]
                event["feature_write"] += perf_counter_ns() - start

                def distance_rows():
                    diff = (
                        self.features[dirty, None, :]
                        - self.features[None, :, :]
                    )
                    return (
                        np.linalg.norm(diff, axis=2)
                        / np.sqrt(2.0)
                    )

                rows, elapsed = self._timed(distance_rows)
                event["distance_compute"] += elapsed

                start = perf_counter_ns()
                self.distances[dirty, :] = rows
                self.distances[:, dirty] = rows.T
                event["distance_write"] += perf_counter_ns() - start

                kernel_rows = []
                start = perf_counter_ns()
                for resolution in self.resolutions:
                    kernel_rows.append(
                        np.exp(
                            -0.5
                            * (rows / resolution) ** 2
                        )
                    )
                event["kernel_compute"] += perf_counter_ns() - start

                start = perf_counter_ns()
                for index, values in enumerate(kernel_rows):
                    kernel = self.kernels[index]
                    kernel[dirty, :] = values
                    kernel[:, dirty] = values.T
                event["kernel_write"] += perf_counter_ns() - start

                start = perf_counter_ns()
                self.kernel_revision += 1
                self.partial_refreshes += 1
                self.rows_refreshed += int(
                    np.sum(dirty)
                )
                event["bookkeeping"] += perf_counter_ns() - start

        total = perf_counter_ns() - refresh_start
        is_dirty = bool(np.any(dirty))
        self.refresh_total_ns += total
        self.profiled_events += 1

        if is_dirty:
            self.dirty_events += 1
            self.dirty_rows += int(np.sum(dirty))
            self.dirty_refresh_total_ns += total

        for name, elapsed in event.items():
            self._record(
                name,
                elapsed,
                dirty_event=is_dirty,
            )

        return dirty.copy()

    def profile(self) -> dict:
        total = max(1, self.refresh_total_ns)
        explicit = sum(self.substage_ns.values())
        dirty_total = max(1, self.dirty_refresh_total_ns)
        anchors = len(self.field.anchors)
        return {
            "refresh_events": self.profiled_events,
            "dirty_events": self.dirty_events,
            "dirty_event_fraction": (
                self.dirty_events
                / max(1, self.profiled_events)
            ),
            "mean_dirty_rows_per_dirty_event": (
                self.dirty_rows
                / max(1, self.dirty_events)
            ),
            "mean_dirty_row_fraction": (
                self.dirty_rows
                / max(1, self.dirty_events * anchors)
            ),
            "refresh_total_ns": self.refresh_total_ns,
            "accounted_fraction": explicit / total,
            **{
                f"{name}_ns": self.substage_ns[name]
                for name in SUBSTAGES
            },
            **{
                f"{name}_share": (
                    self.substage_ns[name] / total
                )
                for name in SUBSTAGES
            },
            **{
                f"dirty_{name}_share": (
                    self.dirty_event_substage_ns[name]
                    / dirty_total
                )
                for name in SUBSTAGES
            },
        }


def _field(anchors: np.ndarray) -> FuzzyPredictiveField:
    return FuzzyPredictiveField(
        anchors,
        outcomes=OUTCOMES,
        relations=3,
        perceptual_temperature=0.03,
        semantic_resolution=0.04,
        evidence_decay=0.998,
    )


def _controller(
    anchors: np.ndarray,
    *,
    profiled: bool,
) -> FusedSupportSparseLocalResolutionController:
    field = _field(anchors.copy())
    controller = FusedSupportSparseLocalResolutionController(
        field,
        resolutions=RESOLUTIONS,
        feature_tolerance=0.0025,
        loss_change_tolerance=0.001,
        full_scan_interval=64,
        complexity_refresh_interval=32,
        retained_mass=1.0,
    )
    if profiled:
        controller.cache = ProfiledSupportSparsePredictiveCache(
            field,
            resolutions=RESOLUTIONS,
            feature_tolerance=0.0025,
        )
    return controller


def run_case(seed: int, anchors_count: int) -> dict:
    anchors = make_anchors(
        seed + 1709 + anchors_count,
        anchors=anchors_count,
        individual_scale=0.35,
    )
    trace = make_trace(
        seed + 5003,
        anchors,
    )

    reference = _controller(
        anchors,
        profiled=False,
    )
    profiled = _controller(
        anchors,
        profiled=True,
    )
    assert isinstance(
        profiled.cache,
        ProfiledSupportSparsePredictiveCache,
    )

    max_candidate_error = 0.0
    max_local_error = 0.0
    max_outcome_count_error = 0.0
    max_feature_error = 0.0
    max_probability_error = 0.0
    max_distance_error = 0.0
    max_kernel_error = 0.0
    structural_disagreements = 0

    for observation, outcome, _phase in trace:
        ref_bundle = reference.prepare(observation)
        prof_bundle = profiled.prepare(observation)

        max_candidate_error = max(
            max_candidate_error,
            float(
                np.max(
                    np.abs(
                        ref_bundle.candidate_predictions
                        - prof_bundle.candidate_predictions
                    )
                )
            ),
        )
        max_local_error = max(
            max_local_error,
            float(
                np.max(
                    np.abs(
                        ref_bundle.local_prediction
                        - prof_bundle.local_prediction
                    )
                )
            ),
        )

        reference.commit(ref_bundle, outcome)
        profiled.commit(prof_bundle, outcome)

        if not np.array_equal(
            reference.current_indices,
            profiled.current_indices,
        ):
            structural_disagreements += 1

        max_outcome_count_error = max(
            max_outcome_count_error,
            float(
                np.max(
                    np.abs(
                        reference.field.outcome_counts
                        - profiled.field.outcome_counts
                    )
                )
            ),
        )
        max_feature_error = max(
            max_feature_error,
            float(
                np.max(
                    np.abs(
                        reference.cache.features
                        - profiled.cache.features
                    )
                )
            ),
        )
        max_probability_error = max(
            max_probability_error,
            float(
                np.max(
                    np.abs(
                        reference.cache.probabilities
                        - profiled.cache.probabilities
                    )
                )
            ),
        )
        max_distance_error = max(
            max_distance_error,
            float(
                np.max(
                    np.abs(
                        reference.cache.distances
                        - profiled.cache.distances
                    )
                )
            ),
        )
        max_kernel_error = max(
            max_kernel_error,
            float(
                np.max(
                    np.abs(
                        reference.cache.kernels
                        - profiled.cache.kernels
                    )
                )
            ),
        )

    profile = profiled.cache.profile()
    steps = len(trace)

    return {
        "protocol": "cache-refresh-attribution-v1",
        "seed": seed,
        "anchors": anchors_count,
        "steps": steps,
        "max_candidate_error": max_candidate_error,
        "max_local_error": max_local_error,
        "max_outcome_count_error": max_outcome_count_error,
        "max_feature_error": max_feature_error,
        "max_probability_error": max_probability_error,
        "max_distance_error": max_distance_error,
        "max_kernel_error": max_kernel_error,
        "structural_disagreements": structural_disagreements,
        "refresh_us_per_step": (
            profile["refresh_total_ns"]
            / steps
            / 1.0e3
        ),
        **{
            f"{name}_us_per_step": (
                profile[f"{name}_ns"]
                / steps
                / 1.0e3
            )
            for name in SUBSTAGES
        },
        **{
            f"{name}_share": profile[
                f"{name}_share"
            ]
            for name in SUBSTAGES
        },
        **{
            f"dirty_{name}_share": profile[
                f"dirty_{name}_share"
            ]
            for name in SUBSTAGES
        },
        "accounted_fraction": profile[
            "accounted_fraction"
        ],
        "dirty_event_fraction": profile[
            "dirty_event_fraction"
        ],
        "mean_dirty_rows_per_dirty_event": profile[
            "mean_dirty_rows_per_dirty_event"
        ],
        "mean_dirty_row_fraction": profile[
            "mean_dirty_row_fraction"
        ],
    }


def _median(rows, key):
    return float(
        np.median(
            [row[key] for row in rows]
        )
    )


def campaign(seeds: int, output: Path) -> dict:
    rows = [
        run_case(seed, anchors)
        for anchors in ANCHOR_CASES
        for seed in range(seeds)
    ]
    output.write_text(
        "".join(
            json.dumps(row, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )

    cases = {}
    for anchors in ANCHOR_CASES:
        selected = [
            row
            for row in rows
            if row["anchors"] == anchors
        ]
        shares = {
            name: _median(
                selected,
                f"{name}_share",
            )
            for name in SUBSTAGES
        }
        dominant = max(
            SUBSTAGES,
            key=shares.get,
        )
        cases[str(anchors)] = {
            "refresh_us_per_step": _median(
                selected,
                "refresh_us_per_step",
            ),
            "accounted_fraction": _median(
                selected,
                "accounted_fraction",
            ),
            "dirty_event_fraction": _median(
                selected,
                "dirty_event_fraction",
            ),
            "mean_dirty_row_fraction": _median(
                selected,
                "mean_dirty_row_fraction",
            ),
            "dominant_substage": dominant,
            "substages": {
                name: {
                    "share": shares[name],
                    "us_per_step": _median(
                        selected,
                        f"{name}_us_per_step",
                    ),
                    "dirty_share": _median(
                        selected,
                        f"dirty_{name}_share",
                    ),
                }
                for name in SUBSTAGES
            },
        }

    summary = {
        "protocol": "cache-refresh-attribution-v1",
        "seeds": seeds,
        "anchor_cases": list(ANCHOR_CASES),
        "cases": cases,
        "max_candidate_error": max(
            row["max_candidate_error"]
            for row in rows
        ),
        "max_local_error": max(
            row["max_local_error"]
            for row in rows
        ),
        "max_outcome_count_error": max(
            row["max_outcome_count_error"]
            for row in rows
        ),
        "max_feature_error": max(
            row["max_feature_error"]
            for row in rows
        ),
        "max_probability_error": max(
            row["max_probability_error"]
            for row in rows
        ),
        "max_distance_error": max(
            row["max_distance_error"]
            for row in rows
        ),
        "max_kernel_error": max(
            row["max_kernel_error"]
            for row in rows
        ),
        "structural_disagreements": sum(
            row["structural_disagreements"]
            for row in rows
        ),
    }
    print(json.dumps(summary, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "cache-refresh-attribution-v1.jsonl"
        ),
    )
    args = parser.parse_args()
    campaign(args.seeds, args.output)


if __name__ == "__main__":
    main()
