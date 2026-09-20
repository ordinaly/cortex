"""Cortex v1.11-R fused support-sparse predictive application.

v1.10-R showed that sparse kernel maintenance begins to pay off around 72
anchors, while sampled SVD is not the dominant cost at the tested scales.

This module targets the remaining dense predictive application:
    w^T K_r
for every candidate resolution and the corresponding local operator readout.

Two changes are separated deliberately:

1. exact fusion: compute perceptual responsibilities once per observation and
   reuse them for candidate prediction, local prediction, regional-loss
   attribution and field update;
2. support sparsification: retain the smallest perceptual support containing a
   declared probability mass, renormalize it, and apply only those kernel rows.

World-model evidence updates always use the full perceptual responsibility
vector. Only predictive readout is support-compressed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from sparse_cached_resolution import (
    PredictiveKernelCache,
    SparseCachedLocalResolutionController,
    SparseRegionalPredictiveLoss,
)


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    totals = values.sum(axis=1, keepdims=True)
    fallback = totals[:, 0] <= 1.0e-15
    normalized = values / np.maximum(totals, 1.0e-15)
    if np.any(fallback):
        normalized[fallback] = 1.0 / values.shape[1]
    return normalized


@dataclass(frozen=True)
class PerceptualSupport:
    indices: np.ndarray
    weights: np.ndarray
    retained_mass: float
    dropped_mass: float
    fraction: float


@dataclass(frozen=True)
class PrequentialBundle:
    candidate_predictions: np.ndarray
    local_prediction: np.ndarray
    decision: object
    full_weights: np.ndarray
    support: PerceptualSupport


def retained_mass_support(
    weights: Sequence[float],
    *,
    target_mass: float,
) -> PerceptualSupport:
    """Return the smallest descending-weight support reaching target_mass.

    The returned support weights are renormalized to one. If M is retained
    mass, the total-variation distance between the original categorical
    responsibility vector and the renormalized support vector is exactly 1-M.
    """
    values = np.asarray(weights, dtype=float)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("weights must be a non-empty vector")
    if np.any(values < 0) or not np.all(np.isfinite(values)):
        raise ValueError("weights must be finite and non-negative")
    total = float(values.sum())
    if total <= 1.0e-15:
        values = np.full(len(values), 1.0 / len(values))
    else:
        values = values / total
    if not (0 < target_mass <= 1):
        raise ValueError("target_mass must be in (0, 1]")

    if target_mass >= 1.0 - 1.0e-15:
        indices = np.arange(len(values), dtype=int)
        return PerceptualSupport(
            indices=indices,
            weights=values.copy(),
            retained_mass=1.0,
            dropped_mass=0.0,
            fraction=1.0,
        )

    order = np.argsort(-values, kind="stable")
    cumulative = np.cumsum(values[order])
    count = int(
        np.searchsorted(
            cumulative,
            target_mass,
            side="left",
        )
        + 1
    )
    indices = order[:count]
    retained = float(values[indices].sum())
    selected = values[indices] / max(retained, 1.0e-15)
    dropped = max(0.0, 1.0 - retained)
    return PerceptualSupport(
        indices=indices,
        weights=selected,
        retained_mass=retained,
        dropped_mass=dropped,
        fraction=float(count / len(values)),
    )


class SupportSparsePredictiveCache(PredictiveKernelCache):
    """Predictive cache with fused dense or retained-mass support readout."""

    def prediction_bundle_from_weights(
        self,
        full_weights: Sequence[float],
        resolution_indices: Sequence[int],
        *,
        retained_mass: float,
    ) -> tuple[np.ndarray, np.ndarray, PerceptualSupport]:
        support = retained_mass_support(
            full_weights,
            target_mass=retained_mass,
        )
        source = support.indices
        weights = support.weights

        if len(source) == len(self.field.anchors):
            # Exact full-support path: avoid fancy-index copying of the whole
            # candidate bank so timing reflects fusion rather than a cache copy.
            activations = np.einsum(
                "a,rab->rb",
                weights,
                self.kernels,
                optimize=True,
            )
        else:
            candidate_rows = self.kernels[:, source, :]
            activations = np.einsum(
                "s,rsb->rb",
                weights,
                candidate_rows,
                optimize=True,
            )
        semantic_fields = _normalize_rows(activations)
        candidate_predictions = (
            semantic_fields @ self.probabilities
        )
        candidate_predictions = _normalize_rows(
            candidate_predictions
        )

        resolution_indexes = np.asarray(
            resolution_indices,
            dtype=int,
        )
        if resolution_indexes.shape != (
            len(self.field.anchors),
        ):
            raise ValueError(
                "one resolution index is required per anchor"
            )
        local_rows = self.kernels[
            resolution_indexes[source],
            source,
            :,
        ]
        local_activation = weights @ local_rows
        local_total = float(local_activation.sum())
        if local_total <= 1.0e-15:
            local_field = np.full(
                len(self.field.anchors),
                1.0 / len(self.field.anchors),
            )
        else:
            local_field = (
                local_activation / local_total
            )
        local_prediction = (
            local_field @ self.probabilities
        )
        prediction_total = float(
            local_prediction.sum()
        )
        if prediction_total <= 1.0e-15:
            local_prediction = np.full(
                self.field.outcomes,
                1.0 / self.field.outcomes,
            )
        else:
            local_prediction = (
                local_prediction / prediction_total
            )

        return (
            candidate_predictions,
            local_prediction,
            support,
        )


class ExactSemanticGeometryPredictiveCache(
    SupportSparsePredictiveCache
):
    """Sparse cache using exact streamed semantic geometry.

    Refresh accumulates squared feature differences directly into a 2-D
    dirty-row distance matrix rather than materializing a
    dirty-rows × anchors × outcomes difference tensor. The same squared
    distance is then reused to regenerate every resolution kernel row.

    This preserves the legacy Euclidean distance arithmetic more closely than
    the mathematically equivalent unit-norm dot-product identity near zero.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        resolutions = np.asarray(
            self.resolutions,
            dtype=float,
        )
        self._inverse_two_resolution_sq = (
            0.5 / np.square(resolutions)
        )

    def _geometry_rows(
        self,
        features: np.ndarray,
        *,
        source_indices: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        # Accumulate exact squared differences into a 2-D dirty-row matrix
        # instead of materializing dirty_rows × anchors × outcomes. This is
        # algebraically identical to the legacy Euclidean path while avoiding
        # the near-zero cancellation of sqrt(1 - dot).
        distance_squared = np.zeros(
            (len(features), len(self.features)),
            dtype=float,
        )
        for dimension in range(features.shape[1]):
            delta = (
                features[:, dimension, None]
                - self.features[None, :, dimension]
            )
            distance_squared += delta * delta
        distance_squared *= 0.5
        if source_indices is not None:
            distance_squared[
                np.arange(len(source_indices)),
                source_indices,
            ] = 0.0
        distances = np.sqrt(distance_squared)
        return distance_squared, distances

    def _kernel_rows_from_squared(
        self,
        distance_squared: np.ndarray,
    ) -> np.ndarray:
        return np.exp(
            -self._inverse_two_resolution_sq[
                :, None, None
            ]
            * distance_squared[None, :, :]
        )

    def refresh(self) -> np.ndarray:
        current_features = self.field.features()
        self.probabilities = (
            self.field.outcome_probabilities()
        )
        anchors = len(self.field.anchors)
        self.refresh_events += 1

        if self.feature_tolerance == 0.0:
            self.features = current_features
            distance_squared, distances = (
                self._geometry_rows(
                    self.features,
                    source_indices=np.arange(
                        anchors,
                        dtype=int,
                    ),
                )
            )
            self.distances = distances
            self.kernels = (
                self._kernel_rows_from_squared(
                    distance_squared
                )
            )
            self.last_dirty_mask = np.ones(
                anchors,
                dtype=bool,
            )
            self.kernel_revision += 1
            self.full_rebuilds += 1
            self.rows_refreshed += anchors
            return self.last_dirty_mask.copy()

        drift = np.linalg.norm(
            current_features - self.features,
            axis=1,
        )
        dirty = drift >= self.feature_tolerance
        self.last_dirty_mask = dirty

        if not np.any(dirty):
            return dirty.copy()

        self.features[dirty] = current_features[dirty]
        dirty_indices = np.flatnonzero(dirty)
        distance_squared, rows = self._geometry_rows(
            self.features[dirty],
            source_indices=dirty_indices,
        )
        self.distances[dirty, :] = rows
        self.distances[:, dirty] = rows.T

        kernel_rows = self._kernel_rows_from_squared(
            distance_squared
        )
        for index in range(len(self.resolutions)):
            kernel = self.kernels[index]
            values = kernel_rows[index]
            kernel[dirty, :] = values
            kernel[:, dirty] = values.T

        self.kernel_revision += 1
        self.partial_refreshes += 1
        self.rows_refreshed += int(np.sum(dirty))
        return dirty.copy()


class LazyDistanceSemanticGeometryPredictiveCache(
    ExactSemanticGeometryPredictiveCache
):
    """Exact cache with lazy materialization of the public distance matrix.

    The hot predictive path consumes kernels, not the distance matrix. Refresh
    therefore computes only squared dirty-row geometry, regenerates kernels,
    and marks the public distance view stale. Accessing distances rebuilds
    the exact symmetric matrix from the current cached features.

    This removes hot-path square-root work and symmetric distance writes while
    preserving the existing public distance view on demand.
    """

    def __init__(self, *args, **kwargs):
        self._distance_cache = None
        self._distance_stale = False
        self.distance_materializations = 0
        super().__init__(*args, **kwargs)

    @property
    def distances(self) -> np.ndarray:
        if self._distance_stale:
            anchors = len(self.features)
            _, values = self._geometry_rows(
                self.features,
                source_indices=np.arange(
                    anchors,
                    dtype=int,
                ),
            )
            self._distance_cache = values
            self._distance_stale = False
            self.distance_materializations += 1
        if self._distance_cache is None:
            raise RuntimeError(
                "distance cache was not initialized"
            )
        return self._distance_cache

    @distances.setter
    def distances(self, value: np.ndarray) -> None:
        self._distance_cache = np.asarray(
            value,
            dtype=float,
        )
        self._distance_stale = False

    @property
    def distance_view_stale(self) -> bool:
        return bool(self._distance_stale)

    def _distance_squared_rows(
        self,
        features: np.ndarray,
        *,
        source_indices: np.ndarray | None = None,
    ) -> np.ndarray:
        distance_squared = np.zeros(
            (len(features), len(self.features)),
            dtype=float,
        )
        for dimension in range(features.shape[1]):
            delta = (
                features[:, dimension, None]
                - self.features[None, :, dimension]
            )
            distance_squared += delta * delta
        distance_squared *= 0.5
        if source_indices is not None:
            distance_squared[
                np.arange(len(source_indices)),
                source_indices,
            ] = 0.0
        return distance_squared

    def refresh(self) -> np.ndarray:
        current_features = self.field.features()
        self.probabilities = (
            self.field.outcome_probabilities()
        )
        anchors = len(self.field.anchors)
        self.refresh_events += 1

        if self.feature_tolerance == 0.0:
            self.features = current_features
            distance_squared = self._distance_squared_rows(
                self.features,
                source_indices=np.arange(
                    anchors,
                    dtype=int,
                ),
            )
            self.kernels = (
                self._kernel_rows_from_squared(
                    distance_squared
                )
            )
            self._distance_stale = True
            self.last_dirty_mask = np.ones(
                anchors,
                dtype=bool,
            )
            self.kernel_revision += 1
            self.full_rebuilds += 1
            self.rows_refreshed += anchors
            return self.last_dirty_mask.copy()

        drift = np.linalg.norm(
            current_features - self.features,
            axis=1,
        )
        dirty = drift >= self.feature_tolerance
        self.last_dirty_mask = dirty

        if not np.any(dirty):
            return dirty.copy()

        self.features[dirty] = current_features[dirty]
        dirty_indices = np.flatnonzero(dirty)
        distance_squared = self._distance_squared_rows(
            self.features[dirty],
            source_indices=dirty_indices,
        )

        kernel_rows = self._kernel_rows_from_squared(
            distance_squared
        )
        for index in range(len(self.resolutions)):
            kernel = self.kernels[index]
            values = kernel_rows[index]
            kernel[dirty, :] = values
            kernel[:, dirty] = values.T

        self._distance_stale = True
        self.kernel_revision += 1
        self.partial_refreshes += 1
        self.rows_refreshed += int(np.sum(dirty))
        return dirty.copy()


class SingleBufferKernelRefreshPredictiveCache(
    LazyDistanceSemanticGeometryPredictiveCache
):
    """v1.22-R candidate using one reusable 2-D kernel scratch buffer.

    The promoted v1.20 lazy cache computes all candidate-resolution rows in
    one temporary array shaped roughly resolutions x dirty_rows x anchors.
    This candidate instead evaluates one resolution at a time into a reusable
    two-dimensional scratch buffer, then copies that slice into the persistent
    kernel bank.

    The scratch bank starts at 25% of the anchor count (comfortably above the
    ~13% dirty-row fraction measured in v1.17) and grows geometrically only if
    a later refresh requires more rows. Numerical kernel semantics, symmetric
    writes, and lazy public-distance materialization are unchanged.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        anchors = len(self.features)
        initial_rows = min(
            anchors,
            max(1, (anchors + 3) // 4),
        )
        self._kernel_scratch = np.empty(
            (initial_rows, anchors),
            dtype=float,
        )
        self.scratch_reallocations = 0
        self.single_buffer_refreshes = 0

    @property
    def scratch_capacity_rows(self) -> int:
        return int(self._kernel_scratch.shape[0])

    @property
    def scratch_bytes(self) -> int:
        return int(self._kernel_scratch.nbytes)

    def _scratch_rows(self, rows: int) -> np.ndarray:
        if rows <= 0:
            raise ValueError("rows must be positive")
        anchors = len(self.features)
        if rows > self._kernel_scratch.shape[0]:
            capacity = min(
                anchors,
                max(
                    rows,
                    2 * self._kernel_scratch.shape[0],
                ),
            )
            self._kernel_scratch = np.empty(
                (capacity, anchors),
                dtype=float,
            )
            self.scratch_reallocations += 1
        return self._kernel_scratch[:rows, :]

    def _write_kernel_rows_single_buffer(
        self,
        distance_squared: np.ndarray,
        dirty: np.ndarray,
        *,
        full_rebuild: bool,
    ) -> None:
        rows = int(distance_squared.shape[0])
        scratch = self._scratch_rows(rows)

        for index, inverse_scale in enumerate(
            self._inverse_two_resolution_sq
        ):
            np.multiply(
                distance_squared,
                -float(inverse_scale),
                out=scratch,
            )
            np.exp(scratch, out=scratch)

            kernel = self.kernels[index]
            if full_rebuild:
                kernel[...] = scratch
            else:
                kernel[dirty, :] = scratch
                kernel[:, dirty] = scratch.T

        self.single_buffer_refreshes += 1

    def refresh(self) -> np.ndarray:
        current_features = self.field.features()
        self.probabilities = (
            self.field.outcome_probabilities()
        )
        anchors = len(self.field.anchors)
        self.refresh_events += 1

        if self.feature_tolerance == 0.0:
            self.features = current_features
            distance_squared = self._distance_squared_rows(
                self.features,
                source_indices=np.arange(
                    anchors,
                    dtype=int,
                ),
            )
            dirty = np.ones(
                anchors,
                dtype=bool,
            )
            self._write_kernel_rows_single_buffer(
                distance_squared,
                dirty,
                full_rebuild=True,
            )
            self._distance_stale = True
            self.last_dirty_mask = dirty
            self.kernel_revision += 1
            self.full_rebuilds += 1
            self.rows_refreshed += anchors
            return dirty.copy()

        drift = np.linalg.norm(
            current_features - self.features,
            axis=1,
        )
        dirty = drift >= self.feature_tolerance
        self.last_dirty_mask = dirty

        if not np.any(dirty):
            return dirty.copy()

        self.features[dirty] = current_features[dirty]
        dirty_indices = np.flatnonzero(dirty)
        distance_squared = self._distance_squared_rows(
            self.features[dirty],
            source_indices=dirty_indices,
        )
        self._write_kernel_rows_single_buffer(
            distance_squared,
            dirty,
            full_rebuild=False,
        )

        self._distance_stale = True
        self.kernel_revision += 1
        self.partial_refreshes += 1
        self.rows_refreshed += int(np.sum(dirty))
        return dirty.copy()


class KernelLadderSemanticGeometryPredictiveCache(
    ExactSemanticGeometryPredictiveCache
):
    """Exact-kernel candidate that reuses one exponential across a scale ladder.

    For the canonical resolutions 0.03, 0.06, 0.12, 0.25, the first three
    inverse squared scales have ratios 16:4:1. The 0.12 kernel is therefore
    exponentiated once, while 0.06 and 0.03 are derived by repeated squaring.
    The 0.25 kernel remains independently exponentiated.

    Non-canonical resolution sets and anchor banks below 192 anchors fall back
    to the parent implementation. The scale gate is a development-stage result:
    repeated-squaring overhead dominated at 72 and 144 anchors, while the
    ladder became beneficial at 288 anchors.
    """

    _canonical_resolutions = np.asarray(
        (0.03, 0.06, 0.12, 0.25),
        dtype=float,
    )
    _minimum_ladder_anchors = 192

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._kernel_ladder_enabled = (
            len(self.features) >= self._minimum_ladder_anchors
            and self._uses_canonical_ladder()
        )

    def _uses_canonical_ladder(self) -> bool:
        values = np.asarray(
            self.resolutions,
            dtype=float,
        )
        return (
            values.shape == self._canonical_resolutions.shape
            and np.allclose(
                values,
                self._canonical_resolutions,
                atol=0.0,
                rtol=1.0e-12,
            )
        )

    def _kernel_rows_from_squared(
        self,
        distance_squared: np.ndarray,
    ) -> np.ndarray:
        if not self._kernel_ladder_enabled:
            return super()._kernel_rows_from_squared(
                distance_squared
            )

        shape = (
            len(self.resolutions),
            *distance_squared.shape,
        )
        kernels = np.empty(shape, dtype=float)

        # Base exponential for r=0.12.
        np.multiply(
            distance_squared,
            -self._inverse_two_resolution_sq[2],
            out=kernels[2],
        )
        np.exp(kernels[2], out=kernels[2])

        # K_0.06 = K_0.12^4.
        np.square(kernels[2], out=kernels[1])
        np.square(kernels[1], out=kernels[1])

        # K_0.03 = K_0.12^16.
        np.square(kernels[1], out=kernels[0])
        np.square(kernels[0], out=kernels[0])

        # r=0.25 has no exact integer-power relation to the ladder above.
        np.multiply(
            distance_squared,
            -self._inverse_two_resolution_sq[3],
            out=kernels[3],
        )
        np.exp(kernels[3], out=kernels[3])
        return kernels


class FusedSupportSparseLocalResolutionController(
    SparseCachedLocalResolutionController
):
    """v1.9 sparse controller with one perceptual pass per observation."""

    def __init__(
        self,
        field,
        *,
        resolutions: Sequence[float] = (
            0.03,
            0.06,
            0.12,
            0.25,
        ),
        epsilon: float = 0.04,
        hysteresis_margin: float = 0.01,
        coarsen_patience: int = 12,
        loss_decay: float = 0.995,
        warmup_observations: int = 60,
        minimum_relative_mass: float = 0.10,
        probability_floor: float = 1.0e-9,
        feature_tolerance: float = 0.0025,
        loss_change_tolerance: float = 0.001,
        full_scan_interval: int = 64,
        complexity_refresh_interval: int = 32,
        retained_mass: float = 1.0,
    ) -> None:
        super().__init__(
            field,
            resolutions=resolutions,
            epsilon=epsilon,
            hysteresis_margin=hysteresis_margin,
            coarsen_patience=coarsen_patience,
            loss_decay=loss_decay,
            warmup_observations=warmup_observations,
            minimum_relative_mass=minimum_relative_mass,
            probability_floor=probability_floor,
            feature_tolerance=feature_tolerance,
            loss_change_tolerance=loss_change_tolerance,
            full_scan_interval=full_scan_interval,
            complexity_refresh_interval=complexity_refresh_interval,
        )
        if not (0 < retained_mass <= 1):
            raise ValueError(
                "retained_mass must be in (0, 1]"
            )
        self.retained_mass = float(retained_mass)
        self.cache = LazyDistanceSemanticGeometryPredictiveCache(
            field,
            resolutions=self.resolutions,
            feature_tolerance=feature_tolerance,
        )
        self.support_observations = 0
        self.support_fraction_sum = 0.0
        self.support_fraction_max = 0.0
        self.dropped_mass_sum = 0.0
        self.dropped_mass_max = 0.0

    def prepare(
        self,
        observation: Sequence[float],
    ) -> PrequentialBundle:
        decision = self.decision()
        full_weights = self.field.perceptual_weights(
            observation
        )
        (
            candidate_predictions,
            local_prediction,
            support,
        ) = self.cache.prediction_bundle_from_weights(
            full_weights,
            self.current_indices,
            retained_mass=self.retained_mass,
        )

        self.support_observations += 1
        self.support_fraction_sum += support.fraction
        self.support_fraction_max = max(
            self.support_fraction_max,
            support.fraction,
        )
        self.dropped_mass_sum += support.dropped_mass
        self.dropped_mass_max = max(
            self.dropped_mass_max,
            support.dropped_mass,
        )

        return PrequentialBundle(
            candidate_predictions=(
                candidate_predictions
            ),
            local_prediction=local_prediction,
            decision=decision,
            full_weights=full_weights,
            support=support,
        )

    def _update_tracker_from_weights(
        self,
        weights: np.ndarray,
        outcome: int,
        candidate_predictions: np.ndarray,
    ) -> None:
        tracker = self.tracker
        if not isinstance(
            tracker,
            SparseRegionalPredictiveLoss,
        ):
            raise TypeError(
                "fused controller requires sparse loss tracker"
            )
        if not 0 <= outcome < self.field.outcomes:
            raise ValueError("invalid outcome")

        predictions = np.asarray(
            candidate_predictions,
            dtype=float,
        )
        expected = (
            len(self.resolutions),
            self.field.outcomes,
        )
        if predictions.shape != expected:
            raise ValueError(
                f"candidate_predictions must have shape {expected}"
            )

        before = tracker.local_losses().copy()
        losses = np.asarray(
            [
                -float(
                    np.log(
                        max(
                            float(prediction[outcome]),
                            tracker.probability_floor,
                        )
                    )
                )
                for prediction in predictions
            ],
            dtype=float,
        )

        tracker.loss_numerator *= (
            tracker.loss_decay
        )
        tracker.anchor_mass *= tracker.loss_decay
        tracker.loss_numerator += (
            losses[:, None] * weights[None, :]
        )
        tracker.anchor_mass += weights
        tracker.observations += 1

        after = tracker.local_losses()
        delta = np.max(
            np.abs(after - before),
            axis=0,
        )
        tracker.last_dirty_mask = (
            delta >= tracker.loss_change_tolerance
        )
        tracker.last_weights = weights.copy()

    def _update_field_from_weights(
        self,
        weights: np.ndarray,
        outcome: int,
    ) -> None:
        if not 0 <= outcome < self.field.outcomes:
            raise ValueError("invalid outcome")
        if self.field.evidence_decay < 1.0:
            self.field.outcome_counts *= (
                self.field.evidence_decay
            )
        self.field.outcome_counts[:, outcome] += weights

    def commit(
        self,
        bundle: PrequentialBundle,
        outcome: int,
    ) -> None:
        # Predictive readout may be support-compressed, but evidence attribution
        # and field learning always retain the full perceptual responsibilities.
        self._update_tracker_from_weights(
            bundle.full_weights,
            outcome,
            bundle.candidate_predictions,
        )
        self._update_field_from_weights(
            bundle.full_weights,
            outcome,
        )
        self.cache.refresh()

    def support_diagnostics(self) -> dict:
        count = max(1, self.support_observations)
        return {
            "retained_mass_target": (
                self.retained_mass
            ),
            "mean_support_fraction": (
                self.support_fraction_sum / count
            ),
            "max_support_fraction": (
                self.support_fraction_max
            ),
            "mean_dropped_mass": (
                self.dropped_mass_sum / count
            ),
            "max_dropped_mass": (
                self.dropped_mass_max
            ),
        }
