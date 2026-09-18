"""Cortex v1.9-R sparse cached local-resolution research controller.

The v1.8-R controller is retained as the behavioral reference. This module
introduces two optimization layers:

1. exact shared-state caching: semantic features, predictive distances and all
   candidate kernels are built once per field state and reused by candidate
   prediction plus local-operator assembly;
2. sparse incremental refresh: semantic rows are refreshed only after their
   accumulated feature drift exceeds a declared tolerance, hysteresis scans only
   materially changed regions (with periodic full scans), and complexity SVD is
   sampled/cached because it is telemetry rather than a control input.

The sparse mode is approximate and must be audited against v1.8-R. The exact
cached mode exists to separate implementation speedup from approximation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from fuzzy_predictive_field import (
    FuzzyPredictiveField,
    predictive_distance_matrix,
)
from hysteretic_local_resolution import (
    HystereticLocalResolutionController,
    LocalResolutionDecision,
    RegionalPredictiveLoss,
    effective_singular_rank,
)


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    totals = values.sum(axis=1, keepdims=True)
    fallback = totals[:, 0] <= 1.0e-15
    safe = np.maximum(totals, 1.0e-15)
    normalized = values / safe
    if np.any(fallback):
        normalized[fallback] = 1.0 / values.shape[1]
    return normalized


@dataclass(frozen=True)
class CacheDiagnostics:
    kernel_revision: int
    full_rebuilds: int
    partial_refreshes: int
    rows_refreshed: int
    refresh_events: int
    last_dirty_fraction: float


class PredictiveKernelCache:
    """Shared candidate-kernel cache for one fuzzy predictive field."""

    def __init__(
        self,
        field: FuzzyPredictiveField,
        *,
        resolutions: Sequence[float],
        feature_tolerance: float = 0.0,
    ) -> None:
        values = tuple(float(r) for r in resolutions)
        if not values or any(r <= 0 or not np.isfinite(r) for r in values):
            raise ValueError("resolutions must be finite and positive")
        if feature_tolerance < 0 or not np.isfinite(feature_tolerance):
            raise ValueError(
                "feature_tolerance must be finite and non-negative"
            )

        self.field = field
        self.resolutions = values
        self.feature_tolerance = float(feature_tolerance)
        self.features = field.features()
        self.probabilities = field.outcome_probabilities()
        self.distances = predictive_distance_matrix(self.features)
        self.kernels = self._kernels_from_distances(self.distances)

        self.kernel_revision = 0
        self.full_rebuilds = 1
        self.partial_refreshes = 0
        self.rows_refreshed = len(field.anchors)
        self.refresh_events = 1
        self.last_dirty_mask = np.ones(len(field.anchors), dtype=bool)

    def _kernels_from_distances(
        self,
        distances: np.ndarray,
    ) -> np.ndarray:
        return np.asarray(
            [
                np.exp(
                    -0.5
                    * (distances / resolution) ** 2
                )
                for resolution in self.resolutions
            ],
            dtype=float,
        )

    def refresh(self) -> np.ndarray:
        """Refresh cached geometry after the field consumes one observation.

        feature_tolerance == 0 performs a full exact rebuild. Positive tolerance
        bounds the drift of every clean cached semantic row until accumulated
        drift crosses the threshold.
        """
        current_features = self.field.features()
        self.probabilities = self.field.outcome_probabilities()
        anchors = len(self.field.anchors)
        self.refresh_events += 1

        if self.feature_tolerance == 0.0:
            self.features = current_features
            self.distances = predictive_distance_matrix(self.features)
            self.kernels = self._kernels_from_distances(
                self.distances
            )
            self.last_dirty_mask = np.ones(anchors, dtype=bool)
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
        diff = (
            self.features[dirty, None, :]
            - self.features[None, :, :]
        )
        rows = np.linalg.norm(diff, axis=2) / np.sqrt(2.0)
        self.distances[dirty, :] = rows
        self.distances[:, dirty] = rows.T

        for index, resolution in enumerate(self.resolutions):
            kernel_rows = np.exp(
                -0.5
                * (rows / resolution) ** 2
            )
            self.kernels[index, dirty, :] = kernel_rows
            self.kernels[index, :, dirty] = kernel_rows.T

        self.kernel_revision += 1
        self.partial_refreshes += 1
        self.rows_refreshed += int(np.sum(dirty))
        return dirty.copy()

    def candidate_predictions(
        self,
        observation: Sequence[float],
    ) -> np.ndarray:
        weights = self.field.perceptual_weights(observation)
        activations = np.einsum(
            "a,rab->rb",
            weights,
            self.kernels,
            optimize=True,
        )
        semantic_fields = _normalize_rows(activations)
        predictions = semantic_fields @ self.probabilities
        return _normalize_rows(predictions)

    def operator(
        self,
        resolution_indices: Sequence[int],
    ) -> np.ndarray:
        indexes = np.asarray(resolution_indices, dtype=int)
        anchors = len(self.field.anchors)
        if indexes.shape != (anchors,):
            raise ValueError(
                "one resolution index is required per anchor"
            )
        if np.any(indexes < 0) or np.any(
            indexes >= len(self.resolutions)
        ):
            raise ValueError("invalid resolution index")
        rows = np.arange(anchors)
        return self.kernels[indexes, rows, :]

    def predict_local_outcome(
        self,
        observation: Sequence[float],
        resolution_indices: Sequence[int],
    ) -> np.ndarray:
        operator = self.operator(resolution_indices)
        weights = self.field.perceptual_weights(observation)
        activation = weights @ operator
        total = float(activation.sum())
        if total <= 1.0e-15:
            semantic_field = np.full(
                len(self.field.anchors),
                1.0 / len(self.field.anchors),
            )
        else:
            semantic_field = activation / total

        prediction = semantic_field @ self.probabilities
        prediction_total = float(prediction.sum())
        if prediction_total <= 1.0e-15:
            return np.full(
                self.field.outcomes,
                1.0 / self.field.outcomes,
            )
        return prediction / prediction_total

    def diagnostics(self) -> CacheDiagnostics:
        anchors = len(self.field.anchors)
        return CacheDiagnostics(
            kernel_revision=self.kernel_revision,
            full_rebuilds=self.full_rebuilds,
            partial_refreshes=self.partial_refreshes,
            rows_refreshed=self.rows_refreshed,
            refresh_events=self.refresh_events,
            last_dirty_fraction=float(
                np.sum(self.last_dirty_mask) / anchors
            ),
        )


class SparseRegionalPredictiveLoss(RegionalPredictiveLoss):
    """Regional loss tracker exposing materially changed loss columns."""

    def __init__(
        self,
        field: FuzzyPredictiveField,
        *,
        resolutions: Sequence[float],
        loss_decay: float,
        minimum_relative_mass: float,
        probability_floor: float,
        loss_change_tolerance: float,
    ) -> None:
        super().__init__(
            field,
            resolutions=resolutions,
            loss_decay=loss_decay,
            minimum_relative_mass=minimum_relative_mass,
            probability_floor=probability_floor,
        )
        if (
            loss_change_tolerance < 0
            or not np.isfinite(loss_change_tolerance)
        ):
            raise ValueError(
                "loss_change_tolerance must be finite and non-negative"
            )
        self.loss_change_tolerance = float(
            loss_change_tolerance
        )
        self.last_dirty_mask = np.ones(
            len(field.anchors),
            dtype=bool,
        )
        self.last_weights = np.zeros(
            len(field.anchors),
            dtype=float,
        )

    def update(
        self,
        observation: Sequence[float],
        outcome: int,
        *,
        candidate_predictions: np.ndarray,
    ) -> None:
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

        before = self.local_losses().copy()
        weights = self.field.perceptual_weights(observation)
        losses = np.asarray(
            [
                -float(
                    np.log(
                        max(
                            float(prediction[outcome]),
                            self.probability_floor,
                        )
                    )
                )
                for prediction in predictions
            ],
            dtype=float,
        )

        self.loss_numerator *= self.loss_decay
        self.anchor_mass *= self.loss_decay
        self.loss_numerator += (
            losses[:, None] * weights[None, :]
        )
        self.anchor_mass += weights
        self.observations += 1

        after = self.local_losses()
        delta = np.max(np.abs(after - before), axis=0)
        self.last_dirty_mask = (
            delta >= self.loss_change_tolerance
        )
        self.last_weights = weights


class CachedLocalResolutionController(
    HystereticLocalResolutionController
):
    """Exact v1.8 behavior with shared semantic/kernels per field state."""

    def __init__(
        self,
        field: FuzzyPredictiveField,
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
        )
        self.cache = PredictiveKernelCache(
            field,
            resolutions=self.resolutions,
            feature_tolerance=0.0,
        )

    def candidate_predictions(
        self,
        observation: Sequence[float],
    ) -> np.ndarray:
        return self.cache.candidate_predictions(observation)

    def decision(self) -> LocalResolutionDecision:
        self._advance_state()
        operator = self.cache.operator(
            self.current_indices
        )
        complexity = effective_singular_rank(operator)
        resolutions = self.resolution_vector()
        refined = float(
            np.mean(
                resolutions < max(self.resolutions)
            )
        )
        return LocalResolutionDecision(
            resolutions=tuple(
                float(x) for x in resolutions
            ),
            complexity=complexity,
            refined_fraction=refined,
            switch_count=int(
                np.sum(self.switch_counts)
            ),
        )

    def predict_outcome(
        self,
        observation: Sequence[float],
    ) -> tuple[np.ndarray, LocalResolutionDecision]:
        decision = self.decision()
        prediction = self.cache.predict_local_outcome(
            observation,
            self.current_indices,
        )
        return prediction, decision

    def observe_outcome(
        self,
        observation: Sequence[float],
        outcome: int,
        *,
        candidate_predictions: np.ndarray,
    ) -> None:
        self.tracker.update(
            observation,
            outcome,
            candidate_predictions=candidate_predictions,
        )
        self.field.observe_outcome(
            observation,
            outcome,
        )
        self.cache.refresh()


class SparseCachedLocalResolutionController(
    HystereticLocalResolutionController
):
    """Approximate sparse v1.9-R controller audited against v1.8-R."""

    def __init__(
        self,
        field: FuzzyPredictiveField,
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
        )
        if full_scan_interval < 1:
            raise ValueError(
                "full_scan_interval must be positive"
            )
        if complexity_refresh_interval < 1:
            raise ValueError(
                "complexity_refresh_interval must be positive"
            )

        self.tracker = SparseRegionalPredictiveLoss(
            field,
            resolutions=self.resolutions,
            loss_decay=loss_decay,
            minimum_relative_mass=minimum_relative_mass,
            probability_floor=probability_floor,
            loss_change_tolerance=loss_change_tolerance,
        )
        self.cache = PredictiveKernelCache(
            field,
            resolutions=self.resolutions,
            feature_tolerance=feature_tolerance,
        )
        self.full_scan_interval = int(
            full_scan_interval
        )
        self.complexity_refresh_interval = int(
            complexity_refresh_interval
        )

        self.scan_events = 0
        self.anchor_scans = 0
        self.full_scans = 0
        self.svd_calls = 0
        self._cached_complexity: float | None = None
        self._complexity_observation = -1
        self._complexity_signature: tuple[int, ...] | None = None

    def _advance_state(self) -> None:
        if (
            self._last_transition_observation
            == self.tracker.observations
        ):
            return
        self._last_transition_observation = (
            self.tracker.observations
        )

        if (
            self.tracker.observations
            < self.warmup_observations
        ):
            for anchor in range(
                len(self.current_indices)
            ):
                self._reset_pending(anchor)
            return

        losses = self.tracker.local_losses()
        active = self.tracker.active_anchors()
        force_full = (
            self.tracker.observations
            == self.warmup_observations
            or self.tracker.observations
            % self.full_scan_interval
            == 0
        )

        if force_full:
            eligible = active.copy()
            self.full_scans += 1
        else:
            pending = self.pending_targets >= 0
            eligible = active & (
                self.tracker.last_dirty_mask
                | pending
            )

        indexes = np.flatnonzero(eligible)
        self.scan_events += 1
        self.anchor_scans += len(indexes)

        for anchor in indexes:
            anchor = int(anchor)
            current = int(
                self.current_indices[anchor]
            )
            mode, target = self._transition_target(
                losses[:, anchor],
                current,
                epsilon=self.epsilon,
                hysteresis_margin=self.hysteresis_margin,
            )

            if mode == "sharpen" and target is not None:
                if target != current:
                    self.current_indices[anchor] = int(
                        target
                    )
                    self.switch_counts[anchor] += 1
                self._reset_pending(anchor)
                continue

            if mode == "coarsen" and target is not None:
                if (
                    self.pending_targets[anchor]
                    == target
                ):
                    self.pending_counts[anchor] += 1
                else:
                    self.pending_targets[anchor] = int(
                        target
                    )
                    self.pending_counts[anchor] = 1

                if (
                    self.pending_counts[anchor]
                    >= self.coarsen_patience
                ):
                    if target != current:
                        self.current_indices[anchor] = int(
                            target
                        )
                        self.switch_counts[anchor] += 1
                    self._reset_pending(anchor)
                continue

            self._reset_pending(anchor)

    def _complexity(self) -> float:
        signature = tuple(
            int(x) for x in self.current_indices
        )
        observation = self.tracker.observations
        due = (
            self._cached_complexity is None
            or signature
            != self._complexity_signature
            or observation
            - self._complexity_observation
            >= self.complexity_refresh_interval
        )
        if due:
            operator = self.cache.operator(
                self.current_indices
            )
            self._cached_complexity = (
                effective_singular_rank(operator)
            )
            self._complexity_observation = (
                observation
            )
            self._complexity_signature = signature
            self.svd_calls += 1
        return float(self._cached_complexity)

    def candidate_predictions(
        self,
        observation: Sequence[float],
    ) -> np.ndarray:
        return self.cache.candidate_predictions(observation)

    def decision(self) -> LocalResolutionDecision:
        self._advance_state()
        resolutions = self.resolution_vector()
        refined = float(
            np.mean(
                resolutions < max(self.resolutions)
            )
        )
        return LocalResolutionDecision(
            resolutions=tuple(
                float(x) for x in resolutions
            ),
            complexity=self._complexity(),
            refined_fraction=refined,
            switch_count=int(
                np.sum(self.switch_counts)
            ),
        )

    def exact_complexity(self) -> float:
        return effective_singular_rank(
            self.cache.operator(
                self.current_indices
            )
        )

    def predict_outcome(
        self,
        observation: Sequence[float],
    ) -> tuple[np.ndarray, LocalResolutionDecision]:
        decision = self.decision()
        prediction = self.cache.predict_local_outcome(
            observation,
            self.current_indices,
        )
        return prediction, decision

    def observe_outcome(
        self,
        observation: Sequence[float],
        outcome: int,
        *,
        candidate_predictions: np.ndarray,
    ) -> None:
        self.tracker.update(
            observation,
            outcome,
            candidate_predictions=candidate_predictions,
        )
        self.field.observe_outcome(
            observation,
            outcome,
        )
        self.cache.refresh()

    def optimization_diagnostics(self) -> dict:
        cache = self.cache.diagnostics()
        anchors = len(self.field.anchors)
        possible_scans = max(
            1,
            self.scan_events * anchors,
        )
        possible_refresh_rows = max(
            1,
            self.cache.refresh_events * anchors,
        )
        return {
            "anchor_scan_fraction": (
                self.anchor_scans
                / possible_scans
            ),
            "full_scans": self.full_scans,
            "svd_calls": self.svd_calls,
            "svd_fraction": (
                self.svd_calls
                / max(
                    1,
                    self.tracker.observations + 1,
                )
            ),
            "kernel_row_refresh_fraction": (
                self.cache.rows_refreshed
                / possible_refresh_rows
            ),
            "kernel_revision": (
                cache.kernel_revision
            ),
            "partial_refreshes": (
                cache.partial_refreshes
            ),
            "full_rebuilds": cache.full_rebuilds,
        }
