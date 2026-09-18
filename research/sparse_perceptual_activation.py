"""Cortex v1.11-R sparse perceptual activation research controller.

v1.9-R reduced redundant semantic geometry and local-resolution maintenance, but
prediction still multiplies a dense perceptual responsibility vector through
every row of every candidate kernel.

This module keeps evidence assimilation exact while allowing predictive readout
to retain only the smallest perceptual support carrying a declared probability
mass. A fused step also reuses the exact perceptual weights across candidate
prediction, local prediction, regional-loss update, and field evidence update.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from hysteretic_local_resolution import LocalResolutionDecision
from sparse_cached_resolution import (
    SparseCachedLocalResolutionController,
    SparseRegionalPredictiveLoss,
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
class ActivationStep:
    candidate_predictions: np.ndarray
    prediction: np.ndarray
    decision: LocalResolutionDecision
    exact_weights: np.ndarray
    support: np.ndarray
    retained_mass: float


def retained_mass_support(
    weights: Sequence[float],
    *,
    retained_mass: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return a deterministic minimum prefix carrying the requested mass."""
    values = np.asarray(weights, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("weights must be a non-empty vector")
    if np.any(values < 0) or not np.all(np.isfinite(values)):
        raise ValueError("weights must be finite and non-negative")
    total = float(values.sum())
    if total <= 1.0e-15:
        values = np.full(
            len(values),
            1.0 / len(values),
            dtype=float,
        )
    else:
        values = values / total

    if not (0 < retained_mass <= 1):
        raise ValueError("retained_mass must be in (0, 1]")

    if retained_mass >= 1.0 - 1.0e-15:
        support = np.arange(len(values), dtype=int)
        return support, values.copy(), 1.0

    order = np.argsort(-values, kind="mergesort")
    cumulative = np.cumsum(values[order])
    count = int(
        np.searchsorted(
            cumulative,
            retained_mass,
            side="left",
        )
        + 1
    )
    support = order[:count]
    mass = float(values[support].sum())
    truncated = values[support] / max(mass, 1.0e-15)
    return support, truncated, mass


class SparsePerceptualActivationController(
    SparseCachedLocalResolutionController
):
    """v1.9-R controller with fused and optionally sparse predictive activation."""

    def __init__(
        self,
        *args,
        retained_mass: float = 0.999,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if not (0 < retained_mass <= 1):
            raise ValueError(
                "retained_mass must be in (0, 1]"
            )
        self.retained_mass = float(retained_mass)
        self.activation_calls = 0
        self.support_items = 0
        self.retained_mass_sum = 0.0
        self.minimum_retained_mass = 1.0

    def _candidate_predictions_from_support(
        self,
        support: np.ndarray,
        weights: np.ndarray,
    ) -> np.ndarray:
        kernel_rows = self.cache.kernels[
            :,
            support,
            :,
        ]
        activations = np.einsum(
            "a,rab->rb",
            weights,
            kernel_rows,
            optimize=True,
        )
        semantic_fields = _normalize_rows(activations)
        predictions = (
            semantic_fields
            @ self.cache.probabilities
        )
        return _normalize_rows(predictions)

    def _local_prediction_from_support(
        self,
        support: np.ndarray,
        weights: np.ndarray,
    ) -> np.ndarray:
        indexes = self.current_indices[support]
        local_rows = self.cache.kernels[
            indexes,
            support,
            :,
        ]
        activation = weights @ local_rows
        total = float(activation.sum())
        if total <= 1.0e-15:
            semantic_field = np.full(
                len(self.field.anchors),
                1.0 / len(self.field.anchors),
            )
        else:
            semantic_field = activation / total

        prediction = (
            semantic_field
            @ self.cache.probabilities
        )
        prediction_total = float(prediction.sum())
        if prediction_total <= 1.0e-15:
            return np.full(
                self.field.outcomes,
                1.0 / self.field.outcomes,
            )
        return prediction / prediction_total

    def predict_step(
        self,
        observation: Sequence[float],
    ) -> ActivationStep:
        decision = self.decision()
        exact_weights = self.field.perceptual_weights(
            observation
        )
        support, weights, mass = retained_mass_support(
            exact_weights,
            retained_mass=self.retained_mass,
        )

        candidates = (
            self._candidate_predictions_from_support(
                support,
                weights,
            )
        )
        prediction = (
            self._local_prediction_from_support(
                support,
                weights,
            )
        )

        self.activation_calls += 1
        self.support_items += len(support)
        self.retained_mass_sum += mass
        self.minimum_retained_mass = min(
            self.minimum_retained_mass,
            mass,
        )

        return ActivationStep(
            candidate_predictions=candidates,
            prediction=prediction,
            decision=decision,
            exact_weights=exact_weights,
            support=support,
            retained_mass=mass,
        )

    def observe_step(
        self,
        observation: Sequence[float],
        outcome: int,
        step: ActivationStep,
    ) -> None:
        if not 0 <= outcome < self.field.outcomes:
            raise ValueError("invalid outcome")

        tracker = self.tracker
        if not isinstance(
            tracker,
            SparseRegionalPredictiveLoss,
        ):
            raise TypeError(
                "fused activation requires SparseRegionalPredictiveLoss"
            )

        predictions = np.asarray(
            step.candidate_predictions,
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

        weights = np.asarray(
            step.exact_weights,
            dtype=float,
        )
        tracker.loss_numerator *= tracker.loss_decay
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
            delta
            >= tracker.loss_change_tolerance
        )
        tracker.last_weights = weights

        if self.field.evidence_decay < 1.0:
            self.field.outcome_counts *= (
                self.field.evidence_decay
            )
        self.field.outcome_counts[:, outcome] += (
            weights
        )

        self.cache.refresh()

    def activation_diagnostics(self) -> dict:
        calls = max(1, self.activation_calls)
        anchors = len(self.field.anchors)
        return {
            "mean_support_size": (
                self.support_items / calls
            ),
            "mean_support_fraction": (
                self.support_items
                / (calls * anchors)
            ),
            "mean_retained_mass": (
                self.retained_mass_sum / calls
            ),
            "minimum_retained_mass": (
                self.minimum_retained_mass
            ),
        }
