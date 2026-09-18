"""Cortex v1.8-R hysteretic local adaptive resolution research controller.

This module extends the v1.7-R predictive rate-distortion controller in two ways:

1. hysteresis: resolution state is sticky inside a dead band, sharpening is fast,
   and coarsening requires sustained evidence;
2. local resolution: each perceptual anchor can select its own predictive
   resolution, producing a row-adaptive predictive operator.

The production runtime and frozen executable specification are unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from fuzzy_predictive_field import (
    FuzzyPredictiveField,
    predictive_distance_matrix,
    responsibilities,
)


@dataclass(frozen=True)
class GlobalResolutionDecision:
    resolution: float
    distortion: float
    best_distortion: float
    complexity: float
    switch_count: int


@dataclass(frozen=True)
class LocalResolutionDecision:
    resolutions: tuple[float, ...]
    complexity: float
    refined_fraction: float
    switch_count: int


def effective_singular_rank(operator: np.ndarray) -> float:
    """Entropy-effective rank for a possibly non-symmetric linear operator."""
    matrix = np.asarray(operator, dtype=float)
    if matrix.ndim != 2 or matrix.size == 0 or not np.all(np.isfinite(matrix)):
        raise ValueError("operator must be a finite non-empty matrix")
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    singular_values = np.clip(singular_values, 0.0, None)
    total = float(singular_values.sum())
    if total <= 1.0e-15:
        return 0.0
    p = singular_values / total
    p = p[p > 1.0e-15]
    return float(np.exp(-np.sum(p * np.log(p))))


def local_resolution_operator(
    field: FuzzyPredictiveField,
    resolutions: Sequence[float],
) -> np.ndarray:
    """Build the row-adaptive semantic operator K_loc[i,j] = K_rho_i[i,j]."""
    rho = np.asarray(resolutions, dtype=float)
    if rho.shape != (len(field.anchors),):
        raise ValueError("one local resolution is required per anchor")
    if np.any(rho <= 0) or not np.all(np.isfinite(rho)):
        raise ValueError("local resolutions must be finite and positive")

    distances = predictive_distance_matrix(field.features())
    return np.exp(-0.5 * (distances / rho[:, None]) ** 2)


def predict_local_outcome(
    field: FuzzyPredictiveField,
    observation: Sequence[float],
    resolutions: Sequence[float],
) -> np.ndarray:
    operator = local_resolution_operator(field, resolutions)
    activation = field.perceptual_weights(observation) @ operator
    semantic_field = responsibilities(activation)
    prediction = semantic_field @ field.outcome_probabilities()
    total = float(prediction.sum())
    if total <= 1.0e-15:
        return np.full(field.outcomes, 1.0 / field.outcomes)
    return prediction / total


class RegionalPredictiveLoss:
    """Prequential per-anchor candidate loss shared by global and local control."""

    def __init__(
        self,
        field: FuzzyPredictiveField,
        *,
        resolutions: Sequence[float],
        loss_decay: float,
        minimum_relative_mass: float,
        probability_floor: float,
    ) -> None:
        if not (0 < loss_decay < 1):
            raise ValueError("loss_decay must be in (0, 1)")
        if not (0 <= minimum_relative_mass <= 1):
            raise ValueError("minimum_relative_mass must be in [0, 1]")
        if not (0 < probability_floor < 1):
            raise ValueError("probability_floor must be in (0, 1)")

        self.field = field
        self.resolutions = tuple(float(r) for r in resolutions)
        self.loss_decay = float(loss_decay)
        self.minimum_relative_mass = float(minimum_relative_mass)
        self.probability_floor = float(probability_floor)

        candidates = len(self.resolutions)
        anchors = len(field.anchors)
        self.loss_numerator = np.zeros((candidates, anchors), dtype=float)
        self.anchor_mass = np.zeros(anchors, dtype=float)
        self.observations = 0

    def candidate_predictions(
        self,
        observation: Sequence[float],
    ) -> np.ndarray:
        return np.asarray(
            [
                self.field.predict_outcome(
                    observation,
                    resolution=resolution,
                )
                for resolution in self.resolutions
            ],
            dtype=float,
        )

    def local_losses(self) -> np.ndarray:
        return self.loss_numerator / np.maximum(
            self.anchor_mass,
            1.0e-12,
        )[None, :]

    def active_anchors(self) -> np.ndarray:
        maximum = float(np.max(self.anchor_mass))
        if maximum <= 1.0e-12:
            return np.ones(len(self.anchor_mass), dtype=bool)
        return (
            self.anchor_mass
            >= self.minimum_relative_mass * maximum
        )

    def global_distortions(self) -> np.ndarray:
        losses = self.local_losses()
        active = self.active_anchors()
        if not np.any(active):
            return np.max(losses, axis=1)
        return np.max(losses[:, active], axis=1)

    def update(
        self,
        observation: Sequence[float],
        outcome: int,
        *,
        candidate_predictions: np.ndarray,
    ) -> None:
        if not 0 <= outcome < self.field.outcomes:
            raise ValueError("invalid outcome")

        predictions = np.asarray(candidate_predictions, dtype=float)
        expected = (len(self.resolutions), self.field.outcomes)
        if predictions.shape != expected:
            raise ValueError(
                f"candidate_predictions must have shape {expected}"
            )

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
        self.loss_numerator += losses[:, None] * weights[None, :]
        self.anchor_mass += weights
        self.observations += 1


class _HysteresisMixin:
    def _validate_hysteresis(
        self,
        *,
        epsilon: float,
        hysteresis_margin: float,
        coarsen_patience: int,
        warmup_observations: int,
    ) -> None:
        if epsilon < 0 or not np.isfinite(epsilon):
            raise ValueError("epsilon must be finite and non-negative")
        if hysteresis_margin < 0 or not np.isfinite(hysteresis_margin):
            raise ValueError(
                "hysteresis_margin must be finite and non-negative"
            )
        if coarsen_patience < 1:
            raise ValueError("coarsen_patience must be positive")
        if warmup_observations < 1:
            raise ValueError("warmup_observations must be positive")

    def _transition_target(
        self,
        distortions: np.ndarray,
        current_index: int,
        *,
        epsilon: float,
        hysteresis_margin: float,
    ) -> tuple[str, int | None]:
        best = float(np.min(distortions))
        outer = epsilon + hysteresis_margin
        inner = max(0.0, epsilon - hysteresis_margin)

        if float(distortions[current_index]) > best + outer:
            admissible = [
                index
                for index in range(current_index)
                if float(distortions[index]) <= best + epsilon + 1.0e-12
            ]
            if admissible:
                return "sharpen", max(admissible)
            return "sharpen", int(np.argmin(distortions[: current_index + 1]))

        candidates = [
            index
            for index in range(
                current_index + 1,
                len(distortions),
            )
            if float(distortions[index]) <= best + inner + 1.0e-12
        ]
        if candidates:
            return "coarsen", max(candidates)
        return "hold", None


class HystereticGlobalResolutionController(_HysteresisMixin):
    """Global v1.7-R controller with stateful hysteresis."""

    def __init__(
        self,
        field: FuzzyPredictiveField,
        *,
        resolutions: Sequence[float] = (0.03, 0.06, 0.12, 0.25),
        epsilon: float = 0.04,
        hysteresis_margin: float = 0.01,
        coarsen_patience: int = 12,
        loss_decay: float = 0.995,
        warmup_observations: int = 60,
        minimum_relative_mass: float = 0.10,
        probability_floor: float = 1.0e-9,
    ) -> None:
        values = tuple(sorted({float(r) for r in resolutions}))
        if not values or any(r <= 0 or not np.isfinite(r) for r in values):
            raise ValueError("resolutions must be finite and positive")
        self._validate_hysteresis(
            epsilon=epsilon,
            hysteresis_margin=hysteresis_margin,
            coarsen_patience=coarsen_patience,
            warmup_observations=warmup_observations,
        )

        self.field = field
        self.resolutions = values
        self.epsilon = float(epsilon)
        self.hysteresis_margin = float(hysteresis_margin)
        self.coarsen_patience = int(coarsen_patience)
        self.warmup_observations = int(warmup_observations)
        self.tracker = RegionalPredictiveLoss(
            field,
            resolutions=values,
            loss_decay=loss_decay,
            minimum_relative_mass=minimum_relative_mass,
            probability_floor=probability_floor,
        )

        self.current_index = len(values) - 1
        self.pending_target: int | None = None
        self.pending_count = 0
        self.switch_count = 0
        self._last_transition_observation = -1

    def _reset_pending(self) -> None:
        self.pending_target = None
        self.pending_count = 0

    def _advance_state(self) -> None:
        if self._last_transition_observation == self.tracker.observations:
            return
        self._last_transition_observation = self.tracker.observations

        if self.tracker.observations < self.warmup_observations:
            self._reset_pending()
            return

        distortions = self.tracker.global_distortions()
        mode, target = self._transition_target(
            distortions,
            self.current_index,
            epsilon=self.epsilon,
            hysteresis_margin=self.hysteresis_margin,
        )

        if mode == "sharpen" and target is not None:
            if target != self.current_index:
                self.current_index = int(target)
                self.switch_count += 1
            self._reset_pending()
            return

        if mode == "coarsen" and target is not None:
            if self.pending_target == target:
                self.pending_count += 1
            else:
                self.pending_target = int(target)
                self.pending_count = 1
            if self.pending_count >= self.coarsen_patience:
                if target != self.current_index:
                    self.current_index = int(target)
                    self.switch_count += 1
                self._reset_pending()
            return

        self._reset_pending()

    def decision(self) -> GlobalResolutionDecision:
        self._advance_state()
        resolution = self.resolutions[self.current_index]
        complexity = self.field.predictive_complexity(
            resolution=resolution
        )
        if self.tracker.observations < self.warmup_observations:
            return GlobalResolutionDecision(
                resolution=resolution,
                distortion=float("nan"),
                best_distortion=float("nan"),
                complexity=float(complexity),
                switch_count=self.switch_count,
            )
        distortions = self.tracker.global_distortions()
        return GlobalResolutionDecision(
            resolution=resolution,
            distortion=float(distortions[self.current_index]),
            best_distortion=float(np.min(distortions)),
            complexity=float(complexity),
            switch_count=self.switch_count,
        )

    def candidate_predictions(
        self,
        observation: Sequence[float],
    ) -> np.ndarray:
        return self.tracker.candidate_predictions(observation)

    def predict_outcome(
        self,
        observation: Sequence[float],
        *,
        candidate_predictions: np.ndarray | None = None,
    ) -> tuple[np.ndarray, GlobalResolutionDecision]:
        predictions = (
            self.candidate_predictions(observation)
            if candidate_predictions is None
            else np.asarray(candidate_predictions, dtype=float)
        )
        decision = self.decision()
        index = self.resolutions.index(decision.resolution)
        return predictions[index], decision

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
        self.field.observe_outcome(observation, outcome)


class HystereticLocalResolutionController(_HysteresisMixin):
    """Per-anchor adaptive resolution with the same prequential evidence."""

    def __init__(
        self,
        field: FuzzyPredictiveField,
        *,
        resolutions: Sequence[float] = (0.03, 0.06, 0.12, 0.25),
        epsilon: float = 0.04,
        hysteresis_margin: float = 0.01,
        coarsen_patience: int = 12,
        loss_decay: float = 0.995,
        warmup_observations: int = 60,
        minimum_relative_mass: float = 0.10,
        probability_floor: float = 1.0e-9,
    ) -> None:
        values = tuple(sorted({float(r) for r in resolutions}))
        if not values or any(r <= 0 or not np.isfinite(r) for r in values):
            raise ValueError("resolutions must be finite and positive")
        self._validate_hysteresis(
            epsilon=epsilon,
            hysteresis_margin=hysteresis_margin,
            coarsen_patience=coarsen_patience,
            warmup_observations=warmup_observations,
        )

        self.field = field
        self.resolutions = values
        self.epsilon = float(epsilon)
        self.hysteresis_margin = float(hysteresis_margin)
        self.coarsen_patience = int(coarsen_patience)
        self.warmup_observations = int(warmup_observations)
        self.tracker = RegionalPredictiveLoss(
            field,
            resolutions=values,
            loss_decay=loss_decay,
            minimum_relative_mass=minimum_relative_mass,
            probability_floor=probability_floor,
        )

        anchors = len(field.anchors)
        self.current_indices = np.full(
            anchors,
            len(values) - 1,
            dtype=int,
        )
        self.pending_targets = np.full(anchors, -1, dtype=int)
        self.pending_counts = np.zeros(anchors, dtype=int)
        self.switch_counts = np.zeros(anchors, dtype=int)
        self._last_transition_observation = -1

    def resolution_vector(self) -> np.ndarray:
        return np.asarray(
            [self.resolutions[int(i)] for i in self.current_indices],
            dtype=float,
        )

    def _reset_pending(self, anchor: int) -> None:
        self.pending_targets[anchor] = -1
        self.pending_counts[anchor] = 0

    def _advance_state(self) -> None:
        if self._last_transition_observation == self.tracker.observations:
            return
        self._last_transition_observation = self.tracker.observations

        if self.tracker.observations < self.warmup_observations:
            for anchor in range(len(self.current_indices)):
                self._reset_pending(anchor)
            return

        losses = self.tracker.local_losses()
        active = self.tracker.active_anchors()

        for anchor in range(len(self.current_indices)):
            if not active[anchor]:
                self._reset_pending(anchor)
                continue

            current = int(self.current_indices[anchor])
            mode, target = self._transition_target(
                losses[:, anchor],
                current,
                epsilon=self.epsilon,
                hysteresis_margin=self.hysteresis_margin,
            )

            if mode == "sharpen" and target is not None:
                if target != current:
                    self.current_indices[anchor] = int(target)
                    self.switch_counts[anchor] += 1
                self._reset_pending(anchor)
                continue

            if mode == "coarsen" and target is not None:
                if self.pending_targets[anchor] == target:
                    self.pending_counts[anchor] += 1
                else:
                    self.pending_targets[anchor] = int(target)
                    self.pending_counts[anchor] = 1
                if (
                    self.pending_counts[anchor]
                    >= self.coarsen_patience
                ):
                    if target != current:
                        self.current_indices[anchor] = int(target)
                        self.switch_counts[anchor] += 1
                    self._reset_pending(anchor)
                continue

            self._reset_pending(anchor)

    def decision(self) -> LocalResolutionDecision:
        self._advance_state()
        resolutions = self.resolution_vector()
        operator = local_resolution_operator(self.field, resolutions)
        complexity = effective_singular_rank(operator)
        refined = float(np.mean(resolutions < max(self.resolutions)))
        return LocalResolutionDecision(
            resolutions=tuple(float(x) for x in resolutions),
            complexity=complexity,
            refined_fraction=refined,
            switch_count=int(np.sum(self.switch_counts)),
        )

    def candidate_predictions(
        self,
        observation: Sequence[float],
    ) -> np.ndarray:
        return self.tracker.candidate_predictions(observation)

    def predict_outcome(
        self,
        observation: Sequence[float],
    ) -> tuple[np.ndarray, LocalResolutionDecision]:
        decision = self.decision()
        prediction = predict_local_outcome(
            self.field,
            observation,
            decision.resolutions,
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
        self.field.observe_outcome(observation, outcome)
