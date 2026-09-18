"""Cortex v1.7-R predictive rate-distortion controller.

The controller selects field resolution online from prequential predictive loss.
It does not use future observations to choose the current resolution.

Decision rule:
    choose the minimum-complexity resolution whose worst supported regional
    predictive loss is within epsilon of the best candidate loss.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from fuzzy_predictive_field import FuzzyPredictiveField


@dataclass(frozen=True)
class ResolutionDecision:
    resolution: float
    distortion: float
    best_distortion: float
    complexity: float
    admissible: tuple[float, ...]


class AdaptiveResolutionController:
    def __init__(
        self,
        field: FuzzyPredictiveField,
        *,
        resolutions: Sequence[float] = (0.03, 0.06, 0.12, 0.25),
        epsilon: float = 0.04,
        loss_decay: float = 0.995,
        warmup_observations: int = 60,
        minimum_relative_mass: float = 0.10,
        probability_floor: float = 1.0e-9,
    ) -> None:
        values = tuple(sorted({float(r) for r in resolutions}))
        if not values or any(r <= 0 or not np.isfinite(r) for r in values):
            raise ValueError("resolutions must be finite and positive")
        if epsilon < 0 or not np.isfinite(epsilon):
            raise ValueError("epsilon must be finite and non-negative")
        if not (0 < loss_decay < 1):
            raise ValueError("loss_decay must be in (0, 1)")
        if warmup_observations < 1:
            raise ValueError("warmup_observations must be positive")
        if not (0 <= minimum_relative_mass <= 1):
            raise ValueError("minimum_relative_mass must be in [0, 1]")
        if not (0 < probability_floor < 1):
            raise ValueError("probability_floor must be in (0, 1)")

        self.field = field
        self.resolutions = values
        self.epsilon = epsilon
        self.loss_decay = loss_decay
        self.warmup_observations = warmup_observations
        self.minimum_relative_mass = minimum_relative_mass
        self.probability_floor = probability_floor

        candidates = len(values)
        anchors = len(field.anchors)
        self.loss_numerator = np.zeros((candidates, anchors), dtype=float)
        self.anchor_mass = np.zeros(anchors, dtype=float)
        self.observations = 0

    def _regional_losses(self) -> np.ndarray:
        denominator = np.maximum(self.anchor_mass, 1.0e-12)
        return self.loss_numerator / denominator[None, :]

    def _active_anchors(self) -> np.ndarray:
        maximum = float(np.max(self.anchor_mass))
        if maximum <= 1.0e-12:
            return np.ones(len(self.anchor_mass), dtype=bool)
        return self.anchor_mass >= self.minimum_relative_mass * maximum

    def distortions(self) -> np.ndarray:
        losses = self._regional_losses()
        active = self._active_anchors()
        if not np.any(active):
            return np.max(losses, axis=1)
        return np.max(losses[:, active], axis=1)

    def complexities(self) -> np.ndarray:
        return np.asarray(
            [
                self.field.predictive_complexity(resolution=resolution)
                for resolution in self.resolutions
            ],
            dtype=float,
        )

    def decision(self) -> ResolutionDecision:
        # Before enough prequential evidence exists, prefer the lowest-complexity
        # representation. No prediction claim is made from the warmup state.
        complexities = self.complexities()
        if self.observations < self.warmup_observations:
            index = int(np.argmin(complexities))
            return ResolutionDecision(
                resolution=self.resolutions[index],
                distortion=float("nan"),
                best_distortion=float("nan"),
                complexity=float(complexities[index]),
                admissible=tuple(self.resolutions),
            )

        distortions = self.distortions()
        best = float(np.min(distortions))
        admissible_indexes = np.flatnonzero(
            distortions <= best + self.epsilon + 1.0e-12
        )
        if len(admissible_indexes) == 0:
            admissible_indexes = np.asarray([int(np.argmin(distortions))])

        # Rate-distortion rule: among resolutions satisfying the predictive
        # distortion constraint, keep the least complex field.
        local = complexities[admissible_indexes]
        minimum_complexity = float(np.min(local))
        tied = admissible_indexes[
            np.flatnonzero(local <= minimum_complexity + 1.0e-12)
        ]
        # If complexity ties numerically, prefer the coarser resolution.
        index = int(max(tied, key=lambda i: self.resolutions[int(i)]))

        return ResolutionDecision(
            resolution=self.resolutions[index],
            distortion=float(distortions[index]),
            best_distortion=best,
            complexity=float(complexities[index]),
            admissible=tuple(
                self.resolutions[int(i)] for i in admissible_indexes
            ),
        )

    def predict_outcome(
        self,
        observation: Sequence[float],
    ) -> tuple[np.ndarray, ResolutionDecision]:
        decision = self.decision()
        prediction = self.field.predict_outcome(
            observation,
            resolution=decision.resolution,
        )
        return prediction, decision

    def observe_outcome(
        self,
        observation: Sequence[float],
        outcome: int,
    ) -> ResolutionDecision:
        if not 0 <= outcome < self.field.outcomes:
            raise ValueError("invalid outcome")

        # Every candidate is scored on the same pre-update state. The observed
        # outcome enters the controller only after all predictions are frozen.
        weights = self.field.perceptual_weights(observation)
        candidate_losses = []
        for resolution in self.resolutions:
            prediction = self.field.predict_outcome(
                observation,
                resolution=resolution,
            )
            probability = max(
                float(prediction[outcome]),
                self.probability_floor,
            )
            candidate_losses.append(-float(np.log(probability)))

        self.loss_numerator *= self.loss_decay
        self.anchor_mass *= self.loss_decay
        for index, loss in enumerate(candidate_losses):
            self.loss_numerator[index] += weights * loss
        self.anchor_mass += weights
        self.observations += 1

        self.field.observe_outcome(observation, outcome)
        return self.decision()
