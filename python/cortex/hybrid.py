"""Stage-I neural-perception adapter for Cortex.

The neural encoder is deliberately external and frozen. Cortex receives only
metric-calibrated embedding vectors; packet metadata is retained at the Python
research boundary and does not alter native state transitions in Stage I.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import Any, Protocol, Sequence, runtime_checkable


@runtime_checkable
class FrozenEncoder(Protocol):
    """Minimal contract for a frozen perceptual encoder."""

    output_dim: int
    name: str

    def encode(self, batch: Sequence[Any]) -> Sequence[Sequence[float]]:
        """Return one embedding per input without updating encoder weights."""


@dataclass(frozen=True, slots=True)
class NeuralObservation:
    """One perceptual observation presented to the hybrid boundary.

    Only embedding enters Cortex in Stage I. The remaining fields are
    provenance/diagnostic metadata reserved for evaluation and later stages.
    """

    embedding: Sequence[float]
    confidence: float | None = None
    uncertainty: Sequence[float] | None = None
    bbox: tuple[float, float, float, float] | None = None
    timestamp: int | float | None = None
    source: str | None = None


def cosine_mse_embedding(embedding: Sequence[float]) -> list[float]:
    """Map an embedding so Cortex MSE equals cosine distance.

    For d-dimensional vectors a,b, each adapted vector is

        x = sqrt(d / 2) * a / ||a||_2.

    Therefore the unweighted mean squared error between adapted vectors is
    exactly 1 - cosine(a, b).
    """

    values = [float(v) for v in embedding]
    if not values:
        raise ValueError("embedding must be non-empty")
    if any(not math.isfinite(v) for v in values):
        raise ValueError("embedding contains non-finite values")
    norm_sq = sum(v * v for v in values)
    if norm_sq <= 1.0e-24:
        raise ValueError("embedding norm must be non-zero")
    scale = math.sqrt(len(values) / (2.0 * norm_sq))
    return [scale * v for v in values]


def _validate_packet(packet: NeuralObservation, dim: int) -> list[float]:
    if len(packet.embedding) != dim:
        raise ValueError(f"expected embedding dimension {dim}, got {len(packet.embedding)}")
    if packet.confidence is not None:
        confidence = float(packet.confidence)
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be finite and in [0, 1]")
    if packet.uncertainty is not None:
        if len(packet.uncertainty) != dim:
            raise ValueError(
                f"expected uncertainty dimension {dim}, got {len(packet.uncertainty)}"
            )
        for value in packet.uncertainty:
            value = float(value)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError("uncertainty values must be finite and non-negative")
    if packet.bbox is not None:
        if len(packet.bbox) != 4 or any(not math.isfinite(float(v)) for v in packet.bbox):
            raise ValueError("bbox must contain four finite values")
    if packet.timestamp is not None and not math.isfinite(float(packet.timestamp)):
        raise ValueError("timestamp must be finite")
    return cosine_mse_embedding(packet.embedding)


class HybridCortexRuntime:
    """Frozen-encoder -> Cortex Stage-I composition.

    Generic neural latent coordinates do not carry Cortex's cyclic coordinate
    action, so the default hybrid configuration uses the identity nuisance
    group. nuisance_mode='native' is available only for representations whose
    coordinate action is intentionally compatible with Cortex.
    """

    def __init__(
        self,
        embedding_dim: int,
        *,
        max_entities: int = 96,
        budget: int = 24,
        config_json: str | None = None,
        nuisance_mode: str = "identity-only",
    ) -> None:
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        from . import CortexRuntime

        if config_json is None:
            seed = CortexRuntime(
                feature_dim=embedding_dim,
                max_entities=max_entities,
                budget=budget,
            )
            cfg = json.loads(seed.config_json())
        else:
            cfg = json.loads(config_json)
            configured_dim = int(cfg["articulation"]["feature_dim"])
            if configured_dim != embedding_dim:
                raise ValueError(
                    f"config feature_dim {configured_dim} does not match embedding_dim {embedding_dim}"
                )

        if nuisance_mode == "identity-only":
            cfg["articulation"]["fixed_group"] = [0]
        elif nuisance_mode == "native":
            cfg["articulation"]["fixed_group"] = None
        else:
            raise ValueError("nuisance_mode must be 'identity-only' or 'native'")

        self.embedding_dim = embedding_dim
        self.nuisance_mode = nuisance_mode
        self._config_json = json.dumps(cfg, separators=(",", ":"), sort_keys=True)
        self._runtime = CortexRuntime(config_json=self._config_json)

    def adapt(self, embedding: Sequence[float]) -> list[float]:
        if len(embedding) != self.embedding_dim:
            raise ValueError(
                f"expected embedding dimension {self.embedding_dim}, got {len(embedding)}"
            )
        return cosine_mse_embedding(embedding)

    def step(
        self,
        observations: Sequence[NeuralObservation],
        relation_obs: Sequence[tuple[int, int, int]] = (),
        intervention_src_det: int | None = None,
        outcomes: Sequence[tuple[int, int]] = (),
        outcome: float | None = None,
    ):
        detections = [_validate_packet(packet, self.embedding_dim) for packet in observations]
        return self._runtime.step(
            detections,
            list(relation_obs),
            intervention_src_det,
            list(outcomes),
            outcome,
        )

    def step_embeddings(
        self,
        embeddings: Sequence[Sequence[float]],
        **kwargs: Any,
    ):
        return self.step([NeuralObservation(embedding=x) for x in embeddings], **kwargs)

    def encode_and_step(
        self,
        encoder: FrozenEncoder,
        batch: Sequence[Any],
        **kwargs: Any,
    ):
        if int(encoder.output_dim) != self.embedding_dim:
            raise ValueError(
                f"encoder output_dim {encoder.output_dim} does not match {self.embedding_dim}"
            )
        embeddings = encoder.encode(batch)
        if len(embeddings) != len(batch):
            raise ValueError("encoder must return one embedding per input")
        return self.step_embeddings(embeddings, **kwargs)

    def snapshot_json(self) -> str:
        return self._runtime.snapshot_json()

    def config_json(self) -> str:
        return self._config_json

    def articulation_vector(self) -> list[float]:
        return list(self._runtime.articulation_vector())

    @property
    def runtime(self):
        """Expose the native runtime for research instrumentation only."""
        return self._runtime
