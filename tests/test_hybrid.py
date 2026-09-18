from __future__ import annotations

import json
import math

import pytest

from cortex import CortexRuntime, HybridCortexRuntime, NeuralObservation, cosine_mse_embedding


def _mse(a, b):
    return sum((x - y) ** 2 for x, y in zip(a, b)) / len(a)


def _cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb)


def test_metric_adapter_maps_mse_to_cosine_distance():
    a = [1.0, 2.0, -1.0, 0.5]
    b = [-0.5, 0.25, 2.0, 1.0]
    xa = cosine_mse_embedding(a)
    xb = cosine_mse_embedding(b)
    assert _mse(xa, xb) == pytest.approx(1.0 - _cosine(a, b), abs=1.0e-12)


def test_adapter_rejects_zero_and_non_finite_embeddings():
    with pytest.raises(ValueError):
        cosine_mse_embedding([0.0, 0.0])
    with pytest.raises(ValueError):
        cosine_mse_embedding([1.0, float("nan")])


def test_hybrid_defaults_to_identity_only_nuisance_group():
    hybrid = HybridCortexRuntime(embedding_dim=4, max_entities=8, budget=4)
    cfg = json.loads(hybrid.config_json())
    assert cfg["articulation"]["fixed_group"] == [0]


def test_hybrid_is_exact_wrapper_over_adapted_native_vectors():
    embeddings = [
        [1.0, 0.0, 0.3, 0.2],
        [0.0, 1.0, -0.4, 0.1],
    ]
    hybrid = HybridCortexRuntime(embedding_dim=4, max_entities=8, budget=4)
    direct = CortexRuntime(config_json=hybrid.config_json())

    hybrid_read = hybrid.step_embeddings(embeddings)
    direct_read = direct.step([cosine_mse_embedding(x) for x in embeddings])

    assert hybrid_read.bindings == direct_read.bindings
    assert hybrid_read.subgroup == direct_read.subgroup
    assert hybrid_read.articulation_vector == pytest.approx(direct_read.articulation_vector)
    assert hybrid.snapshot_json() == direct.snapshot_json()


def test_packet_metadata_does_not_change_stage1_reasoning():
    embedding = [1.0, -0.2, 0.4, 0.1]
    left = HybridCortexRuntime(embedding_dim=4, max_entities=8, budget=4)
    right = HybridCortexRuntime(embedding_dim=4, max_entities=8, budget=4)

    a = NeuralObservation(
        embedding=embedding,
        confidence=0.2,
        uncertainty=[0.0, 0.1, 0.2, 0.3],
        bbox=(1.0, 2.0, 3.0, 4.0),
        timestamp=10,
        source="camera-a",
    )
    b = NeuralObservation(
        embedding=embedding,
        confidence=0.9,
        uncertainty=[0.5, 0.4, 0.3, 0.2],
        bbox=(9.0, 8.0, 7.0, 6.0),
        timestamp=99,
        source="camera-b",
    )

    assert left.step([a]).bindings == right.step([b]).bindings
    assert left.snapshot_json() == right.snapshot_json()


def test_packet_validation_is_explicit():
    hybrid = HybridCortexRuntime(embedding_dim=4)
    with pytest.raises(ValueError):
        hybrid.step([NeuralObservation([1.0, 2.0])])
    with pytest.raises(ValueError):
        hybrid.step([NeuralObservation([1.0, 0.0, 0.0, 0.0], confidence=1.1)])
    with pytest.raises(ValueError):
        hybrid.step(
            [
                NeuralObservation(
                    [1.0, 0.0, 0.0, 0.0],
                    uncertainty=[0.0, -1.0, 0.0, 0.0],
                )
            ]
        )
