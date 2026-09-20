from __future__ import annotations

from typing import Mapping, Sequence

import pytest

import cortex.strict_language_gateway as strict_module
from cortex.cognitive_orchestrator import (
    CognitiveOrchestratorConfig,
    CognitiveOrchestratorGateway,
)
from cortex.gateway_server import (
    build_gateway_from_env,
)


class FakeRead:
    def __init__(self, bindings):
        self.bindings = list(bindings)
        self.subgroup = [0]
        self.subgroup_margin = 0.2
        self.relation_changed = []
        self.causal_changed = []
        self.articulation_vector = [0.0] * 12
        self.memory_stored = 1
        self.memory_unresolved = False
        self.prediction = 0.8
        self.memberships = [0.9, 0.1]
        self.current_id = 0
        self.nearest_id = 0
        self.nearest_dist = 0.05
        self.revision = False
        self.reactivated = False
        self.discovered = False
        self.unresolved = False
        self.budget_pressure = False
        self.stored = 1


class FakeRuntime:
    def __init__(
        self,
        *,
        feature_dim,
        max_entities,
        budget,
    ):
        self.feature_dim = feature_dim
        self.vector_ids = {}

    def step(
        self,
        detections,
        relation_obs,
        intervention_src_det,
        outcomes,
        outcome,
    ):
        bindings = []
        for vector in detections:
            key = tuple(
                float(value)
                for value in vector
            )
            if key not in self.vector_ids:
                self.vector_ids[key] = len(
                    self.vector_ids
                )
            bindings.append(
                self.vector_ids[key]
            )
        return FakeRead(bindings)

    def snapshot_json(self):
        import json

        return json.dumps(
            {
                "feature_dim": self.feature_dim,
                "vector_ids": [
                    [list(key), value]
                    for key, value
                    in sorted(
                        self.vector_ids.items(),
                        key=lambda item: item[1],
                    )
                ],
            },
            sort_keys=True,
        )


class OrchestratorFakeClient:
    def __init__(self):
        self.calls = []
        self.revision_calls = 0

    def chat(
        self,
        *,
        model: str,
        messages: Sequence[
            Mapping[str, str]
        ],
        format_schema=None,
    ) -> str:
        call = {
            "model": model,
            "messages": [
                dict(message)
                for message in messages
            ],
            "structured": (
                format_schema
                is not None
            ),
        }
        self.calls.append(call)

        system = (
            messages[0]["content"]
            if messages
            and messages[0].get(
                "role"
            )
            == "system"
            else ""
        )
        user_text = messages[-1][
            "content"
        ]

        if (
            "candidate-answer analyzer"
            in system
        ):
            if (
                "Alice works with Bob."
                in user_text
                and "unresolved" not in user_text
            ):
                return (
                    '{"entities":['
                    '{"id":"e0","surface":"Alice","kind":"person","evidence":"Alice"},'
                    '{"id":"e1","surface":"Bob","kind":"person","evidence":"Bob"}'
                    '],"claims":['
                    '{"subject":"e0","predicate":"works_with","object":"e1",'
                    '"asserted":true,"evidence":"Alice works with Bob"}'
                    ']}'
                )
            return (
                '{"entities":[],"claims":[]}'
            )

        if (
            "revising a candidate answer"
            in system
        ):
            self.revision_calls += 1
            return (
                "The neural candidate conflicts with Cortex session evidence, "
                "so the relationship remains unresolved."
            )

        if format_schema is not None:
            if (
                user_text
                == "Alice works with Bob."
            ):
                return (
                    '{"language":"en","speech_act":"assertion",'
                    '"entities":['
                    '{"id":"e0","surface":"Alice","kind":"person","evidence":"Alice"},'
                    '{"id":"e1","surface":"Bob","kind":"person","evidence":"Bob"}'
                    '],"observations":['
                    '{"subject":"e0","predicate":"works_with","object":"e1",'
                    '"asserted":true,"evidence":"Alice works with Bob"}'
                    '],"query_operation":"none","query_subject":"",'
                    '"query_predicate":"","query_object":""}'
                )
            if (
                user_text
                == "Alice does not work with Bob."
            ):
                return (
                    '{"language":"en","speech_act":"assertion",'
                    '"entities":['
                    '{"id":"e0","surface":"Alice","kind":"person","evidence":"Alice"},'
                    '{"id":"e1","surface":"Bob","kind":"person","evidence":"Bob"}'
                    '],"observations":['
                    '{"subject":"e0","predicate":"works_with","object":"e1",'
                    '"asserted":false,"evidence":"Alice does not work with Bob"}'
                    '],"query_operation":"none","query_subject":"",'
                    '"query_predicate":"","query_object":""}'
                )
            if (
                user_text
                == "Does Alice work with Bob?"
            ):
                return (
                    '{"language":"en","speech_act":"question",'
                    '"entities":['
                    '{"id":"e0","surface":"Alice","kind":"person","evidence":"Alice"},'
                    '{"id":"e1","surface":"Bob","kind":"person","evidence":"Bob"}'
                    '],"observations":[],'
                    '"query_operation":"relation_lookup","query_subject":"e0",'
                    '"query_predicate":"works_with","query_object":"e1"}'
                )
            return (
                '{"language":"en","speech_act":"other",'
                '"entities":[],"observations":[],'
                '"query_operation":"none","query_subject":"",'
                '"query_predicate":"","query_object":""}'
            )

        if (
            "CORTEX_SESSION_CONTEXT="
            in system
        ):
            if (
                user_text
                == "Does Alice work with Bob?"
            ):
                return "Alice works with Bob."
            return "Neural answer."

        return "Neural answer."

    def embed(
        self,
        *,
        model: str,
        inputs,
    ):
        raise AssertionError(
            "orchestrator must not use neural embeddings"
        )


@pytest.fixture(autouse=True)
def fake_runtime(monkeypatch):
    monkeypatch.setattr(
        strict_module,
        "CortexRuntime",
        FakeRuntime,
    )


def make_gateway(
    *,
    persistence_dir=None,
    adapter_models=None,
    confirmations=3,
):
    client = OrchestratorFakeClient()
    gateway = CognitiveOrchestratorGateway(
        CognitiveOrchestratorConfig(
            chat_model="knowledge",
            parser_model="parser",
            feature_dim=64,
            persistence_dir=(
                None
                if persistence_dir
                is None
                else str(
                    persistence_dir
                )
            ),
            adapter_models=dict(
                adapter_models
                or {}
            ),
            adaptation_min_confirmations=(
                confirmations
            ),
            max_revision_cycles=1,
            allow_weight_updates=False,
        ),
        client=client,
    )
    return gateway, client


def test_cortex_context_reaches_neural_model_before_answer():
    gateway, client = make_gateway()

    gateway.chat(
        "session",
        "Alice works with Bob.",
    )
    result = gateway.chat(
        "session",
        "Does Alice work with Bob?",
    )

    neural_calls = [
        call
        for call in client.calls
        if not call[
            "structured"
        ]
        and "revising a candidate answer"
        not in (
            call["messages"][0][
                "content"
            ]
            if call["messages"]
            and call["messages"][0][
                "role"
            ]
            == "system"
            else ""
        )
    ]
    last = neural_calls[-1]
    system = last[
        "messages"
    ][0]["content"]

    assert (
        "CORTEX_SESSION_CONTEXT="
        in system
    )
    assert "works_with" in system
    assert (
        result.cortex_context[
            "orchestration"
        ]["route"]
        == "mixed_structural"
    )
    assert (
        result.cortex_context[
            "diagnosis"
        ]["status"]
        == "supported"
    )


def test_conflict_triggers_one_bounded_neural_revision():
    gateway, client = make_gateway()

    gateway.chat(
        "conflict",
        "Alice does not work with Bob.",
    )
    result = gateway.chat(
        "conflict",
        "Does Alice work with Bob?",
    )

    assert (
        client.revision_calls
        == 1
    )
    assert (
        result.cortex_context[
            "orchestration"
        ][
            "revision_cycles"
        ]
        == 1
    )
    assert (
        "remains unresolved"
        in result.cortex_context[
            "neural_answer"
        ]
    )


def test_single_observation_is_context_only_not_training():
    gateway, _client = make_gateway(
        confirmations=3
    )

    result = gateway.chat(
        "learn",
        "Alice works with Bob.",
    )
    adaptive = result.cortex_context[
        "adaptive_neural"
    ]

    assert (
        adaptive["learning"][
            "action"
        ]
        == "context_only"
    )
    assert (
        adaptive["learning"][
            "base_weights_mutated"
        ]
        is False
    )
    assert (
        adaptive[
            "weight_updates_enabled"
        ]
        is False
    )


def test_repeated_evidence_only_proposes_adapter_candidate():
    gateway, _client = make_gateway(
        confirmations=3
    )

    for _ in range(
        3
    ):
        result = gateway.chat(
            "learn",
            "Alice works with Bob.",
        )

    proposal = result.cortex_context[
        "adaptive_neural"
    ]["learning"]

    assert (
        proposal["action"]
        == "adapter_candidate"
    )
    assert (
        proposal[
            "evidence_confirmations"
        ]
        == 3
    )
    assert (
        proposal[
            "requires_approval"
        ]
        is True
    )
    assert (
        proposal[
            "base_weights_mutated"
        ]
        is False
    )


def test_pre_registered_adapter_selection_is_reversible():
    gateway, client = make_gateway(
        adapter_models={
            "mixed_structural": (
                "reasoning-adapter:latest"
            ),
        }
    )

    gateway.chat(
        "adapter",
        "Alice works with Bob.",
    )
    result = gateway.chat(
        "adapter",
        "Does Alice work with Bob?",
    )

    activation = result.cortex_context[
        "adaptive_neural"
    ]["activation"]
    assert (
        activation["model"]
        == "reasoning-adapter:latest"
    )
    assert (
        activation[
            "adapter_active"
        ]
        is True
    )

    assert any(
        call["model"]
        == "reasoning-adapter:latest"
        for call in client.calls
    )


def test_orchestrator_persistence_restores_evidence_ledger(
    tmp_path,
):
    gateway, _client = make_gateway(
        persistence_dir=tmp_path,
        confirmations=2,
    )
    gateway.chat(
        "persist",
        "Alice works with Bob.",
    )

    restored, _ = make_gateway(
        persistence_dir=tmp_path,
        confirmations=2,
    )
    result = restored.chat(
        "persist",
        "Alice works with Bob.",
    )

    proposal = result.cortex_context[
        "adaptive_neural"
    ]["learning"]
    assert (
        proposal["action"]
        == "adapter_candidate"
    )
    assert (
        proposal[
            "evidence_confirmations"
        ]
        == 2
    )


def test_server_factory_builds_cognitive_orchestrator(
    tmp_path,
):
    gateway = build_gateway_from_env(
        chat_model="knowledge",
        cognitive_orchestrator=True,
        parser_model="parser",
        persistence_dir=str(
            tmp_path
        ),
        feature_dim=64,
        adapter_models={
            "mixed_structural": (
                "adapter:latest"
            )
        },
        adaptation_min_confirmations=4,
        max_revision_cycles=1,
    )

    assert isinstance(
        gateway,
        CognitiveOrchestratorGateway,
    )
    assert (
        gateway.config.allow_weight_updates
        is False
    )
    assert (
        gateway.config.adapter_models[
            "mixed_structural"
        ]
        == "adapter:latest"
    )
    assert (
        gateway.config.adaptation_min_confirmations
        == 4
    )
