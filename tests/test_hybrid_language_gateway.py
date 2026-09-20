from __future__ import annotations

from typing import Any, Mapping, Sequence

import pytest

import cortex.strict_language_gateway as strict_module
from cortex.gateway_server import build_gateway_from_env
from cortex.language_gateway import openai_chat_completion
from cortex.hybrid_language_gateway import (
    HybridCortexLanguageGateway,
    HybridGatewayConfig,
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


class HybridFakeClient:
    def __init__(self):
        self.calls = []
        self.fail_prompt_parse = False
        self.fail_candidate_parse = False

    def chat(
        self,
        *,
        model: str,
        messages: Sequence[
            Mapping[str, str]
        ],
        format_schema=None,
    ) -> str:
        self.calls.append(
            {
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
        )
        user_text = messages[-1][
            "content"
        ]

        if model == "knowledge":
            if (
                user_text
                == "What is the capital of Mongolia?"
            ):
                return (
                    "The capital of Mongolia is Ulaanbaatar."
                )
            if (
                user_text
                == "Alice works with Bob."
            ):
                return "Understood."
            if (
                user_text
                == "Alice does not work with Bob."
            ):
                return "Understood."
            if (
                user_text
                == "Does Alice work with Bob?"
            ):
                return (
                    "Alice works with Bob."
                )
            if (
                user_text
                == "Tell me a joke."
            ):
                return (
                    "Why did the byte cross the bus? "
                    "To get to the other side."
                )
            if (
                user_text
                == "This parser will fail."
            ):
                return (
                    "The neural answer still works."
                )
            return "A normal neural answer."

        if model == "parser":
            system = messages[0][
                "content"
            ]

            if (
                "candidate-answer analyzer"
                in system
            ):
                if self.fail_candidate_parse:
                    return "not json"
                if (
                    "capital of Mongolia is Ulaanbaatar"
                    in user_text
                ):
                    return (
                        '{"entities":['
                        '{"id":"e0","surface":"Mongolia","kind":"country","evidence":"Mongolia"},'
                        '{"id":"e1","surface":"Ulaanbaatar","kind":"city","evidence":"Ulaanbaatar"}'
                        '],"claims":['
                        '{"subject":"e0","predicate":"capital","object":"e1",'
                        '"asserted":true,"evidence":"capital of Mongolia is Ulaanbaatar"}'
                        ']}'
                    )
                if (
                    user_text
                    == "Alice works with Bob."
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

            if self.fail_prompt_parse:
                return "not json"

            if (
                user_text
                == "What is the capital of Mongolia?"
            ):
                return (
                    '{"language":"en","speech_act":"question",'
                    '"entities":['
                    '{"id":"e0","surface":"Mongolia","kind":"country","evidence":"Mongolia"}'
                    '],"observations":[],'
                    '"query_operation":"relation_lookup","query_subject":"e0",'
                    '"query_predicate":"capital","query_object":""}'
                )
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

        raise AssertionError(
            f"unexpected model {model}"
        )

    def embed(
        self,
        *,
        model: str,
        inputs: Sequence[str],
    ):
        raise AssertionError(
            "hybrid v0.3 must not use neural embeddings"
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
):
    client = HybridFakeClient()
    gateway = HybridCortexLanguageGateway(
        HybridGatewayConfig(
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
        ),
        client=client,
    )
    return gateway, client


def test_neural_pretrained_knowledge_answers_without_cortex_fact():
    gateway, _client = make_gateway()

    result = gateway.chat(
        "hybrid",
        "What is the capital of Mongolia?",
    )

    assert (
        result.content.startswith(
            "The capital of Mongolia is Ulaanbaatar."
        )
    )
    diagnosis = (
        result.cortex_context[
            "diagnosis"
        ]
    )
    assert (
        diagnosis["status"]
        == "neural_prior"
    )
    assert (
        diagnosis["reasoning_mode"]
        == "pretrained_knowledge"
    )
    assert (
        diagnosis["neural_claims"][0][
            "status"
        ]
        == "neural_prior"
    )

    session = gateway.session(
        "hybrid"
    )
    names = {
        name
        for entity
        in session.entities.values()
        for name in entity.names
    }
    assert "Mongolia" in names
    assert "Ulaanbaatar" not in names
    assert not session.relations


def test_explicit_user_observation_enters_cortex_then_corroborates_neural_answer():
    gateway, _client = make_gateway()

    first = gateway.chat(
        "hybrid",
        "Alice works with Bob.",
    )
    assert (
        first.cortex_context[
            "diagnosis"
        ][
            "accepted_user_observations"
        ]
        == 1
    )

    result = gateway.chat(
        "hybrid",
        "Does Alice work with Bob?",
    )
    diagnosis = (
        result.cortex_context[
            "diagnosis"
        ]
    )

    assert (
        diagnosis["status"]
        == "supported"
    )
    assert (
        diagnosis["reasoning_mode"]
        == "neural_plus_cortex"
    )
    assert (
        diagnosis["neural_claims"][0][
            "status"
        ]
        == "corroborated"
    )
    assert (
        "Cortex independently found compatible support"
        in result.content
    )


def test_cortex_reports_conflict_without_overwriting_memory():
    gateway, _client = make_gateway()

    gateway.chat(
        "hybrid",
        "Alice does not work with Bob.",
    )
    result = gateway.chat(
        "hybrid",
        "Does Alice work with Bob?",
    )

    diagnosis = (
        result.cortex_context[
            "diagnosis"
        ]
    )
    assert (
        diagnosis["status"]
        == "conflict"
    )
    assert (
        diagnosis["neural_claims"][0][
            "status"
        ]
        == "conflict"
    )

    session = gateway.session(
        "hybrid"
    )
    stored = list(
        session.relations.values()
    )
    assert len(stored) == 1
    assert stored[0].asserted is False


def test_prompt_parser_failure_never_blocks_neural_answer():
    gateway, client = make_gateway()
    client.fail_prompt_parse = True

    result = gateway.chat(
        "hybrid",
        "This parser will fail.",
    )

    assert (
        result.content.startswith(
            "The neural answer still works."
        )
    )
    diagnosis = (
        result.cortex_context[
            "diagnosis"
        ]
    )
    assert (
        diagnosis["analysis_errors"]
    )


def test_candidate_parser_failure_never_blocks_neural_answer():
    gateway, client = make_gateway()
    client.fail_candidate_parse = True

    result = gateway.chat(
        "hybrid",
        "What is the capital of Mongolia?",
    )

    assert (
        result.content.startswith(
            "The capital of Mongolia is Ulaanbaatar."
        )
    )
    assert (
        result.cortex_context[
            "diagnosis"
        ]["status"]
        == "analysis_partial"
    )


def test_neural_history_never_contains_cortex_diagnosis():
    gateway, client = make_gateway()

    gateway.chat(
        "hybrid",
        "What is the capital of Mongolia?",
    )
    gateway.chat(
        "hybrid",
        "Tell me a joke.",
    )

    knowledge_calls = [
        call
        for call in client.calls
        if call["model"]
        == "knowledge"
    ]
    second = knowledge_calls[
        -1
    ][
        "messages"
    ]
    assistant_history = [
        message["content"]
        for message in second
        if message["role"]
        == "assistant"
    ]
    assert assistant_history
    assert all(
        "Cortex diagnosis:"
        not in content
        for content
        in assistant_history
    )


def test_hybrid_persistence_replays_cortex_state(tmp_path):
    gateway, _client = make_gateway(
        persistence_dir=tmp_path
    )
    gateway.chat(
        "persist",
        "Alice works with Bob.",
    )

    restored_gateway, _ = (
        make_gateway(
            persistence_dir=tmp_path
        )
    )
    session = (
        restored_gateway.session(
            "persist"
        )
    )

    assert len(
        session.relations
    ) == 1


def test_server_factory_builds_hybrid_mode(tmp_path):
    gateway = build_gateway_from_env(
        chat_model="knowledge",
        strict_neural_boundary=False,
        hybrid_cognition=True,
        parser_model="parser",
        persistence_dir=str(tmp_path),
        feature_dim=64,
    )

    assert isinstance(
        gateway,
        HybridCortexLanguageGateway,
    )
    assert (
        gateway.config.chat_model
        == "knowledge"
    )
    assert (
        gateway.config.parser_model
        == "parser"
    )
    assert (
        gateway.config.embedding_model
        == "sha256-symbolic-v1"
    )


def test_openai_response_exposes_structured_hybrid_diagnosis():
    gateway, _client = make_gateway()

    result = gateway.chat(
        "hybrid",
        "What is the capital of Mongolia?",
    )
    payload = openai_chat_completion(
        result
    )

    diagnosis = payload[
        "cortex"
    ][
        "diagnosis"
    ]
    assert diagnosis is not None
    assert (
        diagnosis["status"]
        == "neural_prior"
    )
    assert (
        diagnosis["reasoning_mode"]
        == "pretrained_knowledge"
    )
