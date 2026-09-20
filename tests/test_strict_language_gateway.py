from __future__ import annotations

from typing import Any, Mapping, Sequence

import pytest

import cortex.strict_language_gateway as strict_module
from cortex.gateway_server import build_gateway_from_env
from cortex.strict_language_gateway import (
    DeterministicSymbolicEncoder,
    PromptEntity,
    parse_prompt_ir_payload,
    StrictCortexLanguageGateway,
    StrictGatewayConfig,
)


class FakeRead:
    def __init__(
        self,
        bindings,
    ):
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
        self.vector_ids: dict[
            tuple[float, ...],
            int,
        ] = {}
        self.steps: list[dict[str, Any]] = []

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
        self.steps.append(
            {
                "detections": detections,
                "relations": relation_obs,
            }
        )
        return FakeRead(
            bindings
        )

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
                "steps": len(self.steps),
            },
            sort_keys=True,
        )


class StrictFakeClient:
    def __init__(self):
        self.parser_messages = []
        self.renderer_messages = []
        self.embed_calls = 0

    def chat(
        self,
        *,
        model: str,
        messages: Sequence[
            Mapping[str, str]
        ],
        format_schema=None,
    ) -> str:
        if model == "parser":
            self.parser_messages.append(
                list(messages)
            )
            text = messages[-1][
                "content"
            ]
            if (
                text
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
                text
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
            if (
                text
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
            if text == "How are you doing internally?":
                return (
                    '{"language":"en","speech_act":"question",'
                    '"entities":[],"observations":[],'
                    '"query_operation":"introspection","query_subject":"",'
                    '"query_predicate":"","query_object":""}'
                )
            if text == "Hello Cortex. What do you know?":
                # Deliberately model a plausible parser mistake:
                # self-recall with an undeclared surface reference.
                return (
                    '{"language":"en","speech_act":"question",'
                    '"entities":[],"observations":[],'
                    '"query_operation":"entity_recall","query_subject":"Cortex",'
                    '"query_predicate":"knowledge","query_object":""}'
                )
            raise AssertionError(
                f"unexpected parser input: {text}"
            )

        if model == "renderer":
            self.renderer_messages.append(
                list(messages)
            )
            raw = messages[-1][
                "content"
            ]
            # Adversarial simulation: if semantic query information leaks
            # through, the renderer uses pretrained knowledge.
            if (
                "Mongolia" in raw
                or "capital" in raw
            ):
                return (
                    '{"text":"The capital is Ulaanbaatar.",'
                    '"support_ids":["unknown:0"]}'
                )
            if "Alice works with Bob." in raw:
                return (
                    '{"text":"Alice works with Bob.",'
                    '"support_ids":["claim:0"]}'
                )
            if "enough information" in raw:
                return (
                    '{"text":"I do not currently have enough information.",'
                    '"support_ids":["unknown:0"]}'
                )
            if "incorporated" in raw:
                return (
                    '{"text":"Noted.",'
                    '"support_ids":["ack:0"]}'
                )
            if "structurally stable" in raw:
                return (
                    '{"text":"My current Cortex state is structurally stable.",'
                    '"support_ids":["diagnostic:0"]}'
                )
            return (
                '{"text":"Rendered.",'
                '"support_ids":["ack:0"]}'
            )

        raise AssertionError(
            f"unexpected model: {model}"
        )

    def embed(
        self,
        *,
        model: str,
        inputs: Sequence[str],
    ):
        self.embed_calls += 1
        raise AssertionError(
            "strict neural boundary must never call embeddings"
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
    explain_responses=False,
):
    client = StrictFakeClient()
    gateway = (
        StrictCortexLanguageGateway(
            StrictGatewayConfig(
                parser_model="parser",
                renderer_model="renderer",
                feature_dim=64,
                persistence_dir=(
                    None
                    if persistence_dir
                    is None
                    else str(
                        persistence_dir
                    )
                ),
                explain_responses=(
                    explain_responses
                ),
            ),
            client=client,
        )
    )
    return gateway, client


def test_symbolic_encoder_is_deterministic_and_name_sensitive():
    encoder = (
        DeterministicSymbolicEncoder(
            64
        )
    )
    alice = PromptEntity(
        local_id="e0",
        surface="Alice",
        kind="person",
        evidence="Alice",
    )
    alice_again = PromptEntity(
        local_id="x7",
        surface="Alice",
        kind="person",
        evidence="Alice",
    )
    bob = PromptEntity(
        local_id="e1",
        surface="Bob",
        kind="person",
        evidence="Bob",
    )

    assert (
        encoder.encode_entity(
            alice
        )
        == encoder.encode_entity(
            alice_again
        )
    )
    assert (
        encoder.encode_entity(
            alice
        )
        != encoder.encode_entity(
            bob
        )
    )


def test_strict_gateway_never_calls_embedding_model():
    gateway, client = make_gateway()

    gateway.chat(
        "strict",
        "Alice works with Bob.",
    )
    gateway.chat(
        "strict",
        "Does Alice work with Bob?",
    )

    assert client.embed_calls == 0


def test_parser_is_stateless_and_sees_only_current_user_message():
    gateway, client = make_gateway()

    gateway.chat(
        "strict",
        "Alice works with Bob.",
    )
    gateway.chat(
        "strict",
        "Does Alice work with Bob?",
    )

    assert len(
        client.parser_messages
    ) == 2
    for messages in (
        client.parser_messages
    ):
        assert len(messages) == 2
        assert (
            messages[0]["role"]
            == "system"
        )
        assert (
            messages[1]["role"]
            == "user"
        )


def test_cortex_answer_ir_resolves_learned_relation():
    gateway, _client = make_gateway()

    gateway.chat(
        "strict",
        "Alice works with Bob.",
    )
    result = gateway.chat(
        "strict",
        "Does Alice work with Bob?",
    )

    answer_ir = (
        result.cortex_context[
            "answer_ir"
        ]
    )
    assert (
        answer_ir["status"]
        == "resolved"
    )
    assert len(
        answer_ir["claims"]
    ) == 1
    claim = (
        answer_ir["claims"][0]
    )
    assert claim[
        "subject"
    ] == "Alice"
    assert claim[
        "predicate"
    ] == "works_with"
    assert claim[
        "object"
    ] == "Bob"
    assert (
        result.content
        == "Alice works with Bob."
    )


def test_unknown_query_is_redacted_before_renderer_knowledge_leak():
    gateway, client = make_gateway()

    result = gateway.chat(
        "strict",
        "What is the capital of Mongolia?",
    )

    answer_ir = (
        result.cortex_context[
            "answer_ir"
        ]
    )
    assert (
        answer_ir["status"]
        == "unknown"
    )
    renderer_raw = (
        client.renderer_messages[-1][
            -1
        ]["content"]
    )
    assert "Mongolia" not in renderer_raw
    assert "capital" not in renderer_raw
    assert "Ulaanbaatar" not in (
        result.content
    )
    assert "enough information" in (
        result.content
    )


def test_renderer_never_receives_original_prompt_or_history():
    gateway, client = make_gateway()

    gateway.chat(
        "strict",
        "Alice works with Bob.",
    )
    gateway.chat(
        "strict",
        "Does Alice work with Bob?",
    )

    messages = (
        client.renderer_messages[-1]
    )
    assert len(messages) == 2
    assert messages[0][
        "role"
    ] == "system"
    assert messages[1][
        "role"
    ] == "user"
    raw = messages[1][
        "content"
    ]
    assert (
        "Does Alice work with Bob?"
        not in raw
    )


def test_strict_persistence_replays_without_parser_renderer_or_embeddings(
    tmp_path,
):
    gateway, client = make_gateway(
        persistence_dir=tmp_path
    )
    gateway.chat(
        "persist",
        "Alice works with Bob.",
    )
    first = gateway.session(
        "persist"
    )
    before_snapshot = (
        first.runtime.snapshot_json()
    )
    parser_calls = len(
        client.parser_messages
    )
    renderer_calls = len(
        client.renderer_messages
    )

    restored_gateway, _ = (
        make_gateway(
            persistence_dir=tmp_path
        )
    )
    restored = (
        restored_gateway.session(
            "persist"
        )
    )

    assert (
        restored.runtime.snapshot_json()
        == before_snapshot
    )
    assert len(
        client.parser_messages
    ) == parser_calls
    assert len(
        client.renderer_messages
    ) == renderer_calls


def test_diagnostics_are_cortex_owned_in_strict_mode():
    gateway, _client = make_gateway()

    gateway.chat(
        "diag",
        "Alice works with Bob.",
    )
    result = gateway.chat(
        "diag",
        "How are you doing internally?",
        diagnostics=True,
    )

    assert (
        result.diagnostics
        is not None
    )
    assert (
        result.cortex_context[
            "answer_ir"
        ]["status"]
        == "diagnostic"
    )


def test_server_factory_builds_strict_gateway_mode(tmp_path):
    gateway = build_gateway_from_env(
        chat_model="base-model",
        strict_neural_boundary=True,
        parser_model="parser-model",
        renderer_model="renderer-model",
        persistence_dir=str(tmp_path),
        feature_dim=64,
    )

    assert isinstance(
        gateway,
        StrictCortexLanguageGateway,
    )
    assert (
        gateway.config.parser_model
        == "parser-model"
    )
    assert (
        gateway.config.renderer_model
        == "renderer-model"
    )
    assert (
        gateway.config.embedding_model
        == "sha256-symbolic-v1"
    )
    assert gateway.config.feature_dim == 64


def test_reasoning_report_is_always_available_in_context():
    gateway, _client = make_gateway()

    gateway.chat(
        "report",
        "Alice works with Bob.",
    )
    result = gateway.chat(
        "report",
        "Does Alice work with Bob?",
    )

    report = result.cortex_context[
        "reasoning_report"
    ]
    assert (
        report["query"]["operation"]
        == "relation_lookup"
    )
    assert (
        report["query"]["subject"]
        == "Alice"
    )
    assert (
        report["query"]["predicate"]
        == "works_with"
    )
    assert (
        report["query"]["object"]
        == "Bob"
    )
    assert (
        report["result"]["status"]
        == "resolved"
    )
    assert len(
        report["result"]["claims"]
    ) == 1
    assert {
        "parser",
        "cortex",
        "renderer",
    }.issubset(
        report["timings"]
    )


def test_explain_responses_appends_deterministic_cortex_report():
    gateway, _client = make_gateway(
        explain_responses=True,
    )

    result = gateway.chat(
        "explain",
        "Alice works with Bob.",
    )

    assert "Cortex report" in (
        result.content
    )
    assert (
        "Gathered entities: Alice (person), Bob (person)"
        in result.content
    )
    assert (
        "Accepted observations:"
        in result.content
    )
    assert (
        "Alice works_with Bob"
        in result.content
    )
    assert (
        "Cortex bindings:"
        in result.content
    )
    assert (
        "Decision: acknowledged"
        in result.content
    )
    assert (
        "Processing:"
        in result.content
    )


def test_explain_unknown_query_reports_missing_support_without_renderer_leak():
    gateway, client = make_gateway(
        explain_responses=True,
    )

    result = gateway.chat(
        "unknown-explain",
        "What is the capital of Mongolia?",
    )

    renderer_raw = (
        client.renderer_messages[-1][-1][
            "content"
        ]
    )
    assert "Mongolia" not in renderer_raw
    assert "capital" not in renderer_raw

    assert "Cortex report" in (
        result.content
    )
    assert (
        "Query: relation_lookup [Mongolia capital]"
        in result.content
    )
    assert (
        "Decision: unknown"
        in result.content
    )
    assert (
        "Cortex found no sufficient stored support"
        in result.content
    )
    assert "Ulaanbaatar" not in (
        result.content
    )


def test_server_factory_enables_explain_responses(tmp_path):
    gateway = build_gateway_from_env(
        chat_model="base-model",
        strict_neural_boundary=True,
        parser_model="parser-model",
        renderer_model="renderer-model",
        persistence_dir=str(tmp_path),
        feature_dim=64,
        explain_responses=True,
    )

    assert isinstance(
        gateway,
        StrictCortexLanguageGateway,
    )
    assert (
        gateway.config.explain_responses
        is True
    )


def test_promptir_query_surface_reference_is_canonicalized_to_local_id():
    prompt_ir = parse_prompt_ir_payload(
        {
            "language": "en",
            "speech_act": "question",
            "entities": [
                {
                    "id": "e0",
                    "surface": "Alice",
                    "kind": "person",
                    "evidence": "Alice",
                },
                {
                    "id": "e1",
                    "surface": "Bob",
                    "kind": "person",
                    "evidence": "Bob",
                },
            ],
            "observations": [],
            "query_operation": "relation_lookup",
            "query_subject": "Alice",
            "query_predicate": "works_with",
            "query_object": "Bob",
        },
        source_text=(
            "Does Alice work with Bob?"
        ),
    )

    assert (
        prompt_ir.query_subject
        == "e0"
    )
    assert (
        prompt_ir.query_object
        == "e1"
    )


def test_promptir_observation_surface_references_are_canonicalized():
    prompt_ir = parse_prompt_ir_payload(
        {
            "language": "en",
            "speech_act": "assertion",
            "entities": [
                {
                    "id": "e0",
                    "surface": "Alice",
                    "kind": "person",
                    "evidence": "Alice",
                },
                {
                    "id": "e1",
                    "surface": "Bob",
                    "kind": "person",
                    "evidence": "Bob",
                },
            ],
            "observations": [
                {
                    "subject": "Alice",
                    "predicate": "works_with",
                    "object": "Bob",
                    "asserted": True,
                    "evidence": "Alice works with Bob",
                }
            ],
            "query_operation": "none",
            "query_subject": "",
            "query_predicate": "",
            "query_object": "",
        },
        source_text=(
            "Alice works with Bob."
        ),
    )

    assert (
        prompt_ir.observations[0].subject
        == "e0"
    )
    assert (
        prompt_ir.observations[0].object
        == "e1"
    )


def test_promptir_ambiguous_surface_reference_still_fails_closed():
    with pytest.raises(
        ValueError,
        match="ambiguous",
    ):
        parse_prompt_ir_payload(
            {
                "language": "en",
                "speech_act": "question",
                "entities": [
                    {
                        "id": "e0",
                        "surface": "Alex",
                        "kind": "person",
                        "evidence": "Alex",
                    },
                    {
                        "id": "e1",
                        "surface": "Alex",
                        "kind": "organization",
                        "evidence": "Alex",
                    },
                ],
                "observations": [],
                "query_operation": "entity_recall",
                "query_subject": "Alex",
                "query_predicate": "",
                "query_object": "",
            },
            source_text="Tell me about Alex.",
        )


def test_promptir_unknown_surface_reference_still_fails_closed():
    with pytest.raises(
        ValueError,
        match="unknown",
    ):
        parse_prompt_ir_payload(
            {
                "language": "en",
                "speech_act": "question",
                "entities": [
                    {
                        "id": "e0",
                        "surface": "Alice",
                        "kind": "person",
                        "evidence": "Alice",
                    }
                ],
                "observations": [],
                "query_operation": "entity_recall",
                "query_subject": "Bob",
                "query_predicate": "",
                "query_object": "",
            },
            source_text="Tell me about Alice.",
        )


def test_introspection_ignores_undeclared_cortex_self_reference():
    prompt_ir = parse_prompt_ir_payload(
        {
            "language": "en",
            "speech_act": "question",
            "entities": [],
            "observations": [],
            "query_operation": "introspection",
            "query_subject": "Cortex",
            "query_predicate": "knowledge",
            "query_object": "",
        },
        source_text=(
            "Hello Cortex. What do you know?"
        ),
    )

    assert (
        prompt_ir.query_operation
        == "introspection"
    )
    assert (
        prompt_ir.query_subject
        == ""
    )
    assert (
        prompt_ir.query_object
        == ""
    )


def test_self_entity_recall_is_normalized_to_introspection():
    prompt_ir = parse_prompt_ir_payload(
        {
            "language": "en",
            "speech_act": "question",
            "entities": [],
            "observations": [],
            "query_operation": "entity_recall",
            "query_subject": "Cortex",
            "query_predicate": "knowledge",
            "query_object": "",
        },
        source_text=(
            "Hello Cortex. What do you know?"
        ),
    )

    assert (
        prompt_ir.query_operation
        == "introspection"
    )
    assert (
        prompt_ir.query_subject
        == ""
    )


def test_what_do_you_know_returns_cortex_stored_relations():
    gateway, _client = make_gateway(
        explain_responses=True,
    )

    gateway.chat(
        "knowledge",
        "Alice works with Bob.",
    )
    result = gateway.chat(
        "knowledge",
        "Hello Cortex. What do you know?",
    )

    answer_ir = (
        result.cortex_context[
            "answer_ir"
        ]
    )
    assert (
        answer_ir["status"]
        == "resolved"
    )
    assert len(
        answer_ir["claims"]
    ) == 1
    claim = (
        answer_ir["claims"][0]
    )
    assert (
        claim["subject"]
        == "Alice"
    )
    assert (
        claim["predicate"]
        == "works_with"
    )
    assert (
        claim["object"]
        == "Bob"
    )
    assert (
        "Alice works with Bob."
        in result.content
    )
    assert (
        "Query: introspection"
        in result.content
    )


def test_empty_knowledge_introspection_returns_unknown_not_validation_error():
    gateway, _client = make_gateway(
        explain_responses=True,
    )

    result = gateway.chat(
        "empty-knowledge",
        "Hello Cortex. What do you know?",
    )

    assert (
        result.cortex_context[
            "answer_ir"
        ]["status"]
        == "unknown"
    )
    assert (
        "Decision: unknown"
        in result.content
    )
