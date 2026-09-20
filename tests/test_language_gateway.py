from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import pytest

import cortex.language_gateway as gateway_module
from cortex import (
    CortexRuntime as NativeCortexRuntime,
    NATIVE_AVAILABLE,
)
from cortex.gateway_server import _bool_value, _last_user_text
from cortex.language_gateway import (
    CortexLanguageGateway,
    GatewayConfig,
    OllamaClient,
    diagnose_runtime,
    openai_chat_completion,
    parse_semantic_observation,
)


class FakeRead:
    def __init__(
        self,
        *,
        bindings,
        relation_changed=(),
        unresolved=False,
        budget_pressure=False,
        discovered=False,
    ):
        self.bindings = list(bindings)
        self.subgroup = [0]
        self.subgroup_margin = 0.2
        self.relation_changed = list(relation_changed)
        self.causal_changed = []
        self.articulation_vector = [0.0] * 12
        self.memory_stored = 1
        self.memory_unresolved = False
        self.prediction = 0.75
        self.memberships = [0.8, 0.2]
        self.current_id = 0
        self.nearest_id = 0
        self.nearest_dist = 0.1
        self.revision = False
        self.reactivated = False
        self.discovered = discovered
        self.unresolved = unresolved
        self.budget_pressure = budget_pressure
        self.stored = 2


class FakeRuntime:
    instances: list["FakeRuntime"] = []

    def __init__(self, *, feature_dim, max_entities, budget):
        self.feature_dim = feature_dim
        self.max_entities = max_entities
        self.budget = budget
        self.vector_ids: dict[tuple[float, ...], int] = {}
        self.steps: list[dict[str, Any]] = []
        FakeRuntime.instances.append(self)

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
            key = tuple(float(value) for value in vector)
            if key not in self.vector_ids:
                self.vector_ids[key] = len(self.vector_ids)
            bindings.append(self.vector_ids[key])
        self.steps.append(
            {
                "detections": detections,
                "relation_obs": relation_obs,
                "outcomes": outcomes,
                "outcome": outcome,
            }
        )
        return FakeRead(
            bindings=bindings,
            relation_changed=relation_obs,
            discovered=len(self.steps) == 1,
        )


class FakeClient:
    def __init__(self):
        self.extract_calls = 0
        self.answer_calls = 0
        self.answer_messages: list[list[Mapping[str, str]]] = []

    def chat(
        self,
        *,
        model: str,
        messages: Sequence[Mapping[str, str]],
        format_schema=None,
    ) -> str:
        if format_schema is not None:
            self.extract_calls += 1
            user = messages[-1]["content"]
            if "Alice knows Bob" in user:
                return (
                    '{"entities":['
                    '{"id":"e0","name":"Alice","kind":"person","description":"","evidence":"Alice"},'
                    '{"id":"e1","name":"Bob","kind":"person","description":"","evidence":"Bob"}'
                    '],"relations":['
                    '{"source":"e0","target":"e1","label":"knows","asserted":true,"evidence":"Alice knows Bob"}'
                    ']}'
                )
            if "Alice" in user:
                return (
                    '{"entities":['
                    '{"id":"e0","name":"Alice","kind":"person","description":"","evidence":"Alice"}'
                    '],"relations":[]}'
                )
            return '{"entities":[],"relations":[]}'

        self.answer_calls += 1
        self.answer_messages.append(list(messages))
        return f"answer-{self.answer_calls}"

    def embed(self, *, model: str, inputs: Sequence[str]):
        vectors = []
        for value in inputs:
            if "Alice" in value:
                vectors.append([1.0, 0.0, 0.0, 0.0])
            elif "Bob" in value:
                vectors.append([0.0, 1.0, 0.0, 0.0])
            else:
                vectors.append([0.0, 0.0, 1.0, 0.0])
        return vectors


@pytest.fixture(autouse=True)
def fake_native_runtime(monkeypatch):
    FakeRuntime.instances.clear()
    monkeypatch.setattr(
        gateway_module,
        "CortexRuntime",
        FakeRuntime,
    )


def make_gateway(
    *,
    diagnostics=False,
    persistence_dir=None,
    client=None,
):
    client = client or FakeClient()
    config = GatewayConfig(
        chat_model="test-chat",
        embedding_model="test-embed",
        diagnostics_default=diagnostics,
        history_messages=6,
        persistence_dir=(
            None
            if persistence_dir is None
            else str(persistence_dir)
        ),
    )
    return CortexLanguageGateway(
        config,
        client=client,
    ), client


def test_repeated_entity_is_persistent_across_chat_turns():
    gateway, client = make_gateway()

    first = gateway.chat("research", "Alice is here.")
    second = gateway.chat("research", "Alice is still here.")

    session = gateway.session("research")
    assert first.content == "answer-1"
    assert second.content == "answer-2"
    assert len(session.entities) == 1

    entity = next(iter(session.entities.values()))
    assert entity.names == {"Alice"}
    assert entity.mentions == 2
    assert client.extract_calls == 2
    assert len(FakeRuntime.instances) == 1
    assert len(FakeRuntime.instances[0].steps) == 2


def test_relation_extraction_becomes_cortex_relation_observation():
    gateway, _client = make_gateway()

    result = gateway.chat(
        "relations",
        "Alice knows Bob.",
    )

    session = gateway.session("relations")
    assert len(session.entities) == 2
    assert len(session.relations) == 1
    relation = next(iter(session.relations.values()))
    assert relation.label == "knows"
    assert relation.asserted
    assert len(
        result.cortex_context["relations"]
    ) == 1

    step = FakeRuntime.instances[0].steps[0]
    assert step["relation_obs"] == [(0, 1, 1)]


def test_diagnostics_switch_only_changes_introspection_not_observation():
    gateway, client = make_gateway(diagnostics=False)

    without = gateway.chat(
        "diag",
        "Alice is here.",
    )
    session = gateway.session("diag")
    entity_before = next(iter(session.entities.values()))
    mentions_before = entity_before.mentions
    assert without.diagnostics is None
    assert "diagnostics" not in without.cortex_context

    gateway.set_diagnostics("diag", True)
    with_diag = gateway.chat(
        "diag",
        "Alice is still here.",
    )

    entity_after = next(iter(session.entities.values()))
    assert entity_after.cortex_id == entity_before.cortex_id
    assert entity_after.mentions == mentions_before + 1
    assert with_diag.diagnostics is not None
    assert with_diag.diagnostics.status == "stable"
    assert "diagnostics" in with_diag.cortex_context

    # The diagnostics switch does not create an extra extraction or runtime step.
    assert client.extract_calls == 2
    assert len(FakeRuntime.instances[0].steps) == 2


def test_request_override_can_enable_diagnostics_without_changing_default():
    gateway, _client = make_gateway(diagnostics=False)

    result = gateway.chat(
        "override",
        "Alice is here.",
        diagnostics=True,
    )

    assert result.diagnostics is not None
    assert not gateway.session(
        "override"
    ).diagnostics_enabled


def test_diagnostics_report_is_grounded_in_runtime_flags():
    read = FakeRead(
        bindings=[0],
        unresolved=True,
        budget_pressure=True,
    )

    report = diagnose_runtime(
        read,
        memory_budget=24,
    )

    assert report.status == "uncertain"
    assert "uncertain" in report.summary.lower()
    assert report.metrics["budget_pressure"] is True
    assert report.metrics["unresolved"] is True
    assert any(
        "unresolved" in signal
        for signal in report.signals
    )


def test_openai_chat_completion_contains_cortex_extension():
    gateway, _client = make_gateway(diagnostics=True)
    result = gateway.chat(
        "openai",
        "Alice is here.",
    )

    payload = openai_chat_completion(result)

    assert payload["object"] == "chat.completion"
    assert payload["choices"][0]["message"][
        "content"
    ] == result.content
    assert payload["cortex"]["session_id"] == "openai"
    assert payload["cortex"]["diagnostics"] is not None


def test_parser_rejects_relation_to_unknown_local_entity():
    raw = (
        '{"entities":['
        '{"id":"e0","name":"Alice","kind":"person","description":"","evidence":"Alice"}'
        '],"relations":['
        '{"source":"e0","target":"missing","label":"knows","asserted":true,"evidence":"Alice"}'
        ']}'
    )
    with pytest.raises(ValueError):
        parse_semantic_observation(raw)


def test_persistent_session_restores_without_reextracting_or_reembedding(tmp_path):
    client = FakeClient()
    gateway, _ = make_gateway(
        persistence_dir=tmp_path,
        client=client,
    )

    gateway.chat(
        "persistent",
        "Alice knows Bob.",
    )
    gateway.set_diagnostics(
        "persistent",
        True,
    )

    first_session = gateway.session(
        "persistent"
    )
    first_entity_packets = [
        entity.packet()
        for entity in sorted(
            first_session.entities.values(),
            key=lambda entity: entity.cortex_id,
        )
    ]
    first_relation_packets = [
        relation.packet()
        for relation in first_session.relations.values()
    ]
    extraction_calls = client.extract_calls
    assert extraction_calls == 1
    assert len(FakeRuntime.instances) == 1

    restored_gateway, _ = make_gateway(
        persistence_dir=tmp_path,
        client=client,
    )
    restored = restored_gateway.session(
        "persistent"
    )

    assert client.extract_calls == extraction_calls
    assert len(FakeRuntime.instances) == 2
    assert restored.diagnostics_enabled
    assert len(restored.frames) == 1
    assert len(restored.history) == 2
    assert [
        entity.packet()
        for entity in sorted(
            restored.entities.values(),
            key=lambda entity: entity.cortex_id,
        )
    ] == first_entity_packets
    assert [
        relation.packet()
        for relation in restored.relations.values()
    ] == first_relation_packets


def test_native_runtime_snapshot_is_exact_after_event_replay(tmp_path, monkeypatch):
    if not NATIVE_AVAILABLE:
        pytest.skip(
            "native extension is not built in this test environment"
        )

    monkeypatch.setattr(
        gateway_module,
        "CortexRuntime",
        NativeCortexRuntime,
    )
    client = FakeClient()
    gateway, _ = make_gateway(
        persistence_dir=tmp_path,
        client=client,
    )

    gateway.chat(
        "native-replay",
        "Alice knows Bob.",
    )
    gateway.chat(
        "native-replay",
        "Alice is still here.",
    )
    original = gateway.session(
        "native-replay"
    )
    assert original.runtime is not None
    before = original.runtime.snapshot_json()

    restored_gateway, _ = make_gateway(
        persistence_dir=tmp_path,
        client=client,
    )
    restored = restored_gateway.session(
        "native-replay"
    )
    assert restored.runtime is not None
    after = restored.runtime.snapshot_json()

    assert after == before


def test_persistence_uses_saved_vectors_not_embedding_model_on_restore(tmp_path):
    client = FakeClient()
    gateway, _ = make_gateway(
        persistence_dir=tmp_path,
        client=client,
    )
    gateway.chat(
        "vectors",
        "Alice is here.",
    )
    extraction_calls = client.extract_calls

    class NoEmbedClient(FakeClient):
        def embed(self, *, model: str, inputs: Sequence[str]):
            raise AssertionError(
                "restore must not call the embedding model"
            )

    restored_client = NoEmbedClient()
    restored_client.extract_calls = extraction_calls
    restored_gateway, _ = make_gateway(
        persistence_dir=tmp_path,
        client=restored_client,
    )
    restored = restored_gateway.session(
        "vectors"
    )

    assert len(restored.frames) == 1
    assert len(restored.entities) == 1


def test_reset_deletes_persisted_session_by_default(tmp_path):
    gateway, _client = make_gateway(
        persistence_dir=tmp_path,
    )
    gateway.chat(
        "erase-me",
        "Alice is here.",
    )
    assert (
        "erase-me"
        in gateway.list_persisted_sessions()
    )

    gateway.reset(
        "erase-me"
    )

    assert (
        "erase-me"
        not in gateway.list_persisted_sessions()
    )


def test_save_load_roundtrip_can_replace_in_memory_session(tmp_path):
    gateway, _client = make_gateway(
        persistence_dir=tmp_path,
    )
    gateway.chat(
        "roundtrip",
        "Alice is here.",
    )
    original = gateway.session(
        "roundtrip"
    )
    original_mentions = next(
        iter(
            original.entities.values()
        )
    ).mentions

    path = gateway.save(
        "roundtrip"
    )
    assert path.endswith(".json")

    original.entities.clear()
    assert not original.entities

    loaded = gateway.load(
        "roundtrip"
    )
    assert (
        next(
            iter(
                loaded.entities.values()
            )
        ).mentions
        == original_mentions
    )


def test_parser_rejects_unanchored_extractor_hallucination():
    raw = (
        '{"entities":['
        '{"id":"e0","name":"Paris","kind":"city","description":"","evidence":"Paris"}'
        '],"relations":[]}'
    )
    with pytest.raises(ValueError):
        parse_semantic_observation(
            raw,
            source_text="Alice is here.",
        )


def test_ollama_client_normalizes_host_without_scheme():
    client = OllamaClient(
        "127.0.0.1:11434"
    )
    assert (
        client.base_url
        == "http://127.0.0.1:11434"
    )


def test_failed_answer_still_persists_accepted_observation(tmp_path):
    class FailingAnswerClient(FakeClient):
        def chat(
            self,
            *,
            model: str,
            messages: Sequence[Mapping[str, str]],
            format_schema=None,
        ) -> str:
            if format_schema is not None:
                return super().chat(
                    model=model,
                    messages=messages,
                    format_schema=format_schema,
                )
            raise RuntimeError(
                "answer generation failed"
            )

    client = FailingAnswerClient()
    gateway, _ = make_gateway(
        persistence_dir=tmp_path,
        client=client,
    )

    with pytest.raises(RuntimeError):
        gateway.chat(
            "failed-answer",
            "Alice is here.",
        )

    restored_gateway, _ = make_gateway(
        persistence_dir=tmp_path,
        client=client,
    )
    restored = restored_gateway.session(
        "failed-answer"
    )
    assert len(restored.frames) == 1
    assert len(restored.entities) == 1


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("true", True),
        ("on", True),
        ("1", True),
        ("false", False),
        ("off", False),
        ("0", False),
        (None, None),
    ],
)
def test_gateway_boolean_parser(raw, expected):
    assert _bool_value(raw) is expected


def test_last_user_text_uses_latest_user_message():
    assert _last_user_text(
        [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "answer"},
            {"role": "user", "content": "second"},
        ]
    ) == "second"
