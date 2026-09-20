from __future__ import annotations

from typing import Mapping, Sequence

import pytest

import cortex.strict_language_gateway as strict_module
from cortex.hybrid_core import (
    BOOTSTRAP_SCHEMA,
    HYBRID_TURN_SCHEMA,
    HybridCoreConfig,
    HybridCoreSession,
)
from cortex.neural_runtime import (
    NeuralCapabilities,
    NeuralResponse,
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


class FakeCortexRuntime:
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
                self.vector_ids[
                    key
                ]
            )
        return FakeRead(
            bindings
        )

    def snapshot_json(self):
        import json

        return json.dumps(
            {
                "feature_dim": (
                    self.feature_dim
                ),
                "vector_ids": [
                    [
                        list(key),
                        value,
                    ]
                    for key, value
                    in sorted(
                        self.vector_ids.items(),
                        key=lambda item: (
                            item[1]
                        ),
                    )
                ],
            },
            sort_keys=True,
        )


class FakeNeuralRuntime:
    def __init__(self):
        self.calls = []

    @property
    def backend_name(self):
        return "fake-runtime"

    @property
    def capabilities(self):
        return NeuralCapabilities(
            generate=True,
            structured_generate=True,
        )

    def _response(
        self,
        text,
        model,
        schema,
    ):
        return NeuralResponse(
            text=text,
            model=model,
            backend=self.backend_name,
            duration_s=0.01,
            structured=(
                schema is not None
            ),
            prompt_tokens=10,
            completion_tokens=5,
        )

    def generate(
        self,
        *,
        model: str,
        messages: Sequence[
            Mapping[str, str]
        ],
        schema=None,
        temperature=None,
    ):
        user = messages[-1][
            "content"
        ]
        system = (
            messages[0][
                "content"
            ]
            if messages
            and messages[0][
                "role"
            ]
            == "system"
            else ""
        )
        self.calls.append(
            {
                "model": model,
                "user": user,
                "system": system,
                "schema": schema,
            }
        )

        if schema == BOOTSTRAP_SCHEMA:
            return self._response(
                (
                    '{"claims":['
                    '{"subject":"France","subject_kind":"country",'
                    '"predicate":"capital","object":"Paris",'
                    '"object_kind":"city","asserted":true,'
                    '"confidence":0.95}'
                    ']}'
                ),
                model,
                schema,
            )

        if (
            schema is not None
            and schema != HYBRID_TURN_SCHEMA
        ):
            if (
                user
                == "Alice works with Bob."
            ):
                text = (
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
            elif (
                user
                == "Alice does not work with Bob."
            ):
                text = (
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
            elif (
                user
                == "Does Alice work with Bob?"
            ):
                text = (
                    '{"language":"en","speech_act":"question",'
                    '"entities":['
                    '{"id":"e0","surface":"Alice","kind":"person","evidence":"Alice"},'
                    '{"id":"e1","surface":"Bob","kind":"person","evidence":"Bob"}'
                    '],"observations":[],'
                    '"query_operation":"relation_lookup","query_subject":"e0",'
                    '"query_predicate":"works_with","query_object":"e1"}'
                )
            elif (
                user
                == "What is the capital of France?"
            ):
                text = (
                    '{"language":"en","speech_act":"question",'
                    '"entities":['
                    '{"id":"e0","surface":"France","kind":"country","evidence":"France"}'
                    '],"observations":[],'
                    '"query_operation":"relation_lookup","query_subject":"e0",'
                    '"query_predicate":"capital","query_object":""}'
                )
            else:
                text = (
                    '{"language":"en","speech_act":"other",'
                    '"entities":[],"observations":[],'
                    '"query_operation":"none","query_subject":"",'
                    '"query_predicate":"","query_object":""}'
                )
            return self._response(
                text,
                model,
                schema,
            )

        if schema == HYBRID_TURN_SCHEMA:
            if (
                "cortex_conflicts"
                in user
            ):
                return self._response(
                    (
                        '{"answer":"Session evidence says Alice does not work with Bob.",'
                        '"entities":['
                        '{"id":"e0","surface":"Alice","kind":"person","evidence":"Alice"},'
                        '{"id":"e1","surface":"Bob","kind":"person","evidence":"Bob"}'
                        '],"claims":['
                        '{"subject":"e0","predicate":"works_with","object":"e1",'
                        '"asserted":false,"evidence":"Alice does not work with Bob"}'
                        ']}'
                    ),
                    model,
                    schema,
                )
            if (
                user
                == "Does Alice work with Bob?"
            ):
                return self._response(
                    (
                        '{"answer":"Alice works with Bob.",'
                        '"entities":['
                        '{"id":"e0","surface":"Alice","kind":"person","evidence":"Alice"},'
                        '{"id":"e1","surface":"Bob","kind":"person","evidence":"Bob"}'
                        '],"claims":['
                        '{"subject":"e0","predicate":"works_with","object":"e1",'
                        '"asserted":true,"evidence":"Alice works with Bob"}'
                        ']}'
                    ),
                    model,
                    schema,
                )
            if (
                user
                == "What is the capital of France?"
            ):
                assert (
                    "neural_bootstrap"
                    in system
                )
                return self._response(
                    (
                        '{"answer":"Paris is the capital of France.",'
                        '"entities":['
                        '{"id":"e0","surface":"France","kind":"country","evidence":"France"},'
                        '{"id":"e1","surface":"Paris","kind":"city","evidence":"Paris"}'
                        '],"claims":['
                        '{"subject":"e0","predicate":"capital","object":"e1",'
                        '"asserted":true,"evidence":"capital of France"}'
                        ']}'
                    ),
                    model,
                    schema,
                )
            return self._response(
                (
                    '{"answer":"Understood.",'
                    '"entities":[],"claims":[]}'
                ),
                model,
                schema,
            )

        raise AssertionError(
            "unexpected neural call"
        )

    def embed(
        self,
        *,
        model,
        inputs,
    ):
        raise AssertionError(
            "Hybrid Core must not require neural embeddings"
        )


@pytest.fixture(autouse=True)
def fake_cortex_runtime(monkeypatch):
    monkeypatch.setattr(
        strict_module,
        "CortexRuntime",
        FakeCortexRuntime,
    )


def make_session():
    runtime = FakeNeuralRuntime()
    session = HybridCoreSession(
        "test",
        HybridCoreConfig(
            chat_model="knowledge",
            parser_model="parser",
            feature_dim=64,
            include_diagnosis=False,
            max_revision_cycles=1,
        ),
        runtime,
    )
    return session, runtime


def test_normal_hybrid_turn_uses_two_neural_calls():
    session, runtime = make_session()

    result = session.chat(
        "Does Alice work with Bob?"
    )

    assert (
        len(
            runtime.calls
        )
        == 2
    )
    assert (
        result.cortex_context[
            "hybrid_runtime"
        ][
            "neural_call_count"
        ]
        == 2
    )
    stages = [
        item["stage"]
        for item
        in result.cortex_context[
            "hybrid_runtime"
        ][
            "neural_calls"
        ]
    ]
    assert stages == [
        "prompt_parse",
        "answer_with_sidecar",
    ]


def test_stored_cortex_evidence_corroborates_fused_sidecar():
    session, _runtime = make_session()

    session.chat(
        "Alice works with Bob."
    )
    result = session.chat(
        "Does Alice work with Bob?"
    )

    assert (
        result.cortex_context[
            "diagnosis"
        ][
            "status"
        ]
        == "supported"
    )
    assert (
        result.cortex_context[
            "hybrid_runtime"
        ][
            "neural_call_count"
        ]
        == 2
    )


def test_conflict_adds_one_revision_not_posthoc_parse():
    session, _runtime = make_session()

    session.chat(
        "Alice does not work with Bob."
    )
    result = session.chat(
        "Does Alice work with Bob?"
    )

    runtime = result.cortex_context[
        "hybrid_runtime"
    ]
    assert (
        runtime[
            "revision_cycles"
        ]
        == 1
    )
    assert (
        runtime[
            "neural_call_count"
        ]
        == 3
    )
    assert [
        item["stage"]
        for item
        in runtime[
            "neural_calls"
        ]
    ] == [
        "prompt_parse",
        "answer_with_sidecar",
        "conflict_revision",
    ]
    assert (
        "does not work"
        in result.content
    )


def test_neural_bootstrap_imports_provisional_cortex_memory():
    session, _runtime = make_session()

    imported = session.bootstrap_topics(
        [
            "basic geography",
        ]
    )

    assert len(
        imported
    ) == 1
    assert (
        imported[0].provenance
        == "neural_bootstrap"
    )
    assert (
        imported[0].confidence
        == pytest.approx(
            0.95
        )
    )
    assert len(
        session.relations
    ) == 1

    result = session.chat(
        "What is the capital of France?"
    )

    assert (
        result.cortex_context[
            "bootstrap"
        ][
            "claim_count"
        ]
        == 1
    )
    assert (
        result.cortex_context[
            "cortex_answer_ir"
        ][
            "status"
        ]
        == "resolved"
    )
    assert (
        result.cortex_context[
            "diagnosis"
        ][
            "status"
        ]
        == "supported"
    )


def test_low_confidence_bootstrap_claim_is_not_imported():
    session, runtime = make_session()

    original = runtime.generate

    def low_confidence(**kwargs):
        if kwargs.get(
            "schema"
        ) == BOOTSTRAP_SCHEMA:
            return NeuralResponse(
                text=(
                    '{"claims":['
                    '{"subject":"X","subject_kind":"entity",'
                    '"predicate":"related_to","object":"Y",'
                    '"object_kind":"entity","asserted":true,'
                    '"confidence":0.2}'
                    ']}'
                ),
                model=kwargs[
                    "model"
                ],
                backend=(
                    runtime.backend_name
                ),
                duration_s=0.01,
                structured=True,
            )
        return original(
            **kwargs
        )

    runtime.generate = (
        low_confidence
    )

    imported = session.bootstrap_topics(
        [
            "uncertain",
        ]
    )

    assert imported == tuple()
    assert not session.relations
