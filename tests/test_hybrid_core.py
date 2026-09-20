from __future__ import annotations

import json
from typing import Mapping, Sequence

import pytest

import cortex.strict_language_gateway as strict_module
from cortex.hybrid_core import (
    BOOTSTRAP_SCHEMA,
    HYBRID_TURN_SCHEMA,
    HybridCoreConfig,
    HybridCoreSession,
)
from cortex.hybrid_core_gateway import (
    HybridCoreGateway,
)
from cortex.hybrid_exchange import (
    named_orchestration_policy,
)
from cortex.gateway_server import (
    build_gateway_from_env,
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
                == "France has capital Paris."
            ):
                text = (
                    '{"language":"en","speech_act":"assertion",'
                    '"entities":['
                    '{"id":"e0","surface":"France","kind":"country","evidence":"France"},'
                    '{"id":"e1","surface":"Paris","kind":"city","evidence":"Paris"}'
                    '],"observations":['
                    '{"subject":"e0","predicate":"capital","object":"e1",'
                    '"asserted":true,"evidence":"France has capital Paris"}'
                    '],"query_operation":"none","query_subject":"",'
                    '"query_predicate":"","query_object":""}'
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
        == "bootstrap_aligned"
    )
    statuses = [
        item["status"]
        for item
        in result.cortex_context[
            "diagnosis"
        ][
            "neural_claims"
        ]
    ]
    assert statuses == [
        "bootstrap_aligned"
    ]


def test_user_observation_promotes_bootstrap_relation_to_independent_evidence():
    session, _runtime = make_session()

    session.bootstrap_topics(
        [
            "basic geography",
        ]
    )
    session.chat(
        "France has capital Paris."
    )
    result = session.chat(
        "What is the capital of France?"
    )

    assert (
        result.cortex_context[
            "diagnosis"
        ][
            "status"
        ]
        == "supported"
    )
    assert [
        item["status"]
        for item
        in result.cortex_context[
            "diagnosis"
        ][
            "neural_claims"
        ]
    ] == [
        "corroborated"
    ]
    assert (
        "user_observation"
        in session.relation_provenance.values()
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



def test_hybrid_core_gateway_persists_bootstrap_memory(
    tmp_path,
):
    runtime = FakeNeuralRuntime()
    config = HybridCoreConfig(
        chat_model="knowledge",
        parser_model="parser",
        feature_dim=64,
        include_diagnosis=False,
        persistence_dir=str(
            tmp_path
        ),
    )
    gateway = HybridCoreGateway(
        config,
        runtime=runtime,
    )

    imported = gateway.bootstrap(
        "persist",
        [
            "basic geography",
        ],
    )
    assert len(
        imported
    ) == 1

    restored = HybridCoreGateway(
        config,
        runtime=FakeNeuralRuntime(),
    )
    session = restored.session(
        "persist"
    )

    assert len(
        session.bootstrap_records
    ) == 1
    assert len(
        session.relations
    ) == 1

    result = restored.chat(
        "persist",
        "What is the capital of France?",
    )
    assert (
        result.cortex_context[
            "diagnosis"
        ][
            "status"
        ]
        == "bootstrap_aligned"
    )
    assert (
        "neural_bootstrap"
        in session.relation_provenance.values()
    )


def test_server_factory_builds_hybrid_runtime_mode(
    tmp_path,
):
    gateway = build_gateway_from_env(
        chat_model="qwen-test",
        parser_model="parser-test",
        hybrid_runtime=True,
        persistence_dir=str(
            tmp_path
        ),
        feature_dim=64,
        max_revision_cycles=2,
    )

    assert isinstance(
        gateway,
        HybridCoreGateway,
    )
    assert (
        gateway.config.chat_model
        == "qwen-test"
    )
    assert (
        gateway.config.parser_model
        == "parser-test"
    )
    assert (
        gateway.config.max_revision_cycles
        == 2
    )
    assert (
        gateway.runtime.backend_name
        == "ollama"
    )



def test_neural_direct_policy_uses_one_neural_call():
    runtime = FakeNeuralRuntime()
    session = HybridCoreSession(
        "direct",
        HybridCoreConfig(
            chat_model="knowledge",
            parser_model="parser",
            feature_dim=64,
            include_diagnosis=False,
        ),
        runtime,
        orchestration_policy=(
            named_orchestration_policy(
                "neural-direct"
            )
        ),
    )

    result = session.chat(
        "Does Alice work with Bob?"
    )

    assert len(
        runtime.calls
    ) == 1
    assert (
        result.cortex_context[
            "prompt_ir"
        ]
        is None
    )
    assert (
        result.cortex_context[
            "hybrid_runtime"
        ][
            "neural_call_count"
        ]
        == 1
    )
    assert (
        result.cortex_context[
            "diagnosis"
        ][
            "status"
        ]
        == "neural_only"
    )
    exchange = result.cortex_context[
        "hybrid_exchange"
    ]
    assert (
        exchange["plan"][
            "route"
        ]
        == "neural_direct"
    )
    assert (
        "parse_skipped"
        in [
            event["phase"]
            for event
            in exchange["events"]
        ]
    )


def test_no_revision_policy_preserves_conflict_without_third_call():
    runtime = FakeNeuralRuntime()
    session = HybridCoreSession(
        "no-revision",
        HybridCoreConfig(
            chat_model="knowledge",
            parser_model="parser",
            feature_dim=64,
            include_diagnosis=False,
        ),
        runtime,
        orchestration_policy=(
            named_orchestration_policy(
                "no-revision"
            )
        ),
    )

    session.chat(
        "Alice does not work with Bob."
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
        == "conflict"
    )
    assert (
        result.cortex_context[
            "hybrid_runtime"
        ][
            "neural_call_count"
        ]
        == 2
    )
    assert (
        result.cortex_context[
            "hybrid_exchange"
        ][
            "revision_count"
        ]
        == 0
    )


def test_no_cortex_context_policy_observes_but_withholds_pre_generation_state():
    runtime = FakeNeuralRuntime()
    session = HybridCoreSession(
        "withheld",
        HybridCoreConfig(
            chat_model="knowledge",
            parser_model="parser",
            feature_dim=64,
            include_diagnosis=False,
        ),
        runtime,
        orchestration_policy=(
            named_orchestration_policy(
                "no-cortex-context"
            )
        ),
    )

    session.chat(
        "Alice works with Bob."
    )
    result = session.chat(
        "Does Alice work with Bob?"
    )

    answer_call = runtime.calls[-1]
    system = answer_call[
        "messages"
    ][0][
        "content"
    ]
    payload = json.loads(
        system.split(
            "HYBRID_EXCHANGE=",
            1,
        )[1]
    )

    assert "cortex" not in payload
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
    assert (
        result.cortex_context[
            "hybrid_exchange"
        ][
            "plan"
        ][
            "inject_cortex"
        ]
        is False
    )


def test_server_factory_exposes_named_hybrid_policy(
    tmp_path,
):
    gateway = build_gateway_from_env(
        chat_model="knowledge",
        parser_model="parser",
        hybrid_runtime=True,
        hybrid_policy=(
            "neural-direct"
        ),
        persistence_dir=str(
            tmp_path
        ),
        feature_dim=64,
    )

    plan = (
        gateway.orchestration_policy.plan(
            user_text="hello",
            has_relations=False,
            has_bootstrap=False,
        )
    )
    assert (
        plan.route
        == "neural_direct"
    )
    assert not plan.parse_user
    assert not plan.inject_cortex
    assert not plan.assess_candidate
