from __future__ import annotations

from cortex.neural_runtime import (
    NeuralCapabilities,
    NeuralRequest,
    NeuralResponse,
    NeuralSession,
    OllamaNeuralRuntime,
    RuntimeClientAdapter,
)


class FakeRuntime:
    def __init__(self):
        self.calls = []

    @property
    def backend_name(self):
        return "fake"

    @property
    def capabilities(self):
        return NeuralCapabilities(
            embeddings=True,
        )

    def generate(
        self,
        *,
        model,
        messages,
        schema=None,
        temperature=None,
    ):
        self.calls.append(
            {
                "model": model,
                "messages": [
                    dict(item)
                    for item in messages
                ],
                "schema": schema,
                "temperature": temperature,
            }
        )
        return NeuralResponse(
            text="ok",
            model=model,
            backend="fake",
            duration_s=0.01,
            structured=(
                schema is not None
            ),
        )

    def embed(
        self,
        *,
        model,
        inputs,
    ):
        return [
            [float(index)]
            for index, _ in enumerate(
                inputs
            )
        ]


def test_runtime_client_adapter_preserves_existing_client_contract():
    runtime = FakeRuntime()
    client = RuntimeClientAdapter(
        runtime
    )

    text = client.chat(
        model="test",
        messages=[
            {
                "role": "user",
                "content": "hello",
            }
        ],
        format_schema={
            "type": "object",
        },
    )

    assert text == "ok"
    assert runtime.calls[-1][
        "schema"
    ] == {
        "type": "object",
    }
    assert client.embed(
        model="embed",
        inputs=[
            "a",
            "b",
        ],
    ) == [
        [0.0],
        [1.0],
    ]


def test_ollama_runtime_reports_capability_boundary():
    runtime = OllamaNeuralRuntime()

    capabilities = (
        runtime.capabilities
    )

    assert capabilities.generate
    assert (
        capabilities.structured_generate
    )
    assert capabilities.embeddings
    assert not capabilities.hidden_states
    assert not capabilities.kv_cache
    assert not capabilities.adapters
    assert not capabilities.train_adapters


def test_ollama_runtime_returns_instrumented_response(monkeypatch):
    runtime = OllamaNeuralRuntime()

    def fake_post(path, payload):
        assert path == "/api/chat"
        assert (
            payload["model"]
            == "qwen"
        )
        assert (
            payload["stream"]
            is False
        )
        assert (
            payload["think"]
            is False
        )
        return {
            "message": {
                "role": "assistant",
                "content": (
                    '{"answer":"Paris"}'
                ),
            },
            "prompt_eval_count": 12,
            "eval_count": 4,
            "total_duration": 123,
            "load_duration": 7,
        }

    monkeypatch.setattr(
        runtime,
        "_post",
        fake_post,
    )

    response = runtime.generate(
        model="qwen",
        messages=[
            {
                "role": "user",
                "content": "question",
            }
        ],
        schema={
            "type": "object",
        },
    )

    assert (
        response.text
        == '{"answer":"Paris"}'
    )
    assert response.backend == "ollama"
    assert response.structured
    assert response.prompt_tokens == 12
    assert (
        response.completion_tokens
        == 4
    )
    assert (
        response.metadata[
            "total_duration_ns"
        ]
        == 123
    )



def test_neural_session_owns_per_exchange_call_trace():
    runtime = FakeRuntime()
    session = NeuralSession(
        session_id="s1",
        runtime=runtime,
    )
    session.begin_exchange(
        "exchange-1"
    )

    response = session.execute(
        NeuralRequest(
            stage="answer",
            model="test",
            messages=[
                {
                    "role": "user",
                    "content": "hello",
                }
            ],
            schema={
                "type": "object",
            },
            cache_policy=(
                "reuse_if_supported"
            ),
        )
    )

    assert response.text == "ok"
    packet = session.packet()
    assert (
        packet[
            "active_exchange_id"
        ]
        == "exchange-1"
    )
    assert (
        packet[
            "neural_call_count"
        ]
        == 1
    )
    call = packet[
        "neural_calls"
    ][0]
    assert (
        call["stage"]
        == "answer"
    )
    assert (
        call["exchange_id"]
        == "exchange-1"
    )
    assert (
        call["cache_policy"]
        == "reuse_if_supported"
    )
    assert (
        call["control_keys"]
        == []
    )


def test_neural_session_resets_trace_for_new_exchange():
    runtime = FakeRuntime()
    session = NeuralSession(
        session_id="s1",
        runtime=runtime,
    )
    session.begin_exchange(
        "one"
    )
    session.execute(
        NeuralRequest(
            stage="first",
            model="test",
            messages=[],
        )
    )
    session.begin_exchange(
        "two"
    )

    assert (
        session.packet()[
            "neural_call_count"
        ]
        == 0
    )
    assert (
        session.packet()[
            "active_exchange_id"
        ]
        == "two"
    )


def test_neural_session_rejects_unsupported_structured_generation():
    class LimitedRuntime(
        FakeRuntime
    ):
        @property
        def capabilities(self):
            return NeuralCapabilities(
                generate=True,
                structured_generate=False,
            )

    session = NeuralSession(
        session_id="s1",
        runtime=LimitedRuntime(),
    )

    try:
        session.execute(
            NeuralRequest(
                stage="structured",
                model="test",
                messages=[],
                schema={
                    "type": "object",
                },
            )
        )
    except RuntimeError as exc:
        assert (
            "structured generation"
            in str(
                exc
            )
        )
    else:
        raise AssertionError(
            "unsupported structured generation must fail"
        )



def test_neural_session_preserves_out_of_band_control_objects():
    runtime = FakeRuntime()
    session = NeuralSession(
        session_id="s1",
        runtime=runtime,
    )
    sentinel = object()

    session.execute(
        NeuralRequest(
            stage="answer",
            model="test",
            messages=[],
            control={
                "hybrid_exchange": (
                    sentinel
                ),
            },
        )
    )

    assert (
        session.calls[0].request.control[
            "hybrid_exchange"
        ]
        is sentinel
    )
    assert (
        session.packet()[
            "neural_calls"
        ][0][
            "control_keys"
        ]
        == [
            "hybrid_exchange"
        ]
    )
