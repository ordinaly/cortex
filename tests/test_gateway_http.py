from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from cortex.gateway_server import create_app
from cortex.language_gateway import ChatResult


class DummyGateway:
    def __init__(self):
        self.config = SimpleNamespace(
            ollama_url="http://localhost:11434",
            chat_model="dummy",
            embedding_model="dummy",
        )
        self.sessions = {}
        self.calls = []

    def chat(
        self,
        session_id: str,
        user_text: str,
        *,
        diagnostics=None,
    ):
        self.calls.append(
            {
                "session_id": session_id,
                "user_text": user_text,
                "diagnostics": diagnostics,
            }
        )
        return ChatResult(
            response_id="chatcmpl-test",
            session_id=session_id,
            model="cortex",
            content="Hello from Cortex.",
            cortex_context={
                "diagnosis": {
                    "status": "neural_prior",
                    "reasoning_mode": "pretrained_knowledge",
                    "summary": (
                        "The neural model supplied the answer; "
                        "Cortex found no conflicting stored evidence."
                    ),
                    "neural_claims": [
                        {
                            "status": "neural_prior",
                        }
                    ],
                    "cortex_claims": [],
                    "accepted_user_observations": 0,
                    "memory_changed": False,
                    "analysis_errors": [],
                    "runtime": {
                        "unresolved": False,
                        "revision": False,
                        "reactivated": False,
                        "discovered": False,
                        "budget_pressure": False,
                        "stored": 1,
                    },
                    "timings": {
                        "knowledge": 0.25,
                        "cortex_prompt": 0.05,
                        "candidate": 0.10,
                    },
                }
            },
            diagnostics=None,
        )

    def list_persisted_sessions(self):
        return []

    def set_diagnostics(self, session_id, enabled):
        pass

    def reset(self, session_id):
        pass

    def save(self, session_id):
        return "/tmp/test.json"

    def load(self, session_id):
        return SimpleNamespace(
            history=[],
            frames=[],
        )

    def session(self, session_id):
        return SimpleNamespace(
            diagnostics_enabled=False,
            last_read=None,
            history=[],
            config=SimpleNamespace(
                memory_budget=24,
            ),
            context_packet=lambda diagnostics=None: {},
        )


@pytest.fixture()
def client():
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    gateway = DummyGateway()
    return (
        TestClient(
            create_app(
                gateway
            )
        ),
        gateway,
    )


def test_openai_chat_body_is_not_treated_as_query_parameter(client):
    http, gateway = client

    response = http.post(
        "/v1/chat/completions",
        json={
            "model": "cortex",
            "messages": [
                {
                    "role": "user",
                    "content": "Hello Cortex",
                }
            ],
            "stream": False,
        },
        headers={
            "X-Cortex-Session": "alpaca-test",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert (
        payload["choices"][0]["message"]["content"]
        == "Hello from Cortex."
    )
    assert gateway.calls == [
        {
            "session_id": "alpaca-test",
            "user_text": "Hello Cortex",
            "diagnostics": None,
        }
    ]


def test_alpaca_style_streaming_returns_openai_sse(client):
    http, _gateway = client

    with http.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "cortex",
            "messages": [
                {
                    "role": "user",
                    "content": "Hello Cortex",
                }
            ],
            "stream": True,
            "temperature": 0.7,
        },
    ) as response:
        assert response.status_code == 200
        assert response.headers[
            "content-type"
        ].startswith(
            "text/event-stream"
        )
        body = "".join(
            response.iter_text()
        )

    lines = [
        line
        for line in body.splitlines()
        if line.startswith("data: ")
    ]
    assert lines[-1] == "data: [DONE]"

    chunks = []
    for line in lines[:-1]:
        payload = json.loads(
            line[len("data: ") :]
        )
        assert (
            payload["object"]
            == "chat.completion.chunk"
        )
        chunks.append(
            payload
        )

    text = "".join(
        chunk["choices"][0][
            "delta"
        ].get(
            "content",
            "",
        )
        for chunk in chunks
    )
    assert text == "Hello from Cortex."
    assert (
        chunks[-1]["choices"][0][
            "finish_reason"
        ]
        == "stop"
    )


def test_diagnostics_endpoint_accepts_json_body(client):
    http, _gateway = client

    response = http.post(
        "/cortex/sessions/alpaca-test/diagnostics",
        json={
            "enabled": True,
        },
    )

    assert response.status_code == 200
    assert response.json()[
        "diagnostics_enabled"
    ] is True


def test_alpaca_content_parts_are_extracted_as_user_text(client):
    http, gateway = client

    response = http.post(
        "/v1/chat/completions",
        json={
            "model": "cortex",
            "messages": [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": "System instructions",
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/png;base64,AAAA",
                            },
                        },
                        {
                            "type": "text",
                            "text": "Hello from Alpaca",
                        },
                    ],
                },
            ],
            "stream": True,
        },
    )

    assert response.status_code == 200
    assert gateway.calls[-1]["user_text"] == "Hello from Alpaca"


def test_multiple_text_parts_preserve_order(client):
    http, gateway = client

    response = http.post(
        "/v1/chat/completions",
        json={
            "model": "cortex",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "First part",
                        },
                        {
                            "type": "text",
                            "text": "Second part",
                        },
                    ],
                }
            ],
            "stream": False,
        },
    )

    assert response.status_code == 200
    assert (
        gateway.calls[-1]["user_text"]
        == "First part\nSecond part"
    )


def test_openai_python_sdk_receives_streamed_delta_content(client):
    http, _gateway = client
    openai = pytest.importorskip("openai")

    sdk = openai.OpenAI(
        api_key="cortex",
        base_url="http://testserver/v1",
        http_client=http,
    )

    stream = sdk.chat.completions.create(
        model="cortex",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Hello Cortex",
                    }
                ],
            }
        ],
        stream=True,
    )

    pieces = []
    finish_reason = None
    for chunk in stream:
        assert chunk.choices
        choice = chunk.choices[0]
        if choice.delta.content:
            pieces.append(
                choice.delta.content
            )
        if choice.finish_reason:
            finish_reason = (
                choice.finish_reason
            )

    assert "".join(
        pieces
    ) == "Hello from Cortex."
    assert finish_reason == "stop"


def test_alpaca_title_request_bypasses_cortex_state(client):
    http, gateway = client

    response = http.post(
        "/v1/chat/completions",
        json={
            "model": "cortex",
            "messages": [
                {
                    "role": "system",
                    "content": "Generate a concise title.",
                },
                {
                    "role": "user",
                    "content": "Generate a title for this prompt:\nAlice works with Bob.",
                },
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "ChatTitle",
                    "schema": {
                        "type": "object"
                    },
                },
            },
        },
    )

    assert response.status_code == 200
    assert gateway.calls == []
    payload = response.json()
    content = payload[
        "choices"
    ][0]["message"]["content"]
    parsed = json.loads(content)
    assert parsed["title"]
    assert parsed["emoji"] == ""


def test_alpaca_plain_title_fallback_bypasses_cortex_state(client):
    http, gateway = client

    response = http.post(
        "/v1/chat/completions",
        json={
            "model": "cortex",
            "messages": [
                {
                    "role": "user",
                    "content": "Generate a title for this prompt:\nAlice works with Bob.",
                }
            ],
        },
    )

    assert response.status_code == 200
    assert gateway.calls == []
    content = response.json()[
        "choices"
    ][0]["message"]["content"]
    assert "Alice" in content


def test_alpaca_openai_1_84_stream_consumption_exact_loop(client):
    http, _gateway = client
    openai = pytest.importorskip("openai")

    sdk = openai.OpenAI(
        api_key="cortex",
        base_url="http://testserver/v1",
        http_client=http,
    )

    response = sdk.chat.completions.create(
        model="cortex",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Hello Cortex",
                    }
                ],
            }
        ],
        stream=True,
    )

    collected = []
    for chunk in response:
        if (
            chunk.choices
            and chunk.choices[0].delta
        ):
            delta = chunk.choices[0].delta
            if delta.content:
                collected.append(
                    delta.content
                )

    assert "".join(collected) == "Hello from Cortex."


def test_ollama_tags_match_alpaca_ollama_0_6_1_models(client):
    http, _gateway = client
    ollama = pytest.importorskip("ollama")

    response = http.get(
        "/api/tags"
    )

    assert response.status_code == 200
    parsed = ollama.ListResponse.model_validate(
        response.json()
    )
    assert parsed.models
    assert (
        parsed.models[0].model
        == "cortex:latest"
    )


def test_ollama_show_matches_alpaca_ollama_0_6_1_model(client):
    http, _gateway = client
    ollama = pytest.importorskip("ollama")

    response = http.post(
        "/api/show",
        json={
            "model": "cortex",
        },
    )

    assert response.status_code == 200
    parsed = ollama.ShowResponse.model_validate(
        response.json()
    )
    assert (
        "completion"
        in (
            parsed.capabilities
            or []
        )
    )


def test_ollama_chat_stream_matches_alpaca_ollama_0_6_1(client):
    http, gateway = client
    ollama = pytest.importorskip("ollama")

    with http.stream(
        "POST",
        "/api/chat",
        json={
            "model": "cortex",
            "messages": [
                {
                    "role": "user",
                    "content": "Hello Cortex",
                }
            ],
            "stream": True,
            "think": False,
        },
    ) as response:
        assert response.status_code == 200
        assert response.headers[
            "content-type"
        ].startswith(
            "application/x-ndjson"
        )
        raw_lines = [
            line
            for line
            in response.iter_lines()
            if line.strip()
        ]

    chunks = [
        ollama.ChatResponse.model_validate_json(
            line
        )
        for line in raw_lines
    ]

    text = "".join(
        chunk.message.content
        or ""
        for chunk in chunks
    )

    assert text == "Hello from Cortex."
    assert chunks[-1].done is True
    assert (
        chunks[-1].done_reason
        == "stop"
    )
    thinking = "".join(
        chunk.message.thinking
        or ""
        for chunk in chunks
    )
    assert (
        "Cortex diagnosis"
        in thinking
    )
    assert (
        "Reasoning mode: pretrained_knowledge"
        in thinking
    )
    assert (
        chunks[-1].total_duration
        == 400_000_000
    )
    assert gateway.calls[-1][
        "user_text"
    ] == "Hello Cortex"


def test_ollama_non_streaming_exposes_cortex_thinking_and_timing(client):
    http, _gateway = client
    ollama = pytest.importorskip("ollama")

    response = http.post(
        "/api/chat",
        json={
            "model": "cortex",
            "messages": [
                {
                    "role": "user",
                    "content": "Hello Cortex",
                }
            ],
            "stream": False,
        },
    )

    assert response.status_code == 200
    parsed = ollama.ChatResponse.model_validate(
        response.json()
    )
    assert (
        "Cortex diagnosis"
        in (
            parsed.message.thinking
            or ""
        )
    )
    assert (
        parsed.total_duration
        == 400_000_000
    )


def test_ollama_alpaca_title_request_bypasses_cortex(client):
    http, gateway = client
    ollama = pytest.importorskip("ollama")

    response = http.post(
        "/api/chat",
        json={
            "model": "cortex",
            "stream": False,
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "You are an assistant that generates short chat titles "
                        "based on the prompt. If you want to, you can add a single "
                        "emoji. Format the response as a single JSON object."
                        "\n\nHello Cortex"
                    ),
                }
            ],
            "format": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string"
                    }
                },
                "required": [
                    "title"
                ],
            },
            "think": False,
        },
    )

    assert response.status_code == 200
    assert gateway.calls == []
    parsed = ollama.ChatResponse.model_validate(
        response.json()
    )
    payload = json.loads(
        parsed.message.content
        or "{}"
    )
    assert payload[
        "title"
    ] == "Hello Cortex"
