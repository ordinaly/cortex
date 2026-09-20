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
            cortex_context={},
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
