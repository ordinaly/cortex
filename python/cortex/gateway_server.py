"""HTTP server for Cortex Language Gateway v0.1.

Install the optional gateway dependencies and run:

    cortex-gateway --chat-model qwen3:8b --embedding-model embeddinggemma

The server exposes a deliberately small OpenAI-compatible surface plus
Cortex session controls. OpenAI SSE transport streaming is supported; Cortex
currently computes the complete answer before chunks are emitted.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

from .language_gateway import (
    CortexLanguageGateway,
    GatewayConfig,
    openai_chat_completion,
)
from .strict_language_gateway import (
    STRICT_GATEWAY_VERSION,
    StrictCortexLanguageGateway,
    StrictGatewayConfig,
)


def _bool_value(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"invalid boolean value: {value!r}")


def _message_text_content(content: Any) -> str | None:
    """Extract textual content from OpenAI chat message content.

    OpenAI-compatible clients may send either a plain string or an ordered
    list of content parts such as:
        [{"type": "text", "text": "hello"}, {"type": "image_url", ...}]
    Only textual parts are forwarded into the Cortex language parser.
    """

    if isinstance(content, str):
        text = content.strip()
        return text or None

    if isinstance(content, dict):
        part_type = content.get("type")
        text = content.get("text")
        if (
            part_type in ("text", "input_text")
            and isinstance(text, str)
        ):
            normalized = text.strip()
            return normalized or None
        return None

    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if not isinstance(part, dict):
                continue
            part_type = part.get("type")
            text = part.get("text")
            if (
                part_type in ("text", "input_text")
                and isinstance(text, str)
                and text.strip()
            ):
                parts.append(text.strip())
        if parts:
            return "\n".join(parts)
        return None

    return None


def _last_user_text(messages: Any) -> str:
    if not isinstance(messages, list):
        raise ValueError("messages must be an array")
    for message in reversed(messages):
        if not isinstance(message, dict):
            continue
        if message.get("role") != "user":
            continue
        text = _message_text_content(
            message.get("content")
        )
        if text is not None:
            return text
    raise ValueError(
        "a textual user message is required"
    )


def _openai_stream_events(result: Any):
    """Emit a minimal OpenAI-compatible chat-completion SSE stream."""

    payload = openai_chat_completion(result)
    response_id = payload["id"]
    created = payload["created"]
    model = payload["model"]
    content = payload["choices"][0]["message"]["content"]

    def event(data: dict[str, Any]) -> str:
        import json

        return "data: " + json.dumps(
            data,
            ensure_ascii=False,
            separators=(",", ":"),
        ) + "\n\n"

    yield event(
        {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                    },
                    "finish_reason": None,
                }
            ],
        }
    )

    # Cortex currently computes the answer before HTTP streaming begins.
    # Chunking here is transport compatibility for OpenAI clients such as
    # Alpaca; it is intentionally not token-level generation streaming.
    chunk_size = 64
    for offset in range(
        0,
        len(content),
        chunk_size,
    ):
        yield event(
            {
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "content": content[
                                offset : offset
                                + chunk_size
                            ],
                        },
                        "finish_reason": None,
                    }
                ],
            }
        )

    yield event(
        {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop",
                }
            ],
        }
    )
    yield "data: [DONE]\n\n"


def create_app(
    gateway: Any,
    *,
    public_model_name: str = "cortex",
):
    try:
        from fastapi import Body, FastAPI, Header, HTTPException
        from fastapi.responses import JSONResponse, StreamingResponse
    except ImportError as exc:  # pragma: no cover - optional runtime dependency
        raise RuntimeError(
            "Gateway HTTP dependencies are not installed. "
            "Install cortex-reasoner[gateway]."
        ) from exc

    app = FastAPI(
        title="Cortex Language Gateway",
        version="0.1",
    )

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "gateway": (
                STRICT_GATEWAY_VERSION
                if isinstance(
                    gateway,
                    StrictCortexLanguageGateway,
                )
                else "cortex-language-gateway-v0.1"
            ),
            "ollama_url": gateway.config.ollama_url,
            "chat_model": gateway.config.chat_model,
            "embedding_model": gateway.config.embedding_model,
            "strict_neural_boundary": isinstance(
                gateway,
                StrictCortexLanguageGateway,
            ),
            "parser_model": getattr(
                gateway.config,
                "parser_model",
                None,
            ),
            "renderer_model": getattr(
                gateway.config,
                "renderer_model",
                None,
            ),
        }

    @app.get("/v1/models")
    def models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {
                    "id": public_model_name,
                    "object": "model",
                    "owned_by": "cortex",
                }
            ],
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(
        payload: dict[str, Any] = Body(...),
        x_cortex_session: str | None = Header(
            default=None,
            alias="X-Cortex-Session",
        ),
        x_cortex_diagnostics: str | None = Header(
            default=None,
            alias="X-Cortex-Diagnostics",
        ),
    ):
        try:
            user_text = _last_user_text(
                payload.get("messages")
            )
            session_id = (
                x_cortex_session
                or str(
                    payload.get(
                        "cortex_session",
                        "default",
                    )
                )
            )
            diagnostics = _bool_value(
                x_cortex_diagnostics
                if x_cortex_diagnostics is not None
                else payload.get(
                    "cortex_diagnostics"
                )
            )
            result = gateway.chat(
                session_id,
                user_text,
                diagnostics=diagnostics,
            )
        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail=str(exc),
            ) from exc
        except RuntimeError as exc:
            raise HTTPException(
                status_code=502,
                detail=str(exc),
            ) from exc

        if payload.get("stream", False):
            return StreamingResponse(
                _openai_stream_events(
                    result
                ),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                },
            )

        return JSONResponse(
            openai_chat_completion(result)
        )

    @app.get("/cortex/sessions")
    def list_sessions() -> dict[str, Any]:
        return {
            "active": sorted(
                gateway.sessions
            ),
            "persisted": (
                gateway.list_persisted_sessions()
            ),
        }

    @app.post(
        "/cortex/sessions/{session_id}/save"
    )
    def save_session(
        session_id: str,
    ) -> dict[str, Any]:
        try:
            path = gateway.save(
                session_id
            )
        except RuntimeError as exc:
            raise HTTPException(
                status_code=409,
                detail=str(exc),
            ) from exc
        return {
            "session_id": session_id,
            "saved": True,
            "path": path,
        }

    @app.post(
        "/cortex/sessions/{session_id}/load"
    )
    def load_session(
        session_id: str,
    ) -> dict[str, Any]:
        try:
            session = gateway.load(
                session_id
            )
        except FileNotFoundError as exc:
            raise HTTPException(
                status_code=404,
                detail=str(exc),
            ) from exc
        except (
            RuntimeError,
            ValueError,
        ) as exc:
            raise HTTPException(
                status_code=409,
                detail=str(exc),
            ) from exc
        return {
            "session_id": session_id,
            "loaded": True,
            "history_messages": len(
                session.history
            ),
            "frames": len(
                session.frames
            ),
        }

    @app.get("/cortex/sessions/{session_id}")
    def session_state(
        session_id: str,
    ) -> dict[str, Any]:
        session = gateway.session(session_id)
        diagnostic = None
        if (
            session.diagnostics_enabled
            and session.last_read is not None
        ):
            from .language_gateway import diagnose_runtime

            diagnostic = diagnose_runtime(
                session.last_read,
                memory_budget=session.config.memory_budget,
            )
        return {
            "session_id": session_id,
            "diagnostics_enabled": session.diagnostics_enabled,
            "history_messages": len(session.history),
            "context": session.context_packet(
                diagnostics=diagnostic
            ),
        }

    @app.post(
        "/cortex/sessions/{session_id}/diagnostics"
    )
    async def set_diagnostics(
        session_id: str,
        payload: dict[str, Any] = Body(...),
    ) -> dict[str, Any]:
        try:
            enabled = _bool_value(
                payload.get("enabled")
            )
        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail=str(exc),
            ) from exc
        if enabled is None:
            raise HTTPException(
                status_code=400,
                detail="enabled is required",
            )
        gateway.set_diagnostics(
            session_id,
            enabled,
        )
        return {
            "session_id": session_id,
            "diagnostics_enabled": enabled,
        }

    @app.delete("/cortex/sessions/{session_id}")
    def reset_session(
        session_id: str,
    ) -> dict[str, Any]:
        gateway.reset(session_id)
        return {
            "session_id": session_id,
            "reset": True,
        }

    return app


def build_gateway_from_env(
    *,
    chat_model: str | None = None,
    embedding_model: str | None = None,
    ollama_url: str | None = None,
    diagnostics_default: bool | None = None,
    persistence_dir: str | None = None,
    strict_neural_boundary: bool = False,
    parser_model: str | None = None,
    renderer_model: str | None = None,
    feature_dim: int = 128,
) -> Any:
    model = (
        chat_model
        or os.environ.get("CORTEX_CHAT_MODEL")
        or "qwen3:8b"
    )
    embed = (
        embedding_model
        or os.environ.get("CORTEX_EMBEDDING_MODEL")
        or "embeddinggemma"
    )
    url = (
        ollama_url
        or os.environ.get("OLLAMA_HOST")
        or "http://localhost:11434"
    )
    default_diag = (
        diagnostics_default
        if diagnostics_default is not None
        else _bool_value(
            os.environ.get(
                "CORTEX_DIAGNOSTICS",
                "false",
            )
        )
    )
    state_dir = (
        persistence_dir
        if persistence_dir is not None
        else os.environ.get(
            "CORTEX_STATE_DIR",
            str(
                Path.home()
                / ".cortex"
                / "gateway"
            ),
        )
    )
    state_dir = (
        state_dir.strip()
        if isinstance(
            state_dir,
            str,
        )
        else state_dir
    )
    if strict_neural_boundary:
        strict_config = StrictGatewayConfig(
            parser_model=(
                parser_model
                or os.environ.get(
                    "CORTEX_PARSER_MODEL"
                )
                or model
            ),
            renderer_model=(
                renderer_model
                or os.environ.get(
                    "CORTEX_RENDERER_MODEL"
                )
                or model
            ),
            ollama_url=url,
            feature_dim=int(
                feature_dim
            ),
            diagnostics_default=bool(
                default_diag
            ),
            persistence_dir=(
                state_dir or None
            ),
        )
        return StrictCortexLanguageGateway(
            strict_config
        )

    config = GatewayConfig(
        chat_model=model,
        embedding_model=embed,
        ollama_url=url,
        diagnostics_default=bool(
            default_diag
        ),
        persistence_dir=(
            state_dir or None
        ),
    )
    return CortexLanguageGateway(config)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Cortex Ollama language gateway."
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=11435,
    )
    parser.add_argument(
        "--chat-model",
        default=None,
    )
    parser.add_argument(
        "--embedding-model",
        default=None,
        help="Legacy v0.1 embedding model; unused in strict mode.",
    )
    parser.add_argument(
        "--strict-neural-boundary",
        action="store_true",
        help=(
            "Use Gateway v0.2: stateless parser/renderer, deterministic "
            "Cortex encoding, and Cortex-owned AnswerIR."
        ),
    )
    parser.add_argument(
        "--parser-model",
        default=None,
        help="Ollama model used only for text -> PromptIR.",
    )
    parser.add_argument(
        "--renderer-model",
        default=None,
        help="Ollama model used only for RendererPacket -> text.",
    )
    parser.add_argument(
        "--feature-dim",
        type=int,
        default=128,
        help="Deterministic Cortex feature dimension in strict mode.",
    )
    parser.add_argument(
        "--ollama-url",
        default=None,
    )
    parser.add_argument(
        "--diagnostics",
        action="store_true",
    )
    parser.add_argument(
        "--state-dir",
        default=None,
        help=(
            "Directory for persistent Cortex sessions "
            "(default: CORTEX_STATE_DIR or ~/.cortex/gateway)."
        ),
    )
    parser.add_argument(
        "--no-persistence",
        action="store_true",
        help="Disable gateway session persistence.",
    )
    args = parser.parse_args()

    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover - optional runtime dependency
        raise SystemExit(
            "Install cortex-reasoner[gateway] to run the HTTP gateway."
        ) from exc

    gateway = build_gateway_from_env(
        chat_model=args.chat_model,
        embedding_model=args.embedding_model,
        ollama_url=args.ollama_url,
        diagnostics_default=args.diagnostics,
        persistence_dir=(
            ""
            if args.no_persistence
            else args.state_dir
        ),
        strict_neural_boundary=(
            args.strict_neural_boundary
        ),
        parser_model=args.parser_model,
        renderer_model=(
            args.renderer_model
        ),
        feature_dim=args.feature_dim,
    )
    app = create_app(gateway)
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
    )


if __name__ == "__main__":
    main()
