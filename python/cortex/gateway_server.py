"""HTTP server for Cortex Language Gateway v0.1.

Install the optional gateway dependencies and run:

    cortex-gateway --chat-model qwen3:8b --embedding-model embeddinggemma

The v0.1 server exposes a deliberately small OpenAI-compatible surface plus
Cortex session controls. Streaming is deferred to a later milestone.
"""
from __future__ import annotations

import argparse
import os
from dataclasses import replace
from typing import Any

from .language_gateway import (
    CortexLanguageGateway,
    GatewayConfig,
    openai_chat_completion,
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


def _last_user_text(messages: Any) -> str:
    if not isinstance(messages, list):
        raise ValueError("messages must be an array")
    for message in reversed(messages):
        if not isinstance(message, dict):
            continue
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str):
            return content
    raise ValueError("a string user message is required")


def create_app(
    gateway: CortexLanguageGateway,
    *,
    public_model_name: str = "cortex",
):
    try:
        from fastapi import FastAPI, Header, HTTPException, Request
        from fastapi.responses import JSONResponse
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
            "gateway": "cortex-language-gateway-v0.1",
            "ollama_url": gateway.config.ollama_url,
            "chat_model": gateway.config.chat_model,
            "embedding_model": gateway.config.embedding_model,
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
        request: Request,
        x_cortex_session: str | None = Header(
            default=None,
            alias="X-Cortex-Session",
        ),
        x_cortex_diagnostics: str | None = Header(
            default=None,
            alias="X-Cortex-Diagnostics",
        ),
    ):
        payload = await request.json()
        if not isinstance(payload, dict):
            raise HTTPException(
                status_code=400,
                detail="request body must be an object",
            )
        if payload.get("stream", False):
            raise HTTPException(
                status_code=400,
                detail="streaming is not supported in gateway v0.1",
            )

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

        return JSONResponse(
            openai_chat_completion(result)
        )

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
        request: Request,
    ) -> dict[str, Any]:
        payload = await request.json()
        if not isinstance(payload, dict):
            raise HTTPException(
                status_code=400,
                detail="request body must be an object",
            )
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
) -> CortexLanguageGateway:
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
    config = GatewayConfig(
        chat_model=model,
        embedding_model=embed,
        ollama_url=url,
        diagnostics_default=bool(
            default_diag
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
    )
    parser.add_argument(
        "--ollama-url",
        default=None,
    )
    parser.add_argument(
        "--diagnostics",
        action="store_true",
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
    )
    app = create_app(gateway)
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
    )


if __name__ == "__main__":
    main()
