"""HTTP server for Cortex Language Gateway v0.1.

Install the optional gateway dependencies and run:

    cortex-gateway --chat-model qwen3:8b --embedding-model embeddinggemma

The server exposes a deliberately small OpenAI-compatible surface plus
Cortex session controls. OpenAI SSE transport streaming is supported; Cortex
currently computes the complete answer before chunks are emitted.
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from time import sleep, time
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
from .hybrid_language_gateway import (
    HYBRID_GATEWAY_VERSION,
    HybridCortexLanguageGateway,
    HybridGatewayConfig,
)
from .hybrid_core import (
    HYBRID_CORE_VERSION,
    HybridCoreConfig,
)
from .hybrid_core_gateway import (
    HybridCoreGateway,
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


def _alpaca_title_request(
    user_text: str,
) -> bool:
    return user_text.startswith(
        "Generate a title for this prompt:"
    )


def _alpaca_title_text(
    user_text: str,
) -> str:
    _, _, prompt = user_text.partition(
        ":"
    )
    cleaned = " ".join(
        prompt.strip().split()
    )
    if not cleaned:
        return "New Chat"
    words = cleaned.split()
    title = " ".join(
        words[:6]
    )
    if len(title) > 30:
        title = title[:30].rstrip()
    return title or "New Chat"


def _utility_chat_completion(
    *,
    model: str,
    content: str,
) -> dict[str, Any]:
    created = int(
        time()
    )
    response_id = (
        "chatcmpl-util-"
        + sha256(
            content.encode(
                "utf-8"
            )
        ).hexdigest()[:20]
    )
    return {
        "id": response_id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }


def _openai_stream_events(
    result: Any,
    *,
    trace: bool = False,
):
    """Emit a minimal OpenAI-compatible chat-completion SSE stream."""

    payload = openai_chat_completion(result)
    response_id = payload["id"]
    created = payload["created"]
    model = payload["model"]
    content = payload["choices"][0]["message"]["content"]

    def event(data: dict[str, Any]) -> str:
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
    emitted_chunks = 0
    for offset in range(
        0,
        len(content),
        chunk_size,
    ):
        emitted_chunks += 1
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
    if trace:
        print(
            f"[cortex-gateway:{result.session_id}] "
            f"sse:done ({len(content)} chars, {emitted_chunks} content chunks)",
            flush=True,
        )


def _ollama_title_request(
    user_text: str,
) -> bool:
    return user_text.startswith(
        "You are an assistant that generates short chat titles"
    )


def _ollama_created_at() -> str:
    return (
        datetime.now(
            timezone.utc
        )
        .isoformat()
        .replace("+00:00", "Z")
    )


def _ollama_visible_content(
    result: Any,
) -> str:
    context = getattr(
        result,
        "cortex_context",
        None,
    )
    if isinstance(
        context,
        dict,
    ):
        neural_answer = context.get(
            "neural_answer"
        )
        if (
            isinstance(
                neural_answer,
                str,
            )
            and neural_answer.strip()
        ):
            return neural_answer
    return str(
        getattr(
            result,
            "content",
            "",
        )
    )


def _cortex_thinking_text(
    result: Any,
) -> str:
    context = getattr(
        result,
        "cortex_context",
        None,
    )
    if not isinstance(
        context,
        dict,
    ):
        return ""
    diagnosis = context.get(
        "diagnosis"
    )
    if not isinstance(
        diagnosis,
        dict,
    ):
        return ""

    lines = [
        "Cortex diagnosis",
        (
            "Status: "
            + str(
                diagnosis.get(
                    "status",
                    "unknown",
                )
            )
        ),
        (
            "Reasoning mode: "
            + str(
                diagnosis.get(
                    "reasoning_mode",
                    "unknown",
                )
            )
        ),
    ]
    summary = diagnosis.get(
        "summary"
    )
    if (
        isinstance(
            summary,
            str,
        )
        and summary.strip()
    ):
        lines.append(
            "Assessment: "
            + summary.strip()
        )
    lines.append(
        (
            "Memory: structural state changed on this turn."
            if diagnosis.get(
                "memory_changed"
            )
            else "Memory: no structural state change."
        )
    )

    exchange = context.get(
        "hybrid_exchange"
    )
    if isinstance(
        exchange,
        dict,
    ):
        plan = exchange.get(
            "plan"
        )
        if isinstance(
            plan,
            dict,
        ):
            lines.extend(
                [
                    "",
                    "Orchestration",
                    (
                        "Route: "
                        + str(
                            plan.get(
                                "route",
                                "unknown",
                            )
                        )
                    ),
                    (
                        "Cortex context: "
                        + (
                            "enabled"
                            if plan.get(
                                "inject_cortex"
                            )
                            else "disabled"
                        )
                    ),
                    (
                        "Revision budget: "
                        + str(
                            plan.get(
                                "revision_budget",
                                0,
                            )
                        )
                    ),
                ]
            )
            reason = plan.get(
                "reason"
            )
            if (
                isinstance(
                    reason,
                    str,
                )
                and reason.strip()
            ):
                lines.append(
                    "Route reason: "
                    + reason.strip()
                )
        events = exchange.get(
            "events"
        )
        if isinstance(
            events,
            list,
        ):
            phases = [
                str(
                    item.get(
                        "phase",
                        "",
                    )
                )
                for item in events
                if (
                    isinstance(
                        item,
                        dict,
                    )
                    and item.get(
                        "phase"
                    )
                )
            ]
            if phases:
                lines.append(
                    "Exchange phases: "
                    + " -> ".join(
                        phases
                    )
                )

    runtime = context.get(
        "hybrid_runtime"
    )
    if isinstance(
        runtime,
        dict,
    ):
        lines.extend(
            [
                "",
                "Hybrid runtime",
                (
                    "Backend: "
                    + str(
                        runtime.get(
                            "backend",
                            "unknown",
                        )
                    )
                ),
                (
                    "Neural calls: "
                    + str(
                        runtime.get(
                            "neural_call_count",
                            0,
                        )
                    )
                ),
                (
                    "Revision cycles: "
                    + str(
                        runtime.get(
                            "revision_cycles",
                            0,
                        )
                    )
                ),
            ]
        )
        calls = runtime.get(
            "neural_calls"
        )
        if isinstance(
            calls,
            list,
        ):
            stages = [
                str(
                    item.get(
                        "stage",
                        "unknown",
                    )
                )
                for item in calls
                if isinstance(
                    item,
                    dict,
                )
            ]
            if stages:
                lines.append(
                    "Stages: "
                    + " -> ".join(
                        stages
                    )
                )

    bootstrap = context.get(
        "bootstrap"
    )
    if isinstance(
        bootstrap,
        dict,
    ):
        lines.append(
            (
                "Bootstrap claims: "
                + str(
                    bootstrap.get(
                        "claim_count",
                        0,
                    )
                )
            )
        )

    errors = diagnosis.get(
        "analysis_errors"
    )
    if (
        isinstance(
            errors,
            list,
        )
        and errors
    ):
        lines.append(
            "Analysis warnings: "
            + "; ".join(
                str(
                    item
                )
                for item in errors
            )
        )

    return "\n".join(
        lines
    )


def _ollama_timing_metadata(
    result: Any,
) -> dict[str, int]:
    context = getattr(
        result,
        "cortex_context",
        None,
    )
    diagnosis = (
        context.get(
            "diagnosis"
        )
        if isinstance(
            context,
            dict,
        )
        else None
    )
    timings = (
        diagnosis.get(
            "timings"
        )
        if isinstance(
            diagnosis,
            dict,
        )
        else None
    )
    total_seconds = 0.0
    if isinstance(
        timings,
        dict,
    ):
        for value in timings.values():
            if isinstance(
                value,
                (
                    int,
                    float,
                ),
            ):
                total_seconds += max(
                    0.0,
                    float(
                        value
                    ),
                )
    return {
        "total_duration": int(
            total_seconds
            * 1_000_000_000
        ),
        "load_duration": 0,
        "prompt_eval_count": 0,
        "prompt_eval_duration": 0,
        "eval_count": 0,
        "eval_duration": 0,
    }


def _ollama_chat_payload(
    *,
    model: str,
    content: str,
    done: bool,
    thinking: str = "",
    metrics: dict[str, int] | None = None,
) -> dict[str, Any]:
    return {
        "model": model,
        "created_at": (
            _ollama_created_at()
        ),
        "message": {
            "role": "assistant",
            "content": content,
            "thinking": thinking,
            "tool_calls": [],
        },
        "done": done,
        "done_reason": (
            "stop"
            if done
            else None
        ),
        **(
            metrics
            or {
                "total_duration": 0,
                "load_duration": 0,
                "prompt_eval_count": 0,
                "prompt_eval_duration": 0,
                "eval_count": 0,
                "eval_duration": 0,
            }
        ),
    }


def _ollama_stream_events(
    result: Any,
    *,
    model: str,
    trace: bool = False,
):
    content = (
        _ollama_visible_content(
            result
        )
    )
    thinking = (
        _cortex_thinking_text(
            result
        )
    )
    metrics = (
        _ollama_timing_metadata(
            result
        )
    )
    emitted_content = 0
    emitted_thinking = 0

    if thinking:
        emitted_thinking = 1
        yield (
            json.dumps(
                _ollama_chat_payload(
                    model=model,
                    content="",
                    thinking=thinking,
                    done=False,
                ),
                ensure_ascii=False,
                separators=(",", ":"),
            )
            + "\n"
        )

    if content:
        # Alpaca queues text through GLib.idle_add and can finalize before
        # multiple artificial chunks are rendered. Cortex already has the full
        # answer here, so expose it atomically and leave a tiny GTK flush window.
        emitted_content = 1
        yield (
            json.dumps(
                _ollama_chat_payload(
                    model=model,
                    content=content,
                    done=False,
                ),
                ensure_ascii=False,
                separators=(",", ":"),
            )
            + "\n"
        )
        sleep(
            0.075
        )

    yield (
        json.dumps(
            _ollama_chat_payload(
                model=model,
                content="",
                done=True,
                metrics=metrics,
            ),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        + "\n"
    )

    if trace:
        print(
            f"[cortex-gateway:{result.session_id}] "
            f"ollama:done ({len(content)} answer chars, "
            f"{len(thinking)} diagnosis chars, "
            f"{emitted_content} answer chunks, "
            f"{emitted_thinking} diagnosis chunks)",
            flush=True,
        )


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
                HYBRID_CORE_VERSION
                if isinstance(
                    gateway,
                    HybridCoreGateway,
                )
                else (
                    STRICT_GATEWAY_VERSION
                    if isinstance(
                        gateway,
                        StrictCortexLanguageGateway,
                    )
                    else (
                        HYBRID_GATEWAY_VERSION
                        if isinstance(
                            gateway,
                            HybridCortexLanguageGateway,
                        )
                        else "cortex-language-gateway-v0.1"
                    )
                )
            ),
            "ollama_url": gateway.config.ollama_url,
            "chat_model": gateway.config.chat_model,
            "embedding_model": gateway.config.embedding_model,
            "strict_neural_boundary": isinstance(
                gateway,
                StrictCortexLanguageGateway,
            ),
            "hybrid_cognition": isinstance(
                gateway,
                HybridCortexLanguageGateway,
            ),
            "hybrid_runtime": isinstance(
                gateway,
                HybridCoreGateway,
            ),
            "neural_backend": (
                gateway.runtime.backend_name
                if isinstance(
                    gateway,
                    HybridCoreGateway,
                )
                else None
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
            "explain_responses": bool(
                getattr(
                    gateway.config,
                    "explain_responses",
                    False,
                )
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

    @app.get("/api/tags")
    def ollama_tags() -> dict[str, Any]:
        return {
            "models": [
                {
                    "name": (
                        public_model_name
                        + ":latest"
                    ),
                    "model": (
                        public_model_name
                        + ":latest"
                    ),
                    "modified_at": (
                        _ollama_created_at()
                    ),
                    "size": 0,
                    "digest": (
                        "sha256:"
                        + sha256(
                            public_model_name.encode(
                                "utf-8"
                            )
                        ).hexdigest()
                    ),
                    "details": {
                        "parent_model": "",
                        "format": "cortex-gateway",
                        "family": "cortex",
                        "families": [
                            "cortex"
                        ],
                        "parameter_size": (
                            "hybrid"
                        ),
                        "quantization_level": (
                            "N/A"
                        ),
                    },
                }
            ]
        }

    @app.post("/api/show")
    async def ollama_show(
        payload: dict[str, Any] = Body(...),
    ) -> dict[str, Any]:
        return {
            "modified_at": (
                _ollama_created_at()
            ),
            "template": "",
            "modelfile": "",
            "license": "",
            "details": {
                "parent_model": "",
                "format": "cortex-gateway",
                "family": "cortex",
                "families": [
                    "cortex"
                ],
                "parameter_size": "hybrid",
                "quantization_level": "N/A",
            },
            "model_info": {
                "general.architecture": (
                    "cortex-hybrid-core"
                    if isinstance(
                        gateway,
                        HybridCoreGateway,
                    )
                    else "cortex-hybrid"
                ),
                "cortex.gateway": (
                    HYBRID_CORE_VERSION
                    if isinstance(
                        gateway,
                        HybridCoreGateway,
                    )
                    else (
                        HYBRID_GATEWAY_VERSION
                        if isinstance(
                            gateway,
                            HybridCortexLanguageGateway,
                        )
                        else "cortex"
                    )
                ),
            },
            "parameters": "",
            "capabilities": [
                "completion"
            ],
        }

    @app.get("/api/ps")
    def ollama_ps() -> dict[str, Any]:
        return {
            "models": []
        }

    @app.post("/api/chat")
    async def ollama_chat(
        payload: dict[str, Any] = Body(...),
    ):
        try:
            messages = payload.get(
                "messages"
            )
            user_text = _last_user_text(
                messages
            )
            model_name = str(
                payload.get(
                    "model",
                    public_model_name,
                )
            )

            if (
                _ollama_title_request(
                    user_text
                )
                and isinstance(
                    payload.get(
                        "format"
                    ),
                    dict,
                )
            ):
                _, _, prompt = (
                    user_text.partition(
                        "\n\n"
                    )
                )
                title = _alpaca_title_text(
                    "Generate a title for this prompt:"
                    + prompt
                )
                content = json.dumps(
                    {
                        "title": title,
                    },
                    separators=(",", ":"),
                )
                response = (
                    _ollama_chat_payload(
                        model=model_name,
                        content=content,
                        done=True,
                    )
                )
                if bool(
                    getattr(
                        gateway.config,
                        "trace_stages",
                        False,
                    )
                ):
                    print(
                        "[cortex-gateway:utility] "
                        f"ollama-title:bypass ({len(content)} chars)",
                        flush=True,
                    )
                return JSONResponse(
                    response
                )

            session_id = str(
                payload.get(
                    "cortex_session",
                    "default",
                )
            )
            result = gateway.chat(
                session_id,
                user_text,
                diagnostics=None,
            )
        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail=str(
                    exc
                ),
            ) from exc
        except RuntimeError as exc:
            raise HTTPException(
                status_code=502,
                detail=str(
                    exc
                ),
            ) from exc

        trace_http = bool(
            getattr(
                gateway.config,
                "trace_stages",
                False,
            )
        )
        stream = bool(
            payload.get(
                "stream",
                False,
            )
        )
        if trace_http:
            print(
                f"[cortex-gateway:{result.session_id}] "
                f"ollama:ready ({len(result.content)} chars, stream={stream})",
                flush=True,
            )

        if stream:
            return StreamingResponse(
                _ollama_stream_events(
                    result,
                    model=model_name,
                    trace=trace_http,
                ),
                media_type=(
                    "application/x-ndjson"
                ),
                headers={
                    "Cache-Control": (
                        "no-cache"
                    ),
                    "X-Accel-Buffering": (
                        "no"
                    ),
                },
            )

        return JSONResponse(
            _ollama_chat_payload(
                model=model_name,
                content=(
                    _ollama_visible_content(
                        result
                    )
                ),
                thinking=(
                    _cortex_thinking_text(
                        result
                    )
                ),
                done=True,
                metrics=(
                    _ollama_timing_metadata(
                        result
                    )
                ),
            )
        )

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

            if _alpaca_title_request(
                user_text
            ):
                title = _alpaca_title_text(
                    user_text
                )
                response_format = payload.get(
                    "response_format"
                )
                if isinstance(
                    response_format,
                    dict,
                ):
                    utility_content = json.dumps(
                        {
                            "title": title,
                            "emoji": "",
                        },
                        separators=(",", ":"),
                    )
                else:
                    utility_content = title
                if bool(
                    getattr(
                        gateway.config,
                        "trace_stages",
                        False,
                    )
                ):
                    print(
                        "[cortex-gateway:utility] "
                        f"title:bypass ({len(utility_content)} chars)",
                        flush=True,
                    )
                return JSONResponse(
                    _utility_chat_completion(
                        model=str(
                            payload.get(
                                "model",
                                public_model_name,
                            )
                        ),
                        content=utility_content,
                    )
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

        trace_http = bool(
            getattr(
                gateway.config,
                "trace_stages",
                False,
            )
        )
        if trace_http:
            print(
                f"[cortex-gateway:{result.session_id}] "
                f"response:ready ({len(result.content)} chars, "
                f"stream={bool(payload.get('stream', False))})",
                flush=True,
            )

        if payload.get("stream", False):
            return StreamingResponse(
                _openai_stream_events(
                    result,
                    trace=trace_http,
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
        "/cortex/sessions/{session_id}/bootstrap"
    )
    async def bootstrap_session(
        session_id: str,
        payload: dict[str, Any] = Body(...),
    ) -> dict[str, Any]:
        bootstrap = getattr(
            gateway,
            "bootstrap",
            None,
        )
        if not callable(
            bootstrap
        ):
            raise HTTPException(
                status_code=400,
                detail=(
                    "this gateway mode does not support neural bootstrap"
                ),
            )
        topics = payload.get(
            "topics",
            []
        )
        if (
            not isinstance(
                topics,
                list,
            )
            or not all(
                isinstance(
                    item,
                    str,
                )
                for item in topics
            )
        ):
            raise HTTPException(
                status_code=400,
                detail="topics must be an array of strings",
            )
        records = bootstrap(
            session_id,
            topics,
        )
        return {
            "session_id": session_id,
            "imported": len(
                records
            ),
            "records": [
                record.packet()
                for record in records
            ],
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
    hybrid_cognition: bool = False,
    hybrid_runtime: bool = False,
    parser_model: str | None = None,
    renderer_model: str | None = None,
    feature_dim: int = 128,
    trace_stages: bool = False,
    explain_responses: bool = False,
    max_revision_cycles: int = 1,
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
            trace_stages=bool(
                trace_stages
            ),
            explain_responses=bool(
                explain_responses
            ),
        )
        return StrictCortexLanguageGateway(
            strict_config
        )

    if hybrid_runtime:
        core_config = HybridCoreConfig(
            chat_model=model,
            parser_model=(
                parser_model
                or os.environ.get(
                    "CORTEX_PARSER_MODEL"
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
            trace_stages=bool(
                trace_stages
            ),
            include_diagnosis=False,
            max_revision_cycles=max(
                0,
                int(
                    max_revision_cycles
                ),
            ),
        )
        return HybridCoreGateway(
            core_config
        )

    if hybrid_cognition:
        hybrid_config = HybridGatewayConfig(
            chat_model=model,
            parser_model=(
                parser_model
                or os.environ.get(
                    "CORTEX_PARSER_MODEL"
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
            trace_stages=bool(
                trace_stages
            ),
            include_diagnosis=True,
        )
        return HybridCortexLanguageGateway(
            hybrid_config
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
        "--hybrid-cognition",
        action="store_true",
        help=(
            "Use Gateway v0.3: normal neural knowledge answers plus "
            "Cortex structural reasoning/diagnosis. Recommended for chat."
        ),
    )
    parser.add_argument(
        "--hybrid-runtime",
        action="store_true",
        help=(
            "Use Hybrid Core v0.1: provider-independent NeuralRuntime, "
            "Cortex pre-pass, fused answer+claim sidecar, and bounded revision."
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
        "--trace-stages",
        action="store_true",
        help=(
            "Print strict gateway parser/Cortex/renderer stage timings."
        ),
    )
    parser.add_argument(
        "--max-revision-cycles",
        type=int,
        default=1,
        help=(
            "Maximum Hybrid Core NN/Cortex conflict-revision cycles per turn."
        ),
    )
    parser.add_argument(
        "--explain-responses",
        action="store_true",
        help=(
            "Append a deterministic Cortex report describing parsed "
            "information, accepted evidence, bindings, decision, and runtime state."
        ),
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

    selected_modes = sum(
        (
            bool(
                args.strict_neural_boundary
            ),
            bool(
                args.hybrid_cognition
            ),
            bool(
                args.hybrid_runtime
            ),
        )
    )
    if selected_modes > 1:
        parser.error(
            "--strict-neural-boundary, --hybrid-cognition, and "
            "--hybrid-runtime are mutually exclusive"
        )

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
        hybrid_cognition=(
            args.hybrid_cognition
        ),
        hybrid_runtime=(
            args.hybrid_runtime
        ),
        parser_model=args.parser_model,
        renderer_model=(
            args.renderer_model
        ),
        feature_dim=args.feature_dim,
        trace_stages=(
            args.trace_stages
        ),
        explain_responses=(
            args.explain_responses
        ),
        max_revision_cycles=(
            args.max_revision_cycles
        ),
    )
    app = create_app(gateway)
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
    )


if __name__ == "__main__":
    main()
