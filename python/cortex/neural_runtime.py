"""Provider-independent neural runtime for Cortex hybrid cognition.

The hybrid architecture owns orchestration. Backends own model execution.

The runtime intentionally exposes capabilities instead of assuming every
provider can supply hidden states, KV reuse, adapters, or training. Ollama is
the first backend, but it is not the architectural boundary.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from json import dumps, loads
from time import perf_counter
from typing import Any, Mapping, Protocol, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class NeuralCapabilities:
    generate: bool = True
    structured_generate: bool = True
    stream: bool = False
    embeddings: bool = False
    logits: bool = False
    hidden_states: bool = False
    kv_cache: bool = False
    adapters: bool = False
    train_adapters: bool = False

    def packet(self) -> dict[str, bool]:
        return {
            key: bool(value)
            for key, value
            in self.__dict__.items()
        }


@dataclass(frozen=True)
class NeuralResponse:
    text: str
    model: str
    backend: str
    duration_s: float
    structured: bool = False
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    def packet(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "backend": self.backend,
            "duration_s": self.duration_s,
            "structured": self.structured,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class NeuralRequest:
    """One orchestrator-controlled neural execution."""

    stage: str
    model: str
    messages: Sequence[
        Mapping[str, str]
    ]
    schema: Mapping[
        str,
        Any,
    ] | None = None
    temperature: float | None = None
    exchange_id: str | None = None
    cache_policy: str = "reuse_if_supported"
    adapter: str | None = None

    def packet(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "model": self.model,
            "structured": (
                self.schema
                is not None
            ),
            "exchange_id": (
                self.exchange_id
            ),
            "cache_policy": (
                self.cache_policy
            ),
            "adapter": self.adapter,
            "message_count": len(
                self.messages
            ),
        }


@dataclass(frozen=True)
class NeuralCall:
    request: NeuralRequest
    response: NeuralResponse

    def packet(self) -> dict[str, Any]:
        packet = (
            self.response.packet()
        )
        packet[
            "stage"
        ] = (
            self.request.stage
        )
        packet[
            "exchange_id"
        ] = (
            self.request.exchange_id
        )
        packet[
            "cache_policy"
        ] = (
            self.request.cache_policy
        )
        packet[
            "adapter"
        ] = (
            self.request.adapter
        )
        return packet


class NeuralSession:
    """Stateful execution boundary owned by the hybrid orchestrator.

    v0.1 delegates execution to NeuralRuntime while recording explicit
    per-exchange requests. Future backends can override this boundary to keep
    KV state, active adapters, or other model-local state alive across calls.
    """

    def __init__(
        self,
        *,
        session_id: str,
        runtime: "NeuralRuntime",
    ) -> None:
        self.session_id = session_id
        self.runtime = runtime
        self.active_exchange_id: (
            str
            | None
        ) = None
        self.calls: list[
            NeuralCall
        ] = []

    @property
    def capabilities(
        self,
    ) -> NeuralCapabilities:
        return (
            self.runtime.capabilities
        )

    @property
    def backend_name(
        self,
    ) -> str:
        return (
            self.runtime.backend_name
        )

    def begin_exchange(
        self,
        exchange_id: str,
    ) -> None:
        self.active_exchange_id = (
            exchange_id
        )
        self.calls = []

    def execute(
        self,
        request: NeuralRequest,
    ) -> NeuralResponse:
        if (
            request.schema
            is not None
            and not self.capabilities.structured_generate
        ):
            raise RuntimeError(
                "neural backend does not support structured generation"
            )
        if (
            request.adapter
            is not None
            and not self.capabilities.adapters
        ):
            raise RuntimeError(
                "neural backend does not support adapter activation"
            )
        exchange_id = (
            request.exchange_id
            or self.active_exchange_id
        )
        normalized = NeuralRequest(
            stage=request.stage,
            model=request.model,
            messages=request.messages,
            schema=request.schema,
            temperature=(
                request.temperature
            ),
            exchange_id=(
                exchange_id
            ),
            cache_policy=(
                request.cache_policy
            ),
            adapter=request.adapter,
        )
        response = (
            self.runtime.generate(
                model=(
                    normalized.model
                ),
                messages=(
                    normalized.messages
                ),
                schema=(
                    normalized.schema
                ),
                temperature=(
                    normalized.temperature
                ),
            )
        )
        self.calls.append(
            NeuralCall(
                request=normalized,
                response=response,
            )
        )
        return response

    def packet(self) -> dict[str, Any]:
        return {
            "session_id": (
                self.session_id
            ),
            "backend": (
                self.backend_name
            ),
            "capabilities": (
                self.capabilities.packet()
            ),
            "active_exchange_id": (
                self.active_exchange_id
            ),
            "neural_call_count": len(
                self.calls
            ),
            "neural_calls": [
                call.packet()
                for call
                in self.calls
            ],
        }


class NeuralRuntime(Protocol):
    @property
    def capabilities(
        self,
    ) -> NeuralCapabilities:
        ...

    @property
    def backend_name(
        self,
    ) -> str:
        ...

    def generate(
        self,
        *,
        model: str,
        messages: Sequence[
            Mapping[str, str]
        ],
        schema: Mapping[str, Any] | None = None,
        temperature: float | None = None,
    ) -> NeuralResponse:
        ...

    def embed(
        self,
        *,
        model: str,
        inputs: Sequence[str],
    ) -> list[list[float]]:
        ...


class OllamaNeuralRuntime:
    """Ollama implementation of the provider-independent runtime."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        *,
        timeout: float = 120.0,
    ) -> None:
        normalized = base_url.strip()
        if "://" not in normalized:
            normalized = (
                "http://" + normalized
            )
        self.base_url = (
            normalized.rstrip("/")
        )
        self.timeout = float(timeout)

    @property
    def backend_name(
        self,
    ) -> str:
        return "ollama"

    @property
    def capabilities(
        self,
    ) -> NeuralCapabilities:
        return NeuralCapabilities(
            generate=True,
            structured_generate=True,
            stream=False,
            embeddings=True,
            logits=False,
            hidden_states=False,
            kv_cache=False,
            adapters=False,
            train_adapters=False,
        )

    def _post(
        self,
        path: str,
        payload: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        request = Request(
            self.base_url + path,
            data=dumps(
                payload
            ).encode(
                "utf-8"
            ),
            headers={
                "Content-Type": (
                    "application/json"
                )
            },
            method="POST",
        )
        try:
            with urlopen(
                request,
                timeout=self.timeout,
            ) as response:
                raw = (
                    response.read().decode(
                        "utf-8"
                    )
                )
        except HTTPError as exc:
            detail = exc.read().decode(
                "utf-8",
                errors="replace",
            )
            raise RuntimeError(
                f"Ollama HTTP {exc.code}: {detail}"
            ) from exc
        except URLError as exc:
            raise RuntimeError(
                "Cannot reach Ollama at "
                + self.base_url
                + ": "
                + str(
                    exc.reason
                )
            ) from exc

        payload_out = loads(
            raw
        )
        if not isinstance(
            payload_out,
            dict,
        ):
            raise RuntimeError(
                "Ollama returned a non-object response"
            )
        return payload_out

    def generate(
        self,
        *,
        model: str,
        messages: Sequence[
            Mapping[str, str]
        ],
        schema: Mapping[
            str,
            Any,
        ]
        | None = None,
        temperature: float | None = None,
    ) -> NeuralResponse:
        request: dict[
            str,
            Any,
        ] = {
            "model": model,
            "messages": [
                dict(
                    message
                )
                for message
                in messages
            ],
            "stream": False,
        }
        options: dict[
            str,
            Any,
        ] = {}
        if schema is not None:
            request[
                "format"
            ] = dict(
                schema
            )
            # Structured translation is not a reasoning stage.
            request[
                "think"
            ] = False
            options[
                "temperature"
            ] = 0
        elif temperature is not None:
            options[
                "temperature"
            ] = float(
                temperature
            )
        if options:
            request[
                "options"
            ] = options

        started = perf_counter()
        result = self._post(
            "/api/chat",
            request,
        )
        duration = (
            perf_counter()
            - started
        )
        message = result.get(
            "message"
        )
        if not isinstance(
            message,
            dict,
        ):
            raise RuntimeError(
                "Ollama chat response has no message"
            )
        text = message.get(
            "content"
        )
        if not isinstance(
            text,
            str,
        ):
            raise RuntimeError(
                "Ollama chat response has no text content"
            )

        prompt_tokens = result.get(
            "prompt_eval_count"
        )
        completion_tokens = result.get(
            "eval_count"
        )
        return NeuralResponse(
            text=text,
            model=model,
            backend=self.backend_name,
            duration_s=duration,
            structured=(
                schema
                is not None
            ),
            prompt_tokens=(
                int(
                    prompt_tokens
                )
                if isinstance(
                    prompt_tokens,
                    int,
                )
                else None
            ),
            completion_tokens=(
                int(
                    completion_tokens
                )
                if isinstance(
                    completion_tokens,
                    int,
                )
                else None
            ),
            metadata={
                "total_duration_ns": (
                    result.get(
                        "total_duration"
                    )
                ),
                "load_duration_ns": (
                    result.get(
                        "load_duration"
                    )
                ),
            },
        )

    def embed(
        self,
        *,
        model: str,
        inputs: Sequence[str],
    ) -> list[
        list[
            float
        ]
    ]:
        if not inputs:
            return []
        result = self._post(
            "/api/embed",
            {
                "model": model,
                "input": list(
                    inputs
                ),
            },
        )
        embeddings = result.get(
            "embeddings"
        )
        if not isinstance(
            embeddings,
            list,
        ):
            raise RuntimeError(
                "Ollama embed response has no embeddings"
            )
        vectors: list[
            list[
                float
            ]
        ] = []
        for vector in embeddings:
            if not isinstance(
                vector,
                list,
            ):
                raise RuntimeError(
                    "Ollama returned malformed embedding"
                )
            vectors.append(
                [
                    float(
                        value
                    )
                    for value
                    in vector
                ]
            )
        if len(
            vectors
        ) != len(
            inputs
        ):
            raise RuntimeError(
                "Ollama embedding count does not match inputs"
            )
        return vectors


class RuntimeClientAdapter:
    """Compatibility adapter for existing Cortex parser/session code."""

    def __init__(
        self,
        runtime: NeuralRuntime,
    ) -> None:
        self.runtime = runtime

    def chat(
        self,
        *,
        model: str,
        messages: Sequence[
            Mapping[str, str]
        ],
        format_schema: Mapping[
            str,
            Any,
        ]
        | None = None,
    ) -> str:
        return self.runtime.generate(
            model=model,
            messages=messages,
            schema=format_schema,
        ).text

    def embed(
        self,
        *,
        model: str,
        inputs: Sequence[str],
    ) -> list[list[float]]:
        return self.runtime.embed(
            model=model,
            inputs=inputs,
        )
