"""Ollama-backed natural-language gateway for Cortex.

The gateway deliberately keeps language generation separate from structural
state. Ollama performs semantic extraction and final verbalization. Cortex
receives entity embeddings plus relation observations and owns the persistent
structural runtime state.

Diagnostics are opt-in. Diagnostic reports are derived from Cortex telemetry
and are never re-ingested as observations of the external world.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from json import dumps, loads
from time import time
from typing import Any, Mapping, Protocol, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from uuid import uuid4

try:
    from . import CortexRuntime
except Exception:  # pragma: no cover - native extension may be absent in docs tooling
    CortexRuntime = None  # type: ignore[assignment]


EXTRACTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "name": {"type": "string"},
                    "kind": {"type": "string"},
                    "description": {"type": "string"},
                },
                "required": ["id", "name", "kind", "description"],
            },
        },
        "relations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "source": {"type": "string"},
                    "target": {"type": "string"},
                    "label": {"type": "string"},
                    "asserted": {"type": "boolean"},
                },
                "required": ["source", "target", "label", "asserted"],
            },
        },
    },
    "required": ["entities", "relations"],
}


EXTRACTION_SYSTEM_PROMPT = """You are the semantic perception layer for Cortex.
Extract only entities and explicit relations supported by the current user
message. Do not infer unstated facts. Use short local IDs such as e0, e1 inside
this extraction only. Each relation source/target must reference an entity ID
from the same extraction. Mark asserted=false only when the message explicitly
negates the relation. Respond exactly to the provided JSON schema."""


ANSWER_SYSTEM_PROMPT = """You are the language articulation layer for Cortex.
Answer the user naturally and helpfully. A CORTEX CONTEXT packet may be supplied
as trusted internal state. Use it when relevant, but do not invent additional
Cortex beliefs, diagnoses, or confidence claims. If diagnostics are present,
you may explain them in natural language when useful. Never treat a diagnostic
self-report as an observation about the external world."""


class LanguageModelClient(Protocol):
    def chat(
        self,
        *,
        model: str,
        messages: Sequence[Mapping[str, str]],
        format_schema: Mapping[str, Any] | None = None,
    ) -> str:
        ...

    def embed(
        self,
        *,
        model: str,
        inputs: Sequence[str],
    ) -> list[list[float]]:
        ...


@dataclass(frozen=True)
class GatewayConfig:
    chat_model: str
    embedding_model: str = "embeddinggemma"
    ollama_url: str = "http://localhost:11434"
    max_entities: int = 96
    memory_budget: int = 24
    history_messages: int = 12
    diagnostics_default: bool = False


@dataclass(frozen=True)
class ExtractedEntity:
    local_id: str
    name: str
    kind: str
    description: str

    @property
    def embedding_text(self) -> str:
        description = self.description.strip()
        suffix = f". {description}" if description else ""
        return f"{self.kind}: {self.name}{suffix}"


@dataclass(frozen=True)
class ExtractedRelation:
    source: str
    target: str
    label: str
    asserted: bool


@dataclass(frozen=True)
class SemanticObservation:
    entities: tuple[ExtractedEntity, ...]
    relations: tuple[ExtractedRelation, ...]


@dataclass
class PersistentEntity:
    cortex_id: int
    names: set[str] = field(default_factory=set)
    kinds: set[str] = field(default_factory=set)
    descriptions: list[str] = field(default_factory=list)
    mentions: int = 0

    def absorb(self, entity: ExtractedEntity) -> None:
        if entity.name:
            self.names.add(entity.name)
        if entity.kind:
            self.kinds.add(entity.kind)
        description = entity.description.strip()
        if description and description not in self.descriptions:
            self.descriptions.append(description)
            self.descriptions[:] = self.descriptions[-4:]
        self.mentions += 1

    def packet(self) -> dict[str, Any]:
        return {
            "id": self.cortex_id,
            "names": sorted(self.names),
            "kinds": sorted(self.kinds),
            "descriptions": list(self.descriptions),
            "mentions": self.mentions,
        }


@dataclass(frozen=True)
class PersistentRelation:
    source_id: int
    target_id: int
    label: str
    asserted: bool

    def packet(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "label": self.label,
            "asserted": self.asserted,
        }


@dataclass(frozen=True)
class DiagnosticReport:
    status: str
    summary: str
    signals: tuple[str, ...]
    metrics: Mapping[str, Any]

    def packet(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "summary": self.summary,
            "signals": list(self.signals),
            "metrics": dict(self.metrics),
        }


@dataclass(frozen=True)
class ChatResult:
    response_id: str
    session_id: str
    model: str
    content: str
    cortex_context: Mapping[str, Any]
    diagnostics: DiagnosticReport | None


class OllamaClient:
    """Minimal dependency-free client for Ollama's native chat/embed API."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        *,
        timeout: float = 120.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = float(timeout)

    def _post(
        self,
        path: str,
        payload: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        body = dumps(payload).encode("utf-8")
        request = Request(
            self.base_url + path,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                raw = response.read().decode("utf-8")
        except HTTPError as exc:  # pragma: no cover - network integration
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Ollama HTTP {exc.code}: {detail}"
            ) from exc
        except URLError as exc:  # pragma: no cover - network integration
            raise RuntimeError(
                f"Cannot reach Ollama at {self.base_url}: {exc.reason}"
            ) from exc

        result = loads(raw)
        if not isinstance(result, dict):
            raise RuntimeError("Ollama returned a non-object response")
        return result

    def chat(
        self,
        *,
        model: str,
        messages: Sequence[Mapping[str, str]],
        format_schema: Mapping[str, Any] | None = None,
    ) -> str:
        payload: dict[str, Any] = {
            "model": model,
            "messages": [dict(message) for message in messages],
            "stream": False,
        }
        if format_schema is not None:
            payload["format"] = dict(format_schema)
            payload["options"] = {"temperature": 0}
        result = self._post("/api/chat", payload)
        message = result.get("message")
        if not isinstance(message, dict):
            raise RuntimeError("Ollama chat response has no message")
        content = message.get("content")
        if not isinstance(content, str):
            raise RuntimeError("Ollama chat response has no text content")
        return content

    def embed(
        self,
        *,
        model: str,
        inputs: Sequence[str],
    ) -> list[list[float]]:
        if not inputs:
            return []
        result = self._post(
            "/api/embed",
            {
                "model": model,
                "input": list(inputs),
            },
        )
        embeddings = result.get("embeddings")
        if not isinstance(embeddings, list):
            raise RuntimeError("Ollama embed response has no embeddings")
        vectors: list[list[float]] = []
        for vector in embeddings:
            if not isinstance(vector, list):
                raise RuntimeError("Ollama returned malformed embedding")
            vectors.append([float(value) for value in vector])
        if len(vectors) != len(inputs):
            raise RuntimeError("Ollama embedding count does not match inputs")
        return vectors


def parse_semantic_observation(raw: str) -> SemanticObservation:
    try:
        payload = loads(raw)
    except Exception as exc:
        raise ValueError("semantic extraction is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("semantic extraction must be an object")

    raw_entities = payload.get("entities", [])
    raw_relations = payload.get("relations", [])
    if not isinstance(raw_entities, list) or not isinstance(raw_relations, list):
        raise ValueError("semantic extraction arrays are malformed")

    entities: list[ExtractedEntity] = []
    seen_ids: set[str] = set()
    for item in raw_entities:
        if not isinstance(item, dict):
            raise ValueError("entity extraction item must be an object")
        entity = ExtractedEntity(
            local_id=str(item.get("id", "")).strip(),
            name=str(item.get("name", "")).strip(),
            kind=str(item.get("kind", "")).strip(),
            description=str(item.get("description", "")).strip(),
        )
        if not entity.local_id or entity.local_id in seen_ids:
            raise ValueError("entity IDs must be unique and non-empty")
        seen_ids.add(entity.local_id)
        entities.append(entity)

    relations: list[ExtractedRelation] = []
    for item in raw_relations:
        if not isinstance(item, dict):
            raise ValueError("relation extraction item must be an object")
        relation = ExtractedRelation(
            source=str(item.get("source", "")).strip(),
            target=str(item.get("target", "")).strip(),
            label=str(item.get("label", "")).strip(),
            asserted=bool(item.get("asserted", True)),
        )
        if (
            relation.source not in seen_ids
            or relation.target not in seen_ids
            or not relation.label
        ):
            raise ValueError("relation references unknown entity or empty label")
        relations.append(relation)

    return SemanticObservation(
        entities=tuple(entities),
        relations=tuple(relations),
    )


def diagnose_runtime(read: Any, *, memory_budget: int) -> DiagnosticReport:
    memberships = [float(value) for value in getattr(read, "memberships", [])]
    max_membership = max(memberships, default=0.0)
    nearest_dist = float(getattr(read, "nearest_dist", 0.0))
    unresolved = bool(getattr(read, "unresolved", False))
    memory_unresolved = bool(getattr(read, "memory_unresolved", False))
    budget_pressure = bool(getattr(read, "budget_pressure", False))
    discovered = bool(getattr(read, "discovered", False))
    revision = bool(getattr(read, "revision", False))
    reactivated = bool(getattr(read, "reactivated", False))
    stored = int(getattr(read, "stored", 0))
    memory_stored = int(getattr(read, "memory_stored", 0))
    subgroup_margin = float(getattr(read, "subgroup_margin", 0.0))
    relation_changed = len(getattr(read, "relation_changed", []))
    causal_changed = len(getattr(read, "causal_changed", []))

    signals: list[str] = []
    if unresolved:
        signals.append("current structural state is unresolved")
    if memory_unresolved:
        signals.append("historical memory reports unresolved retrieval")
    if budget_pressure:
        signals.append("continual memory is under budget pressure")
    if discovered:
        signals.append("the latest step discovered a new persistent state")
    if revision:
        signals.append("the latest step revised the active state")
    if reactivated:
        signals.append("a previously stored state was reactivated")
    if relation_changed:
        signals.append(f"{relation_changed} relation evidence cells changed")
    if causal_changed:
        signals.append(f"{causal_changed} causal evidence cells changed")
    if max_membership < 0.5 and memberships:
        signals.append("prototype membership is diffuse")
    if subgroup_margin < 0.05:
        signals.append("nuisance-subgroup evidence has a small margin")

    if unresolved or memory_unresolved:
        status = "uncertain"
    elif budget_pressure or max_membership < 0.5:
        status = "caution"
    else:
        status = "stable"

    if status == "stable":
        summary = (
            "I am currently structurally stable: the latest observation fits "
            "my active state without an unresolved conflict."
        )
    elif status == "caution":
        summary = (
            "I can continue, but my current state deserves caution because "
            "one or more confidence or resource signals are weak."
        )
    else:
        summary = (
            "I am currently uncertain about the latest structural state and "
            "should avoid presenting that uncertainty as a settled fact."
        )

    return DiagnosticReport(
        status=status,
        summary=summary,
        signals=tuple(signals),
        metrics={
            "prediction": float(getattr(read, "prediction", 0.0)),
            "max_membership": max_membership,
            "nearest_distance": nearest_dist,
            "subgroup_margin": subgroup_margin,
            "stored_states": stored,
            "memory_stored": memory_stored,
            "memory_budget": int(memory_budget),
            "budget_pressure": budget_pressure,
            "unresolved": unresolved,
            "memory_unresolved": memory_unresolved,
            "revision": revision,
            "reactivated": reactivated,
            "discovered": discovered,
            "relation_changes": relation_changed,
            "causal_changes": causal_changed,
        },
    )


class CortexChatSession:
    def __init__(
        self,
        session_id: str,
        config: GatewayConfig,
        client: LanguageModelClient,
    ) -> None:
        self.session_id = session_id
        self.config = config
        self.client = client
        self.diagnostics_enabled = bool(config.diagnostics_default)
        self.runtime: Any | None = None
        self.feature_dim: int | None = None
        self.history: list[dict[str, str]] = []
        self.entities: dict[int, PersistentEntity] = {}
        self.relations: dict[tuple[int, int, str], PersistentRelation] = {}
        self.last_read: Any | None = None

    def set_diagnostics(self, enabled: bool) -> None:
        self.diagnostics_enabled = bool(enabled)

    def _extract(self, user_text: str) -> SemanticObservation:
        raw = self.client.chat(
            model=self.config.chat_model,
            messages=[
                {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        "Extract the explicitly stated semantic observations "
                        "from this message as JSON:\n\n" + user_text
                    ),
                },
            ],
            format_schema=EXTRACTION_SCHEMA,
        )
        return parse_semantic_observation(raw)

    def _ensure_runtime(self, feature_dim: int) -> Any:
        if self.runtime is not None:
            if self.feature_dim != feature_dim:
                raise RuntimeError(
                    "embedding dimension changed inside an active Cortex session"
                )
            return self.runtime
        if CortexRuntime is None:
            raise RuntimeError(
                "Cortex native runtime is unavailable; build/install the Rust extension"
            )
        self.feature_dim = int(feature_dim)
        self.runtime = CortexRuntime(
            feature_dim=self.feature_dim,
            max_entities=self.config.max_entities,
            budget=self.config.memory_budget,
        )
        return self.runtime

    def _observe(self, observation: SemanticObservation) -> Any | None:
        if not observation.entities:
            return self.last_read

        vectors = self.client.embed(
            model=self.config.embedding_model,
            inputs=[entity.embedding_text for entity in observation.entities],
        )
        if not vectors:
            return self.last_read
        dimension = len(vectors[0])
        if dimension <= 0 or any(len(vector) != dimension for vector in vectors):
            raise RuntimeError("embedding vectors must share a positive dimension")

        runtime = self._ensure_runtime(dimension)
        local_index = {
            entity.local_id: index
            for index, entity in enumerate(observation.entities)
        }
        relation_obs = [
            (
                local_index[relation.source],
                local_index[relation.target],
                1 if relation.asserted else 0,
            )
            for relation in observation.relations
        ]
        read = runtime.step(
            vectors,
            relation_obs,
            None,
            [],
            None,
        )
        self.last_read = read

        bindings = [int(value) for value in read.bindings]
        for entity, cortex_id in zip(observation.entities, bindings):
            persistent = self.entities.setdefault(
                cortex_id,
                PersistentEntity(cortex_id=cortex_id),
            )
            persistent.absorb(entity)

        for relation in observation.relations:
            source_index = local_index[relation.source]
            target_index = local_index[relation.target]
            if source_index >= len(bindings) or target_index >= len(bindings):
                continue
            persistent_relation = PersistentRelation(
                source_id=bindings[source_index],
                target_id=bindings[target_index],
                label=relation.label,
                asserted=relation.asserted,
            )
            key = (
                persistent_relation.source_id,
                persistent_relation.target_id,
                persistent_relation.label,
            )
            self.relations[key] = persistent_relation

        return read

    def context_packet(self, *, diagnostics: DiagnosticReport | None) -> dict[str, Any]:
        recent_entities = sorted(
            self.entities.values(),
            key=lambda entity: (-entity.mentions, entity.cortex_id),
        )[:12]
        recent_relations = list(self.relations.values())[-16:]
        packet: dict[str, Any] = {
            "session_id": self.session_id,
            "entities": [entity.packet() for entity in recent_entities],
            "relations": [relation.packet() for relation in recent_relations],
        }
        if self.last_read is not None:
            packet["runtime"] = {
                "prediction": float(self.last_read.prediction),
                "unresolved": bool(self.last_read.unresolved),
                "revision": bool(self.last_read.revision),
                "reactivated": bool(self.last_read.reactivated),
                "discovered": bool(self.last_read.discovered),
                "budget_pressure": bool(self.last_read.budget_pressure),
                "stored": int(self.last_read.stored),
            }
        if diagnostics is not None:
            packet["diagnostics"] = diagnostics.packet()
        return packet

    def chat(
        self,
        user_text: str,
        *,
        diagnostics: bool | None = None,
    ) -> ChatResult:
        text = user_text.strip()
        if not text:
            raise ValueError("user message must not be empty")

        observation = self._extract(text)
        read = self._observe(observation)
        diagnostic_enabled = (
            self.diagnostics_enabled
            if diagnostics is None
            else bool(diagnostics)
        )
        diagnostic = (
            diagnose_runtime(
                read,
                memory_budget=self.config.memory_budget,
            )
            if diagnostic_enabled and read is not None
            else None
        )
        packet = self.context_packet(diagnostics=diagnostic)

        context_message = {
            "role": "system",
            "content": "CORTEX CONTEXT\n" + dumps(packet, sort_keys=True),
        }
        messages = [
            {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
            context_message,
            *self.history[-self.config.history_messages :],
            {"role": "user", "content": text},
        ]
        content = self.client.chat(
            model=self.config.chat_model,
            messages=messages,
        )

        self.history.extend(
            [
                {"role": "user", "content": text},
                {"role": "assistant", "content": content},
            ]
        )
        self.history[:] = self.history[-self.config.history_messages :]

        return ChatResult(
            response_id="chatcmpl-" + uuid4().hex,
            session_id=self.session_id,
            model=self.config.chat_model,
            content=content,
            cortex_context=packet,
            diagnostics=diagnostic,
        )


class CortexLanguageGateway:
    def __init__(
        self,
        config: GatewayConfig,
        *,
        client: LanguageModelClient | None = None,
    ) -> None:
        self.config = config
        self.client = client or OllamaClient(config.ollama_url)
        self.sessions: dict[str, CortexChatSession] = {}

    def session(self, session_id: str) -> CortexChatSession:
        normalized = session_id.strip()
        if not normalized:
            raise ValueError("session_id must not be empty")
        session = self.sessions.get(normalized)
        if session is None:
            session = CortexChatSession(
                normalized,
                self.config,
                self.client,
            )
            self.sessions[normalized] = session
        return session

    def reset(self, session_id: str) -> None:
        self.sessions.pop(session_id, None)

    def set_diagnostics(self, session_id: str, enabled: bool) -> None:
        self.session(session_id).set_diagnostics(enabled)

    def chat(
        self,
        session_id: str,
        user_text: str,
        *,
        diagnostics: bool | None = None,
    ) -> ChatResult:
        return self.session(session_id).chat(
            user_text,
            diagnostics=diagnostics,
        )


def openai_chat_completion(result: ChatResult) -> dict[str, Any]:
    created = int(time())
    return {
        "id": result.response_id,
        "object": "chat.completion",
        "created": created,
        "model": result.model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": result.content,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
        "cortex": {
            "session_id": result.session_id,
            "diagnostics": (
                result.diagnostics.packet()
                if result.diagnostics is not None
                else None
            ),
        },
    }
