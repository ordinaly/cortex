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
from os import replace as atomic_replace
from pathlib import Path
from tempfile import NamedTemporaryFile
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
                    "evidence": {"type": "string"},
                },
                "required": ["id", "name", "kind", "description", "evidence"],
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
                    "evidence": {"type": "string"},
                },
                "required": ["source", "target", "label", "asserted", "evidence"],
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
negates the relation. For every entity and relation, include a short evidence
field copied verbatim from the current user message that grounds the extraction.
Do not output an item if you cannot quote supporting text. Respond exactly to
the provided JSON schema."""


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
    persistence_dir: str | None = None


@dataclass(frozen=True)
class ExtractedEntity:
    local_id: str
    name: str
    kind: str
    description: str
    evidence: str

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
    evidence: str


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
    evidence_snippets: list[str] = field(default_factory=list)
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
        evidence = entity.evidence.strip()
        if evidence and evidence not in self.evidence_snippets:
            self.evidence_snippets.append(evidence)
            self.evidence_snippets[:] = self.evidence_snippets[-4:]
        self.mentions += 1

    def packet(self) -> dict[str, Any]:
        return {
            "id": self.cortex_id,
            "names": sorted(self.names),
            "kinds": sorted(self.kinds),
            "descriptions": list(self.descriptions),
            "evidence_snippets": list(self.evidence_snippets),
            "provenance": "user_message",
            "extractor": "ollama_structured",
            "mentions": self.mentions,
        }


@dataclass(frozen=True)
class PersistentRelation:
    source_id: int
    target_id: int
    label: str
    asserted: bool
    evidence: str

    def packet(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "label": self.label,
            "asserted": self.asserted,
            "evidence": self.evidence,
            "provenance": "user_message",
            "extractor": "ollama_structured",
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


@dataclass(frozen=True)
class ObservationFrame:
    entities: tuple[ExtractedEntity, ...]
    relations: tuple[ExtractedRelation, ...]
    vectors: tuple[tuple[float, ...], ...]

    def packet(self) -> dict[str, Any]:
        return {
            "entities": [
                {
                    "id": entity.local_id,
                    "name": entity.name,
                    "kind": entity.kind,
                    "description": entity.description,
                    "evidence": entity.evidence,
                }
                for entity in self.entities
            ],
            "relations": [
                {
                    "source": relation.source,
                    "target": relation.target,
                    "label": relation.label,
                    "asserted": relation.asserted,
                    "evidence": relation.evidence,
                }
                for relation in self.relations
            ],
            "vectors": [
                list(vector)
                for vector in self.vectors
            ],
        }

    @classmethod
    def from_packet(
        cls,
        payload: Mapping[str, Any],
    ) -> "ObservationFrame":
        entities = tuple(
            ExtractedEntity(
                local_id=str(item["id"]),
                name=str(item["name"]),
                kind=str(item["kind"]),
                description=str(
                    item.get("description", "")
                ),
                evidence=str(
                    item.get("evidence", "")
                ),
            )
            for item in payload.get(
                "entities",
                [],
            )
        )
        relations = tuple(
            ExtractedRelation(
                source=str(item["source"]),
                target=str(item["target"]),
                label=str(item["label"]),
                asserted=bool(
                    item.get("asserted", True)
                ),
                evidence=str(
                    item.get("evidence", "")
                ),
            )
            for item in payload.get(
                "relations",
                [],
            )
        )
        vectors = tuple(
            tuple(float(value) for value in vector)
            for vector in payload.get(
                "vectors",
                [],
            )
        )
        if len(vectors) != len(entities):
            raise ValueError(
                "persisted frame vector/entity count mismatch"
            )
        return cls(
            entities=entities,
            relations=relations,
            vectors=vectors,
        )


class FileSessionStore:
    SCHEMA = "cortex-language-session-v1"

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser()
        self.root.mkdir(
            parents=True,
            exist_ok=True,
        )

    @staticmethod
    def _safe_name(
        session_id: str,
    ) -> str:
        from hashlib import sha256

        digest = sha256(
            session_id.encode("utf-8")
        ).hexdigest()[:16]
        readable = "".join(
            ch
            if ch.isalnum()
            or ch in ("-", "_")
            else "_"
            for ch in session_id
        ).strip("_")[:48]
        return (
            f"{readable or 'session'}-{digest}.json"
        )

    def path_for(
        self,
        session_id: str,
    ) -> Path:
        return self.root / self._safe_name(
            session_id
        )

    def exists(
        self,
        session_id: str,
    ) -> bool:
        return self.path_for(
            session_id
        ).is_file()

    def list_sessions(self) -> list[str]:
        sessions: list[str] = []
        for path in sorted(
            self.root.glob("*.json")
        ):
            try:
                payload = loads(
                    path.read_text(
                        encoding="utf-8"
                    )
                )
            except Exception:
                continue
            if (
                isinstance(payload, dict)
                and payload.get("schema")
                == self.SCHEMA
                and isinstance(
                    payload.get("session_id"),
                    str,
                )
            ):
                sessions.append(
                    payload["session_id"]
                )
        return sessions

    def save(
        self,
        session: "CortexChatSession",
    ) -> Path:
        payload = session.persistence_packet()
        target = self.path_for(
            session.session_id
        )
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=str(self.root),
            prefix=".cortex-session-",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(
                dumps(
                    payload,
                    sort_keys=True,
                    indent=2,
                )
            )
            handle.write("\n")
            temporary = Path(handle.name)
        atomic_replace(
            temporary,
            target,
        )
        return target

    def load_packet(
        self,
        session_id: str,
    ) -> Mapping[str, Any]:
        path = self.path_for(
            session_id
        )
        if not path.is_file():
            raise FileNotFoundError(
                f"no persisted Cortex session: {session_id}"
            )
        payload = loads(
            path.read_text(
                encoding="utf-8"
            )
        )
        if not isinstance(payload, dict):
            raise ValueError(
                "persisted Cortex session must be an object"
            )
        if payload.get("schema") != self.SCHEMA:
            raise ValueError(
                "unsupported Cortex session schema"
            )
        if payload.get("session_id") != session_id:
            raise ValueError(
                "persisted Cortex session ID mismatch"
            )
        return payload

    def delete(
        self,
        session_id: str,
    ) -> bool:
        path = self.path_for(
            session_id
        )
        if not path.exists():
            return False
        path.unlink()
        return True


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


def _normalized_text(value: str) -> str:
    return " ".join(value.casefold().split())


def _evidence_is_anchored(evidence: str, source_text: str | None) -> bool:
    if source_text is None:
        return True
    needle = _normalized_text(evidence)
    haystack = _normalized_text(source_text)
    return bool(needle) and needle in haystack


def parse_semantic_observation(
    raw: str,
    *,
    source_text: str | None = None,
) -> SemanticObservation:
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
            evidence=str(item.get("evidence", "")).strip(),
        )
        if not entity.local_id or entity.local_id in seen_ids:
            raise ValueError("entity IDs must be unique and non-empty")
        if not _evidence_is_anchored(entity.evidence, source_text):
            raise ValueError("entity evidence is not anchored in the source message")
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
            evidence=str(item.get("evidence", "")).strip(),
        )
        if not _evidence_is_anchored(relation.evidence, source_text):
            raise ValueError("relation evidence is not anchored in the source message")
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
        self.frames: list[ObservationFrame] = []
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
        return parse_semantic_observation(
            raw,
            source_text=user_text,
        )

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

    def _apply_frame(
        self,
        frame: ObservationFrame,
        *,
        record: bool,
    ) -> Any | None:
        if not frame.entities:
            if record:
                self.frames.append(
                    frame
                )
            return self.last_read

        vectors = [
            list(vector)
            for vector in frame.vectors
        ]
        dimension = len(vectors[0])
        if (
            dimension <= 0
            or any(
                len(vector) != dimension
                for vector in vectors
            )
        ):
            raise RuntimeError(
                "embedding vectors must share a positive dimension"
            )

        runtime = self._ensure_runtime(
            dimension
        )
        local_index = {
            entity.local_id: index
            for index, entity
            in enumerate(frame.entities)
        }
        relation_obs = [
            (
                local_index[
                    relation.source
                ],
                local_index[
                    relation.target
                ],
                1
                if relation.asserted
                else 0,
            )
            for relation in frame.relations
        ]
        read = runtime.step(
            vectors,
            relation_obs,
            None,
            [],
            None,
        )
        self.last_read = read

        bindings = [
            int(value)
            for value in read.bindings
        ]
        for entity, cortex_id in zip(
            frame.entities,
            bindings,
        ):
            persistent = (
                self.entities.setdefault(
                    cortex_id,
                    PersistentEntity(
                        cortex_id=cortex_id
                    ),
                )
            )
            persistent.absorb(
                entity
            )

        for relation in frame.relations:
            source_index = local_index[
                relation.source
            ]
            target_index = local_index[
                relation.target
            ]
            if (
                source_index >= len(bindings)
                or target_index
                >= len(bindings)
            ):
                continue
            persistent_relation = (
                PersistentRelation(
                    source_id=bindings[
                        source_index
                    ],
                    target_id=bindings[
                        target_index
                    ],
                    label=relation.label,
                    asserted=(
                        relation.asserted
                    ),
                    evidence=(
                        relation.evidence
                    ),
                )
            )
            key = (
                persistent_relation.source_id,
                persistent_relation.target_id,
                persistent_relation.label,
            )
            self.relations[
                key
            ] = persistent_relation

        if record:
            self.frames.append(
                frame
            )
        return read

    def _observe(
        self,
        observation: SemanticObservation,
    ) -> Any | None:
        if not observation.entities:
            frame = ObservationFrame(
                entities=observation.entities,
                relations=observation.relations,
                vectors=tuple(),
            )
            return self._apply_frame(
                frame,
                record=True,
            )

        vectors = self.client.embed(
            model=self.config.embedding_model,
            inputs=[
                entity.embedding_text
                for entity
                in observation.entities
            ],
        )
        frame = ObservationFrame(
            entities=observation.entities,
            relations=observation.relations,
            vectors=tuple(
                tuple(
                    float(value)
                    for value in vector
                )
                for vector in vectors
            ),
        )
        return self._apply_frame(
            frame,
            record=True,
        )

    def persistence_packet(
        self,
    ) -> dict[str, Any]:
        return {
            "schema": FileSessionStore.SCHEMA,
            "session_id": self.session_id,
            "chat_model": (
                self.config.chat_model
            ),
            "embedding_model": (
                self.config.embedding_model
            ),
            "feature_dim": self.feature_dim,
            "diagnostics_enabled": (
                self.diagnostics_enabled
            ),
            "history": [
                dict(message)
                for message in self.history
            ],
            "frames": [
                frame.packet()
                for frame in self.frames
            ],
        }

    def restore_packet(
        self,
        payload: Mapping[str, Any],
    ) -> None:
        chat_model = payload.get(
            "chat_model"
        )
        embedding_model = payload.get(
            "embedding_model"
        )
        if (
            chat_model
            and chat_model
            != self.config.chat_model
        ):
            raise ValueError(
                "persisted chat model does not match gateway configuration"
            )
        if (
            embedding_model
            and embedding_model
            != self.config.embedding_model
        ):
            raise ValueError(
                "persisted embedding model does not match gateway configuration"
            )

        history = payload.get(
            "history",
            [],
        )
        if not isinstance(
            history,
            list,
        ):
            raise ValueError(
                "persisted history must be an array"
            )

        frames_payload = payload.get(
            "frames",
            [],
        )
        if not isinstance(
            frames_payload,
            list,
        ):
            raise ValueError(
                "persisted frames must be an array"
            )

        self.runtime = None
        self.feature_dim = None
        self.entities.clear()
        self.relations.clear()
        self.frames.clear()
        self.last_read = None
        self.history = [
            {
                "role": str(
                    message.get(
                        "role",
                        "",
                    )
                ),
                "content": str(
                    message.get(
                        "content",
                        "",
                    )
                ),
            }
            for message in history
            if isinstance(
                message,
                dict,
            )
        ][
            -self.config.history_messages:
        ]
        self.diagnostics_enabled = bool(
            payload.get(
                "diagnostics_enabled",
                self.config.diagnostics_default,
            )
        )

        for frame_payload in (
            frames_payload
        ):
            if not isinstance(
                frame_payload,
                dict,
            ):
                raise ValueError(
                    "persisted frame must be an object"
                )
            frame = (
                ObservationFrame.from_packet(
                    frame_payload
                )
            )
            self._apply_frame(
                frame,
                record=True,
            )


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
        self.client = (
            client
            or OllamaClient(
                config.ollama_url
            )
        )
        self.sessions: dict[
            str,
            CortexChatSession,
        ] = {}
        self.store = (
            FileSessionStore(
                config.persistence_dir
            )
            if config.persistence_dir
            else None
        )

    def _new_session(
        self,
        session_id: str,
    ) -> CortexChatSession:
        return CortexChatSession(
            session_id,
            self.config,
            self.client,
        )

    def session(
        self,
        session_id: str,
    ) -> CortexChatSession:
        normalized = session_id.strip()
        if not normalized:
            raise ValueError(
                "session_id must not be empty"
            )
        session = self.sessions.get(
            normalized
        )
        if session is not None:
            return session

        session = self._new_session(
            normalized
        )
        if (
            self.store is not None
            and self.store.exists(
                normalized
            )
        ):
            session.restore_packet(
                self.store.load_packet(
                    normalized
                )
            )
        self.sessions[
            normalized
        ] = session
        return session

    def save(
        self,
        session_id: str,
    ) -> str:
        if self.store is None:
            raise RuntimeError(
                "gateway persistence is disabled"
            )
        path = self.store.save(
            self.session(
                session_id
            )
        )
        return str(path)

    def load(
        self,
        session_id: str,
    ) -> CortexChatSession:
        if self.store is None:
            raise RuntimeError(
                "gateway persistence is disabled"
            )
        session = self._new_session(
            session_id
        )
        session.restore_packet(
            self.store.load_packet(
                session_id
            )
        )
        self.sessions[
            session_id
        ] = session
        return session

    def list_persisted_sessions(
        self,
    ) -> list[str]:
        if self.store is None:
            return []
        return self.store.list_sessions()

    def reset(
        self,
        session_id: str,
        *,
        delete_persisted: bool = True,
    ) -> None:
        self.sessions.pop(
            session_id,
            None,
        )
        if (
            delete_persisted
            and self.store is not None
        ):
            self.store.delete(
                session_id
            )

    def set_diagnostics(
        self,
        session_id: str,
        enabled: bool,
    ) -> None:
        self.session(
            session_id
        ).set_diagnostics(
            enabled
        )
        if self.store is not None:
            self.save(
                session_id
            )

    def chat(
        self,
        session_id: str,
        user_text: str,
        *,
        diagnostics: bool | None = None,
    ) -> ChatResult:
        result = self.session(
            session_id
        ).chat(
            user_text,
            diagnostics=diagnostics,
        )
        if self.store is not None:
            self.save(
                session_id
            )
        return result


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
