"""Strict neural-boundary language gateway for Cortex.

Ollama is used only at the language boundary:

    text -> PromptIR
    RendererPacket -> text

Cortex owns persistent state, entity binding, relation memory, query resolution,
uncertainty, diagnostics, and AnswerIR construction. The strict path makes no
embedding-model calls and never gives the renderer the original prompt or chat
history.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from json import dumps, loads
from math import sqrt
from os import replace as atomic_replace
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Mapping, Sequence

try:
    from . import (
        CortexRuntime,
        __spec_version__,
        __version__,
    )
except Exception:  # pragma: no cover - docs/source-only environments
    CortexRuntime = None  # type: ignore[assignment]
    __version__ = "unknown"
    __spec_version__ = "unknown"

from .language_gateway import (
    ChatResult,
    DiagnosticReport,
    LanguageModelClient,
    OllamaClient,
    diagnose_runtime,
)


STRICT_GATEWAY_VERSION = "cortex-language-gateway-v0.2"
STRICT_SESSION_SCHEMA = "cortex-language-session-v2"
SYMBOLIC_ENCODER_ID = "sha256-symbolic-v1"


PROMPT_IR_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "language": {"type": "string"},
        "speech_act": {
            "type": "string",
            "enum": [
                "assertion",
                "question",
                "command",
                "mixed",
                "other",
            ],
        },
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "surface": {"type": "string"},
                    "kind": {"type": "string"},
                    "evidence": {"type": "string"},
                },
                "required": [
                    "id",
                    "surface",
                    "kind",
                    "evidence",
                ],
            },
        },
        "observations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "subject": {"type": "string"},
                    "predicate": {"type": "string"},
                    "object": {"type": "string"},
                    "asserted": {"type": "boolean"},
                    "evidence": {"type": "string"},
                },
                "required": [
                    "subject",
                    "predicate",
                    "object",
                    "asserted",
                    "evidence",
                ],
            },
        },
        "query_operation": {
            "type": "string",
            "enum": [
                "none",
                "relation_lookup",
                "entity_recall",
                "introspection",
            ],
        },
        "query_subject": {"type": "string"},
        "query_predicate": {"type": "string"},
        "query_object": {"type": "string"},
    },
    "required": [
        "language",
        "speech_act",
        "entities",
        "observations",
        "query_operation",
        "query_subject",
        "query_predicate",
        "query_object",
    ],
}


RENDER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "text": {"type": "string"},
        "support_ids": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": ["text", "support_ids"],
}


PARSER_SYSTEM_PROMPT = """You are Cortex's language parser.

Your only job is to translate the current user message into PromptIR.

Rules:
- Do not answer the user's question.
- Do not use world knowledge to fill in missing facts.
- Extract only explicit user assertions as observations.
- Questions are queries, not asserted facts.
- Every entity and observation must include a short verbatim evidence span
  copied from the current user message.
- Entity references inside observations/query fields must use local entity IDs.
- Normalize relation predicates to short lowercase snake_case labels.
- For relation_lookup, use query_subject as the source entity. If the user asks
  for an unknown object (for example "What is X's capital?"), leave
  query_object empty.
- Use introspection only when the user explicitly asks about Cortex's internal
  state, uncertainty, memory pressure, diagnosis, or why Cortex revised.
- Respond exactly to the supplied JSON schema."""


RENDERER_SYSTEM_PROMPT = """You are Cortex's language renderer.

You receive a RendererPacket that was produced by Cortex. Your only job is to
express that packet naturally.

Rules:
- Do not answer from pretrained knowledge.
- Do not add facts, names, entities, reasons, confidence values, or conclusions
  not present in the packet.
- If the packet says information is unknown, say that it is unknown.
- If the packet says Cortex is uncertain, preserve the uncertainty.
- Every response must cite only the packet support IDs supplied to you.
- Return exactly the supplied JSON schema."""


def _normalized_text(value: str) -> str:
    return " ".join(value.casefold().split())


def _normalized_predicate(value: str) -> str:
    compact = "_".join(
        piece
        for piece in (
            "".join(
                ch if ch.isalnum() else " "
                for ch in value.casefold()
            )
        ).split()
        if piece
    )
    return compact


def _evidence_is_anchored(
    evidence: str,
    source_text: str,
) -> bool:
    needle = _normalized_text(evidence)
    haystack = _normalized_text(source_text)
    return bool(needle) and needle in haystack


@dataclass(frozen=True)
class PromptEntity:
    local_id: str
    surface: str
    kind: str
    evidence: str


@dataclass(frozen=True)
class PromptObservation:
    subject: str
    predicate: str
    object: str
    asserted: bool
    evidence: str


@dataclass(frozen=True)
class PromptIR:
    language: str
    speech_act: str
    entities: tuple[PromptEntity, ...]
    observations: tuple[PromptObservation, ...]
    query_operation: str
    query_subject: str
    query_predicate: str
    query_object: str


@dataclass(frozen=True)
class AnswerClaim:
    claim_id: str
    subject: str
    predicate: str
    object: str
    truth: str
    evidence: tuple[str, ...]

    def packet(self) -> dict[str, Any]:
        return {
            "id": self.claim_id,
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "truth": self.truth,
            "evidence": list(self.evidence),
        }


@dataclass(frozen=True)
class CortexAnswerIR:
    status: str
    language: str
    claims: tuple[AnswerClaim, ...]
    unknown_count: int
    accepted_observations: int
    diagnostics: DiagnosticReport | None

    def packet(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "language": self.language,
            "claims": [
                claim.packet()
                for claim in self.claims
            ],
            "unknown_count": self.unknown_count,
            "accepted_observations": (
                self.accepted_observations
            ),
            "diagnostics": (
                self.diagnostics.packet()
                if self.diagnostics is not None
                else None
            ),
        }


@dataclass(frozen=True)
class RendererSegment:
    support_id: str
    kind: str
    content: str

    def packet(self) -> dict[str, str]:
        return {
            "id": self.support_id,
            "kind": self.kind,
            "content": self.content,
        }


@dataclass(frozen=True)
class RendererPacket:
    language: str
    segments: tuple[RendererSegment, ...]

    @property
    def support_ids(self) -> frozenset[str]:
        return frozenset(
            segment.support_id
            for segment in self.segments
        )

    def packet(self) -> dict[str, Any]:
        return {
            "language": self.language,
            "segments": [
                segment.packet()
                for segment in self.segments
            ],
        }

    def deterministic_text(self) -> str:
        return " ".join(
            segment.content
            for segment in self.segments
        ).strip()


@dataclass
class StrictPersistentEntity:
    cortex_id: int
    names: set[str] = field(default_factory=set)
    kinds: set[str] = field(default_factory=set)
    evidence: list[str] = field(default_factory=list)
    mentions: int = 0

    def absorb(self, entity: PromptEntity) -> None:
        self.names.add(entity.surface)
        if entity.kind:
            self.kinds.add(entity.kind)
        if (
            entity.evidence
            and entity.evidence
            not in self.evidence
        ):
            self.evidence.append(
                entity.evidence
            )
            self.evidence[:] = self.evidence[-6:]
        self.mentions += 1

    @property
    def display_name(self) -> str:
        if self.names:
            return sorted(
                self.names,
                key=lambda value: (
                    len(value),
                    value.casefold(),
                ),
            )[0]
        return f"entity-{self.cortex_id}"

    def packet(self) -> dict[str, Any]:
        return {
            "id": self.cortex_id,
            "names": sorted(self.names),
            "kinds": sorted(self.kinds),
            "evidence": list(self.evidence),
            "mentions": self.mentions,
            "encoder": SYMBOLIC_ENCODER_ID,
        }


@dataclass(frozen=True)
class StrictPersistentRelation:
    source_id: int
    target_id: int
    predicate: str
    asserted: bool
    evidence: str

    def packet(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "predicate": self.predicate,
            "asserted": self.asserted,
            "evidence": self.evidence,
            "provenance": "user_message",
        }


@dataclass(frozen=True)
class StrictFrame:
    prompt_ir: PromptIR
    vectors: tuple[tuple[float, ...], ...]

    def packet(self) -> dict[str, Any]:
        return {
            "prompt_ir": prompt_ir_packet(
                self.prompt_ir
            ),
            "vectors": [
                list(vector)
                for vector in self.vectors
            ],
        }

    @classmethod
    def from_packet(
        cls,
        payload: Mapping[str, Any],
    ) -> "StrictFrame":
        prompt_payload = payload.get(
            "prompt_ir"
        )
        if not isinstance(
            prompt_payload,
            dict,
        ):
            raise ValueError(
                "persisted strict frame has no PromptIR"
            )
        prompt_ir = parse_prompt_ir_payload(
            prompt_payload,
            source_text=None,
        )
        vectors = tuple(
            tuple(
                float(value)
                for value in vector
            )
            for vector in payload.get(
                "vectors",
                [],
            )
        )
        if (
            len(vectors)
            != len(
                prompt_ir.entities
            )
        ):
            raise ValueError(
                "persisted strict frame vector/entity count mismatch"
            )
        return cls(
            prompt_ir=prompt_ir,
            vectors=vectors,
        )


@dataclass(frozen=True)
class StrictGatewayConfig:
    parser_model: str
    renderer_model: str
    ollama_url: str = "http://localhost:11434"
    feature_dim: int = 128
    max_entities: int = 96
    memory_budget: int = 24
    diagnostics_default: bool = False
    persistence_dir: str | None = None
    chat_model: str = "cortex"
    embedding_model: str = SYMBOLIC_ENCODER_ID


def prompt_ir_packet(
    prompt_ir: PromptIR,
) -> dict[str, Any]:
    return {
        "language": prompt_ir.language,
        "speech_act": prompt_ir.speech_act,
        "entities": [
            {
                "id": entity.local_id,
                "surface": entity.surface,
                "kind": entity.kind,
                "evidence": entity.evidence,
            }
            for entity in prompt_ir.entities
        ],
        "observations": [
            {
                "subject": observation.subject,
                "predicate": observation.predicate,
                "object": observation.object,
                "asserted": observation.asserted,
                "evidence": observation.evidence,
            }
            for observation
            in prompt_ir.observations
        ],
        "query_operation": (
            prompt_ir.query_operation
        ),
        "query_subject": (
            prompt_ir.query_subject
        ),
        "query_predicate": (
            prompt_ir.query_predicate
        ),
        "query_object": (
            prompt_ir.query_object
        ),
    }


def parse_prompt_ir_payload(
    payload: Mapping[str, Any],
    *,
    source_text: str | None,
) -> PromptIR:
    raw_entities = payload.get(
        "entities",
        [],
    )
    raw_observations = payload.get(
        "observations",
        [],
    )
    if (
        not isinstance(
            raw_entities,
            list,
        )
        or not isinstance(
            raw_observations,
            list,
        )
    ):
        raise ValueError(
            "PromptIR entities/observations must be arrays"
        )

    entities: list[PromptEntity] = []
    ids: set[str] = set()
    for item in raw_entities:
        if not isinstance(
            item,
            dict,
        ):
            raise ValueError(
                "PromptIR entity must be an object"
            )
        entity = PromptEntity(
            local_id=str(
                item.get("id", "")
            ).strip(),
            surface=str(
                item.get(
                    "surface",
                    "",
                )
            ).strip(),
            kind=str(
                item.get(
                    "kind",
                    "",
                )
            ).strip(),
            evidence=str(
                item.get(
                    "evidence",
                    "",
                )
            ).strip(),
        )
        if (
            not entity.local_id
            or entity.local_id
            in ids
            or not entity.surface
        ):
            raise ValueError(
                "PromptIR entity IDs/surfaces must be unique and non-empty"
            )
        if (
            source_text is not None
            and not _evidence_is_anchored(
                entity.evidence,
                source_text,
            )
        ):
            raise ValueError(
                "PromptIR entity evidence is not anchored in the source message"
            )
        ids.add(
            entity.local_id
        )
        entities.append(
            entity
        )

    observations: list[
        PromptObservation
    ] = []
    for item in raw_observations:
        if not isinstance(
            item,
            dict,
        ):
            raise ValueError(
                "PromptIR observation must be an object"
            )
        observation = PromptObservation(
            subject=str(
                item.get(
                    "subject",
                    "",
                )
            ).strip(),
            predicate=_normalized_predicate(
                str(
                    item.get(
                        "predicate",
                        "",
                    )
                )
            ),
            object=str(
                item.get(
                    "object",
                    "",
                )
            ).strip(),
            asserted=bool(
                item.get(
                    "asserted",
                    True,
                )
            ),
            evidence=str(
                item.get(
                    "evidence",
                    "",
                )
            ).strip(),
        )
        if (
            observation.subject
            not in ids
            or observation.object
            not in ids
            or not observation.predicate
        ):
            raise ValueError(
                "PromptIR observation references an unknown entity or empty predicate"
            )
        if (
            source_text is not None
            and not _evidence_is_anchored(
                observation.evidence,
                source_text,
            )
        ):
            raise ValueError(
                "PromptIR observation evidence is not anchored in the source message"
            )
        observations.append(
            observation
        )

    operation = str(
        payload.get(
            "query_operation",
            "none",
        )
    ).strip()
    allowed_operations = {
        "none",
        "relation_lookup",
        "entity_recall",
        "introspection",
    }
    if operation not in allowed_operations:
        raise ValueError(
            "unsupported PromptIR query operation"
        )

    query_subject = str(
        payload.get(
            "query_subject",
            "",
        )
    ).strip()
    query_object = str(
        payload.get(
            "query_object",
            "",
        )
    ).strip()
    query_predicate = (
        _normalized_predicate(
            str(
                payload.get(
                    "query_predicate",
                    "",
                )
            )
        )
    )

    if (
        query_subject
        and query_subject
        not in ids
    ):
        raise ValueError(
            "PromptIR query subject references unknown entity"
        )
    if (
        query_object
        and query_object
        not in ids
    ):
        raise ValueError(
            "PromptIR query object references unknown entity"
        )
    if (
        operation
        == "relation_lookup"
        and (
            not query_subject
            or not query_predicate
        )
    ):
        raise ValueError(
            "relation_lookup requires subject and predicate"
        )
    if (
        operation
        == "entity_recall"
        and not query_subject
    ):
        raise ValueError(
            "entity_recall requires a subject"
        )

    return PromptIR(
        language=str(
            payload.get(
                "language",
                "en",
            )
        ).strip()
        or "en",
        speech_act=str(
            payload.get(
                "speech_act",
                "other",
            )
        ).strip()
        or "other",
        entities=tuple(
            entities
        ),
        observations=tuple(
            observations
        ),
        query_operation=operation,
        query_subject=query_subject,
        query_predicate=query_predicate,
        query_object=query_object,
    )


def parse_prompt_ir(
    raw: str,
    *,
    source_text: str,
) -> PromptIR:
    try:
        payload = loads(
            raw
        )
    except Exception as exc:
        raise ValueError(
            "PromptIR is not valid JSON"
        ) from exc
    if not isinstance(
        payload,
        dict,
    ):
        raise ValueError(
            "PromptIR must be an object"
        )
    return parse_prompt_ir_payload(
        payload,
        source_text=source_text,
    )


class DeterministicSymbolicEncoder:
    """Map parsed language symbols into a fixed Cortex feature space."""

    def __init__(
        self,
        feature_dim: int = 128,
    ) -> None:
        if feature_dim < 32:
            raise ValueError(
                "strict symbolic feature_dim must be at least 32"
            )
        self.feature_dim = int(
            feature_dim
        )

    def _add(
        self,
        vector: list[float],
        token: str,
        weight: float,
    ) -> None:
        digest = sha256(
            token.encode(
                "utf-8"
            )
        ).digest()
        for offset in range(
            0,
            16,
            2,
        ):
            value = int.from_bytes(
                digest[
                    offset : offset + 2
                ],
                "big",
            )
            index = (
                value
                % self.feature_dim
            )
            sign = (
                1.0
                if digest[
                    16
                    + offset // 2
                ]
                & 1
                else -1.0
            )
            vector[
                index
            ] += sign * weight

    def encode_entity(
        self,
        entity: PromptEntity,
    ) -> tuple[float, ...]:
        normalized_name = (
            _normalized_text(
                entity.surface
            )
        )
        normalized_kind = (
            _normalized_text(
                entity.kind
            )
        )
        vector = [
            0.0
            for _ in range(
                self.feature_dim
            )
        ]

        self._add(
            vector,
            "name:"
            + normalized_name,
            4.0,
        )
        if normalized_kind:
            self._add(
                vector,
                "kind:"
                + normalized_kind,
                0.5,
            )

        compact = (
            "^"
            + normalized_name.replace(
                " ",
                "_",
            )
            + "$"
        )
        for index in range(
            max(
                0,
                len(compact) - 2,
            )
        ):
            self._add(
                vector,
                "ngram:"
                + compact[
                    index : index + 3
                ],
                0.15,
            )

        norm = sqrt(
            sum(
                value * value
                for value in vector
            )
        )
        if norm <= 0.0:
            raise ValueError(
                "cannot encode an empty entity"
            )
        return tuple(
            value / norm
            for value in vector
        )

    def encode(
        self,
        prompt_ir: PromptIR,
    ) -> tuple[
        tuple[float, ...],
        ...,
    ]:
        return tuple(
            self.encode_entity(
                entity
            )
            for entity
            in prompt_ir.entities
        )


class StrictFileSessionStore:
    def __init__(
        self,
        root: str | Path,
    ) -> None:
        self.root = Path(
            root
        ).expanduser()
        self.root.mkdir(
            parents=True,
            exist_ok=True,
        )

    @staticmethod
    def _safe_name(
        session_id: str,
    ) -> str:
        digest = sha256(
            session_id.encode(
                "utf-8"
            )
        ).hexdigest()[:16]
        readable = "".join(
            ch
            if (
                ch.isalnum()
                or ch in (
                    "-",
                    "_",
                )
            )
            else "_"
            for ch in session_id
        ).strip("_")[:48]
        return (
            f"{readable or 'session'}-v2-{digest}.json"
        )

    def path_for(
        self,
        session_id: str,
    ) -> Path:
        return (
            self.root
            / self._safe_name(
                session_id
            )
        )

    def exists(
        self,
        session_id: str,
    ) -> bool:
        return self.path_for(
            session_id
        ).is_file()

    def list_sessions(
        self,
    ) -> list[str]:
        sessions: list[str] = []
        for path in sorted(
            self.root.glob(
                "*-v2-*.json"
            )
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
                isinstance(
                    payload,
                    dict,
                )
                and payload.get(
                    "schema"
                )
                == STRICT_SESSION_SCHEMA
                and isinstance(
                    payload.get(
                        "session_id"
                    ),
                    str,
                )
            ):
                sessions.append(
                    payload[
                        "session_id"
                    ]
                )
        return sessions

    def save(
        self,
        session: "StrictCortexChatSession",
    ) -> Path:
        target = self.path_for(
            session.session_id
        )
        payload = (
            session.persistence_packet()
        )
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=str(
                self.root
            ),
            prefix=".cortex-v2-",
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
            handle.write(
                "\n"
            )
            temporary = Path(
                handle.name
            )
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
                f"no persisted strict Cortex session: {session_id}"
            )
        payload = loads(
            path.read_text(
                encoding="utf-8"
            )
        )
        if (
            not isinstance(
                payload,
                dict,
            )
            or payload.get(
                "schema"
            )
            != STRICT_SESSION_SCHEMA
        ):
            raise ValueError(
                "unsupported strict Cortex session schema"
            )
        if payload.get(
            "session_id"
        ) != session_id:
            raise ValueError(
                "strict Cortex session ID mismatch"
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


class StrictCortexChatSession:
    def __init__(
        self,
        session_id: str,
        config: StrictGatewayConfig,
        client: LanguageModelClient,
    ) -> None:
        self.session_id = (
            session_id
        )
        self.config = config
        self.client = client
        self.encoder = (
            DeterministicSymbolicEncoder(
                config.feature_dim
            )
        )
        self.diagnostics_enabled = bool(
            config.diagnostics_default
        )
        self.runtime: Any | None = None
        self.last_read: Any | None = None
        self.entities: dict[
            int,
            StrictPersistentEntity,
        ] = {}
        self.relations: dict[
            tuple[
                int,
                int,
                str,
            ],
            StrictPersistentRelation,
        ] = {}
        self.frames: list[
            StrictFrame
        ] = []
        self.history: list[
            dict[str, str]
        ] = []

    def set_diagnostics(
        self,
        enabled: bool,
    ) -> None:
        self.diagnostics_enabled = bool(
            enabled
        )

    def _ensure_runtime(
        self,
    ) -> Any:
        if self.runtime is not None:
            return self.runtime
        if CortexRuntime is None:
            raise RuntimeError(
                "Cortex native runtime is unavailable"
            )
        self.runtime = CortexRuntime(
            feature_dim=(
                self.config.feature_dim
            ),
            max_entities=(
                self.config.max_entities
            ),
            budget=(
                self.config.memory_budget
            ),
        )
        return self.runtime

    def _parse(
        self,
        user_text: str,
    ) -> PromptIR:
        raw = self.client.chat(
            model=(
                self.config.parser_model
            ),
            messages=[
                {
                    "role": "system",
                    "content": (
                        PARSER_SYSTEM_PROMPT
                    ),
                },
                {
                    "role": "user",
                    "content": user_text,
                },
            ],
            format_schema=(
                PROMPT_IR_SCHEMA
            ),
        )
        return parse_prompt_ir(
            raw,
            source_text=user_text,
        )

    def _apply_frame(
        self,
        frame: StrictFrame,
        *,
        record: bool,
    ) -> tuple[
        Any | None,
        dict[str, int],
    ]:
        prompt_ir = (
            frame.prompt_ir
        )
        if not prompt_ir.entities:
            if record:
                self.frames.append(
                    frame
                )
            return (
                self.last_read,
                {},
            )

        runtime = (
            self._ensure_runtime()
        )
        local_index = {
            entity.local_id: index
            for index, entity
            in enumerate(
                prompt_ir.entities
            )
        }
        relation_obs = [
            (
                local_index[
                    observation.subject
                ],
                local_index[
                    observation.object
                ],
                1
                if observation.asserted
                else 0,
            )
            for observation
            in prompt_ir.observations
        ]
        read = runtime.step(
            [
                list(vector)
                for vector
                in frame.vectors
            ],
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
        local_bindings: dict[
            str,
            int,
        ] = {}
        for entity, cortex_id in zip(
            prompt_ir.entities,
            bindings,
        ):
            local_bindings[
                entity.local_id
            ] = cortex_id
            persistent = (
                self.entities.setdefault(
                    cortex_id,
                    StrictPersistentEntity(
                        cortex_id=(
                            cortex_id
                        )
                    ),
                )
            )
            persistent.absorb(
                entity
            )

        for observation in (
            prompt_ir.observations
        ):
            source_id = (
                local_bindings[
                    observation.subject
                ]
            )
            target_id = (
                local_bindings[
                    observation.object
                ]
            )
            predicate = (
                _normalized_predicate(
                    observation.predicate
                )
            )
            relation = (
                StrictPersistentRelation(
                    source_id=source_id,
                    target_id=target_id,
                    predicate=predicate,
                    asserted=(
                        observation.asserted
                    ),
                    evidence=(
                        observation.evidence
                    ),
                )
            )
            self.relations[
                (
                    source_id,
                    target_id,
                    predicate,
                )
            ] = relation

        if record:
            self.frames.append(
                frame
            )

        return (
            read,
            local_bindings,
        )

    def _observe(
        self,
        prompt_ir: PromptIR,
    ) -> tuple[
        Any | None,
        dict[str, int],
    ]:
        frame = StrictFrame(
            prompt_ir=prompt_ir,
            vectors=(
                self.encoder.encode(
                    prompt_ir
                )
            ),
        )
        return self._apply_frame(
            frame,
            record=True,
        )

    def _entity_name(
        self,
        cortex_id: int,
    ) -> str:
        entity = self.entities.get(
            cortex_id
        )
        if entity is None:
            return (
                f"entity-{cortex_id}"
            )
        return entity.display_name

    def _relation_answer(
        self,
        prompt_ir: PromptIR,
        bindings: Mapping[
            str,
            int,
        ],
        diagnostic: DiagnosticReport | None,
    ) -> CortexAnswerIR:
        source_id = bindings.get(
            prompt_ir.query_subject
        )
        if source_id is None:
            return CortexAnswerIR(
                status="unknown",
                language=(
                    prompt_ir.language
                ),
                claims=tuple(),
                unknown_count=1,
                accepted_observations=len(
                    prompt_ir.observations
                ),
                diagnostics=diagnostic,
            )

        predicate = (
            _normalized_predicate(
                prompt_ir.query_predicate
            )
        )
        object_id = (
            bindings.get(
                prompt_ir.query_object
            )
            if prompt_ir.query_object
            else None
        )

        matching = [
            relation
            for (
                key_source,
                key_target,
                key_predicate,
            ), relation
            in self.relations.items()
            if (
                key_source
                == source_id
                and key_predicate
                == predicate
                and (
                    object_id is None
                    or key_target
                    == object_id
                )
            )
        ]
        if not matching:
            return CortexAnswerIR(
                status="unknown",
                language=(
                    prompt_ir.language
                ),
                claims=tuple(),
                unknown_count=1,
                accepted_observations=len(
                    prompt_ir.observations
                ),
                diagnostics=diagnostic,
            )

        claims: list[
            AnswerClaim
        ] = []
        for index, relation in enumerate(
            sorted(
                matching,
                key=lambda item: (
                    item.target_id,
                    item.predicate,
                ),
            )
        ):
            claims.append(
                AnswerClaim(
                    claim_id=(
                        f"claim:{index}"
                    ),
                    subject=(
                        self._entity_name(
                            relation.source_id
                        )
                    ),
                    predicate=(
                        relation.predicate
                    ),
                    object=(
                        self._entity_name(
                            relation.target_id
                        )
                    ),
                    truth=(
                        "supported"
                        if relation.asserted
                        else "refuted"
                    ),
                    evidence=(
                        relation.evidence,
                    ),
                )
            )
        return CortexAnswerIR(
            status="resolved",
            language=prompt_ir.language,
            claims=tuple(
                claims
            ),
            unknown_count=0,
            accepted_observations=len(
                prompt_ir.observations
            ),
            diagnostics=diagnostic,
        )

    def _entity_recall_answer(
        self,
        prompt_ir: PromptIR,
        bindings: Mapping[
            str,
            int,
        ],
        diagnostic: DiagnosticReport | None,
    ) -> CortexAnswerIR:
        entity_id = bindings.get(
            prompt_ir.query_subject
        )
        if entity_id is None:
            return CortexAnswerIR(
                status="unknown",
                language=(
                    prompt_ir.language
                ),
                claims=tuple(),
                unknown_count=1,
                accepted_observations=len(
                    prompt_ir.observations
                ),
                diagnostics=diagnostic,
            )

        matching = [
            relation
            for relation
            in self.relations.values()
            if (
                relation.source_id
                == entity_id
                or relation.target_id
                == entity_id
            )
        ]
        if not matching:
            return CortexAnswerIR(
                status="unknown",
                language=(
                    prompt_ir.language
                ),
                claims=tuple(),
                unknown_count=1,
                accepted_observations=len(
                    prompt_ir.observations
                ),
                diagnostics=diagnostic,
            )

        claims = tuple(
            AnswerClaim(
                claim_id=(
                    f"claim:{index}"
                ),
                subject=(
                    self._entity_name(
                        relation.source_id
                    )
                ),
                predicate=(
                    relation.predicate
                ),
                object=(
                    self._entity_name(
                        relation.target_id
                    )
                ),
                truth=(
                    "supported"
                    if relation.asserted
                    else "refuted"
                ),
                evidence=(
                    relation.evidence,
                ),
            )
            for index, relation
            in enumerate(
                sorted(
                    matching,
                    key=lambda item: (
                        item.source_id,
                        item.target_id,
                        item.predicate,
                    ),
                )
            )
        )
        return CortexAnswerIR(
            status="resolved",
            language=prompt_ir.language,
            claims=claims,
            unknown_count=0,
            accepted_observations=len(
                prompt_ir.observations
            ),
            diagnostics=diagnostic,
        )

    def _answer_ir(
        self,
        prompt_ir: PromptIR,
        bindings: Mapping[
            str,
            int,
        ],
        diagnostic: DiagnosticReport | None,
    ) -> CortexAnswerIR:
        if (
            prompt_ir.query_operation
            == "relation_lookup"
        ):
            return self._relation_answer(
                prompt_ir,
                bindings,
                diagnostic,
            )
        if (
            prompt_ir.query_operation
            == "entity_recall"
        ):
            return (
                self._entity_recall_answer(
                    prompt_ir,
                    bindings,
                    diagnostic,
                )
            )
        if (
            prompt_ir.query_operation
            == "introspection"
        ):
            return CortexAnswerIR(
                status=(
                    "diagnostic"
                    if diagnostic
                    is not None
                    else "unknown"
                ),
                language=(
                    prompt_ir.language
                ),
                claims=tuple(),
                unknown_count=(
                    0
                    if diagnostic
                    is not None
                    else 1
                ),
                accepted_observations=len(
                    prompt_ir.observations
                ),
                diagnostics=diagnostic,
            )
        return CortexAnswerIR(
            status="acknowledged",
            language=prompt_ir.language,
            claims=tuple(),
            unknown_count=0,
            accepted_observations=len(
                prompt_ir.observations
            ),
            diagnostics=diagnostic,
        )

    def _renderer_packet(
        self,
        answer_ir: CortexAnswerIR,
    ) -> RendererPacket:
        segments: list[
            RendererSegment
        ] = []

        if answer_ir.claims:
            for claim in (
                answer_ir.claims
            ):
                predicate = (
                    claim.predicate.replace(
                        "_",
                        " ",
                    )
                )
                truth_prefix = (
                    ""
                    if claim.truth
                    == "supported"
                    else "It is not supported that "
                )
                if claim.truth == "supported":
                    content = (
                        f"{claim.subject} {predicate} {claim.object}."
                    )
                else:
                    content = (
                        f"{truth_prefix}{claim.subject} "
                        f"{predicate} {claim.object}."
                    )
                segments.append(
                    RendererSegment(
                        support_id=(
                            claim.claim_id
                        ),
                        kind="claim",
                        content=content,
                    )
                )

        if (
            answer_ir.status
            == "unknown"
        ):
            segments.append(
                RendererSegment(
                    support_id="unknown:0",
                    kind="unknown",
                    content=(
                        "I do not currently have enough information in Cortex to answer that."
                    ),
                )
            )

        if (
            answer_ir.status
            == "diagnostic"
            and answer_ir.diagnostics
            is not None
        ):
            diagnostic = (
                answer_ir.diagnostics
            )
            segments.append(
                RendererSegment(
                    support_id="diagnostic:0",
                    kind="diagnostic",
                    content=(
                        diagnostic.summary
                    ),
                )
            )
            for index, signal in enumerate(
                diagnostic.signals
            ):
                segments.append(
                    RendererSegment(
                        support_id=(
                            f"diagnostic:{index + 1}"
                        ),
                        kind="diagnostic",
                        content=signal + ".",
                    )
                )

        if (
            answer_ir.status
            == "acknowledged"
        ):
            content = (
                "I have incorporated the explicit observation into Cortex."
                if (
                    answer_ir.accepted_observations
                    > 0
                )
                else "I have no new explicit observation to store."
            )
            segments.append(
                RendererSegment(
                    support_id="ack:0",
                    kind="acknowledgement",
                    content=content,
                )
            )

        if not segments:
            segments.append(
                RendererSegment(
                    support_id="unknown:0",
                    kind="unknown",
                    content=(
                        "I do not currently have enough information in Cortex to answer that."
                    ),
                )
            )

        return RendererPacket(
            language=(
                answer_ir.language
            ),
            segments=tuple(
                segments
            ),
        )

    def _render(
        self,
        packet: RendererPacket,
    ) -> str:
        raw = self.client.chat(
            model=(
                self.config.renderer_model
            ),
            messages=[
                {
                    "role": "system",
                    "content": (
                        RENDERER_SYSTEM_PROMPT
                    ),
                },
                {
                    "role": "user",
                    "content": dumps(
                        packet.packet(),
                        sort_keys=True,
                    ),
                },
            ],
            format_schema=(
                RENDER_SCHEMA
            ),
        )
        try:
            payload = loads(
                raw
            )
        except Exception:
            return (
                packet.deterministic_text()
            )
        if not isinstance(
            payload,
            dict,
        ):
            return (
                packet.deterministic_text()
            )
        text = payload.get(
            "text"
        )
        support_ids = (
            payload.get(
                "support_ids"
            )
        )
        if (
            not isinstance(
                text,
                str,
            )
            or not text.strip()
            or not isinstance(
                support_ids,
                list,
            )
        ):
            return (
                packet.deterministic_text()
            )
        supplied = frozenset(
            str(value)
            for value in support_ids
        )
        if (
            not supplied
            or not supplied.issubset(
                packet.support_ids
            )
        ):
            return (
                packet.deterministic_text()
            )
        return text.strip()

    def context_packet(
        self,
        *,
        diagnostics: DiagnosticReport | None,
    ) -> dict[str, Any]:
        return {
            "session_id": (
                self.session_id
            ),
            "gateway": (
                STRICT_GATEWAY_VERSION
            ),
            "encoder": (
                SYMBOLIC_ENCODER_ID
            ),
            "entities": [
                entity.packet()
                for entity in sorted(
                    self.entities.values(),
                    key=lambda item: (
                        -item.mentions,
                        item.cortex_id,
                    ),
                )[:12]
            ],
            "relations": [
                relation.packet()
                for relation in list(
                    self.relations.values()
                )[-16:]
            ],
            "runtime": (
                {
                    "prediction": float(
                        self.last_read.prediction
                    ),
                    "unresolved": bool(
                        self.last_read.unresolved
                    ),
                    "revision": bool(
                        self.last_read.revision
                    ),
                    "reactivated": bool(
                        self.last_read.reactivated
                    ),
                    "discovered": bool(
                        self.last_read.discovered
                    ),
                    "budget_pressure": bool(
                        self.last_read.budget_pressure
                    ),
                    "stored": int(
                        self.last_read.stored
                    ),
                }
                if self.last_read
                is not None
                else None
            ),
            "diagnostics": (
                diagnostics.packet()
                if diagnostics
                is not None
                else None
            ),
        }

    def chat(
        self,
        user_text: str,
        *,
        diagnostics: bool | None = None,
    ) -> ChatResult:
        text = user_text.strip()
        if not text:
            raise ValueError(
                "user message must not be empty"
            )

        prompt_ir = (
            self._parse(
                text
            )
        )
        read, bindings = (
            self._observe(
                prompt_ir
            )
        )

        diagnostic_enabled = (
            self.diagnostics_enabled
            if diagnostics is None
            else bool(
                diagnostics
            )
        )
        diagnostic = (
            diagnose_runtime(
                read,
                memory_budget=(
                    self.config.memory_budget
                ),
            )
            if (
                diagnostic_enabled
                and read is not None
            )
            else None
        )

        answer_ir = self._answer_ir(
            prompt_ir,
            bindings,
            diagnostic,
        )
        render_packet = (
            self._renderer_packet(
                answer_ir
            )
        )
        content = self._render(
            render_packet
        )

        self.history.extend(
            [
                {
                    "role": "user",
                    "content": text,
                },
                {
                    "role": "assistant",
                    "content": content,
                },
            ]
        )
        self.history[:] = (
            self.history[-12:]
        )

        return ChatResult(
            response_id=(
                "chatcmpl-"
                + sha256(
                    (
                        self.session_id
                        + str(
                            len(
                                self.frames
                            )
                        )
                        + content
                    ).encode(
                        "utf-8"
                    )
                ).hexdigest()[:24]
            ),
            session_id=(
                self.session_id
            ),
            model="cortex",
            content=content,
            cortex_context={
                "prompt_ir": (
                    prompt_ir_packet(
                        prompt_ir
                    )
                ),
                "answer_ir": (
                    answer_ir.packet()
                ),
                "renderer_packet": (
                    render_packet.packet()
                ),
                "state": (
                    self.context_packet(
                        diagnostics=diagnostic
                    )
                ),
            },
            diagnostics=diagnostic,
        )

    def persistence_packet(
        self,
    ) -> dict[str, Any]:
        return {
            "schema": (
                STRICT_SESSION_SCHEMA
            ),
            "session_id": (
                self.session_id
            ),
            "gateway_version": (
                STRICT_GATEWAY_VERSION
            ),
            "encoder": (
                SYMBOLIC_ENCODER_ID
            ),
            "feature_dim": (
                self.config.feature_dim
            ),
            "cortex_version": (
                __version__
            ),
            "cortex_spec_version": (
                __spec_version__
            ),
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
            "runtime_snapshot_json": (
                self.runtime.snapshot_json()
                if (
                    self.runtime
                    is not None
                    and hasattr(
                        self.runtime,
                        "snapshot_json",
                    )
                )
                else None
            ),
        }

    def restore_packet(
        self,
        payload: Mapping[str, Any],
    ) -> None:
        if (
            payload.get(
                "encoder"
            )
            != SYMBOLIC_ENCODER_ID
        ):
            raise ValueError(
                "strict session encoder mismatch"
            )
        if int(
            payload.get(
                "feature_dim",
                -1,
            )
        ) != int(
            self.config.feature_dim
        ):
            raise ValueError(
                "strict session feature dimension mismatch"
            )

        frames = payload.get(
            "frames",
            [],
        )
        if not isinstance(
            frames,
            list,
        ):
            raise ValueError(
                "strict session frames must be an array"
            )

        self.runtime = None
        self.last_read = None
        self.entities.clear()
        self.relations.clear()
        self.frames.clear()
        self.history = [
            {
                "role": str(
                    item.get(
                        "role",
                        "",
                    )
                ),
                "content": str(
                    item.get(
                        "content",
                        "",
                    )
                ),
            }
            for item in payload.get(
                "history",
                [],
            )
            if isinstance(
                item,
                dict,
            )
        ][-12:]
        self.diagnostics_enabled = bool(
            payload.get(
                "diagnostics_enabled",
                self.config.diagnostics_default,
            )
        )

        for item in frames:
            if not isinstance(
                item,
                dict,
            ):
                raise ValueError(
                    "strict session frame must be an object"
                )
            frame = (
                StrictFrame.from_packet(
                    item
                )
            )
            expected_vectors = (
                self.encoder.encode(
                    frame.prompt_ir
                )
            )
            if expected_vectors != frame.vectors:
                raise ValueError(
                    "strict session deterministic encoder replay mismatch"
                )
            self._apply_frame(
                frame,
                record=True,
            )

        expected_snapshot = (
            payload.get(
                "runtime_snapshot_json"
            )
        )
        if (
            expected_snapshot
            is not None
            and self.runtime
            is not None
            and hasattr(
                self.runtime,
                "snapshot_json",
            )
        ):
            actual = loads(
                self.runtime.snapshot_json()
            )
            expected = loads(
                str(
                    expected_snapshot
                )
            )
            if actual != expected:
                raise ValueError(
                    "strict session native replay diverged from saved snapshot"
                )


class StrictCortexLanguageGateway:
    def __init__(
        self,
        config: StrictGatewayConfig,
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
            StrictCortexChatSession,
        ] = {}
        self.store = (
            StrictFileSessionStore(
                config.persistence_dir
            )
            if config.persistence_dir
            else None
        )

    def _new_session(
        self,
        session_id: str,
    ) -> StrictCortexChatSession:
        return StrictCortexChatSession(
            session_id,
            self.config,
            self.client,
        )

    def session(
        self,
        session_id: str,
    ) -> StrictCortexChatSession:
        normalized = (
            session_id.strip()
        )
        if not normalized:
            raise ValueError(
                "session_id must not be empty"
            )
        existing = (
            self.sessions.get(
                normalized
            )
        )
        if existing is not None:
            return existing

        session = (
            self._new_session(
                normalized
            )
        )
        if (
            self.store
            is not None
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

    def chat(
        self,
        session_id: str,
        user_text: str,
        *,
        diagnostics: bool | None = None,
    ) -> ChatResult:
        session = self.session(
            session_id
        )
        try:
            result = session.chat(
                user_text,
                diagnostics=diagnostics,
            )
        except Exception:
            if self.store is not None:
                try:
                    self.store.save(
                        session
                    )
                except Exception:
                    pass
            raise
        if self.store is not None:
            self.store.save(
                session
            )
        return result

    def set_diagnostics(
        self,
        session_id: str,
        enabled: bool,
    ) -> None:
        session = self.session(
            session_id
        )
        session.set_diagnostics(
            enabled
        )
        if self.store is not None:
            self.store.save(
                session
            )

    def save(
        self,
        session_id: str,
    ) -> str:
        if self.store is None:
            raise RuntimeError(
                "gateway persistence is disabled"
            )
        return str(
            self.store.save(
                self.session(
                    session_id
                )
            )
        )

    def load(
        self,
        session_id: str,
    ) -> StrictCortexChatSession:
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
            and self.store
            is not None
        ):
            self.store.delete(
                session_id
            )

    def list_persisted_sessions(
        self,
    ) -> list[str]:
        if self.store is None:
            return []
        return (
            self.store.list_sessions()
        )
