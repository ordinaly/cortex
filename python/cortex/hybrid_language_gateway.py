"""Hybrid language/knowledge + Cortex reasoning gateway.

Gateway v0.3 restores the language model's ordinary role as a broad pretrained
knowledge base and conversational model. Cortex is a metacognitive/structural
sidecar: explicit user observations may update Cortex, candidate neural claims
are treated as neural priors (never as observations), and Cortex emits a
separate diagnosis about support, conflict, uncertainty, and memory changes.

A valid neural answer is never blocked by a Cortex parsing/diagnosis failure.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from json import dumps, loads
from os import replace as atomic_replace
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import perf_counter
from typing import Any, Mapping, Sequence
from uuid import uuid4

from .language_gateway import (
    ChatResult,
    DiagnosticReport,
    LanguageModelClient,
    OllamaClient,
    diagnose_runtime,
)
from .strict_language_gateway import (
    CortexAnswerIR,
    DeterministicSymbolicEncoder,
    PromptIR,
    StrictCortexChatSession,
    StrictFrame,
    SYMBOLIC_ENCODER_ID,
    parse_prompt_ir_payload,
    prompt_ir_packet,
)


HYBRID_GATEWAY_VERSION = "cortex-language-gateway-v0.3"
HYBRID_SESSION_SCHEMA = "cortex-language-session-v3"


CANDIDATE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
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
        "claims": {
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
    },
    "required": [
        "entities",
        "claims",
    ],
}


CANDIDATE_SYSTEM_PROMPT = """You are Cortex's candidate-answer analyzer.

The input is an assistant answer produced from a neural model's pretrained
knowledge. Extract only factual relation claims explicitly stated in that
answer.

Rules:
- Do not add facts that the answer did not state.
- Local entity IDs such as e0/e1 exist only inside this extraction.
- Every claim subject/object must refer to a declared entity.
- Normalize predicates to short lowercase snake_case labels.
- Mark asserted=false only for an explicit negated claim.
- Every entity and claim must contain a short verbatim evidence span copied
  from the assistant answer.
- It is valid to return empty entities/claims when the answer is conversational,
  procedural, uncertain, or cannot be represented as a binary relation.
- Respond exactly to the supplied JSON schema."""


def _norm(value: str) -> str:
    return " ".join(
        value.casefold().split()
    )


def _predicate(value: str) -> str:
    return "_".join(
        "".join(
            ch if ch.isalnum() else " "
            for ch in value.casefold()
        ).split()
    )


def _anchored(
    evidence: str,
    source: str,
) -> bool:
    needle = _norm(
        evidence
    )
    return bool(
        needle
    ) and needle in _norm(
        source
    )


@dataclass(frozen=True)
class CandidateEntity:
    local_id: str
    surface: str
    kind: str
    evidence: str


@dataclass(frozen=True)
class CandidateClaim:
    subject: str
    predicate: str
    object: str
    asserted: bool
    evidence: str

    def packet(
        self,
    ) -> dict[str, Any]:
        return {
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "asserted": self.asserted,
            "evidence": self.evidence,
            "provenance": "neural_prior",
        }


@dataclass(frozen=True)
class CandidateAnswerIR:
    entities: tuple[
        CandidateEntity,
        ...,
    ]
    claims: tuple[
        CandidateClaim,
        ...,
    ]

    def packet(
        self,
    ) -> dict[str, Any]:
        return {
            "entities": [
                {
                    "id": entity.local_id,
                    "surface": entity.surface,
                    "kind": entity.kind,
                    "evidence": entity.evidence,
                }
                for entity in self.entities
            ],
            "claims": [
                claim.packet()
                for claim in self.claims
            ],
        }


@dataclass(frozen=True)
class ClaimAssessment:
    claim: CandidateClaim
    status: str
    cortex_evidence: tuple[
        str,
        ...,
    ] = tuple()

    def packet(
        self,
    ) -> dict[str, Any]:
        return {
            "claim": self.claim.packet(),
            "status": self.status,
            "cortex_evidence": list(
                self.cortex_evidence
            ),
        }


@dataclass(frozen=True)
class HybridDiagnosis:
    status: str
    reasoning_mode: str
    summary: str
    neural_claims: tuple[
        ClaimAssessment,
        ...,
    ]
    cortex_claims: tuple[
        Mapping[str, Any],
        ...,
    ]
    accepted_user_observations: int
    memory_changed: bool
    analysis_errors: tuple[
        str,
        ...,
    ]
    runtime: Mapping[
        str,
        Any,
    ] | None
    timings: Mapping[
        str,
        float,
    ]

    def packet(
        self,
    ) -> dict[str, Any]:
        return {
            "status": self.status,
            "reasoning_mode": (
                self.reasoning_mode
            ),
            "summary": self.summary,
            "neural_claims": [
                item.packet()
                for item
                in self.neural_claims
            ],
            "cortex_claims": [
                dict(
                    item
                )
                for item
                in self.cortex_claims
            ],
            "accepted_user_observations": (
                self.accepted_user_observations
            ),
            "memory_changed": (
                self.memory_changed
            ),
            "analysis_errors": list(
                self.analysis_errors
            ),
            "runtime": (
                dict(
                    self.runtime
                )
                if self.runtime
                is not None
                else None
            ),
            "timings": dict(
                self.timings
            ),
        }

    def text(
        self,
    ) -> str:
        parts = [
            "Cortex diagnosis: "
            + self.summary
        ]
        if self.analysis_errors:
            parts.append(
                "Structural analysis was partial: "
                + "; ".join(
                    self.analysis_errors
                )
                + "."
            )
        return " ".join(
            parts
        )


@dataclass(frozen=True)
class HybridGatewayConfig:
    chat_model: str
    parser_model: str
    ollama_url: str = (
        "http://localhost:11434"
    )
    feature_dim: int = 128
    max_entities: int = 96
    memory_budget: int = 24
    history_messages: int = 12
    diagnostics_default: bool = False
    persistence_dir: str | None = None
    trace_stages: bool = False
    include_diagnosis: bool = True
    embedding_model: str = (
        SYMBOLIC_ENCODER_ID
    )


def _resolve_candidate_ref(
    reference: str,
    entities: Sequence[
        CandidateEntity
    ],
) -> str:
    raw = reference.strip()
    if not raw:
        raise ValueError(
            "candidate claim reference is empty"
        )
    by_id = {
        entity.local_id: (
            entity.surface
        )
        for entity in entities
    }
    if raw in by_id:
        return by_id[
            raw
        ]
    matches = [
        entity.surface
        for entity in entities
        if _norm(
            entity.surface
        ) == _norm(
            raw
        )
    ]
    if len(
        matches
    ) == 1:
        return matches[
            0
        ]
    raise ValueError(
        "candidate claim references an unknown or ambiguous entity"
    )


def parse_candidate_answer(
    raw: str,
    *,
    source_text: str,
) -> CandidateAnswerIR:
    try:
        payload = loads(
            raw
        )
    except Exception as exc:
        raise ValueError(
            "candidate answer extraction is not valid JSON"
        ) from exc
    if not isinstance(
        payload,
        dict,
    ):
        raise ValueError(
            "candidate answer extraction must be an object"
        )

    raw_entities = payload.get(
        "entities",
        [],
    )
    raw_claims = payload.get(
        "claims",
        [],
    )
    if (
        not isinstance(
            raw_entities,
            list,
        )
        or not isinstance(
            raw_claims,
            list,
        )
    ):
        raise ValueError(
            "candidate answer extraction arrays are malformed"
        )

    entities: list[
        CandidateEntity
    ] = []
    ids: set[
        str
    ] = set()
    for item in raw_entities:
        if not isinstance(
            item,
            dict,
        ):
            raise ValueError(
                "candidate entity must be an object"
            )
        entity = CandidateEntity(
            local_id=str(
                item.get(
                    "id",
                    "",
                )
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
            or not _anchored(
                entity.evidence,
                source_text,
            )
        ):
            raise ValueError(
                "candidate entity is malformed or unanchored"
            )
        ids.add(
            entity.local_id
        )
        entities.append(
            entity
        )

    claims: list[
        CandidateClaim
    ] = []
    for item in raw_claims:
        if not isinstance(
            item,
            dict,
        ):
            raise ValueError(
                "candidate claim must be an object"
            )
        predicate = _predicate(
            str(
                item.get(
                    "predicate",
                    "",
                )
            )
        )
        evidence = str(
            item.get(
                "evidence",
                "",
            )
        ).strip()
        if (
            not predicate
            or not _anchored(
                evidence,
                source_text,
            )
        ):
            raise ValueError(
                "candidate claim is malformed or unanchored"
            )
        claims.append(
            CandidateClaim(
                subject=(
                    _resolve_candidate_ref(
                        str(
                            item.get(
                                "subject",
                                "",
                            )
                        ),
                        entities,
                    )
                ),
                predicate=predicate,
                object=(
                    _resolve_candidate_ref(
                        str(
                            item.get(
                                "object",
                                "",
                            )
                        ),
                        entities,
                    )
                ),
                asserted=bool(
                    item.get(
                        "asserted",
                        True,
                    )
                ),
                evidence=evidence,
            )
        )

    return CandidateAnswerIR(
        entities=tuple(
            entities
        ),
        claims=tuple(
            claims
        ),
    )


class HybridFileSessionStore:
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
        ).strip(
            "_"
        )[:48]
        return (
            f"{readable or 'session'}-v3-{digest}.json"
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
        sessions: list[
            str
        ] = []
        for path in sorted(
            self.root.glob(
                "*-v3-*.json"
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
                == HYBRID_SESSION_SCHEMA
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
        session: "HybridCortexChatSession",
    ) -> Path:
        target = self.path_for(
            session.session_id
        )
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=str(
                self.root
            ),
            prefix=".cortex-v3-",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(
                dumps(
                    session.persistence_packet(),
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
    ) -> Mapping[
        str,
        Any,
    ]:
        path = self.path_for(
            session_id
        )
        if not path.is_file():
            raise FileNotFoundError(
                f"no persisted hybrid Cortex session: {session_id}"
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
            != HYBRID_SESSION_SCHEMA
        ):
            raise ValueError(
                "unsupported hybrid Cortex session schema"
            )
        if payload.get(
            "session_id"
        ) != session_id:
            raise ValueError(
                "hybrid Cortex session ID mismatch"
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


class HybridCortexChatSession(
    StrictCortexChatSession
):
    config: HybridGatewayConfig

    def __init__(
        self,
        session_id: str,
        config: HybridGatewayConfig,
        client: LanguageModelClient,
    ) -> None:
        # StrictCortexChatSession only relies on fields also provided by
        # HybridGatewayConfig for structural state/persistence.
        super().__init__(
            session_id,
            config,  # type: ignore[arg-type]
            client,
        )
        self.encoder = (
            DeterministicSymbolicEncoder(
                config.feature_dim
            )
        )

    def _trace_hybrid(
        self,
        stage: str,
        duration: float | None = None,
    ) -> None:
        if not self.config.trace_stages:
            return
        suffix = (
            ""
            if duration is None
            else f" ({duration:.3f}s)"
        )
        print(
            f"[cortex-hybrid:{self.session_id}] "
            f"{stage}{suffix}",
            flush=True,
        )

    def _knowledge_answer(
        self,
        text: str,
    ) -> str:
        messages = [
            *self.history[
                -self.config.history_messages :
            ],
            {
                "role": "user",
                "content": text,
            },
        ]
        return self.client.chat(
            model=self.config.chat_model,
            messages=messages,
        ).strip()

    def _candidate_ir(
        self,
        answer: str,
    ) -> CandidateAnswerIR:
        raw = self.client.chat(
            model=self.config.parser_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        CANDIDATE_SYSTEM_PROMPT
                    ),
                },
                {
                    "role": "user",
                    "content": answer,
                },
            ],
            format_schema=(
                CANDIDATE_SCHEMA
            ),
        )
        return parse_candidate_answer(
            raw,
            source_text=answer,
        )

    def _persistent_id_for_surface(
        self,
        surface: str,
    ) -> int | None:
        normalized = _norm(
            surface
        )
        matches = [
            entity.cortex_id
            for entity
            in self.entities.values()
            if any(
                _norm(
                    name
                ) == normalized
                for name
                in entity.names
            )
        ]
        unique = sorted(
            set(
                matches
            )
        )
        if len(
            unique
        ) == 1:
            return unique[
                0
            ]
        return None

    def _assess_candidate(
        self,
        candidate: CandidateAnswerIR,
    ) -> tuple[
        ClaimAssessment,
        ...,
    ]:
        assessments: list[
            ClaimAssessment
        ] = []
        for claim in candidate.claims:
            source_id = (
                self._persistent_id_for_surface(
                    claim.subject
                )
            )
            target_id = (
                self._persistent_id_for_surface(
                    claim.object
                )
            )
            matching = []
            if (
                source_id is not None
                and target_id is not None
            ):
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
                        and key_target
                        == target_id
                        and key_predicate
                        == claim.predicate
                    )
                ]

            if not matching:
                assessments.append(
                    ClaimAssessment(
                        claim=claim,
                        status="neural_prior",
                    )
                )
                continue

            agreeing = [
                relation
                for relation in matching
                if relation.asserted
                == claim.asserted
            ]
            if agreeing:
                assessments.append(
                    ClaimAssessment(
                        claim=claim,
                        status="corroborated",
                        cortex_evidence=tuple(
                            relation.evidence
                            for relation
                            in agreeing
                        ),
                    )
                )
            else:
                assessments.append(
                    ClaimAssessment(
                        claim=claim,
                        status="conflict",
                        cortex_evidence=tuple(
                            relation.evidence
                            for relation
                            in matching
                        ),
                    )
                )
        return tuple(
            assessments
        )

    def _diagnose(
        self,
        *,
        prompt_ir: PromptIR | None,
        candidate: CandidateAnswerIR | None,
        cortex_answer: CortexAnswerIR | None,
        read: Any | None,
        errors: Sequence[str],
        timings: Mapping[str, float],
    ) -> HybridDiagnosis:
        assessments = (
            self._assess_candidate(
                candidate
            )
            if candidate is not None
            else tuple()
        )
        conflicts = [
            item
            for item in assessments
            if item.status
            == "conflict"
        ]
        corroborated = [
            item
            for item in assessments
            if item.status
            == "corroborated"
        ]
        neural_only = [
            item
            for item in assessments
            if item.status
            == "neural_prior"
        ]
        cortex_claims = (
            tuple(
                claim.packet()
                for claim
                in cortex_answer.claims
            )
            if cortex_answer
            is not None
            else tuple()
        )
        accepted = (
            len(
                prompt_ir.observations
            )
            if prompt_ir
            is not None
            else 0
        )
        memory_changed = (
            accepted
            > 0
            or bool(
                getattr(
                    read,
                    "revision",
                    False,
                )
            )
            or bool(
                getattr(
                    read,
                    "discovered",
                    False,
                )
            )
        )

        if conflicts:
            status = "conflict"
            reasoning_mode = (
                "neural_cortex_conflict"
            )
            summary = (
                "The neural model supplied the answer, but Cortex found "
                f"{len(conflicts)} claim(s) that conflict with stored "
                "session evidence. Cortex did not automatically replace "
                "either source."
            )
        elif (
            cortex_claims
            and corroborated
        ):
            status = "supported"
            reasoning_mode = (
                "neural_plus_cortex"
            )
            summary = (
                "The neural model supplied the answer and Cortex independently "
                "found compatible support in its structural session state."
            )
        elif cortex_claims:
            status = "cortex_supported"
            reasoning_mode = (
                "cortex_structural_reasoning"
            )
            summary = (
                "The neural model supplied the conversational answer. Cortex "
                "also produced a structural result from its current session "
                "state; the structured result is recorded separately below."
            )
        elif candidate is None:
            status = (
                "analysis_partial"
            )
            reasoning_mode = (
                "neural_answer_only"
            )
            summary = (
                "The neural model supplied the answer. Cortex could not "
                "reliably structure the candidate answer, so it made no "
                "claim-level judgment about it."
            )
        elif neural_only:
            status = "neural_prior"
            reasoning_mode = (
                "pretrained_knowledge"
            )
            summary = (
                "The neural model supplied the factual answer from pretrained "
                "knowledge. Cortex found no matching stored session evidence, "
                "so it treated those claims as neural priors and did not write "
                "them into memory."
            )
        else:
            status = "no_structural_claim"
            reasoning_mode = (
                "conversational"
            )
            summary = (
                "The neural model supplied the answer. Cortex found no explicit "
                "binary factual claim requiring structural corroboration or "
                "conflict checking."
            )

        if accepted:
            summary += (
                f" Cortex accepted {accepted} explicit user observation(s) "
                "from this turn into its structural state."
            )

        runtime = (
            {
                "unresolved": bool(
                    read.unresolved
                ),
                "revision": bool(
                    read.revision
                ),
                "reactivated": bool(
                    read.reactivated
                ),
                "discovered": bool(
                    read.discovered
                ),
                "budget_pressure": bool(
                    read.budget_pressure
                ),
                "stored": int(
                    read.stored
                ),
            }
            if read is not None
            else None
        )

        return HybridDiagnosis(
            status=status,
            reasoning_mode=(
                reasoning_mode
            ),
            summary=summary,
            neural_claims=assessments,
            cortex_claims=(
                cortex_claims
            ),
            accepted_user_observations=(
                accepted
            ),
            memory_changed=(
                memory_changed
            ),
            analysis_errors=tuple(
                errors
            ),
            runtime=runtime,
            timings=dict(
                timings
            ),
        )

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

        timings: dict[
            str,
            float,
        ] = {}
        errors: list[
            str
        ] = []

        self._trace_hybrid(
            "knowledge:start"
        )
        started = perf_counter()
        neural_answer = (
            self._knowledge_answer(
                text
            )
        )
        timings["knowledge"] = (
            perf_counter()
            - started
        )
        self._trace_hybrid(
            "knowledge:done",
            timings[
                "knowledge"
            ],
        )
        if not neural_answer:
            neural_answer = (
                "I do not have a usable neural response for this turn."
            )

        prompt_ir: (
            PromptIR
            | None
        ) = None
        read: Any | None = None
        bindings: Mapping[
            str,
            int,
        ] = {}
        cortex_answer: (
            CortexAnswerIR
            | None
        ) = None

        self._trace_hybrid(
            "cortex-prompt:start"
        )
        started = perf_counter()
        try:
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
            cortex_answer = (
                self._answer_ir(
                    prompt_ir,
                    bindings,
                    diagnostic,
                )
            )
        except Exception as exc:
            diagnostic = None
            errors.append(
                "prompt/Cortex analysis: "
                + str(
                    exc
                )
            )
        timings[
            "cortex_prompt"
        ] = (
            perf_counter()
            - started
        )
        self._trace_hybrid(
            "cortex-prompt:done",
            timings[
                "cortex_prompt"
            ],
        )

        candidate: (
            CandidateAnswerIR
            | None
        ) = None
        self._trace_hybrid(
            "candidate:start"
        )
        started = perf_counter()
        try:
            candidate = (
                self._candidate_ir(
                    neural_answer
                )
            )
        except Exception as exc:
            errors.append(
                "candidate analysis: "
                + str(
                    exc
                )
            )
        timings[
            "candidate"
        ] = (
            perf_counter()
            - started
        )
        self._trace_hybrid(
            "candidate:done",
            timings[
                "candidate"
            ],
        )

        diagnosis = self._diagnose(
            prompt_ir=prompt_ir,
            candidate=candidate,
            cortex_answer=(
                cortex_answer
            ),
            read=read,
            errors=errors,
            timings=timings,
        )

        content = neural_answer
        if self.config.include_diagnosis:
            content = (
                neural_answer.rstrip()
                + "\n\n"
                + diagnosis.text()
            ).strip()

        # The knowledge model sees only its own prior answers on subsequent
        # turns. Cortex diagnoses never feed back into neural conversation.
        self.history.extend(
            [
                {
                    "role": "user",
                    "content": text,
                },
                {
                    "role": "assistant",
                    "content": (
                        neural_answer
                    ),
                },
            ]
        )
        self.history[:] = (
            self.history[
                -self.config.history_messages :
            ]
        )

        return ChatResult(
            response_id=(
                "chatcmpl-"
                + uuid4().hex
            ),
            session_id=(
                self.session_id
            ),
            model="cortex",
            content=content,
            cortex_context={
                "gateway": (
                    HYBRID_GATEWAY_VERSION
                ),
                "neural_answer": (
                    neural_answer
                ),
                "prompt_ir": (
                    prompt_ir_packet(
                        prompt_ir
                    )
                    if prompt_ir
                    is not None
                    else None
                ),
                "candidate_ir": (
                    candidate.packet()
                    if candidate
                    is not None
                    else None
                ),
                "cortex_answer_ir": (
                    cortex_answer.packet()
                    if cortex_answer
                    is not None
                    else None
                ),
                "diagnosis": (
                    diagnosis.packet()
                ),
            },
            diagnostics=diagnostic,
        )

    def persistence_packet(
        self,
    ) -> dict[str, Any]:
        packet = (
            super().persistence_packet()
        )
        packet[
            "schema"
        ] = HYBRID_SESSION_SCHEMA
        packet[
            "gateway_version"
        ] = HYBRID_GATEWAY_VERSION
        packet[
            "chat_model"
        ] = self.config.chat_model
        packet[
            "parser_model"
        ] = self.config.parser_model
        return packet


class HybridCortexLanguageGateway:
    def __init__(
        self,
        config: HybridGatewayConfig,
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
            HybridCortexChatSession,
        ] = {}
        self.store = (
            HybridFileSessionStore(
                config.persistence_dir
            )
            if config.persistence_dir
            else None
        )

    def _new_session(
        self,
        session_id: str,
    ) -> HybridCortexChatSession:
        return HybridCortexChatSession(
            session_id,
            self.config,
            self.client,
        )

    def session(
        self,
        session_id: str,
    ) -> HybridCortexChatSession:
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
        result = session.chat(
            user_text,
            diagnostics=diagnostics,
        )
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
    ) -> HybridCortexChatSession:
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
