"""Cortex Cognitive Orchestrator v0.1.

The orchestrator makes Cortex and the neural model bidirectional collaborators:

1. Cortex performs a structural pre-pass over the user turn.
2. An adaptive neural policy selects the base model or a pre-registered adapter.
3. The neural model answers with bounded, provenance-labelled Cortex context.
4. Cortex checks the neural candidate.
5. One constrained neural revision is allowed when Cortex finds a conflict.
6. The adaptive interface emits a reversible learning proposal.

v0.1 never trains or mutates base-model weights. Adapter selection is limited
to pre-existing model variants. Learning proposals are advisory and require
explicit future implementation/approval before any parameter update.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from json import dumps, loads
from os import replace as atomic_replace
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import perf_counter
from typing import Any, Mapping, Sequence
from uuid import uuid4

from .hybrid_language_gateway import (
    CandidateAnswerIR,
    HybridCortexChatSession,
    HybridDiagnosis,
    HybridGatewayConfig,
)
from .language_gateway import (
    ChatResult,
    DiagnosticReport,
    LanguageModelClient,
    OllamaClient,
    diagnose_runtime,
)
from .strict_language_gateway import (
    CortexAnswerIR,
    PromptIR,
    prompt_ir_packet,
)


ORCHESTRATOR_VERSION = "cortex-cognitive-orchestrator-v0.1"
ORCHESTRATOR_SESSION_SCHEMA = "cortex-language-session-v4"


@dataclass(frozen=True)
class LearningProposal:
    action: str
    reason: str
    evidence_confirmations: int
    requires_approval: bool
    base_weights_mutated: bool = False

    def packet(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "reason": self.reason,
            "evidence_confirmations": (
                self.evidence_confirmations
            ),
            "requires_approval": (
                self.requires_approval
            ),
            "base_weights_mutated": (
                self.base_weights_mutated
            ),
        }


@dataclass(frozen=True)
class NeuralActivation:
    route: str
    model: str
    adapter_active: bool

    def packet(self) -> dict[str, Any]:
        return {
            "route": self.route,
            "model": self.model,
            "adapter_active": (
                self.adapter_active
            ),
        }


@dataclass(frozen=True)
class OrchestrationTrace:
    route: str
    cortex_prepass: bool
    cortex_context_used: bool
    candidate_checked: bool
    revision_cycles: int
    selected_model: str
    adapter_active: bool

    def packet(self) -> dict[str, Any]:
        return {
            "route": self.route,
            "cortex_prepass": (
                self.cortex_prepass
            ),
            "cortex_context_used": (
                self.cortex_context_used
            ),
            "candidate_checked": (
                self.candidate_checked
            ),
            "revision_cycles": (
                self.revision_cycles
            ),
            "selected_model": (
                self.selected_model
            ),
            "adapter_active": (
                self.adapter_active
            ),
        }


@dataclass(frozen=True)
class CognitiveOrchestratorConfig(
    HybridGatewayConfig
):
    max_revision_cycles: int = 1
    adaptation_min_confirmations: int = 3
    adapter_models: Mapping[
        str,
        str,
    ] = field(
        default_factory=dict
    )
    allow_adapter_selection: bool = True
    allow_weight_updates: bool = False


class AdaptiveNeuralInterface:
    """Reversible neural-control policy.

    v0.1 can select only pre-existing adapter/model variants. It never invokes
    training and never mutates base weights.
    """

    def __init__(
        self,
        config: CognitiveOrchestratorConfig,
    ) -> None:
        self.config = config

    def activate(
        self,
        route: str,
    ) -> NeuralActivation:
        model = self.config.chat_model
        adapter_active = False
        if (
            self.config.allow_adapter_selection
        ):
            candidate = (
                self.config.adapter_models.get(
                    route
                )
            )
            if candidate:
                model = candidate
                adapter_active = (
                    model
                    != self.config.chat_model
                )
        return NeuralActivation(
            route=route,
            model=model,
            adapter_active=(
                adapter_active
            ),
        )

    def learning_proposal(
        self,
        *,
        diagnosis: HybridDiagnosis,
        max_confirmations: int,
    ) -> LearningProposal:
        runtime = diagnosis.runtime or {}

        if diagnosis.analysis_errors:
            return LearningProposal(
                action="blocked",
                reason=(
                    "Cortex analysis was partial; "
                    "neural adaptation is not justified."
                ),
                evidence_confirmations=(
                    max_confirmations
                ),
                requires_approval=True,
            )

        if (
            diagnosis.status
            == "conflict"
        ):
            return LearningProposal(
                action="blocked",
                reason=(
                    "Neural and Cortex evidence conflict; "
                    "preserve disagreement before learning."
                ),
                evidence_confirmations=(
                    max_confirmations
                ),
                requires_approval=True,
            )

        if (
            runtime.get(
                "unresolved"
            )
            or runtime.get(
                "budget_pressure"
            )
        ):
            return LearningProposal(
                action="blocked",
                reason=(
                    "Cortex state is unresolved or under "
                    "memory-budget pressure."
                ),
                evidence_confirmations=(
                    max_confirmations
                ),
                requires_approval=True,
            )

        if max_confirmations >= int(
            self.config.adaptation_min_confirmations
        ):
            return LearningProposal(
                action="adapter_candidate",
                reason=(
                    "Repeated explicit evidence reached the "
                    "adapter-candidate threshold. v0.1 records "
                    "the proposal but performs no training."
                ),
                evidence_confirmations=(
                    max_confirmations
                ),
                requires_approval=True,
                base_weights_mutated=False,
            )

        if (
            diagnosis.accepted_user_observations
            > 0
        ):
            return LearningProposal(
                action="context_only",
                reason=(
                    "New explicit evidence is retained in Cortex "
                    "memory but has not earned neural adaptation."
                ),
                evidence_confirmations=(
                    max_confirmations
                ),
                requires_approval=False,
            )

        return LearningProposal(
            action="none",
            reason=(
                "No new explicit evidence requires neural adaptation."
            ),
            evidence_confirmations=(
                max_confirmations
            ),
            requires_approval=False,
        )


class CognitiveFileSessionStore:
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
            f"{readable or 'session'}-v4-{digest}.json"
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

    def save(
        self,
        session: "CognitiveOrchestratorSession",
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
            prefix=".cortex-v4-",
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
    ) -> Mapping[str, Any]:
        path = self.path_for(
            session_id
        )
        if not path.is_file():
            raise FileNotFoundError(
                "no persisted cognitive Cortex session: "
                + session_id
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
            != ORCHESTRATOR_SESSION_SCHEMA
        ):
            raise ValueError(
                "unsupported cognitive Cortex session schema"
            )
        if payload.get(
            "session_id"
        ) != session_id:
            raise ValueError(
                "cognitive Cortex session ID mismatch"
            )
        return payload

    def list_sessions(
        self,
    ) -> list[str]:
        sessions: list[str] = []
        for path in sorted(
            self.root.glob(
                "*-v4-*.json"
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
                == ORCHESTRATOR_SESSION_SCHEMA
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


class CognitiveOrchestratorSession(
    HybridCortexChatSession
):
    config: CognitiveOrchestratorConfig

    def __init__(
        self,
        session_id: str,
        config: CognitiveOrchestratorConfig,
        client: LanguageModelClient,
    ) -> None:
        super().__init__(
            session_id,
            config,
            client,
        )
        self.adaptive = (
            AdaptiveNeuralInterface(
                config
            )
        )
        self.evidence_ledger: dict[
            str,
            int,
        ] = {}

    def _trace_orchestrator(
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
            f"[cortex-orchestrator:{self.session_id}] "
            f"{stage}{suffix}",
            flush=True,
        )

    def _route(
        self,
        prompt_ir: PromptIR | None,
        cortex_answer: CortexAnswerIR | None,
    ) -> str:
        if (
            cortex_answer is not None
            and cortex_answer.claims
        ):
            return "mixed_structural"
        if (
            prompt_ir is not None
            and prompt_ir.observations
        ):
            return "memory_update"
        if (
            prompt_ir is not None
            and prompt_ir.query_operation
            in {
                "relation_lookup",
                "entity_recall",
                "introspection",
            }
        ):
            return "mixed_query"
        return "neural_knowledge"

    def _cortex_context(
        self,
        *,
        prompt_ir: PromptIR | None,
        cortex_answer: CortexAnswerIR | None,
        read: Any | None,
    ) -> dict[str, Any]:
        return {
            "provenance": (
                "cortex_session_state"
            ),
            "query_operation": (
                prompt_ir.query_operation
                if prompt_ir
                is not None
                else "none"
            ),
            "supported_claims": (
                [
                    claim.packet()
                    for claim
                    in cortex_answer.claims
                ]
                if cortex_answer
                is not None
                else []
            ),
            "status": (
                cortex_answer.status
                if cortex_answer
                is not None
                else "unavailable"
            ),
            "runtime": (
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
                }
                if read is not None
                else None
            ),
        }

    def _answer_with_context(
        self,
        *,
        text: str,
        activation: NeuralActivation,
        cortex_context: Mapping[str, Any],
    ) -> str:
        context_json = dumps(
            dict(
                cortex_context
            ),
            sort_keys=True,
        )
        system = (
            "You are the neural component of Cortex Cognitive Orchestrator. "
            "Answer the user's request normally using your pretrained knowledge "
            "and language abilities. You may also use the supplied Cortex "
            "session context as provenance-labelled session evidence. "
            "Do not present Cortex session evidence as universal fact. "
            "If it conflicts with your pretrained expectation, explicitly "
            "preserve the disagreement rather than silently overriding it.\n\n"
            "CORTEX_SESSION_CONTEXT="
            + context_json
        )
        messages = [
            {
                "role": "system",
                "content": system,
            },
            *self.history[
                -self.config.history_messages :
            ],
            {
                "role": "user",
                "content": text,
            },
        ]
        return self.client.chat(
            model=activation.model,
            messages=messages,
        ).strip()

    def _revision_answer(
        self,
        *,
        text: str,
        candidate: str,
        activation: NeuralActivation,
        diagnosis: HybridDiagnosis,
    ) -> str:
        conflict_rows = [
            item.packet()
            for item
            in diagnosis.neural_claims
            if item.status
            == "conflict"
        ]
        system = (
            "You are revising a candidate answer after Cortex performed an "
            "independent structural check. Preserve useful pretrained knowledge, "
            "but do not ignore Cortex session evidence. If the evidence is "
            "genuinely inconsistent, state the conflict or uncertainty rather "
            "than pretending one side was proven. Return only the revised answer."
        )
        user = dumps(
            {
                "user_request": text,
                "candidate_answer": (
                    candidate
                ),
                "cortex_review": {
                    "status": (
                        diagnosis.status
                    ),
                    "summary": (
                        diagnosis.summary
                    ),
                    "conflicts": (
                        conflict_rows
                    ),
                },
            },
            sort_keys=True,
        )
        return self.client.chat(
            model=activation.model,
            messages=[
                {
                    "role": "system",
                    "content": system,
                },
                {
                    "role": "user",
                    "content": user,
                },
            ],
        ).strip()

    def _evidence_key(
        self,
        prompt_ir: PromptIR,
        observation: Any,
    ) -> str:
        entities = {
            entity.local_id: (
                entity.surface
            )
            for entity
            in prompt_ir.entities
        }
        return "|".join(
            (
                entities.get(
                    observation.subject,
                    observation.subject,
                ).casefold(),
                observation.predicate.casefold(),
                entities.get(
                    observation.object,
                    observation.object,
                ).casefold(),
                (
                    "1"
                    if observation.asserted
                    else "0"
                ),
            )
        )

    def _record_evidence(
        self,
        prompt_ir: PromptIR | None,
    ) -> int:
        if prompt_ir is None:
            return 0
        maximum = 0
        for observation in (
            prompt_ir.observations
        ):
            key = self._evidence_key(
                prompt_ir,
                observation,
            )
            self.evidence_ledger[
                key
            ] = (
                self.evidence_ledger.get(
                    key,
                    0,
                )
                + 1
            )
            maximum = max(
                maximum,
                self.evidence_ledger[
                    key
                ],
            )
        return maximum

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
        errors: list[str] = []

        prompt_ir: PromptIR | None = None
        read: Any | None = None
        bindings: Mapping[
            str,
            int,
        ] = {}
        cortex_answer: (
            CortexAnswerIR
            | None
        ) = None
        diagnostic: (
            DiagnosticReport
            | None
        ) = None

        self._trace_orchestrator(
            "cortex-prepass:start"
        )
        started = perf_counter()
        try:
            prompt_ir = self._parse(
                text
            )
            read, bindings = self._observe(
                prompt_ir
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
            cortex_answer = self._answer_ir(
                prompt_ir,
                bindings,
                diagnostic,
            )
        except Exception as exc:
            errors.append(
                "Cortex pre-pass: "
                + str(
                    exc
                )
            )
        timings[
            "cortex_prepass"
        ] = (
            perf_counter()
            - started
        )
        self._trace_orchestrator(
            "cortex-prepass:done",
            timings[
                "cortex_prepass"
            ],
        )

        route = self._route(
            prompt_ir,
            cortex_answer,
        )
        activation = (
            self.adaptive.activate(
                route
            )
        )
        cortex_context = (
            self._cortex_context(
                prompt_ir=prompt_ir,
                cortex_answer=(
                    cortex_answer
                ),
                read=read,
            )
        )

        self._trace_orchestrator(
            "neural:start"
        )
        started = perf_counter()
        neural_answer = (
            self._answer_with_context(
                text=text,
                activation=activation,
                cortex_context=(
                    cortex_context
                ),
            )
        )
        timings["neural"] = (
            perf_counter()
            - started
        )
        self._trace_orchestrator(
            "neural:done",
            timings[
                "neural"
            ],
        )
        if not neural_answer:
            neural_answer = (
                "I do not have a usable neural response for this turn."
            )

        candidate: (
            CandidateAnswerIR
            | None
        ) = None
        self._trace_orchestrator(
            "candidate-check:start"
        )
        started = perf_counter()
        try:
            candidate = self._candidate_ir(
                neural_answer
            )
        except Exception as exc:
            errors.append(
                "Candidate analysis: "
                + str(
                    exc
                )
            )
        timings[
            "candidate_check"
        ] = (
            perf_counter()
            - started
        )
        self._trace_orchestrator(
            "candidate-check:done",
            timings[
                "candidate_check"
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

        revision_cycles = 0
        while (
            diagnosis.status
            == "conflict"
            and revision_cycles
            < max(
                0,
                int(
                    self.config.max_revision_cycles
                ),
            )
        ):
            revision_cycles += 1
            self._trace_orchestrator(
                "neural-revision:start"
            )
            started = perf_counter()
            revised = (
                self._revision_answer(
                    text=text,
                    candidate=(
                        neural_answer
                    ),
                    activation=(
                        activation
                    ),
                    diagnosis=(
                        diagnosis
                    ),
                )
            )
            timings[
                "neural_revision"
            ] = (
                timings.get(
                    "neural_revision",
                    0.0,
                )
                + (
                    perf_counter()
                    - started
                )
            )
            self._trace_orchestrator(
                "neural-revision:done",
                timings[
                    "neural_revision"
                ],
            )
            if revised:
                neural_answer = revised

            candidate = None
            try:
                candidate = (
                    self._candidate_ir(
                        neural_answer
                    )
                )
            except Exception as exc:
                errors.append(
                    "Revised candidate analysis: "
                    + str(
                        exc
                    )
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

        max_confirmations = (
            self._record_evidence(
                prompt_ir
            )
        )
        learning = (
            self.adaptive.learning_proposal(
                diagnosis=diagnosis,
                max_confirmations=(
                    max_confirmations
                ),
            )
        )

        trace = OrchestrationTrace(
            route=route,
            cortex_prepass=(
                prompt_ir
                is not None
            ),
            cortex_context_used=bool(
                cortex_context.get(
                    "supported_claims"
                )
                or cortex_context.get(
                    "runtime"
                )
            ),
            candidate_checked=(
                candidate
                is not None
            ),
            revision_cycles=(
                revision_cycles
            ),
            selected_model=(
                activation.model
            ),
            adapter_active=(
                activation.adapter_active
            ),
        )

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
            content=neural_answer,
            cortex_context={
                "gateway": (
                    ORCHESTRATOR_VERSION
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
                "cortex_answer_ir": (
                    cortex_answer.packet()
                    if cortex_answer
                    is not None
                    else None
                ),
                "candidate_ir": (
                    candidate.packet()
                    if candidate
                    is not None
                    else None
                ),
                "diagnosis": (
                    diagnosis.packet()
                ),
                "orchestration": (
                    trace.packet()
                ),
                "adaptive_neural": {
                    "activation": (
                        activation.packet()
                    ),
                    "learning": (
                        learning.packet()
                    ),
                    "weight_updates_enabled": (
                        self.config.allow_weight_updates
                    ),
                },
            },
            diagnostics=diagnostic,
        )

    def persistence_packet(
        self,
    ) -> dict[str, Any]:
        packet = super().persistence_packet()
        packet[
            "schema"
        ] = ORCHESTRATOR_SESSION_SCHEMA
        packet[
            "gateway_version"
        ] = ORCHESTRATOR_VERSION
        packet[
            "evidence_ledger"
        ] = dict(
            self.evidence_ledger
        )
        return packet

    def restore_packet(
        self,
        payload: Mapping[str, Any],
    ) -> None:
        super().restore_packet(
            payload
        )
        raw = payload.get(
            "evidence_ledger",
            {},
        )
        if not isinstance(
            raw,
            dict,
        ):
            raise ValueError(
                "orchestrator evidence ledger must be an object"
            )
        self.evidence_ledger = {
            str(
                key
            ): int(
                value
            )
            for key, value
            in raw.items()
        }


class CognitiveOrchestratorGateway:
    def __init__(
        self,
        config: CognitiveOrchestratorConfig,
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
            CognitiveOrchestratorSession,
        ] = {}
        self.store = (
            CognitiveFileSessionStore(
                config.persistence_dir
            )
            if config.persistence_dir
            else None
        )

    def _new_session(
        self,
        session_id: str,
    ) -> CognitiveOrchestratorSession:
        return CognitiveOrchestratorSession(
            session_id,
            self.config,
            self.client,
        )

    def session(
        self,
        session_id: str,
    ) -> CognitiveOrchestratorSession:
        normalized = session_id.strip()
        if not normalized:
            raise ValueError(
                "session_id must not be empty"
            )
        existing = self.sessions.get(
            normalized
        )
        if existing is not None:
            return existing
        session = self._new_session(
            normalized
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
    ) -> CognitiveOrchestratorSession:
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
