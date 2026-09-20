"""Cortex Hybrid Core v0.1.

This is the provider-independent hybrid cognition path.

Normal turn:
    text -> structured PromptIR -> Cortex pre-pass
         -> one neural turn {answer + claim sidecar}
         -> Cortex assessment

The answer and its candidate claims are generated together so Cortex does not
need a third neural call merely to parse the neural model's own answer.
"""
from __future__ import annotations

from dataclasses import dataclass
from json import dumps, loads
from time import perf_counter
from typing import Any, Mapping, Sequence
from uuid import uuid4

from .hybrid_language_gateway import (
    CandidateAnswerIR,
    ClaimAssessment,
    HybridCortexChatSession,
    HybridDiagnosis,
    HybridGatewayConfig,
    parse_candidate_answer,
)
from .language_gateway import (
    ChatResult,
    DiagnosticReport,
    diagnose_runtime,
)
from .neural_runtime import (
    NeuralResponse,
    NeuralRuntime,
    RuntimeClientAdapter,
)
from .strict_language_gateway import (
    CortexAnswerIR,
    PARSER_SYSTEM_PROMPT,
    PROMPT_IR_SCHEMA,
    PromptEntity,
    PromptIR,
    PromptObservation,
    parse_prompt_ir,
    prompt_ir_packet,
)


HYBRID_CORE_VERSION = "cortex-hybrid-core-v0.1"


HYBRID_TURN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "answer": {
            "type": "string",
        },
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {
                        "type": "string",
                    },
                    "surface": {
                        "type": "string",
                    },
                    "kind": {
                        "type": "string",
                    },
                    "evidence": {
                        "type": "string",
                    },
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
                    "subject": {
                        "type": "string",
                    },
                    "predicate": {
                        "type": "string",
                    },
                    "object": {
                        "type": "string",
                    },
                    "asserted": {
                        "type": "boolean",
                    },
                    "evidence": {
                        "type": "string",
                    },
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
        "answer",
        "entities",
        "claims",
    ],
}


HYBRID_TURN_SYSTEM_PROMPT = """You are the neural component of Cortex Hybrid Core.

Produce a natural answer and, in the same structured response, a factual claim
sidecar that describes only relation claims explicitly stated in your answer.

A CORTEX_CONTEXT packet may be supplied. It is provenance-labelled session
state, not universal truth. Use it when relevant. If it conflicts with your
pretrained expectation, preserve the disagreement instead of silently
overwriting either source.

Sidecar rules:
- Do not add facts that are absent from the answer.
- Entity IDs are local to this response.
- Every claim subject/object references a declared entity ID.
- Normalize predicates to lowercase snake_case.
- Every entity/claim evidence field is a short verbatim span from the answer.
- Empty entities/claims are valid for conversational or non-relational answers.
- Return exactly the supplied schema."""


HYBRID_REVISION_SYSTEM_PROMPT = """Revise a neural answer after Cortex found an
explicit structural conflict with session evidence.

Return a new natural answer plus a factual claim sidecar in the same schema.
Do not silently declare either source universally true. Preserve scope,
provenance, or uncertainty when the conflict cannot be resolved.

Sidecar claims must be explicitly stated by the revised answer, and evidence
must be copied verbatim from that answer. Return exactly the supplied schema."""


BOOTSTRAP_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "subject": {
                        "type": "string",
                    },
                    "subject_kind": {
                        "type": "string",
                    },
                    "predicate": {
                        "type": "string",
                    },
                    "object": {
                        "type": "string",
                    },
                    "object_kind": {
                        "type": "string",
                    },
                    "asserted": {
                        "type": "boolean",
                    },
                    "confidence": {
                        "type": "number",
                    },
                },
                "required": [
                    "subject",
                    "subject_kind",
                    "predicate",
                    "object",
                    "object_kind",
                    "asserted",
                    "confidence",
                ],
            },
        },
    },
    "required": [
        "claims",
    ],
}


BOOTSTRAP_SYSTEM_PROMPT = """You are compiling a small provisional knowledge
seed for Cortex from your pretrained neural knowledge.

Given a topic, emit only concise binary relation claims you consider strongly
supported by your pretrained knowledge. These are NOT verified truths: Cortex
will store them with provenance=neural_bootstrap and may later revise them.

Rules:
- Prefer stable, basic relations over opinions or time-sensitive claims.
- Use lowercase snake_case predicates.
- Confidence is your own heuristic estimate in [0,1], not calibrated truth.
- Do not exceed the requested claim limit.
- Return exactly the supplied schema."""


@dataclass(frozen=True)
class HybridTurnIR:
    answer: str
    candidate: CandidateAnswerIR

    def packet(
        self,
    ) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "candidate": (
                self.candidate.packet()
            ),
        }


@dataclass(frozen=True)
class BootstrapRecord:
    subject: str
    predicate: str
    object: str
    asserted: bool
    confidence: float
    topic: str
    model: str
    provenance: str = "neural_bootstrap"

    @property
    def key(
        self,
    ) -> str:
        return "|".join(
            (
                self.subject.casefold(),
                self.predicate.casefold(),
                self.object.casefold(),
                "1" if self.asserted else "0",
            )
        )

    def packet(
        self,
    ) -> dict[str, Any]:
        return {
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "asserted": self.asserted,
            "confidence": self.confidence,
            "topic": self.topic,
            "model": self.model,
            "provenance": self.provenance,
            "status": "provisional",
        }


@dataclass(frozen=True)
class HybridCoreConfig(
    HybridGatewayConfig
):
    max_revision_cycles: int = 1
    bootstrap_min_confidence: float = 0.75
    bootstrap_max_claims_per_topic: int = 12


def parse_hybrid_turn(
    raw: str,
) -> HybridTurnIR:
    try:
        payload = loads(
            raw
        )
    except Exception as exc:
        raise ValueError(
            "hybrid neural turn is not valid JSON"
        ) from exc
    if not isinstance(
        payload,
        dict,
    ):
        raise ValueError(
            "hybrid neural turn must be an object"
        )

    answer = str(
        payload.get(
            "answer",
            "",
        )
    ).strip()
    if not answer:
        raise ValueError(
            "hybrid neural turn has no answer"
        )

    candidate_raw = dumps(
        {
            "entities": payload.get(
                "entities",
                [],
            ),
            "claims": payload.get(
                "claims",
                [],
            ),
        }
    )
    candidate = parse_candidate_answer(
        candidate_raw,
        source_text=answer,
    )
    return HybridTurnIR(
        answer=answer,
        candidate=candidate,
    )


class HybridCoreSession(
    HybridCortexChatSession
):
    config: HybridCoreConfig

    def __init__(
        self,
        session_id: str,
        config: HybridCoreConfig,
        runtime: NeuralRuntime,
    ) -> None:
        self.neural_runtime = runtime
        self.neural_calls: list[
            Mapping[str, Any]
        ] = []
        self.bootstrap_records: dict[
            str,
            BootstrapRecord,
        ] = {}
        self.relation_provenance: dict[
            str,
            str,
        ] = {}
        super().__init__(
            session_id,
            config,
            RuntimeClientAdapter(
                runtime
            ),
        )

    @staticmethod
    def _provenance_key(
        source_id: int,
        target_id: int,
        predicate: str,
    ) -> str:
        return (
            str(
                source_id
            )
            + "|"
            + str(
                target_id
            )
            + "|"
            + normalize_predicate(
                predicate
            )
        )

    def _mark_observation_provenance(
        self,
        prompt_ir: PromptIR,
        bindings: Mapping[
            str,
            int,
        ],
        provenance: str,
    ) -> None:
        for observation in (
            prompt_ir.observations
        ):
            source_id = bindings.get(
                observation.subject
            )
            target_id = bindings.get(
                observation.object
            )
            if (
                source_id is None
                or target_id is None
            ):
                continue
            self.relation_provenance[
                self._provenance_key(
                    source_id,
                    target_id,
                    observation.predicate,
                )
            ] = provenance

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
            matching: list[
                tuple[
                    Any,
                    str,
                ]
            ] = []
            if (
                source_id is not None
                and target_id is not None
            ):
                for (
                    key_source,
                    key_target,
                    key_predicate,
                ), relation in self.relations.items():
                    if (
                        key_source
                        == source_id
                        and key_target
                        == target_id
                        and key_predicate
                        == claim.predicate
                    ):
                        provenance = (
                            self.relation_provenance.get(
                                self._provenance_key(
                                    key_source,
                                    key_target,
                                    key_predicate,
                                ),
                                "user_observation",
                            )
                        )
                        matching.append(
                            (
                                relation,
                                provenance,
                            )
                        )

            if not matching:
                assessments.append(
                    ClaimAssessment(
                        claim=claim,
                        status="neural_prior",
                    )
                )
                continue

            independent = [
                pair
                for pair in matching
                if pair[1]
                != "neural_bootstrap"
            ]
            bootstrap = [
                pair
                for pair in matching
                if pair[1]
                == "neural_bootstrap"
            ]

            if independent:
                agreeing = [
                    relation
                    for relation, _provenance
                    in independent
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
                                for relation, _provenance
                                in independent
                            ),
                        )
                    )
                continue

            agreeing_bootstrap = [
                relation
                for relation, _provenance
                in bootstrap
                if relation.asserted
                == claim.asserted
            ]
            assessments.append(
                ClaimAssessment(
                    claim=claim,
                    status=(
                        "bootstrap_aligned"
                        if agreeing_bootstrap
                        else "bootstrap_divergence"
                    ),
                    cortex_evidence=tuple(
                        relation.evidence
                        for relation, _provenance
                        in bootstrap
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
        diagnosis = super()._diagnose(
            prompt_ir=prompt_ir,
            candidate=candidate,
            cortex_answer=cortex_answer,
            read=read,
            errors=errors,
            timings=timings,
        )
        statuses = {
            assessment.status
            for assessment
            in diagnosis.neural_claims
        }
        if (
            diagnosis.status
            != "conflict"
            and "bootstrap_divergence"
            in statuses
        ):
            return HybridDiagnosis(
                status="bootstrap_divergence",
                reasoning_mode=(
                    "neural_vs_bootstrap_prior"
                ),
                summary=(
                    "The neural answer diverges from provisional "
                    "neural-bootstrap memory. This is not treated as "
                    "independent Cortex evidence, so no forced revision "
                    "is triggered."
                ),
                neural_claims=(
                    diagnosis.neural_claims
                ),
                cortex_claims=(
                    diagnosis.cortex_claims
                ),
                accepted_user_observations=(
                    diagnosis.accepted_user_observations
                ),
                memory_changed=(
                    diagnosis.memory_changed
                ),
                analysis_errors=(
                    diagnosis.analysis_errors
                ),
                runtime=diagnosis.runtime,
                timings=diagnosis.timings,
            )
        if (
            diagnosis.status
            not in {
                "conflict",
                "supported",
            }
            and "bootstrap_aligned"
            in statuses
        ):
            return HybridDiagnosis(
                status="bootstrap_aligned",
                reasoning_mode=(
                    "neural_bootstrap_alignment"
                ),
                summary=(
                    "The neural answer matches provisional "
                    "neural-bootstrap memory. This is alignment with "
                    "the neural prior, not independent corroboration."
                ),
                neural_claims=(
                    diagnosis.neural_claims
                ),
                cortex_claims=(
                    diagnosis.cortex_claims
                ),
                accepted_user_observations=(
                    diagnosis.accepted_user_observations
                ),
                memory_changed=(
                    diagnosis.memory_changed
                ),
                analysis_errors=(
                    diagnosis.analysis_errors
                ),
                runtime=diagnosis.runtime,
                timings=diagnosis.timings,
            )
        return diagnosis

    def _record_neural(
        self,
        response: NeuralResponse,
        *,
        stage: str,
    ) -> None:
        packet = response.packet()
        packet[
            "stage"
        ] = stage
        self.neural_calls.append(
            packet
        )

    def _parse_user_turn(
        self,
        text: str,
    ) -> tuple[
        PromptIR,
        NeuralResponse,
    ]:
        response = (
            self.neural_runtime.generate(
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
                        "content": text,
                    },
                ],
                schema=(
                    PROMPT_IR_SCHEMA
                ),
                temperature=0,
            )
        )
        self._record_neural(
            response,
            stage="prompt_parse",
        )
        return (
            parse_prompt_ir(
                response.text,
                source_text=text,
            ),
            response,
        )

    def _cortex_context(
        self,
        *,
        prompt_ir: PromptIR,
        cortex_answer: CortexAnswerIR,
        read: Any | None,
    ) -> dict[str, Any]:
        return {
            "provenance": (
                "cortex_session_state"
            ),
            "query": {
                "operation": (
                    prompt_ir.query_operation
                ),
                "predicate": (
                    prompt_ir.query_predicate
                ),
            },
            "claims": [
                claim.packet()
                for claim
                in cortex_answer.claims
            ],
            "answer_status": (
                cortex_answer.status
            ),
            "runtime": (
                {
                    "unresolved": bool(
                        getattr(
                            read,
                            "unresolved",
                            False,
                        )
                    ),
                    "revision": bool(
                        getattr(
                            read,
                            "revision",
                            False,
                        )
                    ),
                    "reactivated": bool(
                        getattr(
                            read,
                            "reactivated",
                            False,
                        )
                    ),
                    "discovered": bool(
                        getattr(
                            read,
                            "discovered",
                            False,
                        )
                    ),
                }
                if read is not None
                else None
            ),
            "bootstrap_claims": [
                record.packet()
                for record
                in self.bootstrap_records.values()
                if (
                    record.subject.casefold()
                    in text_tokens(
                        prompt_ir
                    )
                    or record.object.casefold()
                    in text_tokens(
                        prompt_ir
                    )
                )
            ][
                :8
            ],
        }

    def _generate_turn(
        self,
        *,
        text: str,
        cortex_context: Mapping[
            str,
            Any,
        ],
    ) -> tuple[
        HybridTurnIR,
        NeuralResponse,
    ]:
        system = (
            HYBRID_TURN_SYSTEM_PROMPT
            + "\n\nCORTEX_CONTEXT="
            + dumps(
                dict(
                    cortex_context
                ),
                sort_keys=True,
            )
        )
        response = (
            self.neural_runtime.generate(
                model=(
                    self.config.chat_model
                ),
                messages=[
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
                ],
                schema=(
                    HYBRID_TURN_SCHEMA
                ),
            )
        )
        self._record_neural(
            response,
            stage="answer_with_sidecar",
        )
        return (
            parse_hybrid_turn(
                response.text
            ),
            response,
        )

    def _revise_turn(
        self,
        *,
        text: str,
        prior: HybridTurnIR,
        diagnosis: HybridDiagnosis,
    ) -> tuple[
        HybridTurnIR,
        NeuralResponse,
    ]:
        conflicts = [
            assessment.packet()
            for assessment
            in diagnosis.neural_claims
            if (
                assessment.status
                == "conflict"
            )
        ]
        response = (
            self.neural_runtime.generate(
                model=(
                    self.config.chat_model
                ),
                messages=[
                    {
                        "role": "system",
                        "content": (
                            HYBRID_REVISION_SYSTEM_PROMPT
                        ),
                    },
                    {
                        "role": "user",
                        "content": dumps(
                            {
                                "user_request": text,
                                "candidate_answer": (
                                    prior.answer
                                ),
                                "cortex_conflicts": (
                                    conflicts
                                ),
                            },
                            sort_keys=True,
                        ),
                    },
                ],
                schema=(
                    HYBRID_TURN_SCHEMA
                ),
            )
        )
        self._record_neural(
            response,
            stage="conflict_revision",
        )
        return (
            parse_hybrid_turn(
                response.text
            ),
            response,
        )

    def bootstrap_topics(
        self,
        topics: Sequence[str],
    ) -> tuple[
        BootstrapRecord,
        ...,
    ]:
        imported: list[
            BootstrapRecord
        ] = []
        for topic in topics:
            normalized_topic = (
                topic.strip()
            )
            if not normalized_topic:
                continue
            response = (
                self.neural_runtime.generate(
                    model=(
                        self.config.chat_model
                    ),
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                BOOTSTRAP_SYSTEM_PROMPT
                            ),
                        },
                        {
                            "role": "user",
                            "content": (
                                "Topic: "
                                + normalized_topic
                                + "\nMaximum claims: "
                                + str(
                                    self.config.bootstrap_max_claims_per_topic
                                )
                            ),
                        },
                    ],
                    schema=(
                        BOOTSTRAP_SCHEMA
                    ),
                    temperature=0,
                )
            )
            self._record_neural(
                response,
                stage="bootstrap",
            )
            payload = loads(
                response.text
            )
            claims = (
                payload.get(
                    "claims",
                    [],
                )
                if isinstance(
                    payload,
                    dict,
                )
                else []
            )
            if not isinstance(
                claims,
                list,
            ):
                raise ValueError(
                    "bootstrap claims must be an array"
                )

            for index, item in enumerate(
                claims[
                    : self.config.bootstrap_max_claims_per_topic
                ]
            ):
                if not isinstance(
                    item,
                    dict,
                ):
                    continue
                subject = str(
                    item.get(
                        "subject",
                        "",
                    )
                ).strip()
                object_name = str(
                    item.get(
                        "object",
                        "",
                    )
                ).strip()
                predicate = normalize_predicate(
                    str(
                        item.get(
                            "predicate",
                            "",
                        )
                    )
                )
                confidence = float(
                    item.get(
                        "confidence",
                        0.0,
                    )
                )
                if (
                    not subject
                    or not object_name
                    or not predicate
                    or confidence
                    < float(
                        self.config.bootstrap_min_confidence
                    )
                ):
                    continue
                asserted = bool(
                    item.get(
                        "asserted",
                        True,
                    )
                )
                record = BootstrapRecord(
                    subject=subject,
                    predicate=predicate,
                    object=object_name,
                    asserted=asserted,
                    confidence=min(
                        1.0,
                        max(
                            0.0,
                            confidence,
                        ),
                    ),
                    topic=normalized_topic,
                    model=response.model,
                )

                prompt_ir = PromptIR(
                    language="en",
                    speech_act="assertion",
                    entities=(
                        PromptEntity(
                            local_id="e0",
                            surface=subject,
                            kind=str(
                                item.get(
                                    "subject_kind",
                                    "entity",
                                )
                            ).strip()
                            or "entity",
                            evidence=subject,
                        ),
                        PromptEntity(
                            local_id="e1",
                            surface=object_name,
                            kind=str(
                                item.get(
                                    "object_kind",
                                    "entity",
                                )
                            ).strip()
                            or "entity",
                            evidence=object_name,
                        ),
                    ),
                    observations=(
                        PromptObservation(
                            subject="e0",
                            predicate=predicate,
                            object="e1",
                            asserted=asserted,
                            evidence=(
                                "neural_bootstrap:"
                                + response.model
                                + ":"
                                + normalized_topic
                            ),
                        ),
                    ),
                    query_operation="none",
                    query_subject="",
                    query_predicate="",
                    query_object="",
                )
                _read, bootstrap_bindings = self._observe(
                    prompt_ir
                )
                self._mark_observation_provenance(
                    prompt_ir,
                    bootstrap_bindings,
                    "neural_bootstrap",
                )
                self.bootstrap_records[
                    record.key
                ] = record
                imported.append(
                    record
                )
        return tuple(
            imported
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

        self.neural_calls = []
        timings: dict[
            str,
            float,
        ] = {}
        errors: list[str] = []

        prompt_ir: PromptIR | None = None
        cortex_answer: (
            CortexAnswerIR
            | None
        ) = None
        read: Any | None = None
        diagnostic: (
            DiagnosticReport
            | None
        ) = None

        started = perf_counter()
        try:
            prompt_ir, parse_response = (
                self._parse_user_turn(
                    text
                )
            )
            timings[
                "prompt_parse"
            ] = (
                parse_response.duration_s
            )
            read, bindings = self._observe(
                prompt_ir
            )
            self._mark_observation_provenance(
                prompt_ir,
                bindings,
                "user_observation",
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
            errors.append(
                "Cortex pre-pass: "
                + str(
                    exc
                )
            )
        timings[
            "cortex_prepass_total"
        ] = (
            perf_counter()
            - started
        )

        if (
            prompt_ir is None
            or cortex_answer is None
        ):
            cortex_context: dict[
                str,
                Any,
            ] = {
                "provenance": (
                    "cortex_session_state"
                ),
                "status": "unavailable",
            }
        else:
            cortex_context = (
                self._cortex_context(
                    prompt_ir=prompt_ir,
                    cortex_answer=(
                        cortex_answer
                    ),
                    read=read,
                )
            )

        turn: HybridTurnIR | None = None
        try:
            turn, answer_response = (
                self._generate_turn(
                    text=text,
                    cortex_context=(
                        cortex_context
                    ),
                )
            )
            timings[
                "answer_with_sidecar"
            ] = (
                answer_response.duration_s
            )
        except Exception as exc:
            errors.append(
                "Neural turn: "
                + str(
                    exc
                )
            )

        if turn is None:
            answer = (
                "I do not have a usable neural response for this turn."
            )
            candidate = None
        else:
            answer = turn.answer
            candidate = turn.candidate

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

        revisions = 0
        while (
            turn is not None
            and diagnosis.status
            == "conflict"
            and revisions
            < max(
                0,
                int(
                    self.config.max_revision_cycles
                ),
            )
        ):
            revisions += 1
            try:
                turn, revision_response = (
                    self._revise_turn(
                        text=text,
                        prior=turn,
                        diagnosis=diagnosis,
                    )
                )
                answer = turn.answer
                candidate = (
                    turn.candidate
                )
                timings[
                    "conflict_revision"
                ] = (
                    revision_response.duration_s
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
            except Exception as exc:
                errors.append(
                    "Conflict revision: "
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
                break

        self.history.extend(
            [
                {
                    "role": "user",
                    "content": text,
                },
                {
                    "role": "assistant",
                    "content": answer,
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
            content=answer,
            cortex_context={
                "gateway": (
                    HYBRID_CORE_VERSION
                ),
                "neural_answer": answer,
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
                "hybrid_runtime": {
                    "backend": (
                        self.neural_runtime.backend_name
                    ),
                    "capabilities": (
                        self.neural_runtime.capabilities.packet()
                    ),
                    "neural_call_count": len(
                        self.neural_calls
                    ),
                    "neural_calls": [
                        dict(
                            item
                        )
                        for item
                        in self.neural_calls
                    ],
                    "revision_cycles": (
                        revisions
                    ),
                },
                "bootstrap": {
                    "claim_count": len(
                        self.bootstrap_records
                    ),
                    "claims": [
                        record.packet()
                        for record
                        in self.bootstrap_records.values()
                    ][
                        :32
                    ],
                },
            },
            diagnostics=diagnostic,
        )


def normalize_predicate(
    value: str,
) -> str:
    return "_".join(
        "".join(
            char
            if char.isalnum()
            else " "
            for char in value.casefold()
        ).split()
    )


def text_tokens(
    prompt_ir: PromptIR,
) -> set[str]:
    return {
        entity.surface.casefold()
        for entity
        in prompt_ir.entities
        if entity.surface
    }
