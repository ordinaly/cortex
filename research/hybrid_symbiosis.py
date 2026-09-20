"""Deterministic Hybrid Symbiosis v0.1 characterization.

This campaign tests the NN/Cortex orchestration contract independently of a
specific language model. A scripted neural runtime is used so failures are
attributable to the hybrid architecture rather than stochastic model behavior.

A second live-model campaign can reuse the same scenario families later.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import cortex.strict_language_gateway as strict_module
from cortex.hybrid_core import (
    BOOTSTRAP_SCHEMA,
    HYBRID_TURN_SCHEMA,
    HybridCoreConfig,
    HybridCoreSession,
)
from cortex.neural_runtime import (
    NeuralCapabilities,
    NeuralResponse,
)


PROTOCOL = "hybrid-symbiosis-v0.1"

FAMILIES = (
    "neural-prior",
    "independent-support",
    "context-guided",
    "conflict-revision",
    "conflict-stubborn",
    "bootstrap-aligned",
    "bootstrap-divergence",
    "bootstrap-promoted",
    "parser-failure",
    "alternative-object-divergence",
    "cold-vs-bootstrap",
)

DEVELOPMENT_SEEDS = tuple(
    range(
        10
    )
)


@dataclass(frozen=True)
class World:
    subject: str
    predicate: str
    object_true: str
    object_alt: str


def build_world(
    seed: int,
) -> World:
    return World(
        subject=(
            f"N{seed:02d}_subject"
        ),
        predicate=(
            f"rel_{seed:02d}"
        ),
        object_true=(
            f"N{seed:02d}_true"
        ),
        object_alt=(
            f"N{seed:02d}_alt"
        ),
    )


class FakeRead:
    def __init__(
        self,
        bindings: Sequence[int],
    ) -> None:
        self.bindings = list(
            bindings
        )
        self.subgroup = [0]
        self.subgroup_margin = 0.2
        self.relation_changed = []
        self.causal_changed = []
        self.articulation_vector = (
            [0.0] * 12
        )
        self.memory_stored = 1
        self.memory_unresolved = False
        self.prediction = 0.8
        self.memberships = [
            0.9,
            0.1,
        ]
        self.current_id = 0
        self.nearest_id = 0
        self.nearest_dist = 0.05
        self.revision = False
        self.reactivated = False
        self.discovered = False
        self.unresolved = False
        self.budget_pressure = False
        self.stored = 1


class FakeCortexRuntime:
    def __init__(
        self,
        *,
        feature_dim: int,
        max_entities: int,
        budget: int,
    ) -> None:
        del max_entities
        del budget
        self.feature_dim = (
            feature_dim
        )
        self.vector_ids: dict[
            tuple[float, ...],
            int,
        ] = {}

    def step(
        self,
        detections,
        relation_obs,
        intervention_src_det,
        outcomes,
        outcome,
    ) -> FakeRead:
        del relation_obs
        del intervention_src_det
        del outcomes
        del outcome
        bindings: list[
            int
        ] = []
        for vector in detections:
            key = tuple(
                float(
                    value
                )
                for value in vector
            )
            if (
                key
                not in self.vector_ids
            ):
                self.vector_ids[
                    key
                ] = len(
                    self.vector_ids
                )
            bindings.append(
                self.vector_ids[
                    key
                ]
            )
        return FakeRead(
            bindings
        )

    def snapshot_json(
        self,
    ) -> str:
        return json.dumps(
            {
                "feature_dim": (
                    self.feature_dim
                ),
                "vector_ids": [
                    [
                        list(
                            key
                        ),
                        value,
                    ]
                    for key, value
                    in sorted(
                        self.vector_ids.items(),
                        key=lambda item: (
                            item[1]
                        ),
                    )
                ],
            },
            sort_keys=True,
        )


def _prompt_ir(
    *,
    world: World,
    user: str,
) -> str:
    if (
        user
        == "PARSER_FAIL"
    ):
        return "{not-json"

    positive = (
        f"Observe: {world.subject} "
        f"{world.predicate} "
        f"{world.object_true}."
    )
    negative = (
        f"Observe-not: {world.subject} "
        f"{world.predicate} "
        f"{world.object_true}."
    )
    fixed_query = (
        f"Query: {world.subject} "
        f"{world.predicate} "
        f"{world.object_true}?"
    )
    open_query = (
        f"Query-object: {world.subject} "
        f"{world.predicate}?"
    )

    if user in {
        positive,
        negative,
    }:
        asserted = (
            user == positive
        )
        return json.dumps(
            {
                "language": "en",
                "speech_act": (
                    "assertion"
                ),
                "entities": [
                    {
                        "id": "e0",
                        "surface": (
                            world.subject
                        ),
                        "kind": "node",
                        "evidence": (
                            world.subject
                        ),
                    },
                    {
                        "id": "e1",
                        "surface": (
                            world.object_true
                        ),
                        "kind": "node",
                        "evidence": (
                            world.object_true
                        ),
                    },
                ],
                "observations": [
                    {
                        "subject": "e0",
                        "predicate": (
                            world.predicate
                        ),
                        "object": "e1",
                        "asserted": (
                            asserted
                        ),
                        "evidence": (
                            user.removesuffix(
                                "."
                            )
                        ),
                    }
                ],
                "query_operation": (
                    "none"
                ),
                "query_subject": "",
                "query_predicate": "",
                "query_object": "",
            }
        )

    if user == fixed_query:
        return json.dumps(
            {
                "language": "en",
                "speech_act": (
                    "question"
                ),
                "entities": [
                    {
                        "id": "e0",
                        "surface": (
                            world.subject
                        ),
                        "kind": "node",
                        "evidence": (
                            world.subject
                        ),
                    },
                    {
                        "id": "e1",
                        "surface": (
                            world.object_true
                        ),
                        "kind": "node",
                        "evidence": (
                            world.object_true
                        ),
                    },
                ],
                "observations": [],
                "query_operation": (
                    "relation_lookup"
                ),
                "query_subject": "e0",
                "query_predicate": (
                    world.predicate
                ),
                "query_object": "e1",
            }
        )

    if user == open_query:
        return json.dumps(
            {
                "language": "en",
                "speech_act": (
                    "question"
                ),
                "entities": [
                    {
                        "id": "e0",
                        "surface": (
                            world.subject
                        ),
                        "kind": "node",
                        "evidence": (
                            world.subject
                        ),
                    }
                ],
                "observations": [],
                "query_operation": (
                    "relation_lookup"
                ),
                "query_subject": "e0",
                "query_predicate": (
                    world.predicate
                ),
                "query_object": "",
            }
        )

    return json.dumps(
        {
            "language": "en",
            "speech_act": "other",
            "entities": [],
            "observations": [],
            "query_operation": "none",
            "query_subject": "",
            "query_predicate": "",
            "query_object": "",
        }
    )


def _turn_json(
    *,
    world: World,
    asserted: bool,
    object_name: str | None = None,
    prefix: str = "",
) -> str:
    target = (
        object_name
        or world.object_true
    )
    relation_text = (
        (
            f"{world.subject} "
            f"{world.predicate} "
            f"{target}."
        )
        if asserted
        else (
            f"{world.subject} does not "
            f"{world.predicate} "
            f"{target}."
        )
    )
    answer = (
        prefix
        + relation_text
    ).strip()
    return json.dumps(
        {
            "answer": answer,
            "entities": [
                {
                    "id": "e0",
                    "surface": (
                        world.subject
                    ),
                    "kind": "node",
                    "evidence": (
                        world.subject
                    ),
                },
                {
                    "id": "e1",
                    "surface": (
                        target
                    ),
                    "kind": "node",
                    "evidence": target,
                },
            ],
            "claims": [
                {
                    "subject": "e0",
                    "predicate": (
                        world.predicate
                    ),
                    "object": "e1",
                    "asserted": (
                        asserted
                    ),
                    "evidence": (
                        relation_text
                    ),
                }
            ],
        }
    )


class ScriptedNeuralRuntime:
    def __init__(
        self,
        world: World,
        *,
        neural_asserted: bool = True,
        neural_object: str | None = None,
        respect_cortex_context: bool = False,
        revision_mode: str = "fix",
    ) -> None:
        self.world = world
        self.neural_asserted = (
            neural_asserted
        )
        self.neural_object = (
            neural_object
            or world.object_true
        )
        self.respect_cortex_context = (
            respect_cortex_context
        )
        self.revision_mode = (
            revision_mode
        )
        self.calls: list[
            dict[
                str,
                Any,
            ]
        ] = []

    @property
    def backend_name(
        self,
    ) -> str:
        return (
            "scripted-neural"
        )

    @property
    def capabilities(
        self,
    ) -> NeuralCapabilities:
        return NeuralCapabilities(
            generate=True,
            structured_generate=True,
        )

    def _response(
        self,
        *,
        text: str,
        model: str,
        schema: Mapping[
            str,
            Any,
        ]
        | None,
    ) -> NeuralResponse:
        return NeuralResponse(
            text=text,
            model=model,
            backend=(
                self.backend_name
            ),
            duration_s=0.001,
            structured=(
                schema is not None
            ),
            prompt_tokens=10,
            completion_tokens=5,
        )

    def generate(
        self,
        *,
        model: str,
        messages: Sequence[
            Mapping[str, str]
        ],
        schema=None,
        temperature=None,
    ) -> NeuralResponse:
        del temperature
        user = (
            messages[-1][
                "content"
            ]
        )
        system = (
            messages[0][
                "content"
            ]
            if (
                messages
                and messages[0].get(
                    "role"
                )
                == "system"
            )
            else ""
        )
        self.calls.append(
            {
                "model": model,
                "user": user,
                "system": system,
                "schema": schema,
            }
        )

        if (
            schema
            == BOOTSTRAP_SCHEMA
        ):
            return self._response(
                text=json.dumps(
                    {
                        "claims": [
                            {
                                "subject": (
                                    self.world.subject
                                ),
                                "subject_kind": (
                                    "node"
                                ),
                                "predicate": (
                                    self.world.predicate
                                ),
                                "object": (
                                    self.world.object_true
                                ),
                                "object_kind": (
                                    "node"
                                ),
                                "asserted": True,
                                "confidence": (
                                    0.95
                                ),
                            }
                        ]
                    }
                ),
                model=model,
                schema=schema,
            )

        if (
            schema
            != HYBRID_TURN_SCHEMA
        ):
            return self._response(
                text=_prompt_ir(
                    world=self.world,
                    user=user,
                ),
                model=model,
                schema=schema,
            )

        if (
            "cortex_conflicts"
            in user
        ):
            asserted = (
                True
                if self.revision_mode
                == "fix"
                else self.neural_asserted
            )
            return self._response(
                text=_turn_json(
                    world=self.world,
                    asserted=asserted,
                    object_name=(
                        self.neural_object
                    ),
                    prefix=(
                        "Revised: "
                    ),
                ),
                model=model,
                schema=schema,
            )

        asserted = (
            self.neural_asserted
        )
        object_name = (
            self.neural_object
        )
        if (
            self.respect_cortex_context
            and "HYBRID_EXCHANGE="
            in system
            and self.world.object_true
            in system
        ):
            asserted = True
            object_name = (
                self.world.object_true
            )

        return self._response(
            text=_turn_json(
                world=self.world,
                asserted=asserted,
                object_name=(
                    object_name
                ),
            ),
            model=model,
            schema=schema,
        )

    def embed(
        self,
        *,
        model: str,
        inputs: Sequence[str],
    ):
        del model
        del inputs
        raise AssertionError(
            "Hybrid Symbiosis v0.1 must not use neural embeddings"
        )


def make_session(
    world: World,
    **runtime_kwargs,
) -> tuple[
    HybridCoreSession,
    ScriptedNeuralRuntime,
]:
    runtime = ScriptedNeuralRuntime(
        world,
        **runtime_kwargs,
    )
    session = HybridCoreSession(
        "campaign",
        HybridCoreConfig(
            chat_model="scripted",
            parser_model="scripted-parser",
            feature_dim=64,
            include_diagnosis=False,
            max_revision_cycles=1,
        ),
        runtime,
    )
    return (
        session,
        runtime,
    )


def positive_observation(
    world: World,
) -> str:
    return (
        f"Observe: {world.subject} "
        f"{world.predicate} "
        f"{world.object_true}."
    )


def fixed_query(
    world: World,
) -> str:
    return (
        f"Query: {world.subject} "
        f"{world.predicate} "
        f"{world.object_true}?"
    )


def open_query(
    world: World,
) -> str:
    return (
        f"Query-object: {world.subject} "
        f"{world.predicate}?"
    )


def _claim_statuses(
    result,
) -> list[str]:
    return [
        str(
            item[
                "status"
            ]
        )
        for item
        in result.cortex_context[
            "diagnosis"
        ][
            "neural_claims"
        ]
    ]


def _runtime_packet(
    result,
) -> Mapping[str, Any]:
    return result.cortex_context[
        "hybrid_runtime"
    ]


def run_case(
    family: str,
    seed: int,
) -> dict[str, Any]:
    if family not in FAMILIES:
        raise ValueError(
            "unknown family: "
            + family
        )

    strict_module.CortexRuntime = (
        FakeCortexRuntime
    )
    world = build_world(
        seed
    )

    if family == "neural-prior":
        session, _runtime = (
            make_session(
                world,
            )
        )
        result = session.chat(
            fixed_query(
                world
            )
        )
        return row(
            family,
            seed,
            result,
            {
                "neural_prior": (
                    _claim_statuses(
                        result
                    )
                    == [
                        "neural_prior"
                    ]
                ),
                "no_revision": (
                    _runtime_packet(
                        result
                    )[
                        "revision_cycles"
                    ]
                    == 0
                ),
            },
        )

    if family == "independent-support":
        session, _runtime = (
            make_session(
                world,
            )
        )
        session.chat(
            positive_observation(
                world
            )
        )
        result = session.chat(
            fixed_query(
                world
            )
        )
        return row(
            family,
            seed,
            result,
            {
                "supported": (
                    result.cortex_context[
                        "diagnosis"
                    ][
                        "status"
                    ]
                    == "supported"
                ),
                "corroborated": (
                    _claim_statuses(
                        result
                    )
                    == [
                        "corroborated"
                    ]
                ),
            },
        )

    if family == "context-guided":
        session, _runtime = (
            make_session(
                world,
                neural_asserted=False,
                respect_cortex_context=True,
            )
        )
        session.chat(
            positive_observation(
                world
            )
        )
        result = session.chat(
            fixed_query(
                world
            )
        )
        claims = result.cortex_context[
            "candidate_ir"
        ][
            "claims"
        ]
        return row(
            family,
            seed,
            result,
            {
                "context_changed_neural_candidate": (
                    bool(
                        claims
                    )
                    and claims[0][
                        "asserted"
                    ]
                    is True
                ),
                "supported_without_revision": (
                    result.cortex_context[
                        "diagnosis"
                    ][
                        "status"
                    ]
                    == "supported"
                    and _runtime_packet(
                        result
                    )[
                        "revision_cycles"
                    ]
                    == 0
                ),
            },
        )

    if family == "conflict-revision":
        session, _runtime = (
            make_session(
                world,
                neural_asserted=False,
                revision_mode="fix",
            )
        )
        session.chat(
            positive_observation(
                world
            )
        )
        result = session.chat(
            fixed_query(
                world
            )
        )
        return row(
            family,
            seed,
            result,
            {
                "revision_triggered": (
                    _runtime_packet(
                        result
                    )[
                        "revision_cycles"
                    ]
                    == 1
                ),
                "revision_resolved": (
                    result.cortex_context[
                        "diagnosis"
                    ][
                        "status"
                    ]
                    == "supported"
                    and _claim_statuses(
                        result
                    )
                    == [
                        "corroborated"
                    ]
                ),
            },
        )

    if family == "conflict-stubborn":
        session, _runtime = (
            make_session(
                world,
                neural_asserted=False,
                revision_mode="stubborn",
            )
        )
        session.chat(
            positive_observation(
                world
            )
        )
        result = session.chat(
            fixed_query(
                world
            )
        )
        return row(
            family,
            seed,
            result,
            {
                "bounded_one_revision": (
                    _runtime_packet(
                        result
                    )[
                        "revision_cycles"
                    ]
                    == 1
                ),
                "conflict_preserved": (
                    result.cortex_context[
                        "diagnosis"
                    ][
                        "status"
                    ]
                    == "conflict"
                ),
            },
        )

    if family == "bootstrap-aligned":
        session, _runtime = (
            make_session(
                world,
            )
        )
        session.bootstrap_topics(
            [
                "synthetic world",
            ]
        )
        result = session.chat(
            fixed_query(
                world
            )
        )
        return row(
            family,
            seed,
            result,
            {
                "bootstrap_aligned": (
                    result.cortex_context[
                        "diagnosis"
                    ][
                        "status"
                    ]
                    == "bootstrap_aligned"
                ),
                "not_independent_support": (
                    "corroborated"
                    not in _claim_statuses(
                        result
                    )
                ),
                "no_revision": (
                    _runtime_packet(
                        result
                    )[
                        "revision_cycles"
                    ]
                    == 0
                ),
            },
        )

    if family == "bootstrap-divergence":
        session, _runtime = (
            make_session(
                world,
                neural_asserted=False,
            )
        )
        session.bootstrap_topics(
            [
                "synthetic world",
            ]
        )
        result = session.chat(
            fixed_query(
                world
            )
        )
        return row(
            family,
            seed,
            result,
            {
                "bootstrap_divergence": (
                    result.cortex_context[
                        "diagnosis"
                    ][
                        "status"
                    ]
                    == "bootstrap_divergence"
                ),
                "no_forced_revision": (
                    _runtime_packet(
                        result
                    )[
                        "revision_cycles"
                    ]
                    == 0
                ),
            },
        )

    if family == "bootstrap-promoted":
        session, _runtime = (
            make_session(
                world,
            )
        )
        session.bootstrap_topics(
            [
                "synthetic world",
            ]
        )
        session.chat(
            positive_observation(
                world
            )
        )
        result = session.chat(
            fixed_query(
                world
            )
        )
        return row(
            family,
            seed,
            result,
            {
                "promoted_to_support": (
                    result.cortex_context[
                        "diagnosis"
                    ][
                        "status"
                    ]
                    == "supported"
                ),
                "corroborated": (
                    _claim_statuses(
                        result
                    )
                    == [
                        "corroborated"
                    ]
                ),
            },
        )

    if family == "parser-failure":
        session, _runtime = (
            make_session(
                world,
            )
        )
        result = session.chat(
            "PARSER_FAIL"
        )
        diagnosis = (
            result.cortex_context[
                "diagnosis"
            ]
        )
        return row(
            family,
            seed,
            result,
            {
                "answer_survived": (
                    bool(
                        result.content.strip()
                    )
                ),
                "analysis_error_exposed": (
                    bool(
                        diagnosis[
                            "analysis_errors"
                        ]
                    )
                ),
            },
        )

    if (
        family
        == "alternative-object-divergence"
    ):
        session, _runtime = (
            make_session(
                world,
                neural_object=(
                    world.object_alt
                ),
            )
        )
        session.chat(
            positive_observation(
                world
            )
        )
        result = session.chat(
            open_query(
                world
            )
        )
        statuses = (
            _claim_statuses(
                result
            )
        )
        detected = (
            result.cortex_context[
                "diagnosis"
            ][
                "status"
            ]
            == "conflict"
            or "conflict"
            in statuses
        )
        return row(
            family,
            seed,
            result,
            {
                "alternative_object_conflict_detected": (
                    detected
                ),
                "characterized_without_crash": (
                    bool(
                        result.content.strip()
                    )
                ),
            },
        )

    if family == "cold-vs-bootstrap":
        cold, _ = (
            make_session(
                world,
            )
        )
        cold_result = cold.chat(
            open_query(
                world
            )
        )

        boot, _ = (
            make_session(
                world,
            )
        )
        boot.bootstrap_topics(
            [
                "synthetic world",
            ]
        )
        boot_result = boot.chat(
            open_query(
                world
            )
        )
        base = row(
            family,
            seed,
            boot_result,
            {
                "cold_not_resolved": (
                    cold_result.cortex_context[
                        "cortex_answer_ir"
                    ][
                        "status"
                    ]
                    != "resolved"
                ),
                "bootstrap_resolved": (
                    boot_result.cortex_context[
                        "cortex_answer_ir"
                    ][
                        "status"
                    ]
                    == "resolved"
                ),
            },
        )
        base[
            "cold_cortex_status"
        ] = cold_result.cortex_context[
            "cortex_answer_ir"
        ][
            "status"
        ]
        base[
            "bootstrap_cortex_status"
        ] = boot_result.cortex_context[
            "cortex_answer_ir"
        ][
            "status"
        ]
        return base

    raise AssertionError(
        "unreachable"
    )


def row(
    family: str,
    seed: int,
    result,
    checks: Mapping[
        str,
        bool,
    ],
) -> dict[str, Any]:
    runtime = (
        _runtime_packet(
            result
        )
    )
    return {
        "protocol": PROTOCOL,
        "family": family,
        "seed": seed,
        "diagnosis_status": (
            result.cortex_context[
                "diagnosis"
            ][
                "status"
            ]
        ),
        "claim_statuses": (
            _claim_statuses(
                result
            )
        ),
        "neural_calls": (
            runtime[
                "neural_call_count"
            ]
        ),
        "revision_cycles": (
            runtime[
                "revision_cycles"
            ]
        ),
        "stages": [
            call[
                "stage"
            ]
            for call
            in runtime[
                "neural_calls"
            ]
        ],
        "answer_nonempty": bool(
            result.content.strip()
        ),
        "checks": {
            key: bool(
                value
            )
            for key, value
            in checks.items()
        },
    }
