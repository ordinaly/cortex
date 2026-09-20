"""Typed cognitive exchange for Cortex Hybrid Core.

The exchange is the internal contract between orchestration, Cortex, and the
neural side. Text/JSON prompt serialization is an adapter concern rather than
the cognitive architecture itself.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from json import dumps
from typing import Any, Mapping, Protocol, Sequence
from uuid import uuid4


@dataclass(frozen=True)
class OrchestrationPlan:
    """Explicit per-turn control decisions."""

    route: str = "full_hybrid"
    parse_user: bool = True
    observe_user: bool = True
    inject_cortex: bool = True
    require_sidecar: bool = True
    revision_budget: int = 1
    context_claim_limit: int = 8
    bootstrap_claim_limit: int = 8
    reason: str = (
        "Default v0.1 policy preserves the full hybrid path."
    )

    def packet(self) -> dict[str, Any]:
        return {
            "route": self.route,
            "parse_user": self.parse_user,
            "observe_user": self.observe_user,
            "inject_cortex": self.inject_cortex,
            "require_sidecar": self.require_sidecar,
            "revision_budget": self.revision_budget,
            "context_claim_limit": self.context_claim_limit,
            "bootstrap_claim_limit": self.bootstrap_claim_limit,
            "reason": self.reason,
        }


class HybridOrchestrationPolicy(Protocol):
    def plan(
        self,
        *,
        user_text: str,
        has_relations: bool,
        has_bootstrap: bool,
    ) -> OrchestrationPlan:
        ...


@dataclass(frozen=True)
class DefaultHybridOrchestrationPolicy:
    """Conservative policy used while the exchange boundary is validated.

    The policy is intentionally explicit even where it currently chooses the
    same route for every turn. Future routing can change decisions without
    rewriting HybridCoreSession.
    """

    revision_budget: int = 1
    context_claim_limit: int = 8
    bootstrap_claim_limit: int = 8

    def plan(
        self,
        *,
        user_text: str,
        has_relations: bool,
        has_bootstrap: bool,
    ) -> OrchestrationPlan:
        del user_text
        state = (
            "existing Cortex structure"
            if has_relations
            else "cold Cortex state"
        )
        if has_bootstrap:
            state += " with provisional neural bootstrap"
        return OrchestrationPlan(
            revision_budget=max(
                0,
                int(
                    self.revision_budget
                ),
            ),
            context_claim_limit=max(
                0,
                int(
                    self.context_claim_limit
                ),
            ),
            bootstrap_claim_limit=max(
                0,
                int(
                    self.bootstrap_claim_limit
                ),
            ),
            reason=(
                "Full hybrid route selected from "
                + state
                + "."
            ),
        )


@dataclass(frozen=True)
class StaticHybridOrchestrationPolicy:
    """Return one explicit plan for every turn.

    Useful for ablations and backend experiments where orchestration decisions
    must be controlled independently of Cortex/NN behavior.
    """

    plan_template: OrchestrationPlan

    def plan(
        self,
        *,
        user_text: str,
        has_relations: bool,
        has_bootstrap: bool,
    ) -> OrchestrationPlan:
        del user_text
        del has_relations
        del has_bootstrap
        return self.plan_template


def named_orchestration_policy(
    name: str,
    *,
    revision_budget: int = 1,
) -> HybridOrchestrationPolicy:
    normalized = (
        name.strip().casefold()
    )
    if normalized == "full":
        return (
            DefaultHybridOrchestrationPolicy(
                revision_budget=(
                    revision_budget
                )
            )
        )
    if normalized == "neural-direct":
        return StaticHybridOrchestrationPolicy(
            OrchestrationPlan(
                route="neural_direct",
                parse_user=False,
                observe_user=False,
                inject_cortex=False,
                revision_budget=0,
                reason=(
                    "Ablation route: neural generation without Cortex pre-pass."
                ),
            )
        )
    if normalized == "no-revision":
        return StaticHybridOrchestrationPolicy(
            OrchestrationPlan(
                route="hybrid_no_revision",
                revision_budget=0,
                reason=(
                    "Ablation route: full Cortex context with revision disabled."
                ),
            )
        )
    if normalized == "no-cortex-context":
        return StaticHybridOrchestrationPolicy(
            OrchestrationPlan(
                route=(
                    "hybrid_no_cortex_context"
                ),
                inject_cortex=False,
                revision_budget=(
                    max(
                        0,
                        int(
                            revision_budget
                        ),
                    )
                ),
                reason=(
                    "Ablation route: Cortex observes and assesses, "
                    "but its pre-generation context is withheld from the NN."
                ),
            )
        )
    raise ValueError(
        "unknown hybrid orchestration policy: "
        + name
    )


@dataclass(frozen=True)
class CortexContribution:
    available: bool
    answer_status: str
    query_operation: str
    claims: tuple[
        Mapping[str, Any],
        ...,
    ] = ()
    bootstrap_claims: tuple[
        Mapping[str, Any],
        ...,
    ] = ()
    runtime: Mapping[
        str,
        Any,
    ] | None = None

    def packet(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "answer_status": self.answer_status,
            "query_operation": self.query_operation,
            "claims": [
                dict(
                    item
                )
                for item in self.claims
            ],
            "bootstrap_claims": [
                dict(
                    item
                )
                for item
                in self.bootstrap_claims
            ],
            "runtime": (
                dict(
                    self.runtime
                )
                if self.runtime
                is not None
                else None
            ),
        }


@dataclass(frozen=True)
class NeuralContribution:
    answer: str
    candidate: Mapping[
        str,
        Any,
    ]
    model: str
    backend: str

    def packet(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "candidate": dict(
                self.candidate
            ),
            "model": self.model,
            "backend": self.backend,
        }


@dataclass(frozen=True)
class ExchangeEvent:
    index: int
    phase: str
    actor: str
    action: str
    metadata: Mapping[
        str,
        Any,
    ] = field(
        default_factory=dict
    )

    def packet(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "phase": self.phase,
            "actor": self.actor,
            "action": self.action,
            "metadata": dict(
                self.metadata
            ),
        }


@dataclass
class HybridExchange:
    """Mutable turn state shared by the orchestrator boundaries."""

    session_id: str
    user_text: str
    plan: OrchestrationPlan
    exchange_id: str = field(
        default_factory=lambda: (
            "exchange-"
            + uuid4().hex
        )
    )
    phase: str = "created"
    prompt_ir: Mapping[
        str,
        Any,
    ] | None = None
    cortex: CortexContribution | None = None
    neural: NeuralContribution | None = None
    diagnosis: Mapping[
        str,
        Any,
    ] | None = None
    revision_count: int = 0
    events: list[
        ExchangeEvent
    ] = field(
        default_factory=list
    )

    def record(
        self,
        *,
        phase: str,
        actor: str,
        action: str,
        metadata: Mapping[
            str,
            Any,
        ] | None = None,
    ) -> None:
        self.phase = phase
        self.events.append(
            ExchangeEvent(
                index=len(
                    self.events
                ),
                phase=phase,
                actor=actor,
                action=action,
                metadata=(
                    dict(
                        metadata
                    )
                    if metadata
                    is not None
                    else {}
                ),
            )
        )

    def packet(self) -> dict[str, Any]:
        return {
            "exchange_id": self.exchange_id,
            "session_id": self.session_id,
            "phase": self.phase,
            "plan": (
                self.plan.packet()
            ),
            "prompt_ir": (
                dict(
                    self.prompt_ir
                )
                if self.prompt_ir
                is not None
                else None
            ),
            "cortex": (
                self.cortex.packet()
                if self.cortex
                is not None
                else None
            ),
            "neural": (
                self.neural.packet()
                if self.neural
                is not None
                else None
            ),
            "diagnosis": (
                dict(
                    self.diagnosis
                )
                if self.diagnosis
                is not None
                else None
            ),
            "revision_count": (
                self.revision_count
            ),
            "events": [
                event.packet()
                for event
                in self.events
            ],
        }


class TextHybridExchangeAdapter:
    """Compile a typed exchange for a text-only neural backend.

    Ollama currently consumes this representation. An in-process backend can
    later consume HybridExchange through a different adapter without changing
    Cortex or the orchestration policy.
    """

    def answer_messages(
        self,
        *,
        exchange: HybridExchange,
        history: Sequence[
            Mapping[str, str]
        ],
        system_prompt: str,
    ) -> list[
        Mapping[
            str,
            str,
        ]
    ]:
        context: dict[
            str,
            Any,
        ] = {
            "exchange_id": (
                exchange.exchange_id
            ),
            "plan": (
                exchange.plan.packet()
            ),
        }
        if (
            exchange.plan.inject_cortex
            and exchange.cortex
            is not None
        ):
            context[
                "cortex"
            ] = (
                exchange.cortex.packet()
            )

        return [
            {
                "role": "system",
                "content": (
                    system_prompt
                    + "\n\nHYBRID_EXCHANGE="
                    + dumps(
                        context,
                        sort_keys=True,
                    )
                ),
            },
            *[
                dict(
                    message
                )
                for message
                in history
            ],
            {
                "role": "user",
                "content": (
                    exchange.user_text
                ),
            },
        ]

    def revision_messages(
        self,
        *,
        exchange: HybridExchange,
        system_prompt: str,
        conflicts: Sequence[
            Mapping[str, Any]
        ],
    ) -> list[
        Mapping[
            str,
            str,
        ]
    ]:
        prior = (
            exchange.neural.packet()
            if exchange.neural
            is not None
            else None
        )
        return [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": dumps(
                    {
                        "exchange_id": (
                            exchange.exchange_id
                        ),
                        "user_request": (
                            exchange.user_text
                        ),
                        "prior_neural": prior,
                        "cortex_conflicts": [
                            dict(
                                item
                            )
                            for item
                            in conflicts
                        ],
                        "revision_budget": (
                            exchange.plan.revision_budget
                        ),
                    },
                    sort_keys=True,
                ),
            },
        ]
