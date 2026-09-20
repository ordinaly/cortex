from __future__ import annotations

from cortex.hybrid_exchange import (
    CortexContribution,
    DefaultHybridOrchestrationPolicy,
    HybridExchange,
    NeuralContribution,
    OrchestrationPlan,
    TextHybridExchangeAdapter,
)


def test_default_policy_is_explicit_and_conservative():
    policy = (
        DefaultHybridOrchestrationPolicy(
            revision_budget=2,
            context_claim_limit=5,
            bootstrap_claim_limit=3,
        )
    )

    plan = policy.plan(
        user_text="hello",
        has_relations=True,
        has_bootstrap=True,
    )

    assert (
        plan.route
        == "full_hybrid"
    )
    assert plan.parse_user
    assert plan.observe_user
    assert plan.inject_cortex
    assert plan.assess_candidate
    assert plan.require_sidecar
    assert (
        plan.revision_budget
        == 2
    )
    assert (
        plan.context_claim_limit
        == 5
    )
    assert (
        plan.bootstrap_claim_limit
        == 3
    )
    assert (
        "existing Cortex structure"
        in plan.reason
    )


def test_exchange_tracks_typed_contributions_and_events():
    exchange = HybridExchange(
        session_id="session",
        user_text="question",
        plan=OrchestrationPlan(),
    )
    exchange.record(
        phase="planned",
        actor="orchestrator",
        action="plan_selected",
    )
    exchange.prompt_ir = {
        "query_operation": (
            "relation_lookup"
        )
    }
    exchange.cortex = (
        CortexContribution(
            available=True,
            answer_status="resolved",
            query_operation=(
                "relation_lookup"
            ),
            claims=(
                {
                    "predicate": (
                        "related_to"
                    )
                },
            ),
        )
    )
    exchange.neural = (
        NeuralContribution(
            answer="answer",
            candidate={
                "claims": [],
            },
            model="model",
            backend="fake",
        )
    )
    exchange.diagnosis = {
        "status": "supported",
    }
    exchange.record(
        phase="completed",
        actor="orchestrator",
        action="finalize",
    )

    packet = exchange.packet()

    assert (
        packet["cortex"][
            "answer_status"
        ]
        == "resolved"
    )
    assert (
        packet["neural"][
            "answer"
        ]
        == "answer"
    )
    assert (
        packet["diagnosis"][
            "status"
        ]
        == "supported"
    )
    assert [
        event["phase"]
        for event
        in packet["events"]
    ] == [
        "planned",
        "completed",
    ]


def test_text_adapter_serializes_exchange_as_backend_adapter():
    exchange = HybridExchange(
        session_id="session",
        user_text="What changed?",
        plan=OrchestrationPlan(
            inject_cortex=True,
        ),
    )
    exchange.cortex = (
        CortexContribution(
            available=True,
            answer_status="resolved",
            query_operation="none",
            claims=(
                {
                    "subject": "A",
                    "predicate": "r",
                    "object": "B",
                },
            ),
        )
    )
    adapter = (
        TextHybridExchangeAdapter()
    )

    messages = (
        adapter.answer_messages(
            exchange=exchange,
            history=[
                {
                    "role": "assistant",
                    "content": "Earlier.",
                }
            ],
            system_prompt="SYSTEM",
        )
    )

    system = messages[0][
        "content"
    ]
    assert (
        "HYBRID_EXCHANGE="
        in system
    )
    assert (
        exchange.exchange_id
        in system
    )
    assert (
        '"predicate": "r"'
        in system
    )
    assert (
        messages[-1][
            "content"
        ]
        == "What changed?"
    )


def test_revision_adapter_contains_prior_and_conflict_only_as_exchange_data():
    exchange = HybridExchange(
        session_id="session",
        user_text="question",
        plan=OrchestrationPlan(
            revision_budget=1,
        ),
    )
    exchange.neural = (
        NeuralContribution(
            answer="first",
            candidate={
                "claims": [
                    {
                        "predicate": "r",
                    }
                ],
            },
            model="model",
            backend="fake",
        )
    )

    messages = (
        TextHybridExchangeAdapter().revision_messages(
            exchange=exchange,
            system_prompt="REVISE",
            conflicts=[
                {
                    "status": "conflict",
                }
            ],
        )
    )

    payload = messages[-1][
        "content"
    ]
    assert (
        exchange.exchange_id
        in payload
    )
    assert '"answer": "first"' in payload
    assert '"status": "conflict"' in payload
    assert '"revision_budget": 1' in payload
