"""Hybrid orchestration ablation v0.1.

Measures the marginal contribution of Cortex pre-generation context and
post-generation revision using the same scripted neural behavior.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cortex.strict_language_gateway as strict_module
from cortex.hybrid_core import (
    HybridCoreConfig,
    HybridCoreSession,
)
from cortex.hybrid_exchange import (
    named_orchestration_policy,
)
from hybrid_symbiosis import (
    FakeCortexRuntime,
    ScriptedNeuralRuntime,
    build_world,
    fixed_query,
    positive_observation,
)


PROTOCOL = (
    "hybrid-orchestration-ablation-v0.1"
)
POLICIES = (
    "neural-direct",
    "no-cortex-context",
    "no-revision",
    "full",
)
NEURAL_MODES = (
    "context-responsive",
    "context-blind-revisable",
    "stubborn",
)


def runtime_for(
    world,
    mode: str,
) -> ScriptedNeuralRuntime:
    if (
        mode
        == "context-responsive"
    ):
        return ScriptedNeuralRuntime(
            world,
            neural_asserted=False,
            respect_cortex_context=True,
            revision_mode="fix",
        )
    if (
        mode
        == "context-blind-revisable"
    ):
        return ScriptedNeuralRuntime(
            world,
            neural_asserted=False,
            respect_cortex_context=False,
            revision_mode="fix",
        )
    if mode == "stubborn":
        return ScriptedNeuralRuntime(
            world,
            neural_asserted=False,
            respect_cortex_context=False,
            revision_mode="stubborn",
        )
    raise ValueError(
        "unknown neural mode: "
        + mode
    )


def run_case(
    *,
    seed: int,
    policy_name: str,
    neural_mode: str,
) -> dict[str, Any]:
    strict_module.CortexRuntime = (
        FakeCortexRuntime
    )
    world = build_world(
        seed
    )
    runtime = runtime_for(
        world,
        neural_mode,
    )

    # Seed the exact same independent Cortex observation before applying the
    # target ablation. This keeps the world state fixed across policies.
    session = HybridCoreSession(
        (
            f"ablation-{seed}-"
            f"{policy_name}-"
            f"{neural_mode}"
        ),
        HybridCoreConfig(
            chat_model="scripted",
            parser_model=(
                "scripted-parser"
            ),
            feature_dim=64,
            include_diagnosis=False,
            max_revision_cycles=1,
        ),
        runtime,
        orchestration_policy=(
            named_orchestration_policy(
                "full"
            )
        ),
    )
    session.chat(
        positive_observation(
            world
        )
    )

    session.orchestration_policy = (
        named_orchestration_policy(
            policy_name
        )
    )
    result = session.chat(
        fixed_query(
            world
        )
    )

    context = (
        result.cortex_context
    )
    exchange = context[
        "hybrid_exchange"
    ]
    hybrid_runtime = context[
        "hybrid_runtime"
    ]
    diagnosis = context[
        "diagnosis"
    ]
    candidate = context[
        "candidate_ir"
    ]
    claims = (
        candidate.get(
            "claims",
            [],
        )
        if isinstance(
            candidate,
            dict,
        )
        else []
    )
    final_asserted = (
        claims[0][
            "asserted"
        ]
        if claims
        else None
    )

    return {
        "protocol": PROTOCOL,
        "seed": seed,
        "policy": policy_name,
        "neural_mode": (
            neural_mode
        ),
        "diagnosis_status": (
            diagnosis[
                "status"
            ]
        ),
        "final_asserted": (
            final_asserted
        ),
        "neural_calls": (
            hybrid_runtime[
                "neural_call_count"
            ]
        ),
        "revision_cycles": (
            hybrid_runtime[
                "revision_cycles"
            ]
        ),
        "prompt_ir_present": (
            context[
                "prompt_ir"
            ]
            is not None
        ),
        "cortex_available": bool(
            exchange.get(
                "cortex",
                {},
            ).get(
                "available",
                False,
            )
        ),
        "inject_cortex": bool(
            exchange[
                "plan"
            ][
                "inject_cortex"
            ]
        ),
        "route": (
            exchange[
                "plan"
            ][
                "route"
            ]
        ),
        "answer_nonempty": bool(
            result.content.strip()
        ),
    }


def campaign(
    *,
    seeds: tuple[int, ...],
    output: Path,
) -> dict[str, Any]:
    rows = [
        run_case(
            seed=seed,
            policy_name=policy,
            neural_mode=mode,
        )
        for seed in seeds
        for mode in NEURAL_MODES
        for policy in POLICIES
    ]
    output.write_text(
        "".join(
            json.dumps(
                row,
                sort_keys=True,
            )
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )

    def selected(
        policy: str,
        mode: str,
    ):
        return [
            row
            for row in rows
            if (
                row[
                    "policy"
                ]
                == policy
                and row[
                    "neural_mode"
                ]
                == mode
            )
        ]

    def rate(
        chosen,
        predicate,
    ) -> float:
        chosen = list(
            chosen
        )
        return (
            sum(
                int(
                    predicate(
                        row
                    )
                )
                for row in chosen
            )
            / len(
                chosen
            )
        )

    metrics = {
        "neural_direct_one_call_rate": rate(
            [
                row
                for row in rows
                if row[
                    "policy"
                ]
                == "neural-direct"
            ],
            lambda row: (
                row[
                    "neural_calls"
                ]
                == 1
            ),
        ),
        "full_context_responsive_no_revision_rate": rate(
            selected(
                "full",
                "context-responsive",
            ),
            lambda row: (
                row[
                    "final_asserted"
                ]
                is True
                and row[
                    "revision_cycles"
                ]
                == 0
                and row[
                    "diagnosis_status"
                ]
                == "supported"
            ),
        ),
        "withheld_context_requires_revision_rate": rate(
            selected(
                "no-cortex-context",
                "context-responsive",
            ),
            lambda row: (
                row[
                    "revision_cycles"
                ]
                == 1
                and row[
                    "final_asserted"
                ]
                is True
                and row[
                    "diagnosis_status"
                ]
                == "supported"
            ),
        ),
        "full_blind_revision_success_rate": rate(
            selected(
                "full",
                "context-blind-revisable",
            ),
            lambda row: (
                row[
                    "revision_cycles"
                ]
                == 1
                and row[
                    "final_asserted"
                ]
                is True
                and row[
                    "diagnosis_status"
                ]
                == "supported"
            ),
        ),
        "no_revision_conflict_preservation_rate": rate(
            selected(
                "no-revision",
                "context-blind-revisable",
            ),
            lambda row: (
                row[
                    "revision_cycles"
                ]
                == 0
                and row[
                    "diagnosis_status"
                ]
                == "conflict"
            ),
        ),
        "full_stubborn_bounded_conflict_rate": rate(
            selected(
                "full",
                "stubborn",
            ),
            lambda row: (
                row[
                    "revision_cycles"
                ]
                == 1
                and row[
                    "diagnosis_status"
                ]
                == "conflict"
            ),
        ),
        "nonempty_answer_rate": rate(
            rows,
            lambda row: (
                row[
                    "answer_nonempty"
                ]
            ),
        ),
    }

    summary = {
        "protocol": PROTOCOL,
        "seeds": list(
            seeds
        ),
        "policies": list(
            POLICIES
        ),
        "neural_modes": list(
            NEURAL_MODES
        ),
        "rows": len(
            rows
        ),
        "metrics": metrics,
    }
    print(
        json.dumps(
            summary,
            sort_keys=True,
        )
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seeds",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "hybrid-orchestration-ablation-v0.1.jsonl"
        ),
    )
    args = parser.parse_args()
    seeds = tuple(
        range(
            max(
                1,
                int(
                    args.seeds
                ),
            )
        )
    )
    campaign(
        seeds=seeds,
        output=args.output,
    )


if __name__ == "__main__":
    main()
