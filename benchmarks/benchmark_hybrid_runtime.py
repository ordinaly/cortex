#!/usr/bin/env python3
"""Compare Gateway v0.3 against Hybrid Core v0.1 on a live local model.

This benchmark is intentionally local-only: it requires a running Ollama
instance and does not belong in CI. It measures end-to-end wall time and neural
call count for the same prompts/models.

Example:

    PYTHONPATH=python python benchmarks/benchmark_hybrid_runtime.py \
      --chat-model qwen3:8b \
      --parser-model cortex-parser \
      --repeats 3
"""
from __future__ import annotations

import argparse
import json
from statistics import mean
from time import perf_counter

from cortex.hybrid_core import (
    HybridCoreConfig,
)
from cortex.hybrid_core_gateway import (
    HybridCoreGateway,
)
from cortex.hybrid_language_gateway import (
    HybridCortexLanguageGateway,
    HybridGatewayConfig,
)
from cortex.neural_runtime import (
    OllamaNeuralRuntime,
)


DEFAULT_PROMPTS = [
    "What is the capital of Mongolia?",
    "Alice works with Bob.",
    "Does Alice work with Bob?",
]


def run_v03(
    *,
    chat_model: str,
    parser_model: str,
    ollama_url: str,
    prompts: list[str],
    repeats: int,
) -> dict:
    gateway = (
        HybridCortexLanguageGateway(
            HybridGatewayConfig(
                chat_model=chat_model,
                parser_model=parser_model,
                ollama_url=ollama_url,
                persistence_dir=None,
                include_diagnosis=False,
            )
        )
    )
    samples = []
    for repeat in range(
        repeats
    ):
        session_id = (
            f"bench-v03-{repeat}"
        )
        for prompt in prompts:
            started = perf_counter()
            result = gateway.chat(
                session_id,
                prompt,
            )
            wall = (
                perf_counter()
                - started
            )
            diagnosis = (
                result.cortex_context[
                    "diagnosis"
                ]
            )
            samples.append(
                {
                    "prompt": prompt,
                    "wall_s": wall,
                    "neural_calls": 3,
                    "timings": diagnosis.get(
                        "timings",
                        {},
                    ),
                }
            )
    return summarize(
        "gateway-v0.3",
        samples,
    )


def run_hybrid_core(
    *,
    chat_model: str,
    parser_model: str,
    ollama_url: str,
    prompts: list[str],
    repeats: int,
) -> dict:
    runtime = OllamaNeuralRuntime(
        ollama_url
    )
    gateway = HybridCoreGateway(
        HybridCoreConfig(
            chat_model=chat_model,
            parser_model=parser_model,
            ollama_url=ollama_url,
            persistence_dir=None,
            include_diagnosis=False,
            max_revision_cycles=1,
        ),
        runtime=runtime,
    )
    samples = []
    for repeat in range(
        repeats
    ):
        session_id = (
            f"bench-core-{repeat}"
        )
        for prompt in prompts:
            started = perf_counter()
            result = gateway.chat(
                session_id,
                prompt,
            )
            wall = (
                perf_counter()
                - started
            )
            runtime_packet = (
                result.cortex_context[
                    "hybrid_runtime"
                ]
            )
            samples.append(
                {
                    "prompt": prompt,
                    "wall_s": wall,
                    "neural_calls": (
                        runtime_packet[
                            "neural_call_count"
                        ]
                    ),
                    "calls": (
                        runtime_packet[
                            "neural_calls"
                        ]
                    ),
                }
            )
    return summarize(
        "hybrid-core-v0.1",
        samples,
    )


def summarize(
    name: str,
    samples: list[dict],
) -> dict:
    return {
        "name": name,
        "samples": samples,
        "mean_wall_s": mean(
            sample["wall_s"]
            for sample in samples
        ),
        "mean_neural_calls": mean(
            sample[
                "neural_calls"
            ]
            for sample in samples
        ),
        "sample_count": len(
            samples
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--chat-model",
        default="qwen3:8b",
    )
    parser.add_argument(
        "--parser-model",
        default="cortex-parser",
    )
    parser.add_argument(
        "--ollama-url",
        default="http://localhost:11434",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--prompt",
        action="append",
        default=[],
        help=(
            "Prompt to benchmark. Repeat for multiple prompts. "
            "Defaults to a small factual/memory sequence."
        ),
    )
    args = parser.parse_args()

    prompts = (
        args.prompt
        or DEFAULT_PROMPTS
    )
    repeats = max(
        1,
        int(
            args.repeats
        ),
    )

    v03 = run_v03(
        chat_model=args.chat_model,
        parser_model=(
            args.parser_model
        ),
        ollama_url=args.ollama_url,
        prompts=prompts,
        repeats=repeats,
    )
    core = run_hybrid_core(
        chat_model=args.chat_model,
        parser_model=(
            args.parser_model
        ),
        ollama_url=args.ollama_url,
        prompts=prompts,
        repeats=repeats,
    )

    improvement = (
        (
            v03[
                "mean_wall_s"
            ]
            / core[
                "mean_wall_s"
            ]
        )
        if core[
            "mean_wall_s"
        ] > 0
        else None
    )
    output = {
        "chat_model": (
            args.chat_model
        ),
        "parser_model": (
            args.parser_model
        ),
        "ollama_url": (
            args.ollama_url
        ),
        "gateway_v03": v03,
        "hybrid_core_v01": core,
        "wall_speedup_v03_over_core": (
            improvement
        ),
        "notes": [
            (
                "Results are hardware/model specific and are not "
                "scientific claims until the protocol is frozen."
            ),
            (
                "Gateway v0.3 performs knowledge, PromptIR, and "
                "candidate-analysis neural passes on normal turns."
            ),
            (
                "Hybrid Core fuses answer generation and candidate "
                "claims into one neural turn."
            ),
        ],
    }
    print(
        json.dumps(
            output,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
