#!/usr/bin/env python3
"""Attribute composed native CortexRuntime cost to its major execution stages.

Protocol: stage-attribution-v1

The benchmark calls the opt-in Rust `profile_step` path. The production `step`
path remains uninstrumented. Each campaign case runs in a fresh subprocess and
reuses the deterministic frame generator from `benchmark_native_scaling.py`.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

from benchmark_native_scaling import make_frames, percentile, resolved_config
from provenance import benchmark_provenance

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "stage-attribution-v1"
STAGES = (
    "articulation",
    "binding",
    "graph",
    "summary",
    "memory",
    "continual",
)


def run_case(case: dict, repetition: int) -> dict:
    from cortex import CortexRuntime

    seed = int(case.get("seed", 20260916)) + repetition * 1009
    steps = int(case.get("steps", 1000))
    warmup = int(case.get("warmup", 200))
    n_entities = int(case["n_entities"])
    feature_dim = int(case["feature_dim"])
    visible = int(case["visible"])
    cluster_count = int(case.get("cluster_count", 1))
    budget = int(case.get("budget", 24))
    tensor_profile = str(case.get("tensor_profile", "default"))
    max_entities = int(case.get("max_entities", max(96, n_entities * 2)))

    frames = make_frames(
        steps=steps,
        seed=seed,
        n_entities=n_entities,
        feature_dim=feature_dim,
        visible=visible,
        cluster_count=cluster_count,
    )
    cfg = resolved_config(
        feature_dim=feature_dim,
        max_entities=max_entities,
        budget=budget,
        tensor_profile=tensor_profile,
    )
    runtime = CortexRuntime(config_json=cfg)

    stage_us: dict[str, list[float]] = {stage: [] for stage in STAGES}
    internal_total_us: list[float] = []
    outer_total_us: list[float] = []
    last = None

    for i, frame in enumerate(frames):
        t0 = time.perf_counter_ns()
        last, timings = runtime.profile_step(
            frame["detections"],
            frame["relation_obs"],
            frame["intervention_src_det"],
            frame["outcomes"],
            frame["outcome"],
        )
        t1 = time.perf_counter_ns()
        if i < warmup:
            continue
        outer_total_us.append((t1 - t0) / 1000.0)
        internal_total_us.append(timings.total_ns / 1000.0)
        for stage in STAGES:
            stage_us[stage].append(getattr(timings, f"{stage}_ns") / 1000.0)

    stage_mean_us = {stage: statistics.fmean(values) for stage, values in stage_us.items()}
    attributed_mean_us = sum(stage_mean_us.values())
    internal_mean_us = statistics.fmean(internal_total_us)
    dominant_stage = max(STAGES, key=stage_mean_us.__getitem__)
    snap = json.loads(runtime.snapshot_json())

    return {
        **benchmark_provenance(PROTOCOL),
        **case,
        "repetition": repetition,
        "seed_used": seed,
        "steps_measured": len(outer_total_us),
        "outer_mean_us": statistics.fmean(outer_total_us),
        "outer_p50_us": percentile(outer_total_us, 0.50),
        "outer_p95_us": percentile(outer_total_us, 0.95),
        "outer_p99_us": percentile(outer_total_us, 0.99),
        "internal_mean_us": internal_mean_us,
        "internal_p50_us": percentile(internal_total_us, 0.50),
        "internal_p95_us": percentile(internal_total_us, 0.95),
        "attributed_mean_us": attributed_mean_us,
        "unattributed_internal_mean_us": max(0.0, internal_mean_us - attributed_mean_us),
        "stage_mean_us": stage_mean_us,
        "stage_share": {
            stage: stage_mean_us[stage] / attributed_mean_us if attributed_mean_us else 0.0
            for stage in STAGES
        },
        "dominant_stage": dominant_stage,
        "active_entities": int(snap["articulation"]["active_entities"]),
        "represented_relations": len(snap["graph"]["relations"]),
        "represented_causal": len(snap["graph"]["causal"]),
        "memory_prototypes": len(snap["memory"]["prototypes"]),
        "stored_regimes": len(snap["continual"]["prototypes"]),
        "tensor_updates": int(snap["continual"]["tensor_updates"]),
        "curvature_updates": int(snap["continual"]["curvature_updates"]),
        "fingerprint": {
            "prediction": float(last.prediction),
            "stored": int(last.stored),
        },
    }


def campaign_cases() -> list[dict]:
    base = {
        "n_entities": 32,
        "feature_dim": 16,
        "visible": 8,
        "cluster_count": 1,
        "budget": 24,
        "tensor_profile": "default",
        "steps": 1000,
        "warmup": 200,
        "seed": 20260916,
    }
    cases: list[dict] = []

    def add(name: str, **changes) -> None:
        case = dict(base)
        case.update(changes)
        case["case"] = name
        cases.append(case)

    add("baseline")
    add("entity_96", n_entities=96, visible=8)
    add("visible_16", n_entities=64, visible=16)
    add("feature_dim_64", feature_dim=64)
    add("sparse_topology", n_entities=64, visible=8, cluster_count=8)
    add("tensor_high", tensor_profile="high")
    return cases


def run_campaign(repetitions: int, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf8") as fh:
        for case_id, case in enumerate(campaign_cases()):
            for rep in range(1, repetitions + 1):
                payload = json.dumps({**case, "case_id": case_id, "repetition": rep})
                proc = subprocess.run(
                    [sys.executable, __file__, "--case-json", payload],
                    cwd=ROOT,
                    check=True,
                    text=True,
                    capture_output=True,
                    env={**os.environ, "PYTHONHASHSEED": "0"},
                )
                line = proc.stdout.strip().splitlines()[-1]
                result = json.loads(line.removeprefix("CORTEX_STAGE="))
                fh.write(json.dumps(result, sort_keys=True) + "\n")
                fh.flush()
                dominant = result["dominant_stage"]
                share = result["stage_share"][dominant]
                print(
                    f"[{case_id:02d}] {case['case']} rep={rep} "
                    f"internal={result['internal_mean_us']:.2f}us "
                    f"dominant={dominant}:{share:.1%}",
                    flush=True,
                )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", action="store_true")
    ap.add_argument("--repetitions", type=int, default=3)
    ap.add_argument(
        "--output", type=Path, default=ROOT / "stage-attribution-results.jsonl"
    )
    ap.add_argument("--case-json")
    args = ap.parse_args()

    if args.case_json:
        case = json.loads(args.case_json)
        repetition = int(case.pop("repetition", 1))
        result = run_case(case, repetition)
        print("CORTEX_STAGE=" + json.dumps(result, sort_keys=True))
        return
    if args.campaign:
        run_campaign(args.repetitions, args.output)
        return
    ap.error("choose --campaign or --case-json")


if __name__ == "__main__":
    main()
