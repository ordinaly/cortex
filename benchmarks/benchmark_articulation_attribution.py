#!/usr/bin/env python3
"""Fine-grained attribution of native Cortex articulation cost.

Protocol: articulation-attribution-v1

The benchmark calls the opt-in native `Articulation.profile_step` path. The
production `Articulation.step` path remains uninstrumented. Each campaign case
runs in a fresh subprocess and reuses the deterministic frame generator from
the native scaling campaign.
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

from benchmark_native_scaling import make_frames, percentile
from provenance import benchmark_provenance

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "articulation-attribution-v1"
STAGES = (
    "validation",
    "match_scoring",
    "assignment",
    "bind_finalize",
    "entity_update",
    "subgroup_evidence",
    "dedup",
    "prototype_reliability",
    "noise_state",
    "subgroup_select",
)
WORK_FIELDS = (
    "entities_before",
    "match_group_size",
    "pair_scores",
    "transform_evaluations",
    "assignment_rows",
    "assignment_cols",
    "spawned",
    "subgroup_evidence_detections",
    "subgroup_transform_evaluations",
)


def articulation_config(feature_dim: int, max_entities: int) -> str:
    return json.dumps(
        {
            "feature_dim": feature_dim,
            "max_entities": max_entities,
            "beta": 9.0,
            "spawn_cost": 0.7,
            "bootstrap_spawn_cost": 0.18,
            "residual_threshold": 0.38,
            "group_candidates": None,
        },
        sort_keys=True,
    )


def run_case(case: dict, repetition: int) -> dict:
    from cortex import Articulation

    seed = int(case.get("seed", 20260916)) + repetition * 1009
    steps = int(case.get("steps", 1200))
    warmup = int(case.get("warmup", 300))
    n_entities = int(case["n_entities"])
    feature_dim = int(case["feature_dim"])
    visible = int(case["visible"])
    cluster_count = int(case.get("cluster_count", 1))
    max_entities = int(case.get("max_entities", max(96, n_entities * 2)))

    frames = make_frames(
        steps=steps,
        seed=seed,
        n_entities=n_entities,
        feature_dim=feature_dim,
        visible=visible,
        cluster_count=cluster_count,
    )
    articulation = Articulation(
        feature_dim=feature_dim,
        max_entities=max_entities,
        config_json=articulation_config(feature_dim, max_entities),
    )

    stage_us: dict[str, list[float]] = {stage: [] for stage in STAGES}
    total_us: list[float] = []
    outer_us: list[float] = []
    work_values: dict[str, list[float]] = {field: [] for field in WORK_FIELDS}
    last = None

    for i, frame in enumerate(frames):
        t0 = time.perf_counter_ns()
        last, timings, work = articulation.profile_step(frame["detections"])
        t1 = time.perf_counter_ns()
        if i < warmup:
            continue

        outer_us.append((t1 - t0) / 1000.0)
        total_us.append(timings.total_ns / 1000.0)
        for stage in STAGES:
            stage_us[stage].append(getattr(timings, f"{stage}_ns") / 1000.0)
        for field in WORK_FIELDS:
            work_values[field].append(float(getattr(work, field)))

    stage_mean_us = {stage: statistics.fmean(values) for stage, values in stage_us.items()}
    stage_p50_us = {stage: percentile(values, 0.50) for stage, values in stage_us.items()}
    stage_p95_us = {stage: percentile(values, 0.95) for stage, values in stage_us.items()}
    attributed_mean_us = sum(stage_mean_us.values())
    internal_mean_us = statistics.fmean(total_us)
    dominant_stage = max(STAGES, key=stage_mean_us.__getitem__)
    snap = json.loads(articulation.snapshot_json())

    return {
        **benchmark_provenance(PROTOCOL),
        **case,
        "repetition": repetition,
        "seed_used": seed,
        "steps_measured": len(total_us),
        "outer_mean_us": statistics.fmean(outer_us),
        "outer_p50_us": percentile(outer_us, 0.50),
        "outer_p95_us": percentile(outer_us, 0.95),
        "outer_p99_us": percentile(outer_us, 0.99),
        "internal_mean_us": internal_mean_us,
        "internal_p50_us": percentile(total_us, 0.50),
        "internal_p95_us": percentile(total_us, 0.95),
        "internal_p99_us": percentile(total_us, 0.99),
        "attributed_mean_us": attributed_mean_us,
        "unattributed_internal_mean_us": max(0.0, internal_mean_us - attributed_mean_us),
        "stage_mean_us": stage_mean_us,
        "stage_p50_us": stage_p50_us,
        "stage_p95_us": stage_p95_us,
        "stage_share": {
            stage: stage_mean_us[stage] / attributed_mean_us if attributed_mean_us else 0.0
            for stage in STAGES
        },
        "dominant_stage": dominant_stage,
        "work_mean": {
            field: statistics.fmean(values) for field, values in work_values.items()
        },
        "active_entities": int(snap["active_entities"]),
        "subgroup_size": len(snap["subgroup"]),
        "subgroup_margin": float(snap["subgroup_margin"]),
        "fingerprint": {
            "bindings": list(last.bindings),
            "subgroup": list(last.subgroup),
            "active_entities": int(last.active_entities),
        },
    }


def campaign_cases() -> list[dict]:
    base = {
        "n_entities": 32,
        "feature_dim": 16,
        "visible": 8,
        "cluster_count": 1,
        "steps": 1200,
        "warmup": 300,
        "seed": 20260916,
    }
    cases: list[dict] = []

    def add(name: str, **changes) -> None:
        case = dict(base)
        case.update(changes)
        case["case"] = name
        cases.append(case)

    add("baseline")

    for n_entities in (8, 32, 64, 96):
        add(f"entities_{n_entities}", n_entities=n_entities, visible=min(8, n_entities))

    for visible in (2, 4, 8, 12, 16):
        add(f"visible_{visible}", n_entities=64, visible=visible)

    for feature_dim in (8, 16, 32, 64):
        add(f"feature_dim_{feature_dim}", feature_dim=feature_dim)

    add("sparse_topology", n_entities=64, visible=8, cluster_count=8)
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
                result = json.loads(line.removeprefix("CORTEX_ARTICULATION="))
                fh.write(json.dumps(result, sort_keys=True) + "\n")
                fh.flush()
                dominant = result["dominant_stage"]
                share = result["stage_share"][dominant]
                print(
                    f"[{case_id:02d}] {case['case']} rep={rep} "
                    f"internal={result['internal_mean_us']:.2f}us "
                    f"dominant={dominant}:{share:.1%} "
                    f"entities={result['active_entities']} subgroup={result['subgroup_size']}",
                    flush=True,
                )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", action="store_true")
    ap.add_argument("--repetitions", type=int, default=3)
    ap.add_argument(
        "--output", type=Path, default=ROOT / "articulation-attribution-v1.jsonl"
    )
    ap.add_argument("--case-json")
    args = ap.parse_args()

    if args.case_json:
        case = json.loads(args.case_json)
        repetition = int(case.pop("repetition", 1))
        result = run_case(case, repetition)
        print("CORTEX_ARTICULATION=" + json.dumps(result, sort_keys=True))
        return
    if args.campaign:
        run_campaign(args.repetitions, args.output)
        return
    ap.error("choose --campaign or --case-json")


if __name__ == "__main__":
    main()
