#!/usr/bin/env python3
"""Controlled full-runtime scaling campaign for Cortex v1.0-RC1.

This benchmark characterizes the *native* CortexRuntime under independent
resource-axis sweeps. It is not a Python-vs-Rust speedup benchmark; that is
covered by benchmark_full_runtime.py.

Each case runs in a fresh subprocess so process peak RSS is attributable to one
configuration. Frame generation is deterministic and occurs before the timed
region. The public Python -> PyO3 -> Rust -> Python call is timed end-to-end.

The campaign deliberately includes benchmark-only low/high tensor-duty profiles.
Those profiles are not candidate production defaults; they exist to measure the
cost sensitivity of the conditional tensor/refinement machinery.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def normalize(x: list[float]) -> list[float]:
    mean = sum(x) / len(x)
    centered = [v - mean for v in x]
    var = sum(v * v for v in centered) / len(centered)
    scale = math.sqrt(var) + 1.0e-6
    return [v / scale for v in centered]


def shift(x: list[float], k: int) -> list[float]:
    k %= len(x)
    return x[-k:] + x[:-k] if k else list(x)


def make_frames(
    *,
    steps: int,
    seed: int,
    n_entities: int,
    feature_dim: int,
    visible: int,
    cluster_count: int,
) -> list[dict]:
    if n_entities < 2:
        raise ValueError("n_entities must be >= 2")
    if feature_dim < 4 or feature_dim % 4:
        raise ValueError("feature_dim must be a multiple of 4")
    if not 1 <= visible <= n_entities:
        raise ValueError("visible must be within [1, n_entities]")
    if cluster_count < 1 or n_entities % cluster_count:
        raise ValueError("cluster_count must divide n_entities")
    cluster_size = n_entities // cluster_count
    if visible > cluster_size:
        raise ValueError("visible cannot exceed cluster size")

    rng = random.Random(seed)
    raw = [[rng.gauss(0.0, 1.0) for _ in range(feature_dim)] for _ in range(n_entities)]
    prototypes = []
    for row in raw:
        smoothed = [
            row[i] + 0.55 * row[(i - 1) % feature_dim] + 0.35 * row[(i + 1) % feature_dim]
            for i in range(feature_dim)
        ]
        prototypes.append(normalize(smoothed))

    # Latent relation and causal truth. Only co-visible entity pairs are exposed
    # to the sparse graph carrier, so cluster_count controls represented global
    # graph density without changing missing-relation semantics.
    relation = [[False] * n_entities for _ in range(n_entities)]
    causal = [[False] * n_entities for _ in range(n_entities)]
    for i in range(n_entities):
        for j in range(i + 1, n_entities):
            relation[i][j] = relation[j][i] = rng.random() < 0.18
        for j in range(n_entities):
            if i != j:
                causal[i][j] = rng.random() < 0.07

    clusters = [
        list(range(c * cluster_size, (c + 1) * cluster_size)) for c in range(cluster_count)
    ]
    group = [0, feature_dim // 4, feature_dim // 2, 3 * feature_dim // 4]
    drift_t = max(2, steps // 2)
    frames: list[dict] = []

    for t in range(1, steps + 1):
        if t == drift_t:
            # One local feature drift plus one relation change. This keeps the
            # stream nonstationary without deliberately forcing pathological
            # concept proliferation.
            e = min(2, n_entities - 1)
            prototypes[e] = normalize(
                [0.72 * v + 0.28 * rng.gauss(0.0, 1.0) for v in prototypes[e]]
            )
            j = (e + 1) % n_entities
            relation[e][j] = not relation[e][j]
            relation[j][e] = relation[e][j]

        cluster = clusters[(t - 1) % cluster_count]
        ids = rng.sample(cluster, visible)
        detections: list[list[float]] = []
        for entity in ids:
            g = rng.choice(group)
            x = shift(prototypes[entity], g)
            x = [v + rng.gauss(0.0, 0.035) for v in x]
            for k in range(feature_dim):
                if rng.random() < 0.006:
                    x[k] += rng.gauss(0.0, 0.85)
            detections.append(x)

        relation_obs = []
        for i in range(visible):
            for j in range(i + 1, visible):
                p = 0.92 if relation[ids[i]][ids[j]] else 0.08
                relation_obs.append((i, j, int(rng.random() < p)))

        src_det = rng.randrange(visible) if visible > 1 and rng.random() < 0.35 else None
        outcomes = []
        for j, target in enumerate(ids):
            if src_det is not None and j != src_det:
                source = ids[src_det]
                p = 0.84 if causal[source][target] else 0.18
            else:
                p = 0.18
            outcomes.append((j, int(rng.random() < p)))

        frames.append(
            {
                "detections": detections,
                "relation_obs": relation_obs,
                "intervention_src_det": src_det,
                "outcomes": outcomes,
                "outcome": float(outcomes[0][1]) if outcomes else None,
            }
        )
    return frames


def percentile(values: list[float], p: float) -> float:
    xs = sorted(values)
    if not xs:
        return float("nan")
    idx = int(round((len(xs) - 1) * p))
    return xs[max(0, min(len(xs) - 1, idx))]


def resolved_config(
    *, feature_dim: int, max_entities: int, budget: int, tensor_profile: str
) -> str:
    from cortex import CortexRuntime

    seed_runtime = CortexRuntime(
        feature_dim=feature_dim,
        max_entities=max_entities,
        budget=budget,
    )
    cfg = json.loads(seed_runtime.config_json())
    if tensor_profile == "low":
        cfg["continual"]["initial_burst"] = 4
        cfg["continual"]["post_ready_burst"] = 2
        cfg["continual"]["refresh_interval"] = 1_000_000
        cfg["continual"]["refresh_burst"] = 1
        cfg["continual"]["degradation_ratio"] = 10.0
        cfg["continual"]["degradation_patience"] = 100_000
        cfg["continual"]["curvature_stride"] = 4
    elif tensor_profile == "high":
        # Keep the tensor monitor awake for the entire campaign and evaluate
        # curvature every observation. Benchmark-only stress profile.
        cfg["continual"]["initial_burst"] = 1_000_000
        cfg["continual"]["post_ready_burst"] = 1_000_000
        cfg["continual"]["refresh_interval"] = 1
        cfg["continual"]["refresh_burst"] = 1
        cfg["continual"]["curvature_stride"] = 1
    elif tensor_profile != "default":
        raise ValueError(f"unknown tensor profile: {tensor_profile}")
    return json.dumps(cfg, sort_keys=True)


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

    latency_us: list[float] = []
    last = None
    for i, frame in enumerate(frames):
        t0 = time.perf_counter_ns()
        last = runtime.step(
            frame["detections"],
            frame["relation_obs"],
            frame["intervention_src_det"],
            frame["outcomes"],
            frame["outcome"],
        )
        t1 = time.perf_counter_ns()
        if i >= warmup:
            latency_us.append((t1 - t0) / 1000.0)

    snap = json.loads(runtime.snapshot_json())
    cont = snap["continual"]
    graph = snap["graph"]
    tensor_total = int(cont["tensor_awake_steps"]) + int(cont["tensor_sleep_steps"])
    tensor_duty = float(cont["tensor_awake_steps"]) / tensor_total if tensor_total else 0.0
    relation_den = max(1, int(snap["articulation"]["active_entities"]) * (int(snap["articulation"]["active_entities"]) - 1) // 2)
    causal_den = max(1, int(snap["articulation"]["active_entities"]) * (int(snap["articulation"]["active_entities"]) - 1))
    rss_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    result = {
        **case,
        "repetition": repetition,
        "seed_used": seed,
        "steps_measured": len(latency_us),
        "mean_us": statistics.fmean(latency_us),
        "p50_us": percentile(latency_us, 0.50),
        "p95_us": percentile(latency_us, 0.95),
        "p99_us": percentile(latency_us, 0.99),
        "throughput_steps_s": 1_000_000.0 / statistics.fmean(latency_us),
        "peak_rss_mib": rss_kib / 1024.0,
        "active_entities": int(snap["articulation"]["active_entities"]),
        "represented_relations": len(graph["relations"]),
        "represented_causal": len(graph["causal"]),
        "relation_storage_fraction": len(graph["relations"]) / relation_den,
        "causal_storage_fraction": len(graph["causal"]) / causal_den,
        "memory_prototypes": len(snap["memory"]["prototypes"]),
        "stored_regimes": len(cont["prototypes"]),
        "tensor_duty": tensor_duty,
        "tensor_updates": int(cont["tensor_updates"]),
        "curvature_updates": int(cont["curvature_updates"]),
        "rank_checks": int(cont["rank_checks"]),
        "split_promotions": int(cont["split_promotions"]),
        "merge_promotions": int(cont["merge_promotions"]),
        "budget_pressure_count": int(cont["budget_pressure_count"]),
        "fingerprint": {
            "prediction": float(last.prediction),
            "stored": int(last.stored),
        },
    }
    return result


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

    def add(axis: str, value, **changes) -> None:
        case = dict(base)
        case.update(changes)
        case["axis"] = axis
        case["axis_value"] = value
        cases.append(case)

    for n in (8, 16, 32, 64, 96):
        add("entity_count", n, n_entities=n, visible=min(8, n))
    for visible in (2, 4, 8, 12, 16):
        add("visible_entities", visible, n_entities=64, visible=visible)
    for dim in (8, 16, 32, 64):
        add("feature_dim", dim, feature_dim=dim)
    for clusters in (1, 2, 4, 8):
        add("co_visibility_clusters", clusters, n_entities=64, visible=8, cluster_count=clusters)
    for budget in (4, 8, 24, 64):
        add("regime_budget", budget, budget=budget)
    for profile in ("low", "default", "high"):
        add("tensor_duty_profile", profile, tensor_profile=profile)
    return cases


def run_campaign(repetitions: int, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf8") as fh:
        for case_id, case in enumerate(campaign_cases()):
            for rep in range(1, repetitions + 1):
                payload = json.dumps({**case, "case_id": case_id, "repetition": rep})
                cmd = [sys.executable, __file__, "--case-json", payload]
                proc = subprocess.run(
                    cmd,
                    cwd=ROOT,
                    check=True,
                    text=True,
                    capture_output=True,
                    env={**os.environ, "PYTHONHASHSEED": "0"},
                )
                line = proc.stdout.strip().splitlines()[-1]
                result = json.loads(line.removeprefix("CORTEX_SCALING="))
                fh.write(json.dumps(result, sort_keys=True) + "\n")
                fh.flush()
                print(
                    f"[{case_id:02d}] {case['axis']}={case['axis_value']} rep={rep} "
                    f"mean={result['mean_us']:.2f}us p99={result['p99_us']:.2f}us "
                    f"rss={result['peak_rss_mib']:.1f}MiB tensor={result['tensor_duty']:.3f}",
                    flush=True,
                )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", action="store_true")
    ap.add_argument("--repetitions", type=int, default=3)
    ap.add_argument("--output", type=Path, default=ROOT / "native-scaling-results.jsonl")
    ap.add_argument("--case-json")
    args = ap.parse_args()

    if args.case_json:
        case = json.loads(args.case_json)
        repetition = int(case.pop("repetition", 1))
        result = run_case(case, repetition)
        print("CORTEX_SCALING=" + json.dumps(result, sort_keys=True))
        return
    if args.campaign:
        run_campaign(args.repetitions, args.output)
        return
    ap.error("choose --campaign or --case-json")


if __name__ == "__main__":
    main()
