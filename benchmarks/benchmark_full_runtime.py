#!/usr/bin/env python3
"""Benchmark frozen Python, hybrid, and fully native Cortex execution.

Each invocation runs one mode in a fresh process so peak RSS is attributable to
that deployment shape rather than contaminated by imports from the other modes.
The workload is generated deterministically using only the Python standard
library, then materialized before timing.

Modes
-----
python : frozen v0.8 articulation/memory + frozen v0.9.5 continual controller
hybrid : frozen v0.8 articulation/memory + native Rust continual controller
native : single-call Rust CortexRuntime (articulation + sparse graph + memory + continual)
"""
from __future__ import annotations

import argparse
import json
import math
import random
import resource
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REF_ROOT = ROOT / "reference" / "v0_9_5"


def _normalize(x: list[float]) -> list[float]:
    mean = sum(x) / len(x)
    centered = [v - mean for v in x]
    var = sum(v * v for v in centered) / len(centered)
    scale = math.sqrt(var) + 1.0e-6
    return [v / scale for v in centered]


def _shift(x: list[float], k: int) -> list[float]:
    if not x:
        return []
    k %= len(x)
    return x[-k:] + x[:-k] if k else list(x)


def make_frames(
    *,
    steps: int,
    seed: int,
    n_entities: int = 12,
    feature_dim: int = 16,
    visible: int = 5,
) -> list[dict]:
    rng = random.Random(seed)
    raw = [[rng.gauss(0.0, 1.0) for _ in range(feature_dim)] for _ in range(n_entities)]
    prototypes = []
    for row in raw:
        sm = [
            row[i] + 0.55 * row[(i - 1) % feature_dim] + 0.35 * row[(i + 1) % feature_dim]
            for i in range(feature_dim)
        ]
        prototypes.append(_normalize(sm))

    relation = [[False] * n_entities for _ in range(n_entities)]
    causal = [[False] * n_entities for _ in range(n_entities)]
    rel_p = min(0.22, 4.0 / max(4, n_entities))
    causal_p = min(0.10, 2.0 / max(4, n_entities))
    for i in range(n_entities):
        for j in range(i + 1, n_entities):
            relation[i][j] = relation[j][i] = rng.random() < rel_p
        for j in range(n_entities):
            if i != j:
                causal[i][j] = rng.random() < causal_p

    group = [0, feature_dim // 4, feature_dim // 2, 3 * feature_dim // 4]
    drift_t = max(2, steps // 2)
    drift_entity = 2
    frames: list[dict] = []
    for t in range(1, steps + 1):
        if t == drift_t:
            p = prototypes[drift_entity]
            mixed = [0.70 * v + 0.30 * rng.gauss(0.0, 1.0) for v in p]
            prototypes[drift_entity] = _normalize(mixed)
            j = (drift_entity + 1) % n_entities
            relation[drift_entity][j] = not relation[drift_entity][j]
            relation[j][drift_entity] = relation[drift_entity][j]

        ids = rng.sample(range(n_entities), min(visible, n_entities))
        detections: list[list[float]] = []
        for entity in ids:
            g = rng.choice(group)
            x = _shift(prototypes[entity], g)
            x = [v + rng.gauss(0.0, 0.05) for v in x]
            for k in range(feature_dim):
                if rng.random() < 0.02:
                    x[k] += rng.gauss(0.0, 1.25)
            detections.append(x)

        relation_obs = []
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                p = 0.90 if relation[ids[i]][ids[j]] else 0.10
                relation_obs.append((i, j, int(rng.random() < p)))

        src_det = rng.randrange(len(ids)) if len(ids) > 1 and rng.random() < 0.35 else None
        outcomes = []
        for j, target in enumerate(ids):
            if src_det is not None and j != src_det:
                source = ids[src_det]
                p = 0.86 if causal[source][target] else 0.20
            else:
                p = 0.20
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


def reference_continual(budget: int):
    if str(REF_ROOT) not in sys.path:
        sys.path.insert(0, str(REF_ROOT))
    from cortex_v095_compute_optimized import ComputeOptimizedRefinementReasoner

    return ComputeOptimizedRefinementReasoner(
        dim=12,
        budget=budget,
        evidence_decay=0.85,
        hazard_lr=0.35,
        recurrence_threshold=0.75,
        initial_burst=24,
        post_ready_burst=8,
        refresh_interval=144,
        refresh_burst=4,
        degradation_ratio=1.0,
        degradation_patience=6,
        curvature_stride=2,
    )


def percentile(values: list[float], p: float) -> float:
    xs = sorted(values)
    if not xs:
        return float("nan")
    i = int(round((len(xs) - 1) * p))
    return xs[max(0, min(len(xs) - 1, i))]


def prepare_python_frames(frames: list[dict]):
    import numpy as np
    if str(REF_ROOT) not in sys.path:
        sys.path.insert(0, str(REF_ROOT))
    from cortex_v07 import Detection, Frame

    out = []
    for f in frames:
        out.append(
            Frame(
                detections=tuple(Detection(np.asarray(x, dtype=float)) for x in f["detections"]),
                relation_obs=tuple(tuple(r) for r in f["relation_obs"]),
                intervention_src_det=f["intervention_src_det"],
                outcomes=tuple(tuple(r) for r in f["outcomes"]),
            )
        )
    return out


def run_python(frames: list[dict], warmup: int, budget: int, max_entities: int):
    if str(REF_ROOT) not in sys.path:
        sys.path.insert(0, str(REF_ROOT))
    from cortex_v08 import CortexV08Runtime

    prepared = prepare_python_frames(frames)
    v08 = CortexV08Runtime(
        memory_dim=12,
        memory_budget=budget,
        feature_dim=16,
        max_entities=max_entities,
    )
    cont = reference_continual(budget)

    lat = []
    last = None
    for i, (frame, raw) in enumerate(zip(prepared, frames)):
        t0 = time.perf_counter_ns()
        _, _ = v08.step(frame)
        last = cont.step(v08.articulation_vector(), raw["outcome"])
        t1 = time.perf_counter_ns()
        if i >= warmup:
            lat.append((t1 - t0) / 1000.0)
    return lat, {"stored": len(cont.prototypes), "prediction": float(last.prediction)}


def run_hybrid(frames: list[dict], warmup: int, budget: int, max_entities: int):
    if str(REF_ROOT) not in sys.path:
        sys.path.insert(0, str(REF_ROOT))
    from cortex_v08 import CortexV08Runtime
    from cortex import Cortex

    prepared = prepare_python_frames(frames)
    v08 = CortexV08Runtime(
        memory_dim=12,
        memory_budget=budget,
        feature_dim=16,
        max_entities=max_entities,
    )
    cont = Cortex(dim=12, budget=budget)

    lat = []
    last = None
    for i, (frame, raw) in enumerate(zip(prepared, frames)):
        t0 = time.perf_counter_ns()
        _, _ = v08.step(frame)
        last = cont.step(v08.articulation_vector().tolist(), raw["outcome"])
        t1 = time.perf_counter_ns()
        if i >= warmup:
            lat.append((t1 - t0) / 1000.0)
    return lat, {"stored": int(last.stored), "prediction": float(last.prediction)}


def run_native(frames: list[dict], warmup: int, budget: int, max_entities: int):
    from cortex import CortexRuntime

    rt = CortexRuntime(feature_dim=16, max_entities=max_entities, budget=budget)
    lat = []
    last = None
    for i, f in enumerate(frames):
        t0 = time.perf_counter_ns()
        last = rt.step(
            f["detections"],
            f["relation_obs"],
            f["intervention_src_det"],
            f["outcomes"],
            f["outcome"],
        )
        t1 = time.perf_counter_ns()
        if i >= warmup:
            lat.append((t1 - t0) / 1000.0)
    return lat, {"stored": int(last.stored), "prediction": float(last.prediction)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("python", "hybrid", "native"), required=True)
    ap.add_argument("--steps", type=int, default=1400)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20260916)
    ap.add_argument("--budget", type=int, default=24)
    ap.add_argument("--max-entities", type=int, default=96)
    args = ap.parse_args()
    if args.warmup >= args.steps:
        raise SystemExit("warmup must be smaller than steps")

    frames = make_frames(steps=args.steps, seed=args.seed)
    runners = {"python": run_python, "hybrid": run_hybrid, "native": run_native}
    lat, fingerprint = runners[args.mode](frames, args.warmup, args.budget, args.max_entities)
    rss_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result = {
        "mode": args.mode,
        "steps_total": args.steps,
        "steps_measured": len(lat),
        "seed": args.seed,
        "mean_us": statistics.fmean(lat),
        "p50_us": percentile(lat, 0.50),
        "p95_us": percentile(lat, 0.95),
        "p99_us": percentile(lat, 0.99),
        "peak_rss_mib": rss_kib / 1024.0,
        "fingerprint": fingerprint,
    }
    print("CORTEX_BENCHMARK=" + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
