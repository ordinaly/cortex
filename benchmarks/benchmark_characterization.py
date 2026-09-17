#!/usr/bin/env python3
"""Ground-truthed scientific characterization campaign for Cortex.

Protocol: characterization-v1

The goal is not to tune Cortex for these fixtures. The campaign maps strengths,
weaknesses, and phase boundaries under frozen configurations and deterministic
seeds. It covers continual recurrence, novelty detection, gradual drift,
budget pressure, full-stack identity binding under nuisance/noise, and long-run
state stability. Continual cases also include a deliberately simple bounded
online nearest-centroid baseline using Cortex's configured recurrence tolerance.

Finite success here is evidence only for the stated synthetic ranges; it is not
a proof of general continual-learning performance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from provenance import benchmark_provenance

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "characterization-v1"
DIM = 12


def rms(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)) / len(a))


def shift(x: list[float], k: int) -> list[float]:
    k %= len(x)
    return x[-k:] + x[:-k] if k else list(x)


def normalize(x: list[float]) -> list[float]:
    mean = sum(x) / len(x)
    centered = [v - mean for v in x]
    var = sum(v * v for v in centered) / len(centered)
    scale = math.sqrt(var) + 1.0e-9
    return [v / scale for v in centered]


def config_hash(config_json: str) -> str:
    return hashlib.sha256(config_json.encode("utf8")).hexdigest()


def canonical_continual_config(budget: int) -> tuple[str, dict[str, Any]]:
    from cortex import Cortex

    seed = Cortex(dim=DIM, budget=budget)
    cfg_json = seed.config_json()
    return cfg_json, json.loads(cfg_json)


def latent_prototypes(k: int, separation: float, dim: int = DIM) -> list[list[float]]:
    if k > dim:
        raise ValueError("orthogonal latent fixture requires k <= dim")
    # For two distinct basis vectors with amplitude a, RMS distance is
    # a * sqrt(2 / dim). Choose a so the pairwise RMS separation is exact.
    a = separation * math.sqrt(dim / 2.0)
    out: list[list[float]] = []
    for j in range(k):
        x = [0.0] * dim
        x[j] = a
        out.append(x)
    return out


def sample(proto: list[float], noise: float, rng: random.Random) -> list[float]:
    return [v + rng.gauss(0.0, noise) for v in proto]


def stable_uid(model, read) -> int | None:
    if read.current_id is None:
        return None
    snap = json.loads(model.snapshot_json())
    idx = int(read.current_id)
    if not 0 <= idx < len(snap["prototypes"]):
        return None
    return int(snap["prototypes"][idx]["uid"])


def clustering_metrics(truth: list[int], predicted: list[int | None]) -> dict[str, float]:
    if len(truth) != len(predicted):
        raise ValueError("truth/predicted length mismatch")
    assigned = [(t, p) for t, p in zip(truth, predicted) if p is not None]
    if not truth:
        return {
            "assignment_rate": 0.0,
            "purity": 0.0,
            "fragmentation": 0.0,
            "collision": 0.0,
        }
    if not assigned:
        return {
            "assignment_rate": 0.0,
            "purity": 0.0,
            "fragmentation": 0.0,
            "collision": 0.0,
        }

    by_pred: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    by_true: dict[int, set[int]] = defaultdict(set)
    for t, p0 in assigned:
        p = int(p0)
        by_pred[p][t] += 1
        by_true[t].add(p)

    correct = sum(max(counts.values()) for counts in by_pred.values())
    fragmentation = statistics.fmean(len(ids) for ids in by_true.values())
    collision = statistics.fmean(len(counts) for counts in by_pred.values())
    return {
        "assignment_rate": len(assigned) / len(truth),
        "purity": correct / len(assigned),
        "fragmentation": fragmentation,
        "collision": collision,
    }


class BoundedNearestCentroid:
    """Simple sanity baseline: thresholded running centroids with a hard budget."""

    def __init__(self, *, budget: int, tolerance: float):
        self.budget = budget
        self.tolerance = tolerance
        self.centroids: list[list[float]] = []
        self.counts: list[int] = []
        self.discovery_count = 0
        self.reactivation_count = 0
        self.pressure_count = 0
        self.current_id: int | None = None

    def step(self, x: list[float]) -> tuple[int | None, bool, bool, bool]:
        if not self.centroids:
            self.centroids.append(list(x))
            self.counts.append(1)
            self.current_id = 0
            self.discovery_count += 1
            return 0, True, False, False

        distances = [rms(x, c) for c in self.centroids]
        nearest = min(range(len(distances)), key=lambda i: (distances[i], i))
        if distances[nearest] <= self.tolerance:
            reactivated = self.current_id is not None and nearest != self.current_id
            self.current_id = nearest
            n = self.counts[nearest] + 1
            old = self.centroids[nearest]
            self.centroids[nearest] = [v + (z - v) / n for v, z in zip(old, x)]
            self.counts[nearest] = n
            if reactivated:
                self.reactivation_count += 1
            return nearest, False, reactivated, False

        if len(self.centroids) < self.budget:
            new_id = len(self.centroids)
            self.centroids.append(list(x))
            self.counts.append(1)
            self.current_id = new_id
            self.discovery_count += 1
            return new_id, True, False, False

        self.pressure_count += 1
        self.current_id = None
        return None, False, False, True


def run_recurrence_case(case: dict[str, Any], repetition: int) -> dict[str, Any]:
    from cortex import Cortex

    seed = int(case["seed"]) + repetition * 1009
    rng = random.Random(seed)
    k = 4
    budget = 8
    separation = float(case["separation"])
    noise = float(case["noise"])
    cycles = 12
    block = 6
    protos = latent_prototypes(k, separation)
    cfg_json, cfg = canonical_continual_config(budget)
    model = Cortex(dim=DIM, budget=budget, config_json=cfg_json)
    baseline = BoundedNearestCentroid(
        budget=budget, tolerance=float(cfg["recurrence_tolerance"])
    )

    truth: list[int] = []
    cortex_ids: list[int | None] = []
    baseline_ids: list[int | None] = []
    cortex_discovery_steps: list[int] = []
    baseline_discovery_steps: list[int] = []
    cortex_reactivations = 0
    baseline_reactivations = 0
    step_i = 0

    for cycle in range(cycles):
        order = list(range(k))
        rng.shuffle(order)
        for label in order:
            for _ in range(block):
                step_i += 1
                x = sample(protos[label], noise, rng)
                read = model.step(x, float(label % 2))
                uid = stable_uid(model, read)
                bid, bdisc, breact, _ = baseline.step(x)
                truth.append(label)
                cortex_ids.append(uid)
                baseline_ids.append(bid)
                if read.discovered:
                    cortex_discovery_steps.append(step_i)
                if bdisc:
                    baseline_discovery_steps.append(step_i)
                cortex_reactivations += int(read.reactivated)
                baseline_reactivations += int(breact)

    warmup = k * block * 2
    cm = clustering_metrics(truth[warmup:], cortex_ids[warmup:])
    bm = clustering_metrics(truth[warmup:], baseline_ids[warmup:])
    snap = json.loads(model.snapshot_json())
    false_cortex = sum(s > warmup for s in cortex_discovery_steps)
    false_baseline = sum(s > warmup for s in baseline_discovery_steps)
    denom = max(1, len(truth) - warmup)

    return {
        **benchmark_provenance(PROTOCOL),
        **case,
        "repetition": repetition,
        "seed_used": seed,
        "config_sha256": config_hash(cfg_json),
        "recurrence_tolerance": float(cfg["recurrence_tolerance"]),
        "cortex_assignment_rate": cm["assignment_rate"],
        "cortex_purity": cm["purity"],
        "cortex_fragmentation": cm["fragmentation"],
        "cortex_collision": cm["collision"],
        "cortex_false_discovery_rate": false_cortex / denom,
        "cortex_discoveries": int(snap["discovery_count"]),
        "cortex_reactivations": int(snap["reactivation_count"]),
        "cortex_unresolved": int(snap["unresolved_count"]),
        "cortex_stored": len(snap["prototypes"]),
        "baseline_assignment_rate": bm["assignment_rate"],
        "baseline_purity": bm["purity"],
        "baseline_fragmentation": bm["fragmentation"],
        "baseline_collision": bm["collision"],
        "baseline_false_discovery_rate": false_baseline / denom,
        "baseline_discoveries": baseline.discovery_count,
        "baseline_reactivations": baseline.reactivation_count,
        "baseline_stored": len(baseline.centroids),
        "observed_cortex_reactivation_events": cortex_reactivations,
        "observed_baseline_reactivation_events": baseline_reactivations,
    }


def run_novelty_case(case: dict[str, Any], repetition: int) -> dict[str, Any]:
    from cortex import Cortex

    seed = int(case["seed"]) + repetition * 1009
    rng = random.Random(seed)
    separation = float(case["separation"])
    noise = float(case["noise"])
    budget = 8
    protos = latent_prototypes(3, separation)
    cfg_json, cfg = canonical_continual_config(budget)
    model = Cortex(dim=DIM, budget=budget, config_json=cfg_json)
    baseline = BoundedNearestCentroid(
        budget=budget, tolerance=float(cfg["recurrence_tolerance"])
    )

    pre_labels = ([0] * 6 + [1] * 6) * 12
    novel_labels = [2] * 18
    post_labels = ([0] * 6 + [1] * 6) * 4
    labels = pre_labels + novel_labels + post_labels
    novel_start = len(pre_labels)
    novel_end = novel_start + len(novel_labels)

    c_false = 0
    b_false = 0
    c_delay: int | None = None
    b_delay: int | None = None
    c_new_uid: int | None = None
    b_new_id: int | None = None

    for i, label in enumerate(labels):
        x = sample(protos[label], noise, rng)
        read = model.step(x, float(label % 2))
        uid = stable_uid(model, read)
        bid, bdisc, _, _ = baseline.step(x)
        if i >= 24 and i < novel_start and read.discovered:
            c_false += 1
        if i >= 24 and i < novel_start and bdisc:
            b_false += 1
        if novel_start <= i < novel_end:
            if c_delay is None and read.discovered:
                c_delay = i - novel_start
                c_new_uid = uid
            if b_delay is None and bdisc:
                b_delay = i - novel_start
                b_new_id = bid

    snap = json.loads(model.snapshot_json())
    return {
        **benchmark_provenance(PROTOCOL),
        **case,
        "repetition": repetition,
        "seed_used": seed,
        "config_sha256": config_hash(cfg_json),
        "recurrence_tolerance": float(cfg["recurrence_tolerance"]),
        "cortex_detected": c_delay is not None,
        "cortex_detection_delay": c_delay,
        "cortex_novel_uid": c_new_uid,
        "cortex_false_discoveries_pre": c_false,
        "cortex_discoveries": int(snap["discovery_count"]),
        "cortex_reactivations": int(snap["reactivation_count"]),
        "cortex_stored": len(snap["prototypes"]),
        "baseline_detected": b_delay is not None,
        "baseline_detection_delay": b_delay,
        "baseline_novel_id": b_new_id,
        "baseline_false_discoveries_pre": b_false,
        "baseline_discoveries": baseline.discovery_count,
        "baseline_reactivations": baseline.reactivation_count,
        "baseline_stored": len(baseline.centroids),
    }


def run_drift_case(case: dict[str, Any], repetition: int) -> dict[str, Any]:
    from cortex import Cortex

    seed = int(case["seed"]) + repetition * 1009
    rng = random.Random(seed)
    total_drift = float(case["total_drift"])
    noise = float(case["noise"])
    budget = 10
    separation = 0.9
    protos = latent_prototypes(3, separation)
    cfg_json, cfg = canonical_continual_config(budget)
    model = Cortex(dim=DIM, budget=budget, config_json=cfg_json)
    baseline = BoundedNearestCentroid(
        budget=budget, tolerance=float(cfg["recurrence_tolerance"])
    )

    truth: list[int] = []
    c_ids: list[int | None] = []
    b_ids: list[int | None] = []
    cycles = 30
    block = 4
    total_target_obs = cycles * block
    target_seen = 0
    warmup_steps = 3 * block * 3
    c_false = 0
    b_false = 0
    step_i = 0

    for _cycle in range(cycles):
        for label in (0, 1, 2):
            for _ in range(block):
                step_i += 1
                proto = list(protos[label])
                if label == 0:
                    frac = target_seen / max(1, total_target_obs - 1)
                    # Move target regime by an exact RMS displacement total_drift.
                    proto[5] += frac * total_drift * math.sqrt(DIM)
                    target_seen += 1
                x = sample(proto, noise, rng)
                read = model.step(x, float(label % 2))
                uid = stable_uid(model, read)
                bid, bdisc, _, _ = baseline.step(x)
                truth.append(label)
                c_ids.append(uid)
                b_ids.append(bid)
                if step_i > warmup_steps and read.discovered:
                    c_false += 1
                if step_i > warmup_steps and bdisc:
                    b_false += 1

    cm = clustering_metrics(truth[warmup_steps:], c_ids[warmup_steps:])
    bm = clustering_metrics(truth[warmup_steps:], b_ids[warmup_steps:])
    target_c = clustering_metrics(
        [t for t in truth[warmup_steps:] if t == 0],
        [p for t, p in zip(truth[warmup_steps:], c_ids[warmup_steps:]) if t == 0],
    )
    target_b = clustering_metrics(
        [t for t in truth[warmup_steps:] if t == 0],
        [p for t, p in zip(truth[warmup_steps:], b_ids[warmup_steps:]) if t == 0],
    )
    snap = json.loads(model.snapshot_json())
    return {
        **benchmark_provenance(PROTOCOL),
        **case,
        "repetition": repetition,
        "seed_used": seed,
        "config_sha256": config_hash(cfg_json),
        "recurrence_tolerance": float(cfg["recurrence_tolerance"]),
        "cortex_purity": cm["purity"],
        "cortex_fragmentation": cm["fragmentation"],
        "cortex_target_fragmentation": target_c["fragmentation"],
        "cortex_post_warm_discoveries": c_false,
        "cortex_stored": len(snap["prototypes"]),
        "cortex_split_promotions": int(snap["split_promotions"]),
        "cortex_merge_promotions": int(snap["merge_promotions"]),
        "baseline_purity": bm["purity"],
        "baseline_fragmentation": bm["fragmentation"],
        "baseline_target_fragmentation": target_b["fragmentation"],
        "baseline_post_warm_discoveries": b_false,
        "baseline_stored": len(baseline.centroids),
    }


def run_budget_case(case: dict[str, Any], repetition: int) -> dict[str, Any]:
    from cortex import Cortex

    seed = int(case["seed"]) + repetition * 1009
    rng = random.Random(seed)
    budget = int(case["budget"])
    k = budget + 3
    protos = latent_prototypes(k, 1.8)
    cfg_json, cfg = canonical_continual_config(budget)
    model = Cortex(dim=DIM, budget=budget, config_json=cfg_json)
    baseline = BoundedNearestCentroid(
        budget=budget, tolerance=float(cfg["recurrence_tolerance"])
    )

    for label in range(k):
        for _ in range(8):
            x = sample(protos[label], 0.02, rng)
            read = model.step(x, float(label % 2))
            baseline.step(x)
            if read.stored > budget:
                raise RuntimeError(
                    f"Cortex budget invariant violated: stored={read.stored} budget={budget}"
                )

    snap = json.loads(model.snapshot_json())
    stored = len(snap["prototypes"])
    if stored > budget:
        raise RuntimeError(f"snapshot budget invariant violated: {stored}>{budget}")
    return {
        **benchmark_provenance(PROTOCOL),
        **case,
        "repetition": repetition,
        "seed_used": seed,
        "config_sha256": config_hash(cfg_json),
        "attempted_regimes": k,
        "cortex_stored": stored,
        "cortex_budget_pressure": int(snap["budget_pressure_count"]),
        "cortex_unresolved": int(snap["unresolved_count"]),
        "cortex_discoveries": int(snap["discovery_count"]),
        "baseline_stored": len(baseline.centroids),
        "baseline_budget_pressure": baseline.pressure_count,
        "baseline_discoveries": baseline.discovery_count,
    }


def make_identity_prototypes(n_entities: int, dim: int, rng: random.Random) -> list[list[float]]:
    raw = [[rng.gauss(0.0, 1.0) for _ in range(dim)] for _ in range(n_entities)]
    return [
        normalize(
            [row[i] + 0.55 * row[(i - 1) % dim] + 0.35 * row[(i + 1) % dim] for i in range(dim)]
        )
        for row in raw
    ]


def run_identity_case(case: dict[str, Any], repetition: int) -> dict[str, Any]:
    from cortex import CortexRuntime

    seed = int(case["seed"]) + repetition * 1009
    rng = random.Random(seed)
    dim = 16
    n_entities = 12
    visible = 6
    steps = 700
    noise = float(case["noise"])
    corrupt_p = float(case["corrupt_p"])
    max_entities = 64
    runtime0 = CortexRuntime(feature_dim=dim, max_entities=max_entities, budget=24)
    cfg_json = runtime0.config_json()
    runtime = CortexRuntime(config_json=cfg_json)
    protos = make_identity_prototypes(n_entities, dim, rng)
    group = [0, dim // 4, dim // 2, 3 * dim // 4]

    truth: list[int] = []
    bindings: list[int | None] = []
    capacity_failure = False
    completed = 0
    warmup = 100

    for t in range(steps):
        ids = rng.sample(range(n_entities), visible)
        detections: list[list[float]] = []
        for entity in ids:
            x = shift(protos[entity], rng.choice(group))
            x = [v + rng.gauss(0.0, noise) for v in x]
            for j in range(dim):
                if rng.random() < corrupt_p:
                    x[j] += rng.gauss(0.0, 1.0)
            detections.append(x)
        try:
            read = runtime.step(detections, [], None, [], None)
        except Exception as exc:  # capacity exhaustion is a scientific outcome here
            capacity_failure = True
            error_text = f"{type(exc).__name__}: {exc}"
            break
        completed += 1
        if t >= warmup:
            truth.extend(ids)
            bindings.extend(int(x) for x in read.bindings)
    else:
        error_text = None

    snap = json.loads(runtime.snapshot_json())
    cm = clustering_metrics(truth, bindings)
    active = int(snap["articulation"]["active_entities"])
    return {
        **benchmark_provenance(PROTOCOL),
        **case,
        "repetition": repetition,
        "seed_used": seed,
        "config_sha256": config_hash(cfg_json),
        "steps_completed": completed,
        "capacity_failure": capacity_failure,
        "model_error": error_text,
        "binding_assignment_rate": cm["assignment_rate"],
        "binding_purity": cm["purity"],
        "binding_fragmentation": cm["fragmentation"],
        "binding_collision": cm["collision"],
        "active_entities": active,
        "entity_overfragmentation": active / n_entities,
        "subgroup_size": len(snap["articulation"]["subgroup"]),
        "subgroup_margin": float(snap["articulation"]["subgroup_margin"]),
    }


def count_nonfinite(value: Any) -> int:
    if isinstance(value, float):
        return 0 if math.isfinite(value) else 1
    if isinstance(value, dict):
        return sum(count_nonfinite(v) for v in value.values())
    if isinstance(value, list):
        return sum(count_nonfinite(v) for v in value)
    return 0


def percentile(values: list[float], p: float) -> float:
    xs = sorted(values)
    if not xs:
        return float("nan")
    i = int(round((len(xs) - 1) * p))
    return xs[max(0, min(len(xs) - 1, i))]


def run_long_run_case(case: dict[str, Any], repetition: int) -> dict[str, Any]:
    from cortex import Cortex

    seed = int(case["seed"]) + repetition * 1009
    rng = random.Random(seed)
    noise = float(case["noise"])
    budget = 12
    steps = 8000
    protos = latent_prototypes(4, 0.9)
    cfg_json, _cfg = canonical_continual_config(budget)
    model = Cortex(dim=DIM, budget=budget, config_json=cfg_json)
    latencies_us: list[float] = []

    for t in range(steps):
        label = (t // 12) % 4
        proto = list(protos[label])
        if label == 0:
            # Reversible slow drift: no unbounded movement of the latent process.
            proto[7] += 0.24 * math.sqrt(DIM) * math.sin(2.0 * math.pi * t / 900.0)
        x = sample(proto, noise, rng)
        t0 = time.perf_counter_ns()
        read = model.step(x, float(label % 2))
        t1 = time.perf_counter_ns()
        if t >= 500:
            latencies_us.append((t1 - t0) / 1000.0)
        if read.stored > budget:
            raise RuntimeError(f"long-run budget invariant violated: {read.stored}>{budget}")

    snap = json.loads(model.snapshot_json())
    nonfinite = count_nonfinite(snap)
    stored = len(snap["prototypes"])
    if stored > budget:
        raise RuntimeError(f"long-run snapshot budget invariant violated: {stored}>{budget}")
    return {
        **benchmark_provenance(PROTOCOL),
        **case,
        "repetition": repetition,
        "seed_used": seed,
        "config_sha256": config_hash(cfg_json),
        "steps": steps,
        "nonfinite_values": nonfinite,
        "stored": stored,
        "discoveries": int(snap["discovery_count"]),
        "reactivations": int(snap["reactivation_count"]),
        "unresolved": int(snap["unresolved_count"]),
        "budget_pressure": int(snap["budget_pressure_count"]),
        "split_promotions": int(snap["split_promotions"]),
        "merge_promotions": int(snap["merge_promotions"]),
        "mean_step_us": statistics.fmean(latencies_us),
        "p95_step_us": percentile(latencies_us, 0.95),
        "p99_step_us": percentile(latencies_us, 0.99),
    }


def run_case(case: dict[str, Any], repetition: int) -> dict[str, Any]:
    suite = str(case["suite"])
    if suite == "recurrence":
        return run_recurrence_case(case, repetition)
    if suite == "novelty":
        return run_novelty_case(case, repetition)
    if suite == "drift":
        return run_drift_case(case, repetition)
    if suite == "budget":
        return run_budget_case(case, repetition)
    if suite == "identity":
        return run_identity_case(case, repetition)
    if suite == "long_run":
        return run_long_run_case(case, repetition)
    raise ValueError(f"unknown suite {suite!r}")


def campaign_cases() -> list[dict[str, Any]]:
    seed = 20260917
    cases: list[dict[str, Any]] = []
    for separation in (0.20, 0.35, 0.50, 0.75, 1.00):
        for noise in (0.02, 0.08, 0.18):
            cases.append(
                {
                    "suite": "recurrence",
                    "case": f"rec_sep_{separation:.2f}_noise_{noise:.2f}",
                    "separation": separation,
                    "noise": noise,
                    "seed": seed,
                }
            )
    for separation in (0.35, 0.50, 0.75, 1.00, 1.40):
        for noise in (0.03, 0.12):
            cases.append(
                {
                    "suite": "novelty",
                    "case": f"nov_sep_{separation:.2f}_noise_{noise:.2f}",
                    "separation": separation,
                    "noise": noise,
                    "seed": seed + 100,
                }
            )
    for total_drift in (0.20, 0.40, 0.80, 1.20):
        for noise in (0.03, 0.12):
            cases.append(
                {
                    "suite": "drift",
                    "case": f"drift_{total_drift:.2f}_noise_{noise:.2f}",
                    "total_drift": total_drift,
                    "noise": noise,
                    "seed": seed + 200,
                }
            )
    for budget in (2, 4, 8):
        cases.append(
            {
                "suite": "budget",
                "case": f"budget_{budget}",
                "budget": budget,
                "seed": seed + 300,
            }
        )
    for noise in (0.01, 0.05, 0.12, 0.22):
        for corrupt_p in (0.0, 0.03, 0.10):
            cases.append(
                {
                    "suite": "identity",
                    "case": f"identity_noise_{noise:.2f}_corrupt_{corrupt_p:.2f}",
                    "noise": noise,
                    "corrupt_p": corrupt_p,
                    "seed": seed + 400,
                }
            )
    for noise in (0.05, 0.15):
        cases.append(
            {
                "suite": "long_run",
                "case": f"long_run_noise_{noise:.2f}",
                "noise": noise,
                "seed": seed + 500,
            }
        )
    return cases


def run_campaign(repetitions: int, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    cases = campaign_cases()
    with output.open("w", encoding="utf8") as fh:
        for case_id, case in enumerate(cases):
            for rep in range(1, repetitions + 1):
                payload = json.dumps({**case, "case_id": case_id, "repetition": rep})
                proc = subprocess.run(
                    [sys.executable, __file__, "--case-json", payload],
                    cwd=ROOT,
                    check=False,
                    text=True,
                    capture_output=True,
                    env={
                        **os.environ,
                        "PYTHONHASHSEED": "0",
                        "OMP_NUM_THREADS": "1",
                        "OPENBLAS_NUM_THREADS": "1",
                        "MKL_NUM_THREADS": "1",
                    },
                )
                if proc.returncode != 0:
                    raise RuntimeError(
                        f"characterization child failed for {case['case']} rep={rep}:\n"
                        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
                    )
                line = proc.stdout.strip().splitlines()[-1]
                result = json.loads(line.removeprefix("CORTEX_CHARACTERIZATION="))
                fh.write(json.dumps(result, sort_keys=True) + "\n")
                fh.flush()
                print(
                    f"[{case_id:02d}/{len(cases):02d}] {case['case']} rep={rep}",
                    flush=True,
                )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", action="store_true")
    ap.add_argument("--repetitions", type=int, default=3)
    ap.add_argument(
        "--output", type=Path, default=ROOT / "characterization-v1.jsonl"
    )
    ap.add_argument("--case-json")
    args = ap.parse_args()

    if args.case_json:
        case = json.loads(args.case_json)
        repetition = int(case.pop("repetition", 1))
        result = run_case(case, repetition)
        print("CORTEX_CHARACTERIZATION=" + json.dumps(result, sort_keys=True))
        return
    if args.campaign:
        run_campaign(args.repetitions, args.output)
        return
    ap.error("choose --campaign or --case-json")


if __name__ == "__main__":
    main()
