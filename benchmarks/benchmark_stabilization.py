#!/usr/bin/env python3
"""Long-horizon concept-stabilization probes for Cortex.

Protocol: stabilization-v1

This campaign asks whether repeated chronological experience produces a stable
internal structure rather than merely good local decisions. It deliberately
separates two layers:

1. articulation stability: persistent entity identity under nuisance transforms,
   ordinary noise, long absences, novelty, and transient sparse corruption;
2. continual-regime stability: whether discovery/churn decays on a familiar
   stream, whether a dormant regime is reactivated, and whether localized
   novelty causes a bounded temporary disturbance before restabilization.

The fixtures are synthetic and ground-truthed. They are a prerequisite for, not
an alternative to, a future real-world video/sensor stream. Finite success here
does not establish real-world concept formation.
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
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from provenance import benchmark_provenance

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "stabilization-v1"
ART_DIM = 16
CORE_DIM = 12
TRUE_GROUP = (0, 4, 8, 12)


def config_hash(config_json: str) -> str:
    return hashlib.sha256(config_json.encode("utf8")).hexdigest()


def normalize(x: list[float]) -> list[float]:
    mean = statistics.fmean(x)
    centered = [v - mean for v in x]
    var = statistics.fmean(v * v for v in centered)
    scale = math.sqrt(var) + 1.0e-9
    return [v / scale for v in centered]


def shift(x: list[float], k: int) -> list[float]:
    k %= len(x)
    return x[-k:] + x[:-k] if k else list(x)


def make_identity_prototypes(n: int, rng: random.Random) -> list[list[float]]:
    out: list[list[float]] = []
    for _ in range(n):
        raw = [rng.gauss(0.0, 1.0) for _ in range(ART_DIM)]
        smooth = [
            raw[i]
            + 0.55 * raw[(i - 1) % ART_DIM]
            + 0.35 * raw[(i + 1) % ART_DIM]
            for i in range(ART_DIM)
        ]
        out.append(normalize(smooth))
    return out


def render_identity(
    proto: list[float],
    rng: random.Random,
    *,
    noise: float,
    corrupt_p: float = 0.0,
    corrupt_scale: float = 1.0,
) -> list[float]:
    x = shift(proto, rng.choice(TRUE_GROUP))
    x = [v + rng.gauss(0.0, noise) for v in x]
    if corrupt_p > 0.0:
        for i in range(len(x)):
            if rng.random() < corrupt_p:
                x[i] += rng.gauss(0.0, corrupt_scale)
    return x


def identity_metrics(records: list[tuple[int, int]]) -> dict[str, float]:
    if not records:
        return {
            "purity": 0.0,
            "fragmentation": 0.0,
            "collision": 0.0,
            "dominant_share": 0.0,
            "switch_rate": 0.0,
        }
    by_binding: dict[int, Counter[int]] = defaultdict(Counter)
    by_truth: dict[int, Counter[int]] = defaultdict(Counter)
    last: dict[int, int] = {}
    switches = 0
    comparable = 0
    for truth, binding in records:
        by_binding[binding][truth] += 1
        by_truth[truth][binding] += 1
        if truth in last:
            comparable += 1
            switches += int(last[truth] != binding)
        last[truth] = binding
    correct = sum(max(c.values()) for c in by_binding.values())
    fragmentation = statistics.fmean(len(c) for c in by_truth.values())
    collision = statistics.fmean(len(c) for c in by_binding.values())
    dominant = statistics.fmean(max(c.values()) / sum(c.values()) for c in by_truth.values())
    return {
        "purity": correct / len(records),
        "fragmentation": fragmentation,
        "collision": collision,
        "dominant_share": dominant,
        "switch_rate": switches / max(1, comparable),
    }


def dominant_binding(records: list[tuple[int, int]], truth: int) -> int | None:
    c = Counter(binding for label, binding in records if label == truth)
    return c.most_common(1)[0][0] if c else None


def runtime_config(max_entities: int = 64) -> tuple[str, Any]:
    from cortex import CortexRuntime

    seed = CortexRuntime(feature_dim=ART_DIM, max_entities=max_entities, budget=24)
    cfg = seed.config_json()
    return cfg, CortexRuntime(config_json=cfg)


def runtime_step(runtime: Any, ids: list[int], protos: list[list[float]], rng: random.Random,
                 *, noise: float, corrupt: dict[int, tuple[float, float]] | None = None):
    corrupt = corrupt or {}
    detections = []
    for entity in ids:
        p, scale = corrupt.get(entity, (0.0, 1.0))
        detections.append(
            render_identity(protos[entity], rng, noise=noise, corrupt_p=p, corrupt_scale=scale)
        )
    return runtime.step(detections, [], None, [], None)


def run_identity_stabilization(case: dict[str, Any], repetition: int) -> dict[str, Any]:
    seed = int(case["seed"]) + repetition * 1009
    rng = random.Random(seed)
    n = 8
    visible = 4
    steps = 12_000
    window = 2_000
    noise = float(case["noise"])
    corrupt_p = float(case["corrupt_p"])
    protos = make_identity_prototypes(n, rng)
    cfg_json, runtime = runtime_config()

    records: list[tuple[int, int]] = []
    late_records: list[tuple[int, int]] = []
    active_by_window: list[int] = []
    subgroup_correct = 0
    subgroup_trials = 0
    capacity_failure = False
    error_text = None

    for t in range(steps):
        ids = rng.sample(range(n), visible)
        try:
            read = runtime_step(
                runtime, ids, protos, rng, noise=noise,
                corrupt={i: (corrupt_p, 1.0) for i in ids} if corrupt_p else None,
            )
        except Exception as exc:
            capacity_failure = True
            error_text = f"{type(exc).__name__}: {exc}"
            break
        pairs = list(zip(ids, map(int, read.bindings)))
        records.extend(pairs)
        if t >= steps - 4_000:
            late_records.extend(pairs)
        if t >= 1_000:
            subgroup_trials += 1
            subgroup_correct += int(tuple(read.subgroup) == TRUE_GROUP)
        if (t + 1) % window == 0:
            active_by_window.append(int(read.bindings and runtime.snapshot_json() is not None and json.loads(runtime.snapshot_json())["articulation"]["active_entities"]))

    snap = json.loads(runtime.snapshot_json())
    active = int(snap["articulation"]["active_entities"])
    increments: list[int] = []
    prev = 0
    for value in active_by_window:
        increments.append(value - prev)
        prev = value
    m = identity_metrics(records)
    lm = identity_metrics(late_records)
    return {
        **benchmark_provenance(PROTOCOL),
        **case,
        "repetition": repetition,
        "seed_used": seed,
        "config_sha256": config_hash(cfg_json),
        "steps_requested": steps,
        "steps_completed": snap["articulation"]["time"],
        "capacity_failure": capacity_failure,
        "model_error": error_text,
        "active_entities": active,
        "new_entities_by_window": increments,
        "late_new_entities": sum(increments[len(increments)//2:]),
        "purity": m["purity"],
        "fragmentation": m["fragmentation"],
        "dominant_share": m["dominant_share"],
        "switch_rate": m["switch_rate"],
        "late_purity": lm["purity"],
        "late_fragmentation": lm["fragmentation"],
        "late_dominant_share": lm["dominant_share"],
        "late_switch_rate": lm["switch_rate"],
        "subgroup_correct_fraction": subgroup_correct / max(1, subgroup_trials),
    }


def run_absence_return(case: dict[str, Any], repetition: int) -> dict[str, Any]:
    seed = int(case["seed"]) + repetition * 1009
    rng = random.Random(seed)
    gap = int(case["gap"])
    n = 8
    target = 0
    protos = make_identity_prototypes(n, rng)
    cfg_json, runtime = runtime_config()
    train_records: list[tuple[int, int]] = []

    for t in range(5_000):
        ids = rng.sample(range(n), 4)
        read = runtime_step(runtime, ids, protos, rng, noise=0.05)
        if t >= 3_000:
            train_records.extend(zip(ids, map(int, read.bindings)))
    target_id = dominant_binding(train_records, target)
    before_gap = json.loads(runtime.snapshot_json())["articulation"]["active_entities"]

    for _ in range(gap):
        ids = rng.sample(range(1, n), 4)
        runtime_step(runtime, ids, protos, rng, noise=0.05)
    after_gap = json.loads(runtime.snapshot_json())["articulation"]["active_entities"]

    returned: list[int] = []
    first_discovery_active = None
    for i in range(300):
        ids = [target] + rng.sample(range(1, n), 3)
        read = runtime_step(runtime, ids, protos, rng, noise=0.05)
        returned.append(int(read.bindings[0]))
        if i == 0:
            first_discovery_active = int(read.active_entities)
    final = json.loads(runtime.snapshot_json())["articulation"]["active_entities"]
    same = sum(int(b == target_id) for b in returned) if target_id is not None else 0
    return {
        **benchmark_provenance(PROTOCOL),
        **case,
        "repetition": repetition,
        "seed_used": seed,
        "config_sha256": config_hash(cfg_json),
        "target_binding_before_gap": target_id,
        "first_return_binding": returned[0],
        "first_return_same": target_id is not None and returned[0] == target_id,
        "return_same_fraction": same / len(returned),
        "new_entities_during_gap": int(after_gap - before_gap),
        "new_entities_during_return": int(final - after_gap),
        "active_before_gap": int(before_gap),
        "active_after_gap": int(after_gap),
        "active_after_return": int(final),
        "first_return_active_entities": first_discovery_active,
    }


def run_novelty_recovery(case: dict[str, Any], repetition: int) -> dict[str, Any]:
    seed = int(case["seed"]) + repetition * 1009
    rng = random.Random(seed)
    base_n = 8
    novel = 8
    protos = make_identity_prototypes(9, rng)
    cfg_json, runtime = runtime_config()
    recent: list[tuple[int, int]] = []
    seen_ids: set[int] = set()

    for t in range(6_000):
        ids = rng.sample(range(base_n), 4)
        read = runtime_step(runtime, ids, protos, rng, noise=0.05)
        seen_ids.update(map(int, read.bindings))
        if t >= 4_000:
            recent.extend(zip(ids, map(int, read.bindings)))
    old_map = {e: dominant_binding(recent, e) for e in range(base_n)}
    active_before = json.loads(runtime.snapshot_json())["articulation"]["active_entities"]

    first_novel_new = None
    novel_bindings: list[int] = []
    old_obs = 0
    old_mismatch = 0
    for i in range(500):
        ids = [novel] + rng.sample(range(base_n), 3)
        read = runtime_step(runtime, ids, protos, rng, noise=0.05)
        nb = int(read.bindings[0])
        novel_bindings.append(nb)
        if first_novel_new is None and nb not in seen_ids:
            first_novel_new = i
        for entity, binding in zip(ids[1:], map(int, read.bindings[1:])):
            old_obs += 1
            old_mismatch += int(old_map[entity] is not None and binding != old_map[entity])
        seen_ids.update(map(int, read.bindings))
    active_after_novelty = json.loads(runtime.snapshot_json())["articulation"]["active_entities"]

    post_records: list[tuple[int, int]] = []
    for _ in range(4_000):
        ids = rng.sample(range(9), 4)
        read = runtime_step(runtime, ids, protos, rng, noise=0.05)
        post_records.extend(zip(ids, map(int, read.bindings)))
    active_final = json.loads(runtime.snapshot_json())["articulation"]["active_entities"]
    novel_post = Counter(binding for truth, binding in post_records if truth == novel)
    novel_dom = novel_post.most_common(1)[0][1] / sum(novel_post.values()) if novel_post else 0.0
    return {
        **benchmark_provenance(PROTOCOL),
        **case,
        "repetition": repetition,
        "seed_used": seed,
        "config_sha256": config_hash(cfg_json),
        "active_before_novelty": int(active_before),
        "active_after_novelty": int(active_after_novelty),
        "active_final": int(active_final),
        "novelty_new_identity_delay": first_novel_new,
        "novelty_unique_bindings": len(set(novel_bindings)),
        "old_identity_mismatch_during_novelty": old_mismatch / max(1, old_obs),
        "new_entities_during_novelty": int(active_after_novelty - active_before),
        "new_entities_after_novelty": int(active_final - active_after_novelty),
        "novel_post_dominant_share": novel_dom,
        "post_fragmentation": identity_metrics(post_records)["fragmentation"],
    }


def run_corruption_burst(case: dict[str, Any], repetition: int) -> dict[str, Any]:
    seed = int(case["seed"]) + repetition * 1009
    rng = random.Random(seed)
    n = 8
    target = 0
    burst = int(case["burst"])
    corrupt_p = float(case["corrupt_p"])
    scale = float(case["corrupt_scale"])
    protos = make_identity_prototypes(n, rng)
    cfg_json, runtime = runtime_config()
    recent: list[tuple[int, int]] = []

    for t in range(5_000):
        ids = rng.sample(range(n), 4)
        read = runtime_step(runtime, ids, protos, rng, noise=0.05)
        if t >= 3_000:
            recent.extend(zip(ids, map(int, read.bindings)))
    target_id = dominant_binding(recent, target)
    active_before = json.loads(runtime.snapshot_json())["articulation"]["active_entities"]

    burst_bindings: list[int] = []
    for _ in range(burst):
        ids = [target] + rng.sample(range(1, n), 3)
        read = runtime_step(
            runtime, ids, protos, rng, noise=0.05,
            corrupt={target: (corrupt_p, scale)},
        )
        burst_bindings.append(int(read.bindings[0]))
    active_after_burst = json.loads(runtime.snapshot_json())["articulation"]["active_entities"]

    recovery: list[int] = []
    for _ in range(1_000):
        ids = [target] + rng.sample(range(1, n), 3)
        read = runtime_step(runtime, ids, protos, rng, noise=0.05)
        recovery.append(int(read.bindings[0]))
    active_final = json.loads(runtime.snapshot_json())["articulation"]["active_entities"]
    return {
        **benchmark_provenance(PROTOCOL),
        **case,
        "repetition": repetition,
        "seed_used": seed,
        "config_sha256": config_hash(cfg_json),
        "target_binding_before_burst": target_id,
        "burst_unique_target_bindings": len(set(burst_bindings)),
        "burst_original_identity_fraction": sum(int(b == target_id) for b in burst_bindings) / max(1, len(burst_bindings)),
        "new_entities_during_burst": int(active_after_burst - active_before),
        "recovery_original_identity_fraction": sum(int(b == target_id) for b in recovery) / len(recovery),
        "first_recovery_same": bool(recovery and recovery[0] == target_id),
        "new_entities_during_recovery": int(active_final - active_after_burst),
        "permanent_excess_entities": int(active_final - n),
        "active_before_burst": int(active_before),
        "active_after_burst": int(active_after_burst),
        "active_final": int(active_final),
    }


def latent_prototypes(k: int, separation: float) -> list[list[float]]:
    a = separation * math.sqrt(CORE_DIM / 2.0)
    out = []
    for j in range(k):
        x = [0.0] * CORE_DIM
        x[j] = a
        out.append(x)
    return out


def sample_core(proto: list[float], noise: float, rng: random.Random) -> list[float]:
    return [v + rng.gauss(0.0, noise) for v in proto]


def run_continual_stabilization(case: dict[str, Any], repetition: int) -> dict[str, Any]:
    from cortex import Cortex

    seed = int(case["seed"]) + repetition * 1009
    rng = random.Random(seed)
    noise = float(case["noise"])
    protos = latent_prototypes(5, 0.90)
    seed_model = Cortex(dim=CORE_DIM, budget=12)
    cfg_json = seed_model.config_json()
    model = Cortex(dim=CORE_DIM, budget=12, config_json=cfg_json)

    windows: list[dict[str, int]] = []
    last_counts = {"discovery": 0, "reactivation": 0, "revision": 0, "unresolved": 0}
    acc = dict(last_counts)
    step = 0

    def feed(label: int) -> Any:
        nonlocal step
        step += 1
        read = model.step(sample_core(protos[label], noise, rng), float(label % 2))
        acc["discovery"] += int(read.discovered)
        acc["reactivation"] += int(read.reactivated)
        acc["revision"] += int(read.revision)
        acc["unresolved"] += int(read.unresolved)
        if step % 5_000 == 0:
            windows.append({
                "end": step,
                "discoveries": acc["discovery"] - last_counts["discovery"],
                "reactivations": acc["reactivation"] - last_counts["reactivation"],
                "revisions": acc["revision"] - last_counts["revision"],
                "unresolved": acc["unresolved"] - last_counts["unresolved"],
                "stored": int(read.stored),
            })
            last_counts.update(acc)
        return read

    # Familiarization + long stable exposure.
    for t in range(40_000):
        feed((t // 8) % 4)
    snap_pre_gap = json.loads(model.snapshot_json())

    # Long absence of regime 0 while the rest of the world continues.
    for t in range(10_000):
        feed(1 + ((t // 8) % 3))
    before_return_discovery = acc["discovery"]
    before_return_reactivation = acc["reactivation"]
    first_return = None
    for t in range(2_000):
        r = feed((t // 8) % 4)
        if first_return is None and (t // 8) % 4 == 0:
            first_return = {
                "discovered": bool(r.discovered),
                "reactivated": bool(r.reactivated),
                "stored": int(r.stored),
            }
    return_discoveries = acc["discovery"] - before_return_discovery
    return_reactivations = acc["reactivation"] - before_return_reactivation

    # Localized novelty followed by a longer settling phase.
    before_novel_discovery = acc["discovery"]
    novelty_delay = None
    for t in range(1_000):
        r = feed(4)
        if novelty_delay is None and r.discovered:
            novelty_delay = t
    novelty_discoveries = acc["discovery"] - before_novel_discovery
    after_novel = json.loads(model.snapshot_json())

    post_discovery_start = acc["discovery"]
    for t in range(10_000):
        feed((t // 8) % 5)
    final = json.loads(model.snapshot_json())
    post_discoveries = acc["discovery"] - post_discovery_start

    if len(final["prototypes"]) > 12:
        raise RuntimeError("continual budget invariant violated")
    return {
        **benchmark_provenance(PROTOCOL),
        **case,
        "repetition": repetition,
        "seed_used": seed,
        "config_sha256": config_hash(cfg_json),
        "windows": windows,
        "stored_before_gap": len(snap_pre_gap["prototypes"]),
        "stored_after_novelty": len(after_novel["prototypes"]),
        "stored_final": len(final["prototypes"]),
        "first_return": first_return,
        "return_discoveries": return_discoveries,
        "return_reactivations": return_reactivations,
        "novelty_detection_delay": novelty_delay,
        "novelty_discoveries": novelty_discoveries,
        "post_novelty_discoveries": post_discoveries,
        "final_discovery_count": int(final["discovery_count"]),
        "final_reactivation_count": int(final["reactivation_count"]),
        "final_unresolved_count": int(final["unresolved_count"]),
        "split_promotions": int(final["split_promotions"]),
        "merge_promotions": int(final["merge_promotions"]),
        "budget_pressure_count": int(final["budget_pressure_count"]),
    }


def run_case(case: dict[str, Any], repetition: int) -> dict[str, Any]:
    suite = str(case["suite"])
    if suite == "identity_stabilization":
        return run_identity_stabilization(case, repetition)
    if suite == "absence_return":
        return run_absence_return(case, repetition)
    if suite == "novelty_recovery":
        return run_novelty_recovery(case, repetition)
    if suite == "corruption_burst":
        return run_corruption_burst(case, repetition)
    if suite == "continual_stabilization":
        return run_continual_stabilization(case, repetition)
    raise ValueError(f"unknown suite {suite!r}")


def campaign_cases() -> list[dict[str, Any]]:
    seed = 20260917
    cases: list[dict[str, Any]] = [
        {"suite": "identity_stabilization", "case": "identity_clean", "noise": 0.05, "corrupt_p": 0.0, "seed": seed},
        {"suite": "identity_stabilization", "case": "identity_sparse_002", "noise": 0.05, "corrupt_p": 0.02, "seed": seed + 1},
    ]
    for gap in (100, 1_000, 5_000, 20_000):
        cases.append({"suite": "absence_return", "case": f"absence_{gap}", "gap": gap, "seed": seed + 100})
    cases.append({"suite": "novelty_recovery", "case": "localized_novelty", "seed": seed + 200})
    for burst, corrupt_p in ((1, 0.10), (5, 0.10), (20, 0.10), (100, 0.10), (1, 0.20), (20, 0.20)):
        cases.append({
            "suite": "corruption_burst",
            "case": f"burst_{burst}_p_{corrupt_p:.2f}",
            "burst": burst,
            "corrupt_p": corrupt_p,
            "corrupt_scale": 1.25,
            "seed": seed + 300,
        })
    for noise in (0.03, 0.12):
        cases.append({
            "suite": "continual_stabilization",
            "case": f"continual_noise_{noise:.2f}",
            "noise": noise,
            "seed": seed + 400,
        })
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
                        f"stabilization child failed for {case['case']} rep={rep}:\n"
                        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
                    )
                line = proc.stdout.strip().splitlines()[-1]
                result = json.loads(line.removeprefix("CORTEX_STABILIZATION="))
                fh.write(json.dumps(result, sort_keys=True) + "\n")
                fh.flush()
                print(f"[{case_id:02d}/{len(cases):02d}] {case['case']} rep={rep}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", action="store_true")
    ap.add_argument("--repetitions", type=int, default=3)
    ap.add_argument("--output", type=Path, default=ROOT / "stabilization-v1.jsonl")
    ap.add_argument("--case-json")
    args = ap.parse_args()
    if args.case_json:
        case = json.loads(args.case_json)
        repetition = int(case.pop("repetition", 1))
        result = run_case(case, repetition)
        print("CORTEX_STABILIZATION=" + json.dumps(result, sort_keys=True))
        return
    if args.campaign:
        run_campaign(args.repetitions, args.output)
        return
    ap.error("choose --campaign or --case-json")


if __name__ == "__main__":
    main()
