#!/usr/bin/env python3
"""Long-horizon concept-stabilization probes for Cortex.

Protocol: stabilization-v1

Synthetic, ground-truthed prerequisite for a later real video/sensor stream.
The campaign separates articulation identity stability from continual-regime
stability and records scientific failures as outcomes rather than tuning them
away. Finite success here is not evidence of real-world concept formation.
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


def cfg_hash(s: str) -> str:
    return hashlib.sha256(s.encode("utf8")).hexdigest()


def normalize(x: list[float]) -> list[float]:
    m = statistics.fmean(x)
    z = [v - m for v in x]
    scale = math.sqrt(statistics.fmean(v * v for v in z)) + 1e-9
    return [v / scale for v in z]


def shift(x: list[float], k: int) -> list[float]:
    k %= len(x)
    return x[-k:] + x[:-k] if k else list(x)


def identity_prototypes(n: int, rng: random.Random) -> list[list[float]]:
    out = []
    for _ in range(n):
        raw = [rng.gauss(0.0, 1.0) for _ in range(ART_DIM)]
        out.append(normalize([
            raw[i] + 0.55 * raw[(i - 1) % ART_DIM] + 0.35 * raw[(i + 1) % ART_DIM]
            for i in range(ART_DIM)
        ]))
    return out


def render(proto: list[float], rng: random.Random, noise: float,
           corrupt_p: float = 0.0, corrupt_scale: float = 1.0) -> list[float]:
    x = shift(proto, rng.choice(TRUE_GROUP))
    x = [v + rng.gauss(0.0, noise) for v in x]
    for i in range(len(x)):
        if rng.random() < corrupt_p:
            x[i] += rng.gauss(0.0, corrupt_scale)
    return x


def identity_metrics(records: list[tuple[int, int]]) -> dict[str, float]:
    if not records:
        return {"purity": 0.0, "fragmentation": 0.0, "dominant_share": 0.0, "switch_rate": 0.0}
    by_binding: dict[int, Counter[int]] = defaultdict(Counter)
    by_truth: dict[int, Counter[int]] = defaultdict(Counter)
    last: dict[int, int] = {}
    switches = comparable = 0
    for truth, binding in records:
        by_binding[binding][truth] += 1
        by_truth[truth][binding] += 1
        if truth in last:
            comparable += 1
            switches += int(last[truth] != binding)
        last[truth] = binding
    return {
        "purity": sum(max(c.values()) for c in by_binding.values()) / len(records),
        "fragmentation": statistics.fmean(len(c) for c in by_truth.values()),
        "dominant_share": statistics.fmean(max(c.values()) / sum(c.values()) for c in by_truth.values()),
        "switch_rate": switches / max(1, comparable),
    }


def dominant(records: list[tuple[int, int]], truth: int) -> int | None:
    c = Counter(binding for label, binding in records if label == truth)
    return c.most_common(1)[0][0] if c else None


def runtime_pair(max_entities: int = 64):
    from cortex import CortexRuntime
    seed = CortexRuntime(feature_dim=ART_DIM, max_entities=max_entities, budget=24)
    cfg = seed.config_json()
    return cfg, CortexRuntime(config_json=cfg)


def active_entities(runtime: Any) -> int:
    return int(json.loads(runtime.snapshot_json())["articulation"]["active_entities"])


def rstep(runtime: Any, ids: list[int], protos: list[list[float]], rng: random.Random,
          noise: float = 0.05, corrupt: dict[int, tuple[float, float]] | None = None):
    corrupt = corrupt or {}
    xs = []
    for entity in ids:
        p, scale = corrupt.get(entity, (0.0, 1.0))
        xs.append(render(protos[entity], rng, noise, p, scale))
    return runtime.step(xs, [], None, [], None)


def run_identity_stabilization(case: dict[str, Any], rep: int) -> dict[str, Any]:
    seed = int(case["seed"]) + rep * 1009
    rng = random.Random(seed)
    n, steps, window = 8, 12_000, 2_000
    protos = identity_prototypes(n, rng)
    cfg, runtime = runtime_pair()
    records: list[tuple[int, int]] = []
    late: list[tuple[int, int]] = []
    active_windows: list[int] = []
    sg_ok = sg_n = 0
    error = None
    for t in range(steps):
        ids = rng.sample(range(n), 4)
        try:
            read = rstep(runtime, ids, protos, rng, float(case["noise"]),
                         {i: (float(case["corrupt_p"]), 1.0) for i in ids} if case["corrupt_p"] else None)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            break
        pairs = list(zip(ids, map(int, read.bindings)))
        records.extend(pairs)
        if t >= 8_000:
            late.extend(pairs)
        if t >= 1_000:
            sg_n += 1
            sg_ok += int(tuple(read.subgroup) == TRUE_GROUP)
        if (t + 1) % window == 0:
            active_windows.append(active_entities(runtime))
    increments, prev = [], 0
    for v in active_windows:
        increments.append(v - prev)
        prev = v
    m, lm = identity_metrics(records), identity_metrics(late)
    return {
        **benchmark_provenance(PROTOCOL), **case, "repetition": rep, "seed_used": seed,
        "config_sha256": cfg_hash(cfg), "capacity_failure": error is not None, "model_error": error,
        "active_entities": active_entities(runtime), "new_entities_by_window": increments,
        "late_new_entities": sum(increments[len(increments)//2:]),
        "purity": m["purity"], "fragmentation": m["fragmentation"],
        "dominant_share": m["dominant_share"], "switch_rate": m["switch_rate"],
        "late_purity": lm["purity"], "late_fragmentation": lm["fragmentation"],
        "late_dominant_share": lm["dominant_share"], "late_switch_rate": lm["switch_rate"],
        "subgroup_correct_fraction": sg_ok / max(1, sg_n),
    }


def run_absence_return(case: dict[str, Any], rep: int) -> dict[str, Any]:
    seed = int(case["seed"]) + rep * 1009
    rng = random.Random(seed)
    n, target, gap = 8, 0, int(case["gap"])
    protos = identity_prototypes(n, rng)
    cfg, runtime = runtime_pair()
    recent: list[tuple[int, int]] = []
    for t in range(5_000):
        ids = rng.sample(range(n), 4)
        read = rstep(runtime, ids, protos, rng)
        if t >= 3_000:
            recent.extend(zip(ids, map(int, read.bindings)))
    target_id = dominant(recent, target)
    before = active_entities(runtime)
    for _ in range(gap):
        rstep(runtime, rng.sample(range(1, n), 4), protos, rng)
    after_gap = active_entities(runtime)
    returned = []
    first_active = None
    for i in range(300):
        read = rstep(runtime, [target] + rng.sample(range(1, n), 3), protos, rng)
        returned.append(int(read.bindings[0]))
        if i == 0:
            first_active = active_entities(runtime)
    final = active_entities(runtime)
    return {
        **benchmark_provenance(PROTOCOL), **case, "repetition": rep, "seed_used": seed,
        "config_sha256": cfg_hash(cfg), "target_binding_before_gap": target_id,
        "first_return_binding": returned[0], "first_return_same": returned[0] == target_id,
        "return_same_fraction": sum(b == target_id for b in returned) / len(returned),
        "new_entities_during_gap": after_gap - before, "new_entities_during_return": final - after_gap,
        "active_before_gap": before, "active_after_gap": after_gap, "active_after_return": final,
        "first_return_active_entities": first_active,
    }


def run_novelty_recovery(case: dict[str, Any], rep: int) -> dict[str, Any]:
    seed = int(case["seed"]) + rep * 1009
    rng = random.Random(seed)
    base_n, novel = 8, 8
    protos = identity_prototypes(9, rng)
    cfg, runtime = runtime_pair()
    recent: list[tuple[int, int]] = []
    seen: set[int] = set()
    for t in range(6_000):
        ids = rng.sample(range(base_n), 4)
        read = rstep(runtime, ids, protos, rng)
        seen.update(map(int, read.bindings))
        if t >= 4_000:
            recent.extend(zip(ids, map(int, read.bindings)))
    old_map = {e: dominant(recent, e) for e in range(base_n)}
    before = active_entities(runtime)
    delay = None
    novel_bindings = []
    old_obs = old_bad = 0
    for i in range(500):
        ids = [novel] + rng.sample(range(base_n), 3)
        read = rstep(runtime, ids, protos, rng)
        nb = int(read.bindings[0]); novel_bindings.append(nb)
        if delay is None and nb not in seen:
            delay = i
        for e, b in zip(ids[1:], map(int, read.bindings[1:])):
            old_obs += 1; old_bad += int(old_map[e] is not None and b != old_map[e])
        seen.update(map(int, read.bindings))
    after_novelty = active_entities(runtime)
    post: list[tuple[int, int]] = []
    for _ in range(4_000):
        ids = rng.sample(range(9), 4)
        read = rstep(runtime, ids, protos, rng)
        post.extend(zip(ids, map(int, read.bindings)))
    final = active_entities(runtime)
    c = Counter(b for t, b in post if t == novel)
    return {
        **benchmark_provenance(PROTOCOL), **case, "repetition": rep, "seed_used": seed,
        "config_sha256": cfg_hash(cfg), "active_before_novelty": before,
        "active_after_novelty": after_novelty, "active_final": final,
        "novelty_new_identity_delay": delay, "novelty_unique_bindings": len(set(novel_bindings)),
        "old_identity_mismatch_during_novelty": old_bad / max(1, old_obs),
        "new_entities_during_novelty": after_novelty - before,
        "new_entities_after_novelty": final - after_novelty,
        "novel_post_dominant_share": c.most_common(1)[0][1] / sum(c.values()) if c else 0.0,
        "post_fragmentation": identity_metrics(post)["fragmentation"],
    }


def run_corruption_burst(case: dict[str, Any], rep: int) -> dict[str, Any]:
    seed = int(case["seed"]) + rep * 1009
    rng = random.Random(seed)
    n, target = 8, 0
    protos = identity_prototypes(n, rng)
    cfg, runtime = runtime_pair()
    recent: list[tuple[int, int]] = []
    for t in range(5_000):
        ids = rng.sample(range(n), 4)
        read = rstep(runtime, ids, protos, rng)
        if t >= 3_000:
            recent.extend(zip(ids, map(int, read.bindings)))
    target_id = dominant(recent, target)
    before = active_entities(runtime)
    burst_bindings = []
    for _ in range(int(case["burst"])):
        read = rstep(runtime, [target] + rng.sample(range(1, n), 3), protos, rng,
                     corrupt={target: (float(case["corrupt_p"]), float(case["corrupt_scale"]))})
        burst_bindings.append(int(read.bindings[0]))
    after = active_entities(runtime)
    recovery = []
    for _ in range(1_000):
        read = rstep(runtime, [target] + rng.sample(range(1, n), 3), protos, rng)
        recovery.append(int(read.bindings[0]))
    final = active_entities(runtime)
    return {
        **benchmark_provenance(PROTOCOL), **case, "repetition": rep, "seed_used": seed,
        "config_sha256": cfg_hash(cfg), "target_binding_before_burst": target_id,
        "burst_unique_target_bindings": len(set(burst_bindings)),
        "burst_original_identity_fraction": sum(b == target_id for b in burst_bindings) / max(1, len(burst_bindings)),
        "new_entities_during_burst": after - before,
        "recovery_original_identity_fraction": sum(b == target_id for b in recovery) / len(recovery),
        "first_recovery_same": recovery[0] == target_id,
        "new_entities_during_recovery": final - after,
        "permanent_excess_entities": final - n,
        "active_before_burst": before, "active_after_burst": after, "active_final": final,
    }


def core_prototypes(k: int, sep: float = 0.90) -> list[list[float]]:
    a = sep * math.sqrt(CORE_DIM / 2.0)
    out = []
    for j in range(k):
        x = [0.0] * CORE_DIM; x[j] = a; out.append(x)
    return out


def run_continual_stabilization(case: dict[str, Any], rep: int) -> dict[str, Any]:
    from cortex import Cortex
    seed = int(case["seed"]) + rep * 1009
    rng = random.Random(seed)
    noise = float(case["noise"])
    protos = core_prototypes(5)
    seed_model = Cortex(dim=CORE_DIM, budget=12)
    cfg = seed_model.config_json()
    model = Cortex(dim=CORE_DIM, budget=12, config_json=cfg)
    counts = {"d": 0, "r": 0, "v": 0, "u": 0}
    prev = dict(counts); windows = []; step = 0

    def feed(label: int):
        nonlocal step, prev
        step += 1
        x = [v + rng.gauss(0.0, noise) for v in protos[label]]
        read = model.step(x, float(label % 2))
        counts["d"] += int(read.discovered); counts["r"] += int(read.reactivated)
        counts["v"] += int(read.revision); counts["u"] += int(read.unresolved)
        if step % 5_000 == 0:
            windows.append({"end": step, "discoveries": counts["d"]-prev["d"],
                            "reactivations": counts["r"]-prev["r"],
                            "revisions": counts["v"]-prev["v"],
                            "unresolved": counts["u"]-prev["u"], "stored": int(read.stored)})
            prev = dict(counts)
        return read

    for t in range(40_000): feed((t // 8) % 4)
    pre_gap = json.loads(model.snapshot_json())
    for t in range(10_000): feed(1 + ((t // 8) % 3))
    d0, r0 = counts["d"], counts["r"]
    first_return = None
    for t in range(2_000):
        read = feed((t // 8) % 4)
        if t == 0:
            first_return = {"discovered": bool(read.discovered), "reactivated": bool(read.reactivated), "stored": int(read.stored)}
    return_d, return_r = counts["d"]-d0, counts["r"]-r0
    d0 = counts["d"]; novelty_delay = None
    for t in range(1_000):
        read = feed(4)
        if novelty_delay is None and read.discovered: novelty_delay = t
    novelty_d = counts["d"] - d0
    after_novelty = json.loads(model.snapshot_json())
    d0 = counts["d"]
    for t in range(10_000): feed((t // 8) % 5)
    final = json.loads(model.snapshot_json())
    if len(final["prototypes"]) > 12: raise RuntimeError("continual budget invariant violated")
    return {
        **benchmark_provenance(PROTOCOL), **case, "repetition": rep, "seed_used": seed,
        "config_sha256": cfg_hash(cfg), "windows": windows,
        "stored_before_gap": len(pre_gap["prototypes"]), "stored_after_novelty": len(after_novelty["prototypes"]),
        "stored_final": len(final["prototypes"]), "first_return": first_return,
        "return_discoveries": return_d, "return_reactivations": return_r,
        "novelty_detection_delay": novelty_delay, "novelty_discoveries": novelty_d,
        "post_novelty_discoveries": counts["d"]-d0,
        "final_discovery_count": int(final["discovery_count"]),
        "final_reactivation_count": int(final["reactivation_count"]),
        "final_unresolved_count": int(final["unresolved_count"]),
        "split_promotions": int(final["split_promotions"]), "merge_promotions": int(final["merge_promotions"]),
        "budget_pressure_count": int(final["budget_pressure_count"]),
    }


def run_case(case: dict[str, Any], rep: int) -> dict[str, Any]:
    return {
        "identity_stabilization": run_identity_stabilization,
        "absence_return": run_absence_return,
        "novelty_recovery": run_novelty_recovery,
        "corruption_burst": run_corruption_burst,
        "continual_stabilization": run_continual_stabilization,
    }[str(case["suite"])](case, rep)


def campaign_cases() -> list[dict[str, Any]]:
    seed = 20260917
    cases = [
        {"suite":"identity_stabilization","case":"identity_clean","noise":0.05,"corrupt_p":0.0,"seed":seed},
        {"suite":"identity_stabilization","case":"identity_sparse_002","noise":0.05,"corrupt_p":0.02,"seed":seed+1},
    ]
    for gap in (100, 1_000, 5_000, 20_000):
        cases.append({"suite":"absence_return","case":f"absence_{gap}","gap":gap,"seed":seed+100})
    cases.append({"suite":"novelty_recovery","case":"localized_novelty","seed":seed+200})
    for burst, p in ((1,.10),(5,.10),(20,.10),(100,.10),(1,.20),(20,.20)):
        cases.append({"suite":"corruption_burst","case":f"burst_{burst}_p_{p:.2f}","burst":burst,
                      "corrupt_p":p,"corrupt_scale":1.25,"seed":seed+300})
    for noise in (.03,.12):
        cases.append({"suite":"continual_stabilization","case":f"continual_noise_{noise:.2f}","noise":noise,"seed":seed+400})
    return cases


def run_campaign(repetitions: int, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    cases = campaign_cases()
    with output.open("w", encoding="utf8") as fh:
        for cid, case in enumerate(cases):
            for rep in range(1, repetitions+1):
                payload = json.dumps({**case, "repetition": rep})
                proc = subprocess.run([sys.executable, __file__, "--case-json", payload], cwd=ROOT,
                                      text=True, capture_output=True, check=False,
                                      env={**os.environ,"PYTHONHASHSEED":"0","OMP_NUM_THREADS":"1",
                                           "OPENBLAS_NUM_THREADS":"1","MKL_NUM_THREADS":"1"})
                if proc.returncode != 0:
                    raise RuntimeError(f"stabilization child failed for {case['case']} rep={rep}:\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")
                result = json.loads(proc.stdout.strip().splitlines()[-1].removeprefix("CORTEX_STABILIZATION="))
                fh.write(json.dumps(result, sort_keys=True)+"\n"); fh.flush()
                print(f"[{cid:02d}/{len(cases):02d}] {case['case']} rep={rep}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--campaign", action="store_true")
    ap.add_argument("--repetitions", type=int, default=3); ap.add_argument("--case-json")
    ap.add_argument("--output", type=Path, default=ROOT/"stabilization-v1.jsonl"); args = ap.parse_args()
    if args.case_json:
        case = json.loads(args.case_json); rep = int(case.pop("repetition", 1))
        print("CORTEX_STABILIZATION="+json.dumps(run_case(case, rep), sort_keys=True)); return
    if args.campaign: run_campaign(args.repetitions, args.output); return
    ap.error("choose --campaign or --case-json")


if __name__ == "__main__": main()
