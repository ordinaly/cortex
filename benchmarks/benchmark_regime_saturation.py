#!/usr/bin/env python3
"""Force the continual-regime budget to saturation and measure pressure behavior.

Protocol: regime-saturation-v1

This is a benchmark-only synthetic fixture. It intentionally presents a sequence
of mutually distant 12-D articulation vectors so the continual controller must
fill its regime budget and then expose unresolved/budget-pressure states rather
than silently growing beyond the declared bound.
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

from benchmark_native_scaling import percentile
from provenance import benchmark_provenance

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "regime-saturation-v1"
DIM = 12


def regime_vector(index: int) -> list[float]:
    # Binary codewords at amplitude 2.0 keep even one-bit neighbors far outside
    # the configured stay and recurrence tolerances in RMS distance.
    return [2.0 if (index >> bit) & 1 else -2.0 for bit in range(DIM)]


def resolved_config(budget: int) -> str:
    from cortex import Cortex

    seed = Cortex(dim=DIM, budget=budget)
    cfg = json.loads(seed.config_json())
    cfg["budget"] = budget
    cfg["stay_tolerance"] = 0.20
    cfg["recurrence_tolerance"] = 0.35
    cfg["update_tolerance"] = 0.15
    cfg["redundancy_tolerance"] = 0.10
    cfg["novelty_threshold"] = 1.75
    cfg["recurrence_threshold"] = 0.55
    cfg["recompress_interval"] = 1

    # Keep this fixture focused on the regime budget rather than tensor or split
    # machinery. These are benchmark-only settings, not production defaults.
    cfg["initial_burst"] = 0
    cfg["post_ready_burst"] = 0
    cfg["refresh_interval"] = 1_000_000
    cfg["refresh_burst"] = 1
    cfg["tensor_min_obs"] = 1_000_000
    cfg["max_refinement_depth"] = 0
    cfg["merge_interval"] = 1_000_000
    return json.dumps(cfg, sort_keys=True)


def run_case(case: dict, repetition: int) -> dict:
    from cortex import Cortex

    budget = int(case["budget"])
    extra_regimes = int(case.get("extra_regimes", 4))
    repeats = int(case.get("repeats_per_regime", 6))
    attempted_regimes = budget + extra_regimes
    cfg = resolved_config(budget)
    cortex = Cortex(dim=DIM, budget=budget, config_json=cfg)

    latencies_us: list[float] = []
    first_pressure_step = None
    comparison_count = 0
    step_index = 0
    last = None
    for regime in range(attempted_regimes):
        x = regime_vector(regime)
        y = float(regime % 2)
        for _ in range(repeats):
            step_index += 1
            t0 = time.perf_counter_ns()
            last = cortex.step(x, y)
            t1 = time.perf_counter_ns()
            latencies_us.append((t1 - t0) / 1000.0)
            comparison_count += int(last.comparisons)
            if last.budget_pressure and first_pressure_step is None:
                first_pressure_step = step_index

    snap = json.loads(cortex.snapshot_json())
    stored = len(snap["prototypes"])
    pressure = int(snap["budget_pressure_count"])
    discoveries = int(snap["discovery_count"])
    unresolved = int(snap["unresolved_count"])
    expected_first_pressure = budget * repeats + 1

    if stored > budget:
        raise RuntimeError(f"budget invariant violated: stored={stored}, budget={budget}")
    if stored < budget:
        raise RuntimeError(
            f"saturation fixture did not fill budget: stored={stored}, budget={budget}"
        )
    if pressure <= 0 or first_pressure_step is None:
        raise RuntimeError(
            f"saturation fixture did not expose budget pressure for budget={budget}"
        )
    if first_pressure_step != expected_first_pressure:
        raise RuntimeError(
            "unexpected budget-pressure onset: "
            f"got={first_pressure_step}, expected={expected_first_pressure}, budget={budget}"
        )

    return {
        **benchmark_provenance(PROTOCOL),
        **case,
        "repetition": repetition,
        "attempted_regimes": attempted_regimes,
        "frames": len(latencies_us),
        "stored_regimes": stored,
        "discovery_count": discoveries,
        "budget_pressure_count": pressure,
        "unresolved_count": unresolved,
        "recompression_count": int(snap["recompression_count"]),
        "comparison_count": comparison_count,
        "first_pressure_step": first_pressure_step,
        "expected_first_pressure_step": expected_first_pressure,
        "mean_us": statistics.fmean(latencies_us),
        "p50_us": percentile(latencies_us, 0.50),
        "p95_us": percentile(latencies_us, 0.95),
        "p99_us": percentile(latencies_us, 0.99),
        "fingerprint": {
            "prediction": float(last.prediction),
            "stored": int(last.stored),
            "budget_pressure": bool(last.budget_pressure),
        },
    }


def campaign_cases() -> list[dict]:
    return [
        {
            "case": f"budget_{budget}",
            "budget": budget,
            "extra_regimes": 4,
            "repeats_per_regime": 6,
        }
        for budget in (2, 4, 8, 16)
    ]


def run_campaign(repetitions: int, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf8") as fh:
        for case_id, case in enumerate(campaign_cases()):
            for rep in range(1, repetitions + 1):
                payload = json.dumps({**case, "case_id": case_id, "repetition": rep})
                proc = subprocess.run(
                    [sys.executable, __file__, "--case-json", payload],
                    cwd=ROOT,
                    check=False,
                    text=True,
                    capture_output=True,
                    env={**os.environ, "PYTHONHASHSEED": "0"},
                )
                if proc.returncode != 0:
                    raise RuntimeError(
                        f"regime-saturation child failed for {case['case']} rep={rep}:\n"
                        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
                    )
                line = proc.stdout.strip().splitlines()[-1]
                result = json.loads(line.removeprefix("CORTEX_SATURATION="))
                fh.write(json.dumps(result, sort_keys=True) + "\n")
                fh.flush()
                print(
                    f"[{case_id:02d}] budget={case['budget']} rep={rep} "
                    f"stored={result['stored_regimes']} pressure={result['budget_pressure_count']} "
                    f"first_pressure={result['first_pressure_step']}",
                    flush=True,
                )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", action="store_true")
    ap.add_argument("--repetitions", type=int, default=3)
    ap.add_argument(
        "--output", type=Path, default=ROOT / "regime-saturation-results.jsonl"
    )
    ap.add_argument("--case-json")
    args = ap.parse_args()

    if args.case_json:
        case = json.loads(args.case_json)
        repetition = int(case.pop("repetition", 1))
        result = run_case(case, repetition)
        print("CORTEX_SATURATION=" + json.dumps(result, sort_keys=True))
        return
    if args.campaign:
        run_campaign(args.repetitions, args.output)
        return
    ap.error("choose --campaign or --case-json")


if __name__ == "__main__":
    main()
