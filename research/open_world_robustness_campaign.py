#!/usr/bin/env python3
"""Development/pre-registration harness for Open-World Robustness v1.

Phase 1 only validates benchmark construction. It intentionally does not assert
Cortex performance gates; those are frozen after development characterization
and before any official run.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))

from open_world_robustness_world import (
    PROTOCOL,
    STAGES,
    build_suite,
    suite_integrity,
)


def _stage_specific_checks(worlds):
    checks = {}

    stage_a = [w for w in worlds if w.stage == "A"]
    if stage_a:
        checks["A_hidden_reliability"] = all(
            w.oracle["learner_receives_source_reliability"] is False
            and bool(w.oracle["copy_parent"])
            and bool(w.oracle["drift"])
            for w in stage_a
        )

    stage_b = [w for w in worlds if w.stage == "B"]
    if stage_b:
        checks["B_temporal_disorder"] = all(
            w.oracle["missing_events"] > 0
            and w.oracle["delayed_events"] > 0
            and w.oracle["duplicate_events"] > 0
            and w.oracle["regime_schedule"][0][2]
            == w.oracle["regime_schedule"][-1][2]
            for w in stage_b
        )

    stage_c = [w for w in worlds if w.stage == "C"]
    if stage_c:
        checks["C_compound_and_ambiguous"] = all(
            any(len(tuple(v)) > 1 for v in w.oracle["expected_components"].values())
            and bool(w.oracle["ambiguous_queries"])
            for w in stage_c
        )

    stage_d = [w for w in worlds if w.stage == "D"]
    if stage_d:
        checks["D_pressure_and_recurrence"] = all(
            w.oracle["stream_length"] >= 3_000
            and w.oracle["salient_probability"] <= 0.05
            and any(q.kind == "regime-reactivation" for q in w.queries)
            for w in stage_d
        )

    stage_e = [w for w in worlds if w.stage == "E"]
    if stage_e:
        checks["E_composite_hidden_physics"] = all(
            len(w.oracle["contains"]) >= 7
            and w.oracle["regime_schedule"][0][2]
            == w.oracle["regime_schedule"][-1][2]
            and any(not q.answerable for q in w.queries)
            for w in stage_e
        )
        checks["E_heldout_queries_are_unseen"] = all(
            all(
                q.context not in {
                    obs.context
                    for obs in w.observations
                    if obs.event_time <= q.query_time
                }
                for q in w.queries
                if "heldout-context" in q.tags
            )
            for w in stage_e
        )
    return checks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("development", "official"),
        default="development",
    )
    parser.add_argument(
        "--material",
        default="open-world-robustness-v1-development",
        help=(
            "Development label, or for official mode the 40-hex merge commit "
            "SHA of the frozen benchmark/candidate."
        ),
    )
    parser.add_argument("--worlds-per-stage", type=int, default=3)
    parser.add_argument(
        "--stage",
        action="append",
        choices=STAGES,
        dest="stages",
        help="Repeat to select stages; defaults to A-E.",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    stages = tuple(args.stages or STAGES)
    worlds = build_suite(
        material=args.material,
        count=args.worlds_per_stage,
        stages=stages,
        scale=args.mode,
    )
    integrity = suite_integrity(worlds)
    stage_checks = _stage_specific_checks(worlds)
    checks = {**integrity["checks"], **stage_checks}
    payload = {
        "protocol": PROTOCOL,
        "phase": "benchmark-construction",
        "mode": args.mode,
        "material": args.material,
        "worlds": len(worlds),
        "stage_counts": integrity["stage_counts"],
        "checks": checks,
        "passed": all(checks.values()),
        "world_fingerprints": [
            {
                "stage": world.stage,
                "seed": world.seed,
                "public": world.fingerprint(),
                "oracle": world.oracle_fingerprint(),
                "observations": len(world.observations),
                "queries": len(world.queries),
            }
            for world in worlds
        ],
    }

    rendered = json.dumps(payload, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    if not payload["passed"]:
        failed = ", ".join(name for name, value in checks.items() if not value)
        raise SystemExit(f"benchmark-construction checks failed: {failed}")


if __name__ == "__main__":
    main()
