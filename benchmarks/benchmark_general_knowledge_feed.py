#!/usr/bin/env python3
"""General Knowledge Feed v0.1 smoke campaign."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import time

from cortex import CortexRuntime
from cortex.knowledge import (
    KnowledgeGraphView,
    KnowledgeStatement,
    feed_knowledge,
    make_knowledge_runtime,
)


PROTOCOL = "general-knowledge-feed-v0.1"
SEED = 2401


def fixture() -> tuple[KnowledgeStatement, ...]:
    rows = [
        ("France", "capital", "Paris"),
        ("France", "continent", "Europe"),
        ("Germany", "capital", "Berlin"),
        ("Germany", "continent", "Europe"),
        ("Japan", "capital", "Tokyo"),
        ("Japan", "continent", "Asia"),
        ("Brazil", "capital", "Brasilia"),
        ("Brazil", "continent", "South America"),
        ("Canada", "capital", "Ottawa"),
        ("Canada", "continent", "North America"),
        ("Canada", "official language", "English"),
        ("Canada", "official language", "French"),
        ("Earth", "orbits", "Sun"),
        ("Moon", "orbits", "Earth"),
        ("Mars", "orbits", "Sun"),
        ("Jupiter", "orbits", "Sun"),
        ("Venus", "orbits", "Sun"),
        ("Water", "chemical formula", "H2O"),
        ("Gold", "chemical symbol", "Au"),
        ("Silver", "chemical symbol", "Ag"),
        ("Oxygen", "chemical symbol", "O"),
        ("Hydrogen", "chemical symbol", "H"),
    ]
    return tuple(
        KnowledgeStatement(
            statement_id=f"gk-{index:03d}",
            subject=subject,
            property=property_name,
            value=value,
            provenance=PROTOCOL,
        )
        for index, (subject, property_name, value) in enumerate(rows)
    )


EXPECTED = {
    ("France", "capital"): ("Paris",),
    ("France", "continent"): ("Europe",),
    ("Germany", "capital"): ("Berlin",),
    ("Japan", "capital"): ("Tokyo",),
    ("Brazil", "capital"): ("Brasilia",),
    ("Canada", "capital"): ("Ottawa",),
    ("Canada", "official language"): ("English", "French"),
    ("Earth", "orbits"): ("Sun",),
    ("Moon", "orbits"): ("Earth",),
    ("Mars", "orbits"): ("Sun",),
    ("Water", "chemical formula"): ("H2O",),
    ("Gold", "chemical symbol"): ("Au",),
}


def run_once(statements):
    runtime = make_knowledge_runtime(
        CortexRuntime,
        feature_dim=64,
        max_entities=1024,
        budget=64,
    )
    start = time.perf_counter()
    report = feed_knowledge(runtime, statements)
    elapsed = time.perf_counter() - start
    view = KnowledgeGraphView(report.snapshot_json, report.symbol_bindings)

    results = {
        f"{subject}|{property_name}": list(view.objects(subject, property_name))
        for subject, property_name in EXPECTED
    }
    exact = sum(
        tuple(results[f"{subject}|{property_name}"]) == expected
        for (subject, property_name), expected in EXPECTED.items()
    )
    false_query = view.objects("France", "orbits")

    snapshot = json.loads(report.snapshot_json)
    active_relations = sum(
        int(entry["evidence"]["state"]) == 1
        for entry in snapshot["graph"]["relations"]
    )
    return {
        "statements": report.statements,
        "symbols": report.symbols,
        "cortex_entities": report.cortex_entities,
        "binding_conflicts": report.binding_conflicts,
        "active_relations": active_relations,
        "retrieval_exact": exact,
        "retrieval_total": len(EXPECTED),
        "retrieval_accuracy": exact / len(EXPECTED),
        "false_query_empty": false_query == (),
        "multi_value_preserved": (
            view.objects("Canada", "official language")
            == ("English", "French")
        ),
        "results": results,
        "elapsed_s": elapsed,
    }


def campaign():
    statements = fixture()
    baseline = run_once(statements)

    shuffled = list(statements)
    random.Random(SEED).shuffle(shuffled)
    reordered = run_once(tuple(shuffled))

    query_invariant = baseline["results"] == reordered["results"]
    gates = {
        "zero_binding_conflicts": baseline["binding_conflicts"] == 0
        and reordered["binding_conflicts"] == 0,
        "exact_retrieval": baseline["retrieval_accuracy"] == 1.0,
        "negative_control": baseline["false_query_empty"],
        "multi_value_preserved": baseline["multi_value_preserved"],
        "order_invariant_queries": query_invariant,
        "identity_is_injective": (
            baseline["symbols"] == baseline["cortex_entities"]
            and reordered["symbols"] == reordered["cortex_entities"]
        ),
    }

    return {
        "protocol": PROTOCOL,
        "seed": SEED,
        "baseline": baseline,
        "reordered": reordered,
        "gates": gates,
        "passed": all(gates.values()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = campaign()
    payload = json.dumps(report, indent=2, sort_keys=True)
    print(payload)
    if args.output:
        args.output.write_text(payload + "\n", encoding="utf-8")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
