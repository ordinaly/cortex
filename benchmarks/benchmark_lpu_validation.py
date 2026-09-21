#!/usr/bin/env python3
"""LPU Validation v0.1 campaign.

This campaign validates the Logical Processing Unit as a bounded stateless
coprocessor. Absolute timings are characterization only; correctness and
isolation are the hard gates.
"""
from __future__ import annotations

import argparse
from collections import deque
from fractions import Fraction
import json
import random
import statistics
import time
from pathlib import Path
from typing import Any

from cortex import CortexRuntime, LogicalProcessingUnit

PROTOCOL = "lpu-validation-v0.1"
SEED = 1701


def oracle_reachable(edges, source, target):
    adjacency = {}
    for left, right in edges:
        adjacency.setdefault(left, set()).add(right)
        adjacency.setdefault(right, set())
    queue = deque([source])
    seen = {source}
    while queue:
        node = queue.popleft()
        if node == target:
            return True
        for nxt in adjacency.get(node, ()):
            if nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    return False


def witness_valid(edges, source, target, result):
    if not result.value:
        return result.steps[0].result == ()
    path = result.steps[0].result
    edge_set = set(edges)
    return (
        bool(path)
        and path[0] == source
        and path[-1] == target
        and all((a, b) in edge_set for a, b in zip(path, path[1:]))
    )


def boolean_campaign():
    lpu = LogicalProcessingUnit()
    cases = 0
    mismatches = 0
    for left in (False, True):
        if lpu.evaluate("NOT", left).value is not (not left):
            mismatches += 1
        cases += 1
        for right in (False, True):
            expected = {
                "AND": left and right,
                "OR": left or right,
                "IMPLIES": (not left) or right,
                "EQ": left == right,
                "NEQ": left != right,
            }
            for opcode, value in expected.items():
                if lpu.evaluate(opcode, left, right).value != value:
                    mismatches += 1
                cases += 1
    return cases, mismatches


def arithmetic_campaign(rng, samples=5000):
    lpu = LogicalProcessingUnit()
    cases = 0
    mismatches = 0
    for _ in range(samples):
        a = Fraction(rng.randint(-10000, 10000), rng.randint(1, 97))
        b = Fraction(rng.randint(-10000, 10000), rng.randint(1, 97))
        expected = {
            "ADD": a + b,
            "SUB": a - b,
            "MUL": a * b,
        }
        if b:
            expected["DIV"] = a / b
        for opcode, value in expected.items():
            got = lpu.evaluate(opcode, a, b).value
            normalized = value.numerator if value.denominator == 1 else value
            if got != normalized:
                mismatches += 1
            cases += 1
        for opcode, value in {
            "LT": a < b,
            "LTE": a <= b,
            "GT": a > b,
            "GTE": a >= b,
            "EQ": a == b,
            "NEQ": a != b,
        }.items():
            if lpu.evaluate(opcode, a, b).value != value:
                mismatches += 1
            cases += 1
    return cases, mismatches


def graph_campaign(rng, samples=2000):
    lpu = LogicalProcessingUnit()
    mismatches = 0
    witness_failures = 0
    determinism_failures = 0
    cases = 0
    size_histogram = {}

    sizes = (4, 8, 16, 32, 64, 128)
    for _ in range(samples):
        n = rng.choice(sizes)
        p = min(0.30, 3.0 / n)
        edges = []
        for left in range(n):
            for right in range(n):
                if left != right and rng.random() < p:
                    edges.append((left, right))
        source = rng.randrange(n)
        target = rng.randrange(n)
        expected = oracle_reachable(edges, source, target)
        result = lpu.evaluate("REACHABLE", tuple(edges), source, target)
        if result.value != expected:
            mismatches += 1
        if not witness_valid(edges, source, target, result):
            witness_failures += 1

        shuffled = list(edges)
        rng.shuffle(shuffled)
        repeat = lpu.evaluate("REACHABLE", tuple(shuffled), source, target)
        if (
            repeat.value != result.value
            or repeat.steps[0].result != result.steps[0].result
        ):
            determinism_failures += 1

        cases += 1
        size_histogram[str(n)] = size_histogram.get(str(n), 0) + 1

    return {
        "cases": cases,
        "oracle_mismatches": mismatches,
        "witness_failures": witness_failures,
        "edge_order_determinism_failures": determinism_failures,
        "size_histogram": size_histogram,
    }


def purity_campaign():
    lpu = LogicalProcessingUnit()
    runtime = CortexRuntime(feature_dim=4, max_entities=16, budget=8)
    runtime.step(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
        [(0, 1, 1)],
        None,
        [],
        0.5,
    )
    before = runtime.snapshot_json()

    first = lpu.evaluate(
        "REACHABLE",
        (("A", "B"), ("B", "C")),
        "A",
        "C",
    )
    for index in range(1000):
        lpu.evaluate("ADD", Fraction(index, 7), Fraction(index + 1, 11))
        lpu.evaluate("IMPLIES", bool(index & 1), bool(index & 2))
        lpu.evaluate(
            "REACHABLE",
            (("A", "B"), ("B", "C")),
            "A",
            "C",
        )
    second = lpu.evaluate(
        "REACHABLE",
        (("A", "B"), ("B", "C")),
        "A",
        "C",
    )
    after = runtime.snapshot_json()

    return {
        "cortex_snapshot_unchanged": before == after,
        "repeat_result_equal": first == second,
        "calls_between_repeats": 3000,
    }


def ontology_plasticity_campaign():
    lpu = LogicalProcessingUnit()

    capital_x = ("capital", "X")
    capital_y = ("capital", "Y")
    edges = (
        ("Country-A", capital_x),
        ("Country-A", capital_y),
    )
    x = lpu.evaluate("REACHABLE", edges, "Country-A", capital_x)
    y = lpu.evaluate("REACHABLE", edges, "Country-A", capital_y)

    # The exact unit sees two independent supplied edges. It has no predicate
    # cardinality rule and therefore has no basis to reject either.
    return {
        "first_relation_preserved": x.value,
        "second_relation_preserved": y.value,
        "both_witnesses_valid": (
            witness_valid(edges, "Country-A", capital_x, x)
            and witness_valid(edges, "Country-A", capital_y, y)
        ),
    }


def timing_campaign(rng):
    lpu = LogicalProcessingUnit()
    rows = []
    for n in (8, 32, 128, 512):
        edges = [(i, i + 1) for i in range(n - 1)]
        # Add deterministic sparse shortcuts without making the benchmark rely
        # on random graph connectivity.
        for _ in range(max(1, n // 4)):
            left = rng.randrange(n)
            right = rng.randrange(left, n)
            if left != right:
                edges.append((left, right))

        samples = []
        repeats = 500 if n <= 128 else 200
        for _ in range(repeats):
            start = time.perf_counter_ns()
            result = lpu.evaluate("REACHABLE", tuple(edges), 0, n - 1)
            elapsed = time.perf_counter_ns() - start
            if not result.value:
                raise AssertionError("timing fixture must remain reachable")
            samples.append(elapsed / 1000.0)
        rows.append(
            {
                "nodes": n,
                "edges": len(edges),
                "repeats": repeats,
                "median_us": statistics.median(samples),
                "p95_us": sorted(samples)[int(0.95 * (len(samples) - 1))],
            }
        )
    return rows


def run_campaign():
    rng = random.Random(SEED)
    bool_cases, bool_mismatches = boolean_campaign()
    arithmetic_cases, arithmetic_mismatches = arithmetic_campaign(rng)
    graph = graph_campaign(rng)
    purity = purity_campaign()
    plasticity = ontology_plasticity_campaign()
    timing = timing_campaign(rng)

    gates = {
        "boolean_exact": bool_mismatches == 0,
        "arithmetic_exact": arithmetic_mismatches == 0,
        "graph_oracle_exact": graph["oracle_mismatches"] == 0,
        "witness_valid": graph["witness_failures"] == 0,
        "deterministic": graph["edge_order_determinism_failures"] == 0
        and purity["repeat_result_equal"],
        "cortex_isolation": purity["cortex_snapshot_unchanged"],
        "ontology_plasticity": all(plasticity.values()),
    }

    return {
        "protocol": PROTOCOL,
        "seed": SEED,
        "boolean": {
            "cases": bool_cases,
            "mismatches": bool_mismatches,
        },
        "arithmetic_and_comparison": {
            "cases": arithmetic_cases,
            "mismatches": arithmetic_mismatches,
        },
        "graph": graph,
        "purity": purity,
        "ontology_plasticity": plasticity,
        "timing_characterization": timing,
        "gates": gates,
        "passed": all(gates.values()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    report = run_campaign()
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.output is not None:
        args.output.write_text(text + "\n", encoding="utf-8")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
