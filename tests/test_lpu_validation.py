from __future__ import annotations

from fractions import Fraction
import random

from cortex import CortexRuntime, LogicalProcessingUnit


def _oracle(edges, source, target):
    frontier = [source]
    seen = {source}
    adjacency = {}
    for left, right in edges:
        adjacency.setdefault(left, set()).add(right)
    while frontier:
        node = frontier.pop()
        if node == target:
            return True
        for nxt in adjacency.get(node, ()):
            if nxt not in seen:
                seen.add(nxt)
                frontier.append(nxt)
    return False


def _valid_witness(edges, source, target, result):
    if not result.value:
        return result.steps[0].result == ()
    path = result.steps[0].result
    edge_set = set(edges)
    return (
        path[0] == source
        and path[-1] == target
        and all((a, b) in edge_set for a, b in zip(path, path[1:]))
    )


def test_exhaustive_boolean_truth_tables():
    lpu = LogicalProcessingUnit()
    for left in (False, True):
        assert lpu.evaluate("NOT", left).value is (not left)
        for right in (False, True):
            assert lpu.evaluate("AND", left, right).value is (left and right)
            assert lpu.evaluate("OR", left, right).value is (left or right)
            assert lpu.evaluate("IMPLIES", left, right).value is (
                (not left) or right
            )


def test_random_exact_fraction_arithmetic_matches_reference():
    rng = random.Random(9917)
    lpu = LogicalProcessingUnit()
    for _ in range(1000):
        a = Fraction(rng.randint(-5000, 5000), rng.randint(1, 67))
        b = Fraction(rng.randint(-5000, 5000), rng.randint(1, 67))
        for opcode, expected in (
            ("ADD", a + b),
            ("SUB", a - b),
            ("MUL", a * b),
        ):
            got = lpu.evaluate(opcode, a, b).value
            normalized = (
                expected.numerator
                if expected.denominator == 1
                else expected
            )
            assert got == normalized
        if b:
            expected = a / b
            got = lpu.evaluate("DIV", a, b).value
            normalized = (
                expected.numerator
                if expected.denominator == 1
                else expected
            )
            assert got == normalized


def test_random_reachability_matches_independent_oracle_and_valid_witness():
    rng = random.Random(221)
    lpu = LogicalProcessingUnit()
    for _ in range(500):
        n = rng.choice((4, 8, 16, 32))
        edges = tuple(
            (a, b)
            for a in range(n)
            for b in range(n)
            if a != b and rng.random() < min(0.35, 3.0 / n)
        )
        source = rng.randrange(n)
        target = rng.randrange(n)
        result = lpu.evaluate("REACHABLE", edges, source, target)
        assert result.value == _oracle(edges, source, target)
        assert _valid_witness(edges, source, target, result)


def test_reachability_is_invariant_to_edge_input_order():
    lpu = LogicalProcessingUnit()
    edges = [
        ("A", "B"),
        ("A", "C"),
        ("B", "D"),
        ("C", "D"),
        ("D", "E"),
    ]
    expected = lpu.evaluate("REACHABLE", tuple(edges), "A", "E")
    rng = random.Random(451)
    for _ in range(100):
        rng.shuffle(edges)
        got = lpu.evaluate("REACHABLE", tuple(edges), "A", "E")
        assert got == expected


def test_lpu_calls_do_not_mutate_cortex_memory():
    runtime = CortexRuntime(feature_dim=4, max_entities=16, budget=8)
    runtime.step(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
        [(0, 1, 1)],
        None,
        [],
        0.5,
    )
    before = runtime.snapshot_json()

    lpu = LogicalProcessingUnit()
    for i in range(100):
        assert lpu.evaluate("ADD", Fraction(i, 3), Fraction(1, 7)).exact
        assert lpu.evaluate(
            "REACHABLE",
            (("A", "B"), ("B", "C")),
            "A",
            "C",
        ).value

    assert runtime.snapshot_json() == before


def test_multi_valued_relation_does_not_trigger_cardinality_assumption():
    lpu = LogicalProcessingUnit()
    capital_x = ("capital", "X")
    capital_y = ("capital", "Y")
    edges = (
        ("Country-A", capital_x),
        ("Country-A", capital_y),
    )

    first = lpu.evaluate("REACHABLE", edges, "Country-A", capital_x)
    second = lpu.evaluate("REACHABLE", edges, "Country-A", capital_y)

    assert first.value
    assert second.value
    assert _valid_witness(edges, "Country-A", capital_x, first)
    assert _valid_witness(edges, "Country-A", capital_y, second)
