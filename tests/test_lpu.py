from __future__ import annotations

from fractions import Fraction

import pytest

from cortex.lpu import LogicalProcessingUnit


def test_boolean_and_implication_are_exact():
    lpu = LogicalProcessingUnit()
    assert lpu.evaluate("AND", True, False).value is False
    assert lpu.evaluate("OR", False, True).value is True
    assert lpu.evaluate("NOT", False).value is True
    assert lpu.evaluate("IMPLIES", True, False).value is False


def test_arithmetic_uses_exact_rationals():
    lpu = LogicalProcessingUnit()
    assert lpu.evaluate("DIV", 1, 3).value == Fraction(1, 3)
    assert lpu.evaluate("ADD", Fraction(1, 3), Fraction(2, 3)).value == 1
    assert lpu.evaluate("MUL", 0.1, 10).value == 1


def test_comparisons_do_not_coerce_boolean_to_number():
    lpu = LogicalProcessingUnit()
    with pytest.raises(TypeError):
        lpu.evaluate("LT", True, 2)


def test_division_by_zero_is_rejected():
    with pytest.raises(ZeroDivisionError):
        LogicalProcessingUnit.evaluate("DIV", 1, 0)


def test_reachability_returns_a_deterministic_witness_path():
    result = LogicalProcessingUnit.evaluate(
        "REACHABLE",
        (("A", "B"), ("B", "D"), ("A", "C"), ("C", "D")),
        "A",
        "D",
    )
    assert result.value is True
    assert result.steps[0].result == ("A", "B", "D")


def test_reachability_abstains_when_no_path_exists():
    result = LogicalProcessingUnit.evaluate(
        "REACHABLE",
        (("A", "B"),),
        "B",
        "A",
    )
    assert result.value is False
    assert result.steps[0].result == ()


def test_unsupported_opcode_is_rejected():
    with pytest.raises(ValueError):
        LogicalProcessingUnit.evaluate("ASSERT_REALITY", True)
