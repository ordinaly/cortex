"""Stateless exact logical processing unit for Cortex.

The LPU is deliberately not a world model. It performs small exact operations
over values and temporary articulated structures supplied by Cortex, then
returns a deterministic result and trace. It owns no persistent memory,
ontology, predicates, or learned state.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from fractions import Fraction
import math
from typing import Any, Hashable, Iterable


@dataclass(frozen=True, slots=True)
class LPUStep:
    opcode: str
    operands: tuple[Any, ...]
    result: Any


@dataclass(frozen=True, slots=True)
class LPUResult:
    value: Any
    steps: tuple[LPUStep, ...]
    exact: bool = True


class LogicalProcessingUnit:
    """Tiny stateless exact-computation coprocessor."""

    OPCODES = frozenset(
        {
            "EQ",
            "NEQ",
            "NOT",
            "AND",
            "OR",
            "IMPLIES",
            "LT",
            "LTE",
            "GT",
            "GTE",
            "ADD",
            "SUB",
            "MUL",
            "DIV",
            "REACHABLE",
        }
    )

    @staticmethod
    def _bool(value: Any) -> bool:
        if type(value) is not bool:
            raise TypeError("boolean opcode requires bool operands")
        return value

    @staticmethod
    def _number(value: Any) -> Fraction:
        if isinstance(value, bool):
            raise TypeError("numeric opcode does not accept bool")
        if isinstance(value, Fraction):
            return value
        if isinstance(value, int):
            return Fraction(value)
        if isinstance(value, float):
            if not math.isfinite(value):
                raise ValueError("numeric operand must be finite")
            return Fraction(str(value))
        raise TypeError("numeric opcode requires int, float, or Fraction")

    @staticmethod
    def _normalize_number(value: Fraction) -> int | Fraction:
        return value.numerator if value.denominator == 1 else value

    @staticmethod
    def _node(value: Any) -> Hashable:
        try:
            hash(value)
        except TypeError as exc:
            raise TypeError("graph nodes must be hashable") from exc
        return value

    @classmethod
    def evaluate(cls, opcode: str, *operands: Any) -> LPUResult:
        op = str(opcode).upper()
        if op not in cls.OPCODES:
            raise ValueError(f"unsupported LPU opcode: {opcode}")

        if op == "EQ":
            cls._arity(op, operands, 2)
            value = operands[0] == operands[1]
        elif op == "NEQ":
            cls._arity(op, operands, 2)
            value = operands[0] != operands[1]
        elif op == "NOT":
            cls._arity(op, operands, 1)
            value = not cls._bool(operands[0])
        elif op in {"AND", "OR", "IMPLIES"}:
            cls._arity(op, operands, 2)
            left = cls._bool(operands[0])
            right = cls._bool(operands[1])
            if op == "AND":
                value = left and right
            elif op == "OR":
                value = left or right
            else:
                value = (not left) or right
        elif op in {"LT", "LTE", "GT", "GTE"}:
            cls._arity(op, operands, 2)
            left = cls._number(operands[0])
            right = cls._number(operands[1])
            value = {
                "LT": left < right,
                "LTE": left <= right,
                "GT": left > right,
                "GTE": left >= right,
            }[op]
        elif op in {"ADD", "SUB", "MUL", "DIV"}:
            cls._arity(op, operands, 2)
            left = cls._number(operands[0])
            right = cls._number(operands[1])
            if op == "ADD":
                result = left + right
            elif op == "SUB":
                result = left - right
            elif op == "MUL":
                result = left * right
            else:
                if right == 0:
                    raise ZeroDivisionError("LPU DIV by zero")
                result = left / right
            value = cls._normalize_number(result)
        else:
            cls._arity(op, operands, 3)
            value, path = cls._reachable(
                operands[0],
                operands[1],
                operands[2],
            )
            return LPUResult(
                value=value,
                steps=(
                    LPUStep(
                        opcode=op,
                        operands=(operands[1], operands[2]),
                        result=path,
                    ),
                ),
            )

        return LPUResult(
            value=value,
            steps=(
                LPUStep(
                    opcode=op,
                    operands=tuple(operands),
                    result=value,
                ),
            ),
        )

    @staticmethod
    def _arity(opcode: str, operands: tuple[Any, ...], expected: int) -> None:
        if len(operands) != expected:
            raise ValueError(
                f"{opcode} expects {expected} operands, got {len(operands)}"
            )

    @classmethod
    def _reachable(
        cls,
        edges: Iterable[tuple[Any, Any]],
        source: Any,
        target: Any,
    ) -> tuple[bool, tuple[Hashable, ...]]:
        start = cls._node(source)
        goal = cls._node(target)
        adjacency: dict[Hashable, set[Hashable]] = {}

        for edge in edges:
            try:
                left, right = edge
            except (TypeError, ValueError) as exc:
                raise ValueError("REACHABLE edges must be (source, target) pairs") from exc
            a = cls._node(left)
            b = cls._node(right)
            adjacency.setdefault(a, set()).add(b)
            adjacency.setdefault(b, set())

        queue: deque[Hashable] = deque([start])
        parent: dict[Hashable, Hashable | None] = {start: None}
        while queue:
            node = queue.popleft()
            if node == goal:
                path: list[Hashable] = []
                cursor: Hashable | None = node
                while cursor is not None:
                    path.append(cursor)
                    cursor = parent[cursor]
                path.reverse()
                return True, tuple(path)
            for nxt in sorted(adjacency.get(node, ()), key=repr):
                if nxt not in parent:
                    parent[nxt] = node
                    queue.append(nxt)

        return False, tuple()
