"""Bounded algebraic-form induction for Cortex research.

The inducer receives:
- anonymous elements;
- a partially observed binary operation table;
- a bounded expression grammar over variables and one binary operator.

It is not given named algebraic laws. It enumerates equation forms from the
grammar, canonicalizes them under variable renaming and equation-side exchange,
assigns fuzzy evidence from disjoint witness partitions, and uses only strongly
supported forms for conservative predictive closure.

This is bounded form discovery, not unrestricted theorem proving.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations, permutations, product
from math import isfinite
from typing import Iterable, Mapping, Sequence

Element = str
Pair = tuple[Element, Element]
Table = dict[Pair, Element]
VARIABLES = ("x", "y", "z")


@dataclass(frozen=True)
class Expr:
    variable: str | None = None
    left: "Expr | None" = None
    right: "Expr | None" = None

    @property
    def is_variable(self) -> bool:
        return self.variable is not None

    @property
    def op_count(self) -> int:
        if self.is_variable:
            return 0
        assert self.left is not None and self.right is not None
        return 1 + self.left.op_count + self.right.op_count

    @property
    def variables(self) -> frozenset[str]:
        if self.is_variable:
            assert self.variable is not None
            return frozenset((self.variable,))
        assert self.left is not None and self.right is not None
        return self.left.variables | self.right.variables

    def render(self) -> str:
        if self.is_variable:
            assert self.variable is not None
            return self.variable
        assert self.left is not None and self.right is not None
        return f"({self.left.render()}*{self.right.render()})"

    def rename(self, mapping: Mapping[str, str]) -> "Expr":
        if self.is_variable:
            assert self.variable is not None
            return Expr(variable=mapping[self.variable])
        assert self.left is not None and self.right is not None
        return Expr(
            left=self.left.rename(mapping),
            right=self.right.rename(mapping),
        )


def var(name: str) -> Expr:
    if name not in VARIABLES:
        raise ValueError(f"unknown variable: {name}")
    return Expr(variable=name)


def op(left: Expr, right: Expr) -> Expr:
    return Expr(left=left, right=right)


@lru_cache(maxsize=None)
def expressions_exact(op_count: int) -> tuple[Expr, ...]:
    if op_count < 0:
        raise ValueError("operator count must be non-negative")
    if op_count == 0:
        return tuple(var(name) for name in VARIABLES)

    out: list[Expr] = []
    for left_ops in range(op_count):
        right_ops = op_count - 1 - left_ops
        for left in expressions_exact(left_ops):
            for right in expressions_exact(right_ops):
                out.append(op(left, right))
    return tuple(out)


def expression_grammar(max_ops_per_expression: int = 2) -> tuple[Expr, ...]:
    if max_ops_per_expression < 1:
        raise ValueError("max_ops_per_expression must be >= 1")
    out: list[Expr] = []
    for count in range(max_ops_per_expression + 1):
        out.extend(expressions_exact(count))
    return tuple(out)


def canonical_equation(left: Expr, right: Expr) -> str:
    """Canonical equation key under variable renaming and side exchange."""
    best: str | None = None
    for perm in permutations(VARIABLES):
        mapping = dict(zip(VARIABLES, perm))
        a = left.rename(mapping).render()
        b = right.rename(mapping).render()
        candidate = min(f"{a}={b}", f"{b}={a}")
        if best is None or candidate < best:
            best = candidate
    assert best is not None
    return best


@dataclass(frozen=True)
class CandidateForm:
    key: str
    left: Expr
    right: Expr
    variables: tuple[str, ...]
    complexity: int


def equation_forms(max_ops_per_expression: int = 2) -> tuple[CandidateForm, ...]:
    expressions = expression_grammar(max_ops_per_expression)
    unique: dict[str, CandidateForm] = {}

    for left, right in combinations(expressions, 2):
        if left.op_count == 0 and right.op_count == 0:
            continue
        key = canonical_equation(left, right)
        if key in unique:
            continue
        names = tuple(sorted(left.variables | right.variables))
        unique[key] = CandidateForm(
            key=key,
            left=left,
            right=right,
            variables=names,
            complexity=left.op_count + right.op_count,
        )

    return tuple(unique[key] for key in sorted(unique))


def evaluate(
    expr: Expr,
    env: Mapping[str, Element],
    table: Mapping[Pair, Element],
) -> Element | None:
    if expr.is_variable:
        assert expr.variable is not None
        return env[expr.variable]

    assert expr.left is not None and expr.right is not None
    left = evaluate(expr.left, env, table)
    right = evaluate(expr.right, env, table)
    if left is None or right is None:
        return None
    return table.get((left, right))


def root_missing_pair(
    expr: Expr,
    env: Mapping[str, Element],
    table: Mapping[Pair, Element],
) -> Pair | None:
    """Return a missing root operation if both child values are known."""
    if expr.is_variable:
        return None
    assert expr.left is not None and expr.right is not None
    left = evaluate(expr.left, env, table)
    right = evaluate(expr.right, env, table)
    if left is None or right is None:
        return None
    pair = (left, right)
    if pair in table:
        return None
    return pair


@dataclass(frozen=True)
class Evidence:
    positive: int
    negative: int
    minimum_witnesses: int

    @property
    def total(self) -> int:
        return self.positive + self.negative

    @property
    def consistency(self) -> float:
        if self.total == 0:
            return 0.0
        return self.positive / self.total

    @property
    def evidence_mass(self) -> float:
        return min(1.0, self.total / max(1, self.minimum_witnesses))

    @property
    def membership(self) -> float:
        return self.consistency * self.evidence_mass


@dataclass(frozen=True)
class FuzzyLaw:
    form: CandidateForm
    discovery: Evidence
    validation: Evidence
    membership: float
    score: float
    active: bool


@dataclass(frozen=True)
class FormInference:
    value: Element | None
    resolved: bool
    provenance: tuple[str, ...]


def _minimum_witnesses(variable_count: int, split: str) -> int:
    if split not in {"discovery", "validation"}:
        raise ValueError(split)
    if variable_count <= 1:
        return 2 if split == "discovery" else 1
    if variable_count == 2:
        return 6 if split == "discovery" else 2
    return 12 if split == "discovery" else 4


def _assignment_partition(
    assignment: Sequence[Element],
    index: Mapping[Element, int],
) -> str:
    """Deterministic 3:1 witness split independent of Python hash randomization."""
    code = sum((position + 3) * (index[value] + 1) for position, value in enumerate(assignment))
    return "validation" if code % 4 == 0 else "discovery"


def _evidence_for_form(
    form: CandidateForm,
    elements: Sequence[Element],
    table: Mapping[Pair, Element],
) -> tuple[Evidence, Evidence]:
    index = {value: i for i, value in enumerate(elements)}
    counts = {
        "discovery": [0, 0],
        "validation": [0, 0],
    }

    for values in product(elements, repeat=len(form.variables)):
        env = dict(zip(form.variables, values))
        left = evaluate(form.left, env, table)
        right = evaluate(form.right, env, table)
        if left is None or right is None:
            continue
        split = _assignment_partition(values, index)
        if left == right:
            counts[split][0] += 1
        else:
            counts[split][1] += 1

    discovery = Evidence(
        positive=counts["discovery"][0],
        negative=counts["discovery"][1],
        minimum_witnesses=_minimum_witnesses(len(form.variables), "discovery"),
    )
    validation = Evidence(
        positive=counts["validation"][0],
        negative=counts["validation"][1],
        minimum_witnesses=_minimum_witnesses(len(form.variables), "validation"),
    )
    return discovery, validation


class CortexFuzzyLawInducer:
    """Discover equation forms using fuzzy evidence and predictive closure."""

    def __init__(
        self,
        elements: Iterable[Element],
        observed: Table,
        *,
        max_ops_per_expression: int = 2,
        membership_threshold: float = 0.98,
        complexity_penalty: float = 0.01,
        max_active_laws: int = 24,
        use_validation: bool = True,
    ) -> None:
        self.elements = tuple(sorted(elements))
        self.observed = dict(observed)
        if not self.elements:
            raise ValueError("elements must not be empty")
        if not 0.0 <= membership_threshold <= 1.0:
            raise ValueError("membership threshold must lie in [0,1]")
        if complexity_penalty < 0.0 or not isfinite(complexity_penalty):
            raise ValueError("complexity penalty must be finite and non-negative")
        if max_active_laws < 1:
            raise ValueError("max_active_laws must be positive")

        self.max_ops_per_expression = max_ops_per_expression
        self.membership_threshold = membership_threshold
        self.complexity_penalty = complexity_penalty
        self.max_active_laws = max_active_laws
        self.use_validation = use_validation

        self.forms = equation_forms(max_ops_per_expression)
        self.laws = self._score_forms()
        eligible = [law for law in self.laws if law.active]
        eligible.sort(key=lambda law: (-law.score, law.form.complexity, law.form.key))
        self.active_laws = tuple(eligible[:max_active_laws])

        self.completed = dict(self.observed)
        self.provenance: dict[Pair, tuple[str, ...]] = {}
        self.conflicts = 0
        self.rounds = 0
        self._closed = False

    def _score_forms(self) -> tuple[FuzzyLaw, ...]:
        out: list[FuzzyLaw] = []
        for form in self.forms:
            discovery, validation = _evidence_for_form(
                form,
                self.elements,
                self.observed,
            )
            membership = discovery.membership
            if self.use_validation:
                membership = min(membership, validation.membership)
            score = membership - self.complexity_penalty * form.complexity
            active = membership >= self.membership_threshold
            out.append(
                FuzzyLaw(
                    form=form,
                    discovery=discovery,
                    validation=validation,
                    membership=membership,
                    score=score,
                    active=active,
                )
            )
        return tuple(out)

    def active_keys(self) -> set[str]:
        return {law.form.key for law in self.active_laws}

    def top_laws(self, limit: int = 10) -> list[dict[str, float | int | str]]:
        ranked = sorted(
            self.laws,
            key=lambda law: (-law.score, law.form.complexity, law.form.key),
        )
        return [
            {
                "key": law.form.key,
                "membership": law.membership,
                "score": law.score,
                "complexity": law.form.complexity,
                "discovery_positive": law.discovery.positive,
                "discovery_negative": law.discovery.negative,
                "validation_positive": law.validation.positive,
                "validation_negative": law.validation.negative,
            }
            for law in ranked[:limit]
        ]

    def close(self) -> None:
        if self._closed:
            return

        max_rounds = len(self.elements) ** 2 + 2
        while True:
            self.rounds += 1
            proposals: dict[Pair, list[tuple[Element, str]]] = {}

            for law in self.active_laws:
                form = law.form
                for values in product(self.elements, repeat=len(form.variables)):
                    env = dict(zip(form.variables, values))
                    left = evaluate(form.left, env, self.completed)
                    right = evaluate(form.right, env, self.completed)

                    if left is not None and right is not None:
                        if left != right:
                            self.conflicts += 1
                        continue

                    if left is None and right is not None:
                        pair = root_missing_pair(form.left, env, self.completed)
                        if pair is not None:
                            proposals.setdefault(pair, []).append(
                                (right, form.key)
                            )
                    elif right is None and left is not None:
                        pair = root_missing_pair(form.right, env, self.completed)
                        if pair is not None:
                            proposals.setdefault(pair, []).append(
                                (left, form.key)
                            )

            changed = False
            for pair, candidates in sorted(proposals.items()):
                values = {value for value, _key in candidates}
                if len(values) != 1:
                    self.conflicts += 1
                    continue
                if pair in self.completed:
                    continue
                value = next(iter(values))
                keys = tuple(sorted({key for candidate, key in candidates if candidate == value}))
                self.completed[pair] = value
                self.provenance[pair] = keys
                changed = True

            if not changed:
                break
            if self.rounds > max_rounds:
                raise RuntimeError("form closure did not converge")

        self._closed = True

    def infer(self, a: Element, b: Element) -> FormInference:
        self.close()
        pair = (a, b)
        value = self.completed.get(pair)
        if value is None:
            return FormInference(None, False, tuple())
        if pair in self.observed:
            return FormInference(value, True, ("observed",))
        return FormInference(
            value,
            True,
            self.provenance.get(pair, ("derived",)),
        )


class DirectFormMemory:
    def __init__(self, observed: Table) -> None:
        self.observed = dict(observed)

    def infer(self, a: Element, b: Element) -> FormInference:
        value = self.observed.get((a, b))
        return FormInference(
            value,
            value is not None,
            ("observed",) if value is not None else tuple(),
        )


def score_heldout(
    model: object,
    table: Table,
    heldout: Iterable[Pair],
) -> dict[str, float | int]:
    total = 0
    resolved = 0
    correct = 0

    infer = getattr(model, "infer")
    for pair in heldout:
        total += 1
        result: FormInference = infer(*pair)
        if result.resolved:
            resolved += 1
            correct += int(result.value == table[pair])

    return {
        "heldout": total,
        "resolved": resolved,
        "correct": correct,
        "coverage": resolved / max(1, total),
        "accuracy": correct / max(1, total),
        "resolved_accuracy": correct / max(1, resolved),
    }
