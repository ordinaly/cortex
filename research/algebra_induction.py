"""Finite-algebra law induction for Cortex research.

The experiment is intentionally structured and symbol-blind. Cortex receives a
partial binary operation table over anonymous labels. It is not told which
algebra family generated the table.

The inducer scores a small preregistered candidate-law library:
- associativity;
- commutativity;
- existence of a unique two-sided identity.

Only sufficiently supported laws are activated. Activated laws may then derive
previously unseen products by closure. Contradictory evidence prevents a law
from being used.

This is *candidate-law induction*, not unrestricted theorem discovery.
"""
from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Hashable, Iterable


Element = str
Pair = tuple[Element, Element]
Table = dict[Pair, Element]


@dataclass(frozen=True)
class LawEvidence:
    name: str
    positive: int
    negative: int
    active: bool

    @property
    def total(self) -> int:
        return self.positive + self.negative

    @property
    def confidence(self) -> float:
        if self.total == 0:
            return 0.0
        return self.positive / self.total


@dataclass(frozen=True)
class AlgebraInference:
    value: Element | None
    resolved: bool
    provenance: str | None


@dataclass(frozen=True)
class HiddenAlgebra:
    name: str
    elements: tuple[Hashable, ...]
    table: dict[tuple[Hashable, Hashable], Hashable]
    associative: bool
    commutative: bool
    identity: Hashable | None


def cyclic_group(order: int) -> HiddenAlgebra:
    if order < 2:
        raise ValueError("order must be >= 2")
    elements = tuple(range(order))
    table = {
        (a, b): (a + b) % order
        for a in elements
        for b in elements
    }
    return HiddenAlgebra(
        name=f"cyclic-{order}",
        elements=elements,
        table=table,
        associative=True,
        commutative=True,
        identity=0,
    )


def dihedral_group(rotations: int) -> HiddenAlgebra:
    if rotations < 3:
        raise ValueError("rotations must be >= 3")
    elements = tuple(
        (rotation, reflection)
        for reflection in (0, 1)
        for rotation in range(rotations)
    )
    table = {}
    for a in elements:
        ar, af = a
        for b in elements:
            br, bf = b
            table[(a, b)] = (
                (ar + ((-1) ** af) * br) % rotations,
                af ^ bf,
            )
    return HiddenAlgebra(
        name=f"dihedral-{rotations}",
        elements=elements,
        table=table,
        associative=True,
        commutative=False,
        identity=(0, 0),
    )


def subtraction_magma(order: int) -> HiddenAlgebra:
    """Structured non-associative control: a*b = a-b mod n."""
    if order < 3:
        raise ValueError("order must be >= 3")
    elements = tuple(range(order))
    table = {
        (a, b): (a - b) % order
        for a in elements
        for b in elements
    }
    return HiddenAlgebra(
        name=f"subtraction-{order}",
        elements=elements,
        table=table,
        associative=False,
        commutative=False,
        identity=None,
    )


def anonymize(
    algebra: HiddenAlgebra,
    *,
    seed: int,
) -> tuple[tuple[Element, ...], Table, Element | None]:
    """Randomly relabel every element so numeric/tuple semantics are unavailable."""
    rng = Random(seed)
    aliases = [f"E{i}" for i in range(len(algebra.elements))]
    rng.shuffle(aliases)
    alias = dict(zip(algebra.elements, aliases))
    elements = tuple(sorted(alias.values()))
    table = {
        (alias[a], alias[b]): alias[result]
        for (a, b), result in algebra.table.items()
    }
    identity = alias[algebra.identity] if algebra.identity is not None else None
    return elements, table, identity


def partial_observation(
    table: Table,
    *,
    seed: int,
    holdout_fraction: float = 0.40,
) -> tuple[Table, set[Pair]]:
    if not 0.0 < holdout_fraction < 1.0:
        raise ValueError("holdout_fraction must lie in (0,1)")
    rng = Random(seed)
    pairs = list(table)
    rng.shuffle(pairs)
    held_count = round(len(pairs) * holdout_fraction)
    held = set(pairs[:held_count])
    observed = {
        pair: value
        for pair, value in table.items()
        if pair not in held
    }
    return observed, held


class CortexAlgebraInducer:
    """Evidence-gated candidate-law induction plus conservative closure."""

    def __init__(
        self,
        elements: Iterable[Element],
        observed: Table,
        *,
        minimum_associativity_witnesses: int = 20,
        minimum_commutativity_witnesses: int = 5,
        minimum_identity_witnesses: int = 6,
        confidence_threshold: float = 0.98,
    ) -> None:
        self.elements = tuple(elements)
        self.observed = dict(observed)
        self.minimum_associativity_witnesses = minimum_associativity_witnesses
        self.minimum_commutativity_witnesses = minimum_commutativity_witnesses
        self.minimum_identity_witnesses = minimum_identity_witnesses
        self.confidence_threshold = confidence_threshold

        self.associativity = self._associativity_evidence()
        self.commutativity = self._commutativity_evidence()
        self.identity, self.identity_evidence = self._identity_evidence()

        self.completed = dict(self.observed)
        self.provenance: dict[Pair, str] = {}
        self.conflicts = 0
        self.rounds = 0
        self._closed = False

    def _is_active(self, positive: int, negative: int, minimum: int) -> bool:
        total = positive + negative
        if total < minimum:
            return False
        return positive / total >= self.confidence_threshold

    def _associativity_evidence(self) -> LawEvidence:
        positive = 0
        negative = 0
        for a in self.elements:
            for b in self.elements:
                ab = self.observed.get((a, b))
                if ab is None:
                    continue
                for c in self.elements:
                    bc = self.observed.get((b, c))
                    if bc is None:
                        continue
                    left = self.observed.get((ab, c))
                    right = self.observed.get((a, bc))
                    if left is None or right is None:
                        continue
                    if left == right:
                        positive += 1
                    else:
                        negative += 1
        return LawEvidence(
            name="associativity",
            positive=positive,
            negative=negative,
            active=self._is_active(
                positive,
                negative,
                self.minimum_associativity_witnesses,
            ),
        )

    def _commutativity_evidence(self) -> LawEvidence:
        positive = 0
        negative = 0
        for i, a in enumerate(self.elements):
            for b in self.elements[i + 1 :]:
                ab = self.observed.get((a, b))
                ba = self.observed.get((b, a))
                if ab is None or ba is None:
                    continue
                if ab == ba:
                    positive += 1
                else:
                    negative += 1
        return LawEvidence(
            name="commutativity",
            positive=positive,
            negative=negative,
            active=self._is_active(
                positive,
                negative,
                self.minimum_commutativity_witnesses,
            ),
        )

    def _identity_evidence(self) -> tuple[Element | None, LawEvidence]:
        candidates: list[tuple[Element, int, int]] = []
        for candidate in self.elements:
            positive = 0
            negative = 0
            for a in self.elements:
                for pair in ((candidate, a), (a, candidate)):
                    value = self.observed.get(pair)
                    if value is None:
                        continue
                    if value == a:
                        positive += 1
                    else:
                        negative += 1
            if self._is_active(
                positive,
                negative,
                self.minimum_identity_witnesses,
            ):
                candidates.append((candidate, positive, negative))

        if len(candidates) != 1:
            total_positive = sum(item[1] for item in candidates)
            total_negative = sum(item[2] for item in candidates)
            return None, LawEvidence(
                name="two-sided-identity",
                positive=total_positive,
                negative=total_negative,
                active=False,
            )

        identity, positive, negative = candidates[0]
        return identity, LawEvidence(
            name="two-sided-identity",
            positive=positive,
            negative=negative,
            active=True,
        )

    def law_summary(self) -> dict[str, dict[str, float | int | bool | str | None]]:
        return {
            "associativity": {
                "positive": self.associativity.positive,
                "negative": self.associativity.negative,
                "confidence": self.associativity.confidence,
                "active": self.associativity.active,
            },
            "commutativity": {
                "positive": self.commutativity.positive,
                "negative": self.commutativity.negative,
                "confidence": self.commutativity.confidence,
                "active": self.commutativity.active,
            },
            "identity": {
                "positive": self.identity_evidence.positive,
                "negative": self.identity_evidence.negative,
                "confidence": self.identity_evidence.confidence,
                "active": self.identity_evidence.active,
                "element": self.identity,
            },
        }

    def _add(self, pair: Pair, value: Element, provenance: str) -> bool:
        existing = self.completed.get(pair)
        if existing is not None:
            if existing != value:
                self.conflicts += 1
            return False
        self.completed[pair] = value
        self.provenance[pair] = provenance
        return True

    def close(self) -> None:
        if self._closed:
            return

        while True:
            self.rounds += 1
            changed = False

            if self.identity is not None and self.identity_evidence.active:
                for a in self.elements:
                    changed |= self._add(
                        (self.identity, a),
                        a,
                        "two-sided-identity",
                    )
                    changed |= self._add(
                        (a, self.identity),
                        a,
                        "two-sided-identity",
                    )

            if self.commutativity.active:
                for (a, b), value in list(self.completed.items()):
                    changed |= self._add(
                        (b, a),
                        value,
                        "commutativity",
                    )

            if self.associativity.active:
                for a in self.elements:
                    for b in self.elements:
                        ab = self.completed.get((a, b))
                        if ab is None:
                            continue
                        for c in self.elements:
                            bc = self.completed.get((b, c))
                            if bc is None:
                                continue
                            left_pair = (ab, c)
                            right_pair = (a, bc)
                            left = self.completed.get(left_pair)
                            right = self.completed.get(right_pair)

                            if left is not None and right is None:
                                changed |= self._add(
                                    right_pair,
                                    left,
                                    "associativity",
                                )
                            elif right is not None and left is None:
                                changed |= self._add(
                                    left_pair,
                                    right,
                                    "associativity",
                                )
                            elif (
                                left is not None
                                and right is not None
                                and left != right
                            ):
                                self.conflicts += 1

            if not changed:
                break
            if self.rounds > len(self.elements) ** 2 + 2:
                raise RuntimeError("algebra closure did not converge")

        self._closed = True

    def infer(self, a: Element, b: Element) -> AlgebraInference:
        self.close()
        pair = (a, b)
        value = self.completed.get(pair)
        if value is None:
            return AlgebraInference(None, False, None)
        if pair in self.observed:
            return AlgebraInference(value, True, "observed")
        return AlgebraInference(
            value,
            True,
            self.provenance.get(pair, "derived"),
        )


class DirectTableMemory:
    def __init__(self, observed: Table) -> None:
        self.observed = dict(observed)

    def infer(self, a: Element, b: Element) -> AlgebraInference:
        value = self.observed.get((a, b))
        return AlgebraInference(
            value,
            value is not None,
            "observed" if value is not None else None,
        )


class MajorityProductBaseline:
    def __init__(self, observed: Table) -> None:
        counts: dict[Element, int] = {}
        for value in observed.values():
            counts[value] = counts.get(value, 0) + 1
        self.value = max(counts, key=counts.get)

    def infer(self, a: Element, b: Element) -> AlgebraInference:
        del a, b
        return AlgebraInference(self.value, True, "majority")


def score_heldout(
    model: object,
    table: Table,
    heldout: Iterable[Pair],
) -> dict[str, float | int]:
    total = 0
    resolved = 0
    correct = 0
    for a, b in heldout:
        total += 1
        inference: AlgebraInference = model.infer(a, b)
        if inference.resolved:
            resolved += 1
            correct += int(inference.value == table[(a, b)])

    return {
        "heldout": total,
        "resolved": resolved,
        "correct": correct,
        "coverage": resolved / max(1, total),
        "accuracy": correct / max(1, total),
        "resolved_accuracy": correct / max(1, resolved),
    }
