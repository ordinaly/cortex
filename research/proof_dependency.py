"""Explicit proof/dependency DAG for Cortex Reasoning Map v1.

This research prototype separates a conclusion from the proof alternatives that
currently support or refute it. Facts are source nodes. Rules create derived
proof nodes whose parent links form an explicit acyclic dependency graph.

The module is deliberately finite and monotone within one rebuild. Fact
retraction/restoration changes which proof nodes survive; it does not attempt
open-ended theorem discovery.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from itertools import product
from json import dumps
from typing import Iterable, Mapping


@dataclass(frozen=True, order=True)
class Literal:
    atom: str
    polarity: int = 1

    def __post_init__(self) -> None:
        if not self.atom:
            raise ValueError("literal atom must be non-empty")
        if self.polarity not in (-1, 1):
            raise ValueError("literal polarity must be -1 or 1")

    def negated(self) -> "Literal":
        return Literal(self.atom, -self.polarity)


@dataclass(frozen=True)
class Rule:
    rule_id: str
    premises: tuple[Literal, ...]
    conclusion: Literal

    def __post_init__(self) -> None:
        if not self.rule_id:
            raise ValueError("rule_id must be non-empty")
        if not self.premises:
            raise ValueError("rules require at least one premise")


@dataclass(frozen=True)
class ProofNode:
    proof_id: str
    literal: Literal
    kind: str
    source_id: str
    parents: tuple[str, ...]
    fact_dependencies: tuple[str, ...]
    rule_dependencies: tuple[str, ...]
    depth: int


@dataclass(frozen=True)
class ProofExplanation:
    atom: str
    state: int
    reason: str
    positive_proof_ids: tuple[str, ...]
    negative_proof_ids: tuple[str, ...]
    selected_positive_id: str | None
    selected_negative_id: str | None
    nodes: tuple[ProofNode, ...]

    @property
    def resolved(self) -> bool:
        return self.state != 0


def _literal_key(literal: Literal) -> tuple[str, int]:
    return (literal.atom, literal.polarity)


class ProofDependencyModel:
    """Finite explicit proof DAG with fact-level retraction/restoration."""

    def __init__(
        self,
        facts: Mapping[str, Literal],
        rules: Iterable[Rule],
        *,
        max_proofs_per_literal: int = 256,
    ) -> None:
        self.facts = dict(facts)
        self.rules = tuple(rules)
        if not self.facts:
            raise ValueError("at least one fact is required")
        if len(self.facts) != len(set(self.facts)):
            raise ValueError("fact IDs must be unique")
        if any(not fact_id for fact_id in self.facts):
            raise ValueError("fact IDs must be non-empty")
        rule_ids = [rule.rule_id for rule in self.rules]
        if len(rule_ids) != len(set(rule_ids)):
            raise ValueError("rule IDs must be unique")
        if max_proofs_per_literal < 1:
            raise ValueError("max_proofs_per_literal must be positive")

        self.max_proofs_per_literal = int(max_proofs_per_literal)
        self._active_facts = set(self.facts)
        self._rule_order = self._validate_and_order_rules()
        self._dirty = True
        self._nodes: dict[str, ProofNode] = {}
        self._proofs_by_literal: dict[Literal, tuple[str, ...]] = {}
        self._dependency_index: dict[str, tuple[str, ...]] = {}

    def _validate_and_order_rules(self) -> tuple[Rule, ...]:
        literals = set(self.facts.values())
        for rule in self.rules:
            literals.add(rule.conclusion)
            literals.update(rule.premises)

        adjacency = {literal: set() for literal in literals}
        indegree = {literal: 0 for literal in literals}
        for rule in self.rules:
            for premise in rule.premises:
                if rule.conclusion not in adjacency[premise]:
                    adjacency[premise].add(rule.conclusion)
                    indegree[rule.conclusion] += 1

        ready = sorted(
            (literal for literal, degree in indegree.items() if degree == 0),
            key=_literal_key,
        )
        order: list[Literal] = []
        cursor = 0
        while cursor < len(ready):
            literal = ready[cursor]
            cursor += 1
            order.append(literal)
            for nxt in sorted(adjacency[literal], key=_literal_key):
                indegree[nxt] -= 1
                if indegree[nxt] == 0:
                    ready.append(nxt)

        if len(order) != len(literals):
            raise ValueError("proof rules must form an acyclic literal graph")

        rank = {literal: index for index, literal in enumerate(order)}
        return tuple(
            sorted(
                self.rules,
                key=lambda rule: (
                    rank[rule.conclusion],
                    rule.rule_id,
                ),
            )
        )

    @staticmethod
    def _rule_proof_id(rule_id: str, parents: tuple[str, ...]) -> str:
        payload = rule_id + "|" + "|".join(parents)
        digest = sha256(payload.encode("utf-8")).hexdigest()[:20]
        return f"proof:{digest}"

    def _ensure_built(self) -> None:
        if not self._dirty:
            return

        nodes: dict[str, ProofNode] = {}
        proof_lists: dict[Literal, list[str]] = {}

        for fact_id in sorted(self._active_facts):
            literal = self.facts[fact_id]
            proof_id = f"fact:{fact_id}"
            node = ProofNode(
                proof_id=proof_id,
                literal=literal,
                kind="fact",
                source_id=fact_id,
                parents=tuple(),
                fact_dependencies=(fact_id,),
                rule_dependencies=tuple(),
                depth=0,
            )
            nodes[proof_id] = node
            proof_lists.setdefault(literal, []).append(proof_id)

        for rule in self._rule_order:
            parent_options = [
                tuple(proof_lists.get(premise, ()))
                for premise in rule.premises
            ]
            if any(not options for options in parent_options):
                continue

            for parent_ids in product(*parent_options):
                parents = tuple(parent_ids)
                fact_dependencies = tuple(
                    sorted(
                        {
                            fact_id
                            for proof_id in parents
                            for fact_id in nodes[
                                proof_id
                            ].fact_dependencies
                        }
                    )
                )
                rule_dependencies = tuple(
                    sorted(
                        {
                            rule.rule_id,
                            *(
                                dependency
                                for proof_id in parents
                                for dependency in nodes[
                                    proof_id
                                ].rule_dependencies
                            ),
                        }
                    )
                )
                depth = 1 + max(
                    nodes[proof_id].depth
                    for proof_id in parents
                )
                proof_id = self._rule_proof_id(
                    rule.rule_id,
                    parents,
                )
                if proof_id in nodes:
                    continue
                node = ProofNode(
                    proof_id=proof_id,
                    literal=rule.conclusion,
                    kind="rule",
                    source_id=rule.rule_id,
                    parents=parents,
                    fact_dependencies=fact_dependencies,
                    rule_dependencies=rule_dependencies,
                    depth=depth,
                )
                nodes[proof_id] = node
                bucket = proof_lists.setdefault(
                    rule.conclusion,
                    [],
                )
                bucket.append(proof_id)
                if len(bucket) > self.max_proofs_per_literal:
                    raise RuntimeError(
                        "proof bound exceeded for "
                        f"{rule.conclusion.atom}"
                    )

        self._nodes = nodes
        self._proofs_by_literal = {
            literal: tuple(
                sorted(
                    proof_ids,
                    key=lambda proof_id: self._proof_sort_key(
                        nodes[proof_id]
                    ),
                )
            )
            for literal, proof_ids in proof_lists.items()
        }
        dependency_index: dict[str, list[str]] = {
            fact_id: []
            for fact_id in self.facts
        }
        for proof_id, node in nodes.items():
            for fact_id in node.fact_dependencies:
                dependency_index[fact_id].append(proof_id)
        self._dependency_index = {
            fact_id: tuple(sorted(proof_ids))
            for fact_id, proof_ids in dependency_index.items()
        }
        self._dirty = False

    @staticmethod
    def _proof_sort_key(
        node: ProofNode,
    ) -> tuple[int, int, int, str]:
        return (
            node.depth,
            len(node.fact_dependencies),
            len(node.rule_dependencies),
            node.proof_id,
        )

    @property
    def active_fact_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._active_facts))

    def retract_fact(self, fact_id: str) -> None:
        if fact_id not in self.facts:
            raise KeyError(f"unknown fact: {fact_id}")
        self._active_facts.discard(fact_id)
        self._dirty = True

    def restore_fact(self, fact_id: str) -> None:
        if fact_id not in self.facts:
            raise KeyError(f"unknown fact: {fact_id}")
        self._active_facts.add(fact_id)
        self._dirty = True

    def proof_nodes(self) -> tuple[ProofNode, ...]:
        self._ensure_built()
        return tuple(
            self._nodes[proof_id]
            for proof_id in sorted(self._nodes)
        )

    def proofs_for(
        self,
        literal: Literal,
    ) -> tuple[ProofNode, ...]:
        self._ensure_built()
        return tuple(
            self._nodes[proof_id]
            for proof_id in self._proofs_by_literal.get(
                literal,
                (),
            )
        )

    def proofs_depending_on(
        self,
        fact_id: str,
    ) -> tuple[str, ...]:
        if fact_id not in self.facts:
            raise KeyError(f"unknown fact: {fact_id}")
        self._ensure_built()
        return self._dependency_index.get(fact_id, tuple())

    def proof(self, proof_id: str) -> ProofNode:
        self._ensure_built()
        try:
            return self._nodes[proof_id]
        except KeyError as exc:
            raise KeyError(f"unknown active proof: {proof_id}") from exc

    def _selected(
        self,
        literal: Literal,
    ) -> str | None:
        proof_ids = self._proofs_by_literal.get(literal, ())
        return proof_ids[0] if proof_ids else None

    def explain(self, atom: str) -> ProofExplanation:
        if not atom:
            raise ValueError("atom must be non-empty")
        self._ensure_built()

        positive = Literal(atom, 1)
        negative = Literal(atom, -1)
        positive_ids = self._proofs_by_literal.get(
            positive,
            tuple(),
        )
        negative_ids = self._proofs_by_literal.get(
            negative,
            tuple(),
        )

        if positive_ids and not negative_ids:
            state = 1
            reason = "supported"
        elif negative_ids and not positive_ids:
            state = -1
            reason = "refuted"
        elif positive_ids and negative_ids:
            state = 0
            reason = "conflict"
        else:
            state = 0
            reason = "unproven"

        roots = set(positive_ids) | set(negative_ids)
        included: set[str] = set()
        stack = list(roots)
        while stack:
            proof_id = stack.pop()
            if proof_id in included:
                continue
            included.add(proof_id)
            stack.extend(self._nodes[proof_id].parents)

        ordered_nodes = tuple(
            sorted(
                (self._nodes[proof_id] for proof_id in included),
                key=lambda node: (
                    node.depth,
                    node.proof_id,
                ),
            )
        )
        return ProofExplanation(
            atom=atom,
            state=state,
            reason=reason,
            positive_proof_ids=tuple(positive_ids),
            negative_proof_ids=tuple(negative_ids),
            selected_positive_id=self._selected(positive),
            selected_negative_id=self._selected(negative),
            nodes=ordered_nodes,
        )

    def proof_ids_for_atom(
        self,
        atom: str,
    ) -> tuple[str, ...]:
        explanation = self.explain(atom)
        return tuple(
            sorted(
                {
                    *explanation.positive_proof_ids,
                    *explanation.negative_proof_ids,
                }
            )
        )

    def independent(
        self,
        left_proof_id: str,
        right_proof_id: str,
    ) -> bool:
        left = self.proof(left_proof_id)
        right = self.proof(right_proof_id)
        return not (
            set(left.fact_dependencies)
            & set(right.fact_dependencies)
        )

    def target_fingerprint(self, atom: str) -> str:
        explanation = self.explain(atom)
        payload = {
            "state": explanation.state,
            "reason": explanation.reason,
            "positive": explanation.positive_proof_ids,
            "negative": explanation.negative_proof_ids,
            "nodes": [
                {
                    "proof_id": node.proof_id,
                    "literal": (
                        node.literal.atom,
                        node.literal.polarity,
                    ),
                    "kind": node.kind,
                    "source_id": node.source_id,
                    "parents": node.parents,
                    "facts": node.fact_dependencies,
                    "rules": node.rule_dependencies,
                    "depth": node.depth,
                }
                for node in explanation.nodes
            ],
        }
        return sha256(
            dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()

    def validate_active_dag(self) -> bool:
        self._ensure_built()
        for proof_id, node in self._nodes.items():
            if not set(node.fact_dependencies).issubset(
                self._active_facts
            ):
                return False
            for parent_id in node.parents:
                parent = self._nodes.get(parent_id)
                if parent is None:
                    return False
                if parent.depth >= node.depth:
                    return False
            if node.kind == "fact" and node.parents:
                return False
            if node.kind == "rule" and not node.parents:
                return False
        return True


class SingleProofBaseline:
    """Baseline that commits to one initially selected positive proof."""

    def __init__(
        self,
        model: ProofDependencyModel,
        atom: str,
    ) -> None:
        explanation = model.explain(atom)
        if explanation.selected_positive_id is None:
            raise ValueError(
                "baseline requires an initial positive proof"
            )
        selected = model.proof(
            explanation.selected_positive_id
        )
        self.dependencies = selected.fact_dependencies

    def state(self, model: ProofDependencyModel) -> int:
        active = set(model.active_fact_ids)
        return (
            1
            if set(self.dependencies).issubset(active)
            else 0
        )
