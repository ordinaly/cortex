"""Campaign for Cortex explicit-proof-dependency-v1."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from random import Random

from proof_dependency import (
    Literal,
    ProofDependencyModel,
    Rule,
    SingleProofBaseline,
)

PROTOCOL = "explicit-proof-dependency-v1"
FAMILIES = (
    "redundant-branches",
    "short-vs-long",
    "shared-root",
    "contradictory",
    "nested-redundancy",
    "irrelevant-premise",
)
DEVELOPMENT_SEEDS = tuple(range(10))
OFFICIAL_SEEDS = tuple(range(1300, 1320))


@dataclass(frozen=True)
class ProofWorld:
    family: str
    seed: int
    facts: dict[str, Literal]
    rules: tuple[Rule, ...]
    target_atom: str
    fact_roles: dict[str, str]


def _opaque_names(
    rng: Random,
    prefix: str,
    roles: list[str],
) -> dict[str, str]:
    values = [
        f"{prefix}{index:02d}_{rng.randrange(1_000_000):06d}"
        for index in range(len(roles))
    ]
    rng.shuffle(values)
    return {
        role: value
        for role, value in zip(roles, values)
    }


def build_world(family: str, seed: int) -> ProofWorld:
    if family not in FAMILIES:
        raise ValueError(f"unknown family: {family}")

    rng = Random(
        seed * 10_007
        + sum(ord(ch) for ch in family)
    )

    atom_roles = [
        "a",
        "b",
        "c",
        "d",
        "e",
        "s",
        "x",
        "y",
        "w",
        "goal",
    ]
    atom_names = _opaque_names(
        rng,
        "A",
        atom_roles,
    )

    if family == "redundant-branches":
        fact_specs = [
            ("left-a", "a", 1),
            ("left-b", "b", 1),
            ("right-c", "c", 1),
            ("right-d", "d", 1),
        ]
        rule_specs = [
            ("left-mid", (("a", 1), ("b", 1)), ("x", 1)),
            ("left-goal", (("x", 1),), ("goal", 1)),
            ("right-mid", (("c", 1), ("d", 1)), ("y", 1)),
            ("right-goal", (("y", 1),), ("goal", 1)),
        ]
    elif family == "short-vs-long":
        fact_specs = [
            ("short-a", "a", 1),
            ("short-b", "b", 1),
            ("long-c", "c", 1),
            ("long-d", "d", 1),
            ("long-e", "e", 1),
        ]
        rule_specs = [
            ("short-goal", (("a", 1), ("b", 1)), ("goal", 1)),
            ("long-mid", (("c", 1), ("d", 1)), ("x", 1)),
            ("long-goal", (("x", 1), ("e", 1)), ("goal", 1)),
        ]
    elif family == "shared-root":
        fact_specs = [
            ("shared", "s", 1),
            ("left", "a", 1),
            ("right", "b", 1),
        ]
        rule_specs = [
            ("left-mid", (("s", 1), ("a", 1)), ("x", 1)),
            ("right-mid", (("s", 1), ("b", 1)), ("y", 1)),
            ("left-goal", (("x", 1),), ("goal", 1)),
            ("right-goal", (("y", 1),), ("goal", 1)),
        ]
    elif family == "contradictory":
        fact_specs = [
            ("positive-a", "a", 1),
            ("positive-b", "b", 1),
            ("negative-c", "c", 1),
            ("negative-d", "d", 1),
        ]
        rule_specs = [
            ("positive-goal", (("a", 1), ("b", 1)), ("goal", 1)),
            ("negative-goal", (("c", 1), ("d", 1)), ("goal", -1)),
        ]
    elif family == "nested-redundancy":
        fact_specs = [
            ("x-left", "a", 1),
            ("x-right", "b", 1),
            ("x-tail", "c", 1),
            ("independent-d", "d", 1),
            ("independent-e", "e", 1),
        ]
        rule_specs = [
            ("x-from-left", (("a", 1),), ("x", 1)),
            ("x-from-right", (("b", 1),), ("x", 1)),
            ("x-goal", (("x", 1), ("c", 1)), ("goal", 1)),
            ("y-mid", (("d", 1), ("e", 1)), ("y", 1)),
            ("y-goal", (("y", 1),), ("goal", 1)),
        ]
    else:
        fact_specs = [
            ("target-a", "a", 1),
            ("target-b", "b", 1),
            ("irrelevant", "e", 1),
        ]
        rule_specs = [
            ("target-goal", (("a", 1), ("b", 1)), ("goal", 1)),
            ("irrelevant-rule", (("e", 1),), ("w", 1)),
        ]

    fact_roles = [spec[0] for spec in fact_specs]
    fact_ids = _opaque_names(
        rng,
        "F",
        fact_roles,
    )
    rule_roles = [spec[0] for spec in rule_specs]
    rule_ids = _opaque_names(
        rng,
        "R",
        rule_roles,
    )

    facts = {
        fact_ids[role]: Literal(
            atom_names[atom_role],
            polarity,
        )
        for role, atom_role, polarity in fact_specs
    }

    rules = tuple(
        Rule(
            rule_id=rule_ids[role],
            premises=tuple(
                Literal(
                    atom_names[premise_role],
                    polarity,
                )
                for premise_role, polarity in premises
            ),
            conclusion=Literal(
                atom_names[conclusion_role],
                conclusion_polarity,
            ),
        )
        for (
            role,
            premises,
            (conclusion_role, conclusion_polarity),
        ) in rule_specs
    )

    return ProofWorld(
        family=family,
        seed=seed,
        facts=facts,
        rules=rules,
        target_atom=atom_names["goal"],
        fact_roles={
            role: fact_ids[role]
            for role in fact_roles
        },
    )


def _provenance_valid(
    model: ProofDependencyModel,
    rules: tuple[Rule, ...],
) -> bool:
    rules_by_id = {
        rule.rule_id: rule
        for rule in rules
    }
    nodes = {
        node.proof_id: node
        for node in model.proof_nodes()
    }

    for node in nodes.values():
        if node.kind == "fact":
            if node.parents:
                return False
            if node.source_id not in model.facts:
                return False
            if model.facts[node.source_id] != node.literal:
                return False
            continue

        rule = rules_by_id.get(node.source_id)
        if rule is None:
            return False
        if rule.conclusion != node.literal:
            return False
        if len(node.parents) != len(rule.premises):
            return False
        parent_literals = tuple(
            nodes[parent_id].literal
            for parent_id in node.parents
        )
        if parent_literals != rule.premises:
            return False
        expected_facts = tuple(
            sorted(
                {
                    fact_id
                    for parent_id in node.parents
                    for fact_id in nodes[
                        parent_id
                    ].fact_dependencies
                }
            )
        )
        if expected_facts != node.fact_dependencies:
            return False

    return True


def _target_nodes(
    model: ProofDependencyModel,
    atom: str,
):
    explanation = model.explain(atom)
    return {
        proof_id: model.proof(proof_id)
        for proof_id in (
            *explanation.positive_proof_ids,
            *explanation.negative_proof_ids,
        )
    }


def _selected_is_shortest(
    model: ProofDependencyModel,
    atom: str,
) -> bool:
    explanation = model.explain(atom)
    if not explanation.positive_proof_ids:
        return True
    selected = model.proof(
        explanation.selected_positive_id
    )
    minimum = min(
        model.proof(proof_id).depth
        for proof_id in explanation.positive_proof_ids
    )
    return selected.depth == minimum


def _independent_positive_pair(
    model: ProofDependencyModel,
    atom: str,
) -> tuple[str, str] | None:
    ids = model.explain(atom).positive_proof_ids
    for index, left in enumerate(ids):
        for right in ids[index + 1 :]:
            if model.independent(left, right):
                return left, right
    return None


def _exact_invalidation(
    before_nodes,
    after_ids: set[str],
    removed_fact: str,
) -> bool:
    expected = {
        proof_id
        for proof_id, node in before_nodes.items()
        if removed_fact not in node.fact_dependencies
    }
    return after_ids == expected


def run_case(family: str, seed: int) -> dict:
    world = build_world(family, seed)
    model = ProofDependencyModel(
        world.facts,
        world.rules,
    )
    initial = model.explain(world.target_atom)
    initial_nodes = _target_nodes(
        model,
        world.target_atom,
    )
    initial_fingerprint = model.target_fingerprint(
        world.target_atom
    )
    row = {
        "protocol": PROTOCOL,
        "family": family,
        "seed": seed,
        "dag_valid": model.validate_active_dag(),
        "provenance_valid": _provenance_valid(
            model,
            world.rules,
        ),
        "initial_state": initial.state,
        "initial_reason": initial.reason,
        "initial_positive_proofs": len(
            initial.positive_proof_ids
        ),
        "initial_negative_proofs": len(
            initial.negative_proof_ids
        ),
        "selected_is_shortest": _selected_is_shortest(
            model,
            world.target_atom,
        ),
    }

    if family == "redundant-branches":
        pair = _independent_positive_pair(
            model,
            world.target_atom,
        )
        if pair is None:
            raise RuntimeError(
                "redundant fixture lacks independent proofs"
            )
        selected_id = initial.selected_positive_id
        if selected_id is None:
            raise RuntimeError("missing selected proof")
        selected = model.proof(selected_id)
        alternatives = [
            proof_id
            for proof_id in initial.positive_proof_ids
            if proof_id != selected_id
            and model.independent(
                selected_id,
                proof_id,
            )
        ]
        if not alternatives:
            raise RuntimeError("no independent alternate proof")
        alternate_id = alternatives[0]
        removal = selected.fact_dependencies[0]
        baseline = SingleProofBaseline(
            model,
            world.target_atom,
        )

        model.retract_fact(removal)
        after = model.explain(world.target_atom)
        after_ids = set(after.positive_proof_ids)
        row.update(
            {
                "independent_pair_present": True,
                "alternate_survives": (
                    alternate_id in after_ids
                ),
                "after_removal_state": after.state,
                "after_removal_reason": after.reason,
                "single_proof_baseline_state": (
                    baseline.state(model)
                ),
                "exact_invalidation": _exact_invalidation(
                    initial_nodes,
                    after_ids,
                    removal,
                ),
            }
        )
        model.restore_fact(removal)

    elif family == "short-vs-long":
        selected_id = initial.selected_positive_id
        if selected_id is None:
            raise RuntimeError("missing selected proof")
        selected = model.proof(selected_id)
        initial_depths = [
            model.proof(proof_id).depth
            for proof_id in initial.positive_proof_ids
        ]
        removal = selected.fact_dependencies[0]
        baseline = SingleProofBaseline(
            model,
            world.target_atom,
        )

        model.retract_fact(removal)
        after = model.explain(world.target_atom)
        if after.selected_positive_id is None:
            new_depth = None
        else:
            new_depth = model.proof(
                after.selected_positive_id
            ).depth
        row.update(
            {
                "longer_proof_present": (
                    max(initial_depths) > selected.depth
                ),
                "selected_depth": selected.depth,
                "after_removal_state": after.state,
                "takeover_depth": new_depth,
                "deeper_takeover": (
                    new_depth is not None
                    and new_depth > selected.depth
                ),
                "single_proof_baseline_state": (
                    baseline.state(model)
                ),
                "exact_invalidation": _exact_invalidation(
                    initial_nodes,
                    set(after.positive_proof_ids),
                    removal,
                ),
            }
        )
        model.restore_fact(removal)

    elif family == "shared-root":
        shared = world.fact_roles["shared"]
        model.retract_fact(shared)
        after = model.explain(world.target_atom)
        row.update(
            {
                "positive_proofs_share_root": all(
                    shared
                    in model_node.fact_dependencies
                    for model_node in initial_nodes.values()
                ),
                "after_shared_removal_state": after.state,
                "after_shared_removal_reason": after.reason,
                "after_shared_positive_proofs": len(
                    after.positive_proof_ids
                ),
                "exact_invalidation": _exact_invalidation(
                    initial_nodes,
                    set(after.positive_proof_ids),
                    shared,
                ),
            }
        )
        model.restore_fact(shared)

    elif family == "contradictory":
        negative_fact = world.fact_roles["negative-c"]
        positive_fact = world.fact_roles["positive-a"]

        model.retract_fact(negative_fact)
        without_negative = model.explain(
            world.target_atom
        )
        model.restore_fact(negative_fact)
        restored_conflict = model.explain(
            world.target_atom
        )
        model.retract_fact(positive_fact)
        without_positive = model.explain(
            world.target_atom
        )
        model.restore_fact(positive_fact)
        row.update(
            {
                "without_negative_state": (
                    without_negative.state
                ),
                "without_negative_reason": (
                    without_negative.reason
                ),
                "restored_conflict_state": (
                    restored_conflict.state
                ),
                "restored_conflict_reason": (
                    restored_conflict.reason
                ),
                "without_positive_state": (
                    without_positive.state
                ),
                "without_positive_reason": (
                    without_positive.reason
                ),
            }
        )

    elif family == "nested-redundancy":
        removal = world.fact_roles["x-left"]
        model.retract_fact(removal)
        after = model.explain(world.target_atom)
        row.update(
            {
                "after_nested_removal_state": after.state,
                "proof_count_decreased": (
                    len(after.positive_proof_ids)
                    < len(initial.positive_proof_ids)
                ),
                "multiple_proofs_remain": (
                    len(after.positive_proof_ids) >= 2
                ),
                "exact_invalidation": _exact_invalidation(
                    initial_nodes,
                    set(after.positive_proof_ids),
                    removal,
                ),
            }
        )
        model.restore_fact(removal)

    else:
        irrelevant = world.fact_roles["irrelevant"]
        model.retract_fact(irrelevant)
        after_fingerprint = model.target_fingerprint(
            world.target_atom
        )
        after = model.explain(world.target_atom)
        row.update(
            {
                "after_irrelevant_state": after.state,
                "target_fingerprint_unchanged": (
                    after_fingerprint
                    == initial_fingerprint
                ),
            }
        )
        model.restore_fact(irrelevant)

    row["restoration_exact"] = (
        model.target_fingerprint(world.target_atom)
        == initial_fingerprint
    )
    row["restored_dag_valid"] = (
        model.validate_active_dag()
    )
    return row


def campaign(
    seeds: tuple[int, ...],
    output: Path,
) -> dict:
    rows = [
        run_case(family, seed)
        for seed in seeds
        for family in FAMILIES
    ]
    output.write_text(
        "".join(
            json.dumps(row, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )

    summary = {
        "protocol": PROTOCOL,
        "seeds": list(seeds),
        "families": list(FAMILIES),
        "rows": len(rows),
        "common": {
            "dag_valid_rate": sum(
                int(row["dag_valid"])
                for row in rows
            )
            / len(rows),
            "provenance_valid_rate": sum(
                int(row["provenance_valid"])
                for row in rows
            )
            / len(rows),
            "restoration_exact_rate": sum(
                int(row["restoration_exact"])
                for row in rows
            )
            / len(rows),
        },
    }
    print(json.dumps(summary, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--development", action="store_true")
    group.add_argument("--official", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "explicit-proof-dependency-v1.jsonl"
        ),
    )
    args = parser.parse_args()
    seeds = (
        OFFICIAL_SEEDS
        if args.official
        else DEVELOPMENT_SEEDS
    )
    campaign(seeds, args.output)


if __name__ == "__main__":
    main()
