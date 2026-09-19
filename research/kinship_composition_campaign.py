"""Campaign for Cortex structured kinship composition."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from kinship_composition import (
    DOWN,
    UP,
    ClippedCountBaseline,
    CortexRelationAlgebra,
    ExactSequenceMemory,
    LastRelationBaseline,
    score,
    test_examples,
    train_examples,
)


def _train_models(examples):
    models = {
        "cortex": CortexRelationAlgebra(),
        "sequence-memory": ExactSequenceMemory(),
        "last-relation": LastRelationBaseline(),
        "clipped-count": ClippedCountBaseline(),
    }
    for example in examples:
        for model in models.values():
            model.observe_solved(example)
    return models


def _entity_ids(examples):
    entities = set()
    for example in examples:
        for source, _relation, target in example.facts:
            entities.add(source)
            entities.add(target)
    return entities


def run_seed(seed: int) -> dict:
    train, primitive_alias, answer_alias = train_examples(seed=seed)
    test = test_examples(
        seed=seed,
        primitive_alias=primitive_alias,
        answer_alias=answer_alias,
    )
    models = _train_models(train)
    metrics = {
        name: score(model, test)
        for name, model in models.items()
    }

    held_train, held_primitive, held_answer = train_examples(
        seed=seed + 100_000,
        omitted_sequence=(UP, DOWN, DOWN),
    )
    held_test = test_examples(
        seed=seed + 100_000,
        primitive_alias=held_primitive,
        answer_alias=held_answer,
        require_prefix=(UP, DOWN, DOWN),
    )
    held_model = CortexRelationAlgebra()
    for example in held_train:
        held_model.observe_solved(example)
    held_metrics = score(held_model, held_test)

    train_entities = _entity_ids(train)
    test_entities = _entity_ids(test)

    return {
        "protocol": "kinship-composition-v1",
        "seed": seed,
        "train_max_path_length": max(x.path_length for x in train),
        "test_min_path_length": min(x.path_length for x in test),
        "test_max_path_length": max(x.path_length for x in test),
        "train_examples": len(train),
        "test_examples": len(test),
        "entity_overlap": len(train_entities.intersection(test_entities)),
        "primitive_alias": primitive_alias,
        "answer_alias": answer_alias,
        "metrics": metrics,
        "missing_rule": {
            "omitted_sequence": [UP, DOWN, DOWN],
            "test_examples": len(held_test),
            **held_metrics,
        },
        "learned_transitions": len(
            models["cortex"].transition_evidence()
        ),
    }


def campaign(seeds: int, output: Path) -> dict:
    rows = [run_seed(seed) for seed in range(seeds)]
    output.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )

    methods = list(rows[0]["metrics"])
    summary = {
        "protocol": "kinship-composition-v1",
        "seeds": seeds,
        "train_max_path_length": rows[0]["train_max_path_length"],
        "test_min_path_length": rows[0]["test_min_path_length"],
        "test_max_path_length": rows[0]["test_max_path_length"],
        "entity_overlap_max": max(row["entity_overlap"] for row in rows),
        "methods": {},
        "missing_rule": {},
    }
    for method in methods:
        summary["methods"][method] = {
            key: sum(
                row["metrics"][method][key]
                for row in rows
            ) / seeds
            for key in ("accuracy", "coverage", "resolved_accuracy")
        }

    summary["missing_rule"] = {
        key: sum(row["missing_rule"][key] for row in rows) / seeds
        for key in ("accuracy", "coverage", "resolved_accuracy")
    }

    print(json.dumps(summary, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("kinship-composition-v1.jsonl"),
    )
    args = parser.parse_args()
    campaign(args.seeds, args.output)


if __name__ == "__main__":
    main()
