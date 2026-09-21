#!/usr/bin/env python3
"""Held-out relational inference over Cortex hypergeometry."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
from random import Random

from cortex import CortexRuntime
from cortex.hypergraph_geometry import HypergraphGeometry
from cortex.hypergraph_inference import HypergraphCompositionLearner
from cortex.knowledge import (
    KnowledgeGraphView,
    KnowledgeStatement,
    feed_knowledge,
    make_knowledge_runtime,
)


PROTOCOL = "hypergraph-heldout-inference-v0.1"


@dataclass(frozen=True)
class Query:
    source: str
    relation: str
    target: str
    candidates: tuple[str, ...]


def _rank_metrics(scores: dict[str, float], query: Query) -> tuple[float, float]:
    target_score = scores[query.target]
    greater = sum(
        scores[candidate] > target_score + 1.0e-15
        for candidate in query.candidates
    )
    tied = sum(
        abs(scores[candidate] - target_score) <= 1.0e-15
        for candidate in query.candidates
    )
    ranks = range(greater + 1, greater + tied + 1)
    mrr = sum(1.0 / rank for rank in ranks) / tied
    hit1 = sum(rank == 1 for rank in ranks) / tied
    return mrr, hit1


def _opaque_name(rng: Random, prefix: str, index: int) -> str:
    return f"{prefix}-{index}-{rng.getrandbits(48):012x}"


def make_world(seed: int):
    rng = Random(seed)
    relation_tokens = ["R0", "R1", "R2"]
    rng.shuffle(relation_tokens)
    capital_relation, region_relation, target_relation = relation_tokens

    regions = tuple(_opaque_name(rng, "z", i) for i in range(4))
    statements: list[KnowledgeStatement] = []
    queries: list[Query] = []
    index = 0

    for region_index, region in enumerate(regions):
        for member_index in range(3):
            country = _opaque_name(
                rng,
                f"h{region_index}",
                member_index,
            )
            city = _opaque_name(
                rng,
                f"m{region_index}",
                member_index,
            )
            statements.append(
                KnowledgeStatement(
                    f"s{index}",
                    country,
                    capital_relation,
                    city,
                    provenance=PROTOCOL,
                )
            )
            index += 1
            statements.append(
                KnowledgeStatement(
                    f"s{index}",
                    country,
                    region_relation,
                    region,
                    provenance=PROTOCOL,
                )
            )
            index += 1

            if member_index < 2:
                statements.append(
                    KnowledgeStatement(
                        f"s{index}",
                        city,
                        target_relation,
                        region,
                        provenance=PROTOCOL,
                    )
                )
                index += 1
            else:
                queries.append(
                    Query(
                        source=city,
                        relation=target_relation,
                        target=region,
                        candidates=regions,
                    )
                )

    # Distractors use the same opaque relation tokens, but are disconnected
    # from held-out sources so they test rule confidence without leaking the
    # answers.
    for distractor in range(8):
        left = _opaque_name(rng, "noise-a", distractor)
        right = _opaque_name(rng, "noise-b", distractor)
        relation = rng.choice(relation_tokens)
        statements.append(
            KnowledgeStatement(
                f"s{index}",
                left,
                relation,
                right,
                provenance=PROTOCOL,
            )
        )
        index += 1

    rng.shuffle(statements)
    return tuple(statements), tuple(queries), relation_tokens


def _prototype_distance(snapshot, left_id: int, right_id: int) -> float:
    prototypes = snapshot["articulation"]["prototypes"]
    reliability = snapshot["articulation"]["reliability"][left_id]
    left = prototypes[left_id]
    right = prototypes[right_id]
    denominator = max(1.0e-12, sum(reliability))
    return sum(
        weight * (a - b) ** 2
        for a, b, weight in zip(left, right, reliability)
    ) / denominator


def run_seed(seed: int) -> dict:
    statements, queries, aliases = make_world(seed)
    runtime = make_knowledge_runtime(
        CortexRuntime,
        feature_dim=64,
        max_entities=2048,
        budget=64,
    )
    report = feed_knowledge(runtime, statements)
    view = KnowledgeGraphView(
        report.snapshot_json,
        report.symbol_bindings,
        runtime.config_json(),
        report.statement_incidence,
    )
    geometry = HypergraphGeometry(view.hyperedges())
    learner = HypergraphCompositionLearner(
        geometry,
        path_length=2,
        minimum_examples=4,
        confidence_threshold=0.80,
        resolution=0.5,
    )
    rules = learner.fit()

    snapshot = json.loads(report.snapshot_json)
    hyper_mrr = []
    hyper_hit1 = []
    identity_mrr = []
    identity_hit1 = []
    direct_coverage = []
    resolved = []
    predictions = []

    for query in queries:
        # The held-out target relation is absent by construction.
        direct = view.objects(query.source, query.relation)
        direct_coverage.append(float(bool(direct)))

        inference = learner.infer(query.source, query.relation)
        hyper_scores = {
            candidate: 0.0
            for candidate in query.candidates
        }
        for candidate in inference.candidates:
            if candidate.target in hyper_scores:
                hyper_scores[candidate.target] = max(
                    hyper_scores[candidate.target],
                    candidate.score,
                )
        mrr, hit1 = _rank_metrics(hyper_scores, query)
        hyper_mrr.append(mrr)
        hyper_hit1.append(hit1)
        resolved.append(float(inference.resolved))

        source_id = report.symbol_bindings[query.source]
        identity_scores = {
            candidate: -_prototype_distance(
                snapshot,
                source_id,
                report.symbol_bindings[candidate],
            )
            for candidate in query.candidates
        }
        mrr, hit1 = _rank_metrics(identity_scores, query)
        identity_mrr.append(mrr)
        identity_hit1.append(hit1)

        predictions.append(
            {
                "source": query.source,
                "target": query.target,
                "hyper_top": (
                    inference.top.target
                    if inference.top is not None
                    else None
                ),
                "hyper_score": (
                    inference.top.score
                    if inference.top is not None
                    else 0.0
                ),
            }
        )

    target_alias = queries[0].relation
    learned_target_rules = [
        rule
        for rule in rules
        if rule.target_relation == target_alias
    ]

    return {
        "seed": seed,
        "statements": len(statements),
        "queries": len(queries),
        "relation_aliases": aliases,
        "binding_conflicts": report.binding_conflicts,
        "learned_rules": len(rules),
        "learned_target_rules": len(learned_target_rules),
        "hyper_mrr": sum(hyper_mrr) / len(hyper_mrr),
        "hyper_hit1": sum(hyper_hit1) / len(hyper_hit1),
        "hyper_coverage": sum(resolved) / len(resolved),
        "identity_mrr": sum(identity_mrr) / len(identity_mrr),
        "identity_hit1": sum(identity_hit1) / len(identity_hit1),
        "direct_memory_coverage": sum(direct_coverage) / len(direct_coverage),
        "predictions": predictions,
    }


def campaign(seeds: int) -> dict:
    rows = [run_seed(seed) for seed in range(seeds)]

    def mean(name):
        return sum(float(row[name]) for row in rows) / len(rows)

    summary = {
        "protocol": PROTOCOL,
        "seeds": seeds,
        "queries": sum(row["queries"] for row in rows),
        "hyper_mrr": mean("hyper_mrr"),
        "hyper_hit1": mean("hyper_hit1"),
        "hyper_coverage": mean("hyper_coverage"),
        "identity_mrr": mean("identity_mrr"),
        "identity_hit1": mean("identity_hit1"),
        "direct_memory_coverage": mean("direct_memory_coverage"),
        "all_zero_binding_conflicts": all(
            row["binding_conflicts"] == 0
            for row in rows
        ),
        "rule_discovered_every_seed": all(
            row["learned_target_rules"] >= 1
            for row in rows
        ),
    }
    gates = {
        "heldout_is_genuine": summary["direct_memory_coverage"] == 0.0,
        "rule_discovery": summary["rule_discovered_every_seed"],
        "hypergraph_coverage": summary["hyper_coverage"] >= 0.99,
        "hypergraph_hit1": summary["hyper_hit1"] >= 0.95,
        "hypergraph_beats_identity": (
            summary["hyper_hit1"]
            >= summary["identity_hit1"] + 0.40
        ),
        "identity_not_semantic": summary["identity_hit1"] <= 0.45,
        "binding_stable": summary["all_zero_binding_conflicts"],
    }
    return {
        **summary,
        "gates": gates,
        "passed": all(gates.values()),
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=50)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.seeds < 1:
        raise SystemExit("--seeds must be positive")
    report = campaign(args.seeds)
    payload = json.dumps(report, indent=2, sort_keys=True)
    print(payload)
    if args.output:
        args.output.write_text(payload + "\n", encoding="utf-8")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
