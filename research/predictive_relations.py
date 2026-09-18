"""Prospective relation prediction for Cortex v1.2-R.

Finite synthetic experiment only. The hidden domain labels used by the
generator are never passed to the learner.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.linalg import expm

from multiresolution_semantic import (
    pairwise_semantic_distances,
    sqrt_probability_features,
)


@dataclass
class PairEvidence:
    successes: int = 0
    exposures: int = 0


def _pair(i: int, j: int) -> tuple[int, int]:
    return (i, j) if i < j else (j, i)


def learned_affinity(
    n_nodes: int,
    evidence: dict[tuple[int, int], PairEvidence],
    *,
    alpha0: float = 1.0,
    beta0: float = 1.0,
    confidence_tau: float = 8.0,
    background_rate: float = 0.03,
) -> np.ndarray:
    adjacency = np.zeros((n_nodes, n_nodes), dtype=float)
    for (i, j), cell in evidence.items():
        if cell.exposures <= 0:
            continue
        posterior = (alpha0 + cell.successes) / (
            alpha0 + beta0 + cell.exposures
        )
        confidence = cell.exposures / (cell.exposures + confidence_tau)
        weight = confidence * max(0.0, posterior - background_rate)
        adjacency[i, j] = adjacency[j, i] = weight
    return adjacency


def heat_kernel(adjacency: np.ndarray, rho: float) -> np.ndarray:
    degree = np.diag(adjacency.sum(axis=1))
    laplacian = degree - adjacency
    return expm(-rho * laplacian)


def _rank_metrics(labels: list[int], scores: list[float]) -> dict[str, float]:
    order = sorted(range(len(scores)), key=lambda k: (-scores[k], k))
    positives = [k for k, y in enumerate(labels) if y == 1]
    positive_ranks = []
    for positive in positives:
        rank = order.index(positive) + 1
        positive_ranks.append(rank)
    return {
        "mrr": float(np.mean([1.0 / r for r in positive_ranks])),
        "hit_at_1": float(np.mean([r <= 1 for r in positive_ranks])),
        "hit_at_3": float(np.mean([r <= 3 for r in positive_ranks])),
    }


def _auc(labels: list[int], scores: list[float]) -> float:
    positives = [s for y, s in zip(labels, scores) if y == 1]
    negatives = [s for y, s in zip(labels, scores) if y == 0]
    wins = ties = total = 0
    for p in positives:
        for n in negatives:
            total += 1
            if p > n:
                wins += 1
            elif p == n:
                ties += 1
    return (wins + 0.5 * ties) / total


def _average_precision(labels: list[int], scores: list[float]) -> float:
    order = sorted(range(len(scores)), key=lambda k: (-scores[k], k))
    hits = 0
    precisions = []
    for rank, index in enumerate(order, start=1):
        if labels[index]:
            hits += 1
            precisions.append(hits / rank)
    return float(np.mean(precisions))


def run_seed(
    seed: int,
    *,
    domains: int = 4,
    per_role: int = 3,
    exposures_per_observed_pair: int = 25,
    within_probability: float = 0.75,
    cross_probability: float = 0.16,
    background_rate: float = 0.03,
    rho: float = 1.0,
) -> dict:
    rng = np.random.default_rng(seed)
    n_nodes = domains * per_role * 2

    domain = []
    role = []
    names = []
    for d in range(domains):
        for i in range(per_role):
            domain.append(d)
            role.append(0)
            names.append(f"actor-{d}-{i}")
        for i in range(per_role):
            domain.append(d)
            role.append(1)
            names.append(f"receiver-{d}-{i}")

    # Fine informational roles deliberately identify actor/receiver function,
    # but contain no domain identity.
    role_features = sqrt_probability_features(
        [[0.90, 0.08, 0.02] if r == 0 else [0.08, 0.90, 0.02] for r in role]
    )
    d0 = pairwise_semantic_distances(role_features)

    held_positive = []
    held_negative = []
    for d in range(domains):
        actor = d * (2 * per_role)
        receiver_same = actor + per_role
        held_positive.append(_pair(actor, receiver_same))
        for other in range(domains):
            if other == d:
                continue
            receiver_other = other * (2 * per_role) + per_role
            held_negative.append(_pair(actor, receiver_other))

    held = set(held_positive + held_negative)
    evidence: dict[tuple[int, int], PairEvidence] = {}

    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if role[i] == role[j] or (i, j) in held:
                continue
            probability = (
                within_probability if domain[i] == domain[j] else cross_probability
            )
            successes = int(
                rng.binomial(exposures_per_observed_pair, probability)
            )
            evidence[(i, j)] = PairEvidence(
                successes=successes,
                exposures=exposures_per_observed_pair,
            )

    adjacency = learned_affinity(
        n_nodes,
        evidence,
        background_rate=background_rate,
    )
    kernel = heat_kernel(adjacency, rho)
    path3 = adjacency @ adjacency @ adjacency
    common = adjacency @ adjacency

    candidates = held_positive + held_negative
    labels = [1] * len(held_positive) + [0] * len(held_negative)

    # All evaluated pairs have zero direct exposure by construction.
    direct_scores = [0.5 for _ in candidates]
    fine_scores = [-float(d0[i, j]) for i, j in candidates]
    common_scores = [float(common[i, j]) for i, j in candidates]
    path3_scores = [float(path3[i, j]) for i, j in candidates]
    heat_scores = [float(kernel[i, j]) for i, j in candidates]

    methods = {
        "direct-prior": direct_scores,
        "fine-distance": fine_scores,
        "common-neighbors": common_scores,
        "path3": path3_scores,
        "heat-kernel": heat_scores,
    }
    metrics = {}
    for name, scores in methods.items():
        metrics[name] = {
            "auc": _auc(labels, scores),
            "average_precision": _average_precision(labels, scores),
            **_rank_metrics(labels, scores),
        }

    return {
        "protocol": "predictive-relations-v1",
        "seed": seed,
        "domains": domains,
        "per_role": per_role,
        "nodes": n_nodes,
        "training_exposures_per_observed_pair": exposures_per_observed_pair,
        "within_probability": within_probability,
        "cross_probability": cross_probability,
        "background_rate": background_rate,
        "rho": rho,
        "heldout_positive_pairs": len(held_positive),
        "heldout_negative_pairs": len(held_negative),
        "all_evaluated_pairs_unexposed": all(
            pair not in evidence for pair in candidates
        ),
        "metrics": metrics,
        "positive_heat_min": min(
            float(kernel[i, j]) for i, j in held_positive
        ),
        "negative_heat_max": max(
            float(kernel[i, j]) for i, j in held_negative
        ),
        "names": names,
    }


def campaign(seeds: int, output: Path) -> None:
    rows = [run_seed(seed) for seed in range(seeds)]
    output.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )

    for method in rows[0]["metrics"]:
        print(
            json.dumps(
                {
                    "method": method,
                    "seeds": seeds,
                    **{
                        key: float(
                            np.mean([row["metrics"][method][key] for row in rows])
                        )
                        for key in [
                            "auc",
                            "average_precision",
                            "mrr",
                            "hit_at_1",
                            "hit_at_3",
                        ]
                    },
                },
                sort_keys=True,
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=50)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("predictive-relations-v1.jsonl"),
    )
    args = parser.parse_args()
    campaign(args.seeds, args.output)


if __name__ == "__main__":
    main()
