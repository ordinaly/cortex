"""Variable-depth relational prediction for Cortex v1.2-R.

This fixture asks whether one diffusion operator can expose prospective relation
support whose first non-zero path contribution occurs at different depths.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.linalg import expm


def make_query_graph(
    depth: int,
    *,
    seed: int,
    distractor_offsets: tuple[int, ...] = (2, 4, 6),
) -> tuple[np.ndarray, int, list[int]]:
    if depth < 1 or depth % 2 == 0:
        raise ValueError("depth must be a positive odd integer")

    rng = np.random.default_rng(seed)
    lengths = [depth, *(depth + offset for offset in distractor_offsets)]
    edges: list[tuple[int, int, float]] = []
    targets: list[int] = []
    next_node = 1
    source = 0

    for length in lengths:
        previous = source
        for _ in range(length):
            node = next_node
            next_node += 1
            # Mild deterministic noise prevents the fixture from relying on
            # exact unit-weight symmetries.
            weight = float(rng.uniform(0.85, 1.15))
            edges.append((previous, node, weight))
            previous = node
        targets.append(previous)

    adjacency = np.zeros((next_node, next_node), dtype=float)
    for i, j, weight in edges:
        adjacency[i, j] = adjacency[j, i] = weight
    return adjacency, source, targets


def heat_scores(adjacency: np.ndarray, source: int, targets: list[int], rho: float):
    laplacian = np.diag(adjacency.sum(axis=1)) - adjacency
    kernel = expm(-rho * laplacian)
    return [float(kernel[source, target]) for target in targets]


def path_scores(adjacency: np.ndarray, source: int, targets: list[int], order: int):
    power = np.linalg.matrix_power(adjacency, order)
    return [float(power[source, target]) for target in targets]


def _tie_aware_query_metrics(scores: list[float]) -> dict[str, float]:
    """Target 0 is the designated future-positive endpoint."""

    positive = scores[0]
    greater = sum(score > positive + 1.0e-15 for score in scores)
    tied = sum(abs(score - positive) <= 1.0e-15 for score in scores)
    best_rank = greater + 1
    ranks = list(range(best_rank, best_rank + tied))
    return {
        "mrr": float(np.mean([1.0 / rank for rank in ranks])),
        "hit_at_1": float(np.mean([rank <= 1 for rank in ranks])),
        "hit_at_3": float(np.mean([rank <= 3 for rank in ranks])),
        "positive_score": float(positive),
        "max_negative_score": float(max(scores[1:])),
    }


def run_seed(seed: int, rho: float = 1.0) -> dict:
    rows = {}
    for depth in (3, 5, 7):
        adjacency, source, targets = make_query_graph(
            depth,
            seed=seed * 101 + depth,
        )
        methods = {
            "path3": path_scores(adjacency, source, targets, 3),
            "path5": path_scores(adjacency, source, targets, 5),
            "path7": path_scores(adjacency, source, targets, 7),
            "heat-kernel": heat_scores(adjacency, source, targets, rho),
        }
        rows[str(depth)] = {
            name: _tie_aware_query_metrics(scores)
            for name, scores in methods.items()
        }

    return {
        "protocol": "predictive-relations-depth-v1",
        "seed": seed,
        "rho": rho,
        "depths": rows,
    }


def campaign(seeds: int, output: Path) -> None:
    rows = [run_seed(seed) for seed in range(seeds)]
    output.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )

    for depth in ("3", "5", "7"):
        for method in ("path3", "path5", "path7", "heat-kernel"):
            summary = {
                metric: float(
                    np.mean(
                        [
                            row["depths"][depth][method][metric]
                            for row in rows
                        ]
                    )
                )
                for metric in ("mrr", "hit_at_1", "hit_at_3")
            }
            print(
                json.dumps(
                    {
                        "depth": int(depth),
                        "method": method,
                        "seeds": seeds,
                        **summary,
                    },
                    sort_keys=True,
                )
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("predictive-relations-depth-v1.jsonl"),
    )
    args = parser.parse_args()
    campaign(args.seeds, args.output)


if __name__ == "__main__":
    main()
