"""Unsupervised latent event discovery from trajectory/consequence signatures."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


EVENT_TYPES = 6
SIGNATURE_DIM = 10


def event_center(event_type: int) -> np.ndarray:
    """Generator-only latent dynamics templates.

    Coordinates encode:
    delta distance, relative speed, source velocity change, target velocity
    change, source state change, target state change, contact, alignment,
    transfer balance, persistence.
    """
    centers = np.asarray(
        [
            [-0.90, -0.45, 0.05, 0.05, 0.00, 0.00, 0.20, 0.15, 0.00, 0.30],
            [0.90, 0.45, 0.05, 0.05, 0.00, 0.00, 0.05, 0.05, 0.00, 0.20],
            [0.00, 0.05, 0.10, 0.10, -0.75, 0.75, 0.80, 0.20, 0.95, 0.45],
            [0.05, -0.15, -0.85, 0.85, 0.00, 0.00, 0.90, -0.20, 0.00, 0.20],
            [0.00, 0.00, 0.05, 0.05, 0.00, 0.00, 0.10, 0.95, 0.00, 0.85],
            [0.00, 0.00, 0.00, 0.05, 0.00, 0.95, 0.95, 0.10, 0.00, 0.70],
        ],
        dtype=float,
    )
    if not 0 <= event_type < len(centers):
        raise ValueError("invalid event type")
    return centers[event_type].copy()


def sample_signature(
    event_type: int,
    rng: np.random.Generator,
    *,
    noise: float,
) -> np.ndarray:
    center = event_center(event_type)
    return center + rng.normal(0.0, noise, size=center.shape)


@dataclass
class _EventCluster:
    mean: np.ndarray
    count: int

    def update(self, signature: np.ndarray) -> None:
        self.count += 1
        self.mean += (signature - self.mean) / self.count


class LatentEventDiscoverer:
    """Online DP-means-like event discovery.

    The threshold is fixed before the campaign. No ground-truth event label is
    consulted during clustering.
    """

    def __init__(self, *, novelty_threshold: float = 0.95) -> None:
        if novelty_threshold <= 0:
            raise ValueError("novelty_threshold must be positive")
        self.novelty_threshold = novelty_threshold
        self.clusters: list[_EventCluster] = []

    def observe(self, signature: Sequence[float]) -> int:
        x = np.asarray(signature, dtype=float)
        if x.shape != (SIGNATURE_DIM,) or not np.all(np.isfinite(x)):
            raise ValueError("invalid event signature")

        if not self.clusters:
            self.clusters.append(_EventCluster(x.copy(), 1))
            return 0

        distances = [
            float(np.linalg.norm(x - cluster.mean))
            for cluster in self.clusters
        ]
        index = int(np.argmin(distances))
        if distances[index] > self.novelty_threshold:
            self.clusters.append(_EventCluster(x.copy(), 1))
            return len(self.clusters) - 1

        self.clusters[index].update(x)
        return index

    def predict(self, signature: Sequence[float]) -> int:
        if not self.clusters:
            raise ValueError("no event clusters")
        x = np.asarray(signature, dtype=float)
        distances = [
            float(np.linalg.norm(x - cluster.mean))
            for cluster in self.clusters
        ]
        return int(np.argmin(distances))

    def cluster_frequencies(self) -> np.ndarray:
        counts = np.asarray([cluster.count for cluster in self.clusters], dtype=float)
        return counts / counts.sum()


def clustering_metrics(
    truth: Sequence[int],
    clusters: Sequence[int],
    *,
    truth_count: int = EVENT_TYPES,
) -> dict[str, float]:
    if len(truth) != len(clusters) or not truth:
        raise ValueError("truth/clusters must be equally non-empty")

    by_cluster: dict[int, dict[int, int]] = {}
    by_truth: dict[int, set[int]] = {}
    for label, cluster in zip(truth, clusters):
        counts = by_cluster.setdefault(cluster, {})
        counts[label] = counts.get(label, 0) + 1
        by_truth.setdefault(label, set()).add(cluster)

    correct = sum(max(counts.values()) for counts in by_cluster.values())
    purity = correct / len(truth)
    fragmentation = float(
        np.mean(
            [
                len(by_truth.get(label, set()))
                for label in range(truth_count)
            ]
        )
    )

    # Pairwise F1 is label-permutation invariant.
    tp = fp = fn = 0
    for i in range(len(truth)):
        for j in range(i + 1, len(truth)):
            same_truth = truth[i] == truth[j]
            same_cluster = clusters[i] == clusters[j]
            if same_truth and same_cluster:
                tp += 1
            elif not same_truth and same_cluster:
                fp += 1
            elif same_truth and not same_cluster:
                fn += 1
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    pair_f1 = (
        2 * precision * recall / max(precision + recall, 1.0e-15)
    )
    return {
        "purity": purity,
        "fragmentation": fragmentation,
        "pair_f1": pair_f1,
        "clusters": float(len(by_cluster)),
    }
