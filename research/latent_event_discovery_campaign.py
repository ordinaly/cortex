"""Characterize unsupervised latent event discovery."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from latent_event_discovery import (
    EVENT_TYPES,
    LatentEventDiscoverer,
    clustering_metrics,
    event_center,
    sample_signature,
)


EVENT_PROBABILITIES = np.asarray([0.42, 0.25, 0.13, 0.09, 0.07, 0.04])


def run_case(seed: int, noise: float, events: int = 600) -> dict:
    rng = np.random.default_rng(seed)
    discoverer = LatentEventDiscoverer(novelty_threshold=0.95)

    truth = []
    clusters = []
    for _ in range(events):
        label = int(rng.choice(EVENT_TYPES, p=EVENT_PROBABILITIES))
        signature = sample_signature(label, rng, noise=noise)
        cluster = discoverer.observe(signature)
        truth.append(label)
        clusters.append(cluster)

    metrics = clustering_metrics(truth, clusters)

    dominant_truth = {}
    for cluster in sorted(set(clusters)):
        labels = [t for t, c in zip(truth, clusters) if c == cluster]
        counts = np.bincount(labels, minlength=EVENT_TYPES)
        dominant_truth[cluster] = int(np.argmax(counts))

    frequencies = discoverer.cluster_frequencies()
    surprisals = -np.log(frequencies + 1.0e-12)
    cluster_for_center = {
        label: discoverer.predict(event_center(label))
        for label in range(EVENT_TYPES)
    }

    rare_cluster = cluster_for_center[5]
    common_cluster = cluster_for_center[0]
    return {
        "protocol": "latent-event-discovery-v1",
        "seed": seed,
        "noise": noise,
        "events": events,
        **metrics,
        "rare_event_cluster": rare_cluster,
        "common_event_cluster": common_cluster,
        "rare_cluster_surprisal": float(surprisals[rare_cluster]),
        "common_cluster_surprisal": float(surprisals[common_cluster]),
        "rare_more_specific": bool(
            surprisals[rare_cluster] > surprisals[common_cluster]
        ),
        "center_cluster_map": cluster_for_center,
        "dominant_truth": dominant_truth,
    }


def campaign(seeds: int, output: Path) -> None:
    rows = [
        run_case(seed, noise)
        for noise in (0.08, 0.18, 0.30)
        for seed in range(seeds)
    ]
    output.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    for noise in (0.08, 0.18, 0.30):
        subset = [row for row in rows if row["noise"] == noise]
        print(
            json.dumps(
                {
                    "noise": noise,
                    "seeds": seeds,
                    "purity": float(np.mean([r["purity"] for r in subset])),
                    "fragmentation": float(
                        np.mean([r["fragmentation"] for r in subset])
                    ),
                    "pair_f1": float(
                        np.mean([r["pair_f1"] for r in subset])
                    ),
                    "clusters": float(
                        np.mean([r["clusters"] for r in subset])
                    ),
                    "rare_specificity_fraction": float(
                        np.mean(
                            [
                                float(r["rare_more_specific"])
                                for r in subset
                            ]
                        )
                    ),
                },
                sort_keys=True,
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=12)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("latent-event-discovery-v1.jsonl"),
    )
    args = parser.parse_args()
    campaign(args.seeds, args.output)


if __name__ == "__main__":
    main()
