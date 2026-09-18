"""Combined v1.5-R campaign: robust identity + latent events + prediction."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from cortex import HybridCortexRuntime
from dual_geometry_tensor import (
    fit_tensor_bridge,
    rarity_weights,
    relation_surprisal,
)
from latent_event_discovery import (
    LatentEventDiscoverer,
    clustering_metrics,
    event_center,
    sample_signature,
)
from robust_articulation import IdentityToken, RobustArticulator, identity_metrics


FAMILIES = 4
INSTANCES = 3
SOURCE_COUNT = FAMILIES * INSTANCES
TARGET_COUNT = FAMILIES * INSTANCES
ENTITIES = SOURCE_COUNT + TARGET_COUNT
OUTCOMES = 8
EMBEDDING_DIM = 32
BURST = 4


def _normalize(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x)


def centers_for(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    q, _r = np.linalg.qr(rng.normal(size=(EMBEDDING_DIM, ENTITIES)))
    return q[:, :ENTITIES].T.copy()


def side_family(entity: int) -> tuple[int, int]:
    if entity < SOURCE_COUNT:
        return 0, entity // INSTANCES
    local = entity - SOURCE_COUNT
    return 1, local // INSTANCES


def probe_distribution(entity: int) -> np.ndarray:
    side, family = side_family(entity)
    primary = family if side == 0 else FAMILIES + family
    q = np.full(OUTCOMES, 0.03, dtype=float)
    q[side * FAMILIES : (side + 1) * FAMILIES] += 0.02
    q[primary] += 0.68
    return q / q.sum()


def heldout_pairs() -> list[tuple[int, int, int]]:
    rows = []
    for sf in range(FAMILIES):
        source = sf * INSTANCES
        for tf in range(FAMILIES):
            target = SOURCE_COUNT + tf * INSTANCES
            event_type = 2 + ((sf + tf) % FAMILIES)
            rows.append((source, target, event_type))
    return rows


def event_distribution(source: int, target: int) -> np.ndarray:
    source_side, sf = side_family(source)
    target_side, tf = side_family(target)
    if source_side != 0 or target_side != 1:
        raise ValueError("invalid source/target sides")
    rare = 2 + ((sf + tf) % FAMILIES)
    p = np.full(6, 0.015, dtype=float)
    p[0] = 0.64
    p[1] = 0.18
    p[rare] = 0.12
    return p / p.sum()


def noisy_embedding(
    center: np.ndarray,
    rng: np.random.Generator,
    noise: float,
) -> list[float]:
    return _normalize(center + rng.normal(0.0, noise, size=center.shape)).tolist()


class Evidence:
    def __init__(self) -> None:
        self.semantic: list[tuple[object, int]] = []
        self.events: list[tuple[object, object, int]] = []
        self.last: dict[int, object] = {}
        self.identity_records: list[tuple[int, object]] = []


def observe_single_tracklet(
    true_entity: int,
    *,
    centers: np.ndarray,
    rng: np.random.Generator,
    noise: float,
    cortex: HybridCortexRuntime,
    robust: RobustArticulator,
    native: Evidence,
    robust_evidence: Evidence,
) -> tuple[int, IdentityToken]:
    native_last = -1
    robust_last: IdentityToken | None = None
    for _ in range(BURST):
        embedding = noisy_embedding(centers[true_entity], rng, noise)
        native_read = cortex.step_embeddings([embedding])
        native_last = int(native_read.bindings[0])
        robust_last = robust.observe_frame([embedding])[0]
        native.identity_records.append((true_entity, native_last))
        robust_evidence.identity_records.append((true_entity, robust_last))
        native.last[true_entity] = native_last
        robust_evidence.last[true_entity] = robust_last
    assert robust_last is not None
    return native_last, robust_last


def observe_pair_tracklet(
    source: int,
    target: int,
    *,
    centers: np.ndarray,
    rng: np.random.Generator,
    noise: float,
    cortex: HybridCortexRuntime,
    robust: RobustArticulator,
    native: Evidence,
    robust_evidence: Evidence,
) -> tuple[tuple[int, int], tuple[IdentityToken, IdentityToken]]:
    native_by_true = {}
    robust_by_true = {}
    for _ in range(BURST):
        items = [
            (source, noisy_embedding(centers[source], rng, noise)),
            (target, noisy_embedding(centers[target], rng, noise)),
        ]
        rng.shuffle(items)
        native_read = cortex.step_embeddings([x for _truth, x in items])
        robust_read = robust.observe_frame([x for _truth, x in items])
        for (truth, _embedding), native_id, robust_token in zip(
            items,
            native_read.bindings,
            robust_read,
        ):
            native_id = int(native_id)
            native_by_true[truth] = native_id
            robust_by_true[truth] = robust_token
            native.identity_records.append((truth, native_id))
            robust_evidence.identity_records.append((truth, robust_token))
            native.last[truth] = native_id
            robust_evidence.last[truth] = robust_token
    return (
        (native_by_true[source], native_by_true[target]),
        (robust_by_true[source], robust_by_true[target]),
    )


def semantic_matrix(
    records: list[tuple[int, int]],
    max_identity: int,
    *,
    alpha: float = 0.5,
) -> np.ndarray:
    counts = np.zeros((max_identity + 1, OUTCOMES), dtype=float)
    for identity, outcome in records:
        counts[identity, outcome] += 1.0
    probs = (counts + alpha) / (
        counts.sum(axis=1, keepdims=True) + alpha * OUTCOMES
    )
    return np.sqrt(probs)


def build_tensor_prediction(
    semantic_records: list[tuple[int, int]],
    event_records: list[tuple[int, int, int]],
    last_by_true: dict[int, int],
    held: list[tuple[int, int, int]],
    discoverer: LatentEventDiscoverer,
    *,
    shuffle_seed: int,
) -> dict:
    all_ids = [
        identity
        for identity, _outcome in semantic_records
    ] + [
        identity
        for source, target, _cluster in event_records
        for identity in (source, target)
    ] + list(last_by_true.values())
    if not all_ids:
        return {
            "available": False,
            "reason": "no-identities",
            "fit_samples": 0,
            "internal_heldout_collisions": 0,
            "full_hit_at_1": 0.0,
            "interaction_only_hit_at_1": 0.0,
            "shuffled_hit_at_1": 0.0,
        }

    max_identity = max(all_ids)
    semantics = semantic_matrix(semantic_records, max_identity)

    held_internal = {
        (last_by_true[source], last_by_true[target])
        for source, target, _truth in held
        if source in last_by_true and target in last_by_true
    }

    pair_counts: dict[tuple[int, int], np.ndarray] = defaultdict(
        lambda: np.zeros(len(discoverer.clusters), dtype=float)
    )
    pair_n: Counter[tuple[int, int]] = Counter()
    source_counts: dict[int, np.ndarray] = defaultdict(
        lambda: np.zeros(len(discoverer.clusters), dtype=float)
    )
    source_n: Counter[int] = Counter()
    target_counts: dict[int, np.ndarray] = defaultdict(
        lambda: np.zeros(len(discoverer.clusters), dtype=float)
    )
    target_n: Counter[int] = Counter()
    global_counts = np.zeros(len(discoverer.clusters), dtype=float)

    for source, target, cluster in event_records:
        pair_counts[(source, target)][cluster] += 1.0
        pair_n[(source, target)] += 1
        source_counts[source][cluster] += 1.0
        source_n[source] += 1
        target_counts[target][cluster] += 1.0
        target_n[target] += 1
        global_counts[cluster] += 1.0

    samples = []
    for pair, counts in pair_counts.items():
        if pair in held_internal:
            continue
        samples.append(
            (
                pair[0],
                pair[1],
                (counts / max(pair_n[pair], 1)).tolist(),
            )
        )
    if len(samples) < 8:
        return {
            "available": False,
            "reason": "insufficient-distinct-pairs",
            "fit_samples": len(samples),
            "internal_heldout_collisions": 0,
            "full_hit_at_1": 0.0,
            "interaction_only_hit_at_1": 0.0,
            "shuffled_hit_at_1": 0.0,
        }

    bridge = fit_tensor_bridge(semantics, samples, ridge=1.0e-3)
    surprisal = relation_surprisal(global_counts + 1.0)
    eta_fine = float(np.quantile(surprisal, 0.70))
    weights = rarity_weights(surprisal, eta_fine, bandwidth=0.55)

    rng = np.random.default_rng(shuffle_seed)
    shuffled = semantics.copy()
    rng.shuffle(shuffled, axis=0)
    shuffled_bridge = fit_tensor_bridge(shuffled, samples, ridge=1.0e-3)

    full_hits = []
    marginal_hits = []
    shuffled_hits = []
    internal_collisions = 0

    for true_source, true_target, hidden_event in held:
        if true_source not in last_by_true or true_target not in last_by_true:
            continue
        source = last_by_true[true_source]
        target = last_by_true[true_target]
        if pair_n[(source, target)] > 0:
            internal_collisions += 1

        target_cluster = discoverer.predict(event_center(hidden_event))

        full_scores = weights * bridge.predict(
            semantics[source], semantics[target]
        )
        source_rate = source_counts[source] / max(source_n[source], 1)
        target_rate = target_counts[target] / max(target_n[target], 1)
        marginal_scores = weights * 0.5 * (source_rate + target_rate)
        shuffled_scores = weights * shuffled_bridge.predict(
            shuffled[source], shuffled[target]
        )

        full_hits.append(int(int(np.argmax(full_scores)) == target_cluster))
        marginal_hits.append(
            int(int(np.argmax(marginal_scores)) == target_cluster)
        )
        shuffled_hits.append(
            int(int(np.argmax(shuffled_scores)) == target_cluster)
        )

    return {
        "available": bool(full_hits),
        "fit_samples": len(samples),
        "internal_heldout_collisions": internal_collisions,
        "full_hit_at_1": float(np.mean(full_hits)) if full_hits else 0.0,
        "interaction_only_hit_at_1": (
            float(np.mean(marginal_hits)) if marginal_hits else 0.0
        ),
        "shuffled_hit_at_1": (
            float(np.mean(shuffled_hits)) if shuffled_hits else 0.0
        ),
    }


def run_case(
    seed: int,
    *,
    perception_noise: float,
    event_noise: float = 0.08,
    probes_per_entity: int = 30,
    events_per_pair: int = 8,
) -> dict:
    rng = np.random.default_rng(seed)
    centers = centers_for(seed + 31337)
    cortex = HybridCortexRuntime(
        embedding_dim=EMBEDDING_DIM,
        max_entities=128,
        budget=24,
        nuisance_mode="identity-only",
    )
    robust = RobustArticulator(max_entities=128)
    discoverer = LatentEventDiscoverer(novelty_threshold=0.95)

    native = Evidence()
    robust_evidence = Evidence()
    oracle = Evidence()
    for entity in range(ENTITIES):
        oracle.last[entity] = entity

    probe_schedule = [
        entity
        for _ in range(probes_per_entity)
        for entity in range(ENTITIES)
    ]
    rng.shuffle(probe_schedule)
    for entity in probe_schedule:
        native_id, robust_token = observe_single_tracklet(
            entity,
            centers=centers,
            rng=rng,
            noise=perception_noise,
            cortex=cortex,
            robust=robust,
            native=native,
            robust_evidence=robust_evidence,
        )
        outcome = int(rng.choice(OUTCOMES, p=probe_distribution(entity)))
        native.semantic.append((native_id, outcome))
        robust_evidence.semantic.append((robust_token, outcome))
        oracle.semantic.append((entity, outcome))

    held = heldout_pairs()
    held_set = {(source, target) for source, target, _event in held}
    schedule = []
    for source in range(SOURCE_COUNT):
        for target in range(SOURCE_COUNT, ENTITIES):
            if (source, target) in held_set:
                continue
            schedule.extend([(source, target)] * events_per_pair)
    rng.shuffle(schedule)

    event_truth = []
    event_clusters = []
    for source, target in schedule:
        native_pair, robust_pair = observe_pair_tracklet(
            source,
            target,
            centers=centers,
            rng=rng,
            noise=perception_noise,
            cortex=cortex,
            robust=robust,
            native=native,
            robust_evidence=robust_evidence,
        )
        hidden_event = int(
            rng.choice(6, p=event_distribution(source, target))
        )
        signature = sample_signature(hidden_event, rng, noise=event_noise)
        cluster = discoverer.observe(signature)

        event_truth.append(hidden_event)
        event_clusters.append(cluster)
        native.events.append((native_pair[0], native_pair[1], cluster))
        robust_evidence.events.append(
            (robust_pair[0], robust_pair[1], cluster)
        )
        oracle.events.append((source, target, cluster))

    latent_metrics = clustering_metrics(event_truth, event_clusters)

    robust_semantic = []
    for token, outcome in robust_evidence.semantic:
        resolved = robust.resolve(token)
        if resolved is not None:
            robust_semantic.append((resolved, outcome))

    robust_events = []
    for source_token, target_token, cluster in robust_evidence.events:
        source = robust.resolve(source_token)
        target = robust.resolve(target_token)
        if source is not None and target is not None:
            robust_events.append((source, target, cluster))

    robust_last = {
        truth: resolved
        for truth, token in robust_evidence.last.items()
        if (resolved := robust.resolve(token)) is not None
    }

    native_prediction = build_tensor_prediction(
        native.semantic,
        native.events,
        {truth: int(identity) for truth, identity in native.last.items()},
        held,
        discoverer,
        shuffle_seed=seed + 1,
    )
    robust_prediction = build_tensor_prediction(
        robust_semantic,
        robust_events,
        robust_last,
        held,
        discoverer,
        shuffle_seed=seed + 2,
    )
    oracle_prediction = build_tensor_prediction(
        [(int(identity), outcome) for identity, outcome in oracle.semantic],
        [(int(s), int(t), c) for s, t, c in oracle.events],
        {truth: int(identity) for truth, identity in oracle.last.items()},
        held,
        discoverer,
        shuffle_seed=seed + 3,
    )

    native_identity = identity_metrics(
        [(truth, int(identity)) for truth, identity in native.identity_records],
        truth_count=ENTITIES,
    )
    robust_identity = identity_metrics(
        [
            (truth, robust.resolve(token))
            for truth, token in robust_evidence.identity_records
        ],
        truth_count=ENTITIES,
    )

    return {
        "protocol": "robust-latent-world-model-v1",
        "seed": seed,
        "perception_noise": perception_noise,
        "event_noise": event_noise,
        "latent_event": latent_metrics,
        "native_identity": native_identity,
        "robust_identity": robust_identity,
        "native_prediction": native_prediction,
        "robust_prediction": robust_prediction,
        "oracle_prediction": oracle_prediction,
        "robust_reconciliations": robust.reconciliations,
    }


def campaign(seeds: int, output: Path) -> None:
    rows = [
        run_case(seed, perception_noise=noise)
        for noise in (0.12, 0.25)
        for seed in range(seeds)
    ]
    output.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )

    for noise in (0.12, 0.25):
        subset = [row for row in rows if row["perception_noise"] == noise]
        print(
            json.dumps(
                {
                    "perception_noise": noise,
                    "seeds": seeds,
                    "latent_pair_f1": float(
                        np.mean(
                            [row["latent_event"]["pair_f1"] for row in subset]
                        )
                    ),
                    "native_identity_purity": float(
                        np.mean(
                            [row["native_identity"]["purity"] for row in subset]
                        )
                    ),
                    "robust_identity_purity": float(
                        np.mean(
                            [row["robust_identity"]["purity"] for row in subset]
                        )
                    ),
                    "robust_identity_coverage": float(
                        np.mean(
                            [row["robust_identity"]["coverage"] for row in subset]
                        )
                    ),
                    "native_future_hit1": float(
                        np.mean(
                            [
                                row["native_prediction"]["full_hit_at_1"]
                                for row in subset
                            ]
                        )
                    ),
                    "robust_future_hit1": float(
                        np.mean(
                            [
                                row["robust_prediction"]["full_hit_at_1"]
                                for row in subset
                            ]
                        )
                    ),
                    "oracle_future_hit1": float(
                        np.mean(
                            [
                                row["oracle_prediction"]["full_hit_at_1"]
                                for row in subset
                            ]
                        )
                    ),
                },
                sort_keys=True,
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("robust-latent-world-model-v1.jsonl"),
    )
    args = parser.parse_args()
    campaign(args.seeds, args.output)


if __name__ == "__main__":
    main()
