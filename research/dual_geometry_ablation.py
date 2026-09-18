"""Ablation campaign for Cortex v1.3-R dual geometry.

The generator uses hidden semantic-family integers only to create ground truth.
Predictors receive semantic vectors and observed typed interaction rates.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from dual_geometry_tensor import (
    fit_tensor_bridge,
    rarity_weights,
    relation_surprisal,
)


FAMILIES = 4
INSTANCES = 4
RELATIONS = 2 + FAMILIES
NEAR = 0
CONTACT = 1


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / norms


def make_world(
    seed: int,
    *,
    semantic_noise: float = 0.025,
    rate_noise_scale: float = 1.0,
):
    rng = np.random.default_rng(seed)
    if semantic_noise < 0 or rate_noise_scale < 0:
        raise ValueError("noise scales must be non-negative")

    source_semantics = np.repeat(np.eye(FAMILIES), INSTANCES, axis=0)
    target_semantics = np.repeat(np.eye(FAMILIES), INSTANCES, axis=0)
    source_semantics += rng.normal(0.0, semantic_noise, size=source_semantics.shape)
    target_semantics += rng.normal(0.0, semantic_noise, size=target_semantics.shape)
    source_semantics = _normalize_rows(source_semantics)
    target_semantics = _normalize_rows(target_semantics)

    semantics = np.vstack([source_semantics, target_semantics])
    target_offset = len(source_semantics)

    train_samples = []
    heldout = []
    pair_rates = {}

    for sf in range(FAMILIES):
        for tf in range(FAMILIES):
            true_rare = 2 + ((sf + tf) % FAMILIES)
            held_source = sf * INSTANCES
            held_target = target_offset + tf * INSTANCES
            heldout.append((held_source, held_target, true_rare))

            for si in range(INSTANCES):
                source = sf * INSTANCES + si
                for ti in range(INSTANCES):
                    target = target_offset + tf * INSTANCES + ti
                    if source == held_source and target == held_target:
                        continue

                    rates = np.full(RELATIONS, 0.004, dtype=float)
                    rates[NEAR] = 0.65
                    rates[CONTACT] = 0.20
                    rates[true_rare] = 0.08
                    noise = rng.normal(
                        0.0,
                        rate_noise_scale
                        * np.asarray(
                            [0.018, 0.012, 0.004, 0.004, 0.004, 0.004]
                        ),
                    )
                    rates = np.clip(rates + noise, 0.0005, 0.95)
                    pair_rates[(source, target)] = rates
                    train_samples.append((source, target, rates.tolist()))

    global_counts = np.asarray([10000.0, 3000.0, 150.0, 150.0, 150.0, 150.0])
    surprisal = relation_surprisal(global_counts)
    return semantics, train_samples, heldout, pair_rates, surprisal


def marginal_profiles(
    n_entities: int,
    samples,
):
    source_sum = np.zeros((n_entities, RELATIONS), dtype=float)
    source_n = np.zeros(n_entities, dtype=int)
    target_sum = np.zeros((n_entities, RELATIONS), dtype=float)
    target_n = np.zeros(n_entities, dtype=int)

    for source, target, rates in samples:
        values = np.asarray(rates, dtype=float)
        source_sum[source] += values
        source_n[source] += 1
        target_sum[target] += values
        target_n[target] += 1

    return (
        source_sum / np.maximum(source_n[:, None], 1),
        target_sum / np.maximum(target_n[:, None], 1),
    )


def _rank_metrics(scores: np.ndarray, truth: int):
    true_score = float(scores[truth])
    greater = int(np.sum(scores > true_score + 1.0e-15))
    tied = int(np.sum(np.abs(scores - true_score) <= 1.0e-15))
    ranks = np.arange(greater + 1, greater + tied + 1)
    mrr = float(np.mean(1.0 / ranks))
    hit1 = float(np.mean(ranks <= 1))

    rare_scores = scores[2:]
    truth_rare = truth - 2
    wrong = np.delete(rare_scores, truth_rare)
    rare_margin = true_score - float(np.max(wrong))
    return {
        "hit_at_1": hit1,
        "mrr": mrr,
        "rare_margin": rare_margin,
    }


def run_seed(seed: int) -> dict:
    semantics, samples, heldout, _pair_rates, surprisal = make_world(seed)
    bridge = fit_tensor_bridge(semantics, samples, ridge=1.0e-3)

    n_entities = semantics.shape[0]
    source_marginal, target_marginal = marginal_profiles(n_entities, samples)

    rng = np.random.default_rng(seed + 99991)
    permutation = rng.permutation(n_entities)
    shuffled_bridge = fit_tensor_bridge(
        semantics[permutation],
        samples,
        ridge=1.0e-3,
    )

    eta_common = float(surprisal[NEAR])
    eta_rare = float(surprisal[2])
    common_weights = rarity_weights(surprisal, eta_common, bandwidth=0.35)
    rare_weights = rarity_weights(surprisal, eta_rare, bandwidth=0.35)

    methods = {
        "semantic-only": [],
        "interaction-marginal-only": [],
        "tensor-no-resolution": [],
        "full-dual-geometry": [],
        "shuffled-semantic-control": [],
    }

    coarse_correct = 0
    fine_correct = 0

    for source, target, truth in heldout:
        # Semantic proximity can tell us that the pair is structurally
        # compatible, but by itself supplies no typed-relation map.
        semantic_scores = np.zeros(RELATIONS, dtype=float)
        semantic_scores[2:] = 1.0

        marginal_rates = 0.5 * (
            source_marginal[source] + target_marginal[target]
        )
        interaction_scores = rare_weights * marginal_rates

        tensor_rates = bridge.predict(semantics[source], semantics[target])
        no_resolution_scores = tensor_rates
        full_scores = rare_weights * tensor_rates

        shuffled_rates = shuffled_bridge.predict(
            semantics[permutation][source],
            semantics[permutation][target],
        )
        shuffled_scores = rare_weights * shuffled_rates

        for name, scores in [
            ("semantic-only", semantic_scores),
            ("interaction-marginal-only", interaction_scores),
            ("tensor-no-resolution", no_resolution_scores),
            ("full-dual-geometry", full_scores),
            ("shuffled-semantic-control", shuffled_scores),
        ]:
            methods[name].append(_rank_metrics(scores, truth))

        coarse = common_weights * tensor_rates
        fine = full_scores
        coarse_correct += int(int(np.argmax(coarse)) == NEAR)
        fine_correct += int(int(np.argmax(fine)) == truth)

    summary = {}
    for name, rows in methods.items():
        summary[name] = {
            key: float(np.mean([row[key] for row in rows]))
            for key in ("hit_at_1", "mrr", "rare_margin")
        }

    return {
        "protocol": "dual-geometry-ablation-v1",
        "seed": seed,
        "families": FAMILIES,
        "instances_per_family": INSTANCES,
        "relations": RELATIONS,
        "heldout_pairs": len(heldout),
        "heldout_pair_leakage": any(
            source == hs and target == ht
            for source, target, _rates in samples
            for hs, ht, _truth in heldout
        ),
        "eta_common": eta_common,
        "eta_rare": eta_rare,
        "methods": summary,
        "coarse_near_accuracy": coarse_correct / len(heldout),
        "fine_rare_accuracy": fine_correct / len(heldout),
    }


def campaign(seeds: int, output: Path) -> None:
    rows = [run_seed(seed) for seed in range(seeds)]
    output.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )

    for method in rows[0]["methods"]:
        print(
            json.dumps(
                {
                    "method": method,
                    "seeds": seeds,
                    **{
                        metric: float(
                            np.mean(
                                [
                                    row["methods"][method][metric]
                                    for row in rows
                                ]
                            )
                        )
                        for metric in ("hit_at_1", "mrr", "rare_margin")
                    },
                },
                sort_keys=True,
            )
        )

    print(
        json.dumps(
            {
                "coarse_near_accuracy": float(
                    np.mean([row["coarse_near_accuracy"] for row in rows])
                ),
                "fine_rare_accuracy": float(
                    np.mean([row["fine_rare_accuracy"] for row in rows])
                ),
                "heldout_pair_leakage": any(
                    row["heldout_pair_leakage"] for row in rows
                ),
                "seeds": seeds,
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
        default=Path("dual-geometry-ablation-v1.jsonl"),
    )
    args = parser.parse_args()
    campaign(args.seeds, args.output)


if __name__ == "__main__":
    main()
