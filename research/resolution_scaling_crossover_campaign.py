"""Cortex v1.10-R resolution scaling/crossover campaign.

This campaign does not introduce a new adaptive-resolution rule. It profiles
three already-defined v1.9-R execution modes over increasing anchor counts:

1. exact-svd: exact cached kernels + exact per-observation local complexity SVD;
2. dense-sampled: exact dense cache refresh + full hysteresis scan + sampled SVD;
3. sparse: drift-triggered kernel refresh + sparse scans + sampled SVD.

The goal is to locate empirical cost crossovers before promoting additional
sparse machinery into the runtime.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from fuzzy_predictive_field import FuzzyPredictiveField
from hysteretic_local_resolution_campaign import make_anchors, observe
from sparse_cached_resolution import (
    CachedLocalResolutionController,
    SparseCachedLocalResolutionController,
)


RESOLUTIONS = (0.03, 0.06, 0.12, 0.25)
ANCHOR_CASES = (9, 18, 36, 72, 144)
OUTCOMES = 3


def _field(anchors: np.ndarray) -> FuzzyPredictiveField:
    return FuzzyPredictiveField(
        anchors,
        outcomes=OUTCOMES,
        relations=3,
        perceptual_temperature=0.03,
        semantic_resolution=0.04,
        evidence_decay=0.998,
    )


def make_trace(
    seed: int,
    anchors: np.ndarray,
    *,
    equivalent_steps: int = 90,
    divergent_steps: int = 135,
    reconverged_steps: int = 135,
) -> list[tuple[np.ndarray, int, str]]:
    rng = np.random.default_rng(seed)
    stable = np.asarray([0.95, 0.04, 0.01])
    divergent = np.asarray([0.01, 0.04, 0.95])
    rows: list[tuple[np.ndarray, int, str]] = []

    def add_phase(
        name: str,
        steps: int,
        *,
        is_divergent: bool,
    ) -> None:
        for _ in range(steps):
            identity = int(
                rng.integers(0, len(anchors))
            )
            x = observe(
                anchors[identity],
                rng,
                noise=0.20,
            )
            distribution = (
                divergent
                if is_divergent and identity == 0
                else stable
            )
            outcome = int(
                rng.choice(
                    OUTCOMES,
                    p=distribution,
                )
            )
            rows.append((x, outcome, name))

    add_phase(
        "equivalent",
        equivalent_steps,
        is_divergent=False,
    )
    add_phase(
        "divergent",
        divergent_steps,
        is_divergent=True,
    )
    add_phase(
        "reconverged",
        reconverged_steps,
        is_divergent=False,
    )
    return rows


def _controller(
    kind: str,
    anchors: np.ndarray,
):
    field = _field(anchors.copy())

    if kind == "exact-svd":
        return CachedLocalResolutionController(
            field,
            resolutions=RESOLUTIONS,
        )

    if kind == "dense-sampled":
        return SparseCachedLocalResolutionController(
            field,
            resolutions=RESOLUTIONS,
            feature_tolerance=0.0,
            loss_change_tolerance=0.0,
            full_scan_interval=1,
            complexity_refresh_interval=32,
        )

    if kind == "sparse":
        return SparseCachedLocalResolutionController(
            field,
            resolutions=RESOLUTIONS,
            feature_tolerance=0.0025,
            loss_change_tolerance=0.001,
            full_scan_interval=64,
            complexity_refresh_interval=32,
        )

    raise ValueError(kind)


def run_controller(
    kind: str,
    anchors: np.ndarray,
    trace: list[tuple[np.ndarray, int, str]],
) -> dict:
    controller = _controller(kind, anchors)
    predictions = []
    resolutions = []
    elapsed = 0.0

    for x, outcome, _phase in trace:
        start = perf_counter()
        candidates = controller.candidate_predictions(x)
        prediction, decision = controller.predict_outcome(x)
        controller.observe_outcome(
            x,
            outcome,
            candidate_predictions=candidates,
        )
        elapsed += perf_counter() - start

        predictions.append(prediction.copy())
        resolutions.append(decision.resolutions)

    result = {
        "seconds": elapsed,
        "predictions": np.asarray(
            predictions,
            dtype=float,
        ),
        "resolutions": resolutions,
    }

    if isinstance(
        controller,
        SparseCachedLocalResolutionController,
    ):
        result["diagnostics"] = (
            controller.optimization_diagnostics()
        )

    return result


def _resolution_disagreement(
    left: list[tuple[float, ...]],
    right: list[tuple[float, ...]],
) -> float:
    return float(
        np.mean(
            [
                a != b
                for a, b in zip(left, right)
            ]
        )
    )


def run_case(seed: int, anchors_count: int) -> dict:
    anchors = make_anchors(
        seed + 1009 + anchors_count,
        anchors=anchors_count,
        individual_scale=0.35,
    )
    trace = make_trace(
        seed + 2003,
        anchors,
    )

    dense = run_controller(
        "dense-sampled",
        anchors,
        trace,
    )
    sparse = run_controller(
        "sparse",
        anchors,
        trace,
    )

    exact = None
    if anchors_count <= 36:
        exact = run_controller(
            "exact-svd",
            anchors,
            trace,
        )

    tv = 0.5 * np.sum(
        np.abs(
            dense["predictions"]
            - sparse["predictions"]
        ),
        axis=1,
    )

    row = {
        "protocol": "resolution-scaling-crossover-v1",
        "seed": seed,
        "anchors": anchors_count,
        "steps": len(trace),
        "dense_sampled_us_per_step": (
            1.0e6 * dense["seconds"] / len(trace)
        ),
        "sparse_us_per_step": (
            1.0e6 * sparse["seconds"] / len(trace)
        ),
        "sparse_speedup_vs_dense_sampled": (
            dense["seconds"] / sparse["seconds"]
        ),
        "sparse_mean_tv": float(np.mean(tv)),
        "sparse_max_tv": float(np.max(tv)),
        "sparse_resolution_disagreement": (
            _resolution_disagreement(
                dense["resolutions"],
                sparse["resolutions"],
            )
        ),
        **{
            "sparse_" + key: value
            for key, value
            in sparse["diagnostics"].items()
        },
        **{
            "dense_" + key: value
            for key, value
            in dense["diagnostics"].items()
        },
    }

    if exact is not None:
        exact_error = np.abs(
            exact["predictions"]
            - dense["predictions"]
        )
        row.update(
            {
                "exact_svd_us_per_step": (
                    1.0e6
                    * exact["seconds"]
                    / len(trace)
                ),
                "sampled_speedup_vs_exact_svd": (
                    exact["seconds"]
                    / dense["seconds"]
                ),
                "dense_exact_max_prediction_error": (
                    float(np.max(exact_error))
                ),
                "dense_exact_resolution_disagreement": (
                    _resolution_disagreement(
                        exact["resolutions"],
                        dense["resolutions"],
                    )
                ),
            }
        )
    else:
        row.update(
            {
                "exact_svd_us_per_step": None,
                "sampled_speedup_vs_exact_svd": None,
                "dense_exact_max_prediction_error": None,
                "dense_exact_resolution_disagreement": None,
            }
        )

    return row


def campaign(
    seeds: int,
    output: Path,
) -> None:
    rows = [
        run_case(seed, anchors)
        for anchors in ANCHOR_CASES
        for seed in range(seeds)
    ]
    output.write_text(
        "".join(
            json.dumps(row, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )

    summary = {
        "protocol": "resolution-scaling-crossover-v1",
        "seeds": seeds,
        "cases": {},
    }

    for anchors in ANCHOR_CASES:
        selected = [
            row
            for row in rows
            if row["anchors"] == anchors
        ]

        def median(key: str):
            values = [
                row[key]
                for row in selected
                if row[key] is not None
            ]
            if not values:
                return None
            return float(np.median(values))

        summary["cases"][str(anchors)] = {
            "dense_sampled_us_per_step": median(
                "dense_sampled_us_per_step"
            ),
            "sparse_us_per_step": median(
                "sparse_us_per_step"
            ),
            "sparse_speedup_vs_dense_sampled": median(
                "sparse_speedup_vs_dense_sampled"
            ),
            "sparse_mean_tv": median(
                "sparse_mean_tv"
            ),
            "sparse_max_tv": median(
                "sparse_max_tv"
            ),
            "sparse_resolution_disagreement": median(
                "sparse_resolution_disagreement"
            ),
            "sparse_kernel_row_refresh_fraction": median(
                "sparse_kernel_row_refresh_fraction"
            ),
            "sparse_anchor_scan_fraction": median(
                "sparse_anchor_scan_fraction"
            ),
            "sparse_svd_fraction": median(
                "sparse_svd_fraction"
            ),
            "exact_svd_us_per_step": median(
                "exact_svd_us_per_step"
            ),
            "sampled_speedup_vs_exact_svd": median(
                "sampled_speedup_vs_exact_svd"
            ),
        }

    crossover = None
    for anchors in ANCHOR_CASES:
        speedup = summary["cases"][str(anchors)][
            "sparse_speedup_vs_dense_sampled"
        ]
        if speedup is not None and speedup >= 1.10:
            crossover = anchors
            break

    summary["sparse_1_10x_crossover_anchors"] = (
        crossover
    )
    print(json.dumps(summary, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seeds",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "resolution-scaling-crossover-v1.jsonl"
        ),
    )
    args = parser.parse_args()
    campaign(args.seeds, args.output)


if __name__ == "__main__":
    main()
