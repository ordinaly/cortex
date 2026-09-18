"""Cortex v1.9-R optimization campaign: reference vs exact cache vs sparse cache."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from fuzzy_predictive_field import FuzzyPredictiveField
from hysteretic_local_resolution import (
    HystereticLocalResolutionController,
)
from hysteretic_local_resolution_campaign import (
    make_anchors,
    observe,
)
from sparse_cached_resolution import (
    CachedLocalResolutionController,
    SparseCachedLocalResolutionController,
)


RESOLUTIONS = (0.03, 0.06, 0.12, 0.25)
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
    *,
    equivalent_cycles: int = 60,
    divergent_cycles: int = 100,
    reconverged_cycles: int = 140,
) -> tuple[np.ndarray, list[tuple[np.ndarray, int, str]]]:
    rng = np.random.default_rng(seed)
    anchors = make_anchors(seed + 701)
    stable = np.asarray([0.95, 0.04, 0.01])
    divergent = np.asarray([0.01, 0.04, 0.95])
    rows: list[tuple[np.ndarray, int, str]] = []

    def add_phase(
        name: str,
        cycles: int,
        *,
        is_divergent: bool,
    ) -> None:
        for _ in range(cycles):
            for identity in rng.permutation(len(anchors)):
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
        equivalent_cycles,
        is_divergent=False,
    )
    add_phase(
        "divergent",
        divergent_cycles,
        is_divergent=True,
    )
    add_phase(
        "reconverged",
        reconverged_cycles,
        is_divergent=False,
    )
    return anchors, rows


def _controller(kind: str, anchors: np.ndarray):
    field = _field(anchors.copy())
    if kind == "reference":
        return HystereticLocalResolutionController(
            field,
            resolutions=RESOLUTIONS,
        )
    if kind == "cached":
        return CachedLocalResolutionController(
            field,
            resolutions=RESOLUTIONS,
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
    complexities = []
    losses = []
    phases = []
    complexity_audits = []
    elapsed = 0.0

    for step, (x, outcome, phase) in enumerate(trace):
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
        complexities.append(decision.complexity)
        losses.append(
            -float(
                np.log(
                    max(
                        float(prediction[outcome]),
                        1.0e-9,
                    )
                )
            )
        )
        phases.append(phase)

        if (
            kind == "sparse"
            and step % 64 == 0
        ):
            exact = controller.exact_complexity()
            complexity_audits.append(
                abs(
                    float(decision.complexity)
                    - exact
                )
            )

    result = {
        "seconds": elapsed,
        "predictions": np.asarray(predictions),
        "resolutions": resolutions,
        "complexities": np.asarray(
            complexities,
            dtype=float,
        ),
        "losses": np.asarray(losses, dtype=float),
        "phases": np.asarray(phases),
        "complexity_audits": complexity_audits,
    }
    if kind == "sparse":
        result["diagnostics"] = (
            controller.optimization_diagnostics()
        )
    return result


def run_seed(seed: int) -> dict:
    anchors, trace = make_trace(seed)
    reference = run_controller(
        "reference",
        anchors,
        trace,
    )
    cached = run_controller(
        "cached",
        anchors,
        trace,
    )
    sparse = run_controller(
        "sparse",
        anchors,
        trace,
    )

    exact_prediction_error = float(
        np.max(
            np.abs(
                reference["predictions"]
                - cached["predictions"]
            )
        )
    )
    exact_resolution_disagreement = float(
        np.mean(
            [
                a != b
                for a, b in zip(
                    reference["resolutions"],
                    cached["resolutions"],
                )
            ]
        )
    )
    exact_complexity_error = float(
        np.max(
            np.abs(
                reference["complexities"]
                - cached["complexities"]
            )
        )
    )

    tv = 0.5 * np.sum(
        np.abs(
            cached["predictions"]
            - sparse["predictions"]
        ),
        axis=1,
    )
    sparse_resolution_disagreement = float(
        np.mean(
            [
                a != b
                for a, b in zip(
                    cached["resolutions"],
                    sparse["resolutions"],
                )
            ]
        )
    )
    divergent = (
        cached["phases"] == "divergent"
    )
    reconverged = (
        sparse["phases"] == "reconverged"
    )
    reconverged_tail_indexes = np.flatnonzero(
        reconverged
    )
    reconverged_tail_indexes = (
        reconverged_tail_indexes[
            len(reconverged_tail_indexes) // 2 :
        ]
    )
    coarsest = max(RESOLUTIONS)
    sparse_recoarsened = float(
        np.mean(
            [
                all(
                    np.isclose(value, coarsest)
                    for value in sparse["resolutions"][index]
                )
                for index in reconverged_tail_indexes
            ]
        )
    )

    sparse_audits = sparse[
        "complexity_audits"
    ]
    return {
        "protocol": "sparse-cached-resolution-v1",
        "seed": seed,
        "trace_steps": len(trace),
        "reference_seconds": reference["seconds"],
        "cached_seconds": cached["seconds"],
        "sparse_seconds": sparse["seconds"],
        "cached_speedup_vs_reference": (
            reference["seconds"]
            / cached["seconds"]
        ),
        "sparse_speedup_vs_reference": (
            reference["seconds"]
            / sparse["seconds"]
        ),
        "sparse_speedup_vs_cached": (
            cached["seconds"]
            / sparse["seconds"]
        ),
        "exact_max_prediction_error": (
            exact_prediction_error
        ),
        "exact_resolution_disagreement": (
            exact_resolution_disagreement
        ),
        "exact_max_complexity_error": (
            exact_complexity_error
        ),
        "sparse_mean_tv": float(np.mean(tv)),
        "sparse_max_tv": float(np.max(tv)),
        "sparse_resolution_disagreement": (
            sparse_resolution_disagreement
        ),
        "cached_divergent_nll": float(
            np.mean(cached["losses"][divergent])
        ),
        "sparse_divergent_nll": float(
            np.mean(sparse["losses"][divergent])
        ),
        "sparse_recoarsened_fraction": (
            sparse_recoarsened
        ),
        "sparse_complexity_audit_mean_abs_error": (
            float(np.mean(sparse_audits))
            if sparse_audits
            else 0.0
        ),
        "sparse_complexity_audit_max_abs_error": (
            float(np.max(sparse_audits))
            if sparse_audits
            else 0.0
        ),
        **{
            "sparse_" + key: value
            for key, value
            in sparse["diagnostics"].items()
        },
    }


def campaign(seeds: int, output: Path) -> None:
    rows = [run_seed(seed) for seed in range(seeds)]
    output.write_text(
        "".join(
            json.dumps(row, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )

    metrics = [
        "cached_speedup_vs_reference",
        "sparse_speedup_vs_reference",
        "sparse_speedup_vs_cached",
        "exact_max_prediction_error",
        "exact_resolution_disagreement",
        "exact_max_complexity_error",
        "sparse_mean_tv",
        "sparse_max_tv",
        "sparse_resolution_disagreement",
        "cached_divergent_nll",
        "sparse_divergent_nll",
        "sparse_recoarsened_fraction",
        "sparse_complexity_audit_mean_abs_error",
        "sparse_complexity_audit_max_abs_error",
        "sparse_anchor_scan_fraction",
        "sparse_svd_fraction",
        "sparse_kernel_row_refresh_fraction",
    ]
    summary = {
        "protocol": "sparse-cached-resolution-v1",
        "seeds": seeds,
    }
    for metric in metrics:
        summary[metric] = float(
            np.mean(
                [row[metric] for row in rows]
            )
        )
    print(json.dumps(summary, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seeds",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "sparse-cached-resolution-v1.jsonl"
        ),
    )
    args = parser.parse_args()
    campaign(args.seeds, args.output)


if __name__ == "__main__":
    main()
