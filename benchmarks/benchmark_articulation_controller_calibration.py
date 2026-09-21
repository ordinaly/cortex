#!/usr/bin/env python3
"""Calibrate and validate learned articulation-controller decision surfaces."""
from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import json
import math
from pathlib import Path
from random import Random
import statistics
import time

from cortex import CortexRuntime
from cortex.articulation_controller import (
    ArticulationController,
    ArticulationState,
    ExecutionBackend,
    profile_rules,
)
from cortex.controller_calibration import (
    BACKEND_FEATURE_NAMES,
    EXECUTE_FEATURE_NAMES,
    UNRESOLVED_FEATURE_NAMES,
    CalibrationBundle,
    CalibratedArticulationController,
    SurfaceObservation,
    backend_features,
    epistemic_execute_features,
    epistemic_unresolved_features,
    fit_logistic_surface,
    select_f1_threshold,
    select_precision_threshold,
)
from cortex.hypergraph_geometry import HypergraphGeometry, RelationalHyperedge
from cortex.hypergraph_inference import (
    CompositionRule,
    HypergraphCompositionLearner,
    RelationStep,
)
from cortex.hypergraph_lpu import EnumeratingRuleExecutor, LPURuleExecutor
from cortex.knowledge import KnowledgeGraphView

import benchmark_hypergraph_lpu_stress as stress


PROTOCOL = "articulation-controller-calibration-v0.2"
SEED = 982451653


@dataclass(frozen=True)
class EpistemicEpisode:
    execute_features: tuple[float, ...]
    unresolved_features: tuple[float, ...]
    label: str


@dataclass(frozen=True)
class BackendEpisode:
    features: tuple[float, ...]
    label_lpu: bool
    cortex_us: float
    lpu_us: float
    depth: int
    width: int
    paths: int
    edges: int


def _binary_entropy(p: float) -> float:
    p = min(1.0, max(0.0, p))
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -(
        p * math.log(p)
        + (1.0 - p) * math.log(1.0 - p)
    ) / math.log(2.0)


def _wilson(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    if trials <= 0:
        return 0.0, 1.0
    p = successes / trials
    z2 = z * z
    denominator = 1.0 + z2 / trials
    center = (p + z2 / (2.0 * trials)) / denominator
    spread = (
        z
        * math.sqrt(
            (p * (1.0 - p) / trials)
            + z2 / (4.0 * trials * trials)
        )
        / denominator
    )
    return max(0.0, center - spread), min(1.0, center + spread)


def _bernoulli_count(rng: Random, probability: float, n: int) -> int:
    return sum(rng.random() < probability for _ in range(n))


def generate_epistemic_episodes(
    count: int,
    *,
    seed: int,
) -> tuple[EpistemicEpisode, ...]:
    """Generate a stratified calibration set from a future-evidence oracle.

    Five regimes are sampled deliberately so calibration learns the decision
    boundary rather than the ambient class frequency:

    - mature stable structure;
    - stable structure with sparse current evidence;
    - persistent relational conflict;
    - transient source ambiguity;
    - weak structural support.

    EXECUTE means that the latent/future regime has a dominant relation and
    current evidence already supports articulation. EXPLORE means the future
    regime is usable but current evidence is still sparse/ambiguous. UNRESOLVED
    means mature evidence excludes the declared safe articulation region.
    """

    rng = Random(seed)
    rows = []
    for index in range(count):
        regime = index % 5

        if regime == 0:  # mature stable
            latent_relation = 0.76 + 0.23 * rng.random()
            latent_source = 0.86 + 0.13 * rng.random()
            latent_support = 0.62 + 0.34 * rng.random()
            relation_trials = rng.randint(9, 28)
            source_trials = rng.randint(9, 20)
        elif regime == 1:  # stable but sparse
            latent_relation = 0.76 + 0.23 * rng.random()
            latent_source = 0.86 + 0.13 * rng.random()
            latent_support = 0.62 + 0.34 * rng.random()
            relation_trials = rng.randint(2, 7)
            source_trials = rng.randint(2, 7)
        elif regime == 2:  # persistent relation conflict
            latent_relation = 0.50 + 0.19 * rng.random()
            latent_source = 0.86 + 0.13 * rng.random()
            latent_support = 0.62 + 0.34 * rng.random()
            relation_trials = rng.randint(10, 28)
            source_trials = rng.randint(9, 20)
        elif regime == 3:  # source ambiguity that more context can resolve
            latent_relation = 0.84 + 0.15 * rng.random()
            latent_source = 0.86 + 0.13 * rng.random()
            latent_support = 0.62 + 0.34 * rng.random()
            relation_trials = rng.randint(9, 24)
            source_trials = rng.randint(2, 5)
        else:  # structurally weak regime
            latent_relation = 0.84 + 0.15 * rng.random()
            latent_source = 0.86 + 0.13 * rng.random()
            latent_support = 0.34 + 0.20 * rng.random()
            relation_trials = rng.randint(10, 28)
            source_trials = rng.randint(9, 20)

        relation_success = _bernoulli_count(
            rng,
            latent_relation,
            relation_trials,
        )
        source_success = _bernoulli_count(
            rng,
            latent_source,
            source_trials,
        )

        confidence = relation_success / relation_trials
        competitor = 1.0 - confidence
        margin = confidence - competitor
        source_support = source_success / source_trials
        source_entropy = _binary_entropy(source_support)
        observed_support = min(
            1.0,
            max(
                0.0,
                latent_support + rng.gauss(0.0, 0.05),
            ),
        )

        # One-sigma Wilson bounds are appropriate here because EXECUTE does not
        # mean "assert the premise is certainly true"; it means "formalize this
        # sufficiently dominant articulation and preserve its confidence".
        relation_lb, relation_ub = _wilson(
            relation_success,
            relation_trials,
            z=1.0,
        )
        source_lb, source_ub = _wilson(
            source_success,
            source_trials,
            z=1.0,
        )

        future_relation = _bernoulli_count(
            rng,
            latent_relation,
            256,
        ) / 256.0
        future_source = _bernoulli_count(
            rng,
            latent_source,
            192,
        ) / 192.0

        future_safe = (
            future_relation >= 0.75
            and future_source >= 0.80
            and latent_support >= 0.55
        )
        current_resolved = (
            relation_lb >= 0.60
            and source_lb >= 0.60
            and observed_support >= 0.50
        )
        safe_region_excluded = (
            relation_ub < 0.75
            or source_ub < 0.80
            or (
                latent_support < 0.55
                and relation_trials >= 10
            )
        )

        if future_safe and current_resolved:
            label = "execute"
        elif (not future_safe) and safe_region_excluded:
            label = "unresolved"
        else:
            label = "explore"

        rows.append(
            EpistemicEpisode(
                execute_features=epistemic_execute_features(
                    total_examples=relation_trials,
                    confidence=confidence,
                    margin=margin,
                    mean_support=observed_support,
                    source_support=source_support,
                    source_entropy=source_entropy,
                ),
                unresolved_features=epistemic_unresolved_features(
                    total_examples=relation_trials,
                    confidence=confidence,
                    competitor_confidence=competitor,
                    margin=margin,
                    mean_support=observed_support,
                    source_entropy=source_entropy,
                ),
                label=label,
            )
        )
    return tuple(rows)

def _split(rows, train_fraction=0.60, val_fraction=0.20):
    n = len(rows)
    train_end = int(n * train_fraction)
    val_end = train_end + int(n * val_fraction)
    return rows[:train_end], rows[train_end:val_end], rows[val_end:]


def _classification_metrics(surface, rows, positive_label, feature_attr):
    tp = fp = tn = fn = 0
    brier = 0.0
    for row in rows:
        features = getattr(row, feature_attr)
        truth = row.label == positive_label
        probability = surface.probability(features)
        prediction = probability + 1.0e-12 >= surface.threshold
        brier += (probability - float(truth)) ** 2
        if truth and prediction:
            tp += 1
        elif truth:
            fn += 1
        elif prediction:
            fp += 1
        else:
            tn += 1
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    accuracy = (tp + tn) / max(1, tp + tn + fp + fn)
    return {
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
        "brier": brier / max(1, len(rows)),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def calibrate_epistemic():
    rows = list(generate_epistemic_episodes(6000, seed=SEED))
    Random(SEED + 1).shuffle(rows)
    train, validation, test = _split(rows)

    execute_train = tuple(
        SurfaceObservation(
            row.execute_features,
            row.label == "execute",
        )
        for row in train
    )
    execute_validation = tuple(
        SurfaceObservation(
            row.execute_features,
            row.label == "execute",
        )
        for row in validation
    )
    execute = fit_logistic_surface(
        execute_train,
        EXECUTE_FEATURE_NAMES,
        epochs=1250,
    )
    execute = select_precision_threshold(
        execute,
        execute_validation,
        minimum_precision=0.95,
    )

    non_execute_train = [row for row in train if row.label != "execute"]
    non_execute_validation = [
        row for row in validation if row.label != "execute"
    ]
    unresolved_train = tuple(
        SurfaceObservation(
            row.unresolved_features,
            row.label == "unresolved",
        )
        for row in non_execute_train
    )
    unresolved_validation = tuple(
        SurfaceObservation(
            row.unresolved_features,
            row.label == "unresolved",
        )
        for row in non_execute_validation
    )
    unresolved = fit_logistic_surface(
        unresolved_train,
        UNRESOLVED_FEATURE_NAMES,
        epochs=1250,
    )
    unresolved = select_f1_threshold(
        unresolved,
        unresolved_validation,
    )

    execute_metrics = _classification_metrics(
        execute,
        test,
        "execute",
        "execute_features",
    )
    unresolved_metrics = _classification_metrics(
        unresolved,
        [row for row in test if row.label != "execute"],
        "unresolved",
        "unresolved_features",
    )

    # End-to-end 3-state classification on held-out episodes.
    confusion = {
        truth: {prediction: 0 for prediction in ("execute", "explore", "unresolved")}
        for truth in ("execute", "explore", "unresolved")
    }
    for row in test:
        if execute.decide(row.execute_features):
            prediction = "execute"
        elif unresolved.decide(row.unresolved_features):
            prediction = "unresolved"
        else:
            prediction = "explore"
        confusion[row.label][prediction] += 1

    correct = sum(confusion[state][state] for state in confusion)
    total = sum(sum(row.values()) for row in confusion.values())
    dangerous_execute = (
        confusion["explore"]["execute"]
        + confusion["unresolved"]["execute"]
    )

    return (
        execute,
        unresolved,
        {
            "train": len(train),
            "validation": len(validation),
            "test": len(test),
            "label_counts": {
                label: sum(row.label == label for row in rows)
                for label in ("execute", "explore", "unresolved")
            },
            "execute_test": execute_metrics,
            "unresolved_test": unresolved_metrics,
            "three_state_accuracy": correct / max(1, total),
            "dangerous_false_execute_rate": dangerous_execute / max(1, total),
            "confusion": confusion,
        },
    )


def _lattice(depth: int, width: int):
    layers = [("A",)]
    layers.extend(
        tuple(f"L{level}-{column}" for column in range(width))
        for level in range(1, depth)
    )
    layers.append(("Z",))

    edges = []
    signature = []
    serial = 0
    for level in range(depth):
        relation = f"R{level}"
        signature.append(RelationStep(relation, "forward"))
        for left in layers[level]:
            for right in layers[level + 1]:
                edges.append(
                    RelationalHyperedge(
                        f"e{serial}",
                        serial,
                        100 + level,
                        serial + 1,
                        left,
                        relation,
                        right,
                        0.8,
                    )
                )
                serial += 1

    geometry = HypergraphGeometry(edges)
    rule = CompositionRule(
        signature=tuple(signature),
        target_relation="RT",
        examples=10,
        confidence=1.0,
        mean_support=0.8,
    )
    profile = profile_rules(
        geometry,
        "A",
        (rule,),
        resolution=0.5,
    )
    return geometry, rule, profile


def _median_us(fn, repeats=5):
    # warm-up
    fn()
    values = []
    result = None
    for _ in range(repeats):
        start = time.perf_counter_ns()
        result = fn()
        values.append((time.perf_counter_ns() - start) / 1000.0)
    return result, statistics.median(values)


def generate_backend_episodes():
    rows = []
    for depth in range(2, 7):
        for width in range(2, 10):
            geometry, rule, profile = _lattice(depth, width)
            cortex = EnumeratingRuleExecutor(geometry, resolution=0.5)
            lpu = LPURuleExecutor(geometry, resolution=0.5)
            _read_c, cortex_us = _median_us(
                lambda: cortex.infer("A", "RT", (rule,)),
            )
            _read_l, lpu_us = _median_us(
                lambda: lpu.infer("A", "RT", (rule,)),
            )
            # Require a modest measured advantage to choose the coprocessor.
            label_lpu = lpu_us <= cortex_us * 0.95
            rows.append(
                BackendEpisode(
                    features=backend_features(
                        path_multiplicity=profile.path_multiplicity,
                        materialized_edges=profile.materialized_edges,
                    ),
                    label_lpu=label_lpu,
                    cortex_us=cortex_us,
                    lpu_us=lpu_us,
                    depth=depth,
                    width=width,
                    paths=profile.path_multiplicity,
                    edges=profile.materialized_edges,
                )
            )
    Random(SEED + 2).shuffle(rows)
    return tuple(rows)


def _backend_regret(surface, rows):
    regret = []
    correct = 0
    for row in rows:
        choose_lpu = surface.decide(row.features)
        chosen = row.lpu_us if choose_lpu else row.cortex_us
        optimal = min(row.lpu_us, row.cortex_us)
        regret.append(chosen / max(1.0e-12, optimal))
        correct += choose_lpu == row.label_lpu
    return {
        "accuracy": correct / max(1, len(rows)),
        "mean_runtime_regret": statistics.mean(regret),
        "median_runtime_regret": statistics.median(regret),
        "max_runtime_regret": max(regret),
    }


def calibrate_backend():
    rows = list(generate_backend_episodes())
    train, validation, test = _split(rows, 0.60, 0.20)

    surface = fit_logistic_surface(
        tuple(
            SurfaceObservation(row.features, row.label_lpu)
            for row in train
        ),
        BACKEND_FEATURE_NAMES,
        epochs=1600,
    )

    # Learn the probability threshold by minimizing measured runtime regret on
    # validation, not classification error.
    candidate_thresholds = sorted(
        {
            0.0,
            0.5,
            1.0,
            *(
                surface.probability(row.features)
                for row in validation
            ),
        }
    )
    best = None
    for threshold in candidate_thresholds:
        candidate = surface.with_threshold(threshold)
        metrics = _backend_regret(candidate, validation)
        key = (
            metrics["mean_runtime_regret"],
            metrics["max_runtime_regret"],
            threshold,
        )
        if best is None or key < best[0]:
            best = (key, candidate)
    surface = best[1]

    return (
        surface,
        {
            "cases": len(rows),
            "train": len(train),
            "validation": len(validation),
            "test": len(test),
            "lpu_wins": sum(row.label_lpu for row in rows),
            "validation": _backend_regret(surface, validation),
            "test": _backend_regret(surface, test),
            "test_cases": [
                {
                    "depth": row.depth,
                    "width": row.width,
                    "paths": row.paths,
                    "edges": row.edges,
                    "cortex_us": row.cortex_us,
                    "lpu_us": row.lpu_us,
                    "oracle": "lpu" if row.label_lpu else "cortex",
                    "chosen": (
                        "lpu"
                        if surface.decide(row.features)
                        else "cortex"
                    ),
                }
                for row in test
            ],
        },
    )


def _permissive_rules(training, depth):
    return HypergraphCompositionLearner(
        training,
        path_length=depth,
        minimum_examples=1,
        confidence_threshold=0.0,
        resolution=0.5,
    ).fit()


def validate_native(bundle: CalibrationBundle, seeds: int = 10):
    clean = []
    conflict20 = []
    conflict40 = []
    ambiguous = []

    for offset in range(seeds):
        seed = 1000 + offset
        for depth in range(2, 6):
            world = stress.make_world(
                seed=seed,
                depth=depth,
                width=8,
                contradiction_fraction=0.0,
            )
            runtime, report, training, full = stress._build(world)
            rules = _permissive_rules(training, depth)
            view = KnowledgeGraphView(
                report.snapshot_json,
                report.symbol_bindings,
                runtime.config_json(),
                report.statement_incidence,
            )
            local = view.local_articulation(world.source, resolution=0.5)
            before = runtime.snapshot_json()
            read = CalibratedArticulationController(
                full,
                bundle,
            ).execute(
                world.source,
                world.target_relation,
                rules,
                source_support=local.topology.top_entity.support,
                source_entropy=local.topology.entity_entropy,
            )
            after = runtime.snapshot_json()
            targets = (
                {row.target for row in read.execution.candidates}
                if read.execution is not None
                else set()
            )
            clean.append(
                {
                    "seed": seed,
                    "depth": depth,
                    "state": read.decision.state.value,
                    "backend": (
                        read.decision.backend.value
                        if read.decision.backend is not None
                        else None
                    ),
                    "correct": targets == {world.target},
                    "memory_unchanged": before == after,
                    "binding_conflicts": report.binding_conflicts,
                    "resolution": read.decision.resolution,
                    "execute_probability": (
                        read.decision.assessments[0].execute_probability
                        if read.decision.assessments
                        else 0.0
                    ),
                }
            )

        for fraction, output in ((0.2, conflict20), (0.4, conflict40)):
            world = stress.make_world(
                seed=seed,
                depth=3,
                width=8,
                contradiction_fraction=fraction,
            )
            runtime, report, training, full = stress._build(world)
            rules = _permissive_rules(training, 3)
            decision = CalibratedArticulationController(
                full,
                bundle,
            ).decide(
                world.source,
                world.target_relation,
                rules,
            )
            output.append(
                {
                    "state": decision.state.value,
                    "execute_probability": (
                        decision.assessments[0].execute_probability
                        if decision.assessments else 0.0
                    ),
                    "unresolved_probability": (
                        decision.assessments[0].unresolved_probability
                        if decision.assessments else 0.0
                    ),
                }
            )

        world = stress.make_world(
            seed=seed,
            depth=2,
            width=8,
            contradiction_fraction=0.0,
        )
        _runtime, _report, training, full = stress._build(world)
        rules = _permissive_rules(training, 2)
        decision = CalibratedArticulationController(
            full,
            bundle,
        ).decide(
            world.source,
            world.target_relation,
            rules,
            source_support=0.55,
            source_entropy=0.70,
        )
        ambiguous.append(decision.state.value)

    return {
        "clean_cases": len(clean),
        "clean_execute_rate": sum(
            row["state"] == "execute" for row in clean
        ) / len(clean),
        "clean_accuracy": sum(row["correct"] for row in clean) / len(clean),
        "memory_isolation_rate": sum(
            row["memory_unchanged"] for row in clean
        ) / len(clean),
        "zero_binding_conflicts": all(
            row["binding_conflicts"] == 0 for row in clean
        ),
        "backend_counts": {
            backend: sum(row["backend"] == backend for row in clean)
            for backend in ("cortex", "lpu")
        },
        "resolution_counts": {
            str(value): sum(row["resolution"] == value for row in clean)
            for value in sorted(
                {row["resolution"] for row in clean if row["resolution"] is not None}
            )
        },
        "noise20_execute_rate": sum(
            row["state"] == "execute" for row in conflict20
        ) / len(conflict20),
        "noise40_execute_rate": sum(
            row["state"] == "execute" for row in conflict40
        ) / len(conflict40),
        "noise40_unresolved_rate": sum(
            row["state"] == "unresolved" for row in conflict40
        ) / len(conflict40),
        "ambiguous_source_explore_rate": sum(
            state == "explore" for state in ambiguous
        ) / len(ambiguous),
        "clean": clean,
        "noise20": conflict20,
        "noise40": conflict40,
    }


def surface_payload(surface):
    return {
        "features": list(surface.feature_names),
        "weights": list(surface.weights),
        "intercept": surface.intercept,
        "threshold": surface.threshold,
    }


def campaign(native_seeds: int):
    execute, unresolved, epistemic = calibrate_epistemic()
    backend, backend_report = calibrate_backend()

    bundle = CalibrationBundle(
        version="controller-calibration-v0.2",
        execute_surface=execute,
        unresolved_surface=unresolved,
        backend_surface=backend,
        risk_contract="future-evidence-oracle-v0.1",
    )
    native = validate_native(bundle, seeds=native_seeds)

    gates = {
        "execute_precision": epistemic["execute_test"]["precision"] >= 0.93,
        "dangerous_false_execute": (
            epistemic["dangerous_false_execute_rate"] <= 0.03
        ),
        "three_state_generalization": (
            epistemic["three_state_accuracy"] >= 0.72
        ),
        "backend_regret": (
            backend_report["test"]["mean_runtime_regret"] <= 1.25
        ),
        "native_clean_executes": native["clean_execute_rate"] >= 0.95,
        "native_clean_correct": native["clean_accuracy"] == 1.0,
        "native_memory_isolated": native["memory_isolation_rate"] == 1.0,
        "native_binding_stable": native["zero_binding_conflicts"],
        "native_moderate_conflict_executes": (
            native["noise20_execute_rate"] >= 0.80
        ),
        "native_strong_conflict_no_execute": (
            native["noise40_execute_rate"] <= 0.20
        ),
        "native_ambiguous_source_explores": (
            native["ambiguous_source_explore_rate"] >= 0.80
        ),
    }

    return {
        "protocol": PROTOCOL,
        "seed": SEED,
        "risk_contract": {
            "future_relation_floor": 0.75,
            "future_source_floor": 0.80,
            "future_support_floor": 0.55,
            "current_wilson_z": 1.0,
            "current_wilson_floor": 0.60,
            "execute_validation_precision_target": 0.95,
            "unresolved_validation_precision_target": 0.92,
        },
        "surfaces": {
            "execute": surface_payload(execute),
            "unresolved": surface_payload(unresolved),
            "backend": surface_payload(backend),
        },
        "epistemic": epistemic,
        "backend": backend_report,
        "native_holdout": native,
        "gates": gates,
        "passed": all(gates.values()),
        "calibration_json": json.loads(bundle.to_json()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--native-seeds", type=int, default=10)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--calibration-output", type=Path)
    args = parser.parse_args()

    report = campaign(args.native_seeds)
    payload = json.dumps(report, indent=2, sort_keys=True)
    print(payload)

    if args.output:
        args.output.write_text(payload + "\n", encoding="utf-8")
    if args.calibration_output:
        args.calibration_output.write_text(
            json.dumps(
                report["calibration_json"],
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
