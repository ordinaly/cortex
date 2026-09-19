"""Prepared campaign for algebra-transfer-v1.6.

The protocol is frozen in docs/ALGEBRA_TRANSFER_V1_PROTOCOL.md.
This module is prepared for the official campaign but is not wired into the
research workflow yet.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from algebra_form_induction import (
    CortexFuzzyLawInducer,
    DirectFormMemory,
)
from algebra_form_induction_campaign import (
    left_zero,
    pattern_keys,
    random_magma,
    semilattice_min,
)
from algebra_induction import (
    HiddenAlgebra,
    anonymize,
    cyclic_group,
    dihedral_group,
    partial_observation,
    subtraction_magma,
)
from algebra_transfer import (
    CortexTransferredLawInducer,
    build_transfer_library,
    score_transfer_heldout,
)


SOURCE_SEEDS_PER_ORDER = 4
TARGET_SEED_VALUES = tuple(range(100, 110))
SOURCE_HOLDOUT = 0.40
TARGET_OBSERVED_FRACTIONS = (0.20, 0.30, 0.40)
CONTROL_OBSERVED_FRACTION = 0.30


def namespace_case(
    elements: tuple[str, ...],
    table: dict[tuple[str, str], str],
    prefix: str,
) -> tuple[tuple[str, ...], dict[tuple[str, str], str]]:
    mapping = {
        element: f"{prefix}:{element}"
        for element in elements
    }
    namespaced = tuple(sorted(mapping.values()))
    namespaced_table = {
        (mapping[left], mapping[right]): mapping[value]
        for (left, right), value in table.items()
    }
    return namespaced, namespaced_table


def source_specs() -> dict[str, tuple[HiddenAlgebra, ...]]:
    return {
        "cyclic": (
            cyclic_group(5),
            cyclic_group(7),
        ),
        "dihedral": (
            dihedral_group(3),
            dihedral_group(5),
        ),
        "semilattice": (
            semilattice_min(5),
            semilattice_min(6),
        ),
        "left-zero": (
            left_zero(5),
            left_zero(6),
        ),
    }


def target_specs() -> dict[str, HiddenAlgebra]:
    return {
        "cyclic": cyclic_group(8),
        "dihedral": dihedral_group(4),
        "semilattice": semilattice_min(7),
        "left-zero": left_zero(7),
    }


def build_family_library(
    family: str,
    algebras: tuple[HiddenAlgebra, ...],
) -> tuple[tuple, set[str], tuple[int, ...]]:
    models = []
    source_labels: set[str] = set()
    orders: list[int] = []

    for algebra_index, algebra in enumerate(algebras):
        orders.append(len(algebra.elements))
        for seed in range(SOURCE_SEEDS_PER_ORDER):
            raw_elements, raw_table, _identity = anonymize(
                algebra,
                seed=100_003 * (algebra_index + 1) + 997 * seed + 17,
            )
            elements, table = namespace_case(
                raw_elements,
                raw_table,
                f"source:{family}:{len(algebra.elements)}:{seed}",
            )
            observed, _heldout = partial_observation(
                table,
                seed=1009 * seed + 53 * (algebra_index + 1),
                holdout_fraction=SOURCE_HOLDOUT,
            )
            models.append(
                CortexFuzzyLawInducer(
                    elements,
                    observed,
                )
            )
            source_labels.update(elements)

    library = build_transfer_library(
        models,
        minimum_source_support=0.75,
        minimum_source_membership=0.98,
        maximum_forms=24,
    )
    return library, source_labels, tuple(orders)


def target_case(
    algebra: HiddenAlgebra,
    *,
    family: str,
    seed: int,
    observed_fraction: float,
) -> tuple[
    tuple[str, ...],
    dict[tuple[str, str], str],
    dict[tuple[str, str], str],
    set[tuple[str, str]],
]:
    raw_elements, raw_table, _identity = anonymize(
        algebra,
        seed=500_009 + 1009 * seed,
    )
    elements, table = namespace_case(
        raw_elements,
        raw_table,
        f"target:{family}:{len(algebra.elements)}:{seed}:{observed_fraction:.2f}",
    )
    observed, heldout = partial_observation(
        table,
        seed=700_001 + 1013 * seed + round(observed_fraction * 100),
        holdout_fraction=1.0 - observed_fraction,
    )
    return elements, table, observed, heldout


def compatible_row(
    *,
    family: str,
    algebra: HiddenAlgebra,
    library: tuple,
    source_labels: set[str],
    source_orders: tuple[int, ...],
    seed: int,
    observed_fraction: float,
) -> dict:
    elements, table, observed, heldout = target_case(
        algebra,
        family=family,
        seed=seed,
        observed_fraction=observed_fraction,
    )

    transfer = CortexTransferredLawInducer(
        elements,
        observed,
        library,
    )
    scratch = CortexFuzzyLawInducer(
        elements,
        observed,
    )
    direct = DirectFormMemory(observed)

    transfer_metrics = score_transfer_heldout(
        transfer,
        table,
        heldout,
    )
    scratch_metrics = score_transfer_heldout(
        scratch,
        table,
        heldout,
    )
    direct_metrics = score_transfer_heldout(
        direct,
        table,
        heldout,
    )

    transfer.close()
    scratch.close()

    return {
        "protocol": "algebra-transfer-v1.6",
        "kind": "compatible",
        "family": family,
        "seed": seed,
        "source_orders": list(source_orders),
        "target_order": len(algebra.elements),
        "observed_fraction": observed_fraction,
        "source_target_label_overlap": len(
            source_labels.intersection(elements)
        ),
        "library_size": len(library),
        "scratch_grammar_forms": len(scratch.forms),
        "transfer_structurally_supported": len(
            transfer.structurally_supported_keys()
        ),
        "transfer_selected": len(transfer.active_laws),
        "transfer_selected_keys": sorted(transfer.selected_keys()),
        "transfer_selection_mode": transfer.selection_mode,
        "transfer_hypothesis_budget": transfer.hypothesis_budget,
        "transfer_required_predictions": transfer.required_predictions,
        "transfer_validation_positive": transfer.joint_validation.positive,
        "transfer_validation_negative": transfer.joint_validation.negative,
        "transfer_validation_membership": transfer.joint_validation.membership,
        "transfer_validation_conflicts": transfer.joint_validation_conflicts,
        "scratch_supported": len(scratch.active_keys()),
        "scratch_selected": len(scratch.active_laws),
        "transfer_conflicts": transfer.conflicts,
        "scratch_conflicts": scratch.conflicts,
        "coverage_gain": (
            transfer_metrics["coverage"]
            - scratch_metrics["coverage"]
        ),
        "metrics": {
            "transfer": transfer_metrics,
            "scratch": scratch_metrics,
            "direct-memory": direct_metrics,
        },
    }


def control_row(
    *,
    control: str,
    library_name: str,
    library: tuple,
    source_labels: set[str],
    algebra: HiddenAlgebra,
    seed: int,
) -> dict:
    family = f"{control}:{library_name}"
    elements, table, observed, heldout = target_case(
        algebra,
        family=family,
        seed=seed,
        observed_fraction=CONTROL_OBSERVED_FRACTION,
    )
    transfer = CortexTransferredLawInducer(
        elements,
        observed,
        library,
    )
    metrics = score_transfer_heldout(
        transfer,
        table,
        heldout,
    )
    transfer.close()

    rebracket_key = pattern_keys()["rebracket"]

    return {
        "protocol": "algebra-transfer-v1.6",
        "kind": control,
        "library_name": library_name,
        "seed": seed,
        "target_order": len(algebra.elements),
        "observed_fraction": CONTROL_OBSERVED_FRACTION,
        "source_target_label_overlap": len(
            source_labels.intersection(elements)
        ),
        "library_size": len(library),
        "transfer_structurally_supported": len(
            transfer.structurally_supported_keys()
        ),
        "transfer_selected": len(transfer.active_laws),
        "transfer_selected_keys": sorted(transfer.selected_keys()),
        "transfer_selection_mode": transfer.selection_mode,
        "transfer_hypothesis_budget": transfer.hypothesis_budget,
        "transfer_required_predictions": transfer.required_predictions,
        "transfer_validation_positive": transfer.joint_validation.positive,
        "transfer_validation_negative": transfer.joint_validation.negative,
        "transfer_validation_membership": transfer.joint_validation.membership,
        "transfer_validation_conflicts": transfer.joint_validation_conflicts,
        "rebracket_selected": (
            rebracket_key in transfer.selected_keys()
        ),
        "transfer_conflicts": transfer.conflicts,
        "metrics": {
            "transfer": metrics,
        },
    }


def campaign(output: Path) -> dict:
    libraries = {}
    source_labels = {}
    source_orders = {}

    for family, algebras in source_specs().items():
        library, labels, orders = build_family_library(
            family,
            algebras,
        )
        libraries[family] = library
        source_labels[family] = labels
        source_orders[family] = orders

    rows = []

    for family, algebra in target_specs().items():
        for fraction in TARGET_OBSERVED_FRACTIONS:
            for seed in TARGET_SEED_VALUES:
                rows.append(
                    compatible_row(
                        family=family,
                        algebra=algebra,
                        library=libraries[family],
                        source_labels=source_labels[family],
                        source_orders=source_orders[family],
                        seed=seed,
                        observed_fraction=fraction,
                    )
                )

    for library_name, library in libraries.items():
        for seed in TARGET_SEED_VALUES:
            rows.append(
                control_row(
                    control="random-control",
                    library_name=library_name,
                    library=library,
                    source_labels=source_labels[library_name],
                    algebra=random_magma(
                        6,
                        900_001 + 10_007 * seed,
                    ),
                    seed=seed,
                )
            )

    for seed in TARGET_SEED_VALUES:
        rows.append(
            control_row(
                control="subtraction-control",
                library_name="cyclic",
                library=libraries["cyclic"],
                source_labels=source_labels["cyclic"],
                algebra=subtraction_magma(7),
                seed=seed,
            )
        )

    output.write_text(
        "".join(
            json.dumps(row, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )

    compatible = [
        row for row in rows
        if row["kind"] == "compatible"
    ]

    gains = {}
    for fraction in TARGET_OBSERVED_FRACTIONS:
        subset = [
            row for row in compatible
            if row["observed_fraction"] == fraction
        ]
        gains[str(fraction)] = {
            "mean_transfer_coverage": sum(
                row["metrics"]["transfer"]["coverage"]
                for row in subset
            ) / len(subset),
            "mean_scratch_coverage": sum(
                row["metrics"]["scratch"]["coverage"]
                for row in subset
            ) / len(subset),
            "mean_coverage_gain": sum(
                row["coverage_gain"]
                for row in subset
            ) / len(subset),
        }

    summary = {
        "protocol": "algebra-transfer-v1.6",
        "status": "official-campaign-output",
        "compatible_rows": len(compatible),
        "control_rows": len(rows) - len(compatible),
        "target_seeds": list(TARGET_SEED_VALUES),
        "target_observed_fractions": list(
            TARGET_OBSERVED_FRACTIONS
        ),
        "libraries": {
            family: {
                "source_orders": list(source_orders[family]),
                "forms": len(libraries[family]),
                "keys": [
                    prior.form.key
                    for prior in libraries[family]
                ],
            }
            for family in sorted(libraries)
        },
        "coverage_by_fraction": gains,
    }
    print(json.dumps(summary, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("algebra-transfer-v1.6.jsonl"),
    )
    args = parser.parse_args()
    campaign(args.output)


if __name__ == "__main__":
    main()
