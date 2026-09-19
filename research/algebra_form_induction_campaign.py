"""Campaign for bounded fuzzy algebraic-form induction."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from random import Random

from algebra_form_induction import (
    CortexFuzzyLawInducer,
    DirectFormMemory,
    canonical_equation,
    op,
    score_heldout,
    var,
)
from algebra_induction import (
    HiddenAlgebra,
    anonymize,
    cyclic_group,
    dihedral_group,
    partial_observation,
    subtraction_magma,
)


def semilattice_min(order: int) -> HiddenAlgebra:
    elements = tuple(range(order))
    return HiddenAlgebra(
        name=f"semilattice-min-{order}",
        elements=elements,
        table={(a, b): min(a, b) for a in elements for b in elements},
        associative=True,
        commutative=True,
        identity=order - 1,
    )


def left_zero(order: int) -> HiddenAlgebra:
    elements = tuple(range(order))
    return HiddenAlgebra(
        name=f"left-zero-{order}",
        elements=elements,
        table={(a, b): a for a in elements for b in elements},
        associative=True,
        commutative=False,
        identity=None,
    )


def random_magma(order: int, seed: int) -> HiddenAlgebra:
    rng = Random(seed)
    elements = tuple(range(order))
    return HiddenAlgebra(
        name=f"random-magma-{order}",
        elements=elements,
        table={
            (a, b): rng.randrange(order)
            for a in elements
            for b in elements
        },
        associative=False,
        commutative=False,
        identity=None,
    )


def pattern_keys() -> dict[str, str]:
    x, y, z = var("x"), var("y"), var("z")
    return {
        "rebracket": canonical_equation(
            op(op(x, y), z),
            op(x, op(y, z)),
        ),
        "swap": canonical_equation(
            op(x, y),
            op(y, x),
        ),
        "repeat": canonical_equation(
            op(x, x),
            x,
        ),
        "left_projection": canonical_equation(
            op(x, y),
            x,
        ),
    }


def static_families() -> tuple[HiddenAlgebra, ...]:
    return (
        cyclic_group(7),
        dihedral_group(4),
        semilattice_min(6),
        left_zero(6),
        subtraction_magma(7),
    )


def run_case(
    algebra: HiddenAlgebra,
    *,
    seed: int,
    holdout_fraction: float,
) -> dict:
    elements, table, _identity = anonymize(
        algebra,
        seed=seed * 101 + 37,
    )
    observed, heldout = partial_observation(
        table,
        seed=seed * 1009 + 41,
        holdout_fraction=holdout_fraction,
    )

    cortex = CortexFuzzyLawInducer(elements, observed)
    fit_only = CortexFuzzyLawInducer(
        elements,
        observed,
        use_validation=False,
    )
    direct = DirectFormMemory(observed)

    metrics = {
        "cortex-fuzzy": score_heldout(cortex, table, heldout),
        "fit-only-enumerator": score_heldout(fit_only, table, heldout),
        "direct-memory": score_heldout(direct, table, heldout),
    }

    keys = pattern_keys()
    active = cortex.active_keys()
    detected = {
        name: key in active
        for name, key in keys.items()
    }

    cortex.close()
    derived = {
        key: sum(
            int(key in provenance)
            for provenance in cortex.provenance.values()
        )
        for key in keys.values()
    }

    return {
        "protocol": "algebra-form-induction-v1.3",
        "family": algebra.name,
        "seed": seed,
        "order": len(elements),
        "holdout_fraction": holdout_fraction,
        "observed_cells": len(observed),
        "heldout_cells": len(heldout),
        "grammar_forms": len(cortex.forms),
        "supported_laws": len(cortex.active_keys()),
        "selected_laws": len(cortex.active_laws),
        "detected_patterns": detected,
        "top_laws": cortex.top_laws(10),
        "conflicts": cortex.conflicts,
        "closure_rounds": cortex.rounds,
        "derived_by_target_pattern": derived,
        "metrics": metrics,
    }


def run_random_case(
    *,
    seed: int,
    holdout_fraction: float,
) -> dict:
    algebra = random_magma(6, seed * 10007 + 97)
    return run_case(
        algebra,
        seed=seed,
        holdout_fraction=holdout_fraction,
    )


def campaign(
    *,
    seeds: int,
    holdout_fraction: float,
    output: Path,
) -> dict:
    rows = [
        run_case(
            algebra,
            seed=seed,
            holdout_fraction=holdout_fraction,
        )
        for algebra in static_families()
        for seed in range(seeds)
    ]
    rows.extend(
        run_random_case(
            seed=seed,
            holdout_fraction=holdout_fraction,
        )
        for seed in range(seeds)
    )

    output.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )

    families = sorted({row["family"] for row in rows})
    summary = {
        "protocol": "algebra-form-induction-v1.3",
        "seeds_per_family": seeds,
        "holdout_fraction": holdout_fraction,
        "grammar_forms": rows[0]["grammar_forms"],
        "families": {},
    }

    for family in families:
        family_rows = [row for row in rows if row["family"] == family]
        family_summary = {
            "mean_supported_laws": sum(
                row["supported_laws"] for row in family_rows
            ) / len(family_rows),
            "mean_selected_laws": sum(
                row["selected_laws"] for row in family_rows
            ) / len(family_rows),
            "mean_conflicts": sum(
                row["conflicts"] for row in family_rows
            ) / len(family_rows),
            "detected_pattern_rates": {
                name: sum(
                    int(row["detected_patterns"][name])
                    for row in family_rows
                ) / len(family_rows)
                for name in pattern_keys()
            },
            "methods": {},
        }
        for method in (
            "cortex-fuzzy",
            "fit-only-enumerator",
            "direct-memory",
        ):
            family_summary["methods"][method] = {
                metric: sum(
                    row["metrics"][method][metric]
                    for row in family_rows
                ) / len(family_rows)
                for metric in (
                    "accuracy",
                    "coverage",
                    "resolved_accuracy",
                )
            }
        summary["families"][family] = family_summary

    print(json.dumps(summary, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--holdout", type=float, default=0.40)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("algebra-form-induction-v1.jsonl"),
    )
    args = parser.parse_args()
    campaign(
        seeds=args.seeds,
        holdout_fraction=args.holdout,
        output=args.output,
    )


if __name__ == "__main__":
    main()
