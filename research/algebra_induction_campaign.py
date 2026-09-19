"""Campaign for evidence-gated finite-algebra induction."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from algebra_induction import (
    CortexAlgebraInducer,
    DirectTableMemory,
    MajorityProductBaseline,
    anonymize,
    cyclic_group,
    dihedral_group,
    partial_observation,
    score_heldout,
    subtraction_magma,
)


def families():
    return (
        cyclic_group(7),
        dihedral_group(4),
        subtraction_magma(7),
    )


def run_case(algebra, seed: int, holdout_fraction: float) -> dict:
    elements, table, hidden_identity = anonymize(
        algebra,
        seed=seed * 101 + 17,
    )
    observed, heldout = partial_observation(
        table,
        seed=seed * 1009 + 23,
        holdout_fraction=holdout_fraction,
    )

    cortex = CortexAlgebraInducer(elements, observed)
    direct = DirectTableMemory(observed)
    majority = MajorityProductBaseline(observed)

    cortex_metrics = score_heldout(cortex, table, heldout)
    direct_metrics = score_heldout(direct, table, heldout)
    majority_metrics = score_heldout(majority, table, heldout)

    cortex.close()
    derived_by_provenance: dict[str, int] = {}
    example_derivation = None
    for pair in sorted(heldout):
        if pair not in cortex.provenance:
            continue
        provenance = cortex.provenance[pair]
        derived_by_provenance[provenance] = (
            derived_by_provenance.get(provenance, 0) + 1
        )
        if example_derivation is None:
            example_derivation = {
                "left": pair[0],
                "right": pair[1],
                "value": cortex.completed[pair],
                "provenance": provenance,
            }

    laws = cortex.law_summary()
    detected = {
        "associative": bool(laws["associativity"]["active"]),
        "commutative": bool(laws["commutativity"]["active"]),
        "has_identity": bool(laws["identity"]["active"]),
    }
    expected = {
        "associative": algebra.associative,
        "commutative": algebra.commutative,
        "has_identity": algebra.identity is not None,
    }

    return {
        "protocol": "algebra-induction-v1",
        "family": algebra.name,
        "seed": seed,
        "order": len(elements),
        "holdout_fraction": holdout_fraction,
        "observed_cells": len(observed),
        "heldout_cells": len(heldout),
        "expected_laws": expected,
        "detected_laws": detected,
        "hidden_identity_alias": hidden_identity,
        "detected_identity_alias": cortex.identity,
        "law_evidence": laws,
        "law_detection_exact": detected == expected,
        "conflicts": cortex.conflicts,
        "closure_rounds": cortex.rounds,
        "derived_by_provenance": derived_by_provenance,
        "example_derivation": example_derivation,
        "metrics": {
            "cortex": cortex_metrics,
            "direct-memory": direct_metrics,
            "majority-product": majority_metrics,
        },
    }


def campaign(
    seeds: int,
    holdout_fraction: float,
    output: Path,
) -> dict:
    rows = [
        run_case(algebra, seed, holdout_fraction)
        for algebra in families()
        for seed in range(seeds)
    ]

    output.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )

    summary = {
        "protocol": "algebra-induction-v1",
        "seeds_per_family": seeds,
        "holdout_fraction": holdout_fraction,
        "families": {},
    }

    for algebra in families():
        family_rows = [
            row for row in rows if row["family"] == algebra.name
        ]
        family = {}
        for method in ("cortex", "direct-memory", "majority-product"):
            family[method] = {
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
        family["law_detection_rate"] = sum(
            int(row["law_detection_exact"])
            for row in family_rows
        ) / len(family_rows)
        family["mean_conflicts"] = sum(
            row["conflicts"] for row in family_rows
        ) / len(family_rows)
        family["detected_law_rates"] = {
            law: sum(
                int(row["detected_laws"][law])
                for row in family_rows
            ) / len(family_rows)
            for law in ("associative", "commutative", "has_identity")
        }
        summary["families"][algebra.name] = family

    print(json.dumps(summary, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--holdout", type=float, default=0.40)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("algebra-induction-v1.jsonl"),
    )
    args = parser.parse_args()
    campaign(args.seeds, args.holdout, args.output)


if __name__ == "__main__":
    main()
