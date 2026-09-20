"""Gates for Hybrid Core orchestration ablation v0.1."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from hybrid_orchestration_ablation import (
    NEURAL_MODES,
    POLICIES,
    PROTOCOL,
)


def rate(
    rows,
    predicate,
) -> float:
    rows = list(
        rows
    )
    if not rows:
        return 0.0
    return (
        sum(
            int(
                predicate(
                    row
                )
            )
            for row in rows
        )
        / len(
            rows
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "results",
        type=Path,
    )
    args = parser.parse_args()

    rows = [
        json.loads(
            line
        )
        for line
        in args.results.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    if not rows:
        raise SystemExit(
            "no ablation rows"
        )

    seeds = sorted(
        {
            int(
                row[
                    "seed"
                ]
            )
            for row in rows
        }
    )

    def selected(
        policy: str,
        mode: str | None = None,
    ):
        return [
            row
            for row in rows
            if (
                row[
                    "policy"
                ]
                == policy
                and (
                    mode is None
                    or row[
                        "neural_mode"
                    ]
                    == mode
                )
            )
        ]

    metrics = {
        "neural_direct_one_call_rate": rate(
            selected(
                "neural-direct"
            ),
            lambda row: (
                row[
                    "neural_calls"
                ]
                == 1
                and not row[
                    "prompt_ir_present"
                ]
                and not row[
                    "cortex_available"
                ]
            ),
        ),
        "full_context_guidance_rate": rate(
            selected(
                "full",
                "context-responsive",
            ),
            lambda row: (
                row[
                    "diagnosis_status"
                ]
                == "supported"
                and row[
                    "revision_cycles"
                ]
                == 0
                and row[
                    "final_asserted"
                ]
                is True
            ),
        ),
        "withheld_context_revision_rate": rate(
            selected(
                "no-cortex-context",
                "context-responsive",
            ),
            lambda row: (
                row[
                    "diagnosis_status"
                ]
                == "supported"
                and row[
                    "revision_cycles"
                ]
                == 1
                and row[
                    "final_asserted"
                ]
                is True
            ),
        ),
        "full_blind_revision_rate": rate(
            selected(
                "full",
                "context-blind-revisable",
            ),
            lambda row: (
                row[
                    "diagnosis_status"
                ]
                == "supported"
                and row[
                    "revision_cycles"
                ]
                == 1
                and row[
                    "final_asserted"
                ]
                is True
            ),
        ),
        "no_revision_conflict_rate": rate(
            selected(
                "no-revision",
                "context-blind-revisable",
            ),
            lambda row: (
                row[
                    "diagnosis_status"
                ]
                == "conflict"
                and row[
                    "revision_cycles"
                ]
                == 0
            ),
        ),
        "stubborn_conflict_bounded_rate": rate(
            selected(
                "full",
                "stubborn",
            ),
            lambda row: (
                row[
                    "diagnosis_status"
                ]
                == "conflict"
                and row[
                    "revision_cycles"
                ]
                == 1
            ),
        ),
        "nonempty_answer_rate": rate(
            rows,
            lambda row: bool(
                row[
                    "answer_nonempty"
                ]
            ),
        ),
    }

    checks = {
        "protocol_exact": all(
            row[
                "protocol"
            ]
            == PROTOCOL
            for row in rows
        ),
        "policy_coverage": (
            sorted(
                {
                    row[
                        "policy"
                    ]
                    for row in rows
                }
            )
            == sorted(
                POLICIES
            )
        ),
        "mode_coverage": (
            sorted(
                {
                    row[
                        "neural_mode"
                    ]
                    for row in rows
                }
            )
            == sorted(
                NEURAL_MODES
            )
        ),
        "row_count": (
            len(
                rows
            )
            == (
                len(
                    seeds
                )
                * len(
                    POLICIES
                )
                * len(
                    NEURAL_MODES
                )
            )
        ),
        **{
            key: (
                value
                == 1.0
            )
            for key, value
            in metrics.items()
        },
    }

    report = {
        "protocol": PROTOCOL,
        "rows": len(
            rows
        ),
        "seeds": seeds,
        "metrics": metrics,
        "checks": checks,
    }
    print(
        json.dumps(
            report,
            sort_keys=True,
        )
    )

    failed = [
        key
        for key, value
        in checks.items()
        if not value
    ]
    if failed:
        raise SystemExit(
            "failed gates: "
            + ", ".join(
                failed
            )
        )


if __name__ == "__main__":
    main()
