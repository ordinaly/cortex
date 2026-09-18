"""Characterize provisional articulation against current Hybrid Cortex."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from cortex import HybridCortexRuntime
from robust_articulation import IdentityToken, RobustArticulator, identity_metrics


ENTITIES = 12
DIM = 32
BURST = 4
CYCLES = 16


def centers_for(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    q, _r = np.linalg.qr(rng.normal(size=(DIM, ENTITIES)))
    return q[:, :ENTITIES].T.copy()


def observe(
    center: np.ndarray,
    rng: np.random.Generator,
    *,
    noise: float,
    corruption_probability: float,
) -> list[float]:
    x = center + rng.normal(0.0, noise, size=DIM)
    if rng.random() < corruption_probability:
        indexes = rng.choice(DIM, size=DIM // 4, replace=False)
        x[indexes] += rng.normal(0.0, 1.5, size=len(indexes))
    x /= np.linalg.norm(x)
    return x.tolist()


def run_case(seed: int, noise: float, corruption_probability: float) -> dict:
    rng = np.random.default_rng(seed)
    centers = centers_for(seed + 991)

    cortex = HybridCortexRuntime(
        embedding_dim=DIM,
        max_entities=96,
        budget=24,
        nuisance_mode="identity-only",
    )
    robust = RobustArticulator()

    cortex_records = []
    robust_tokens: list[tuple[int, IdentityToken]] = []

    for _cycle in range(CYCLES):
        for truth in rng.permutation(ENTITIES):
            for _ in range(BURST):
                embedding = observe(
                    centers[truth],
                    rng,
                    noise=noise,
                    corruption_probability=corruption_probability,
                )
                cortex_read = cortex.step_embeddings([embedding])
                cortex_records.append((truth, int(cortex_read.bindings[0])))
                token = robust.observe_frame([embedding])[0]
                robust_tokens.append((truth, token))

    robust_records = [
        (truth, robust.resolve(token))
        for truth, token in robust_tokens
    ]
    return {
        "protocol": "robust-articulation-v1",
        "seed": seed,
        "noise": noise,
        "corruption_probability": corruption_probability,
        "cortex": identity_metrics(cortex_records, truth_count=ENTITIES),
        "robust": identity_metrics(robust_records, truth_count=ENTITIES),
        "robust_committed_entities": robust.committed_entities,
        "robust_reconciliations": robust.reconciliations,
        "robust_new_commits": robust.new_commits,
    }


def campaign(seeds: int, output: Path) -> None:
    rows = []
    for noise in (0.03, 0.12, 0.25):
        for corruption in (0.0, 0.03, 0.10):
            for seed in range(seeds):
                rows.append(run_case(seed, noise, corruption))

    output.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )

    for noise in (0.03, 0.12, 0.25):
        for corruption in (0.0, 0.03, 0.10):
            subset = [
                row
                for row in rows
                if row["noise"] == noise
                and row["corruption_probability"] == corruption
            ]
            print(
                json.dumps(
                    {
                        "noise": noise,
                        "corruption_probability": corruption,
                        "seeds": seeds,
                        "cortex_purity": float(
                            np.mean([r["cortex"]["purity"] for r in subset])
                        ),
                        "cortex_fragmentation": float(
                            np.mean(
                                [r["cortex"]["fragmentation"] for r in subset]
                            )
                        ),
                        "robust_purity": float(
                            np.mean([r["robust"]["purity"] for r in subset])
                        ),
                        "robust_fragmentation": float(
                            np.mean(
                                [r["robust"]["fragmentation"] for r in subset]
                            )
                        ),
                        "robust_coverage": float(
                            np.mean([r["robust"]["coverage"] for r in subset])
                        ),
                        "reconciliations": float(
                            np.mean(
                                [r["robust_reconciliations"] for r in subset]
                            )
                        ),
                    },
                    sort_keys=True,
                )
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("robust-articulation-v1.jsonl"),
    )
    args = parser.parse_args()
    campaign(args.seeds, args.output)


if __name__ == "__main__":
    main()
