#!/usr/bin/env python3
"""Frozen ResNet-18 -> Cortex benchmark on CAVIAR oracle boxes.

Protocol: hybrid-caviar-v1

The visual model is frozen and generic (ImageNet classification pretraining).
Ground-truth IDs are used only after Cortex inference for scoring. Manual CAVIAR
boxes remain an oracle detector so the experiment isolates visual
representation + Cortex structural inference from object-detection quality.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment

from benchmark_camera_pixels import (
    DEFAULT_SCENES,
    BoxTruth,
    active_entities,
    identity_metrics,
    inventory_windows,
    load_scene,
)
from provenance import benchmark_provenance

from cortex import (
    HybridCortexRuntime,
    NeuralObservation,
    cosine_mse_embedding,
)

PROTOCOL = "hybrid-caviar-v1"
ENCODER_NAME = "torchvision-resnet18-imagenet-default"
EMBEDDING_DIM = 512
MAX_ENTITIES = 256
SPAWN_THRESHOLD = 0.7


def make_encoder():
    import torch
    import torchvision
    from torchvision.models import ResNet18_Weights, resnet18

    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    model.fc = torch.nn.Identity()
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return torch, torchvision, model, weights, weights.transforms()


def crop(image: Image.Image, box: BoxTruth) -> Image.Image:
    width, height = image.size
    x0 = max(0, int(math.floor(box.xc - box.width / 2.0)))
    y0 = max(0, int(math.floor(box.yc - box.height / 2.0)))
    x1 = min(width, int(math.ceil(box.xc + box.width / 2.0)))
    y1 = min(height, int(math.ceil(box.yc + box.height / 2.0)))
    if x1 <= x0 or y1 <= y0:
        return Image.new("RGB", (8, 8))
    return image.crop((x0, y0, x1, y1))


def encode_scene(frames, model, transform, torch):
    encoded = []
    with torch.inference_mode():
        for frame_number, image_bytes, boxes in frames:
            if not boxes:
                encoded.append((frame_number, []))
                continue
            with Image.open(io.BytesIO(image_bytes)) as image:
                rgb = image.convert("RGB")
                batch = torch.stack([transform(crop(rgb, box)) for box in boxes])
            features = model(batch).detach().cpu()
            encoded.append(
                (
                    frame_number,
                    [
                        (box, features[index].tolist())
                        for index, box in enumerate(boxes)
                    ],
                )
            )
    return encoded


class OnlinePrototypeBaseline:
    """Label-free online identity baseline using the same neural metric.

    Every frame uses a rectangular Hungarian assignment over existing
    prototypes plus spawn columns. This preserves simultaneous injectivity and
    uses the same 0.7 spawn threshold as fixed-group Cortex, but omits Cortex
    reliability/noise state, memory, recurrence and graph reasoning.
    """

    def __init__(self, update_mode: str) -> None:
        if update_mode not in {"centroid", "last-observation"}:
            raise ValueError("unsupported baseline update mode")
        self.update_mode = update_mode
        self.prototypes: list[np.ndarray] = []
        self.counts: list[int] = []

    def step_embeddings(self, embeddings: list[list[float]]) -> list[int]:
        if not embeddings:
            return []
        vectors = np.asarray(
            [cosine_mse_embedding(x) for x in embeddings],
            dtype=np.float64,
        )
        k = int(vectors.shape[0])
        ne = len(self.prototypes)
        if ne == 0:
            if k > MAX_ENTITIES:
                raise RuntimeError("baseline entity capacity exhausted")
            self.prototypes.extend([x.copy() for x in vectors])
            self.counts.extend([1] * k)
            return list(range(k))

        prototypes = np.stack(self.prototypes, axis=0)
        existing_cost = np.mean(
            (vectors[:, None, :] - prototypes[None, :, :]) ** 2,
            axis=2,
        )
        cols = ne + k
        cost = np.full((k, cols), SPAWN_THRESHOLD, dtype=np.float64)
        cost[:, :ne] = existing_cost
        row_index = np.arange(k)[:, None]
        spawn_index = np.arange(k)[None, :]
        cost[:, ne:] += 1.0e-8 * np.abs(row_index - spawn_index)

        rows, assigned_cols = linear_sum_assignment(cost)
        assignment = np.full(k, -1, dtype=np.int64)
        assignment[rows] = assigned_cols

        bindings: list[int] = []
        for i, x in enumerate(vectors):
            col = int(assignment[i])
            if col < ne and float(cost[i, col]) <= SPAWN_THRESHOLD:
                entity = col
                bindings.append(entity)
                if self.update_mode == "centroid":
                    n = self.counts[entity]
                    self.prototypes[entity] = (
                        n * self.prototypes[entity] + x
                    ) / (n + 1)
                    self.counts[entity] = n + 1
                else:
                    self.prototypes[entity] = x.copy()
                    self.counts[entity] += 1
            else:
                if len(self.prototypes) >= MAX_ENTITIES:
                    raise RuntimeError("baseline entity capacity exhausted")
                entity = len(self.prototypes)
                self.prototypes.append(x.copy())
                self.counts.append(1)
                bindings.append(entity)

        if len(bindings) != len(set(bindings)):
            raise RuntimeError("baseline simultaneous binding lost injectivity")
        return bindings

    @property
    def active_entities(self) -> int:
        return len(self.prototypes)


class RunAccumulator:
    def __init__(self) -> None:
        self.records: list[tuple[str, int]] = []
        self.inventory: list[int] = []
        self.truth_ids: set[str] = set()
        self.injectivity_violations = 0
        self.max_covisible = 0
        self.last_seen: dict[str, tuple[int, int]] = {}
        self.gap_counts = {10: 0, 25: 0, 100: 0}
        self.gap_same = {10: 0, 25: 0, 100: 0}
        self.error: str | None = None

    def observe(
        self,
        chronological_index: int,
        detections,
        bindings: list[int],
        active_count: int,
    ) -> None:
        if len(bindings) != len(set(bindings)):
            self.injectivity_violations += 1
        self.max_covisible = max(self.max_covisible, len(bindings))
        for (box, _embedding), binding in zip(detections, bindings):
            self.truth_ids.add(box.truth_id)
            self.records.append((box.truth_id, binding))
            previous = self.last_seen.get(box.truth_id)
            if previous is not None:
                previous_index, previous_binding = previous
                gap = chronological_index - previous_index
                for threshold in self.gap_counts:
                    if gap >= threshold:
                        self.gap_counts[threshold] += 1
                        self.gap_same[threshold] += int(binding == previous_binding)
            self.last_seen[box.truth_id] = (chronological_index, binding)
        self.inventory.append(active_count)


def _result_row(
    *,
    scene: str,
    repetition: int,
    seed: int,
    source_provenance: dict[str, Any],
    torch_version: str,
    torchvision_version: str,
    weights_name: str,
    model: str,
    accumulator: RunAccumulator,
    config_sha256: str,
    nuisance_mode: str,
    baseline_update: str | None,
) -> dict[str, Any]:
    metrics = identity_metrics(accumulator.records)
    checkpoints, increments = inventory_windows(accumulator.inventory)
    final_active = accumulator.inventory[-1] if accumulator.inventory else 0
    return {
        **benchmark_provenance(PROTOCOL),
        **source_provenance,
        "scene": scene,
        "repetition": repetition,
        "order_seed": seed,
        "model": model,
        "encoder": ENCODER_NAME,
        "encoder_weights": weights_name,
        "torch_version": torch_version,
        "torchvision_version": torchvision_version,
        "embedding_dim": EMBEDDING_DIM,
        "metric_adapter": "l2-normalize;sqrt(d/2)-scale;MSE=1-cosine",
        "nuisance_mode": nuisance_mode,
        "baseline_update": baseline_update,
        "spawn_threshold": SPAWN_THRESHOLD,
        "frozen_encoder": True,
        "oracle_detection": True,
        "ground_truth_input": False,
        "config_sha256": config_sha256,
        "capacity_failure": accumulator.error is not None,
        "model_error": accumulator.error,
        "frames": len(accumulator.inventory),
        "observations": len(accumulator.records),
        "true_tracks": len(accumulator.truth_ids),
        "max_covisible": accumulator.max_covisible,
        "active_entities": final_active,
        "oversegmentation_ratio": final_active / max(1, len(accumulator.truth_ids)),
        "inventory_checkpoints": checkpoints,
        "new_entities_by_window": increments,
        "injectivity_violations": accumulator.injectivity_violations,
        **metrics,
        "gap_return_10_count": accumulator.gap_counts[10],
        "gap_return_10_same_fraction": (
            accumulator.gap_same[10] / max(1, accumulator.gap_counts[10])
        ),
        "gap_return_25_count": accumulator.gap_counts[25],
        "gap_return_25_same_fraction": (
            accumulator.gap_same[25] / max(1, accumulator.gap_counts[25])
        ),
        "gap_return_100_count": accumulator.gap_counts[100],
        "gap_return_100_same_fraction": (
            accumulator.gap_same[100] / max(1, accumulator.gap_counts[100])
        ),
    }


def run_scene(
    scene: str,
    encoded_frames,
    source_provenance: dict[str, Any],
    repetition: int,
    torch_version: str,
    torchvision_version: str,
    weights_name: str,
) -> list[dict[str, Any]]:
    seed = 20260918 + repetition * 1009
    rng = random.Random(seed)
    hybrid = HybridCortexRuntime(
        embedding_dim=EMBEDDING_DIM,
        max_entities=MAX_ENTITIES,
        budget=24,
        nuisance_mode="identity-only",
    )
    centroid = OnlinePrototypeBaseline("centroid")
    last_observation = OnlinePrototypeBaseline("last-observation")

    accumulators = {
        "cortex": RunAccumulator(),
        "online-centroid": RunAccumulator(),
        "last-observation": RunAccumulator(),
    }

    for chronological_index, (_frame_number, detections) in enumerate(encoded_frames):
        shuffled = list(detections)
        rng.shuffle(shuffled)
        embeddings = [embedding for _box, embedding in shuffled]
        observations = [
            NeuralObservation(
                embedding=embedding,
                bbox=(box.xc, box.yc, box.width, box.height),
                timestamp=_frame_number,
                source=scene,
            )
            for box, embedding in shuffled
        ]

        if accumulators["cortex"].error is None:
            try:
                read = hybrid.step(observations)
                accumulators["cortex"].observe(
                    chronological_index,
                    shuffled,
                    list(map(int, read.bindings)),
                    active_entities(hybrid.runtime),
                )
            except Exception as exc:
                accumulators["cortex"].error = f"{type(exc).__name__}: {exc}"

        for name, baseline in [
            ("online-centroid", centroid),
            ("last-observation", last_observation),
        ]:
            if accumulators[name].error is not None:
                continue
            try:
                bindings = baseline.step_embeddings(embeddings)
                accumulators[name].observe(
                    chronological_index,
                    shuffled,
                    bindings,
                    baseline.active_entities,
                )
            except Exception as exc:
                accumulators[name].error = f"{type(exc).__name__}: {exc}"

    hybrid_config = hybrid.config_json()
    cortex_hash = hashlib.sha256(hybrid_config.encode()).hexdigest()
    baseline_hashes = {
        mode: hashlib.sha256(
            json.dumps(
                {
                    "metric": "1-cosine",
                    "spawn_threshold": SPAWN_THRESHOLD,
                    "max_entities": MAX_ENTITIES,
                    "assignment": "hungarian+spawn-columns",
                    "update": mode,
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()
        for mode in ["centroid", "last-observation"]
    }

    return [
        _result_row(
            scene=scene,
            repetition=repetition,
            seed=seed,
            source_provenance=source_provenance,
            torch_version=torch_version,
            torchvision_version=torchvision_version,
            weights_name=weights_name,
            model="cortex",
            accumulator=accumulators["cortex"],
            config_sha256=cortex_hash,
            nuisance_mode="fixed-identity",
            baseline_update=None,
        ),
        _result_row(
            scene=scene,
            repetition=repetition,
            seed=seed,
            source_provenance=source_provenance,
            torch_version=torch_version,
            torchvision_version=torchvision_version,
            weights_name=weights_name,
            model="online-centroid",
            accumulator=accumulators["online-centroid"],
            config_sha256=baseline_hashes["centroid"],
            nuisance_mode="none",
            baseline_update="centroid",
        ),
        _result_row(
            scene=scene,
            repetition=repetition,
            seed=seed,
            source_provenance=source_provenance,
            torch_version=torch_version,
            torchvision_version=torchvision_version,
            weights_name=weights_name,
            model="last-observation",
            accumulator=accumulators["last-observation"],
            config_sha256=baseline_hashes["last-observation"],
            nuisance_mode="none",
            baseline_update="last-observation",
        ),
    ]


def campaign(scenes: list[str], repetitions: int, cache_dir: Path, output: Path) -> None:
    torch, torchvision, model, weights, transform = make_encoder()
    rows = []
    total = len(scenes) * repetitions
    n = 0
    for scene in scenes:
        print(f"loading/encoding neural CAVIAR stream: {scene}", file=sys.stderr, flush=True)
        frames, provenance = load_scene(scene, cache_dir)
        encoded = encode_scene(frames, model, transform, torch)
        for repetition in range(repetitions):
            n += 1
            print(f"[{n:02d}/{total:02d}] {scene} rep={repetition + 1}", file=sys.stderr, flush=True)
            rows.extend(
                run_scene(
                    scene,
                    encoded,
                    provenance,
                    repetition,
                    torch.__version__,
                    torchvision.__version__,
                    str(weights),
                )
            )
    output.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", action="store_true")
    parser.add_argument("--scene", action="append", choices=sorted(DEFAULT_SCENES), default=[])
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--cache-dir", type=Path, default=Path(".hybrid-caviar-v1-cache"))
    parser.add_argument("--output", type=Path, default=Path("hybrid-caviar-v1.jsonl"))
    args = parser.parse_args()
    scenes = args.scene or list(DEFAULT_SCENES)
    if args.repetitions < 1:
        raise SystemExit("--repetitions must be >= 1")
    if not args.campaign and len(scenes) > 1:
        scenes = scenes[:1]
    campaign(scenes, args.repetitions, args.cache_dir, args.output)


if __name__ == "__main__":
    main()
