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

from PIL import Image

from benchmark_camera_pixels import (
    DEFAULT_SCENES,
    BoxTruth,
    active_entities,
    identity_metrics,
    inventory_windows,
    load_scene,
)
from provenance import benchmark_provenance

from cortex import HybridCortexRuntime, NeuralObservation

PROTOCOL = "hybrid-caviar-v1"
ENCODER_NAME = "torchvision-resnet18-imagenet-default"
EMBEDDING_DIM = 512
MAX_ENTITIES = 256


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


def run_scene(
    scene: str,
    encoded_frames,
    source_provenance: dict[str, Any],
    repetition: int,
    torch_version: str,
    torchvision_version: str,
    weights_name: str,
) -> dict[str, Any]:
    seed = 20260918 + repetition * 1009
    rng = random.Random(seed)
    hybrid = HybridCortexRuntime(
        embedding_dim=EMBEDDING_DIM,
        max_entities=MAX_ENTITIES,
        budget=24,
        nuisance_mode="identity-only",
    )

    records: list[tuple[str, int]] = []
    inventory: list[int] = []
    truth_ids: set[str] = set()
    injectivity_violations = 0
    max_covisible = 0
    last_seen: dict[str, tuple[int, int]] = {}
    gap_counts = {10: 0, 25: 0, 100: 0}
    gap_same = {10: 0, 25: 0, 100: 0}
    error = None

    for chronological_index, (frame_number, detections) in enumerate(encoded_frames):
        shuffled = list(detections)
        rng.shuffle(shuffled)
        observations = [
            NeuralObservation(
                embedding=embedding,
                bbox=(box.xc, box.yc, box.width, box.height),
                timestamp=frame_number,
                source=scene,
            )
            for box, embedding in shuffled
        ]
        try:
            read = hybrid.step(observations)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            break

        bindings = list(map(int, read.bindings))
        if len(bindings) != len(set(bindings)):
            injectivity_violations += 1
        max_covisible = max(max_covisible, len(bindings))
        for (box, _embedding), binding in zip(shuffled, bindings):
            truth_ids.add(box.truth_id)
            records.append((box.truth_id, binding))
            previous = last_seen.get(box.truth_id)
            if previous is not None:
                previous_index, previous_binding = previous
                gap = chronological_index - previous_index
                for threshold in gap_counts:
                    if gap >= threshold:
                        gap_counts[threshold] += 1
                        gap_same[threshold] += int(binding == previous_binding)
            last_seen[box.truth_id] = (chronological_index, binding)
        inventory.append(active_entities(hybrid.runtime))

    metrics = identity_metrics(records)
    checkpoints, increments = inventory_windows(inventory)
    final_active = active_entities(hybrid.runtime)
    config_json = hybrid.config_json()
    return {
        **benchmark_provenance(PROTOCOL),
        **source_provenance,
        "scene": scene,
        "repetition": repetition,
        "order_seed": seed,
        "encoder": ENCODER_NAME,
        "encoder_weights": weights_name,
        "torch_version": torch_version,
        "torchvision_version": torchvision_version,
        "embedding_dim": EMBEDDING_DIM,
        "metric_adapter": "l2-normalize;sqrt(d/2)-scale;Cortex-MSE=1-cosine",
        "nuisance_mode": "identity-only",
        "frozen_encoder": True,
        "oracle_detection": True,
        "ground_truth_input": False,
        "config_sha256": hashlib.sha256(config_json.encode()).hexdigest(),
        "capacity_failure": error is not None,
        "model_error": error,
        "frames": len(inventory),
        "observations": len(records),
        "true_tracks": len(truth_ids),
        "max_covisible": max_covisible,
        "active_entities": final_active,
        "oversegmentation_ratio": final_active / max(1, len(truth_ids)),
        "inventory_checkpoints": checkpoints,
        "new_entities_by_window": increments,
        "injectivity_violations": injectivity_violations,
        **metrics,
        "gap_return_10_count": gap_counts[10],
        "gap_return_10_same_fraction": gap_same[10] / max(1, gap_counts[10]),
        "gap_return_25_count": gap_counts[25],
        "gap_return_25_same_fraction": gap_same[25] / max(1, gap_counts[25]),
        "gap_return_100_count": gap_counts[100],
        "gap_return_100_same_fraction": gap_same[100] / max(1, gap_counts[100]),
    }


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
            rows.append(
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
