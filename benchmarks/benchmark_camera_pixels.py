#!/usr/bin/env python3
"""Raw-pixel appearance benchmark for Cortex on CAVIAR.

Protocol: camera-pixels-v1

This test keeps object detection oracle-controlled with CAVIAR's hand-labelled
bounding boxes, but derives every Cortex observation from the pixels inside the
box. Ground-truth object IDs are used only after inference for scoring.

Descriptor contract (frozen before observing benchmark results):
- 16-bin HSV hue histogram over the current bounding-box crop;
- each pixel weighted by saturation * (0.25 + 0.75 * value);
- probability histogram p mapped to x_i = 4 * sqrt(p_i).

With Cortex's weighted MSE, this embedding makes squared distance proportional
to Hellinger distance between hue distributions. A cyclic feature shift is also
a genuine circular hue rotation, giving the articulation nuisance action a
meaningful image-domain interpretation without changing Cortex thresholds.

This is still not end-to-end detection or learned vision: manual GT boxes are
an oracle detector and the pixel descriptor is hand-designed.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import random
import re
import statistics
import sys
import tarfile
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from provenance import benchmark_provenance

PROTOCOL = "camera-pixels-v1"
FEATURE_DIM = 16
MAX_ENTITIES = 256
HIST_SCALE = 4.0
OFFICIAL_BASE = "https://homepages.inf.ed.ac.uk/rbf/CAVIARDATA1"
GT_REPO = "jasonfilippou/Prob-EC"
GT_SHA = "c6b1a8d4aaf82e7b44efb3329d425cdbf92e0f31"

SCENES = {
    "Walk2": {
        "derived_dir": "02-Walk2",
        "xml": "wk2gt.xml",
        "archive": "Walk2_jpg.tar.gz",
    },
    "Meet_WalkTogether1": {
        "derived_dir": "19-Meet_WalkTogether1",
        "xml": "mwt1gt.xml",
        "archive": "Meet_WalkTogether1_jpg.tar.gz",
    },
    "Meet_Crowd": {
        "derived_dir": "23-Meet_Crowd",
        "xml": "mc1gt.xml",
        "archive": "Meet_Crowd_jpg.tar.gz",
    },
}
DEFAULT_SCENES = tuple(SCENES)


@dataclass(frozen=True)
class BoxTruth:
    truth_id: str
    xc: float
    yc: float
    width: float
    height: float


def _download(url: str, path: Path) -> None:
    if path.exists() and path.stat().st_size:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    retry_codes = {403, 429, 500, 502, 503, 504}
    last_error: Exception | None = None
    for attempt in range(5):
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (X11; Linux x86_64) "
                    "AppleWebKit/537.36 Chrome/140 Safari/537.36 "
                    "CortexResearch/1.0"
                ),
                "Accept": "*/*",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=90) as response:
                data = response.read()
            path.write_bytes(data)
            return
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code not in retry_codes or attempt == 4:
                raise
        except urllib.error.URLError as exc:
            last_error = exc
            if attempt == 4:
                raise
        time.sleep(2 ** attempt)
    if last_error is not None:
        raise last_error


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _ground_truth_url(scene: str) -> str:
    cfg = SCENES[scene]
    return (
        f"https://raw.githubusercontent.com/{GT_REPO}/{GT_SHA}/"
        f"dataset/clean/{cfg['derived_dir']}/{cfg['xml']}"
    )


def _archive_url(scene: str) -> str:
    cfg = SCENES[scene]
    return f"{OFFICIAL_BASE}/{scene}/{cfg['archive']}"


def load_ground_truth(scene: str, cache_dir: Path) -> dict[int, list[BoxTruth]]:
    cfg = SCENES[scene]
    xml_path = cache_dir / scene / str(cfg["xml"])
    _download(_ground_truth_url(scene), xml_path)
    root = ET.parse(xml_path).getroot()
    frames: dict[int, list[BoxTruth]] = defaultdict(list)
    for frame in root.iter():
        if frame.tag.lower().endswith("frame") and "number" in frame.attrib:
            number = int(frame.attrib["number"])
            for obj in frame.iter():
                if not obj.tag.lower().endswith("object") or "id" not in obj.attrib:
                    continue
                box = next(
                    (node for node in obj.iter() if node.tag.lower().endswith("box")),
                    None,
                )
                if box is None:
                    continue
                try:
                    frames[number].append(
                        BoxTruth(
                            truth_id=str(obj.attrib["id"]),
                            xc=float(box.attrib["xc"]),
                            yc=float(box.attrib["yc"]),
                            width=float(box.attrib["w"]),
                            height=float(box.attrib["h"]),
                        )
                    )
                except KeyError:
                    continue
    if not frames:
        raise RuntimeError(f"no individual boxes parsed from {xml_path}")
    return dict(frames)


def _safe_image_members(archive_path: Path) -> list[tuple[str, bytes]]:
    out: list[tuple[str, bytes]] = []
    with tarfile.open(archive_path, "r:gz") as tf:
        for member in tf.getmembers():
            if not member.isfile() or not member.name.lower().endswith((".jpg", ".jpeg")):
                continue
            extracted = tf.extractfile(member)
            if extracted is None:
                continue
            out.append((member.name, extracted.read()))
    if not out:
        raise RuntimeError(f"no JPEG images found in {archive_path}")
    return out


def _trailing_number(name: str) -> int | None:
    stem = Path(name).stem
    matches = re.findall(r"(\d+)", stem)
    return int(matches[-1]) if matches else None


def map_images_to_frames(
    images: list[tuple[str, bytes]], gt_frames: set[int]
) -> tuple[dict[int, bytes], str, int]:
    numbered = [(_trailing_number(name), data) for name, data in images]
    if all(number is not None for number, _ in numbered):
        by_number = {int(number): data for number, data in numbered if number is not None}
        # Archives have historically used both zero- and one-based names. Infer
        # only this indexing offset; no visual or identity information is used.
        candidates = range(-3, 4)
        offset = max(candidates, key=lambda off: sum((f + off) in by_number for f in gt_frames))
        coverage = sum((f + offset) in by_number for f in gt_frames)
        if coverage:
            return ({f: by_number[f + offset] for f in gt_frames if f + offset in by_number}, "filename", offset)

    # Fallback: align sorted images to the contiguous frame-number range. This
    # uses frame indices only and is reported explicitly in provenance.
    images_sorted = sorted(images, key=lambda item: item[0])
    min_frame = min(gt_frames)
    mapping = {min_frame + i: data for i, (_name, data) in enumerate(images_sorted)}
    return ({f: mapping[f] for f in gt_frames if f in mapping}, "sorted", 0)


def load_scene(scene: str, cache_dir: Path) -> tuple[list[tuple[int, bytes, list[BoxTruth]]], dict[str, Any]]:
    gt = load_ground_truth(scene, cache_dir)
    cfg = SCENES[scene]
    archive_path = cache_dir / scene / str(cfg["archive"])
    _download(_archive_url(scene), archive_path)
    images = _safe_image_members(archive_path)
    image_map, alignment_mode, alignment_offset = map_images_to_frames(images, set(gt))
    rows = [(frame, image_map[frame], gt[frame]) for frame in sorted(gt) if frame in image_map]
    if not rows:
        raise RuntimeError(f"no aligned annotated pixel frames for {scene}")
    provenance = {
        "image_source": _archive_url(scene),
        "image_archive_sha256": sha256_file(archive_path),
        "gt_source": _ground_truth_url(scene),
        "gt_sha256": sha256_file(cache_dir / scene / str(cfg["xml"])),
        "image_alignment_mode": alignment_mode,
        "image_alignment_offset": alignment_offset,
        "annotated_frames_in_gt": len(gt),
        "annotated_frames_with_image": len(rows),
        "archive_images": len(images),
    }
    return rows, provenance


def hue_descriptor(image_bytes: bytes, box: BoxTruth) -> list[float]:
    with Image.open(io.BytesIO(image_bytes)) as image:
        image = image.convert("HSV")
        width, height = image.size
        x0 = max(0, int(math.floor(box.xc - box.width / 2.0)))
        y0 = max(0, int(math.floor(box.yc - box.height / 2.0)))
        x1 = min(width, int(math.ceil(box.xc + box.width / 2.0)))
        y1 = min(height, int(math.ceil(box.yc + box.height / 2.0)))
        if x1 <= x0 or y1 <= y0:
            return [HIST_SCALE / math.sqrt(FEATURE_DIM)] * FEATURE_DIM
        crop = np.asarray(image.crop((x0, y0, x1, y1)), dtype=np.float64)
    hue = crop[..., 0]
    sat = crop[..., 1] / 255.0
    val = crop[..., 2] / 255.0
    weights = sat * (0.25 + 0.75 * val)
    bins = np.minimum((hue * FEATURE_DIM / 256.0).astype(np.int64), FEATURE_DIM - 1)
    hist = np.bincount(bins.ravel(), weights=weights.ravel(), minlength=FEATURE_DIM).astype(np.float64)
    total = float(hist.sum())
    if total <= 1.0e-12:
        p = np.full(FEATURE_DIM, 1.0 / FEATURE_DIM, dtype=np.float64)
    else:
        p = hist / total
    return (HIST_SCALE * np.sqrt(p)).tolist()


def make_runtime(group_mode: str):
    from cortex import CortexRuntime

    seed = CortexRuntime(feature_dim=FEATURE_DIM, max_entities=MAX_ENTITIES, budget=24)
    cfg = json.loads(seed.config_json())
    if group_mode == "identity-only":
        cfg["articulation"]["group_candidates"] = [[0]]
    elif group_mode != "default":
        raise ValueError(group_mode)
    config_json = json.dumps(cfg, separators=(",", ":"), sort_keys=True)
    return config_json, CortexRuntime(config_json=config_json)


def active_entities(runtime: Any) -> int:
    return int(json.loads(runtime.snapshot_json())["articulation"]["active_entities"])


def identity_metrics(records: list[tuple[str, int]]) -> dict[str, float]:
    by_binding: dict[int, Counter[str]] = defaultdict(Counter)
    by_truth: dict[str, Counter[int]] = defaultdict(Counter)
    last: dict[str, int] = {}
    switches = 0
    comparable = 0
    for truth, binding in records:
        by_binding[binding][truth] += 1
        by_truth[truth][binding] += 1
        if truth in last:
            comparable += 1
            switches += int(last[truth] != binding)
        last[truth] = binding
    if not records:
        return {"purity": 0.0, "fragmentation": 0.0, "dominant_share": 0.0, "switch_rate": 0.0, "max_fragmentation": 0.0}
    fragments = [len(c) for c in by_truth.values()]
    return {
        "purity": sum(max(c.values()) for c in by_binding.values()) / len(records),
        "fragmentation": statistics.fmean(fragments),
        "dominant_share": statistics.fmean(max(c.values()) / sum(c.values()) for c in by_truth.values()),
        "switch_rate": switches / max(1, comparable),
        "max_fragmentation": float(max(fragments)),
    }


def inventory_windows(trace: list[int], n: int = 10) -> tuple[list[int], list[int]]:
    if not trace:
        return [], []
    checkpoints = [trace[min(len(trace) - 1, math.ceil(i * len(trace) / n) - 1)] for i in range(1, n + 1)]
    previous = 0
    increments = []
    for value in checkpoints:
        increments.append(value - previous)
        previous = value
    return checkpoints, increments


def run_scene(
    scene: str,
    frames: list[tuple[int, bytes, list[BoxTruth]]],
    source_provenance: dict[str, Any],
    group_mode: str,
    repetition: int,
) -> dict[str, Any]:
    seed = 20260917 + repetition * 1009
    rng = random.Random(seed)
    config_json, runtime = make_runtime(group_mode)
    records: list[tuple[str, int]] = []
    inventory: list[int] = []
    truth_ids: set[str] = set()
    max_covisible = 0
    injectivity_violations = 0
    error = None
    subgroup_changes = 0
    previous_subgroup: tuple[int, ...] | None = None
    last_seen: dict[str, tuple[int, int]] = {}
    gap_counts = {10: 0, 25: 0, 100: 0}
    gap_same = {10: 0, 25: 0, 100: 0}

    for chronological_index, (_frame, image_bytes, boxes) in enumerate(frames):
        detections = [(box, hue_descriptor(image_bytes, box)) for box in boxes]
        rng.shuffle(detections)
        try:
            read = runtime.step([x for _box, x in detections], [], None, [], None)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            break
        bindings = list(map(int, read.bindings))
        if len(bindings) != len(set(bindings)):
            injectivity_violations += 1
        max_covisible = max(max_covisible, len(bindings))
        subgroup = tuple(map(int, read.subgroup))
        if previous_subgroup is not None and subgroup != previous_subgroup:
            subgroup_changes += 1
        previous_subgroup = subgroup

        for (box, _x), binding in zip(detections, bindings):
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
        inventory.append(active_entities(runtime))

    metrics = identity_metrics(records)
    checkpoints, increments = inventory_windows(inventory)
    final_active = active_entities(runtime)
    return {
        **benchmark_provenance(PROTOCOL),
        **source_provenance,
        "scene": scene,
        "group_mode": group_mode,
        "repetition": repetition,
        "order_seed": seed,
        "config_sha256": hashlib.sha256(config_json.encode()).hexdigest(),
        "feature_contract": "16-bin saturation-weighted HSV hue histogram; x=4*sqrt(p); current GT crop pixels only",
        "raw_pixels": True,
        "oracle_detection": True,
        "learned_visual_model": False,
        "histogram_scale": HIST_SCALE,
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
        "subgroup_changes": subgroup_changes,
        "final_subgroup": list(previous_subgroup or ()),
        **metrics,
        "gap_return_10_count": gap_counts[10],
        "gap_return_10_same_fraction": gap_same[10] / max(1, gap_counts[10]),
        "gap_return_25_count": gap_counts[25],
        "gap_return_25_same_fraction": gap_same[25] / max(1, gap_counts[25]),
        "gap_return_100_count": gap_counts[100],
        "gap_return_100_same_fraction": gap_same[100] / max(1, gap_counts[100]),
    }


def campaign(scenes: list[str], repetitions: int, cache_dir: Path, output: Path) -> None:
    rows: list[dict[str, Any]] = []
    total = len(scenes) * 2 * repetitions
    n = 0
    for scene in scenes:
        print(f"loading raw pixels: {scene}", file=sys.stderr, flush=True)
        frames, provenance = load_scene(scene, cache_dir)
        for group_mode in ("default", "identity-only"):
            for repetition in range(repetitions):
                n += 1
                print(f"[{n:02d}/{total:02d}] {scene} {group_mode} rep={repetition + 1}", file=sys.stderr, flush=True)
                rows.append(run_scene(scene, frames, provenance, group_mode, repetition))
    output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", action="store_true")
    parser.add_argument("--scene", action="append", choices=sorted(SCENES), default=[])
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--cache-dir", type=Path, default=Path(".camera-pixels-v1-cache"))
    parser.add_argument("--output", type=Path, default=Path("camera-pixels-v1.jsonl"))
    args = parser.parse_args()
    scenes = args.scene or list(DEFAULT_SCENES)
    if args.repetitions < 1:
        raise SystemExit("--repetitions must be >= 1")
    if not args.campaign and len(scenes) > 1:
        scenes = scenes[:1]
    campaign(scenes, args.repetitions, args.cache_dir, args.output)


if __name__ == "__main__":
    main()
