#!/usr/bin/env python3
"""Camera-derived chronological identity benchmark for Cortex.

Protocol: camera-v1

This is the first bridge from synthetic stabilization probes to real camera
observations. It intentionally does *not* claim raw-pixel vision: it consumes
per-frame structured observations derived from the CAVIAR fixed-camera corpus
and distributed by the public Prob-EC repository. Ground-truth track IDs are
used only for post-step evaluation and are never included in Cortex features.

The external corpus is pinned to an immutable commit. No dataset files are
vendored into Cortex.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import statistics
import sys
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from provenance import benchmark_provenance

PROTOCOL = "camera-v1"
FEATURE_DIM = 16
MAX_ENTITIES = 256
EXTERNAL_REPO = "jasonfilippou/Prob-EC"
EXTERNAL_SHA = "c6b1a8d4aaf82e7b44efb3329d425cdbf92e0f31"
DATASET_VARIANT = "clean"
DEFAULT_SCENES = (
    "01-Walk1",
    "02-Walk2",
    "08-Browse_WhileWaiting1",
    "19-Meet_WalkTogether1",
    "23-Meet_Crowd",
)

MOVE_RE = re.compile(
    r"happensAt\(\s*([A-Za-z_]+)\(\s*(id\d+)\s*\)\s*,\s*(\d+)\s*\)"
)
COORD_RE = re.compile(
    r"holdsAt\(\s*coord\(\s*(id\d+)\s*\)\s*=\s*\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)\s*,\s*(\d+)\s*\)"
)
ORIENT_RE = re.compile(
    r"holdsAt\(\s*orientation\(\s*(id\d+)\s*\)\s*=\s*(-?\d+(?:\.\d+)?)\s*,\s*(\d+)\s*\)"
)
APPEAR_RE = re.compile(
    r"holdsAt\(\s*appearance\(\s*(id\d+)\s*\)\s*=\s*([A-Za-z_]+)\s*,\s*(\d+)\s*\)"
)


@dataclass
class Observation:
    truth_id: str
    time: int
    x: float
    y: float
    movement: str | None = None
    orientation: float | None = None
    appearance: str | None = None


def _http_text(url: str) -> str:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "cortex-camera-v1",
            "Accept": "application/vnd.github+json",
        },
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read().decode("utf-8")


def _scene_listing(scene: str) -> list[dict[str, Any]]:
    quoted_scene = urllib.parse.quote(scene, safe="")
    url = (
        f"https://api.github.com/repos/{EXTERNAL_REPO}/contents/"
        f"dataset/{DATASET_VARIANT}/{quoted_scene}?ref={EXTERNAL_SHA}"
    )
    data = json.loads(_http_text(url))
    if not isinstance(data, list):
        raise RuntimeError(f"unexpected GitHub listing for {scene}: {type(data).__name__}")
    return data


def _pick_source_files(scene: str) -> tuple[str, str]:
    names = [str(item["name"]) for item in _scene_listing(scene)]
    movement = next((n for n in names if n.endswith("MovementIndv_new.pl")), None)
    if movement is None:
        movement = next((n for n in names if n.endswith("MovementIndv.pl")), None)
    appearance = next((n for n in names if n.endswith("AppearenceIndv.pl")), None)
    if movement is None or appearance is None:
        raise RuntimeError(
            f"missing individual movement/appearance sources for {scene}: "
            f"movement={movement!r}, appearance={appearance!r}"
        )
    return movement, appearance


def _raw_url(scene: str, filename: str) -> str:
    scene_q = urllib.parse.quote(scene, safe="")
    file_q = urllib.parse.quote(filename, safe="")
    return (
        f"https://raw.githubusercontent.com/{EXTERNAL_REPO}/{EXTERNAL_SHA}/"
        f"dataset/{DATASET_VARIANT}/{scene_q}/{file_q}"
    )


def _load_cached(cache_dir: Path, scene: str, filename: str) -> str:
    scene_dir = cache_dir / scene
    scene_dir.mkdir(parents=True, exist_ok=True)
    path = scene_dir / filename
    if not path.exists():
        path.write_text(_http_text(_raw_url(scene, filename)), encoding="utf-8")
    return path.read_text(encoding="utf-8")


def load_scene(scene: str, cache_dir: Path) -> list[tuple[int, list[Observation]]]:
    movement_name, appearance_name = _pick_source_files(scene)
    movement_text = _load_cached(cache_dir, scene, movement_name)
    appearance_text = _load_cached(cache_dir, scene, appearance_name)

    by_key: dict[tuple[int, str], Observation] = {}
    movement_by_key: dict[tuple[int, str], str] = {}
    for match in MOVE_RE.finditer(movement_text):
        movement, truth_id, time_s = match.groups()
        movement_by_key[(int(time_s), truth_id)] = movement.lower()
    for match in COORD_RE.finditer(movement_text):
        truth_id, x_s, y_s, time_s = match.groups()
        time = int(time_s)
        key = (time, truth_id)
        by_key[key] = Observation(
            truth_id=truth_id,
            time=time,
            x=float(x_s),
            y=float(y_s),
            movement=movement_by_key.get(key),
        )

    for match in ORIENT_RE.finditer(appearance_text):
        truth_id, angle_s, time_s = match.groups()
        obs = by_key.get((int(time_s), truth_id))
        if obs is not None:
            obs.orientation = float(angle_s)
    for match in APPEAR_RE.finditer(appearance_text):
        truth_id, state, time_s = match.groups()
        obs = by_key.get((int(time_s), truth_id))
        if obs is not None:
            obs.appearance = state.lower()

    frames: dict[int, list[Observation]] = defaultdict(list)
    for obs in by_key.values():
        frames[obs.time].append(obs)
    ordered = [(time, frames[time]) for time in sorted(frames)]
    if not ordered:
        raise RuntimeError(f"no camera observations parsed for {scene}")
    return ordered


def camera_feature(obs: Observation) -> list[float]:
    """Causal 16-D feature vector built only from the current detection.

    CAVIAR's annotation coordinates are approximately half-resolution camera
    coordinates. Fixed camera reference scales (192 x 144) are used a priori;
    there is no per-scene fitting and no future-data normalization.

    Track IDs are deliberately absent. We also avoid velocity features because
    calculating them from these files would require using the ground-truth ID
    to join observations across frames, which would leak the answer.
    """
    x = (obs.x - 96.0) / 96.0
    y = (obs.y - 72.0) / 72.0
    angle = math.radians(obs.orientation) if obs.orientation is not None else 0.0
    movement = obs.movement or "unknown"
    appearance = obs.appearance or "unknown"
    return [
        x,
        y,
        x * x,
        y * y,
        math.sin(math.pi * x),
        math.cos(math.pi * x),
        math.sin(math.pi * y),
        math.cos(math.pi * y),
        math.sin(angle),
        math.cos(angle),
        float(movement == "walking"),
        float(movement == "running"),
        float(movement == "active"),
        float(movement == "inactive"),
        float(appearance == "visible"),
        float(appearance in {"appear", "disappear", "occluded"}),
    ]


def cfg_hash(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def make_runtime(group_mode: str):
    from cortex import CortexRuntime

    seed = CortexRuntime(
        feature_dim=FEATURE_DIM,
        max_entities=MAX_ENTITIES,
        budget=24,
    )
    cfg = json.loads(seed.config_json())
    if group_mode == "identity-only":
        # Mixed camera coordinates/state features do not have a meaningful
        # cyclic feature-permutation nuisance action. This is a supported
        # domain configuration, not a changed matching threshold.
        cfg["articulation"]["group_candidates"] = [[0]]
    elif group_mode != "default":
        raise ValueError(f"unknown group mode {group_mode!r}")
    config_json = json.dumps(cfg, separators=(",", ":"), sort_keys=True)
    return config_json, CortexRuntime(config_json=config_json)


def active_entities(runtime: Any) -> int:
    return int(json.loads(runtime.snapshot_json())["articulation"]["active_entities"])


def identity_metrics(records: list[tuple[str, int]]) -> dict[str, float]:
    if not records:
        return {
            "purity": 0.0,
            "fragmentation": 0.0,
            "dominant_share": 0.0,
            "switch_rate": 0.0,
            "max_fragmentation": 0.0,
        }
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
    fragment_counts = [len(c) for c in by_truth.values()]
    return {
        "purity": sum(max(c.values()) for c in by_binding.values()) / len(records),
        "fragmentation": statistics.fmean(fragment_counts),
        "dominant_share": statistics.fmean(
            max(c.values()) / sum(c.values()) for c in by_truth.values()
        ),
        "switch_rate": switches / max(1, comparable),
        "max_fragmentation": float(max(fragment_counts)),
    }


def _inventory_windows(active_trace: list[int], n_windows: int = 10) -> tuple[list[int], list[int]]:
    if not active_trace:
        return [], []
    checkpoints = []
    for i in range(1, n_windows + 1):
        index = min(len(active_trace) - 1, math.ceil(i * len(active_trace) / n_windows) - 1)
        checkpoints.append(active_trace[index])
    increments = []
    previous = 0
    for value in checkpoints:
        increments.append(value - previous)
        previous = value
    return checkpoints, increments


def run_scene(
    scene: str,
    frames: list[tuple[int, list[Observation]]],
    group_mode: str,
    repetition: int,
    base_seed: int,
) -> dict[str, Any]:
    seed = base_seed + repetition * 1009
    rng = random.Random(seed)
    cfg, runtime = make_runtime(group_mode)
    records: list[tuple[str, int]] = []
    active_trace: list[int] = []
    truth_ids: set[str] = set()
    max_covisible = 0
    injectivity_violations = 0
    subgroup_changes = 0
    previous_subgroup: tuple[int, ...] | None = None
    last_seen: dict[str, tuple[int, int]] = {}
    gap_counts = {10: 0, 25: 0, 100: 0}
    gap_same = {10: 0, 25: 0, 100: 0}
    error = None

    for frame_index, (_time, observations) in enumerate(frames):
        shuffled = list(observations)
        # Prevent source annotation ordering (which often follows truth ID) from
        # becoming a hidden cue. The permutation is independent of truth IDs.
        rng.shuffle(shuffled)
        xs = [camera_feature(obs) for obs in shuffled]
        try:
            read = runtime.step(xs, [], None, [], None)
        except Exception as exc:  # research benchmark: record model failure
            error = f"{type(exc).__name__}: {exc}"
            break

        bindings = list(map(int, read.bindings))
        if len(bindings) != len(set(bindings)):
            injectivity_violations += 1
        max_covisible = max(max_covisible, len(shuffled))
        subgroup = tuple(map(int, read.subgroup))
        if previous_subgroup is not None and subgroup != previous_subgroup:
            subgroup_changes += 1
        previous_subgroup = subgroup

        for obs, binding in zip(shuffled, bindings):
            truth_ids.add(obs.truth_id)
            records.append((obs.truth_id, binding))
            previous = last_seen.get(obs.truth_id)
            if previous is not None:
                previous_frame, previous_binding = previous
                gap = frame_index - previous_frame
                for threshold in gap_counts:
                    if gap >= threshold:
                        gap_counts[threshold] += 1
                        gap_same[threshold] += int(binding == previous_binding)
            last_seen[obs.truth_id] = (frame_index, binding)
        active_trace.append(active_entities(runtime))

    metrics = identity_metrics(records)
    checkpoints, increments = _inventory_windows(active_trace)
    final_active = active_entities(runtime)
    return {
        **benchmark_provenance(PROTOCOL),
        "external_dataset": "CAVIAR via Prob-EC",
        "external_repo": EXTERNAL_REPO,
        "external_commit": EXTERNAL_SHA,
        "dataset_variant": DATASET_VARIANT,
        "scene": scene,
        "group_mode": group_mode,
        "repetition": repetition,
        "order_seed": seed,
        "config_sha256": cfg_hash(cfg),
        "feature_contract": "current-frame geometry+orientation+state; no track history; no GT ID input",
        "raw_pixels": False,
        "capacity_failure": error is not None,
        "model_error": error,
        "frames": len(active_trace),
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
        "purity": metrics["purity"],
        "fragmentation": metrics["fragmentation"],
        "max_fragmentation": metrics["max_fragmentation"],
        "dominant_share": metrics["dominant_share"],
        "switch_rate": metrics["switch_rate"],
        "gap_return_10_count": gap_counts[10],
        "gap_return_10_same_fraction": gap_same[10] / max(1, gap_counts[10]),
        "gap_return_25_count": gap_counts[25],
        "gap_return_25_same_fraction": gap_same[25] / max(1, gap_counts[25]),
        "gap_return_100_count": gap_counts[100],
        "gap_return_100_same_fraction": gap_same[100] / max(1, gap_counts[100]),
    }


def campaign(scenes: list[str], repetitions: int, cache_dir: Path, output: Path) -> None:
    loaded: dict[str, list[tuple[int, list[Observation]]]] = {}
    for scene in scenes:
        print(f"loading {scene}", file=sys.stderr, flush=True)
        loaded[scene] = load_scene(scene, cache_dir)

    rows: list[dict[str, Any]] = []
    total = len(scenes) * 2 * repetitions
    index = 0
    for scene in scenes:
        for group_mode in ("default", "identity-only"):
            for rep in range(repetitions):
                print(
                    f"[{index + 1:02d}/{total:02d}] {scene} {group_mode} rep={rep + 1}",
                    file=sys.stderr,
                    flush=True,
                )
                rows.append(
                    run_scene(
                        scene,
                        loaded[scene],
                        group_mode,
                        rep,
                        base_seed=20260917,
                    )
                )
                index += 1
    output.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", action="store_true")
    parser.add_argument("--scene", action="append", default=[])
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--cache-dir", type=Path, default=Path(".camera-v1-cache"))
    parser.add_argument("--output", type=Path, default=Path("camera-v1.jsonl"))
    args = parser.parse_args()
    scenes = args.scene or list(DEFAULT_SCENES)
    if args.repetitions < 1:
        raise SystemExit("--repetitions must be >= 1")
    if not args.campaign and len(scenes) != 1:
        scenes = scenes[:1]
    campaign(scenes, args.repetitions, args.cache_dir, args.output)


if __name__ == "__main__":
    main()
