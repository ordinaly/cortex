"""Open-World Robustness v1 benchmark-world generator.

This module defines the benchmark side only. It deliberately does not modify
Cortex or provide the hidden oracle fields to the learner-facing observation
stream.

Protocol ID: open-world-robustness-v1
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import re
from random import Random
from typing import Any, Iterable

PROTOCOL = "open-world-robustness-v1"
STAGES = ("A", "B", "C", "D", "E")
OFFICIAL_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


def derive_seed(material: str, stage: str, index: int) -> int:
    """Derive a reproducible 63-bit seed from frozen public material."""
    if stage not in STAGES:
        raise ValueError(f"unknown stage: {stage}")
    if index < 0:
        raise ValueError("index must be non-negative")
    payload = f"{PROTOCOL}|{stage}|{index}|{material}".encode("utf-8")
    return int.from_bytes(sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)


def validate_official_material(material: str) -> str:
    material = str(material).strip().lower()
    if not OFFICIAL_SHA_RE.fullmatch(material):
        raise ValueError(
            "official seed material must be the 40-hex merge commit SHA "
            "of the frozen benchmark/candidate"
        )
    return material


@dataclass(frozen=True, slots=True)
class Observation:
    event_id: str
    event_time: int
    arrival_time: int
    source_id: str
    context: tuple[str, ...]
    observed_relation: str | None
    duplicate_of: str | None = None
    bootstrap: bool = False
    tags: tuple[str, ...] = ()

    def public_dict(self) -> dict[str, Any]:
        # Tags are benchmark-harness metadata and are never learner-facing.
        return {
            "event_id": self.event_id,
            "event_time": self.event_time,
            "arrival_time": self.arrival_time,
            "source_id": self.source_id,
            "context": self.context,
            "observed_relation": self.observed_relation,
            "duplicate_of": self.duplicate_of,
            "bootstrap": self.bootstrap,
        }


@dataclass(frozen=True, slots=True)
class Query:
    query_id: str
    query_time: int
    context: tuple[str, ...]
    expected_relation: str | None
    kind: str
    answerable: bool = True
    tags: tuple[str, ...] = ()

    def public_dict(self) -> dict[str, Any]:
        # Expected answers, answerability, kind, and tags are evaluator-only.
        return {
            "query_id": self.query_id,
            "query_time": self.query_time,
            "context": self.context,
        }


@dataclass(frozen=True)
class HiddenWorld:
    protocol: str
    stage: str
    seed: int
    observations: tuple[Observation, ...]
    queries: tuple[Query, ...]
    oracle: dict[str, Any]

    def public_payload(self) -> dict[str, Any]:
        return {
            "protocol": self.protocol,
            "stage": self.stage,
            "seed": self.seed,
            "observations": [row.public_dict() for row in self.observations],
            "queries": [row.public_dict() for row in self.queries],
        }

    def fingerprint(self) -> str:
        encoded = json.dumps(
            self.public_payload(),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return sha256(encoded).hexdigest()

    def oracle_fingerprint(self) -> str:
        encoded = json.dumps(
            self.oracle,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return sha256(encoded).hexdigest()


def _opaque_tokens(rng: Random, prefix: str, count: int) -> tuple[str, ...]:
    values = [f"{prefix}{rng.randrange(10**9):09d}" for _ in range(count)]
    if len(set(values)) != len(values):
        # Vanishingly unlikely, but benchmark identity must be injective.
        return _opaque_tokens(rng, prefix, count)
    return tuple(values)


def _flip_relation(rng: Random, truth: str, alternatives: tuple[str, ...]) -> str:
    choices = tuple(value for value in alternatives if value != truth)
    return truth if not choices else rng.choice(choices)


def _report(
    rng: Random,
    truth: str,
    relation_vocab: tuple[str, ...],
    reliability: float,
) -> str:
    return truth if rng.random() < reliability else _flip_relation(
        rng, truth, relation_vocab
    )


def _stage_a(seed: int) -> HiddenWorld:
    """Hidden source reliability, copying, and source drift."""
    rng = Random(seed)
    relations = _opaque_tokens(rng, "R", 3)
    contexts = tuple((_opaque_tokens(rng, "C", 1)[0],) for _ in range(12))
    sources = _opaque_tokens(rng, "S", 5)

    truth_by_context = {
        context: relations[index % len(relations)]
        for index, context in enumerate(contexts)
    }
    # Contextual reliability. S0/S4 are strong independent sources; S1-3 are a
    # weak copied cluster. S0 later degrades on family B.
    reliabilities = {
        sources[0]: {"A": 0.95, "B"