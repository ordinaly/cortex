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
        sources[0]: {"A": 0.95, "B": 0.93},
        sources[1]: {"A": 0.62, "B": 0.58},
        sources[2]: {"A": 0.60, "B": 0.56},
        sources[3]: {"A": 0.57, "B": 0.54},
        sources[4]: {"A": 0.88, "B": 0.91},
    }
    copy_parent = {
        sources[1]: sources[1],
        sources[2]: sources[1],
        sources[3]: sources[1],
    }
    drift_step = 112
    observations: list[Observation] = []
    event_id = 0
    for round_index in range(18):
        for context_index, context in enumerate(contexts):
            family = "A" if context_index < 6 else "B"
            truth = truth_by_context[context]
            reports: dict[str, str] = {}
            for source_index, source in enumerate(sources):
                reliability = reliabilities[source][family]
                if source == sources[0] and event_id >= drift_step and family == "B":
                    reliability = 0.34
                if source in (sources[2], sources[3]) and rng.random() < 0.82:
                    parent = copy_parent[source]
                    observed = reports.get(parent)
                    if observed is None:
                        observed = _report(
                            rng,
                            truth,
                            relations,
                            reliabilities[parent][family],
                        )
                else:
                    observed = _report(rng, truth, relations, reliability)
                reports[source] = observed
                observations.append(
                    Observation(
                        event_id=f"A{event_id:05d}",
                        event_time=event_id,
                        arrival_time=event_id,
                        source_id=source,
                        context=context,
                        observed_relation=observed,
                        tags=("source-discovery", family),
                    )
                )
                event_id += 1

    # Queries include deceptive contexts where copied majority can oppose truth.
    queries = tuple(
        Query(
            query_id=f"AQ{i:03d}",
            query_time=len(observations) - 1,
            context=context,
            expected_relation=truth_by_context[context],
            kind="source-weighted-current-truth",
            tags=("hidden-reliability",),
        )
        for i, context in enumerate(contexts)
    )
    oracle = {
        "relation_vocab": relations,
        "truth_by_context": {"|".join(k): v for k, v in truth_by_context.items()},
        "source_profiles": reliabilities,
        "copy_parent": copy_parent,
        "drift": {
            "source": sources[0],
            "family": "B",
            "after_event": drift_step,
            "reliability": 0.34,
        },
        "learner_receives_source_reliability": False,
    }
    return HiddenWorld(PROTOCOL, "A", seed, tuple(observations), queries, oracle)


def _stage_b(seed: int) -> HiddenWorld:
    """Delayed, missing, duplicated, and recurrent temporal evidence."""
    rng = Random(seed)
    relations = _opaque_tokens(rng, "R", 3)
    contexts = tuple((_opaque_tokens(rng, "T", 1)[0],) for _ in range(8))
    source = _opaque_tokens(rng, "S", 1)[0]
    schedule = (
        (0, 44, relations[0]),
        (45, 84, relations[1]),
        (85, 124, relations[2]),
        (125, 169, relations[0]),
    )
    observations: list[Observation] = []
    event_counter = 0
    duplicates = 0
    missing = 0
    delayed = 0

    def truth_at(t: int) -> str:
        for start, end, relation in schedule:
            if start <= t <= end:
                return relation
        raise AssertionError(t)

    for event_time in range(170):
        truth = truth_at(event_time)
        for context in contexts:
            eid = f"B{event_counter:05d}"
            event_counter += 1
            if rng.random() < 0.18:
                observed = None
                missing += 1
            else:
                observed = truth if rng.random() < 0.94 else _flip_relation(
                    rng, truth, relations
                )
            delay = 0 if rng.random() < 0.57 else rng.randint(1, 9)
            delayed += int(delay > 0)
            arrival = event_time + delay
            observations.append(
                Observation(
                    event_id=eid,
                    event_time=event_time,
                    arrival_time=arrival,
                    source_id=source,
                    context=context,
                    observed_relation=observed,
                    tags=("temporal-disorder",),
                )
            )
            if observed is not None and rng.random() < 0.07:
                duplicates += 1
                observations.append(
                    Observation(
                        event_id=f"{eid}D",
                        event_time=event_time,
                        arrival_time=arrival + rng.randint(1, 4),
                        source_id=source,
                        context=context,
                        observed_relation=observed,
                        duplicate_of=eid,
                        tags=("temporal-disorder", "duplicate"),
                    )
                )
    observations.sort(key=lambda row: (row.arrival_time, row.event_time, row.event_id))
    checkpoints = (35, 70, 110, 160)
    queries = tuple(
        Query(
            query_id=f"BQ{q:03d}_{i}",
            query_time=q,
            context=context,
            expected_relation=truth_at(q),
            kind="current-regime-under-temporal-disorder",
            tags=("recurrence" if q == 160 else "temporal",),
        )
        for q in checkpoints
        for i, context in enumerate(contexts)
    )
    oracle = {
        "relation_vocab": relations,
        "regime_schedule": schedule,
        "missing_events": missing,
        "delayed_events": delayed,
        "duplicate_events": duplicates,
        "event_time_distinct_from_arrival_time": True,
        "recurrence_relation": relations[0],
    }
    return HiddenWorld(PROTOCOL, "B", seed, tuple(observations), queries, oracle)


def _stage_c(seed: int) -> HiddenWorld:
    """Compound structural novelty and deliberately ambiguous explanations."""
    rng = Random(seed)
    relations = _opaque_tokens(rng, "R", 4)
    contexts = tuple((_opaque_tokens(rng, "N", 1)[0],) for _ in range(6))
    sources = _opaque_tokens(rng, "S", 2)
    observations: list[Observation] = []
    event_counter = 0

    # Stable prelude.
    for t in range(40):
        for i, context in enumerate(contexts[:4]):
            truth = relations[i % 2]
            observations.append(
                Observation(
                    event_id=f"C{event_counter:05d}",
                    event_time=t,
                    arrival_time=t,
                    source_id=sources[0],
                    context=context,
                    observed_relation=truth,
                    bootstrap=t < 4,
                    tags=("stable-prelude",),
                )
            )
            event_counter += 1

    # Compound novelty: new contexts and new relation while old contexts shift.
    for t in range(40, 70):
        for i, context in enumerate(contexts):
            if i < 2:
                truth = relations[2]  # regime component
                tags = ("compound", "new-regime")
            elif i < 4:
                truth = relations[i % 2]
                tags = ("compound", "stable-control")
            else:
                truth = relations[3]  # new type/context + new relation
                tags = ("compound", "new-context", "new-relation")
            observations.append(
                Observation(
                    event_id=f"C{event_counter:05d}",
                    event_time=t,
                    arrival_time=t,
                    source_id=sources[t % 2],
                    context=context,
                    observed_relation=truth,
                    tags=tags,
                )
            )
            event_counter += 1

    queries = (
        Query(
            "CQ000",
            69,
            contexts[0],
            relations[2],
            "compound-novelty",
            tags=("expected:new_regime", "expected:new_relation"),
        ),
        Query(
            "CQ001",
            69,
            contexts[4],
            relations[3],
            "compound-novelty",
            tags=("expected:new_entity_type", "expected:new_relation"),
        ),
        Query(
            "CQ002",
            52,
            contexts[5],
            None,
            "compound-ambiguity",
            answerable=False,
            tags=("correct:unresolved",),
        ),
    )
    oracle = {
        "expected_components": {
            "CQ000": ("new_regime", "new_relation"),
            "CQ001": ("new_entity_type", "new_relation"),
            "CQ002": ("new_entity_type", "new_regime"),
        },
        "ambiguous_queries": ("CQ002",),
        "single-label-attribution_is_insufficient": True,
    }
    return HiddenWorld(PROTOCOL, "C", seed, tuple(observations), queries, oracle)


def _stage_d(seed: int, *, scale: str) -> HiddenWorld:
    """Long distractor stream under bounded-memory pressure."""
    rng = Random(seed)
    length = 3_000 if scale == "development" else 20_000
    relation_vocab = _opaque_tokens(rng, "R", 5)
    salient_contexts = tuple((_opaque_tokens(rng, "K", 1)[0],) for _ in range(12))
    distractor_contexts = tuple(
        (_opaque_tokens(rng, "D", 1)[0],) for _ in range(320)
    )
    source = _opaque_tokens(rng, "S", 1)[0]
    stable_truth = {
        context: relation_vocab[i % 3]
        for i, context in enumerate(salient_contexts)
    }
    observations: list[Observation] = []

    for t in range(length):
        salient = rng.random() < 0.035
        if salient:
            context = rng.choice(salient_contexts)
            truth = stable_truth[context]
            # One salient family temporarily changes then returns.
            if context == salient_contexts[0] and length // 2 <= t < (3 * length) // 4:
                truth = relation_vocab[4]
            observed = truth if rng.random() < 0.96 else _flip_relation(
                rng, truth, relation_vocab
            )
            tags = ("salient",)
        else:
            context = rng.choice(distractor_contexts)
            observed = rng.choice(relation_vocab)
            tags = ("distractor",)
        observations.append(
            Observation(
                event_id=f"D{t:06d}",
                event_time=t,
                arrival_time=t,
                source_id=source,
                context=context,
                observed_relation=observed,
                tags=tags,
            )
        )

    queries = []
    for i, context in enumerate(salient_contexts):
        expected = stable_truth[context]
        queries.append(
            Query(
                query_id=f"DQ{i:03d}",
                query_time=length - 1,
                context=context,
                expected_relation=expected,
                kind="long-horizon-retention",
                tags=("early-structure",),
            )
        )
    # Explicit recurrence query for the temporarily changed salient context.
    queries.append(
        Query(
            query_id="DQREC",
            query_time=length - 1,
            context=salient_contexts[0],
            expected_relation=stable_truth[salient_contexts[0]],
            kind="regime-reactivation",
            tags=("recurrence",),
        )
    )
    oracle = {
        "stream_length": length,
        "salient_probability": 0.035,
        "salient_contexts": [list(c) for c in salient_contexts],
        "distractor_context_count": len(distractor_contexts),
        "stable_truth": {"|".join(k): v for k, v in stable_truth.items()},
        "temporary_regime": {
            "context": list(salient_contexts[0]),
            "start": length // 2,
            "end": (3 * length) // 4,
            "relation": relation_vocab[4],
        },
    }
    return HiddenWorld(PROTOCOL, "D", seed, tuple(observations), tuple(queries), oracle)


def _stage_e(seed: int, *, scale: str) -> HiddenWorld:
    """Composite hidden-physics world with opaque cyclic composition laws."""
    rng = Random(seed)
    order = 7
    relation_tokens = list(_opaque_tokens(rng, "R", order))
    rng.shuffle(relation_tokens)
    # token_for_latent is the hidden name map. Surface tokens carry no meaning.
    token_for_latent = {i: relation_tokens[i] for i in range(order)}
    latent_for_token = {value: key for key, value in token_for_latent.items()}
    sources = _opaque_tokens(rng, "S", 4)
    base_reliability = {
        sources[0]: 0.94,
        sources[1]: 0.84,
        sources[2]: 0.61,
        sources[3]: 0.58,
    }
    copy_parent = {sources[3]: sources[2]}
    offsets = (0, 2, 4, 0)
    phase_len = 90 if scale == "development" else 220
    total_time = phase_len * 4
    regime_schedule = []
    for phase, offset in enumerate(offsets):
        start = phase * phase_len
        end = (phase + 1) * phase_len - 1
        regime_schedule.append((start, end, offset))

    all_contexts = tuple(
        (token_for_latent[a], token_for_latent[b])
        for a in range(order)
        for b in range(order)
    )
    # Hold out contexts from direct observation; they remain inferable from the
    # cyclic law if the representation is discovered.
    heldout = set(rng.sample(list(all_contexts), 8))
    observations: list[Observation] = []
    event_counter = 0

    def truth_for(context: tuple[str, str], offset: int) -> str:
        a = latent_for_token[context[0]]
        b = latent_for_token[context[1]]
        return token_for_latent[(a + b + offset) % order]

    for t in range(total_time):
        phase = min(3, t // phase_len)
        offset = offsets[phase]
        candidates = [c for c in all_contexts if c not in heldout]
        context = rng.choice(candidates)
        truth = truth_for(context, offset)
        source = rng.choice(sources)
        reliability = base_reliability[source]
        if source == sources[3] and rng.random() < 0.80:
            # Correlated source follows S2's reliability/error tendency.
            reliability = base_reliability[sources[2]]
        if phase == 1 and source == sources[0]:
            reliability = 0.45  # trusted-source degradation
        if rng.random() < 0.11:
            observed = None
        else:
            observed = _report(rng, truth, tuple(relation_tokens), reliability)
        delay = 0 if rng.random() < 0.72 else rng.randint(1, 7)
        observations.append(
            Observation(
                event_id=f"E{event_counter:06d}",
                event_time=t,
                arrival_time=t + delay,
                source_id=source,
                context=context,
                observed_relation=observed,
                tags=(
                    "hidden-physics",
                    f"phase-{phase}",
                    "source-drift" if phase == 1 else "normal-source",
                ),
            )
        )
        event_counter += 1

    observations.sort(key=lambda row: (row.arrival_time, row.event_time, row.event_id))
    queries: list[Query] = []
    heldout_list = sorted(heldout)
    for phase, (_start, end, offset) in enumerate(regime_schedule):
        for i, context in enumerate(heldout_list[:4]):
            queries.append(
                Query(
                    query_id=f"EQ{phase}{i:02d}",
                    query_time=end,
                    context=context,
                    expected_relation=truth_for(context, offset),
                    kind="unseen-composition-under-open-world-dynamics",
                    tags=(
                        "heldout-context",
                        "regime-recurrence" if phase == 3 else "novel-regime",
                    ),
                )
            )
    # Ambiguous query deliberately has no resolvable answer at the boundary.
    queries.append(
        Query(
            query_id="EQAMB",
            query_time=regime_schedule[1][1],
            context=heldout_list[5],
            expected_relation=None,
            kind="epistemic-ambiguity",
            answerable=False,
            tags=("correct:unresolved",),
        )
    )
    oracle = {
        "order": order,
        "token_for_latent": token_for_latent,
        "composition": "latent(a+b+regime_offset) mod 7",
        "regime_schedule": regime_schedule,
        "source_reliability": base_reliability,
        "copy_parent": copy_parent,
        "heldout_contexts": [list(c) for c in heldout_list],
        "source_drift": {
            "source": sources[0],
            "phase": 1,
            "reliability": 0.45,
        },
        "contains": (
            "source-conflict",
            "source-drift",
            "missingness",
            "delays",
            "novel-regimes",
            "regime-recurrence",
            "heldout-composition",
            "epistemic-ambiguity",
        ),
    }
    return HiddenWorld(PROTOCOL, "E", seed, tuple(observations), tuple(queries), oracle)


def build_world(stage: str, seed: int, *, scale: str = "development") -> HiddenWorld:
    if stage not in STAGES:
        raise ValueError(f"unknown stage: {stage}")
    if scale not in {"development", "official"}:
        raise ValueError("scale must be development or official")
    if stage == "A":
        return _stage_a(seed)
    if stage == "B":
        return _stage_b(seed)
    if stage == "C":
        return _stage_c(seed)
    if stage == "D":
        return _stage_d(seed, scale=scale)
    return _stage_e(seed, scale=scale)


def build_suite(
    *,
    material: str,
    count: int = 3,
    stages: Iterable[str] = STAGES,
    scale: str = "development",
) -> tuple[HiddenWorld, ...]:
    if count < 1:
        raise ValueError("count must be positive")
    if scale == "official":
        material = validate_official_material(material)
    selected = tuple(stages)
    worlds = []
    for stage in selected:
        if stage not in STAGES:
            raise ValueError(f"unknown stage: {stage}")
        for index in range(count):
            worlds.append(
                build_world(
                    stage,
                    derive_seed(material, stage, index),
                    scale=scale,
                )
            )
    return tuple(worlds)


def suite_integrity(worlds: Iterable[HiddenWorld]) -> dict[str, Any]:
    rows = tuple(worlds)
    if not rows:
        raise ValueError("suite must not be empty")
    fingerprints = [world.fingerprint() for world in rows]
    stage_counts = {stage: 0 for stage in STAGES}
    for world in rows:
        stage_counts[world.stage] += 1

    checks = {
        "protocol_exact": all(world.protocol == PROTOCOL for world in rows),
        "unique_world_fingerprints": len(fingerprints) == len(set(fingerprints)),
        "nonempty_observations": all(world.observations for world in rows),
        "nonempty_queries": all(world.queries for world in rows),
        "arrival_not_before_event": all(
            obs.arrival_time >= obs.event_time
            for world in rows
            for obs in world.observations
        ),
        "oracle_not_in_public_payload": all(
            "oracle" not in world.public_payload() for world in rows
        ),
        "all_contexts_opaque": all(
            all(token and not token.isspace() for token in obs.context)
            for world in rows
            for obs in world.observations
        ),
    }
    return {
        "protocol": PROTOCOL,
        "worlds": len(rows),
        "stage_counts": stage_counts,
        "checks": checks,
        "passed": all(checks.values()),
    }
