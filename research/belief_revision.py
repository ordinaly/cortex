"""Belief-revision helpers for Cortex Reasoning Map v1."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from json import dumps
from random import Random

from counterfactual_reasoning import (
    CausalCell,
    CortexStructuralCounterfactual,
    Edge,
    Node,
)


CAUSAL_BATCH = {
    "do_positive": 18,
    "do_negative": 2,
    "ctrl_positive": 2,
    "ctrl_negative": 18,
}

STRONG_CONTRADICTION_BATCH = {
    "do_positive": 2,
    "do_negative": 18,
    "ctrl_positive": 2,
    "ctrl_negative": 18,
}

AMBIGUOUS_CONTRADICTION_BATCH = {
    "do_positive": 4,
    "do_negative": 16,
    "ctrl_positive": 2,
    "ctrl_negative": 18,
}


@dataclass(frozen=True)
class RevisionWorld:
    seed: int
    nodes: tuple[Node, ...]
    target_edge: Edge
    downstream_edge: Edge
    downstream_query: Edge
    unrelated_causal_edges: tuple[Edge, ...]


@dataclass(frozen=True)
class RevisionCheckpoint:
    batch: int
    direct_state: int
    effect: float
    downstream_state: int


def build_world(seed: int) -> RevisionWorld:
    rng = Random(seed * 1009 + 211)
    labels = [f"N{index}" for index in range(8)]
    rng.shuffle(labels)
    mapping = {
        index: labels[index]
        for index in range(8)
    }

    return RevisionWorld(
        seed=seed,
        nodes=tuple(sorted(mapping.values())),
        target_edge=(mapping[0], mapping[1]),
        downstream_edge=(mapping[1], mapping[2]),
        downstream_query=(mapping[0], mapping[2]),
        unrelated_causal_edges=(
            (mapping[3], mapping[4]),
            (mapping[5], mapping[6]),
        ),
    )


def _observe_batch(
    model: CortexStructuralCounterfactual,
    edge: Edge,
    batch: dict[str, int],
) -> None:
    model.observe_counts(
        edge[0],
        edge[1],
        **batch,
    )


def build_model(
    world: RevisionWorld,
    *,
    prior_batches: int,
) -> CortexStructuralCounterfactual:
    if prior_batches < 1:
        raise ValueError("prior_batches must be positive")

    model = CortexStructuralCounterfactual(world.nodes)
    causal_edges = {
        world.target_edge,
        world.downstream_edge,
        *world.unrelated_causal_edges,
    }

    for source in world.nodes:
        for target in world.nodes:
            if source == target:
                continue
            edge = (source, target)
            if edge == world.target_edge:
                for _ in range(prior_batches):
                    _observe_batch(model, edge, CAUSAL_BATCH)
            elif edge in causal_edges:
                _observe_batch(model, edge, CAUSAL_BATCH)
            else:
                _observe_batch(
                    model,
                    edge,
                    STRONG_CONTRADICTION_BATCH,
                )

    return model


def target_cell(
    model: CortexStructuralCounterfactual,
    world: RevisionWorld,
) -> CausalCell:
    cell = model.causal(*world.target_edge)
    if cell is None:
        raise RuntimeError("target belief cell is missing")
    return cell


def checkpoint(
    model: CortexStructuralCounterfactual,
    world: RevisionWorld,
    batch: int,
) -> RevisionCheckpoint:
    cell = target_cell(model, world)
    query = model.query(*world.downstream_query)
    return RevisionCheckpoint(
        batch=batch,
        direct_state=cell.state,
        effect=cell.effect,
        downstream_state=query.state,
    )


def apply_causal_batch(
    model: CortexStructuralCounterfactual,
    world: RevisionWorld,
) -> RevisionCheckpoint:
    _observe_batch(model, world.target_edge, CAUSAL_BATCH)
    return checkpoint(model, world, 0)


def apply_strong_contradiction(
    model: CortexStructuralCounterfactual,
    world: RevisionWorld,
) -> RevisionCheckpoint:
    _observe_batch(
        model,
        world.target_edge,
        STRONG_CONTRADICTION_BATCH,
    )
    return checkpoint(model, world, 0)


def apply_ambiguous_contradiction(
    model: CortexStructuralCounterfactual,
    world: RevisionWorld,
) -> RevisionCheckpoint:
    _observe_batch(
        model,
        world.target_edge,
        AMBIGUOUS_CONTRADICTION_BATCH,
    )
    return checkpoint(model, world, 0)


def unrelated_fingerprint(
    model: CortexStructuralCounterfactual,
    world: RevisionWorld,
) -> str:
    payload = []
    for (source, target), cell in sorted(model.cells.items()):
        if (source, target) == world.target_edge:
            continue
        payload.append(
            {
                "source": source,
                "target": target,
                "do_a": cell.do_a,
                "do_b": cell.do_b,
                "ctrl_a": cell.ctrl_a,
                "ctrl_b": cell.ctrl_b,
                "do_n": cell.do_n,
                "ctrl_n": cell.ctrl_n,
                "state": cell.state,
            }
        )

    return sha256(
        dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def monotone_nonincreasing(values: list[float]) -> bool:
    return all(
        right <= left + 1e-15
        for left, right in zip(values, values[1:])
    )


def monotone_nondecreasing(values: list[float]) -> bool:
    return all(
        right + 1e-15 >= left
        for left, right in zip(values, values[1:])
    )


def latest_batch_state(batch: dict[str, int]) -> int:
    """Memoryless baseline using only the latest evidence batch."""
    model = CortexStructuralCounterfactual(("S", "T"))
    model.observe_counts("S", "T", **batch)
    return model.direct_state("S", "T")
