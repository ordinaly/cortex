"""Theorem-inspired routing probes for layered Cortex execution graphs.

This module is research-only. It does not change semantic confidence,
resolution, memory, or execution results.

The Kim--Vu sandwich theorem concerns random regular graph distributions.
Cortex layered graphs are arbitrary learned structures, so the code below uses
only a deliberately weaker engineering idea: near-regular local degree samples
can sometimes bound branching well enough to choose an execution backend.
When the sampled graph is irregular or the bounds overlap the routing boundary,
the probe returns FALLBACK and the existing exact profiler remains authoritative.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import ceil, isfinite, prod, sqrt
from statistics import fmean
from typing import Iterable

from .hypergraph_geometry import HypergraphGeometry
from .hypergraph_inference import CompositionRule, RelationStep


EPS = 1.0e-12


class ProbeRoute(str, Enum):
    CORTEX = "cortex"
    LPU = "lpu"
    FALLBACK = "fallback"


@dataclass(frozen=True, slots=True)
class SandwichProbePolicy:
    max_states_per_layer: int = 8
    minimum_regular_samples: int = 4
    maximum_degree_cv: float = 0.15
    base_relative_slack: float = 0.15
    lpu_min_paths: int = 32
    lpu_min_compression_ratio: float = 2.0

    def __post_init__(self) -> None:
        if self.max_states_per_layer < 1:
            raise ValueError("max_states_per_layer must be positive")
        if self.minimum_regular_samples < 1:
            raise ValueError("minimum_regular_samples must be positive")
        for name, value in (
            ("maximum_degree_cv", self.maximum_degree_cv),
            ("base_relative_slack", self.base_relative_slack),
            ("lpu_min_compression_ratio", self.lpu_min_compression_ratio),
        ):
            if not isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.lpu_min_paths < 1:
            raise ValueError("lpu_min_paths must be positive")


@dataclass(frozen=True, slots=True)
class LayerProbe:
    rule_index: int
    depth: int
    frontier_seen: int
    sampled_states: int
    matching_queries: int
    mean_degree: float
    degree_cv: float
    minimum_degree: int
    maximum_degree: int
    next_states_seen: int
    truncated: bool


@dataclass(frozen=True, slots=True)
class SandwichRoutingEstimate:
    resolution: float
    route: ProbeRoute
    eligible: bool
    lower_paths: float
    upper_paths: float
    lower_edges: float
    upper_edges: float
    matching_queries: int
    layers: tuple[LayerProbe, ...]
    reasons: tuple[str, ...]

    @property
    def lower_compression(self) -> float:
        return self.lower_paths / max(1.0, self.upper_edges)

    @property
    def upper_compression(self) -> float:
        return self.upper_paths / max(1.0, self.lower_edges)


def _matching_targets(
    geometry: HypergraphGeometry,
    node: str,
    step: RelationStep,
    *,
    resolution: float,
) -> tuple[str, ...]:
    targets: list[str] = []
    for edge in geometry.incident(node, resolution=resolution):
        if edge.property != step.relation:
            continue
        if step.orientation == "forward" and edge.subject == node:
            targets.append(edge.object)
        elif step.orientation == "reverse" and edge.object == node:
            targets.append(edge.subject)
    return tuple(sorted(targets))


def _coefficient_of_variation(values: tuple[int, ...]) -> float:
    if not values:
        return float("inf")
    mean = fmean(values)
    if mean <= EPS:
        return 0.0 if all(value == 0 for value in values) else float("inf")
    variance = fmean((value - mean) ** 2 for value in values)
    return sqrt(variance) / mean


def probe_rules(
    geometry: HypergraphGeometry,
    source: str,
    rules: Iterable[CompositionRule],
    *,
    resolution: float,
    policy: SandwichProbePolicy | None = None,
) -> SandwichRoutingEstimate:
    """Estimate routing cost from bounded local degree samples.

    The returned interval is intentionally heuristic, not a theorem-certified
    confidence interval. Its only authority is to select a backend when both
    path and compression bounds lie wholly on one side of the current routing
    boundary. All ambiguous or visibly irregular cases return FALLBACK.
    """

    policy = policy or SandwichProbePolicy()
    if not isfinite(resolution) or not 0.0 <= resolution <= 1.0:
        raise ValueError("resolution must be finite and in [0,1]")

    rules = tuple(rules)
    layer_rows: list[LayerProbe] = []
    lower_paths_total = 0.0
    upper_paths_total = 0.0
    lower_edges_total = 0.0
    upper_edges_total = 0.0
    total_queries = 0
    eligible = True
    reasons: list[str] = []

    for rule_index, rule in enumerate(rules):
        current = (source,)
        rule_lower_paths = 1.0
        rule_upper_paths = 1.0
        frontier_low = 1.0
        frontier_high = 1.0

        for depth, step in enumerate(rule.signature):
            if not current:
                rule_lower_paths = 0.0
                rule_upper_paths = 0.0
                break

            ordered = tuple(sorted(set(current)))
            truncated = len(ordered) > policy.max_states_per_layer
            sample = ordered[: policy.max_states_per_layer]
            degree_targets = tuple(
                _matching_targets(
                    geometry,
                    node,
                    step,
                    resolution=resolution,
                )
                for node in sample
            )
            degrees = tuple(len(row) for row in degree_targets)
            total_queries += len(sample)
            next_seen = tuple(
                sorted({target for rows in degree_targets for target in rows})
            )

            mean_degree = fmean(degrees) if degrees else 0.0
            cv = _coefficient_of_variation(degrees)
            enough_samples = (
                len(sample) >= policy.minimum_regular_samples
                or not truncated
            )
            regular = cv <= policy.maximum_degree_cv + EPS
            positive = mean_degree > EPS

            if depth > 0 and (not enough_samples or not regular):
                eligible = False
                if not enough_samples:
                    reasons.append(
                        f"layer-{rule_index}-{depth}:insufficient-sample"
                    )
                if not regular:
                    reasons.append(
                        f"layer-{rule_index}-{depth}:degree-irregular"
                    )

            if not positive:
                rule_lower_paths = 0.0
                rule_upper_paths = 0.0
                layer_rows.append(
                    LayerProbe(
                        rule_index=rule_index,
                        depth=depth,
                        frontier_seen=len(ordered),
                        sampled_states=len(sample),
                        matching_queries=len(sample),
                        mean_degree=0.0,
                        degree_cv=cv,
                        minimum_degree=min(degrees, default=0),
                        maximum_degree=max(degrees, default=0),
                        next_states_seen=0,
                        truncated=truncated,
                    )
                )
                current = ()
                break

            relative_slack = max(
                policy.base_relative_slack,
                2.0 * cv,
                0.25 if truncated else 0.0,
            )
            degree_low = max(0.0, mean_degree * (1.0 - relative_slack))
            degree_high = mean_degree * (1.0 + relative_slack)

            rule_lower_paths *= degree_low
            rule_upper_paths *= degree_high
            lower_edges_total += frontier_low * degree_low
            upper_edges_total += frontier_high * degree_high

            if truncated:
                # The sampled next-state set is only a lower observation. Carry
                # a conservative multiplicative uncertainty into the next
                # frontier rather than pretending it is complete.
                frontier_low = max(1.0, float(len(next_seen)))
                frontier_high = max(
                    frontier_low,
                    float(len(next_seen))
                    * (len(ordered) / max(1, len(sample))),
                )
            else:
                frontier_low = float(len(next_seen))
                frontier_high = float(len(next_seen))

            layer_rows.append(
                LayerProbe(
                    rule_index=rule_index,
                    depth=depth,
                    frontier_seen=len(ordered),
                    sampled_states=len(sample),
                    matching_queries=len(sample),
                    mean_degree=mean_degree,
                    degree_cv=cv,
                    minimum_degree=min(degrees),
                    maximum_degree=max(degrees),
                    next_states_seen=len(next_seen),
                    truncated=truncated,
                )
            )
            current = next_seen

        lower_paths_total += rule_lower_paths
        upper_paths_total += rule_upper_paths

    if not rules:
        eligible = False
        reasons.append("no-rules")

    lower_compression = lower_paths_total / max(1.0, upper_edges_total)
    upper_compression = upper_paths_total / max(1.0, lower_edges_total)

    if not eligible:
        route = ProbeRoute.FALLBACK
    elif (
        lower_paths_total + EPS >= policy.lpu_min_paths
        and lower_compression + EPS >= policy.lpu_min_compression_ratio
    ):
        route = ProbeRoute.LPU
        reasons.append("lower-envelope-selects-lpu")
    elif (
        upper_paths_total + EPS < policy.lpu_min_paths
        or upper_compression + EPS < policy.lpu_min_compression_ratio
    ):
        route = ProbeRoute.CORTEX
        reasons.append("upper-envelope-selects-cortex")
    else:
        route = ProbeRoute.FALLBACK
        reasons.append("routing-boundary-overlap")

    return SandwichRoutingEstimate(
        resolution=float(resolution),
        route=route,
        eligible=eligible,
        lower_paths=lower_paths_total,
        upper_paths=upper_paths_total,
        lower_edges=lower_edges_total,
        upper_edges=upper_edges_total,
        matching_queries=total_queries,
        layers=tuple(layer_rows),
        reasons=tuple(dict.fromkeys(reasons)),
    )


@dataclass(frozen=True, slots=True)
class SaturatingProfile:
    """Deterministic control that caps path counts but still scans all states."""

    path_multiplicity_lower_bound: int
    materialized_edges: int
    layered_states: int
    matching_queries: int
    saturated: bool


def saturating_profile_rules(
    geometry: HypergraphGeometry,
    source: str,
    rules: Iterable[CompositionRule],
    *,
    resolution: float,
    path_cap: int = 1_000_000,
) -> SaturatingProfile:
    """Exact structural traversal with saturated integer path counts.

    This is a deterministic baseline for the experiment. It can reduce large
    integer arithmetic, but unlike the sandwich probe it cannot avoid scanning
    all reachable layered states.
    """

    if path_cap < 1:
        raise ValueError("path_cap must be positive")

    total_paths = 0
    matching_queries = 0
    saturated = False
    layered_edges: set[
        tuple[tuple[int, str], tuple[int, str], str]
    ] = set()
    layered_states: set[tuple[int, str]] = {(0, source)}

    for rule_index, rule in enumerate(rules):
        current: dict[str, int] = {source: 1}
        for depth, step in enumerate(rule.signature):
            nxt: dict[str, int] = {}
            for node, count in current.items():
                matching_queries += 1
                targets = _matching_targets(
                    geometry,
                    node,
                    step,
                    resolution=resolution,
                )
                for target in targets:
                    value = nxt.get(target, 0) + count
                    if value > path_cap:
                        value = path_cap
                        saturated = True
                    nxt[target] = value
                    layered_states.add((depth + 1, target))
                    layered_edges.add(
                        (
                            (depth, node),
                            (depth + 1, target),
                            f"{rule_index}:{node}:{target}",
                        )
                    )
            current = nxt
            if not current:
                break

        rule_paths = sum(current.values())
        if rule_paths > path_cap:
            rule_paths = path_cap
            saturated = True
        total_paths += rule_paths
        if total_paths > path_cap:
            total_paths = path_cap
            saturated = True

    return SaturatingProfile(
        path_multiplicity_lower_bound=total_paths,
        materialized_edges=len(layered_edges),
        layered_states=len(layered_states),
        matching_queries=matching_queries,
        saturated=saturated,
    )
