"""Teacher-free autonomy evaluation helpers for Cortex research.

This module contains no neural-model dependency. It provides a chronological
knowledge timeline plus controller/LPU ablations used by Autonomy Campaign v1.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable

from .articulation_controller import (
    ArticulationController,
    ArticulationState,
    ExecutionBackend,
)
from .controller_calibration import (
    CalibrationBundle,
    CalibratedArticulationController,
)
from .hypergraph_geometry import HypergraphGeometry
from .hypergraph_inference import CompositionRule
from .hypergraph_lpu import (
    EnumeratingRuleExecutor,
    LPURuleExecutor,
    RuleExecutionRead,
)
from .knowledge import (
    KnowledgeGraphView,
    KnowledgeStatement,
    KnowledgeStatementIncidence,
    KnowledgeSymbolEncoder,
    feed_knowledge,
)


class PolicyMode(str, Enum):
    NONE = "none"
    FIXED_V0_1 = "fixed-v0.1"
    CALIBRATED_V0_2 = "calibrated-v0.2"


@dataclass(frozen=True, slots=True)
class TimelineReport:
    phase: str
    statements: int
    cortex_entities: int
    phase_binding_conflicts: int
    cumulative_binding_conflicts: int


@dataclass(frozen=True, slots=True)
class PolicyExecution:
    mode: PolicyMode
    lpu_enabled: bool
    state: ArticulationState
    backend: ExecutionBackend | None
    source: str
    relation: str
    candidates: tuple[str, ...]
    witness_count: int
    resolution: float | None
    reasons: tuple[str, ...]

    @property
    def resolved(self) -> bool:
        return bool(self.candidates)


class KnowledgeTimeline:
    """Accumulate chronological native Cortex knowledge across phases.

    The knowledge feed reports bindings for one call. Autonomy experiments
    need a persistent experiment-side index so observations arriving after
    teacher detachment can be read from the same native Cortex memory without
    replaying the childhood stream.
    """

    def __init__(
        self,
        runtime,
        *,
        encoder: KnowledgeSymbolEncoder | None = None,
    ) -> None:
        self.runtime = runtime
        self.encoder = encoder or KnowledgeSymbolEncoder()
        self.bindings: dict[str, int] = {}
        self.incidence: list[KnowledgeStatementIncidence] = []
        self.binding_conflicts = 0
        self.statement_ids: set[str] = set()
        self.phase_reports: list[TimelineReport] = []

    def feed(
        self,
        phase: str,
        statements: Iterable[KnowledgeStatement],
    ) -> TimelineReport:
        rows = tuple(statements)
        duplicate_ids = [
            row.statement_id
            for row in rows
            if row.statement_id in self.statement_ids
        ]
        if duplicate_ids:
            raise ValueError(
                "timeline statement ids must be unique: "
                + ", ".join(sorted(set(duplicate_ids)))
            )

        report = feed_knowledge(
            self.runtime,
            rows,
            encoder=self.encoder,
        )
        phase_conflicts = int(report.binding_conflicts)
        for symbol, binding in report.symbol_bindings.items():
            previous = self.bindings.get(symbol)
            if previous is not None and previous != binding:
                phase_conflicts += 1
            self.bindings[symbol] = int(binding)

        self.incidence.extend(report.statement_incidence)
        self.statement_ids.update(row.statement_id for row in rows)
        self.binding_conflicts += phase_conflicts

        out = TimelineReport(
            phase=phase,
            statements=len(rows),
            cortex_entities=report.cortex_entities,
            phase_binding_conflicts=phase_conflicts,
            cumulative_binding_conflicts=self.binding_conflicts,
        )
        self.phase_reports.append(out)
        return out

    def view(self) -> KnowledgeGraphView:
        return KnowledgeGraphView(
            self.runtime.snapshot_json(),
            self.bindings,
            self.runtime.config_json(),
            self.incidence,
        )

    def direct_objects(
        self,
        subject: str,
        relation: str,
    ) -> tuple[str, ...]:
        return self.view().objects(subject, relation)

    @property
    def cortex_entities(self) -> int:
        import json

        return int(
            json.loads(self.runtime.snapshot_json())["articulation"][
                "active_entities"
            ]
        )


def _force_executor(
    *,
    geometry: HypergraphGeometry,
    source: str,
    relation: str,
    rules: tuple[CompositionRule, ...],
    resolution: float,
    use_lpu: bool,
) -> RuleExecutionRead:
    executor = (
        LPURuleExecutor(geometry, resolution=resolution)
        if use_lpu
        else EnumeratingRuleExecutor(geometry, resolution=resolution)
    )
    return executor.infer(source, relation, rules)


def execute_policy(
    *,
    geometry: HypergraphGeometry,
    rules: Iterable[CompositionRule],
    source: str,
    relation: str,
    mode: PolicyMode,
    lpu_enabled: bool,
    calibration: CalibrationBundle | None = None,
    source_support: float = 1.0,
    source_entropy: float = 0.0,
) -> PolicyExecution:
    """Execute one autonomy query under a declared policy ablation."""

    rules = tuple(rules)

    if mode is PolicyMode.NONE:
        selected = tuple(
            rule for rule in rules if rule.target_relation == relation
        )
        if not selected:
            return PolicyExecution(
                mode=mode,
                lpu_enabled=lpu_enabled,
                state=ArticulationState.EXPLORE,
                backend=None,
                source=source,
                relation=relation,
                candidates=(),
                witness_count=0,
                resolution=None,
                reasons=("no-composition-evidence",),
            )
        read = _force_executor(
            geometry=geometry,
            source=source,
            relation=relation,
            rules=selected,
            resolution=0.5,
            use_lpu=lpu_enabled,
        )
        return PolicyExecution(
            mode=mode,
            lpu_enabled=lpu_enabled,
            state=ArticulationState.EXECUTE,
            backend=(
                ExecutionBackend.LPU
                if lpu_enabled
                else ExecutionBackend.CORTEX
            ),
            source=source,
            relation=relation,
            candidates=tuple(row.target for row in read.candidates),
            witness_count=len(read.witnesses),
            resolution=0.5,
            reasons=("controller-bypassed",),
        )

    if mode is PolicyMode.FIXED_V0_1:
        controller = ArticulationController(geometry)
        decision = controller.decide(
            source,
            relation,
            rules,
            source_support=source_support,
            source_entropy=source_entropy,
        )
        selected = decision.selected_rules
        resolution = decision.resolution
        state = decision.state
        reasons = decision.reasons
        preferred = decision.backend
    elif mode is PolicyMode.CALIBRATED_V0_2:
        if calibration is None:
            raise ValueError("calibrated policy requires a calibration bundle")
        controller = CalibratedArticulationController(
            geometry,
            calibration,
        )
        decision = controller.decide(
            source,
            relation,
            rules,
            source_support=source_support,
            source_entropy=source_entropy,
        )
        selected = decision.selected_rules
        resolution = decision.resolution
        state = decision.state
        reasons = decision.reasons
        preferred = decision.backend
    else:
        raise ValueError(f"unsupported policy mode: {mode}")

    if (
        state is not ArticulationState.EXECUTE
        or resolution is None
        or not selected
    ):
        return PolicyExecution(
            mode=mode,
            lpu_enabled=lpu_enabled,
            state=state,
            backend=None,
            source=source,
            relation=relation,
            candidates=(),
            witness_count=0,
            resolution=resolution,
            reasons=tuple(reasons),
        )

    use_lpu = lpu_enabled and preferred is ExecutionBackend.LPU
    read = _force_executor(
        geometry=geometry,
        source=source,
        relation=relation,
        rules=tuple(selected),
        resolution=resolution,
        use_lpu=use_lpu,
    )
    return PolicyExecution(
        mode=mode,
        lpu_enabled=lpu_enabled,
        state=state,
        backend=(
            ExecutionBackend.LPU
            if use_lpu
            else ExecutionBackend.CORTEX
        ),
        source=source,
        relation=relation,
        candidates=tuple(row.target for row in read.candidates),
        witness_count=len(read.witnesses),
        resolution=resolution,
        reasons=tuple(reasons),
    )
