"""Oracle-assisted benchmark for the five-operator Cortex research loop.

This benchmark deliberately tests the architecture before learning. Evaluator-owned
world structure may define distinction neurons and expected answers; Cortex receives
only the resulting public observations/operators. Later phases can replace these
oracle-supplied pieces with learned ones without changing the acceptance metrics.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from fractions import Fraction as F

from cortex.five_operator_loop import (
    ArticulationInput,
    CausalModel,
    CausalObservation,
    DistinctionBank,
    DistinctionNeuron,
    DynamicComplexPolicy,
    DynamicDistinguishabilityComplex,
    FiveOperatorArticulationLoop,
    Integrator,
    LocalView,
    OrientationField,
    TemporalTransport,
)
from research.sl_bench.oracle import exact_quotient, tv
from research.sl_bench.worlds import exact_worlds


EPS = 1.0e-12


@dataclass(frozen=True)
class CaseResult:
    name: str
    passed: bool
    details: dict


def _canonical(blocks):
    return tuple(sorted(tuple(sorted(block)) for block in blocks))


def _zero_distance_partition(bank: DistinctionBank, states: tuple[int, ...]):
    unseen = set(states)
    blocks = []
    while unseen:
        seed = min(unseen)
        stack = [seed]
        component = set()
        while stack:
            current = stack.pop()
            if current in component:
                continue
            component.add(current)
            for other in states:
                if other in component:
                    continue
                if bank.compare(current, other).aggregate <= EPS:
                    stack.append(other)
        unseen -= component
        blocks.append(tuple(sorted(component)))
    return _canonical(blocks)


def delta_oracle_case() -> CaseResult:
    """Delta must recover the exact finite predictive quotient when given exact probes."""
    world = exact_worlds()["action_blindness"]
    profiles = world.profiles(2)
    probes = tuple(sorted(profiles[0]))
    neurons = tuple(
        DistinctionNeuron(
            neuron_id=f"probe-{index}",
            channel="predictive",
            operator=lambda left, right, probe=probe: float(
                tv(profiles[left][probe], profiles[right][probe])
            ),
        )
        for index, probe in enumerate(probes)
    )
    bank = DistinctionBank(neurons)
    recovered = _zero_distance_partition(bank, tuple(range(world.n)))
    expected = exact_quotient(profiles)
    return CaseResult(
        "delta_exact_quotient",
        recovered == expected,
        {"recovered": recovered, "expected": expected, "neurons": len(neurons)},
    )


def temporal_identity_case() -> CaseResult:
    """K must preserve continuity independently of static similarity."""
    transport = TemporalTransport()
    for _ in range(6):
        transport.observe("kitten", "young-cat")
    for _ in range(2):
        transport.observe("kitten", "other")
    successor = transport.best_successor("kitten")
    passed = successor is not None and successor[0] == "young-cat" and abs(successor[1] - 0.75) <= EPS
    return CaseResult(
        "temporal_identity",
        passed,
        {"best_successor": successor, "surprise": transport.surprise("kitten", "young-cat")},
    )


def causal_case() -> CaseResult:
    """C must remain zero without a control and become directional after both arms."""
    model = CausalModel()
    model.observe(CausalObservation("push", "move", 1.0, True))
    before = model.influence("push", "move")
    model.observe(CausalObservation("push", "move", 0.0, False))
    after = model.signed_effect("push", "move")
    return CaseResult(
        "intervention_control",
        before == 0.0 and after == 1.0,
        {"before_control": before, "signed_effect": after},
    )


def _make_context_loop():
    neurons = (
        DistinctionNeuron(
            "shape",
            "structural",
            lambda a, b: abs(a["shape"] - b["shape"]),
            value_axes=("goal",),
        ),
        DistinctionNeuron(
            "motion",
            "predictive",
            lambda a, b: abs(a["motion"] - b["motion"]),
            value_axes=("safety",),
        ),
        DistinctionNeuron(
            "cause",
            "causal",
            lambda a, b: abs(a["impact"] - b["impact"]),
            value_axes=("safety",),
        ),
    )
    bank = DistinctionBank(neurons)
    policy = DynamicComplexPolicy(
        activation_threshold=0.40,
        deactivation_threshold=0.25,
        novelty_weight=0.40,
        temporal_weight=0.10,
        causal_weight=0.20,
        value_weight=0.20,
        base_weight=0.10,
        synergy_gain=0.40,
    )
    complex_ = DynamicDistinguishabilityComplex(
        bank.neuron_ids,
        potential_simplices=(("shape", "motion"), ("motion", "cause")),
        policy=policy,
    )
    return FiveOperatorArticulationLoop(
        bank,
        TemporalTransport(),
        CausalModel(),
        OrientationField({"goal": 1.0, "safety": 1.0}),
        Integrator(conflict_tolerance=0.25),
        complex_,
    )


def dynamic_topology_case() -> CaseResult:
    """The same substrate must select different active topology for different phenomena."""
    loop = _make_context_loop()
    loop.step(
        ArticulationInput(
            "rest",
            {"shape": 0.0, "motion": 0.0, "impact": 0.0},
            orientation={"goal": 0.0, "safety": 0.0},
        )
    )
    shape_motion = frozenset(("shape", "motion"))
    motion_cause = frozenset(("motion", "cause"))
    motion = loop.step(
        ArticulationInput(
            "moving",
            {"shape": 1.0, "motion": 1.0, "impact": 0.0},
            orientation={"goal": 1.0, "safety": 0.0},
            simplex_synergy={shape_motion: 1.0},
            local_views=(
                LocalView(frozenset(("shape",)), {"entity": 0.90}),
                LocalView(frozenset(("motion",)), {"entity": 0.80}),
                LocalView(shape_motion, {"entity": 0.85}),
            ),
        )
    )
    causal = loop.step(
        ArticulationInput(
            "collision",
            {"shape": 1.0, "motion": 1.0, "impact": 1.0},
            orientation={"goal": 0.0, "safety": 1.0},
            causal_updates=(
                CausalObservation("push", "move", 1.0, True, neuron_ids=("cause",)),
                CausalObservation("push", "move", 0.0, False, neuron_ids=("cause",)),
            ),
            simplex_synergy={motion_cause: 1.0},
            local_views=(
                LocalView(frozenset(("motion",)), {"event": 0.80}),
                LocalView(frozenset(("cause",)), {"event": 0.90}),
                LocalView(motion_cause, {"event": 0.85}),
            ),
        )
    )
    motion_active = set(motion.complex.active)
    causal_active = set(causal.complex.active)
    passed = (
        shape_motion in motion_active
        and motion_cause not in motion_active
        and motion_cause in causal_active
        and shape_motion not in causal_active
        and motion.integration.used_views == 3
        and causal.integration.used_views == 3
        and not motion.integration.conflicts
        and not causal.integration.conflicts
    )
    return CaseResult(
        "dynamic_context_topology",
        passed,
        {
            "motion_active": [tuple(sorted(x)) for x in motion.complex.active],
            "causal_active": [tuple(sorted(x)) for x in causal.complex.active],
            "causal_relevance": dict(causal.causal_relevance),
            "motion_orientation_utility": motion.orientation_utility,
            "causal_orientation_utility": causal.orientation_utility,
        },
    )


def integration_conflict_case() -> CaseResult:
    """I must expose an incompatible overlap instead of hiding it in the mean."""
    simplex = frozenset(("shape",))
    read = Integrator(conflict_tolerance=0.20).integrate(
        (
            LocalView(simplex, {"position": 0.0}),
            LocalView(simplex, {"position": 1.0}),
        ),
        active_complex=(simplex,),
    )
    return CaseResult(
        "integration_conflict",
        read.conflicts == ("position",) and read.residual > 0.0,
        {"articulation": read.articulation, "residual": read.residual, "conflicts": read.conflicts},
    )


def run_benchmark():
    cases = (
        delta_oracle_case(),
        temporal_identity_case(),
        causal_case(),
        dynamic_topology_case(),
        integration_conflict_case(),
    )
    return {
        "version": "five-operator-bench-v0.1",
        "passed": all(case.passed for case in cases),
        "passed_cases": sum(case.passed for case in cases),
        "total_cases": len(cases),
        "cases": [asdict(case) for case in cases],
    }


if __name__ == "__main__":
    print(json.dumps(run_benchmark(), indent=2, sort_keys=True, default=str))
