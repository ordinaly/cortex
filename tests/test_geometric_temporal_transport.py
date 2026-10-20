from __future__ import annotations

from itertools import combinations, product
from math import log

from cortex.geometric_temporal_transport import GeometricTemporalTransport, GeometricTransportPolicy
from cortex.simplicial_integration import SimplicialIntegrationRead


PROBES = ((0,), (1,), (2,), (3,))


def _read(kind: str, marker_p: float, *, filled: bool = True, drop_last: bool = False) -> SimplicialIntegrationRead:
    probes = PROBES[:-1] if drop_last else PROBES
    distribution = {}
    for marker in (0, 1):
        p_marker = marker_p if marker else 1.0 - marker_p
        for bits in product((0, 1), repeat=3):
            parity = bits[0] ^ bits[1] ^ bits[2]
            if kind == "even":
                p_bits = 0.25 if parity == 0 else 0.0
            elif kind == "odd":
                p_bits = 0.25 if parity == 1 else 0.0
            elif kind == "uniform":
                p_bits = 0.125
            else:
                raise ValueError(kind)
            state = (marker, *bits)
            if drop_last:
                state = state[:-1]
            distribution[state] = distribution.get(state, 0.0) + p_marker * p_bits

    constraints = [(i,) for i in range(len(probes))]
    constraints.extend(combinations(range(len(probes)), 2))
    if filled and len(probes) == 4:
        constraints.append((1, 2, 3))
    entropy = -sum(p * log(p) for p in distribution.values() if p > 0.0)
    return SimplicialIntegrationRead(
        probes=probes,
        distribution=tuple(sorted(distribution.items())),
        residual=0.0,
        entropy=entropy,
        used_constraints=tuple(tuple(row) for row in constraints),
        max_dimension=2 if filled and len(probes) == 4 else 1,
        iterations=0,
    )


def test_low_dimensional_transport_preserves_identity_when_full_joint_similarity_fails() -> None:
    transport = GeometricTemporalTransport()
    source_a = _read("even", 0.30, filled=True)
    source_b = _read("odd", 0.70, filled=True)
    target_a = _read("uniform", 0.30, filled=False)
    target_b = _read("even", 0.70, filled=True)

    aa = transport.compare(source_a, target_a)
    ab = transport.compare(source_a, target_b)
    bb = transport.compare(source_b, target_b)
    ba = transport.compare(source_b, target_a)

    assert aa.distance == 0.0
    assert bb.distance == 0.0
    assert ab.distance > 0.10
    assert ba.distance > 0.10

    # Deliberately adversarial: raw full-joint similarity chooses the wrong identity.
    assert ab.full_joint_distance < aa.full_joint_distance
    assert ba.full_joint_distance < bb.full_joint_distance

    assert transport.best_successor(source_a, (("A", target_a), ("B", target_b)))[0] == "A"
    assert transport.best_successor(source_b, (("A", target_a), ("B", target_b)))[0] == "B"


def test_transport_exposes_topology_change_without_forcing_identity_break() -> None:
    transport = GeometricTemporalTransport()
    source = _read("even", 0.30, filled=True)
    target = _read("uniform", 0.30, filled=False)
    comparison = transport.compare(source, target)

    assert comparison.distance == 0.0
    assert comparison.full_joint_distance == 0.5
    assert comparison.topology_change > 0.0
    assert comparison.by_dimension == ((0, 0.0), (1, 0.0))


def test_transport_survives_nonmarker_probe_loss() -> None:
    transport = GeometricTemporalTransport(
        policy=GeometricTransportPolicy(overlap_penalty=0.20, inverse_temperature=8.0)
    )
    source = _read("even", 0.30, filled=True)
    target_a = _read("uniform", 0.30, filled=False, drop_last=True)
    target_b = _read("even", 0.70, filled=False, drop_last=True)

    correct = transport.compare(source, target_a)
    cross = transport.compare(source, target_b)
    assert correct.overlap == 0.75
    assert correct.distance < cross.distance
    assert transport.best_successor(source, (("A", target_a), ("B", target_b)))[0] == "A"
