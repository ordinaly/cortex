from cortex.distinction_connectivity import (
    DistinctionConnectivity,
    DistinctionConnectivityPolicy,
    HystereticDistinctionGraph,
)
from cortex.learned_distinguishability import (
    EmpiricalTraceDistinguishability,
    LearnedDistinctionPolicy,
)
from cortex.probe_economy import HystereticProbePopulation, ProbeEconomy


CONTEXTS = ("a", "b", "c")


def _learner():
    return EmpiricalTraceDistinguishability(
        actions=(0, 1),
        symbols=(0, 1),
        horizon=1,
        max_contexts=4,
        window=8,
        policy=LearnedDistinctionPolicy(
            epsilon=0.25,
            min_samples=4,
            confidence=False,
        ),
    )


def _fill(learner, probe, outcomes):
    for context, outcome in zip(CONTEXTS, outcomes):
        for _ in range(8):
            learner.observe(context, probe, (outcome,))


def _complementary_fixture():
    learner = _learner()
    # q0 partitions a | bc => vetoes ab, ac.
    _fill(learner, (0,), (0, 1, 1))
    # q1 partitions ab | c => vetoes ac, bc.
    _fill(learner, (1,), (0, 0, 1))
    economy = ProbeEconomy(learner, CONTEXTS)
    return learner, economy


def test_connectivity_requires_mutual_nonredundancy():
    learner, economy = _complementary_fixture()
    connectivity = DistinctionConnectivity(economy)

    assert economy.veto_edges((0,)) == frozenset({(0, 1), (0, 2)})
    assert economy.veto_edges((1,)) == frozenset({(0, 2), (1, 2)})
    assert connectivity.unique_edges((0,), (1,)) == frozenset({(0, 1)})
    assert connectivity.unique_edges((1,), (0,)) == frozenset({(1, 2)})
    assert connectivity.complementarity((0,), (1,)) == 1 / 3
    assert connectivity.joint_coverage_gain((0,), (1,)) == 1 / 3

    # Flush q1 until it becomes identical to q0. Correlated/redundant probes do
    # not earn an edge merely because their responses agree strongly.
    _fill(learner, (1,), (0, 1, 1))
    assert connectivity.complementarity((0,), (1,)) == 0.0


def test_same_vertices_can_induce_different_topology_after_regime_change():
    learner, economy = _complementary_fixture()
    population = HystereticProbePopulation(economy, active=((0,), (1,)))
    graph = HystereticDistinctionGraph(
        DistinctionConnectivity(economy),
        population,
        policy=DistinctionConnectivityPolicy(edge_on=0.25, edge_off=0.10),
    )

    assert graph.vertices == ((0,), (1,))
    assert graph.step() == (((0,), (1,)),)

    # Keep the same two vertices alive while the phenomenon changes. q1 now
    # duplicates q0, so the learned topology loses the edge.
    _fill(learner, (1,), (0, 1, 1))
    assert graph.vertices == ((0,), (1,))
    assert graph.step() == ()
    assert graph.snapshot["retirements"] == 1


def test_graph_removes_edges_incident_to_a_retired_probe():
    _, economy = _complementary_fixture()
    population = HystereticProbePopulation(economy, active=((0,), (1,)))
    graph = HystereticDistinctionGraph(
        DistinctionConnectivity(economy),
        population,
        policy=DistinctionConnectivityPolicy(edge_on=0.25, edge_off=0.10),
    )
    assert graph.step()

    population.active.remove((1,))
    assert graph.step() == ()
    assert graph.vertices == ((0,),)
