from research.five_operator_bench.higher_order_irreducibility import (
    build_case,
    run_transition,
)


def test_complete_pairwise_graph_does_not_force_filled_triangle():
    case = build_case(3, synergistic=False)
    assert len(case["active"]) == 3
    assert len(case["graph"].edges) == 3
    assert case["triangles"] == ()
    assert case["score"] < 0.25


def test_pairwise_invisible_parity_residual_fills_triangle():
    case = build_case(3, synergistic=True)
    assert len(case["active"]) == 3
    assert len(case["graph"].edges) == 3
    assert len(case["triangles"]) == 1
    assert case["score"] >= 0.50
    assert case["joint_45"] > case["boundary_45"]


def test_face_loss_immediately_removes_coface_and_reads_are_pure():
    case = build_case(7, synergistic=True)
    complex_ = case["complex"]
    graph = case["graph"]
    assert len(complex_.triangles) == 1

    before_steps = complex_.steps
    weighted = complex_.weighted_triangles
    assert len(weighted) == 1
    assert complex_.steps == before_steps

    graph.edges.remove(next(iter(graph.edges)))
    assert complex_.step() == ()
    assert complex_.snapshot["retirements"] == 1


def test_same_k3_boundary_can_gain_and_lose_only_its_two_simplex():
    result = run_transition(91)
    assert result["passed"]
    assert result["initial_edges"] == result["reduced_edges"] == result["restored_edges"] == 3
    assert result["initial_triangles"] == 1
    assert result["reduced_triangles"] == 0
    assert result["restored_triangles"] == 1
