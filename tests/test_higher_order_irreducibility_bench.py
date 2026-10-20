from research.five_operator_bench.higher_order_irreducibility import (
    run_higher_order_irreducibility_experiment,
)


def test_higher_order_irreducibility_benchmark_passes():
    result = run_higher_order_irreducibility_experiment()
    assert result["passed"]
