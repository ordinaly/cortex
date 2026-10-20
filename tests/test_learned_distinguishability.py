from cortex.learned_distinguishability import (
    EmpiricalTraceDistinguishability,
    LearnedDistinctionPolicy,
)
from research.five_operator_bench.learn_delta import evaluate_horizon, run_learning_experiment


def test_unsupported_contexts_are_not_folded():
    learner = EmpiricalTraceDistinguishability(
        horizon=1,
        policy=LearnedDistinctionPolicy(min_samples=4, confidence=False),
    )
    assert learner.probe_distance("a", "b", (0,)) == 1.0
    assert not learner.compatible("a", "b")


def test_learned_bank_exposes_supremum_predictive_distinguishability():
    result = evaluate_horizon(2)
    assert result["passed"], result
    assert result["bank_zero_pair_mean"] == 0.0
    assert result["population_witness_distance"] > 0.9


def test_horizon_one_preserves_the_known_alias():
    result = evaluate_horizon(1)
    assert result["passed"], result
    assert result["proposed"] == ((0, 1, 2), (3,))


def test_horizon_two_learns_the_delayed_action_distinction():
    result = evaluate_horizon(2)
    assert result["passed"], result
    assert result["proposed"] == ((0, 2), (1,), (3,))
    assert result["oracle_audit"]["minimum_feasible_cells"] == 3


def test_learning_experiment_refines_only_when_new_probe_information_arrives():
    report = run_learning_experiment()
    assert report["passed"], report
    assert report["refinement_observed"]
