from cortex.learned_distinguishability import (
    EmpiricalTraceDistinguishability,
    LearnedDistinctionPolicy,
)


def test_support_tracks_retained_window_not_cumulative_history():
    learner = EmpiricalTraceDistinguishability(
        actions=(0, 1),
        symbols=(0, 1),
        horizon=1,
        window=8,
        policy=LearnedDistinctionPolicy(min_samples=2, confidence=True),
    )
    for index in range(40):
        learner.observe("ctx", (0,), (index % 2,))

    assert learner.snapshot["observations"] == 40
    assert learner.support("ctx", (0,)) == 8
    assert learner.snapshot["retained_support"] == 8
    assert learner._radius(learner.support("ctx", (0,))) > learner._radius(40)


def test_prefix_observations_count_when_they_drive_the_prefix_predictor():
    learner = EmpiricalTraceDistinguishability(
        actions=(0, 1),
        symbols=(0, 1),
        horizon=2,
        window=16,
        policy=LearnedDistinctionPolicy(min_samples=2, confidence=False),
    )
    # A length-2 observation updates both the (0,) prefix predictor and the
    # full (0,1) predictor inside TracePredictiveMemory.
    learner.observe("ctx", (0, 1), (1, 0))

    assert learner.support("ctx", (0,)) == 1
    assert learner.support("ctx", (0, 1)) == 1
    # The cumulative direct-observation ledger intentionally records only the
    # submitted full probe; it is no longer used as statistical support.
    assert learner.snapshot["observations"] == 1
    assert learner.snapshot["retained_support"] == 2
