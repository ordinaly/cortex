from research.five_operator_bench.distinction_topology_robustness import (
    PAIR_INDICES,
    POST_PROBABILITIES,
    PRE_PROBABILITIES,
    STABLE_TICKS,
    SWITCH_TICKS,
    TARGET_POST,
    TARGET_PRE,
    _run_policy,
    build_score_trace,
    ideal_complementarity,
)


def test_latent_regime_rewires_the_high_complementarity_edge():
    pre = ideal_complementarity(PRE_PROBABILITIES)
    post = ideal_complementarity(POST_PROBABILITIES)

    assert pre[(0, 1)] == 1 / 6
    assert pre[(0, 2)] == 1 / 3
    assert pre[(1, 2)] == 1 / 6

    assert post[(0, 1)] == 1 / 6
    assert post[(0, 2)] == 1 / 6
    assert post[(1, 2)] == 1 / 3

    assert TARGET_PRE == frozenset({(0, 2)})
    assert TARGET_POST == frozenset({(1, 2)})


def test_noisy_score_trace_is_bounded_and_has_expected_length():
    trace = build_score_trace(0)
    assert len(trace) == STABLE_TICKS + SWITCH_TICKS
    assert all(len(scores) == len(PAIR_INDICES) for scores in trace)
    assert all(0.0 <= score <= 1.0 for scores in trace for score in scores)


def test_hysteresis_state_machine_retains_mid_band_evidence_but_releases_below_off():
    # Pair order is (0,1), (0,2), (1,2). Keep the desired pre edge above
    # edge_on, let it dip into the hysteresis band, then force it below edge_off
    # while the desired post edge becomes strong.
    trace = (
        ((0.0, 0.34, 0.0),) * STABLE_TICKS
        + ((0.0, 0.25, 0.34),) * 4
        + ((0.0, 0.10, 0.34),) * (SWITCH_TICKS - 4)
    )
    result = _run_policy(trace, edge_on=0.30, edge_off=0.20)

    assert result["stable_error_rate"] == 0.0
    assert result["rewire_delay"] == 5
    assert result["post_tail_error_rate"] == 0.0
