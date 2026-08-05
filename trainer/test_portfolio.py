import numpy as np

from trainer.portfolio import (
    combine,
    curve_stats,
    diversified_stats,
    inverse_vol_weights,
    step_returns,
)


def test_step_returns_of_a_known_curve():
    r = step_returns([100.0, 110.0, 99.0])
    assert np.allclose(r, [0.1, -0.1])


def test_step_returns_neutralises_non_finite():
    # a zero price would make the next ratio inf/nan -> neutralised to 0, never propagated
    assert np.all(np.isfinite(step_returns([100.0, 0.0, 50.0])))


def test_curve_stats_total_return_and_drawdown():
    s = curve_stats([100.0, 120.0, 90.0, 108.0])
    assert abs(s["total_return_pct"] - 8.0) < 1e-9  # 100 -> 108
    assert abs(s["max_drawdown_pct"] - (-25.0)) < 1e-9  # 120 -> 90 = -25%


def test_curve_stats_flat_curve_is_zero_not_nan():
    s = curve_stats([100.0, 100.0, 100.0])
    assert s["sharpe"] == 0.0 and s["total_return_pct"] == 0.0 and np.isfinite(s["calmar"])


def test_combine_equal_weight_is_the_mean_of_step_returns():
    a = [100.0, 110.0, 121.0]  # +10%, +10%
    b = [100.0, 90.0, 99.0]    # -10%, +10%
    port = combine([a, b])
    # step 1: mean(+0.10, -0.10) = 0 -> stays 100000; step 2: mean(+0.10, +0.10) = +0.10 -> 110000
    assert np.allclose(port, [100000.0, 100000.0, 110000.0])


def test_combine_diversification_cuts_volatility_below_either_leg():
    # two anti-correlated legs -> the equal-weight portfolio's step-return vol is BELOW each leg's (the whole
    # point of breadth). This is the property the diversified-trend experiment relies on.
    up_down = [100.0, 110.0, 100.0, 110.0, 100.0]
    down_up = [100.0, 90.0, 100.0, 90.0, 100.0]
    port = combine([up_down, down_up])
    v_port = step_returns(port).std()
    assert v_port < step_returns(up_down).std()
    assert v_port < step_returns(down_up).std()


def test_combine_truncates_to_shortest_curve():
    port = combine([[100.0, 110.0, 120.0], [100.0, 90.0]])  # lengths 3 and 2 -> 1 aligned step
    assert len(port) == 2  # initial + 1 combined step


def test_inverse_vol_weights_downweights_the_noisier_leg():
    calm = [100.0, 101.0, 102.0, 103.0]        # low vol
    wild = [100.0, 130.0, 80.0, 140.0]         # high vol
    w = inverse_vol_weights([calm, wild])
    assert w[0] > w[1] and abs(w.sum() - 1.0) < 1e-9


def test_diversified_stats_matches_combine_then_stats():
    curves = [[100.0, 110.0, 121.0], [100.0, 105.0, 110.0]]
    assert diversified_stats(curves) == curve_stats(combine(curves))
