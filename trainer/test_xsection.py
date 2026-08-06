"""Direct unit tests for the cross-sectional (B1 §1) screen core.

These are GATING correctness tests, not coverage: a cross-sectional backtest has failure modes the
single-asset line never had — a misaligned N-symbol join silently fabricates P&L, and ranking against
today's symbol list is survivorship bias. Each is pinned here before the screen is allowed to produce a
number, alongside the same time-prefix causality property the single-asset providers carry.
"""

import numpy as np
import pandas as pd

from trainer import xsection


def _frame(dates, prices):
    return pd.DataFrame({"timestamp_close": pd.to_datetime(dates), "price": prices})


def _matrix(**symbols):
    return xsection.align_prices({k: _frame(*v) for k, v in symbols.items()})


# --- alignment: N symbols on one clock ---------------------------------------------------------------


def test_align_prices_puts_every_symbol_on_one_clock_without_shifting_returns():
    # B trades on a subset of A's calendar (a holiday on the 2nd). A misaligned join would pair B's
    # 3rd-of-the-month price against A's 2nd and invent a return.
    m = _matrix(
        A=(["2022-01-01", "2022-01-02", "2022-01-03"], [100.0, 110.0, 120.0]),
        B=(["2022-01-01", "2022-01-03"], [50.0, 55.0]),
    )
    assert list(m.columns) == ["A", "B"]
    assert len(m) == 3
    assert m.loc[pd.Timestamp("2022-01-02"), "A"] == 110.0
    assert np.isnan(m.loc[pd.Timestamp("2022-01-02"), "B"])  # a hole stays a HOLE, never a forward-fill
    assert m.loc[pd.Timestamp("2022-01-03"), "B"] == 55.0


def test_returns_skip_a_symbols_own_gap_rather_than_pairing_across_it():
    # B's return on the 3rd must be measured against its OWN previous bar (the 1st), not treated as a
    # one-day move, and must never leak into A's series.
    m = _matrix(
        A=(["2022-01-01", "2022-01-02", "2022-01-03"], [100.0, 110.0, 121.0]),
        B=(["2022-01-01", "2022-01-03"], [50.0, 55.0]),
    )
    rets = xsection.step_returns(m)
    assert abs(rets.loc[pd.Timestamp("2022-01-02"), "A"] - 0.1) < 1e-12
    assert rets.loc[pd.Timestamp("2022-01-02"), "B"] == 0.0  # no bar -> no return, not a fabricated one
    assert abs(rets.loc[pd.Timestamp("2022-01-03"), "B"] - 0.1) < 1e-12


# --- survivorship -------------------------------------------------------------------------------------


def test_a_symbol_is_not_tradeable_before_it_has_enough_history():
    # C lists late. Ranking it on day one (or on a partial lookback) is survivorship bias + a bogus signal.
    dates = pd.date_range("2022-01-01", periods=6, freq="D")
    m = _matrix(
        A=(dates, [100.0, 101, 102, 103, 104, 105]),
        C=(dates[3:], [10.0, 11, 12]),
    )
    mask = xsection.tradeable_mask(m, lookback=2)
    assert not mask.loc[dates[0], "A"]  # A itself needs 2 PRIOR bars
    assert mask.loc[dates[2], "A"]
    assert not mask.loc[dates[3], "C"]  # C's first bar
    assert not mask.loc[dates[4], "C"]  # only 1 prior bar
    assert mask.loc[dates[5], "C"]  # 2 prior bars -> tradeable


def test_weights_never_hold_a_symbol_that_is_not_yet_tradeable():
    dates = pd.date_range("2022-01-01", periods=8, freq="D")
    m = _matrix(
        A=(dates, [100.0, 101, 102, 103, 104, 105, 106, 107]),
        B=(dates, [100.0, 99, 98, 97, 96, 95, 94, 93]),
        C=(dates[6:], [10.0, 20.0]),
    )
    w = xsection.build_weights(m, lookback=2, k=1, rebalance_days=1, long_only=False)
    assert (w["C"] == 0.0).all()  # C never has 2 prior bars inside this span


# --- the rule -------------------------------------------------------------------------------------


def test_longs_the_top_k_and_shorts_the_bottom_k_by_trailing_return():
    dates = pd.date_range("2022-01-01", periods=5, freq="D")
    m = _matrix(
        WIN=(dates, [100.0, 110, 120, 130, 140]),
        MID=(dates, [100.0, 100, 100, 100, 100]),
        LOSE=(dates, [100.0, 90, 80, 70, 60]),
    )
    w = xsection.build_weights(m, lookback=2, k=1, rebalance_days=1, long_only=False)
    last = w.iloc[-1]
    assert last["WIN"] > 0 and last["LOSE"] < 0 and last["MID"] == 0.0
    assert abs(last.sum()) < 1e-12  # long/short is NET FLAT by construction — the point of the fork
    assert abs(last.abs().sum() - 2.0) < 1e-12  # gross 2 (1 long + 1 short)


def test_long_only_holds_only_the_top_k_at_gross_one():
    dates = pd.date_range("2022-01-01", periods=5, freq="D")
    m = _matrix(
        WIN=(dates, [100.0, 110, 120, 130, 140]),
        LOSE=(dates, [100.0, 90, 80, 70, 60]),
    )
    w = xsection.build_weights(m, lookback=2, k=1, rebalance_days=1, long_only=True)
    last = w.iloc[-1]
    assert last["WIN"] == 1.0 and last["LOSE"] == 0.0


# --- causality: the whole reason this is trustworthy ---------------------------------------------------


def test_weights_are_time_prefix_causal():
    # Corrupt EVERY bar strictly after t and assert the weights at and before t are byte-identical.
    dates = pd.date_range("2022-01-01", periods=20, freq="D")
    rng = np.random.default_rng(0)
    base = {s: (dates, list(100 + np.cumsum(rng.normal(0, 1, 20)))) for s in ("A", "B", "C", "D")}
    m = _matrix(**base)
    clean = xsection.build_weights(m, lookback=3, k=1, rebalance_days=2, long_only=False)
    dirty_m = m.copy()
    dirty_m.iloc[12:] = dirty_m.iloc[12:] * 5.0
    dirty = xsection.build_weights(dirty_m, lookback=3, k=1, rebalance_days=2, long_only=False)
    pd.testing.assert_frame_equal(clean.iloc[:12], dirty.iloc[:12])


def test_a_signal_is_never_traded_on_its_own_bar():
    # The weight decided from data up to t must apply to the NEXT step's return, never to t's own.
    dates = pd.date_range("2022-01-01", periods=4, freq="D")
    m = _matrix(A=(dates, [100.0, 110, 120, 130]), B=(dates, [100.0, 90, 80, 70]))
    w = xsection.build_weights(m, lookback=1, k=1, rebalance_days=1, long_only=True)
    # Row t carries the weight HELD INTO t (decided at t-1), so the first row must be flat.
    assert w.iloc[0].abs().sum() == 0.0


# --- costs + the benchmark -----------------------------------------------------------------------------


def test_turnover_is_charged_both_ways():
    dates = pd.date_range("2022-01-01", periods=4, freq="D")
    m = _matrix(A=(dates, [100.0, 100, 100, 100]), B=(dates, [100.0, 100, 100, 100]))
    flat = pd.DataFrame(0.0, index=m.index, columns=m.columns)
    flipping = flat.copy()
    flipping.iloc[1:, 0] = 1.0  # enter A and hold: one entry's worth of turnover
    eq_free = xsection.backtest(m, flipping, fee=0.0)
    eq_paid = xsection.backtest(m, flipping, fee=0.01)
    assert eq_free.iloc[-1] == 1.0  # flat prices, no fee -> no change
    assert eq_paid.iloc[-1] < 1.0  # the entry cost real money


def test_basket_benchmark_is_the_equal_weight_tradeable_universe():
    dates = pd.date_range("2022-01-01", periods=4, freq="D")
    m = _matrix(A=(dates, [100.0, 110, 121, 133.1]), B=(dates, [100.0, 100, 100, 100]))
    curve = xsection.basket_curve(m, xsection.tradeable_mask(m, lookback=1), fee=0.0)
    # equal-weight of +10%/step and 0%/step ~= +5%/step compounded
    assert abs(curve.iloc[-1] - 1.05**3) < 1e-6


def test_backtest_handles_an_all_flat_book_and_an_empty_universe():
    dates = pd.date_range("2022-01-01", periods=3, freq="D")
    m = _matrix(A=(dates, [100.0, 101, 102]))
    flat = pd.DataFrame(0.0, index=m.index, columns=m.columns)
    assert xsection.backtest(m, flat, fee=0.001).iloc[-1] == 1.0
    empty = xsection.align_prices({})
    assert empty.empty
