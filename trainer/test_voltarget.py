"""Direct unit tests for the volatility-target (B1 §2) screen core.

These are GATING correctness tests, not coverage. The screen exists because trailing 20d VOL predicts next
20d vol (+0.426, positive on 12/12 symbols on disk) while trailing 20d RETURN does not (+0.015) — but a
vol-targeting rule is trivially easy to write with lookahead: the moment the sizing target is a full-sample
constant, the backtest knows the future volatility regime and every number it emits is fiction. The
expanding, strictly-past target is therefore pinned here before the screen is allowed to produce a number,
alongside the time-prefix causality property the other providers carry, the cap that keeps a collapsing
denominator from fabricating infinite leverage, and the turnover accounting that killed several earlier arms.
"""

import numpy as np
import pandas as pd

from trainer import voltarget


def _prices(returns, start=100.0):
    """A daily price path whose per-step returns are EXACTLY `returns` — so the expected trailing vol is
    known in closed form rather than re-derived from the code under test."""
    vals = [float(start)]
    for r in returns:
        vals.append(vals[-1] * (1.0 + r))
    return pd.Series(vals, index=pd.date_range("2022-01-01", periods=len(vals), freq="D"))


def _alternating(n, magnitude):
    return [magnitude if i % 2 == 0 else -magnitude for i in range(n)]


def _rising_vol_prices(n=24, step=0.002):
    """A path whose trailing vol RISES at every bar (alternating returns of growing magnitude), so the
    expanding median of that vol also moves at every bar.

    This shape exists because a median is deliberately robust, and robustness is what hides an off-by-one: on
    a calm path one extra vol observation leaves the running median bit-for-bit unchanged, so a target that
    peeks one bar into the future is indistinguishable from the causal one. Only on a monotone path do the two
    rules disagree everywhere, which is the condition a causality guard has to be tested under to be worth
    anything."""
    return _prices([step * (i + 1) * (1.0 if i % 2 == 0 else -1.0) for i in range(n)])


# --- the trailing estimator ---------------------------------------------------------------------------


def test_trailing_vol_is_the_stdev_of_the_last_window_returns_up_to_and_including_t():
    px = _prices([0.01, -0.02, 0.03, -0.04, 0.05])
    vols = voltarget.trailing_vol(px, 3)
    rets = voltarget.step_returns(px)
    assert vols.iloc[:3].isna().all()  # the first bar has no return, so 3 returns only exist from bar 3
    assert abs(vols.iloc[3] - np.std(rets.values[1:4], ddof=0)) < 1e-15
    assert abs(vols.iloc[5] - np.std(rets.values[3:6], ddof=0)) < 1e-15


def test_expanding_target_is_the_running_median_of_every_vol_observed_through_that_bar():
    vols = pd.Series([np.nan, 1.0, 3.0, 5.0, 100.0])
    target = voltarget.expanding_vol_target(vols)
    assert np.isnan(target.iloc[0])
    assert target.iloc[1] == 1.0
    assert target.iloc[2] == 2.0
    assert target.iloc[3] == 3.0
    assert target.iloc[4] == 4.0  # median(1,3,5,100) — a MEDIAN, so the late shock cannot drag the target


# --- causality: the whole reason this is trustworthy ---------------------------------------------------


def test_the_target_uses_only_strictly_past_vols_not_the_full_sample_median():
    # 20 calm bars then 60 wild ones. A full-sample median target would size the calm bars against the
    # volatility of a regime that has not happened yet — the single failure that would invalidate the screen.
    px = _prices(_alternating(20, 0.001) + _alternating(60, 0.05))
    cap = 5.0
    w = voltarget.build_weights(px, vol_window=4, weight_cap=cap, rebalance_days=1)
    vols = voltarget.trailing_vol(px, 4)

    decision = 12  # a calm bar; its weight is traded into bar 13
    v_prev = float(vols.iloc[decision])
    causal = min(float(np.nanmedian(vols.values[: decision + 1])) / v_prev, cap)
    full_sample = min(float(np.nanmedian(vols.values)) / v_prev, cap)

    assert full_sample - causal > 1.0  # the two rules genuinely disagree here, by a lot
    assert abs(w.iloc[decision + 1] - causal) < 1e-12
    assert abs(w.iloc[decision + 1] - full_sample) > 1.0


def test_the_weight_traded_into_the_next_bar_uses_the_median_of_vols_through_the_decision_bar_only():
    # The highest-risk property in the module, pinned to the BAR rather than to the shape of the rule. The
    # full-sample test above catches a target that knows the whole sample; it cannot catch a target that knows
    # ONE bar too much, and one bar is the entire difference between a screen and a fiction. On a rising-vol
    # path the causal target and the target that peeks a single bar ahead disagree at every bar, so the second
    # assertion below is what proves the first one is capable of biting.
    px = _rising_vol_prices()
    cap, window = 50.0, 4
    w = voltarget.build_weights(px, vol_window=window, weight_cap=cap, rebalance_days=1)
    vols = voltarget.trailing_vol(px, window).values
    checked = 0
    for t in range(window, len(px) - 1):
        causal = min(max(float(np.nanmedian(vols[: t + 1])) / vols[t], 0.0), cap)
        peeking = min(max(float(np.nanmedian(vols[: t + 2])) / vols[t], 0.0), cap)
        assert abs(peeking - causal) > 1e-6  # the off-by-one is observable at this bar
        assert abs(w.iloc[t + 1] - causal) < 1e-12
        checked += 1
    assert checked >= 15


def test_weights_are_time_prefix_causal():
    # Corrupt the path from bar `cut` onward: the book held into bars 0..cut may not move, because row t is
    # decided at t-1 and cannot know price t. The corruption has to be a genuinely DIFFERENT path rather than
    # a rescaled one — multiplying a tail by a constant changes exactly one return, which leaves a one-bar
    # peek nothing to see — and the mirror assertion (the very next row MUST move) is what stops the
    # invariance half from passing vacuously on a book that simply never trades.
    rng = np.random.default_rng(0)
    px = _prices(list(rng.normal(0.0, 0.02, 39)))
    clean = voltarget.build_weights(px, vol_window=5, weight_cap=2.0, rebalance_days=1)
    for cut in range(6, len(px) - 1):
        dirty_px = px.copy()
        shock = np.array([1.4 if k % 2 == 0 else 0.7 for k in range(len(px) - cut)])
        dirty_px.iloc[cut:] = px.iloc[cut:].values * shock
        dirty = voltarget.build_weights(dirty_px, vol_window=5, weight_cap=2.0, rebalance_days=1)
        pd.testing.assert_series_equal(clean.iloc[: cut + 1], dirty.iloc[: cut + 1])
        assert abs(clean.iloc[cut + 1] - dirty.iloc[cut + 1]) > 1e-9


def test_the_start_gate_delays_the_first_trade_without_restarting_the_formation_history():
    # The seam `run` actually uses and no other test reaches: decisions may not begin before the test window,
    # but the estimators must still carry every vol observed during formation. A gate that also truncated the
    # history would size the first test bar against a one-observation median — a target of exactly its own
    # vol, i.e. a weight of 1.0 — which looks perfectly reasonable and is a different rule from the published
    # one.
    px = _rising_vol_prices(n=40)
    cap, window, gate = 50.0, 4, 25
    gated = voltarget.build_weights(px, window, cap, 1, start=px.index[gate])
    assert (gated.iloc[: gate + 1] == 0.0).all()  # nothing is held into the first eligible bar
    vols = voltarget.trailing_vol(px, window).values
    full = min(max(float(np.nanmedian(vols[: gate + 1])) / vols[gate], 0.0), cap)
    restarted = min(max(float(np.nanmedian(vols[gate : gate + 1])) / vols[gate], 0.0), cap)
    assert abs(full - restarted) > 1e-6  # the two rules genuinely disagree at this bar
    assert abs(gated.iloc[gate + 1] - full) < 1e-12


def test_a_signal_is_never_traded_on_its_own_bar():
    # A price shock at bar 20 must not move the weight applied to the 19 -> 20 return; it may only change
    # the book from bar 21 onward.
    px = _prices(_alternating(30, 0.005))
    shocked = px.copy()
    shocked.iloc[20:] = shocked.iloc[20:] * 1.5
    base_w = voltarget.build_weights(px, vol_window=4, weight_cap=3.0, rebalance_days=1)
    shock_w = voltarget.build_weights(shocked, vol_window=4, weight_cap=3.0, rebalance_days=1)
    assert base_w.iloc[0] == 0.0  # row t carries what was decided at t-1, so bar 0 is necessarily flat
    pd.testing.assert_series_equal(base_w.iloc[:21], shock_w.iloc[:21])
    assert abs(base_w.iloc[21] - shock_w.iloc[21]) > 1e-6


# --- the rule ------------------------------------------------------------------------------------------


def test_constant_volatility_produces_a_constant_weight_after_warm_up():
    px = _prices(_alternating(40, 0.01))
    w = voltarget.build_weights(px, vol_window=4, weight_cap=2.0, rebalance_days=1)
    assert (w.iloc[:5] == 0.0).all()  # 4 returns are needed, so the first sized bar is 5
    live = w.iloc[5:]
    assert (live - 1.0).abs().max() < 1e-9  # vol == its own running median -> fully invested, never levered
    assert live.max() - live.min() < 1e-9


def test_warm_up_holds_zero_weight_rather_than_nan_and_the_curve_stays_finite():
    px = _prices(list(np.random.default_rng(3).normal(0.0, 0.02, 19)))
    w = voltarget.build_weights(px, vol_window=10, weight_cap=1.0, rebalance_days=1)
    assert not w.isna().any()
    assert (w.iloc[:11] == 0.0).all()
    equity = voltarget.backtest(px, w, fee=0.0002)
    assert np.isfinite(equity.values).all()


def test_weight_cap_is_respected_when_vol_collapses_toward_zero():
    # 30 live bars then a dead-flat tape: the denominator goes to exactly 0 and the naive ratio is infinite.
    px = _prices(_alternating(30, 0.01) + [0.0] * 30)
    cap = 2.0
    w = voltarget.build_weights(px, vol_window=5, weight_cap=cap, rebalance_days=1)
    assert np.isfinite(w.values).all()
    assert w.max() <= cap + 1e-12
    assert w.iloc[-1] == cap  # the collapse path is exercised, and the cap IS the answer there
    equity = voltarget.backtest(px, w, fee=0.001)
    assert np.isfinite(equity.values).all()


def test_sizing_never_returns_an_infinite_or_nan_weight():
    assert voltarget.size_weight(0.02, 0.01, 3.0) == 2.0  # the raw ratio, when the cap does not bind
    assert voltarget.size_weight(0.02, 0.01, 1.5) == 1.5
    assert voltarget.size_weight(0.02, 0.0, 1.5) == 1.5
    assert voltarget.size_weight(0.02, 1e-300, 1.5) == 1.5
    assert voltarget.size_weight(float("nan"), 0.01, 1.5) is None
    assert voltarget.size_weight(0.02, float("nan"), 1.5) is None


# --- costs ---------------------------------------------------------------------------------------------


def test_row_t_weight_multiplies_the_t_minus_one_to_t_return():
    # The causality contract restated in the ACCOUNTING: build_weights can be perfectly causal and the
    # backtest can still smuggle the future in by pairing a row with the wrong step's return.
    px = _prices([0.10, 0.20])
    w = pd.Series([0.0, 1.0, 0.0], index=px.index)
    equity = voltarget.backtest(px, w, fee=0.0)
    assert abs(equity.iloc[1] - 1.10) < 1e-12  # row 1 was invested, so it earns the 0 -> 1 step
    assert abs(equity.iloc[2] - 1.10) < 1e-12  # row 2 is flat, so the 1 -> 2 step is NOT earned


def test_fee_is_charged_on_turnover_both_ways():
    px = _prices([0.0, 0.0, 0.0])
    w = pd.Series([0.0, 1.0, 1.0, 0.0], index=px.index)
    assert voltarget.backtest(px, w, fee=0.0).iloc[-1] == 1.0
    paid = voltarget.backtest(px, w, fee=0.01).iloc[-1]
    assert abs(paid - 0.99 * 0.99) < 1e-15  # the entry AND the exit are each charged


def test_turnover_counts_the_trade_that_opens_the_book():
    # Row 0 carries a book, and the book did not appear out of thin air: it was bought from cash. Measuring
    # turnover as a bare diff() silently makes that first trade free, which is the direction that FLATTERS.
    steps = voltarget.turnover_steps(pd.Series([1.0, 1.0, 0.4, 0.0]))
    assert list(steps.values) == [1.0, 0.0, 0.6, 0.4]
    assert float(steps.sum()) == 2.0


def test_a_book_already_open_at_the_first_bar_pays_to_have_got_there():
    # The one case a plain diff() cannot see: open at row 0, closed at row 1. That is two full trades, and
    # charging neither of them would let a caller hand the screen a pre-loaded book for free.
    px = _prices([0.0, 0.0])
    w = pd.Series([1.0, 0.0, 0.0], index=px.index)
    assert abs(voltarget.backtest(px, w, fee=0.01).iloc[-1] - 0.99 * 0.99) < 1e-15


def test_a_longer_rebalance_interval_charges_strictly_less_turnover():
    px = _prices(list(np.random.default_rng(7).normal(0.0, 0.02, 119)))
    fast = voltarget.build_weights(px, vol_window=10, weight_cap=2.0, rebalance_days=1)
    slow = voltarget.build_weights(px, vol_window=10, weight_cap=2.0, rebalance_days=10)
    fast_turnover = float(fast.diff().abs().sum())
    slow_turnover = float(slow.diff().abs().sum())
    assert fast_turnover > 0.0
    assert slow_turnover < fast_turnover
    assert voltarget.backtest(px, slow, 0.01).iloc[-1] != voltarget.backtest(px, fast, 0.01).iloc[-1]


def test_weight_cap_zero_gives_a_flat_book_with_no_fees():
    px = _prices(list(np.random.default_rng(11).normal(0.0, 0.02, 39)))
    w = voltarget.build_weights(px, vol_window=5, weight_cap=0.0, rebalance_days=1)
    assert (w == 0.0).all()
    assert voltarget.backtest(px, w, fee=0.01).iloc[-1] == 1.0


# --- the benchmark -------------------------------------------------------------------------------------


def test_hold_benchmark_on_a_flat_series_is_exactly_the_fee_drag():
    px = _prices([0.0, 0.0, 0.0])
    curve = voltarget.hold_curve(px, fee=0.002)
    assert curve.iloc[-1] == 1.0 * (1.0 - 0.002)
    assert voltarget.hold_curve(px, fee=0.0).iloc[-1] == 1.0


def test_hold_benchmark_compounds_the_assets_own_returns():
    px = _prices([0.1, 0.1, 0.1])
    assert abs(voltarget.hold_curve(px, fee=0.0).iloc[-1] - 1.1**3) < 1e-12


def test_a_permanently_fully_invested_book_is_the_hold_benchmark_exactly():
    # Benchmark symmetry stated as an IDENTITY rather than as a claim in a docstring: the hold curve must be
    # nothing more than what the strategy machinery produces for a book that is always fully invested. Equal
    # final values are not enough — every gate here reads `_oos_stats` and `_max_drawdown_pct`, which see the
    # per-step returns, so a fee booked into a different bar on one side than the other tilts the comparison.
    px = _prices([0.03, -0.02, 0.05, -0.01])
    ones = pd.Series(1.0, index=px.index)
    pd.testing.assert_series_equal(
        voltarget.backtest(px, ones, fee=0.002), voltarget.hold_curve(px, fee=0.002)
    )


def test_de_risking_reads_as_a_win_on_drawdown_and_sharpe_even_though_it_earns_less():
    # The whole point of the screen: it will usually earn LESS than buy-and-hold, so the gate reads
    # sharpe_vs_hold and drawdown_vs_hold_pct. Both must be signed so that "better" is POSITIVE.
    hold = [1.0, 1.5, 0.9, 1.2]
    equity = [1.0, 1.1, 1.05, 1.08]
    m = voltarget.benchmark_metrics(equity, hold)
    assert equity[-1] < hold[-1]  # de-risked: strictly less raw return
    assert abs(m["hold_max_drawdown_pct"] - (0.9 / 1.5 - 1.0) * 100) < 1e-9
    assert m["drawdown_vs_hold_pct"] > 0  # shallower drawdown reads POSITIVE
    assert abs(m["drawdown_vs_hold_pct"] - ((1.05 / 1.1 - 1.0) * 100 - (0.9 / 1.5 - 1.0) * 100)) < 1e-9
    assert m["sharpe_vs_hold"] > 0
    assert abs(m["sharpe_vs_hold"] - (m["oos_sharpe"] - m["hold_sharpe"])) < 1e-12


def test_benchmark_metrics_are_skippable_when_a_curve_is_too_short():
    assert voltarget.benchmark_metrics([1.0], [1.0]) == {}


# --- the cell ------------------------------------------------------------------------------------------


def _stub_loader(monkeypatch, series):
    """Replace the on-disk load so the accounting boundary can be pinned without touching a data directory."""

    def _fake(symbol, pairs):
        return series

    monkeypatch.setattr(voltarget, "_load_asset", _fake)


def test_only_the_test_windows_pnl_is_accounted_while_the_estimators_keep_their_formation_history(monkeypatch):
    # A price path that TRIPLES through the train span and then drifts gently through 2024. If the accounting
    # boundary slipped even one month earlier, the train run-up would land in the reported return and a null
    # would read as a triumph — so the hold benchmark is pinned to the exact arithmetic of the test span, and
    # the bar count to the exact number of test-window bars.
    dates = pd.date_range("2023-06-01", "2024-12-31", freq="D")
    test_start = pd.Timestamp("2024-01-01")
    n_train = int((dates < test_start).sum())
    vals = list(np.linspace(100.0, 300.0, n_train))
    tail = len(dates) - n_train
    vals += [300.0 * (1.0 + 0.01 * np.sin(k / 4.0)) * (1.0 + 0.0002 * k) for k in range(tail)]
    px = pd.Series(vals, index=dates)
    _stub_loader(monkeypatch, px)

    fee = 0.001
    cell = voltarget.run(
        {"asset": "SPY", "vol_window": 10, "weight_cap": 1.0, "rebalance_days": 5,
         "transaction_fee": fee, "walk_forward_window": "stk-2024"}
    )
    oos = px[px.index >= test_start]
    assert cell["metrics"]["bars"] == len(oos) == tail
    assert cell["dataset"]["from"] == str(oos.index[0])
    assert cell["dataset"]["to"] == str(oos.index[-1])
    expected_hold = (float(oos.iloc[-1]) / float(oos.iloc[0]) * (1.0 - fee) - 1.0) * 100
    assert abs(cell["metrics"]["hold_return_pct"] - expected_hold) < 1e-9
    slipped = px[px.index >= (test_start - pd.Timedelta(days=31))]
    slipped_hold = (float(slipped.iloc[-1]) / float(slipped.iloc[0]) * (1.0 - fee) - 1.0) * 100
    assert slipped_hold - expected_hold > 10.0  # one month of slip is loud here, so the pin above can bite

    # Exposure is the gate's own degeneracy check (mean deployed exposure >= 0.5), so it has to be averaged
    # over the ACCOUNTED window and not over a series two-thirds of which is pre-gate zeros.
    book = voltarget.build_weights(px, 10, 1.0, 5, start=test_start)[px.index >= test_start]
    assert abs(cell["metrics"]["mean_exposure"] - float(book.mean())) < 1e-12
    assert abs(cell["metrics"]["max_exposure"] - float(book.max())) < 1e-12
    # The formation history is still read: the book is live on the very FIRST accounted return, which is only
    # possible because the 10-bar estimator warmed up in the train span instead of inside the scored window.
    assert book.iloc[1] > 0.0
    assert cell["metrics"]["n_trades"] > 0
    assert abs(cell["metrics"]["realized_cost_bps"] - cell["metrics"]["turnover"] * fee * 10000) < 1e-9
