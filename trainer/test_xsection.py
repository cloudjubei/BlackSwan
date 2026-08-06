"""Direct unit tests for the cross-sectional (B1 §1) screen core.

These are GATING correctness tests, not coverage: a cross-sectional backtest has failure modes the
single-asset line never had — a misaligned N-symbol join silently fabricates P&L, and ranking against
today's symbol list is survivorship bias. Each is pinned here before the screen is allowed to produce a
number, alongside the same time-prefix causality property the single-asset providers carry.
"""

import numpy as np
import pandas as pd
import pytest

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


def test_a_non_positive_lookback_is_refused_rather_than_silently_reading_the_future():
    # momentum() ranks on `s / s.shift(lookback)`. With a NEGATIVE lookback that shift runs backwards, so
    # the ratio is computed against a bar that has not happened yet and the rank becomes a peek at the
    # future — while tradeable_mask admits every present bar, because `prior >= lookback` is trivially
    # true for a negative bound. Nothing else in the screen would notice: the result simply looks good.
    dates = pd.date_range("2022-01-01", periods=5, freq="D")
    m = _matrix(A=(dates, [100.0, 110, 120, 130, 140]), B=(dates, [100.0, 90, 80, 70, 60]))
    for bad in (-3, 0):
        with pytest.raises(ValueError):
            xsection.build_weights(m, lookback=bad, k=1, rebalance_days=1, long_only=False)


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


# --- the reversal arm ---------------------------------------------------------------------------------
#
# The B1 screen ran 96 momentum cells and failed its gate; its single worst window was stk-2023 at mean
# -63.50%, a documented momentum-crash year in which the beaten-down names this rule SHORTS were the ones
# that ripped. A loss that large IS structure — with the opposite sign. `signal="reversal"` is the exact
# rank inversion, and nothing else changes, so a reversal cell is directly comparable to its momentum twin.


def _golden_matrix():
    """One fixture that exercises every branch of the loop: a warm-up with no book, a rebalance, a HOLD
    between rebalances, and a rank flip when the leader rolls over."""
    dates = pd.date_range("2022-01-01", periods=8, freq="D")
    return _matrix(
        AAA=(dates, [100.0, 104, 108, 112, 109, 105, 101, 99]),
        BBB=(dates, [100.0, 99, 101, 103, 108, 114, 121, 129]),
        CCC=(dates, [100.0, 101, 100, 99, 100, 101, 100, 99]),
    )


def test_the_default_signal_reproduces_todays_momentum_book_byte_for_byte():
    # BACKWARD COMPATIBILITY, non-negotiable. 96 screen cells are already persisted against the momentum
    # rule; if adding the lever moved the default book by so much as a rounding step, every one of those
    # numbers would quietly stop meaning what its record says it means. The literal below was captured
    # from the pre-lever implementation, so this pins TODAY's book rather than merely pinning the code
    # against itself.
    m = _golden_matrix()
    expected = pd.DataFrame(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, -1.0],
            [1.0, 0.0, -1.0],
            [1.0, 0.0, -1.0],
            [-1.0, 1.0, 0.0],
            [-1.0, 1.0, 0.0],
        ],
        index=m.index,
        columns=["AAA", "BBB", "CCC"],
    )
    absent = xsection.build_weights(m, lookback=2, k=1, rebalance_days=3, long_only=False)
    explicit = xsection.build_weights(
        m, lookback=2, k=1, rebalance_days=3, long_only=False, signal="momentum"
    )
    pd.testing.assert_frame_equal(absent, expected)
    pd.testing.assert_frame_equal(explicit, expected)


def test_the_default_book_is_unchanged_at_k_above_one_long_only_and_behind_the_start_gate():
    """The k=1 golden above covers one corner of the lever space; the persisted cells were swept across all
    of it (k defaults to 3, and every cell is start-gated to its walk-forward test window). A momentum
    regression that only bites at k>=2, or only on the long-only reference arm, or only in how the start
    gate opens, would slip past a single k=1 matrix and silently re-price the screen's whole evidence base.
    All three literals below were captured from the pre-lever implementation, same as the k=1 one."""
    dates = pd.date_range("2022-01-01", periods=10, freq="D")
    m = _matrix(
        AAA=(dates, [100.0, 104, 109, 115, 112, 106, 101, 98, 96, 99]),
        BBB=(dates, [100.0, 99, 101, 104, 110, 117, 125, 132, 138, 141]),
        CCC=(dates, [100.0, 101, 100, 99, 100, 101, 100, 99, 100, 101]),
        DDD=(dates, [100.0, 97, 95, 92, 94, 98, 103, 109, 114, 112]),
        EEE=(dates, [100.0, 103, 106, 110, 107, 103, 99, 95, 92, 90]),
    )
    kwargs = dict(lookback=2, k=2, rebalance_days=3)
    cases = [
        (
            dict(long_only=False),
            [[0.0] * 5] * 3
            + [[0.5, 0.0, -0.5, -0.5, 0.5]] * 3
            + [[-0.5, 0.5, 0.0, 0.5, -0.5]] * 4,
        ),
        (
            dict(long_only=True),
            [[0.0] * 5] * 3
            + [[0.5, 0.0, 0.0, 0.0, 0.5]] * 3
            + [[0.0, 0.5, 0.0, 0.5, 0.0]] * 4,
        ),
        (
            dict(long_only=False, start=dates[5]),
            [[0.0] * 5] * 6 + [[-0.5, 0.5, 0.0, 0.5, -0.5]] * 4,
        ),
    ]
    for extra, rows in cases:
        expected = pd.DataFrame(rows, index=m.index, columns=list(m.columns))
        pd.testing.assert_frame_equal(xsection.build_weights(m, **kwargs, **extra), expected)
        pd.testing.assert_frame_equal(
            xsection.build_weights(m, **kwargs, signal="momentum", **extra), expected
        )


def test_reversal_longs_the_bottom_k_and_shorts_the_top_k_by_trailing_return():
    # The mirror of test_longs_the_top_k_and_shorts_the_bottom_k_by_trailing_return: same fixture, same
    # net-flat / gross-2 construction, opposite sign on every leg.
    dates = pd.date_range("2022-01-01", periods=5, freq="D")
    m = _matrix(
        WIN=(dates, [100.0, 110, 120, 130, 140]),
        MID=(dates, [100.0, 100, 100, 100, 100]),
        LOSE=(dates, [100.0, 90, 80, 70, 60]),
    )
    w = xsection.build_weights(
        m, lookback=2, k=1, rebalance_days=1, long_only=False, signal="reversal"
    )
    last = w.iloc[-1]
    assert last["LOSE"] > 0 and last["WIN"] < 0 and last["MID"] == 0.0
    assert abs(last.sum()) < 1e-12  # inverting the rank does not cost the beta-neutrality
    assert abs(last.abs().sum() - 2.0) < 1e-12


def test_reversal_is_the_exact_negation_of_momentum_on_a_strictly_ordered_universe():
    # The mirror holds exactly when the rank is UNAMBIGUOUS — i.e. no ties among the rankable names. Ties
    # are the only thing that breaks it: `sort_values` defaults to quicksort, which is not stable, so tied
    # names are ordered arbitrarily and the two directions need not disagree symmetrically about which of
    # them they take. Universe parity and a k that leaves a middle name out do NOT break it (see
    # test_the_mirror_survives_an_odd_universe_and_an_untradeable_symbol) — a name outside both legs is
    # zero on both arms, and zero is its own negation.
    dates = pd.date_range("2022-01-01", periods=6, freq="D")
    m = _matrix(
        A=(dates, [100.0, 110, 121, 118, 112, 105]),
        B=(dates, [100.0, 105, 103, 108, 116, 126]),
        C=(dates, [100.0, 98, 99, 101, 99, 97]),
        D=(dates, [100.0, 90, 85, 88, 94, 103]),
    )
    mom = xsection.momentum(m, 2)
    for ts in m.index[2:]:
        row = mom.loc[ts].dropna()
        assert len(row) == 4 and len(set(row.round(12))) == 4  # strict order, whole universe rankable
    kwargs = dict(lookback=2, k=2, rebalance_days=1, long_only=False)
    momentum_book = xsection.build_weights(m, signal="momentum", **kwargs)
    reversal_book = xsection.build_weights(m, signal="reversal", **kwargs)
    assert momentum_book.abs().sum(axis=1).iloc[-1] == 2.0  # all four names held, nothing dropped
    pd.testing.assert_frame_equal(reversal_book, -momentum_book)


def test_reversal_respects_the_survivorship_mask_exactly_as_momentum_does():
    # C lists two bars from the end with a halving. Without the tradeable/ranking gate it would look like
    # the universe's extreme on BOTH signals at once — momentum would short it, reversal would long it —
    # so a mask that leaks shows up as a held position under either lever.
    dates = pd.date_range("2022-01-01", periods=8, freq="D")
    m = _matrix(
        A=(dates, [100.0, 102, 104, 106, 108, 110, 112, 114]),
        B=(dates, [100.0, 100.5, 101, 101.5, 102, 102.5, 103, 103.5]),
        C=(dates[6:], [20.0, 10.0]),
    )
    for signal in ("momentum", "reversal"):
        w = xsection.build_weights(
            m, lookback=2, k=1, rebalance_days=1, long_only=False, signal=signal
        )
        assert (w["C"] == 0.0).all()
        assert abs(w.iloc[-1].abs().sum() - 2.0) < 1e-12  # the book is still full, just not from C


def test_long_only_composes_with_reversal_to_hold_the_bottom_k_at_gross_one():
    dates = pd.date_range("2022-01-01", periods=5, freq="D")
    m = _matrix(
        WIN=(dates, [100.0, 110, 120, 130, 140]),
        LOSE=(dates, [100.0, 90, 80, 70, 60]),
    )
    w = xsection.build_weights(
        m, lookback=2, k=1, rebalance_days=1, long_only=True, signal="reversal"
    )
    last = w.iloc[-1]
    assert last["LOSE"] == 1.0 and last["WIN"] == 0.0
    assert abs(last.abs().sum() - 1.0) < 1e-12


def test_the_mirror_survives_an_odd_universe_and_an_untradeable_symbol():
    # Guards the SCOPE of the negation property, because a precondition stated more broadly than it is true
    # gets copied into the next arm and quietly narrows what anyone bothers to check. Five strictly-ordered
    # names with k=2 leaves C outside both legs, and LATE lists too late to rank at all: the mirror still
    # holds exactly, because a name held by neither arm is zero on both sides.
    dates = pd.date_range("2022-01-01", periods=8, freq="D")
    m = _matrix(
        A=(dates, [100.0, 112, 125, 139, 152, 166, 181, 197]),
        B=(dates, [100.0, 106, 111, 117, 122, 128, 133, 139]),
        C=(dates, [100.0, 101, 102, 103, 104, 105, 106, 107]),
        D=(dates, [100.0, 97, 95, 92, 90, 87, 85, 82]),
        E=(dates, [100.0, 92, 85, 78, 71, 65, 59, 54]),
        LATE=(dates[6:], [500.0, 250.0]),
    )
    kwargs = dict(lookback=2, k=2, rebalance_days=1, long_only=False)
    momentum_book = xsection.build_weights(m, signal="momentum", **kwargs)
    reversal_book = xsection.build_weights(m, signal="reversal", **kwargs)
    assert (momentum_book["C"] == 0.0).all() and (reversal_book["C"] == 0.0).all()
    assert (momentum_book["LATE"] == 0.0).all() and (reversal_book["LATE"] == 0.0).all()
    assert momentum_book.iloc[-1].abs().sum() == 2.0
    pd.testing.assert_frame_equal(reversal_book, -momentum_book)


def test_tied_ranks_still_produce_a_net_flat_book_on_both_arms():
    # Where the mirror legitimately stops (two exactly-tied pairs), the properties the fork actually rests
    # on must NOT stop with it: whichever tied name each arm happens to pick, the book is still one long and
    # one short at full size. This is the invariant worth pinning here — the arbitrary tie ORDER is not.
    dates = pd.date_range("2022-01-01", periods=5, freq="D")
    m = _matrix(
        A=(dates, [100.0, 110, 121, 133.1, 146.41]),
        B=(dates, [100.0, 110, 121, 133.1, 146.41]),
        C=(dates, [100.0, 90, 81, 72.9, 65.61]),
        D=(dates, [100.0, 90, 81, 72.9, 65.61]),
    )
    for signal in xsection.SIGNALS:
        last = xsection.build_weights(
            m, lookback=2, k=1, rebalance_days=1, long_only=False, signal=signal
        ).iloc[-1]
        assert abs(last.sum()) < 1e-12
        assert abs(last.abs().sum() - 2.0) < 1e-12


def test_a_universe_too_small_to_split_is_left_flat_rather_than_double_booked():
    # k=2 over three rankable names: the top-2 and the bottom-2 OVERLAP on the middle one. Sizing that name
    # twice would leave it short while it is also long — a book that is neither net flat nor gross 2k, i.e.
    # the beta-neutrality the whole fork rests on lost at the boundary rather than in the ranking. Both arms
    # must decline to trade instead, and the eligibility guard that makes them is otherwise untested.
    dates = pd.date_range("2022-01-01", periods=5, freq="D")
    m = _matrix(
        WIN=(dates, [100.0, 110, 121, 133.1, 146.41]),
        MID=(dates, [100.0, 100, 100, 100, 100]),
        LOSE=(dates, [100.0, 90, 81, 72.9, 65.61]),
    )
    for signal in xsection.SIGNALS:
        w = xsection.build_weights(
            m, lookback=2, k=2, rebalance_days=1, long_only=False, signal=signal
        )
        assert (w == 0.0).all().all()


def test_an_unrecognised_signal_is_rejected_rather_than_silently_ranked_as_momentum():
    # A typo'd lever that falls back to the default would file a momentum result under a reversal label —
    # the one failure mode that corrupts the evidence trail instead of merely producing a bad number.
    dates = pd.date_range("2022-01-01", periods=4, freq="D")
    m = _matrix(A=(dates, [100.0, 110, 120, 130]), B=(dates, [100.0, 90, 80, 70]))
    with pytest.raises(ValueError):
        xsection.build_weights(
            m, lookback=1, k=1, rebalance_days=1, long_only=False, signal="mean_reversion"
        )


# --- the lever as a CELL sees it -----------------------------------------------------------------------
#
# build_weights' own guards only protect a direct library caller. Everything a persisted cell depends on
# lives one level up, in run(): whether the cfg lever reaches the book at all, and whether an absent lever
# still means momentum. Nothing inside build_weights can catch a run() that drops the argument — and a run()
# that drops it files a MOMENTUM book under a reversal label, which corrupts the evidence trail rather than
# merely losing money. That wiring is exactly what the 96 persisted momentum cells and every reversal twin
# are read through, so it is pinned here rather than checked by hand.

_RUN_CFG = {
    "walk_forward_window": "stk-2024",
    "lookback": 2,
    "k": 1,
    "rebalance_days": 1,
    "long_only": False,
    "transaction_fee": 0.0,
}
_RUN_TEST_START = pd.Timestamp("2024-01-01")


def _oscillating_frames():
    """Two symbols held in opposite phase, spanning the train tail and the whole stk-2024 test window:
    chasing the rank and fading it land far apart here, so a cell that silently ran the other arm cannot
    hide inside a rounding tolerance."""
    dates = pd.date_range("2023-11-01", periods=120, freq="D")
    steps = np.arange(120)
    return {
        "AAA": _frame(dates, list(100.0 * (1.0 + 0.05 * np.sin(steps / 3.0)))),
        "BBB": _frame(dates, list(100.0 * (1.0 + 0.05 * np.cos(steps / 3.0)))),
    }


def _stub_loader(monkeypatch, frames):
    """Replace the on-disk universe load, returning the list of calls so a test can assert it never ran."""
    calls = []

    def _fake(symbols, pairs):
        calls.append((tuple(symbols), len(pairs)))
        return dict(frames)

    monkeypatch.setattr(xsection, "_load_universe", _fake)
    return calls


def test_a_cell_without_the_lever_is_the_momentum_cell_metric_for_metric(monkeypatch):
    _stub_loader(monkeypatch, _oscillating_frames())
    absent = xsection.run(dict(_RUN_CFG))
    explicit = xsection.run({**_RUN_CFG, "signal": "momentum"})
    assert absent["objective"] == explicit["objective"]
    assert absent["metrics"] == explicit["metrics"]


def test_a_reversal_cell_actually_trades_the_reversal_book(monkeypatch):
    frames = _oscillating_frames()
    _stub_loader(monkeypatch, frames)
    momentum_cell = xsection.run({**_RUN_CFG, "signal": "momentum"})
    reversal_cell = xsection.run({**_RUN_CFG, "signal": "reversal"})

    prices = xsection.align_prices(frames)
    oos = prices.index >= _RUN_TEST_START
    book = xsection.build_weights(prices, 2, 1, 1, False, "reversal", start=_RUN_TEST_START)
    equity = xsection.backtest(prices[oos], book[oos], fee=0.0)
    assert reversal_cell["objective"] == pytest.approx(float(equity.iloc[-1] - 1.0) * 100)
    assert abs(reversal_cell["objective"] - momentum_cell["objective"]) > 1.0
    # The persisted record has to say which arm produced it, or the two are indistinguishable after the fact.
    assert reversal_cell["config"]["signal"] == "reversal"


def test_a_typod_signal_kills_the_cell_before_any_data_is_read(monkeypatch):
    # Same shape as the unknown-universe check it sits beside: a clean SystemExit, and no wasted disk load.
    calls = _stub_loader(monkeypatch, _oscillating_frames())
    with pytest.raises(SystemExit) as raised:
        xsection.run({**_RUN_CFG, "signal": "mean_reversion"})
    assert "mean_reversion" in str(raised.value)
    assert calls == []


# --- causality: the whole reason this is trustworthy ---------------------------------------------------
#
# Parametrized over SIGNALS rather than run on the default alone: an arm that is only ever exercised through
# the rule tests is an arm whose zero-lookahead property nobody is checking, and adding one is a one-word
# change to a sort. Driving it off the tuple means a third arm inherits both guards the day it is declared.


@pytest.mark.parametrize("signal", xsection.SIGNALS)
def test_weights_are_time_prefix_causal(signal):
    """Corrupt every bar from `cut` onward; the book held into `cut` and everything before it must be
    byte-identical.

    Three details are what make this bite, and getting any of them wrong turns the test into a formality
    that a badly lookahead-ing implementation sails through:

    * The shock is a DIFFERENT factor per symbol. A uniform one rescales every symbol's trailing return by
      the same constant, leaves the cross-sectional RANK untouched, and therefore cannot change the book at
      any bar — clean or lookahead-ing.
    * The comparison includes row `cut` itself. Row t is the book HELD INTO t, decided at t-1, so it too
      must survive the corruption; stopping one row short leaves exactly one bar of slack, which is the
      shape a lookahead bug actually takes.
    * Every cut must be shown to MOVE the book somewhere after the boundary. An assertion that nothing
      changed is worthless without evidence that something could have.
    """
    dates = pd.date_range("2022-01-01", periods=20, freq="D")
    rng = np.random.default_rng(0)
    base = {s: (dates, list(100 + np.cumsum(rng.normal(0, 1, 20)))) for s in ("A", "B", "C", "D")}
    m = _matrix(**base)
    kwargs = dict(lookback=3, k=1, rebalance_days=2, long_only=False, signal=signal)
    clean = xsection.build_weights(m, **kwargs)
    for cut in range(5, 18):
        dirty_m = m.copy()
        dirty_m.iloc[cut:] = dirty_m.iloc[cut:] * [5.0, 0.2, 1.0, 3.0]
        dirty = xsection.build_weights(dirty_m, **kwargs)
        pd.testing.assert_frame_equal(clean.iloc[: cut + 1], dirty.iloc[: cut + 1])
        assert not clean.iloc[cut + 1 :].equals(dirty.iloc[cut + 1 :])


@pytest.mark.parametrize("signal", xsection.SIGNALS)
def test_a_signal_is_never_traded_on_its_own_bar(signal):
    # Shock ONE bar hard enough to own the rank, then read the book either side of it: the weight held into
    # that bar was decided a bar earlier and must not have seen it, while the NEXT row must move — that
    # second half is what stops this collapsing into "row zero is flat", which is true of any book,
    # lookahead or not, because the loop writes the opening row before it has decided anything.
    dates = pd.date_range("2022-01-01", periods=8, freq="D")
    base = {
        "A": [100.0, 101, 102, 103, 104, 105, 106, 107],
        "B": [100.0, 99, 98, 97, 96, 95, 94, 93],
        "C": [100.0, 100.5, 101, 101.5, 102, 102.5, 103, 103.5],
    }
    shocked = {s: list(p) for s, p in base.items()}
    shocked["B"][4] = 400.0
    kwargs = dict(lookback=1, k=1, rebalance_days=1, long_only=False, signal=signal)
    clean = xsection.build_weights(_matrix(**{s: (dates, p) for s, p in base.items()}), **kwargs)
    dirty = xsection.build_weights(_matrix(**{s: (dates, p) for s, p in shocked.items()}), **kwargs)
    assert clean.iloc[0].abs().sum() == 0.0
    pd.testing.assert_frame_equal(clean.iloc[:5], dirty.iloc[:5])
    assert not clean.iloc[5].equals(dirty.iloc[5])


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
