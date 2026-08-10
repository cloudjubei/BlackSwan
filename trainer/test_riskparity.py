"""Direct unit tests for the cross-class risk-parity (Probe A) screen core.

These are GATING correctness properties, not coverage. A vol-scaled, periodically-rebalanced long-only
basket fails in ways the single-asset line never could: the vol estimate is trivially easy to compute with
one bar of lookahead (and the moment it peeks, the inverse-vol book knows tomorrow's risk regime and every
number it emits is fiction), a not-yet-listed symbol silently sized on a partial window is survivorship
bias, and a hole on one asset's calendar counted as a flat return understates its vol and inflates its
weight. Each is pinned here before the screen is allowed to produce a number, alongside the same time-prefix
causality property the other B1 screens carry.
"""

import numpy as np
import pandas as pd
import pytest

from trainer import riskparity, xsection


def _frame(dates, prices):
    return pd.DataFrame({"timestamp_close": pd.to_datetime(dates), "price": prices})


def _matrix(**symbols):
    return xsection.align_prices({k: _frame(*v) for k, v in symbols.items()})


# --- the inverse-vol weighting rule --------------------------------------------------------------------


def test_inverse_vol_gives_the_low_vol_symbol_more_weight_and_a_late_lister_none():
    # LO drifts smoothly (small realized vol); HI whipsaws (large realized vol); risk parity must hand LO the
    # larger slice. LATE lists a single bar from the end with no formation window, so it cannot be risk-sized
    # at all and its weight stays exactly 0 rather than being waved in on a partial estimate.
    dates = pd.date_range("2022-01-01", periods=6, freq="D")
    m = _matrix(
        LO=(dates, [100.0, 101.0, 102.0, 103.0, 104.0, 105.0]),
        HI=(dates, [100.0, 130.0, 95.0, 140.0, 88.0, 135.0]),
        LATE=(dates[5:], [50.0]),
    )
    w = riskparity.build_weights(m, vol_window=3, rebalance_days=1, weighting="inverse_vol")
    last = w.iloc[-1]
    assert last["LO"] > last["HI"] > 0.0  # low vol gets MORE, both are actually held
    assert last["LATE"] == 0.0  # a name with no formation window is never sized
    assert abs(last.sum() - 1.0) < 1e-12  # fully invested over the tradeable set


def test_inverse_vol_weight_is_the_normalized_reciprocal_of_realized_vol():
    # Pin the actual arithmetic, not just the ordering: the two held names split the book in proportion to
    # 1/v, so their weight RATIO must equal the inverse of their vol ratio. The last row's book was decided
    # one bar EARLIER (rebalance_days=1 -> row t carries the weight sized at t-1), so it is measured against
    # the SECOND-TO-LAST bar's vol — which also pins the one-bar causal lag rather than papering over it.
    dates = pd.date_range("2022-01-01", periods=6, freq="D")
    m = _matrix(
        LO=(dates, [100.0, 101.0, 102.0, 103.0, 104.0, 105.0]),
        HI=(dates, [100.0, 130.0, 95.0, 140.0, 88.0, 135.0]),
    )
    vol = riskparity.trailing_vol(m, 3)
    w = riskparity.build_weights(m, vol_window=3, rebalance_days=1, weighting="inverse_vol").iloc[-1]
    v = vol.iloc[-2]  # the bar the last row's weights were decided on
    assert w["LO"] / w["HI"] == pytest.approx(v["HI"] / v["LO"])


def test_equal_weight_splits_the_tradeable_set_evenly_ignoring_vol():
    dates = pd.date_range("2022-01-01", periods=6, freq="D")
    m = _matrix(
        LO=(dates, [100.0, 101.0, 102.0, 103.0, 104.0, 105.0]),
        HI=(dates, [100.0, 130.0, 95.0, 140.0, 88.0, 135.0]),
    )
    last = riskparity.build_weights(m, vol_window=3, rebalance_days=1, weighting="equal_weight").iloc[-1]
    assert last["LO"] == pytest.approx(0.5) and last["HI"] == pytest.approx(0.5)


# --- warm-up / survivorship ----------------------------------------------------------------------------


def test_a_not_yet_tradeable_symbol_is_weight_zero_never_nan():
    # C lists three bars from the end. Until it has a full vol_window of prior bars it is invisible to the
    # book; its weight must be a hard 0, and no bar anywhere may be NaN (a NaN weight silently poisons the
    # whole equity curve through the backtest's sum).
    dates = pd.date_range("2022-01-01", periods=8, freq="D")
    m = _matrix(
        A=(dates, [100.0, 101, 103, 102, 104, 106, 105, 107]),
        B=(dates, [100.0, 99, 98, 100, 101, 99, 98, 100]),
        C=(dates[5:], [10.0, 12.0, 11.0]),
    )
    w = riskparity.build_weights(m, vol_window=3, rebalance_days=1, weighting="inverse_vol")
    assert (w["C"] == 0.0).all()  # never sized inside this span
    assert not w.isna().any().any()  # warm-up is a HOLE, never a NaN


def test_the_book_stays_fully_invested_over_whatever_is_tradeable():
    # Once at least one name is tradeable, the long-only book is fully invested across the tradeable set
    # (weights sum to 1), and it is 0 before any name has a formation window.
    dates = pd.date_range("2022-01-01", periods=7, freq="D")
    m = _matrix(
        A=(dates, [100.0, 101, 103, 102, 104, 106, 105]),
        B=(dates, [100.0, 98, 101, 99, 102, 100, 103]),
    )
    w = riskparity.build_weights(m, vol_window=3, rebalance_days=1, weighting="inverse_vol")
    sums = w.sum(axis=1)
    assert (sums.iloc[:4] == 0.0).all()  # A/B tradeable only from bar 3 -> book fills at bar 4
    assert (abs(sums.iloc[4:] - 1.0) < 1e-12).all()


# --- the causality contract ----------------------------------------------------------------------------


def test_inverse_vol_weights_are_time_prefix_causal():
    """The gold standard: corrupt every bar from `cut` onward, rebuild, and the book held into `cut` and
    everything before it must be byte-identical.

    Three details are what make this bite, and getting any of them wrong turns it into a formality a
    lookahead-ing implementation sails through:

    * The shock is a DIFFERENT factor per symbol. A single uniform factor would still perturb each vol, but a
      per-symbol shock guarantees the RELATIVE inverse-vol weights move rather than relying on a coincidence.
    * The comparison includes row `cut` itself. Row t is the book HELD INTO t, decided at t-1 from vols
      observed through t-1, so it too must survive a corruption that starts AT t; stopping one row short
      leaves exactly one bar of slack, which is the shape a one-bar vol lookahead actually takes.
    * Every cut is shown to MOVE the book somewhere after the boundary. An assertion that nothing changed is
      worthless without evidence that the corruption could have changed something.
    """
    dates = pd.date_range("2022-01-01", periods=20, freq="D")
    rng = np.random.default_rng(0)
    base = {s: (dates, list(100 + np.cumsum(rng.normal(0, 1, 20)))) for s in ("A", "B", "C", "D")}
    m = _matrix(**base)
    kwargs = dict(vol_window=3, rebalance_days=2, weighting="inverse_vol")
    clean = riskparity.build_weights(m, **kwargs)
    for cut in range(5, 18):
        dirty_m = m.copy()
        dirty_m.iloc[cut:] = dirty_m.iloc[cut:] * [5.0, 0.2, 1.0, 3.0]
        dirty = riskparity.build_weights(dirty_m, **kwargs)
        pd.testing.assert_frame_equal(clean.iloc[: cut + 1], dirty.iloc[: cut + 1])
        assert not clean.iloc[cut + 1 :].equals(dirty.iloc[cut + 1 :])


def test_equal_weight_ignores_price_values_entirely():
    # The reference arm depends only on WHICH names are tradeable, never on their prices, so any value
    # corruption anywhere leaves the whole book byte-identical. That is causality in its strongest form for
    # this arm — it cannot peek at a return it never reads — and it also pins that equal_weight is genuinely
    # a different rule from inverse_vol rather than a mislabeled copy that secretly consults vol.
    dates = pd.date_range("2022-01-01", periods=12, freq="D")
    rng = np.random.default_rng(1)
    base = {s: (dates, list(100 + np.cumsum(rng.normal(0, 1, 12)))) for s in ("A", "B", "C")}
    m = _matrix(**base)
    kwargs = dict(vol_window=3, rebalance_days=2, weighting="equal_weight")
    clean = riskparity.build_weights(m, **kwargs)
    dirty_m = m.copy()
    dirty_m.iloc[6:] = dirty_m.iloc[6:] * [7.0, 0.1, 4.0]
    dirty = riskparity.build_weights(dirty_m, **kwargs)
    pd.testing.assert_frame_equal(clean, dirty)


@pytest.mark.parametrize("weighting", riskparity.WEIGHTINGS)
def test_a_signal_is_never_traded_on_its_own_bar(weighting):
    # Shock ONE bar hard enough to dominate every vol, then read the book either side of it: the weight held
    # INTO the shocked bar was decided a bar earlier and must not have seen it, while the NEXT row must move
    # (for inverse_vol — the shock reshapes the vols) — that second half is what stops this collapsing into
    # "row zero is flat", which is true of any book because the loop writes the opening row before it has
    # decided anything. equal_weight rows never move on a value shock, so it only carries the first half.
    dates = pd.date_range("2022-01-01", periods=8, freq="D")
    base = {
        "A": [100.0, 101, 102, 103, 104, 105, 106, 107],
        "B": [100.0, 99, 98, 97, 96, 95, 94, 93],
        "C": [100.0, 100.5, 101, 101.5, 102, 102.5, 103, 103.5],
    }
    shocked = {s: list(p) for s, p in base.items()}
    shocked["B"][4] = 400.0
    kwargs = dict(vol_window=2, rebalance_days=1, weighting=weighting)
    clean = riskparity.build_weights(_matrix(**{s: (dates, p) for s, p in base.items()}), **kwargs)
    dirty = riskparity.build_weights(_matrix(**{s: (dates, p) for s, p in shocked.items()}), **kwargs)
    assert clean.iloc[0].abs().sum() == 0.0  # the opening row is flat
    pd.testing.assert_frame_equal(clean.iloc[:5], dirty.iloc[:5])  # the shock at bar 4 is not traded at bar 4
    if weighting == "inverse_vol":
        assert not clean.iloc[5].equals(dirty.iloc[5])  # but it reshapes the book held into bar 5


# --- costs + turnover control --------------------------------------------------------------------------


def test_fee_is_charged_on_turnover_both_ways():
    # Run the real book through the reused backtest with and without a fee: entering (and re-weighting) the
    # positions costs real money, so the fee-paying curve must end strictly below the free one.
    dates = pd.date_range("2022-01-01", periods=6, freq="D")
    m = _matrix(
        A=(dates, [100.0, 101, 103, 102, 104, 106]),
        B=(dates, [100.0, 130, 95, 140, 88, 135]),
    )
    w = riskparity.build_weights(m, vol_window=3, rebalance_days=1, weighting="inverse_vol")
    eq_free = xsection.backtest(m, w, fee=0.0)
    eq_paid = xsection.backtest(m, w, fee=0.01)
    assert eq_paid.iloc[-1] < eq_free.iloc[-1]


def test_rebalance_days_strictly_reduces_turnover():
    # On a series whose relative vols drift every bar, inverse_vol would re-weight every bar at
    # rebalance_days=1; holding the book for several bars must strictly cut the summed turnover, which is the
    # whole point of the cost control (turnover is the term that killed earlier arms of this campaign).
    dates = pd.date_range("2022-01-01", periods=16, freq="D")
    steps = np.arange(16)
    m = _matrix(
        A=(dates, list(100.0 + 0.3 * steps)),  # calm, near-constant small returns
        B=(dates, list(100.0 * (1.0 + 0.02 * steps * np.sin(steps)))),  # vol grows with the step index
        C=(dates, list(100.0 + 2.0 * np.cos(steps / 2.0))),  # a third, differently-phased path
    )
    fast = riskparity.build_weights(m, vol_window=3, rebalance_days=1, weighting="inverse_vol")
    slow = riskparity.build_weights(m, vol_window=3, rebalance_days=6, weighting="inverse_vol")
    turn_fast = float(fast.diff().abs().sum().sum())
    turn_slow = float(slow.diff().abs().sum().sum())
    assert turn_fast > 0.0
    assert turn_slow < turn_fast


# --- consistency with the basket benchmark -------------------------------------------------------------


def test_equal_weight_book_reproduces_the_basket_step_for_step():
    # A consistency pin: an equal-weight, every-bar-rebalanced, fee-free book IS the equal-weight basket
    # benchmark. Once both are past warm-up and hold the same constant tradeable set, their per-step returns
    # must match exactly — if this ever drifts, one of the two accounting paths has a bug. (They differ only
    # by a single one-bar entry lag right at the start, so the comparison begins at the first traded bar.)
    dates = pd.date_range("2022-01-01", periods=10, freq="D")
    m = _matrix(
        A=(dates, [100.0, 102, 101, 104, 103, 106, 105, 108, 107, 110]),
        B=(dates, [100.0, 99, 101, 100, 103, 101, 104, 102, 105, 103]),
        C=(dates, [100.0, 101, 100, 102, 101, 103, 102, 104, 103, 105]),
    )
    w = riskparity.build_weights(m, vol_window=2, rebalance_days=1, weighting="equal_weight")
    book = xsection.backtest(m, w, fee=0.0)
    basket = xsection.basket_curve(m, xsection.tradeable_mask(m, 2), fee=0.0)
    first_traded = int(np.argmax(w.sum(axis=1).values > 0.0))
    book_steps = (book / book.shift(1)).iloc[first_traded + 1 :]
    basket_steps = (basket / basket.shift(1)).iloc[first_traded + 1 :]
    assert np.allclose(book_steps.values, basket_steps.values)


# --- guardrails ----------------------------------------------------------------------------------------


def test_an_unrecognised_weighting_is_rejected_rather_than_silently_sized():
    # A typo'd lever that fell back to a default would file (say) an equal_weight book under an inverse_vol
    # label — the failure mode that corrupts the evidence trail rather than merely producing a bad number.
    dates = pd.date_range("2022-01-01", periods=5, freq="D")
    m = _matrix(A=(dates, [100.0, 110, 120, 130, 140]), B=(dates, [100.0, 90, 80, 70, 60]))
    with pytest.raises(ValueError):
        riskparity.build_weights(m, vol_window=2, rebalance_days=1, weighting="min_variance")


def test_a_vol_window_below_two_is_refused_rather_than_silently_zeroing_vol():
    # A stdev needs at least two returns. A window of 1 makes every realized vol 0, which inverts to an
    # infinite weight and, on the fallback, silently turns inverse_vol INTO equal_weight — an equal-weight
    # result filed under a risk-parity label. Refuse it the same way a non-positive lookback is refused.
    dates = pd.date_range("2022-01-01", periods=5, freq="D")
    m = _matrix(A=(dates, [100.0, 110, 120, 130, 140]), B=(dates, [100.0, 90, 80, 70, 60]))
    for bad in (1, 0, -2):
        with pytest.raises(ValueError):
            riskparity.build_weights(m, vol_window=bad, rebalance_days=1, weighting="inverse_vol")


def test_an_empty_universe_and_an_all_flat_book_stay_finite():
    empty = xsection.align_prices({})
    assert riskparity.build_weights(empty, vol_window=3, rebalance_days=1).empty
    dates = pd.date_range("2022-01-01", periods=3, freq="D")
    m = _matrix(A=(dates, [100.0, 101, 102]))
    flat = pd.DataFrame(0.0, index=m.index, columns=m.columns)
    assert xsection.backtest(m, flat, fee=0.001).iloc[-1] == 1.0


# --- the cell as run() assembles it --------------------------------------------------------------------
#
# build_weights' guards protect a direct caller; everything a persisted cell depends on lives one level up in
# run(): whether the weighting lever reaches the book, whether the FULL metric vocabulary is emitted so every
# gate/lens reads the cell unchanged, and whether the fully-invested book actually reports ~1 exposure.

_RUN_CFG = {
    "universe": "cross-class",
    "walk_forward_window": "stk-2024",
    "vol_window": 20,
    "rebalance_days": 5,
    "transaction_fee": 0.0,
}
_RUN_TEST_START = pd.Timestamp("2024-01-01")

_METRIC_VOCAB = (
    "total_return_pct", "hold_return_pct", "return_vs_hold_pct",
    "oos_sharpe", "oos_n_obs", "oos_ret_skew", "oos_ret_kurt",
    "hold_sharpe", "sharpe_vs_hold", "hold_max_drawdown_pct", "drawdown_vs_hold_pct",
    "max_drawdown_pct", "beta", "up_capture", "down_capture",
    "realized_cost_bps", "mean_exposure", "universe_size", "bars",
)


def _spanning_frames():
    """Three heterogeneous-vol symbols spanning the stk-2024 train tail and the whole test window, on one
    shared daily clock so the tradeable set is constant and exposure is clean. Deterministic (seeded)."""
    dates = pd.date_range("2023-01-01", periods=730, freq="D")
    rng = np.random.default_rng(7)
    scales = {"GOLD": 0.4, "TLT": 0.8, "SPY": 1.6}
    return {
        s: _frame(dates, list(100.0 * np.exp(np.cumsum(rng.normal(0.0002, sc / 100.0, 730)))))
        for s, sc in scales.items()
    }


def _stub_loader(monkeypatch, frames):
    calls = []

    def _fake(symbols, pairs):
        calls.append((tuple(symbols), len(pairs)))
        return dict(frames)

    monkeypatch.setattr(riskparity, "_load_universe", _fake)
    return calls


def test_a_cell_emits_the_full_metric_vocabulary_and_unit_exposure(monkeypatch):
    _stub_loader(monkeypatch, _spanning_frames())
    summary = riskparity.run(dict(_RUN_CFG))
    metrics = summary["metrics"]
    for key in _METRIC_VOCAB:
        assert key in metrics, f"missing metric {key}"
    assert summary["objective"] == metrics["oos_sharpe"]
    assert metrics["universe_size"] == 3
    assert 0.9 < metrics["mean_exposure"] <= 1.0 + 1e-9  # a fully-invested book
    assert "provenance" in summary


def test_inverse_vol_and_equal_weight_are_actually_different_cells(monkeypatch):
    _stub_loader(monkeypatch, _spanning_frames())
    inv = riskparity.run({**_RUN_CFG, "weighting": "inverse_vol"})
    eq = riskparity.run({**_RUN_CFG, "weighting": "equal_weight"})
    assert inv["metrics"]["total_return_pct"] != eq["metrics"]["total_return_pct"]
    assert inv["config"]["weighting"] == "inverse_vol"  # the record says which arm produced it


def test_a_typod_lever_kills_the_cell_before_any_disk_is_read(monkeypatch):
    calls = _stub_loader(monkeypatch, _spanning_frames())
    for bad in ({"weighting": "min_variance"}, {"universe": "everything"}):
        with pytest.raises(SystemExit):
            riskparity.run({**_RUN_CFG, **bad})
    assert calls == []  # a bad lever fails the same way a bad universe does — before any work
