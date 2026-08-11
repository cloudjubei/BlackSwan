"""Direct unit tests for the pre-FOMC announcement drift probe (Lucca-Moench 2015).

Lucca & Moench (2015) find US equity returns concentrate in the ~24h before scheduled FOMC announcements;
Kurov et al. (2021) find it disappeared. We test it as an EVENT-CALENDAR spread — long the pre-FOMC day,
short the complement, exposure-balanced so the differential is isolated from market drift — on SPY (its home)
and BTC (the 24/7 crypto window, never cost+DSR-tested). Like the seasonal probe, the book is a PURE FUNCTION
of the (published, scheduled) FOMC calendar and never of price; that price-independence is the guard pinned
hardest here, alongside the exact pre-window definition and the mirror control.
"""

import numpy as np
import pandas as pd
import pytest

from trainer import fomc


def _frame(dates, prices):
    return pd.DataFrame({"timestamp_close": pd.to_datetime(dates), "price": prices})


def _prices(dates, prices=None):
    idx = pd.to_datetime(dates)
    prices = prices if prices is not None else list(100.0 + np.arange(len(idx)))
    return fomc.align_prices({"SPY": _frame(idx, prices)})


# --- the event-window rule ----------------------------------------------------------------------------


def test_pre_fomc_days_are_the_n_trading_days_before_an_announcement():
    idx = pd.bdate_range("2022-01-03", "2022-02-11")
    # a fake announcement on 2022-01-26 (a Wednesday)
    pos = fomc.fomc_position(idx, [pd.Timestamp("2022-01-26")], pre_days=1)
    assert pos.loc[pd.Timestamp("2022-01-25")] == 1.0  # the trading day before -> pre-FOMC
    assert pos.loc[pd.Timestamp("2022-01-26")] == -1.0  # announcement day itself is NOT the pre-window
    assert pos.loc[pd.Timestamp("2022-01-24")] == -1.0  # two days before, with pre_days=1
    assert pos.loc[pd.Timestamp("2022-01-27")] == -1.0  # the day after


def test_pre_days_two_covers_the_two_trading_days_before():
    idx = pd.bdate_range("2022-01-03", "2022-02-11")
    pos = fomc.fomc_position(idx, [pd.Timestamp("2022-01-26")], pre_days=2)
    assert pos.loc[pd.Timestamp("2022-01-25")] == 1.0
    assert pos.loc[pd.Timestamp("2022-01-24")] == 1.0
    assert pos.loc[pd.Timestamp("2022-01-21")] == -1.0  # three before -> out


def test_an_announcement_off_the_calendar_maps_to_the_prior_trading_day():
    # FOMC on a Wednesday; the pre-FOMC day is the Tuesday. If the announcement date itself is a holiday not in
    # the index, the pre-window is still the last trading day strictly before it.
    idx = pd.bdate_range("2022-06-13", "2022-06-24")  # 06-15 is the announcement (present)
    pos = fomc.fomc_position(idx, [pd.Timestamp("2022-06-15")], pre_days=1)
    assert pos.loc[pd.Timestamp("2022-06-14")] == 1.0


def test_the_bundled_calendar_is_scheduled_meetings_only_and_sane():
    ds = fomc.FOMC_DATES
    assert all(isinstance(d, str) and len(d) == 10 for d in ds)
    yrs = [int(d[:4]) for d in ds]
    assert min(yrs) <= 2015 and max(yrs) >= 2024
    # ~8 scheduled meetings a year, no emergency clutter (2020 trimmed): sane per-year counts
    from collections import Counter
    per = Counter(yrs)
    assert all(6 <= per[y] <= 9 for y in range(2015, 2025))


# --- price independence (the calendar-probe leakage guard) --------------------------------------------


@pytest.mark.parametrize("signal", fomc.SIGNALS)
def test_the_book_is_a_pure_function_of_the_fomc_calendar_never_the_price(signal):
    idx = pd.bdate_range("2019-01-02", "2021-12-31")
    rng = np.random.default_rng(0)
    clean = _prices(idx, list(100 + np.cumsum(rng.normal(0, 1, len(idx)))))
    dirty = clean.copy()
    dirty["SPY"] = dirty["SPY"] * (3.0 + rng.normal(0, 2, len(idx)))
    wc = fomc.build_weights(clean, signal=signal, pre_days=1)
    wd = fomc.build_weights(dirty, signal=signal, pre_days=1)
    pd.testing.assert_frame_equal(wc, wd)
    assert wc["SPY"].abs().sum() > 0 and wc["SPY"].nunique() > 1  # non-vacuous


def test_inverse_is_the_exact_negation_and_the_book_is_time_neutral():
    idx = pd.bdate_range("2018-01-02", "2020-12-31")
    m = _prices(idx)
    fwd = fomc.build_weights(m, signal="drift", pre_days=1)
    inv = fomc.build_weights(m, signal="drift_inverse", pre_days=1)
    pd.testing.assert_frame_equal(inv, -fwd)
    assert abs(fwd.sum(axis=1).mean()) < 1e-12  # exposure-balanced: time-averaged net exposure is zero


def test_an_unknown_signal_is_refused():
    m = _prices(pd.bdate_range("2019-01-02", periods=60))
    with pytest.raises(ValueError):
        fomc.build_weights(m, signal="fade", pre_days=1)


# --- the cell wiring ----------------------------------------------------------------------------------

_RUN_CFG = {"universe": "spy", "walk_forward_window": "2022", "pre_days": 1, "transaction_fee": 0.0}


def _stub_loader(monkeypatch, frames):
    calls = []

    def _fake(symbols, pairs):
        calls.append((tuple(symbols), len(pairs)))
        return {s: frames[s] for s in symbols if s in frames}

    monkeypatch.setattr(fomc, "_load_universe", _fake)
    return calls


def _spy_frames():
    idx = pd.bdate_range("2018-01-02", "2022-12-31")
    rng = np.random.default_rng(5)
    return {"SPY": _frame(idx, list(100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, len(idx))))))}


def test_a_reversal_of_the_arm_changes_the_cell(monkeypatch):
    _stub_loader(monkeypatch, _spy_frames())
    fwd = fomc.run({**_RUN_CFG, "signal": "drift"})
    inv = fomc.run({**_RUN_CFG, "signal": "drift_inverse"})
    assert abs(fwd["objective"] - inv["objective"]) > 0.05
    assert fwd["config"]["signal"] == "drift"


def test_a_typod_signal_kills_the_cell_before_any_data_is_read(monkeypatch):
    calls = _stub_loader(monkeypatch, _spy_frames())
    with pytest.raises(SystemExit) as raised:
        fomc.run({**_RUN_CFG, "signal": "fade"})
    assert "fade" in str(raised.value)
    assert calls == []


def test_the_cell_reports_the_gate_metric_and_window(monkeypatch):
    _stub_loader(monkeypatch, _spy_frames())
    out = fomc.run(dict(_RUN_CFG))
    assert "oos_sharpe" in out["metrics"]
    assert out["walk_forward_window"] == "2022"
    assert out["metrics"]["bars"] > 0
