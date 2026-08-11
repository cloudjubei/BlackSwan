"""Direct unit tests for the calendar / seasonal anomaly core (the published-anomaly battery).

Calendar anomalies (turn-of-the-month — Ariel 1987 / Lakonishok-Smidt 1988; sell-in-May / Halloween —
Bouman-Jacobsen 2002; the Monday effect — French 1980) are deterministic date rules, so their leakage
surface is unusual: the position is a pure function of the CALENDAR and must be completely independent of
price — a rule that ever reads a price is either a bug or a disguised different strategy. That price
independence is the guard pinned hardest here, alongside the exact window definitions and the mirror control.
"""

import numpy as np
import pandas as pd
import pytest

from trainer import seasonal


def _spy(dates, prices=None):
    idx = pd.to_datetime(dates)
    prices = prices if prices is not None else list(100 + np.arange(len(idx), dtype=float))
    return {"SPY": pd.DataFrame({"timestamp_close": idx, "price": prices})}


def _prices(dates, prices=None):
    return seasonal.align_prices(_spy(dates, prices))


# --- the calendar rules -------------------------------------------------------------------------------


def test_turn_of_month_is_the_last_trading_day_plus_the_first_n():
    # Two months of business days. TOM window = last trading day of the month + the first `tom_days` of the
    # next; everything else is the complement (-1).
    idx = pd.bdate_range("2022-01-03", "2022-02-28")
    pos = seasonal.calendar_position(idx, "turn_of_month", tom_days=3)
    jan = idx[idx.month == 1]
    feb = idx[idx.month == 2]
    assert pos.loc[jan[-1]] == 1.0  # last trading day of January
    assert pos.loc[feb[0]] == 1.0 and pos.loc[feb[1]] == 1.0 and pos.loc[feb[2]] == 1.0  # first 3 of Feb
    assert pos.loc[feb[3]] == -1.0  # 4th trading day of Feb is outside the window
    assert pos.loc[jan[10]] == -1.0  # mid-January is outside


def test_sell_in_may_is_long_november_to_april_short_may_to_october():
    idx = pd.to_datetime(["2022-01-14", "2022-04-14", "2022-05-13", "2022-07-15", "2022-10-14", "2022-11-15"])
    pos = seasonal.calendar_position(idx, "sell_in_may")
    assert pos.loc[pd.Timestamp("2022-01-14")] == 1.0  # Jan (Nov-Apr window)
    assert pos.loc[pd.Timestamp("2022-04-14")] == 1.0  # Apr
    assert pos.loc[pd.Timestamp("2022-05-13")] == -1.0  # May (out)
    assert pos.loc[pd.Timestamp("2022-07-15")] == -1.0  # Jul (out)
    assert pos.loc[pd.Timestamp("2022-10-14")] == -1.0  # Oct (out)
    assert pos.loc[pd.Timestamp("2022-11-15")] == 1.0  # Nov (in)


def test_day_of_week_shorts_monday_and_longs_the_rest():
    # 2022-01-03 is a Monday.
    idx = pd.bdate_range("2022-01-03", periods=5)
    pos = seasonal.calendar_position(idx, "day_of_week")
    assert pos.iloc[0] == -1.0  # Monday
    assert (pos.iloc[1:] == 1.0).all()  # Tue-Fri


def test_seasonal_inverse_is_the_exact_negation():
    idx = pd.bdate_range("2022-01-03", "2022-03-31")
    m = _prices(idx)
    for rule in seasonal.RULES:
        fwd = seasonal.build_weights(m, rule=rule, signal="seasonal")
        inv = seasonal.build_weights(m, rule=rule, signal="seasonal_inverse")
        pd.testing.assert_frame_equal(inv, -fwd)


def test_an_unknown_rule_or_signal_is_refused():
    idx = pd.bdate_range("2022-01-03", periods=10)
    m = _prices(idx)
    with pytest.raises(ValueError):
        seasonal.build_weights(m, rule="triple_witching", signal="seasonal")
    with pytest.raises(ValueError):
        seasonal.build_weights(m, rule="sell_in_may", signal="fade")


# --- the leakage guard unique to calendar rules: PRICE INDEPENDENCE -----------------------------------


@pytest.mark.parametrize("rule", seasonal.RULES)
@pytest.mark.parametrize("signal", seasonal.SIGNALS)
def test_the_book_is_a_pure_function_of_the_calendar_never_the_price(rule, signal):
    # A calendar position must depend ONLY on the dates. Corrupting every price (even wildly) must leave the
    # book byte-identical; if it moves, the rule is secretly reading price and is not the anomaly it claims.
    idx = pd.bdate_range("2021-06-01", "2022-06-30")
    rng = np.random.default_rng(0)
    clean = _prices(idx, list(100 + np.cumsum(rng.normal(0, 1, len(idx)))))
    dirty = clean.copy()
    dirty["SPY"] = dirty["SPY"] * (5.0 + rng.normal(0, 3, len(idx)))  # arbitrary per-bar price shock
    wc = seasonal.build_weights(clean, rule=rule, signal=signal)
    wd = seasonal.build_weights(dirty, rule=rule, signal=signal)
    pd.testing.assert_frame_equal(wc, wd)
    assert wc["SPY"].abs().sum() > 0 and wc["SPY"].nunique() > 1  # non-vacuous: the book actually moves


def test_the_book_spreads_equally_and_is_time_exposure_neutral():
    # 20 business days = 16 non-Monday (frac_in 0.8) + 4 Monday (frac_out 0.2). Each asset carries the same
    # sign; the long leg is scaled +0.5/0.8 and the short leg -0.5/0.2, so the TIME-AVERAGE net exposure is
    # zero — the property that isolates the seasonal differential from market drift.
    idx = pd.bdate_range("2022-01-03", periods=20)
    frames = {s: pd.DataFrame({"timestamp_close": idx, "price": list(100 + np.arange(20.0))}) for s in ("SPY", "TLT", "IEF")}
    m = seasonal.align_prices(frames)
    w = seasonal.build_weights(m, rule="day_of_week", signal="seasonal")
    tue = w.iloc[1]  # a Tuesday (non-Monday, long leg): +0.5/0.8 spread across 3 assets
    mon = w.iloc[0]  # a Monday (short leg): -0.5/0.2 spread across 3 assets
    assert all(abs(tue[s] - (0.5 / 0.8) / 3) < 1e-12 for s in ("SPY", "TLT", "IEF"))
    assert all(abs(mon[s] - (-0.5 / 0.2) / 3) < 1e-12 for s in ("SPY", "TLT", "IEF"))
    assert abs(w.sum(axis=1).mean()) < 1e-12  # time-averaged net exposure is ZERO


# --- costs + the cell wiring --------------------------------------------------------------------------


def test_turnover_is_charged_when_the_window_flips():
    idx = pd.bdate_range("2022-01-03", periods=10)
    m = _prices(idx, [100.0] * 10)  # flat prices: only cost can move equity
    w = seasonal.build_weights(m, rule="day_of_week", signal="seasonal")
    eq_free = seasonal.backtest(m, w, fee=0.0)
    eq_paid = seasonal.backtest(m, w, fee=0.02)
    assert eq_free.iloc[-1] == pytest.approx(1.0)  # flat prices, no fee -> flat equity
    assert eq_paid.iloc[-1] < 1.0  # the Monday->Tuesday flip flips the book and costs money


_RUN_CFG = {"universe": "spy", "walk_forward_window": "2024", "rule": "turn_of_month", "tom_days": 3, "transaction_fee": 0.0}


def _osc_frames():
    idx = pd.bdate_range("2023-06-01", "2024-12-31")
    steps = np.arange(len(idx))
    return {"SPY": pd.DataFrame({"timestamp_close": idx, "price": list(100 * (1 + 0.1 * np.sin(steps / 15.0)))})}


def _stub_loader(monkeypatch, frames):
    calls = []

    def _fake(symbols, pairs):
        calls.append((tuple(symbols), len(pairs)))
        return dict(frames)

    monkeypatch.setattr(seasonal, "_load_universe", _fake)
    return calls


def test_a_reversal_of_the_arm_changes_the_cell(monkeypatch):
    _stub_loader(monkeypatch, _osc_frames())
    fwd = seasonal.run({**_RUN_CFG, "signal": "seasonal"})
    inv = seasonal.run({**_RUN_CFG, "signal": "seasonal_inverse"})
    assert abs(fwd["objective"] - inv["objective"]) > 0.1
    assert fwd["config"]["signal"] == "seasonal"


def test_a_typod_rule_kills_the_cell_before_any_data_is_read(monkeypatch):
    calls = _stub_loader(monkeypatch, _osc_frames())
    with pytest.raises(SystemExit) as raised:
        seasonal.run({**_RUN_CFG, "rule": "triple_witching"})
    assert "triple_witching" in str(raised.value)
    assert calls == []


def test_the_cell_reports_the_gate_metric_and_window(monkeypatch):
    _stub_loader(monkeypatch, _osc_frames())
    out = seasonal.run(dict(_RUN_CFG))
    assert "oos_sharpe" in out["metrics"]
    assert out["walk_forward_window"] == "2024"
    assert out["metrics"]["bars"] > 0
