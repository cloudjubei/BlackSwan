"""Direct unit tests for the overnight-vs-intraday return decomposition (Lou-Polk-Skouras 2019).

The stylized fact: (nearly) all of the equity premium is earned OVERNIGHT (close→open); the intraday
(open→close) component is flat-to-negative. We test the tradeable long-overnight / short-intraday SPREAD. The
failure mode unique to this decomposition is MIS-ALIGNMENT — the overnight return must pair open[t] with
close[t-1] (not close[t]); a one-bar slip fabricates the whole effect. That alignment, plus the mirror and the
(heavy, 2-flips-per-day) cost, are what these tests pin.
"""

import numpy as np
import pandas as pd
import pytest

from trainer import overnight as ov


def _frame(dates, opens, closes):
    return pd.DataFrame({"timestamp_close": pd.to_datetime(dates), "price_open": opens, "price": closes})


def test_overnight_pairs_open_with_the_prior_close_and_intraday_open_with_close():
    dates = pd.to_datetime(["2022-01-03", "2022-01-04", "2022-01-05"])
    # close: 100, 110, 121 ; open: (na), 105, 115.5
    frames = {"SPY": _frame(dates, [100.0, 105.0, 115.5], [100.0, 110.0, 121.0])}
    on = ov.overnight_returns(frames)["SPY"]
    intra = ov.intraday_returns(frames)["SPY"]
    # day 2: overnight = open2/close1 - 1 = 105/100 - 1 = 0.05 ; intraday = close2/open2 - 1 = 110/105 - 1
    assert abs(on.iloc[1] - 0.05) < 1e-12
    assert abs(intra.iloc[1] - (110.0 / 105.0 - 1)) < 1e-12
    # day 3: overnight = 115.5/110 - 1 = 0.05 ; intraday = 121/115.5 - 1
    assert abs(on.iloc[2] - 0.05) < 1e-12
    assert abs(intra.iloc[2] - (121.0 / 115.5 - 1)) < 1e-12
    # overnight * intraday compounding recovers the close-to-close return
    assert abs((1 + on.iloc[1]) * (1 + intra.iloc[1]) - 110.0 / 100.0) < 1e-12


def test_the_spread_is_overnight_minus_intraday_and_inverse_negates_it():
    dates = pd.date_range("2022-01-03", periods=40, freq="B")
    rng = np.random.default_rng(0)
    o = list(100 + np.cumsum(rng.normal(0, 1, 40)))
    cl = [x * (1 + rng.normal(0, 0.005)) for x in o]
    frames = {"SPY": _frame(dates, o, cl)}
    fwd = ov.spread_returns(frames, signal="overnight")
    inv = ov.spread_returns(frames, signal="overnight_inverse")
    on = ov.overnight_returns(frames)["SPY"]
    intra = ov.intraday_returns(frames)["SPY"]
    pd.testing.assert_series_equal(fwd, (on - intra).dropna(), check_names=False)
    pd.testing.assert_series_equal(inv, -fwd, check_names=False)


def test_an_unknown_signal_is_refused():
    dates = pd.date_range("2022-01-03", periods=5, freq="B")
    frames = {"SPY": _frame(dates, [100.0] * 5, [100.0] * 5)}
    with pytest.raises(ValueError):
        ov.spread_returns(frames, signal="carry")


def test_a_misaligned_overnight_return_would_change_the_spread():
    # Guard the alignment: pairing open[t] with close[t] (intraday) instead of close[t-1] (overnight) collapses
    # the spread to ~0. A correct overnight uses the PRIOR close, so corrupting only close[t-1] must move the
    # overnight return but a hypothetical same-bar version would not.
    dates = pd.date_range("2022-01-03", periods=6, freq="B")
    frames = {"SPY": _frame(dates, [100, 102, 104, 106, 108, 110.0], [101, 103, 105, 107, 109, 111.0])}
    on = ov.overnight_returns(frames)["SPY"]
    # overnight day2 = open2/close1 = 102/101 - 1 (uses PRIOR close 101, not close2 103)
    assert abs(on.iloc[1] - (102.0 / 101.0 - 1)) < 1e-12
    assert on.iloc[1] != (102.0 / 103.0 - 1)  # not paired with the same-day close


def test_cost_is_charged_on_the_daily_double_flip():
    dates = pd.date_range("2022-01-03", periods=10, freq="B")
    frames = {"SPY": _frame(dates, [100.0] * 10, [100.0] * 10)}  # flat -> only cost can move equity
    eq_free = ov.spread_equity(frames, signal="overnight", fee=0.0)
    eq_paid = ov.spread_equity(frames, signal="overnight", fee=0.001)
    assert abs(eq_free.iloc[-1] - 1.0) < 1e-9
    assert eq_paid.iloc[-1] < 1.0  # two flips per day cost real money


# --- the cell wiring ----------------------------------------------------------------------------------

_RUN_CFG = {"universe": "spy", "walk_forward_window": "2022", "transaction_fee": 0.0}


def _stub_loader(monkeypatch, frames):
    calls = []

    def _fake(symbols, pairs):
        calls.append((tuple(symbols), len(pairs)))
        return {s: frames[s] for s in symbols if s in frames}

    monkeypatch.setattr(ov, "_load_ohlc", _fake)
    return calls


def _spy_frames():
    dates = pd.date_range("2020-01-02", "2022-12-31", freq="B")
    rng = np.random.default_rng(3)
    cl = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.01, len(dates))))
    op = cl / (1 + rng.normal(0.0003, 0.006, len(dates)))  # open below close on average -> overnight premium
    return {"SPY": _frame(dates, list(op), list(cl))}


def test_a_reversal_of_the_arm_changes_the_cell(monkeypatch):
    _stub_loader(monkeypatch, _spy_frames())
    fwd = ov.run({**_RUN_CFG, "signal": "overnight"})
    inv = ov.run({**_RUN_CFG, "signal": "overnight_inverse"})
    assert abs(fwd["objective"] - inv["objective"]) > 0.1
    assert fwd["config"]["signal"] == "overnight"


def test_a_typod_signal_kills_the_cell_before_any_data_is_read(monkeypatch):
    calls = _stub_loader(monkeypatch, _spy_frames())
    with pytest.raises(SystemExit) as raised:
        ov.run({**_RUN_CFG, "signal": "carry"})
    assert "carry" in str(raised.value)
    assert calls == []


def test_the_cell_reports_the_gate_metric_and_window(monkeypatch):
    _stub_loader(monkeypatch, _spy_frames())
    out = ov.run(dict(_RUN_CFG))
    assert "oos_sharpe" in out["metrics"]
    assert out["walk_forward_window"] == "2022"
    assert out["metrics"]["bars"] > 0
