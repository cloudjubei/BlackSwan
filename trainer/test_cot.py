"""Direct gating tests for the COT probe — the POSITIONING/FLOW signal class (the one class the program had not
tested). CFTC managed-money net positioning (net = spec long - short, as a fraction of open interest) is an
extreme-of-past signal: a crowded net-long extreme is either FADED (cot_contrarian: crowded longs unwind) or
RIDDEN (cot_momentum). It reuses the shared past-only extreme core and the intraday backtest, so the surface new
here is the COT JOIN, which carries the one leakage trap: the COT measured on a Tuesday is PUBLISHED the
following Friday, so a bar dated D may use only a report whose (report date + release lag) is at/before D.

Guards pinned here:
* the join is RELEASE-LAGGED — a report dated D is invisible until D + release_lag_days;
* the extreme is judged past-only (delegated to signal_extremes) and the position applies to the NEXT bar;
* cot_contrarian FADES the crowd (top-tail net-long -> short), cot_momentum RIDES it (top-tail -> long).
"""

import json

import numpy as np
import pytest

from trainer import cot

_DAY = 86_400_000


def _days(n, start_ms=1_704_153_600_000):  # 2024-01-02 00:00 UTC (a Tuesday)
    return [start_ms + i * _DAY for i in range(n)]


# --- COT join: release-lagged, a report is invisible until report_date + lag -------------------------


def test_cot_values_are_release_lagged(tmp_path, monkeypatch):
    table = {"2024-01-02": {"net": 100.0, "oi": 1000.0}, "2024-01-09": {"net": -200.0, "oi": 1000.0}}
    (tmp_path / "GOLD.json").write_text(json.dumps(table))
    monkeypatch.setattr(cot, "COT_DIR", str(tmp_path))
    ts = _days(13)  # 2024-01-02 .. 2024-01-14
    vals = cot.cot_values("GOLD", ts, release_lag_days=4)
    assert vals[0] is None            # 01-02 report not released until 01-06
    assert vals[3] is None            # 01-05 still before release
    assert vals[4] == pytest.approx(0.1)   # 01-06 = 01-02 report + 4d -> net/oi = 100/1000
    assert vals[10] == pytest.approx(0.1)  # 01-12 still the 01-02 report (01-09's release is 01-13)
    assert vals[11] == pytest.approx(-0.2)  # 01-13 = 01-09 report + 4d -> -200/1000


def test_cot_values_missing_file_is_all_none(monkeypatch, tmp_path):
    monkeypatch.setattr(cot, "COT_DIR", str(tmp_path))
    assert cot.cot_values("NOPE", _days(3), release_lag_days=4) == [None, None, None]


# --- extreme signal: contrarian fades, momentum rides ------------------------------------------------


def test_contrarian_fades_and_momentum_rides_a_positioning_extreme():
    ts = _days(31)
    vals = [0.10 + (0.01 if i % 2 else -0.01) for i in range(30)] + [0.60]  # crowded-long extreme at the end
    con = cot.cot_sides(ts, vals, "cot_contrarian", cot_pct=0.90, min_history=10)
    mom = cot.cot_sides(ts, vals, "cot_momentum", cot_pct=0.90, min_history=10)
    assert con[-1] == -1.0   # fade the crowded long -> short
    assert mom[-1] == 1.0    # ride it -> long


def test_cot_sides_refuses_unknown_signal():
    with pytest.raises(SystemExit):
        cot.cot_sides(_days(3), [0.1, 0.2, 0.3], "cot_wat", 0.9, 1)


def test_flow_values_are_the_change_not_the_level():
    v = [0.10, 0.15, 0.20, 0.18]
    out = cot._flow_values(v, 1)
    assert out[0] is None
    assert out[1] == pytest.approx(0.05)   # 0.15 - 0.10
    assert out[2] == pytest.approx(0.05)   # 0.20 - 0.15
    assert out[3] == pytest.approx(-0.02)  # 0.18 - 0.20
    # a None on either end of the window yields None (no fabricated change)
    assert cot._flow_values([0.1, None, 0.3], 1) == [None, None, None]


def test_cot_index_uses_a_trailing_window_not_expanding():
    # value[6]=0.07 is the MAX of its trailing [2..6] (index 100 -> top), but the MIDDLE of the full range
    # [0..6] (min 0, max 0.14 -> index 50 -> no extreme): trailing fires, expanding does not.
    vals = [0.0, 0.14, 0.05, 0.04, 0.06, 0.05, 0.07]
    ts = range(len(vals))
    trailing = cot.cot_sides(ts, vals, "cot_momentum", cot_pct=0.90, min_history=2, index_window=5)
    assert trailing[6] == 1.0   # trailing max -> index 100 -> ride -> long
    expanding = cot.cot_sides(ts, vals, "cot_momentum", cot_pct=0.90, min_history=2, index_window=1000)
    assert expanding[6] is None  # index 50 vs the full [0,0.14] range -> no extreme


# --- causality: positioning known at bar -> applied to next bar; positions ignore price --------------


def _extreme_series(n=60):
    ts = _days(n)
    vals = [0.10 + (0.02 if i % 2 else -0.02) for i in range(n)]
    for i in (18, 32, 46):
        vals[i] = 0.80  # engineered crowded extremes
    closes = list(100.0 * np.cumprod(1.0 + np.array([0.01 if i % 2 else -0.008 for i in range(n)])))
    return ts, closes, vals


@pytest.mark.parametrize("signal", ["cot_contrarian", "cot_momentum"])
def test_positions_are_time_prefix_causal(signal):
    ts, closes, vals = _extreme_series()
    cfg = {"signal": signal, "cot_pct": 0.90, "hold_bars": 3, "min_history": 10}
    clean = cot.positions_from_bars(ts, closes, vals, cfg)
    moved = False
    for cut in range(20, 55):
        dirty = list(vals)
        for i in range(cut, len(ts)):
            dirty[i] = 0.95
        dpos = cot.positions_from_bars(ts, closes, dirty, cfg)
        assert np.array_equal(clean[: cut + 1], dpos[: cut + 1])
        moved = moved or not np.array_equal(clean[cut + 1 :], dpos[cut + 1 :])
    assert moved


def test_positions_ignore_price():
    ts, closes, vals = _extreme_series()
    cfg = {"signal": "cot_contrarian", "cot_pct": 0.90, "hold_bars": 3, "min_history": 10}
    base = cot.positions_from_bars(ts, closes, vals, cfg)
    scaled = cot.positions_from_bars(ts, [c * 3.3 for c in closes], vals, cfg)
    assert np.array_equal(base, scaled)


# --- the run() contract ------------------------------------------------------------------------------


def _daily_bars(n=400):
    rng = np.random.default_rng(2)
    closes = 2000.0 * np.cumprod(1.0 + rng.normal(0.0004, 0.011, n))
    base = cot._month_start_ms("2020-01")
    return {
        "timestamp": [base + i * _DAY for i in range(n)],
        "open": list(closes), "high": list(closes * 1.006), "low": list(closes * 0.994),
        "close": list(closes), "volume": [1.0] * n,
    }


def test_run_emits_the_full_metric_vocabulary(monkeypatch):
    bars = _daily_bars(1100)  # span into the 2022 test window
    n = len(bars["timestamp"])
    vals = [0.10 + 0.30 * np.sin(i / 20.0) for i in range(n)]  # oscillating positioning so it trades
    monkeypatch.setattr(cot, "_load_bars", lambda asset, pairs, bar_minutes: bars)
    monkeypatch.setattr(cot, "cot_values", lambda asset, timestamps, release_lag_days: vals)
    cfg = {
        "asset": "GOLD", "signal": "cot_contrarian", "cot_pct": 0.85, "hold_bars": 10,
        "release_lag_days": 4, "transaction_fee": 0.0005, "walk_forward_window": "2022", "seed": 0,
    }
    summary = cot.run(cfg)
    m = summary["metrics"]
    for key in ("total_return_pct", "oos_sharpe", "return_vs_hold_pct", "signal_expectancy", "n_trades", "n_signals"):
        assert key in m, f"missing metric {key}"
        assert np.isfinite(m[key])
    assert summary["objective"] == m["oos_sharpe"]
    assert m["n_trades"] > 0 and m["n_signals"] > 0
    assert summary["dataset"]["asset"] == "GOLD"


def test_run_refuses_unknown_signal(monkeypatch):
    monkeypatch.setattr(cot, "_load_bars", lambda asset, pairs, bar_minutes: _daily_bars(60))
    monkeypatch.setattr(cot, "cot_values", lambda asset, timestamps, release_lag_days: [0.1] * 60)
    with pytest.raises(SystemExit):
        cot.run({"signal": "cot_wat", "walk_forward_window": "2022"})
