"""Direct gating tests for the attention probe — the first SENTIMENT/attention stream (B3).

Wikipedia daily pageviews for an asset's article proxy retail ATTENTION. This probe asks whether an attention
SURGE — pageviews in the top tail of their own past — predicts the next day's move: ride it (attention_momentum,
FOMO continues) or fade it (attention_reversion, the hype day is a local top). It reuses the shared past-only
extreme core and the intraday backtest mechanics, so the surface new here is the pageview JOIN, which carries
the one leakage trap: Wikimedia publishes a day's count only AFTER that day, so a bar on day D may use pageviews
from day D-1 at the earliest, NEVER day D's own (still-unpublished) count. Guards pinned here:

* the join is PUBLICATION-LAGGED — a crypto bar dated D reads pageviews[D - lag_days] (lag_days>=1), so a spike
  on day D first becomes visible one bar later, and the traded return is two days after the pageview day;
* "is attention extreme?" is judged past-only (delegated to signal_extremes), and the position is applied to
  the next bar; positions never read price.
"""

import numpy as np
import pytest

from trainer import attention

_DAY = 86_400_000


def _days(n, start_ms=1_704_067_200_000):  # 2024-01-01 00:00 UTC
    return [start_ms + i * _DAY for i in range(n)]


# --- pageview join: publication-lagged, a day's own count is never used ------------------------------


def test_attention_values_are_publication_lagged(tmp_path, monkeypatch):
    import json
    ts = _days(4)  # 2024-01-01..04
    # A huge spike on 2024-01-02; with lag_days=1 it must appear at the 2024-01-03 bar, never 01-02.
    pv = {"2024-01-01": 100, "2024-01-02": 99999, "2024-01-03": 200, "2024-01-04": 150}
    (tmp_path / "BTCUSDT.json").write_text(json.dumps(pv))
    monkeypatch.setattr(attention, "PAGEVIEWS_DIR", str(tmp_path))
    vals = attention.attention_values("BTCUSDT", ts, lag_days=1)
    assert vals[0] is None            # bar 01-01 needs pageviews[2023-12-31] which is absent
    assert vals[1] == 100.0           # bar 01-02 sees 01-01, NOT its own-day 99999 spike
    assert vals[2] == 99999.0         # the spike surfaces one bar later (01-03 sees 01-02)
    assert vals[3] == 200.0


def test_attention_values_missing_file_is_all_none(monkeypatch, tmp_path):
    monkeypatch.setattr(attention, "PAGEVIEWS_DIR", str(tmp_path))
    assert attention.attention_values("NOPE", _days(3), lag_days=1) == [None, None, None]


def test_lag_days_below_one_is_refused(monkeypatch, tmp_path):
    # A day's pageviews are published only AFTER that day, so lag_days<1 would read the bar's own (or a future)
    # still-unpublished count — a same-day lookahead. It must be refused, not silently joined.
    monkeypatch.setattr(attention, "PAGEVIEWS_DIR", str(tmp_path))
    for bad in (0, -1):
        with pytest.raises(SystemExit):
            attention.attention_values("BTCUSDT", _days(3), lag_days=bad)


# --- surge signal: momentum rides, reversion fades ---------------------------------------------------


def test_momentum_rides_and_reversion_fades_an_attention_surge():
    ts = _days(31)
    vals = [1000.0 + (50.0 if i % 2 else -50.0) for i in range(30)] + [50000.0]  # a surge at the end
    mom = attention.attention_sides(ts, vals, "attention_momentum", attention_pct=0.90, min_history=10)
    rev = attention.attention_sides(ts, vals, "attention_reversion", attention_pct=0.90, min_history=10)
    assert mom[-1] == 1.0    # ride the surge -> long
    assert rev[-1] == -1.0   # fade it -> short


def test_attention_sides_refuses_unknown_signal():
    with pytest.raises(SystemExit):
        attention.attention_sides(_days(3), [1.0, 2.0, 3.0], "attention_wat", 0.9, 1)


# --- causality: surge known at bar -> applied to next bar; positions ignore price --------------------


def _surge_series(n=60):
    ts = _days(n)
    vals = [1000.0 + (30.0 if i % 2 else -30.0) for i in range(n)]
    for i in (18, 32, 46):
        vals[i] = 60000.0  # engineered surges
    closes = list(100.0 * np.cumprod(1.0 + np.array([0.01 if i % 2 else -0.008 for i in range(n)])))
    return ts, closes, vals


@pytest.mark.parametrize("signal", ["attention_momentum", "attention_reversion"])
def test_positions_are_time_prefix_causal(signal):
    ts, closes, vals = _surge_series()
    cfg = {"signal": signal, "attention_pct": 0.90, "hold_bars": 3, "min_history": 10}
    clean = attention.positions_from_bars(ts, closes, vals, cfg)
    moved = False
    for cut in range(20, 55):
        dirty = list(vals)
        for i in range(cut, len(ts)):
            dirty[i] = 90000.0 * (1 if i % 2 else -1) if dirty[i] else dirty[i]
            dirty[i] = 90000.0  # corrupt attention at/after the cut
        dpos = attention.positions_from_bars(ts, closes, dirty, cfg)
        assert np.array_equal(clean[: cut + 1], dpos[: cut + 1])
        moved = moved or not np.array_equal(clean[cut + 1 :], dpos[cut + 1 :])
    assert moved


def test_positions_ignore_price():
    ts, closes, vals = _surge_series()
    cfg = {"signal": "attention_momentum", "attention_pct": 0.90, "hold_bars": 3, "min_history": 10}
    base = attention.positions_from_bars(ts, closes, vals, cfg)
    scaled = attention.positions_from_bars(ts, [c * 4.2 for c in closes], vals, cfg)
    assert np.array_equal(base, scaled)


def test_surge_never_traded_on_its_own_bar_and_holds():
    ts, closes, vals = _surge_series()
    cfg = {"signal": "attention_momentum", "attention_pct": 0.90, "hold_bars": 3, "min_history": 10}
    pos = attention.positions_from_bars(ts, closes, vals, cfg)
    assert pos[18] == 0.0
    assert pos[19] == 1.0 and pos[20] == 1.0 and pos[21] == 1.0 and pos[22] == 0.0


# --- the run() contract ------------------------------------------------------------------------------


def _daily_bars(n=500):
    rng = np.random.default_rng(1)
    closes = 100.0 * np.cumprod(1.0 + rng.normal(0.001, 0.02, n))
    base = attention._month_start_ms("2024-01")
    return {
        "timestamp": [base + i * _DAY for i in range(n)],
        "open": list(closes), "high": list(closes * 1.01), "low": list(closes * 0.99),
        "close": list(closes), "volume": [1.0] * n,
    }


def test_run_emits_the_full_metric_vocabulary(monkeypatch):
    bars = _daily_bars()
    vals = [1000.0 + 100.0 * np.sin(i / 3.0) for i in range(len(bars["timestamp"]))]
    for i in range(0, len(vals), 20):
        vals[i] = 50000.0  # periodic surges so it trades
    monkeypatch.setattr(attention, "_load_bars", lambda asset, pairs, bar_minutes: bars)
    monkeypatch.setattr(attention, "attention_values", lambda asset, timestamps, lag_days: vals)
    cfg = {
        "asset": "BTCUSDT", "signal": "attention_momentum", "attention_pct": 0.85, "hold_bars": 2,
        "lag_days": 1, "transaction_fee": 0.001, "walk_forward_window": "2024", "seed": 0,
    }
    summary = attention.run(cfg)
    m = summary["metrics"]
    for key in (
        "total_return_pct", "oos_sharpe", "return_vs_hold_pct", "hold_return_pct",
        "trades_per_day", "realized_cost_bps", "signal_expectancy", "n_trades", "n_surges",
        "beta", "up_capture", "down_capture",
    ):
        assert key in m, f"missing metric {key}"
        assert np.isfinite(m[key])
    assert summary["objective"] == m["oos_sharpe"]
    assert m["n_trades"] > 0 and m["n_surges"] > 0
    assert summary["dataset"]["timeframe"] == "1440m"
    assert "provenance" in summary and "config" in summary


def test_run_refuses_unknown_signal(monkeypatch):
    monkeypatch.setattr(attention, "_load_bars", lambda asset, pairs, bar_minutes: _daily_bars(30))
    monkeypatch.setattr(attention, "attention_values", lambda asset, timestamps, lag_days: [1.0] * 30)
    with pytest.raises(SystemExit):
        attention.run({"signal": "attention_wat", "walk_forward_window": "2024"})
