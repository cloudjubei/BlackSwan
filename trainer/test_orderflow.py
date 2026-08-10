"""Direct gating tests for the order-flow (taker-imbalance) probe — the second NON-PRICE stream.

Binance 1m klines already carry the aggressor split: ``asset_volume_taker_base`` is the volume where the TAKER
was the buyer (an aggressive market buy), so aggressive selling is ``volume - taker_buy`` and the per-bar
imbalance is (2*taker_buy - volume)/volume in [-1, +1] — net aggressive buying above 0, net selling below.
This probe asks whether conditioning a directional trade on an EXTREME of that imbalance (ride it — informed
flow persists — or fade it — aggressors overpay and get run over) beats buy-and-hold net of the ~0.2% round
trip. It reuses the intraday backtest mechanics and the shared past-only extreme core, so the surfaces new
here are the flow aggregation and the imbalance sign, each pinned before a number is emitted:

* imbalance is built from ONLY the bar's own minutes (no straddle into the next bar's taker volume), and is
  the SIGNED aggressor ratio — an all-buy bar is +1, an all-sell bar is -1, a balanced bar is 0. A sign slip
  or a total-volume/taker mix-up would invert or flatten the whole signal silently.

* the imbalance measured over bar t is known at t's close and the position it implies is applied to bars
  t+1.. — never bar t's own return — so the position pipeline is time-prefix causal in flow, and positions
  never read price at all (only the imbalance does).

* "is this imbalance extreme?" is judged past-only (delegated to signal_extremes, guarded there); an
  integration check confirms the delegation is wired to the flow series, not to price.
"""

import numpy as np
import pytest

from trainer import orderflow


def _row(ts, close, volume, taker_buy):
    # Prices/volumes arrive as STRINGS in the binance 1m files; only close matters for OHLC here (the probe
    # trades on flow, not OHLC shape), so open/high/low track close.
    return {
        "timestamp": ts,
        "price_open": f"{close}", "price_high": f"{close}", "price_low": f"{close}", "price": f"{close}",
        "volume": f"{volume}", "asset_volume_taker_base": f"{taker_buy}",
    }


# --- flow aggregation: signed imbalance, own-minutes-only ---------------------------------------------


def test_imbalance_sign_all_buy_all_sell_balanced():
    # Three 1-minute bars each their own 1m decision bar: all-buy -> +1, all-sell -> -1, half -> 0.
    rows = [_row(0, 100, 10.0, 10.0), _row(60000, 100, 10.0, 0.0), _row(120000, 100, 10.0, 5.0)]
    bars = orderflow.flow_bars(rows, bar_minutes=1)
    assert bars["imbalance"][0] == pytest.approx(1.0)   # taker_buy == volume -> fully aggressive buying
    assert bars["imbalance"][1] == pytest.approx(-1.0)  # taker_buy == 0 -> fully aggressive selling
    assert bars["imbalance"][2] == pytest.approx(0.0)   # half buy / half sell -> balanced


def test_imbalance_uses_only_the_bars_own_minutes():
    # One 5-minute bar (minutes 0-4) then the next (minute 5). Minute 5 is 100% sell and huge; it must NOT
    # bleed into bar 0's imbalance. Bar 0: buy volume 8 of total 10 -> +0.6.
    rows = [
        _row(0, 100, 4.0, 4.0), _row(60000, 100, 2.0, 2.0), _row(120000, 100, 2.0, 0.0),
        _row(180000, 100, 1.0, 1.0), _row(240000, 100, 1.0, 1.0),  # bar 0 totals: vol 10, buy 8
        _row(300000, 100, 999.0, 0.0),  # bar 1: a colossal all-sell minute
    ]
    bars = orderflow.flow_bars(rows, bar_minutes=5)
    assert bars["volume"][0] == pytest.approx(10.0) and bars["taker_buy"][0] == pytest.approx(8.0)
    assert bars["imbalance"][0] == pytest.approx((2 * 8.0 - 10.0) / 10.0)  # +0.6, unaffected by minute 5
    assert bars["imbalance"][1] == pytest.approx(-1.0)


def test_imbalance_is_none_for_zero_volume_bar():
    bars = orderflow.flow_bars([_row(0, 100, 0.0, 0.0)], bar_minutes=1)
    assert bars["imbalance"][0] is None  # no trades -> undefined imbalance, never a fake 0-or-NaN


def test_flow_bars_empty_stays_finite():
    bars = orderflow.flow_bars([], bar_minutes=15)
    assert bars["imbalance"] == [] and bars["close"] == []


# --- flow signal: extreme-of-past, ride vs fade ------------------------------------------------------


def test_flow_momentum_rides_and_contrarian_fades_an_extreme():
    ts = [i * 60000 for i in range(31)]
    imb = [(0.01 if i % 2 else -0.01) for i in range(30)] + [0.9]  # a lopsided BUY extreme at the end
    assert orderflow.flow_sides(ts, imb, "flow_momentum", 0.90, 10)[-1] == 1.0   # ride the buying -> long
    assert orderflow.flow_sides(ts, imb, "flow_contrarian", 0.90, 10)[-1] == -1.0  # fade it -> short
    imb2 = [(0.01 if i % 2 else -0.01) for i in range(30)] + [-0.9]
    assert orderflow.flow_sides(ts, imb2, "flow_momentum", 0.90, 10)[-1] == -1.0
    assert orderflow.flow_sides(ts, imb2, "flow_contrarian", 0.90, 10)[-1] == 1.0


def test_flow_sides_is_past_only_wired_to_the_flow_series():
    # The regime-style fixture in IMBALANCE space: a held level extreme-vs-sample but ordinary-vs-past must
    # be flat — proving the delegation reads the past-only quantile of the flow, not a whole-sample or price one.
    imb = [0.001] * 40 + [0.9] * 5 + [0.05] * 45
    ts = [i * 60000 for i in range(len(imb))]
    sides = orderflow.flow_sides(ts, imb, "flow_momentum", 0.90, 10)
    assert sides[47] is None   # ordinary vs its burst-dominated past
    assert sides[42] == 1.0    # a burst bar fires (non-vacuous)


def test_flow_sides_refuses_unknown_signal():
    with pytest.raises(SystemExit):
        orderflow.flow_sides([0, 60000], [0.1, 0.2], "flow_wat", 0.9, 1)


# --- causality: positions are time-prefix causal in flow, and never read price -----------------------


def _causal_flow():
    n = 60
    ts = [i * 60000 for i in range(n)]
    imb = [(0.02 if i % 2 else -0.02) for i in range(n)]
    for i in (18, 30, 44):
        imb[i] = 0.95  # engineered buy extremes far enough apart to each open a position
    closes = list(100.0 * np.cumprod(1.0 + np.array([0.001 if i % 2 else -0.001 for i in range(n)])))
    return ts, closes, imb


@pytest.mark.parametrize("signal", ["flow_momentum", "flow_contrarian"])
def test_positions_are_time_prefix_causal_in_flow(signal):
    ts, closes, imb = _causal_flow()
    cfg = {"signal": signal, "flow_pct": 0.90, "hold_bars": 3, "min_history": 10}
    clean = orderflow.positions_from_bars(ts, closes, imb, cfg)
    moved = False
    for cut in range(20, 55):
        dirty = list(imb)
        for i in range(cut, len(ts)):
            dirty[i] = 0.99 * (1 if i % 2 else -1)  # corrupt imbalance at/after the cut
        dpos = orderflow.positions_from_bars(ts, closes, dirty, cfg)
        assert np.array_equal(clean[: cut + 1], dpos[: cut + 1])  # prefix decided before the cut is frozen
        moved = moved or not np.array_equal(clean[cut + 1 :], dpos[cut + 1 :])
    assert moved  # non-vacuous


def test_positions_ignore_price_entirely():
    ts, closes, imb = _causal_flow()
    cfg = {"signal": "flow_momentum", "flow_pct": 0.90, "hold_bars": 3, "min_history": 10}
    base = orderflow.positions_from_bars(ts, closes, imb, cfg)
    scaled = orderflow.positions_from_bars(ts, [c * 3.7 for c in closes], imb, cfg)
    assert np.array_equal(base, scaled)


def test_signal_never_traded_on_its_own_bar_and_holds_hold_bars():
    ts, closes, imb = _causal_flow()
    cfg = {"signal": "flow_momentum", "flow_pct": 0.90, "hold_bars": 3, "min_history": 10}
    pos = orderflow.positions_from_bars(ts, closes, imb, cfg)
    assert pos[18] == 0.0  # extreme detected at 18, applied to 19..21, never bar 18's own return
    assert pos[19] == 1.0 and pos[20] == 1.0 and pos[21] == 1.0
    assert pos[22] == 0.0


# --- the run() contract: the full metric vocabulary the scorecard consumes ---------------------------


def _synthetic_flow_bars(n=1200, bar_ms=3_600_000):
    rng = np.random.default_rng(5)
    rets = rng.normal(0.0002, 0.01, n)
    closes = 100.0 * np.cumprod(1.0 + rets)
    imb = rng.normal(0.0, 0.2, n)
    for i in range(0, n, 11):
        imb[i] = 0.9 * (1 if (i // 11) % 2 else -1)  # periodic extremes so it trades
    base = orderflow._month_start_ms("2024-01")
    return {
        "timestamp": [base + i * bar_ms for i in range(n)],
        "close": list(closes), "volume": [10.0] * n, "taker_buy": [5.0] * n,
        "imbalance": [float(x) for x in imb],
    }


def test_run_emits_the_full_metric_vocabulary(monkeypatch):
    bars = _synthetic_flow_bars()
    monkeypatch.setattr(orderflow, "_load_flow_bars", lambda asset, pairs, bar_minutes: bars)
    cfg = {
        "asset": "BTCUSDT", "bar": 60, "signal": "flow_momentum", "flow_pct": 0.85,
        "hold_bars": 1, "transaction_fee": 0.001, "walk_forward_window": "2024", "seed": 0,
    }
    summary = orderflow.run(cfg)
    m = summary["metrics"]
    for key in (
        "total_return_pct", "oos_sharpe", "return_vs_hold_pct", "hold_return_pct",
        "trades_per_day", "realized_cost_bps", "signal_expectancy", "n_trades",
        "beta", "up_capture", "down_capture", "flow_coverage",
    ):
        assert key in m, f"missing metric {key}"
        assert np.isfinite(m[key])
    assert summary["objective"] == m["oos_sharpe"]
    assert m["n_trades"] > 0
    assert 0.0 <= m["flow_coverage"] <= 1.0
    assert summary["dataset"]["timeframe"] == "60m"
    assert "provenance" in summary and "config" in summary


def test_run_refuses_unknown_signal(monkeypatch):
    monkeypatch.setattr(orderflow, "_load_flow_bars", lambda asset, pairs, bar_minutes: _synthetic_flow_bars(50))
    with pytest.raises(SystemExit):
        orderflow.run({"signal": "flow_wat", "bar": 60, "walk_forward_window": "2024"})
