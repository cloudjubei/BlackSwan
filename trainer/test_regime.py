"""Direct gating tests for the macro-regime probe — a long-or-flat TIMING overlay (the B4 world-model line).

Every prior line sought a per-trade directional edge and found none. This asks a different question: can a
slow MACRO regime estimate (rates easing/tightening, curve, jobless-claims trend) time crypto EXPOSURE — hold
in risk-on, sit in cash in risk-off — and beat buy-and-hold by dodging drawdowns? It is not a per-trade signal
but an allocation overlay, so the honest test is beating hold across a BULL and a BEAR window at once: a real
regime signal must stay invested in the bull AND go flat in the bear. The macro values are joined POINT-IN-TIME
(pit_fusion: known only from their release instant, DST-aware, never the reference period), which is the whole
leakage game here. Guards pinned before a number:

* the regime rule is PAST-ONLY — risk-on/off at day t compares the as-of macro value at t against the as-of
  value `lookback` days earlier, both known at/before t; no whole-sample or future-revision peeking;
* the macro value at a day is the latest RELEASED by that day (a bar before the first release is undefined,
  never forward-guessed);
* the regime decided at day t sets EXPOSURE for day t+1's return, never day t's own; the switch fee is charged
  on every risk-on<->off flip.
"""

import numpy as np
import pytest

from trainer import regime


# --- apply_rule: past-only trend logic, per named rule ------------------------------------------------


def test_rates_easing_is_on_when_the_10y_has_fallen_over_the_lookback():
    # DGS10 falling over the lookback -> easing -> risk-on; rising -> risk-off; the first `lookback` bars are
    # undefined (no value `lookback` ago).
    dgs10 = [4.0, 4.1, 4.2, 4.3, 4.2, 4.1, 4.0, 3.9]  # rises then falls
    on = regime.apply_rule({"DGS10": dgs10}, "rates_easing", lookback=3)
    assert on[0] is None and on[1] is None and on[2] is None  # warmup: no value 3 bars ago
    assert on[3] is False  # 4.3 vs 4.0 -> rose -> risk-off
    assert on[6] is True   # 4.0 vs 4.3 -> fell -> risk-on (easing)
    assert on[7] is True   # 3.9 vs 4.2 -> fell -> risk-on


def test_curve_steepening_and_claims_falling_directions():
    t10y2y = [-0.5, -0.4, -0.3, -0.1]  # curve steepening (rising from inversion) -> risk-on
    assert regime.apply_rule({"T10Y2Y": t10y2y}, "curve_steepening", lookback=2)[2] is True   # -0.3 >= -0.5
    icsa = [200.0, 210.0, 230.0, 240.0]  # claims RISING -> labour deteriorating -> risk-off
    assert regime.apply_rule({"ICSA": icsa}, "claims_falling", lookback=2)[2] is False          # 230 > 200


def test_risk_composite_needs_two_of_three_votes():
    vals = {
        "DGS10": [4.0, 4.0, 3.5],    # fell -> easing vote TRUE
        "T10Y2Y": [-0.5, -0.5, 0.1],  # steepened -> vote TRUE
        "ICSA": [200.0, 200.0, 260.0],  # rose -> claims vote FALSE
    }
    on = regime.apply_rule(vals, "risk_composite", lookback=2)
    assert on[2] is True   # 2 of 3 risk-on -> composite risk-on
    vals["T10Y2Y"] = [-0.5, -0.5, -0.9]  # now curve also FALSE -> only 1 of 3
    assert regime.apply_rule(vals, "risk_composite", lookback=2)[2] is False


def test_apply_rule_refuses_unknown_rule():
    with pytest.raises(SystemExit):
        regime.apply_rule({"DGS10": [1, 2, 3]}, "vibes", lookback=1)


# --- positions: overlay holds in risk-on, inverse is the mirror; exposure set for t+1 ----------------


def _series(n=20):
    ts = list(range(n))
    closes = list(100.0 * np.cumprod(1.0 + np.array([0.01 if i % 2 else -0.005 for i in range(n)])))
    return ts, closes


def test_overlay_holds_in_risk_on_and_inverse_is_the_mirror(monkeypatch):
    ts, closes = _series()
    # A regime that is risk-on for the first half, risk-off for the second.
    ron = [True] * 10 + [False] * 10
    monkeypatch.setattr(regime, "regime_on", lambda timestamps, cfg: ron)
    ov = regime.regime_positions(ts, closes, {"signal": "regime_overlay"})
    inv = regime.regime_positions(ts, closes, {"signal": "regime_inverse"})
    # Exposure decided at day t applies to day t+1: risk-on at day 0..8 -> held days 1..9; flat after.
    assert ov[0] == 0.0 and ov[1] == 1.0 and ov[9] == 1.0 and ov[10] == 1.0  # day9 risk-on -> day10 held
    assert ov[11] == 0.0 and ov[19] == 0.0                                   # risk-off -> flat
    assert all(inv[t] == (1.0 - ov[t]) for t in range(1, 20))                # inverse is 1 - overlay


def test_warmup_none_regime_is_flat(monkeypatch):
    ts, closes = _series()
    ron = [None] * 5 + [True] * 15
    monkeypatch.setattr(regime, "regime_on", lambda timestamps, cfg: ron)
    pos = regime.regime_positions(ts, closes, {"signal": "regime_overlay"})
    assert all(pos[t] == 0.0 for t in range(6))  # undefined regime -> no exposure


# --- causality: regime at t sets exposure for t+1; past-only in macro --------------------------------


def test_positions_are_time_prefix_causal_in_macro():
    # risk_on built from a real past-only rule over a macro series; corrupting the macro AFTER a cut must not
    # change the exposure prefix. Drive apply_rule directly through regime_on's pure core.
    n = 40
    ts = list(range(n))
    closes = list(100.0 * np.cumprod(1.0 + np.array([0.005 if i % 3 else -0.004 for i in range(n)])))
    dgs = list(4.0 + 0.5 * np.sin(np.arange(n) / 3.0))  # oscillating yields -> alternating regime
    cfg = {"signal": "regime_overlay", "rule": "rates_easing", "lookback": 4}

    def positions_from_macro(macro):
        ron = regime.apply_rule({"DGS10": macro}, cfg["rule"], cfg["lookback"])
        return regime.regime_positions(ts, closes, cfg, risk_on=ron)

    clean = positions_from_macro(dgs)
    moved = False
    for cut in range(10, 35):
        dirty = list(dgs)
        for i in range(cut, n):
            dirty[i] = dgs[i] + 2.0  # corrupt macro at/after the cut
        dpos = positions_from_macro(dirty)
        assert np.array_equal(clean[: cut + 1], dpos[: cut + 1])  # exposure decided before the cut is frozen
        moved = moved or not np.array_equal(clean[cut + 1 :], dpos[cut + 1 :])
    assert moved  # non-vacuous


def test_switch_fee_is_charged_on_regime_flips():
    closes = [100.0] * 8  # flat prices: only the switch fee can move equity
    pos = np.array([0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0])  # in, out, in, out -> flips cost fee
    fee = 0.001
    free = regime.backtest(closes, pos, fee=0.0)
    paid = regime.backtest(closes, pos, fee=fee)
    assert free[-1] == 1.0 and paid[-1] < 1.0  # flips are charged


# --- the run() contract ------------------------------------------------------------------------------


def _daily_bars(n=500, start="2024-01"):
    rng = np.random.default_rng(2)
    closes = 100.0 * np.cumprod(1.0 + rng.normal(0.001, 0.02, n))
    base = regime._month_start_ms(start)
    day = 86_400_000
    return {
        "timestamp": [base + i * day for i in range(n)],
        "open": list(closes), "high": list(closes * 1.01), "low": list(closes * 0.99),
        "close": list(closes), "volume": [1.0] * n,
    }


def test_run_emits_the_full_metric_vocabulary(monkeypatch):
    bars = _daily_bars()
    monkeypatch.setattr(regime, "_load_bars", lambda asset, pairs, bar_minutes: bars)
    # a regime that toggles a few times over the span
    ron = [(i // 40) % 2 == 0 for i in range(len(bars["timestamp"]))]
    monkeypatch.setattr(regime, "regime_on", lambda timestamps, cfg: ron)
    cfg = {
        "asset": "BTCUSDT", "signal": "regime_overlay", "rule": "rates_easing", "lookback": 63,
        "transaction_fee": 0.001, "walk_forward_window": "2024", "seed": 0,
    }
    summary = regime.run(cfg)
    m = summary["metrics"]
    for key in (
        "total_return_pct", "oos_sharpe", "return_vs_hold_pct", "sharpe_vs_hold", "hold_return_pct",
        "max_drawdown_pct", "time_in_market_pct", "n_switches", "n_trades",
    ):
        assert key in m, f"missing metric {key}"
        assert np.isfinite(m[key])
    assert summary["objective"] == m["oos_sharpe"]
    assert 0.0 <= m["time_in_market_pct"] <= 100.0
    assert summary["dataset"]["timeframe"] == "1440m"
    assert "provenance" in summary and "config" in summary


def test_run_refuses_unknown_signal(monkeypatch):
    monkeypatch.setattr(regime, "_load_bars", lambda asset, pairs, bar_minutes: _daily_bars(30))
    monkeypatch.setattr(regime, "regime_on", lambda timestamps, cfg: [True] * 30)
    with pytest.raises(SystemExit):
        regime.run({"signal": "regime_wat", "rule": "rates_easing", "walk_forward_window": "2024"})
