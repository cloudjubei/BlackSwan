"""Direct gating tests for the liquidation-cascade probe — the mechanically-asymmetric line.

A liquidation cascade is FORCED deleveraging: an exchange market-closes blown-up leveraged positions
regardless of price, so a long cascade dumps aggressive SELLS into the book (price craters on a volume spike
with one-sided taker selling) and a short squeeze does the mirror. The bet is that this selling is
NON-informational — the seller has no choice — so the dislocation OVER-shoots and REVERTS. Real historical
liquidation feeds are not freely available, so the cascade is DETECTED from a proxy already on disk: a bar
whose (1) move clears k*trailing-vol, (2) volume clears a past-only spike quantile, and (3) taker imbalance is
extreme ON THE SAME SIDE as the move (the forced flow that CAUSED it). All three are required — any one alone
is just a volatile bar. The thesis arm (cascade_reversion) fades the cascade; the control (cascade_momentum)
rides it.

The guards pinned here before a number is emitted:
* the compound gate is a real AND — a bar passing only two of the three conditions must NOT fire (else the
  probe degenerates into the already-null vol-breakout or order-flow arms);
* the flow must CONFIRM the move's direction — a crash on aggressive BUYING is not a sell-cascade;
* every threshold (trailing vol, the volume-spike quantile, the imbalance-extreme quantile) is judged
  past-only, never whole-sample;
* the cascade is detected at bar t from data known at t's close and the reversion/momentum position is applied
  to bars t+1.. (the AFTERMATH), never bar t's own dislocation return — so the pipeline is time-prefix causal.
"""

import numpy as np
import pytest

from trainer import liquidation


def _calm(n=40, seed=0):
    # A calm base: tiny alternating returns, low volume ~1, imbalance noisy around 0 (so its tails are
    # meaningful, not a degenerate 0). Returned as (closes, volume, imbalance) arrays to drive the detector
    # directly, so a test controls the exact return/volume/flow of every bar.
    rng = np.random.default_rng(seed)
    rets = np.array([0.0008 if i % 2 else -0.0008 for i in range(n)])
    closes = list(100.0 * np.cumprod(1.0 + rets))
    volume = list(1.0 + 0.1 * rng.standard_normal(n))
    imbalance = list(0.05 * rng.standard_normal(n))
    return closes, volume, imbalance


def _inject_down_cascade(closes, volume, imbalance, t, drop=0.05, vol=100.0, imb=-0.9):
    # Turn bar t into a down-cascade: a hard drop, a volume spike, and extreme aggressive SELLING.
    closes = list(closes)
    for i in range(t, len(closes)):
        closes[i] = closes[i] * (1.0 - drop) if i == t else closes[i] * (1.0 - drop)
    volume = list(volume); volume[t] = vol
    imbalance = list(imbalance); imbalance[t] = imb
    return closes, volume, imbalance


CFG = {"vol_window": 5, "k_move": 3.0, "vol_pct": 0.9, "flow_pct": 0.8, "min_history": 10}


def test_down_cascade_fires_and_reversion_goes_long():
    closes, volume, imbalance = _calm()
    closes, volume, imbalance = _inject_down_cascade(closes, volume, imbalance, 30)
    rev = liquidation.cascade_sides(closes, volume, imbalance, {**CFG, "signal": "cascade_reversion"})
    mom = liquidation.cascade_sides(closes, volume, imbalance, {**CFG, "signal": "cascade_momentum"})
    assert rev[30] == 1.0    # fade the crash -> LONG the forced-sell dislocation
    assert mom[30] == -1.0   # ride it -> short; the control is the exact negation
    assert all(rev[i] is None for i in range(30))  # nothing fires in the calm base


def test_up_cascade_fires_and_reversion_goes_short():
    closes, volume, imbalance = _calm()
    # An up-cascade (short squeeze): hard rise, volume spike, extreme aggressive BUYING.
    closes = list(closes)
    for i in range(30, len(closes)):
        closes[i] = closes[i] * 1.05
    volume = list(volume); volume[30] = 100.0
    imbalance = list(imbalance); imbalance[30] = 0.9
    rev = liquidation.cascade_sides(closes, volume, imbalance, {**CFG, "signal": "cascade_reversion"})
    assert rev[30] == -1.0  # fade the squeeze -> SHORT


# --- the compound gate: all three conditions are required --------------------------------------------


def test_a_big_move_without_volume_spike_does_not_fire():
    closes, volume, imbalance = _calm()
    closes, volume, imbalance = _inject_down_cascade(closes, volume, imbalance, 30, vol=1.0)  # NO volume spike
    sides = liquidation.cascade_sides(closes, volume, imbalance, {**CFG, "signal": "cascade_reversion"})
    assert sides[30] is None


def test_a_volume_spike_without_a_big_move_does_not_fire():
    closes, volume, imbalance = _calm()
    volume = list(volume); volume[30] = 100.0
    imbalance = list(imbalance); imbalance[30] = -0.9  # spike + flow, but the move stays calm
    sides = liquidation.cascade_sides(closes, volume, imbalance, {**CFG, "signal": "cascade_reversion"})
    assert sides[30] is None


def test_a_move_and_spike_without_confirming_flow_does_not_fire():
    closes, volume, imbalance = _calm()
    closes, volume, imbalance = _inject_down_cascade(closes, volume, imbalance, 30, imb=0.0)  # flat flow
    sides = liquidation.cascade_sides(closes, volume, imbalance, {**CFG, "signal": "cascade_reversion"})
    assert sides[30] is None


def test_flow_must_confirm_the_move_direction():
    # A hard DROP but with aggressive BUYING (positive imbalance) is not a sell-cascade — the forced flow must
    # be on the same side as the move. A wrong-side detector (|imbalance| extreme regardless of sign) would fire.
    closes, volume, imbalance = _calm()
    closes, volume, imbalance = _inject_down_cascade(closes, volume, imbalance, 30, imb=+0.9)
    sides = liquidation.cascade_sides(closes, volume, imbalance, {**CFG, "signal": "cascade_reversion"})
    assert sides[30] is None


# --- past-only thresholds ----------------------------------------------------------------------------


def test_thresholds_are_past_only_not_whole_sample():
    # A volume BURST in the PAST (bars 40-44) then a moderate volume at the probed bar and low volume after.
    # Bar 47's volume is top-decile versus the WHOLE sample (dragged down by the many low future bars) but
    # ordinary versus its own burst-dominated past. Paired with a real drop + confirming aggressive selling,
    # so ONLY the volume-quantile lookahead could make it fire.
    n = 90
    rets = [0.0008 if i % 2 else -0.0008 for i in range(n)]
    rets[47] = -0.05  # a real drop at 47
    closes = list(100.0 * np.cumprod(1.0 + np.array(rets)))
    volume = [1.0] * 40 + [100.0] * 5 + [5.0] * (n - 45)  # burst 40-44 (PAST), then moderate 5
    imbalance = [0.02 * (1 if i % 2 else -1) for i in range(n)]; imbalance[47] = -0.9
    sides = liquidation.cascade_sides(closes, volume, imbalance, {**CFG, "signal": "cascade_reversion"})
    assert volume[47] >= np.quantile(volume, 0.90)      # spike vs the FULL sample (~5)
    assert volume[47] < np.quantile(volume[:47], 0.90)  # ordinary vs its OWN burst-dominated past (~43)
    assert sides[47] is None  # past-only volume gate leaves it flat; a whole-sample gate would fire it


# --- causality: detected at t, applied to t+1, never bar t's own return ------------------------------


def _cascade_series(n=60):
    closes, volume, imbalance = _calm(n)
    for t in (18, 32, 46):
        closes, volume, imbalance = _inject_down_cascade(closes, volume, imbalance, t)
    return closes, volume, imbalance


def test_position_applies_to_next_bars_and_holds_never_its_own_bar():
    closes, volume, imbalance = _cascade_series()
    cfg = {**CFG, "signal": "cascade_reversion", "hold_bars": 3}
    pos = liquidation.positions_from_bars(list(range(len(closes))), closes, volume, imbalance, cfg)
    assert pos[18] == 0.0  # cascade detected at 18, never traded on its own dislocation bar
    assert pos[19] == 1.0 and pos[20] == 1.0 and pos[21] == 1.0 and pos[22] == 0.0


@pytest.mark.parametrize("signal", ["cascade_reversion", "cascade_momentum"])
def test_positions_are_time_prefix_causal(signal):
    closes, volume, imbalance = _cascade_series()
    ts = list(range(len(closes)))
    cfg = {**CFG, "signal": signal, "hold_bars": 3}
    clean = liquidation.positions_from_bars(ts, closes, volume, imbalance, cfg)
    moved = False
    for cut in range(20, 55):
        dclose = list(closes); dvol = list(volume); dimb = list(imbalance)
        for i in range(cut, len(closes)):
            dclose[i] = closes[i] * 1.5; dvol[i] = volume[i] * 3.0; dimb[i] = -imbalance[i]
        dpos = liquidation.positions_from_bars(ts, dclose, dvol, dimb, cfg)
        assert np.array_equal(clean[: cut + 1], dpos[: cut + 1])  # prefix decided before the cut is frozen
        moved = moved or not np.array_equal(clean[cut + 1 :], dpos[cut + 1 :])
    assert moved  # non-vacuous


def test_cascade_sides_refuses_unknown_signal():
    closes, volume, imbalance = _calm()
    with pytest.raises(SystemExit):
        liquidation.cascade_sides(closes, volume, imbalance, {**CFG, "signal": "cascade_wat"})


# --- the run() contract ------------------------------------------------------------------------------


def _synthetic_flow_bars(n=1200, bar_ms=900_000):
    rng = np.random.default_rng(4)
    rets = rng.normal(0.0, 0.004, n)
    vol = list(1.0 + 0.2 * np.abs(rng.standard_normal(n)))
    imb = list(0.1 * rng.standard_normal(n))
    for i in range(20, n, 40):  # periodic down-cascades
        rets[i] = -0.06; vol[i] = 50.0; imb[i] = -0.9
    closes = 100.0 * np.cumprod(1.0 + rets)
    base = liquidation._month_start_ms("2024-01")
    return {
        "timestamp": [base + i * bar_ms for i in range(n)],
        "close": list(closes), "volume": vol, "taker_buy": [0.0] * n, "imbalance": imb,
    }


def test_run_emits_the_full_metric_vocabulary(monkeypatch):
    bars = _synthetic_flow_bars()
    monkeypatch.setattr(liquidation, "_load_flow_bars", lambda asset, pairs, bar_minutes: bars)
    cfg = {
        "asset": "BTCUSDT", "bar": 15, "signal": "cascade_reversion", "vol_window": 20,
        "k_move": 3.0, "vol_pct": 0.9, "flow_pct": 0.8, "hold_bars": 4, "transaction_fee": 0.001,
        "walk_forward_window": "2024", "seed": 0,
    }
    summary = liquidation.run(cfg)
    m = summary["metrics"]
    for key in (
        "total_return_pct", "oos_sharpe", "return_vs_hold_pct", "hold_return_pct",
        "trades_per_day", "realized_cost_bps", "signal_expectancy", "n_trades", "n_cascades",
        "beta", "up_capture", "down_capture",
    ):
        assert key in m, f"missing metric {key}"
        assert np.isfinite(m[key])
    assert summary["objective"] == m["oos_sharpe"]
    assert m["n_cascades"] > 0 and m["n_trades"] > 0
    assert summary["dataset"]["timeframe"] == "15m"
    assert "provenance" in summary and "config" in summary


def test_run_refuses_unknown_signal(monkeypatch):
    monkeypatch.setattr(liquidation, "_load_flow_bars", lambda asset, pairs, bar_minutes: _synthetic_flow_bars(50))
    with pytest.raises(SystemExit):
        liquidation.run({"signal": "cascade_wat", "bar": 15, "walk_forward_window": "2024"})
