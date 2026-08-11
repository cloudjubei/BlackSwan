"""Direct gating tests for the world-model probe — a DRIVER-CONDITIONED directional timing model (the B4
"world model" line, generalised beyond crypto).

Instead of reading price, it reads an asset's economically-motivated FUNDAMENTAL DRIVERS point-in-time and takes
a long / short / flat position from their past-only trend. The seed case is GOLD with its three canonical macro
drivers — real rate (DFII10, gold up when it FALLS), the broad dollar (DTWEXBGS, gold up when it FALLS), and the
inflation breakeven (DGS10 - DFII10, gold up when it RISES). The driver set is configurable so any commodity can
declare its own world model. worldmodel is the thesis; worldmodel_inverse is the mirror control that must FAIL if
the drivers carry real directional content.

Guards pinned here: each driver's scalar is past-only-trended (value[t] vs value[t-lookback], both known by t);
the "breakeven" spread reads TWO DISTINCT series (DGS10 - DFII10), never one collapsed to itself; the composite
side decided at day t sets the position for day t+1, never day t's own; and positions never read price.
"""

import numpy as np
import pytest

from trainer import worldmodel


# --- driver scalar: level vs spread of two distinct series -------------------------------------------


def test_level_driver_is_the_single_series():
    driver = {"name": "real_rate", "series": ("DFII10",), "combine": "level", "bullish_on": "falling"}
    vals = worldmodel.driver_scalar(driver, {"DFII10": [1.0, 2.0, None, 4.0]})
    assert vals == [1.0, 2.0, None, 4.0]


def test_spread_driver_uses_two_distinct_series():
    driver = {"name": "breakeven", "series": ("DGS10", "DFII10"), "combine": "spread", "bullish_on": "rising"}
    vals = worldmodel.driver_scalar(driver, {"DGS10": [4.0, 4.5, 5.0], "DFII10": [1.0, 2.0, None]})
    assert vals == [3.0, 2.5, None]  # DGS10 - DFII10 ; None where either leg absent


# --- driver vote: past-only trend in the bullish direction -------------------------------------------


def test_driver_bullish_is_past_only_trend():
    driver = {"name": "real_rate", "series": ("DFII10",), "combine": "level", "bullish_on": "falling"}
    # lookback 2: bar t compares value[t] vs value[t-2]. falling => bullish.
    vals = {"DFII10": [3.0, 3.0, 2.0, 5.0, 1.0]}  # t2: 2<=3 bull; t3: 5<=3 no; t4: 1<=2 bull
    out = worldmodel.driver_bullish(driver, vals, lookback=2)
    assert out == [None, None, True, False, True]


def test_rising_driver_bullish_flips_direction():
    driver = {"name": "breakeven", "series": ("DGS10", "DFII10"), "combine": "spread", "bullish_on": "rising"}
    vals = {"DGS10": [4.0, 4.0, 4.0], "DFII10": [2.0, 2.0, 1.0]}  # spread 2,2,3 ; t2: 3>=2 bull (rising)
    out = worldmodel.driver_bullish(driver, vals, lookback=2)
    assert out == [None, None, True]


# --- composite side: majority of DEFINED drivers, all-defined required -------------------------------


def _gold_set():
    return worldmodel.resolve_driver_set("gold_macro3")


def test_composite_side_is_majority_and_requires_all_drivers_defined():
    ds = _gold_set()
    # Construct series so at the last bar: real_rate falling (bull), usd falling (bull), breakeven rising (bull)
    n = 5
    vals = {
        "DFII10": [2.0, 2.0, 2.0, 2.0, 1.0],          # falling by t4 -> real_rate bull; breakeven leg
        "DTWEXBGS": [100.0, 100.0, 100.0, 100.0, 98.0],  # falling -> usd bull
        "DGS10": [3.0, 3.0, 3.0, 3.0, 3.5],           # breakeven = DGS10-DFII10 = 1,1,1,1,2.5 rising -> bull
    }
    sides = worldmodel.worldmodel_sides(vals, ds, lookback=2)
    assert sides[-1] == 1                # all three bullish -> long
    assert sides[0] is None and sides[1] is None  # warm-up: not all drivers defined yet


def test_composite_none_when_a_driver_is_undefined():
    ds = _gold_set()
    vals = {
        "DFII10": [2.0, 2.0, 1.0],
        "DTWEXBGS": [100.0, 100.0, 98.0],
        "DGS10": [3.0, 3.0, None],  # breakeven undefined at t2 -> whole composite None (all-defined rule)
    }
    assert worldmodel.worldmodel_sides(vals, ds, lookback=2)[-1] is None


def test_bearish_majority_is_short():
    ds = _gold_set()
    vals = {
        "DFII10": [1.0, 1.0, 2.0],          # RISING -> real_rate bearish
        "DTWEXBGS": [98.0, 98.0, 100.0],    # rising -> usd bearish
        "DGS10": [3.5, 3.5, 3.0],           # breakeven 2.5,2.5,1.0 FALLING -> bearish
    }
    assert worldmodel.worldmodel_sides(vals, ds, lookback=2)[-1] == -1


# --- causality: side at t sets pos t+1; inverse flips; positions ignore price ------------------------


def test_positions_are_next_bar_causal_and_inverse_flips():
    sides = [None, 1, 1, -1, 0, 1]
    pos = worldmodel.positions_from_sides(range(6), sides, "worldmodel")
    assert list(pos) == [0.0, 0.0, 1.0, 1.0, -1.0, 0.0]  # side[t] -> pos[t+1]
    inv = worldmodel.positions_from_sides(range(6), sides, "worldmodel_inverse")
    assert list(inv) == [0.0, 0.0, -1.0, -1.0, 1.0, 0.0]  # sign flipped


def test_positions_refuse_unknown_signal():
    with pytest.raises(SystemExit):
        worldmodel.positions_from_sides(range(3), [1, -1, 0], "worldmodel_wat")


def test_resolve_driver_set_refuses_unknown():
    with pytest.raises(SystemExit):
        worldmodel.resolve_driver_set("nope")


def test_new_driver_sets_resolve_and_are_de_collinearised():
    for name in ("gold_macro3b", "silver_macro3", "copper_macro3"):
        ds = worldmodel.resolve_driver_set(name)
        assert len(ds) == 3
    # single-driver set: composite is just that driver's side (long when the lone driver is bullish)
    solo = worldmodel.resolve_driver_set("gold_realrate1")
    assert len(solo) == 1
    sides = worldmodel.worldmodel_sides({"DFII10": [3.0, 3.0, 2.0, 5.0]}, solo, lookback=2)
    assert sides[2] == 1 and sides[3] == -1  # real rate falling -> long, rising -> short
    # gold_macro3b must NOT double-count DFII10 (the fair-test fix): each raw series appears at most once.
    used = [s for d in worldmodel.resolve_driver_set("gold_macro3b") for s in d["series"]]
    assert used.count("DFII10") == 1
    assert sorted(used) == sorted(set(used))  # no series repeated across legs


def test_level_only_driver_set_composite():
    ds = worldmodel.resolve_driver_set("gold_macro3b")  # three 'level' drivers, no spread
    vals = {
        "DFII10": [2.0, 2.0, 1.0],      # falling -> real_rate bull
        "DTWEXBGS": [100.0, 100.0, 98.0],  # falling -> usd bull
        "T10YIE": [2.2, 2.2, 2.5],      # rising -> breakeven bull
    }
    assert worldmodel.worldmodel_sides(vals, ds, lookback=2)[-1] == 1


# --- the run() contract ------------------------------------------------------------------------------


def _daily_bars(n=400):
    rng = np.random.default_rng(3)
    closes = 1500.0 * np.cumprod(1.0 + rng.normal(0.0003, 0.01, n))
    base = worldmodel._month_start_ms("2020-01")
    day = 86_400_000
    return {
        "timestamp": [base + i * day for i in range(n)],
        "open": list(closes), "high": list(closes * 1.005), "low": list(closes * 0.995),
        "close": list(closes), "volume": [1.0] * n,
    }


def test_run_emits_the_full_metric_vocabulary(monkeypatch):
    bars = _daily_bars(1100)  # span 2020-01 into the 2022 test window
    n = len(bars["timestamp"])
    # oscillating drivers so the composite flips and the book trades
    dfii = [2.0 + 0.5 * np.sin(i / 15.0) for i in range(n)]
    usd = [100.0 + 3.0 * np.sin(i / 15.0) for i in range(n)]
    dgs = [3.5 + 0.2 * np.cos(i / 15.0) for i in range(n)]
    monkeypatch.setattr(worldmodel, "_load_bars", lambda asset, pairs, bar_minutes: bars)
    series = {"DFII10": dfii, "DTWEXBGS": usd, "DGS10": dgs}
    monkeypatch.setattr(worldmodel, "_macro_asof_values", lambda sid, ts: series[sid])
    cfg = {
        "asset": "GOLD", "signal": "worldmodel", "driver_set": "gold_macro3", "lookback": 21,
        "transaction_fee": 0.0005, "walk_forward_window": "2022", "seed": 0,
    }
    summary = worldmodel.run(cfg)
    m = summary["metrics"]
    for key in ("total_return_pct", "oos_sharpe", "return_vs_hold_pct", "sharpe_vs_hold", "n_trades", "time_in_market_pct"):
        assert key in m, f"missing metric {key}"
        assert np.isfinite(m[key])
    assert summary["objective"] == m["oos_sharpe"]
    assert m["n_trades"] > 0
    assert summary["dataset"]["asset"] == "GOLD"


def test_run_refuses_unknown_signal(monkeypatch):
    monkeypatch.setattr(worldmodel, "_load_bars", lambda asset, pairs, bar_minutes: _daily_bars(60))
    monkeypatch.setattr(worldmodel, "_macro_asof_values", lambda sid, ts: [1.0] * 60)
    with pytest.raises(SystemExit):
        worldmodel.run({"signal": "worldmodel_wat", "walk_forward_window": "2022"})
