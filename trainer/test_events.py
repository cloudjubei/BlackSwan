"""Direct gating tests for the macro-event probe — event-CONDITIONED trading (the B2 line).

Unconditional intraday direction is null (per the intraday probe). This line asks a narrower, exogenous
question: in the window right after a SCHEDULED macro release (CPI, jobs, retail, PCE, GDP), does the crypto
reaction PERSIST (drift) or REVERSE (fade) far enough to clear cost? The release is genuinely exogenous
information; the only signal is the first post-release bar's move, ridden or faded. It reuses the shared
point-in-time release timing (pit_fusion) and the intraday backtest mechanics, so the surfaces new here are
the event timing and the reaction gate, each pinned before a number is emitted:

* an event fires at the series' real RELEASE datetime (releaseDate + the series' publish wall-clock, DST-aware),
  NEVER the reference period it describes — stamping at refPeriod is the classic macro look-ahead (January's
  number "known" all January). Events outside the accounted span are dropped, and coincident releases across
  series (CPI NSA + SA drop together) collapse to ONE event.

* the reaction bar is the FIRST decision bar at/after the release; its return is known at that bar's close and
  the position it implies is applied to the NEXT bars, never the reaction bar's own return — so the pipeline
  is time-prefix causal, and event times (fixed by the macro calendar) never depend on price.
"""

import json

import numpy as np
import pytest

from trainer import events
from trainer.pit_fusion import publish_time_for, release_datetime_ms


# --- event timing: release-based, DST-aware, window-filtered, refPeriod ignored ----------------------


def _obs(ref, rel, val):
    return {"refPeriod": ref, "releaseDate": rel, "value": val, "vintage": rel}


def test_event_ms_uses_release_datetime_not_refperiod():
    # Two releases of a default-08:30-ET series: a summer one (EDT, UTC-4 -> 12:30 UTC) and a winter one
    # (EST, UTC-5 -> 13:30 UTC). The refPeriod is deliberately a month before the release; the event instant
    # must depend ONLY on releaseDate + publish time, so a refPeriod-based stamp would land on the wrong day.
    obs = [_obs("2024-06-01", "2024-07-11", 300.0), _obs("2023-12-01", "2024-01-11", 290.0)]
    lo, hi = release_datetime_ms("2024-01-01"), release_datetime_ms("2024-12-31")
    got = events.event_ms_from_observations(obs, "CPIAUCNS", lo, hi)
    jul = release_datetime_ms("2024-07-11", publish_time_for("CPIAUCNS"))
    jan = release_datetime_ms("2024-01-11", publish_time_for("CPIAUCNS"))
    assert set(got) == {jul, jan}  # both instants, depending ONLY on releaseDate (refPeriod is a month earlier)
    # summer release is 12:30 UTC (EDT, UTC-4), winter is 13:30 UTC (EST, UTC-5) — the DST-correct instants.
    import datetime
    summer = datetime.datetime.utcfromtimestamp(jul / 1000)
    winter = datetime.datetime.utcfromtimestamp(jan / 1000)
    assert (summer.hour, summer.minute) == (12, 30)
    assert (winter.hour, winter.minute) == (13, 30)


def test_event_ms_filters_to_the_window():
    obs = [_obs("2024-06-01", "2024-07-11", 1.0), _obs("2025-06-01", "2025-07-11", 2.0)]
    lo, hi = release_datetime_ms("2024-01-01"), release_datetime_ms("2024-12-31")
    got = events.event_ms_from_observations(obs, "CPIAUCNS", lo, hi)
    assert got == [release_datetime_ms("2024-07-11", "08:30")]  # the 2025 release is outside [lo, hi]


def test_load_event_ms_dedupes_coincident_releases(tmp_path, monkeypatch):
    # Two series that release at the SAME instant (CPI NSA + SA) must collapse to ONE event.
    (tmp_path / "CPIAUCNS.json").write_text(json.dumps([_obs("2024-06-01", "2024-07-11", 1.0)]))
    (tmp_path / "CPIAUCSL.json").write_text(json.dumps([_obs("2024-06-01", "2024-07-11", 2.0)]))
    monkeypatch.setattr(events, "MACRO_DIR", str(tmp_path))
    lo, hi = release_datetime_ms("2024-01-01"), release_datetime_ms("2024-12-31")
    got = events.load_event_ms("cpi", lo, hi)
    assert got == [release_datetime_ms("2024-07-11", "08:30")]  # one instant, not two


def test_load_event_ms_refuses_unknown_group(tmp_path, monkeypatch):
    monkeypatch.setattr(events, "MACRO_DIR", str(tmp_path))
    with pytest.raises(SystemExit):
        events.load_event_ms("not_a_group", 0, 10**13)


def test_event_groups_exclude_daily_rate_series():
    # A daily series (a value every calendar day) is NOT a discrete market event; it must not sit in any group,
    # or every day would fire. Guards against the DFEDTARU-is-daily trap.
    for daily in ("DFEDTARU", "DFF", "DGS10", "T10Y2Y", "DFII10"):
        for series in events.EVENT_GROUPS.values():
            assert daily not in series


# --- reaction gate: first bar at/after the release, ride vs fade -------------------------------------


def _bars(n=40, bar_ms=3_600_000, start="2024-01-02"):
    base = events._month_start_ms(start[:7])
    ts = [base + i * bar_ms for i in range(n)]
    closes = list(100.0 * np.cumprod(1.0 + np.array([0.001 if i % 2 else -0.001 for i in range(n)])))
    return ts, closes


def test_reaction_fires_on_first_bar_at_or_after_release_and_never_on_its_own_bar():
    ts, closes = _bars()
    # An event landing between bar 9 and bar 10: the reaction bar is bar 10 (first ts >= event), and the
    # position is applied to bar 11.. never bar 10's own return.
    event_ms = ts[10] - 1
    sides = events.reaction_sides(ts, closes, [event_ms], "event_drift")
    assert sides[10] is not None and all(sides[i] is None for i in range(10))
    returns = events.bar_returns(closes)
    assert sides[10] == events._sign(returns[10])  # drift rides the reaction bar's move
    fade = events.reaction_sides(ts, closes, [event_ms], "event_fade")
    assert fade[10] == -sides[10]  # fade is the exact negation


def test_reaction_positions_apply_to_next_bars_and_hold():
    ts, closes = _bars()
    cfg = {"signal": "event_drift", "hold_bars": 3}
    pos = events.positions_from_events(ts, closes, [ts[10] - 1], cfg)
    assert pos[10] == 0.0  # reaction detected at 10, never traded on its own bar
    side = events.reaction_sides(ts, closes, [ts[10] - 1], "event_drift")[10]
    assert pos[11] == side and pos[12] == side and pos[13] == side and pos[14] == 0.0


def test_no_event_no_position():
    ts, closes = _bars()
    pos = events.positions_from_events(ts, closes, [], {"signal": "event_drift", "hold_bars": 3})
    assert np.count_nonzero(pos) == 0


def test_reaction_sides_refuses_unknown_signal():
    ts, closes = _bars()
    with pytest.raises(SystemExit):
        events.reaction_sides(ts, closes, [ts[5]], "event_wat")


# --- causality: positions are time-prefix causal; event times never depend on price ------------------


@pytest.mark.parametrize("signal", ["event_drift", "event_fade"])
def test_positions_are_time_prefix_causal(signal):
    ts, closes = _bars(60)
    event_ms = [ts[15] - 1, ts[30] - 1, ts[44] - 1]  # three events, spaced so each opens a position
    cfg = {"signal": signal, "hold_bars": 3}
    clean = events.positions_from_events(ts, closes, event_ms, cfg)
    moved = False
    for cut in range(18, 55):
        dirty = list(closes)
        for i in range(cut, len(closes)):
            dirty[i] = closes[i] * 1.5  # corrupt price at/after the cut
        dpos = events.positions_from_events(ts, dirty, event_ms, cfg)
        assert np.array_equal(clean[: cut + 1], dpos[: cut + 1])  # prefix decided before the cut is frozen
        moved = moved or not np.array_equal(clean[cut + 1 :], dpos[cut + 1 :])
    assert moved  # non-vacuous


def test_event_times_do_not_depend_on_price():
    # The SAME events on scaled prices produce reaction bars at the SAME indices (only the side can change).
    ts, closes = _bars(60)
    ev = [ts[20] - 1]
    a = events.reaction_sides(ts, closes, ev, "event_drift")
    b = events.reaction_sides(ts, [c * 2.0 for c in closes], ev, "event_drift")
    assert [i for i, s in enumerate(a) if s is not None] == [i for i, s in enumerate(b) if s is not None]


# --- the run() contract: the full metric vocabulary the scorecard consumes ---------------------------


def _synthetic_bars(n=1200, bar_ms=3_600_000):
    rng = np.random.default_rng(9)
    closes = 100.0 * np.cumprod(1.0 + rng.normal(0.0002, 0.01, n))
    base = events._month_start_ms("2024-01")
    return {
        "timestamp": [base + i * bar_ms for i in range(n)],
        "open": list(closes), "high": list(closes * 1.001), "low": list(closes * 0.999),
        "close": list(closes), "volume": [1.0] * n,
    }


def test_run_emits_the_full_metric_vocabulary(monkeypatch):
    bars = _synthetic_bars()
    monkeypatch.setattr(events, "_load_bars", lambda asset, pairs, bar_minutes: bars)
    # ~40 synthetic events sprinkled across the span so the probe actually trades.
    ev = [bars["timestamp"][i] - 1 for i in range(30, 1200, 30)]
    monkeypatch.setattr(events, "load_event_ms", lambda group, lo, hi: [e for e in ev if lo <= e <= hi])
    cfg = {
        "asset": "BTCUSDT", "bar": 60, "signal": "event_drift", "event_group": "macro_all",
        "hold_bars": 4, "transaction_fee": 0.001, "walk_forward_window": "2024", "seed": 0,
    }
    summary = events.run(cfg)
    m = summary["metrics"]
    for key in (
        "total_return_pct", "oos_sharpe", "return_vs_hold_pct", "hold_return_pct",
        "trades_per_day", "realized_cost_bps", "signal_expectancy", "n_trades", "n_events",
        "beta", "up_capture", "down_capture",
    ):
        assert key in m, f"missing metric {key}"
        assert np.isfinite(m[key])
    assert summary["objective"] == m["oos_sharpe"]
    assert m["n_trades"] > 0 and m["n_events"] > 0
    assert summary["dataset"]["timeframe"] == "60m"
    assert "provenance" in summary and "config" in summary


def test_run_refuses_unknown_signal(monkeypatch):
    monkeypatch.setattr(events, "_load_bars", lambda asset, pairs, bar_minutes: _synthetic_bars(50))
    monkeypatch.setattr(events, "load_event_ms", lambda group, lo, hi: [])
    with pytest.raises(SystemExit):
        events.run({"signal": "event_wat", "bar": 60, "walk_forward_window": "2024"})
