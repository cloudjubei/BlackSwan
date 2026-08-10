"""Direct gating tests for the funding-rate probe — the first NON-PRICE stream in the program.

Perp funding is a positioning/sentiment signal that is NOT in the spot price series: when funding is
extreme-positive longs are crowded (pay shorts), extreme-negative shorts are crowded. This probe asks
whether conditioning an 8h-cadence directional trade on a funding EXTREME (fade it — contrarian — or ride
it — momentum) beats buy-and-hold net of the ~0.2% round trip. It reuses the intraday backtest mechanics
verbatim (resample -> bars, hold-then-flat positions, both-way fees, per-trade edge), so the only new
surfaces are the funding join and the funding signal — and those carry the failure modes pinned here:

* The funding<->price JOIN is by TIMESTAMP, not index. Binance settles majors' funding every 8h on the
  00:00/08:00/16:00 UTC grid (the exact bucket a 480m resample lands on), but rows carry sub-second jitter
  and, under stress, Binance inserts OFF-CYCLE 4h fundings that sit MID-8h-bar. A decision taken at an 8h
  bar close knows only the funding that settled AT that close — the mid-bar 4h one is future-relative to the
  prior close and stale-relative to this one, so it must be DROPPED, never silently floored into a bucket.

* The "is this funding extreme?" threshold must be judged against the funding distribution known BEFORE the
  bar — the same past-only-quantile lookahead the regime gate guards, in a new place. A funding level that
  is extreme versus the WHOLE sample but ordinary versus its own past must NOT fire; a whole-sample quantile
  would fabricate the signal.

* A position decided from the funding that settled at bar t is applied to bars t+1.. — never bar t's own
  interval — so the whole position pipeline is time-prefix causal in funding: corrupt funding at/after a cut
  and every earlier position is byte-identical. Positions never read price at all (only the signal does),
  which is a stronger guarantee than the intraday breakout arm needs.
"""

import numpy as np
import pytest

from trainer import funding

W = funding.FUNDING_BUCKET_MS  # 8h in ms


# --- load_funding: tolerance join drops off-cycle, parses string rates -------------------------------


def _frow(ft, rate):
    return {"fundingTime": ft, "fundingRate": f"{rate}", "markPrice": ""}


def test_load_funding_joins_on_grid_and_drops_offcycle(tmp_path):
    import json
    # Three on-grid settlements (with the real ms jitter binance emits) plus one OFF-CYCLE 4h funding that
    # sits mid-8h-bar. The off-cycle row must be dropped — keeping it would either collide with a real
    # bucket or fabricate a funding value the 8h decision never actually observed.
    rows = [
        _frow(0 * W + 6, 0.0001),        # bucket 0, +6ms jitter
        _frow(1 * W - 12, 0.0002),       # bucket 1, -12ms jitter
        _frow(2 * W + 3, -0.0003),       # bucket 2 — the REAL 8h settlement
        _frow(2 * W - W // 2, 0.0099),   # OFF-CYCLE: 4h BEFORE bucket 2 (rounds to it) and LAST in the file,
                                         # so without the tolerance drop it would OVERWRITE bucket 2 with a
                                         # value the 8h decision never observed. Must be dropped.
    ]
    path = tmp_path / "X-funding.json"
    path.write_text(json.dumps(rows))
    fmap = funding.load_funding(str(path))
    assert set(fmap.keys()) == {0 * W, 1 * W, 2 * W}  # exactly the three on-grid buckets
    assert fmap[0 * W] == pytest.approx(0.0001)  # string rate parsed to float
    assert fmap[1 * W] == pytest.approx(0.0002)
    assert fmap[2 * W] == pytest.approx(-0.0003)  # the REAL settlement, NOT the mid-bar 0.0099 that follows it


def test_load_funding_missing_file_is_empty():
    assert funding.load_funding("/no/such/funding.json") == {}


# --- the funding signal: contrarian fades an extreme, momentum rides it ------------------------------


def _flat_then_spike(n_calm=40, spike=0.02, calm=0.0001):
    # A long calm funding history then one clearly-extreme value at the end, keyed to the 8h grid.
    ts = [i * W for i in range(n_calm + 1)]
    fmap = {i * W: calm for i in range(n_calm)}
    fmap[n_calm * W] = spike
    return ts, fmap


def test_contrarian_and_momentum_take_opposite_sides_of_an_extreme():
    ts, fmap = _flat_then_spike(spike=0.02)
    t = len(ts) - 1
    con = funding.funding_sides(ts, fmap, "funding_contrarian", funding_pct=0.90, min_history=10)
    mom = funding.funding_sides(ts, fmap, "funding_momentum", funding_pct=0.90, min_history=10)
    assert con[t] == -1.0  # crowded-long funding spike -> contrarian SHORTS it
    assert mom[t] == +1.0  # ...and momentum RIDES it long
    # A negative extreme flips both.
    ts2, fmap2 = _flat_then_spike(spike=-0.02)
    t2 = len(ts2) - 1
    assert funding.funding_sides(ts2, fmap2, "funding_contrarian", 0.90, 10)[t2] == +1.0
    assert funding.funding_sides(ts2, fmap2, "funding_momentum", 0.90, 10)[t2] == -1.0


def test_funding_side_is_flat_when_not_extreme():
    # A symmetric funding history (alternating +/-) then a probe bar sitting DEAD CENTRE: it is neither at
    # the high tail nor the low tail of its past, so neither arm fires on it (the tail bars around it may).
    ts = [i * W for i in range(31)]
    fmap = {i * W: (0.0002 if i % 2 else -0.0002) for i in range(30)}
    fmap[30 * W] = 0.0  # the probe: exactly mid-distribution
    sides = funding.funding_sides(ts, fmap, "funding_contrarian", funding_pct=0.90, min_history=10)
    assert sides[30] is None  # a mid value is strictly between the low and high quantiles -> flat


def test_funding_warmup_requires_min_history():
    # Even a colossal funding value cannot fire before min_history observations exist to judge it against.
    ts = [i * W for i in range(6)]
    fmap = {i * W: 0.0001 for i in range(5)}
    fmap[5 * W] = 0.05
    sides = funding.funding_sides(ts, fmap, "funding_contrarian", funding_pct=0.90, min_history=10)
    assert all(s is None for s in sides)


# --- past-only quantile: extreme-vs-sample but ordinary-vs-its-own-past must NOT fire ----------------


def _past_only_funding_fixture():
    # Mirror of the regime-vol lookahead fixture, in funding space: 40 calm, a 5-long burst, then 45 held
    # at a level ABOVE calm but well BELOW the burst. A held-level bar is top-decile versus the WHOLE
    # sample (the calm floor drags the 90th pctile down to the held level) yet unremarkable versus its OWN
    # past (the burst keeps the past 90th pctile far above it). Past-only must leave it flat.
    vals = [0.0001] * 40 + [0.02] * 5 + [0.002] * 45
    ts = [i * W for i in range(len(vals))]
    fmap = {i * W: v for i, v in enumerate(vals)}
    return ts, fmap


def test_funding_quantile_is_past_only_not_whole_sample():
    ts, fmap = _past_only_funding_fixture()
    t = 47  # a held-level (0.002) bar just after the burst — its past 90th pctile is still burst-dominated
    vals = [fmap[x] for x in ts]
    assert vals[t] >= np.quantile(vals, 0.90)          # extreme versus the FULL sample
    assert vals[t] < np.quantile(vals[:t], 0.90)       # ordinary versus its OWN past
    sides = funding.funding_sides(ts, fmap, "funding_contrarian", funding_pct=0.90, min_history=10)
    assert sides[t] is None  # past-only leaves it flat; a whole-sample quantile would fire a short here


def test_funding_quantile_still_fires_on_a_genuine_extreme():
    ts, fmap = _past_only_funding_fixture()
    sides = funding.funding_sides(ts, fmap, "funding_contrarian", funding_pct=0.90, min_history=10)
    # A bar inside the 0.02 burst towers over everything seen before it -> contrarian short (non-vacuous).
    assert sides[42] == -1.0


# --- join by timestamp: a bar with no matching funding bucket is flat --------------------------------


def test_bars_without_a_funding_bucket_stay_flat():
    # Enough calm history to warm up, an extreme at bar 20, but bar 20's TIMESTAMP is shifted off-grid so
    # no funding maps to it: the signal must be flat there (the join found nothing), not carry a stale value.
    ts = [i * W for i in range(30)]
    fmap = {i * W: 0.0001 for i in range(30)}
    fmap[20 * W] = 0.05
    on_grid = funding.funding_sides(ts, fmap, "funding_contrarian", 0.90, 10)
    assert on_grid[20] == -1.0  # sanity: on-grid it fires
    ts_offgrid = list(ts)
    ts_offgrid[20] = 20 * W + 12345  # move bar 20 off the funding grid
    off_grid = funding.funding_sides(ts_offgrid, fmap, "funding_contrarian", 0.90, 10)
    assert off_grid[20] is None  # no funding joined -> flat


# --- causality: positions are time-prefix causal in funding, and never read price --------------------


def _causal_funding():
    # A long calm base with three engineered extremes far enough apart to each open a real position.
    n = 60
    ts = [i * W for i in range(n)]
    fmap = {i * W: (0.0001 if i % 2 else -0.0001) for i in range(n)}
    for i in (18, 30, 44):
        fmap[i * W] = 0.03
    closes = list(100.0 * np.cumprod(1.0 + np.array([0.001 if i % 2 else -0.001 for i in range(n)])))
    return ts, closes, fmap


@pytest.mark.parametrize("signal", ["funding_contrarian", "funding_momentum"])
def test_positions_are_time_prefix_causal_in_funding(signal):
    ts, closes, fmap = _causal_funding()
    cfg = {"signal": signal, "funding_pct": 0.90, "hold_bars": 3, "min_history": 10}
    clean = funding.positions_from_bars(ts, closes, fmap, cfg)
    moved = False
    for cut in range(20, 55):
        dirty = dict(fmap)
        for i in range(cut, len(ts)):  # corrupt the funding at/after the cut
            dirty[ts[i]] = 0.05 * (1 if i % 2 else -1)
        dpos = funding.positions_from_bars(ts, closes, dirty, cfg)
        assert np.array_equal(clean[: cut + 1], dpos[: cut + 1])  # prefix decided before the cut is frozen
        moved = moved or not np.array_equal(clean[cut + 1 :], dpos[cut + 1 :])
    assert moved  # non-vacuous: corrupting future funding did move the later book


def test_positions_ignore_price_entirely():
    # Positions come from the funding signal alone; scaling every close must leave the book byte-identical.
    ts, closes, fmap = _causal_funding()
    cfg = {"signal": "funding_contrarian", "funding_pct": 0.90, "hold_bars": 3, "min_history": 10}
    base = funding.positions_from_bars(ts, closes, fmap, cfg)
    scaled = funding.positions_from_bars(ts, [c * 3.7 for c in closes], fmap, cfg)
    assert np.array_equal(base, scaled)


def test_signal_is_never_traded_on_its_own_bar_and_holds_hold_bars():
    ts, closes, fmap = _causal_funding()
    cfg = {"signal": "funding_contrarian", "funding_pct": 0.90, "hold_bars": 3, "min_history": 10}
    pos = funding.positions_from_bars(ts, closes, fmap, cfg)
    # The extreme at bar 18 is decided at bar 18 and applied to 19..21 (never bar 18 itself), held 3 bars.
    assert pos[18] == 0.0
    assert pos[19] == -1.0 and pos[20] == -1.0 and pos[21] == -1.0
    assert pos[22] == 0.0


# --- the run() contract: the full metric vocabulary the scorecard consumes ---------------------------


def _synthetic_bars(n=1200):
    # 8h bars across ~400 days with a mild upward drift, wrapped as the (year,month)->bars loader shape.
    rng = np.random.default_rng(11)
    rets = rng.normal(0.0002, 0.01, n)
    closes = 100.0 * np.cumprod(1.0 + rets)
    base = funding._month_start_ms("2024-01")
    return {
        "timestamp": [base + i * W for i in range(n)],
        "open": list(closes), "high": list(closes * 1.001), "low": list(closes * 0.999),
        "close": list(closes), "volume": [1.0] * n,
    }, base


def test_run_emits_the_full_metric_vocabulary(monkeypatch):
    bars, base = _synthetic_bars()
    # Funding with periodic extremes so the probe actually trades; keyed to the same 8h grid as the bars.
    rng = np.random.default_rng(3)
    vals = rng.normal(0.0, 0.0005, len(bars["timestamp"]))
    for i in range(0, len(vals), 13):
        vals[i] = 0.02 * (1 if (i // 13) % 2 else -1)
    fmap = {ts: float(v) for ts, v in zip(bars["timestamp"], vals)}
    monkeypatch.setattr(funding, "_load_bars", lambda asset, pairs, bar_minutes: bars)
    monkeypatch.setattr(funding, "load_funding", lambda path: fmap)
    cfg = {
        "asset": "BTCUSDT", "signal": "funding_contrarian", "funding_pct": 0.85,
        "hold_bars": 1, "transaction_fee": 0.001, "walk_forward_window": "2024", "seed": 0,
    }
    summary = funding.run(cfg)
    m = summary["metrics"]
    for key in (
        "total_return_pct", "oos_sharpe", "return_vs_hold_pct", "hold_return_pct",
        "trades_per_day", "realized_cost_bps", "signal_expectancy", "n_trades",
        "beta", "up_capture", "down_capture", "funding_coverage",
    ):
        assert key in m, f"missing metric {key}"
        assert np.isfinite(m[key])
    assert summary["objective"] == m["oos_sharpe"]
    assert m["n_trades"] > 0  # the funding extremes actually opened positions
    assert 0.0 <= m["funding_coverage"] <= 1.0
    assert "provenance" in summary and "config" in summary
    assert summary["dataset"]["timeframe"] == "480m"


def test_run_refuses_unknown_signal(monkeypatch):
    bars, _ = _synthetic_bars(50)
    monkeypatch.setattr(funding, "_load_bars", lambda asset, pairs, bar_minutes: bars)
    monkeypatch.setattr(funding, "load_funding", lambda path: {})
    with pytest.raises(SystemExit):
        funding.run({"signal": "funding_wat", "walk_forward_window": "2024"})
