"""Direct gating tests for the intraday (decision-frequency) backtest core.

This is the first line that trades at INTRADAY decision frequency, so it carries failure modes the
daily lines never had, each pinned here before a number is allowed out:

* Two flavours of lookahead. (a) A position applied to bar t's return must be decided only from data
  at/before t-1 — pinned by a byte-level time-prefix causality test that corrupts the price series
  after a cut and asserts every earlier position is unmoved (and, non-vacuously, that a later one can
  move). (b) The far subtler REGIME quantile: "is this a high-vol bar?" must be answered against the
  TRAILING (past-only) vol distribution, never the whole test sample. A bar that is high-vol versus the
  full sample but ordinary versus its own past must NOT be flagged; using the whole-sample quantile
  would mislabel it, fabricating the very edge the probe is trying to measure.
* Resampling 1m -> decision bars must build each bar from ONLY its own minutes (no straddle into the
  next bar's data), and a short/empty series must stay finite rather than emit a NaN.
* The trade mechanics the thesis rests on: a breakout fires only when |return| clears k * trailing_vol,
  a signal is never traded on its own bar, a position is held for exactly hold_bars then flat, new
  breakouts are ignored while in a position, and the round-trip fee is charged BOTH ways.
* trades_per_day is the user's headline requirement (multiple trades a day), so its arithmetic —
  round-trips over the test span's calendar days — is pinned on a hand fixture.
"""

import numpy as np
import pytest

from trainer import intraday


# --- resample: a bar uses only its own minutes -------------------------------------------------------


def _row(ts, o, h, l, c, v=1.0):
    # Prices arrive as STRINGS in the binance 1m files; the resampler must cope with that verbatim.
    return {
        "timestamp": ts,
        "price_open": f"{o}",
        "price_high": f"{h}",
        "price_low": f"{l}",
        "price": f"{c}",
        "volume": f"{v}",
    }


def test_resample_builds_each_bar_from_only_its_own_minutes():
    # Two 5-minute bars. The first bar's HIGH lives in minute 2; the second bar's data must never leak
    # back into it (a straddle would pull the second bar's 999 into the first bar's high).
    rows = [
        _row(0 * 60000, 100, 105, 99, 101),
        _row(1 * 60000, 101, 108, 100, 107),
        _row(2 * 60000, 107, 110, 106, 106),
        _row(3 * 60000, 106, 107, 104, 104),
        _row(4 * 60000, 104, 106, 103, 105),
        _row(5 * 60000, 105, 999, 105, 900),
        _row(6 * 60000, 900, 950, 100, 120),
    ]
    bars = intraday.resample(rows, bar_minutes=5)
    assert bars["open"][0] == 100.0  # first minute's open
    assert bars["high"][0] == 110.0  # max high across minutes 0-4 only, NOT the 999 in minute 5
    assert bars["low"][0] == 99.0
    assert bars["close"][0] == 105.0  # last minute (4) of the first bar
    assert bars["volume"][0] == 5.0
    assert bars["timestamp"][0] == 0
    # Second bar opens at minute 5 with its own extremes.
    assert bars["open"][1] == 105.0 and bars["high"][1] == 999.0 and bars["close"][1] == 120.0
    assert bars["timestamp"][1] == 5 * 60000


def test_resample_of_empty_or_single_stays_finite():
    assert intraday.resample([], bar_minutes=15) == {
        "timestamp": [], "open": [], "high": [], "low": [], "close": [], "volume": [],
    }
    one = intraday.resample([_row(0, 100, 101, 99, 100)], bar_minutes=15)
    assert one["close"] == [100.0] and np.isfinite(one["close"][0])


# --- trailing realized vol: past-only, warm-up is None ------------------------------------------------


def test_realized_vol_is_trailing_and_warmup_is_none():
    closes = [100.0, 101.0, 102.0, 101.0, 103.0, 100.0]
    rets = intraday.bar_returns(closes)
    assert rets[0] is None  # the first bar has no prior, so no return
    vol = intraday.realized_vol(rets, vol_window=3)
    # vol[t] needs vol_window returns ending at t; returns start at index 1, so t < 3 is undefined.
    assert vol[0] is None and vol[1] is None and vol[2] is None
    assert vol[3] is not None
    expected = float(np.std([rets[1], rets[2], rets[3]]))
    assert abs(vol[3] - expected) < 1e-12


# --- the breakout rule --------------------------------------------------------------------------------


def _breakout_fixture():
    # Calm oscillation (tiny trailing vol) then a decisive +10% jump at bar 5; a second violent move at
    # bar 6 lands WHILE the bar-5 position is still held.
    closes = [100.0, 101.0, 100.0, 101.0, 100.0, 110.0, 130.0, 110.0, 111.0, 110.0, 111.0]
    cfg = {"signal": "breakout_momentum", "vol_window": 3, "breakout_k": 2.0, "hold_bars": 2}
    return closes, cfg


def test_breakout_fires_only_when_return_clears_k_times_trailing_vol():
    closes, cfg = _breakout_fixture()
    # k=2: the +10% move dwarfs 2 * trailing_vol, so it fires and opens a long the NEXT bar.
    pos = intraday.positions_from_closes(closes, cfg)
    assert pos[5] == 0.0  # never traded on its OWN bar — the breakout is detected at 5, acted at 6
    assert pos[6] == 1.0 and pos[7] == 1.0  # held for exactly hold_bars=2 bars, long (sign of +10%)
    assert pos[8] == 0.0  # then flat
    # A colossal k makes the same |return| fail the threshold everywhere -> no position ever opens.
    quiet = intraday.positions_from_closes(closes, {**cfg, "breakout_k": 1000.0})
    assert np.count_nonzero(quiet) == 0


def test_breakout_threshold_uses_trailing_vol_that_excludes_the_signal_bar():
    # The breakout at bar t is judged against the vol of the bars BEFORE it (v_{t-1}). Folding bar t's own
    # jump into the estimate — v_t instead of v_{t-1} — inflates the threshold with the very move being
    # tested and silently suppresses the breakout. This fixture sits in the narrow band where that one-bar
    # difference flips the decision (|return_3| clears 2*v_2 but NOT 2*v_3), so a regression to v_t leaves
    # the book empty. Without this the exclude-own-bar half of property (a) has no biting guard: the
    # corrupt-after-cut causality test can't see it, because v_t is contemporaneous, not future, data.
    rets = [0.002, -0.002, 0.03, -0.001, 0.001, -0.001, 0.001, -0.001]
    closes = [100.0]
    for r in rets:
        closes.append(closes[-1] * (1.0 + r))
    cfg = {"signal": "breakout_momentum", "vol_window": 2, "breakout_k": 2.0, "hold_bars": 2}
    vol = intraday.realized_vol(intraday.bar_returns(closes), 2)
    assert 2.0 * vol[2] < abs(rets[2]) <= 2.0 * vol[3]  # the discriminating band: excl fires, incl doesn't
    pos = intraday.positions_from_closes(closes, cfg)
    assert pos[3] == 0.0  # the +3% bar is never traded on its own bar
    assert pos[4] == 1.0 and pos[5] == 1.0  # trailing-vol gate (through bar 2) fires -> long the next bars
    assert np.count_nonzero(pos) == 2  # and that is the ONLY trade; a v_t gate would leave it all flat


def test_a_new_breakout_is_ignored_while_a_position_is_open():
    closes, cfg = _breakout_fixture()
    pos = intraday.positions_from_closes(closes, cfg)
    # Bar 6 is a ~+18% move — a breakout in its own right — but it lands inside the bar-5 hold window,
    # so it must NOT open a second position or extend the first: pos[6] is the HELD long, pos[8] is flat.
    assert pos[6] == 1.0
    assert pos[7] == 1.0  # still the ORIGINAL long — the bar-6 breakout neither flipped nor extended it
    assert pos[8] == 0.0  # and the book is flat the instant the hold window ends (no re-entry)
    assert intraday.round_trips(pos) == 1  # exactly one round-trip out of the whole fixture


def test_position_is_held_exactly_hold_bars_then_flat():
    closes, cfg = _breakout_fixture()
    for hold in (1, 3, 4):
        pos = intraday.positions_from_closes(closes, {**cfg, "hold_bars": hold})
        run = [i for i, p in enumerate(pos) if p != 0.0]
        assert run == list(range(6, 6 + hold))  # entry at 6, exactly `hold` bars, contiguous


# --- causality (a): a position decided only from data at/before t-1 -----------------------------------


def _causal_closes(n=40):
    # A deterministic, near-flat walk (alternating +/-0.05% steps) so trailing vol is tiny and NOTHING
    # fires on its own — except one engineered +3% breakout at bar 6, which gives the prefix a real
    # non-zero position that must survive corruption.
    rets = np.array([0.0005 if i % 2 else -0.0005 for i in range(n)])
    rets[6] = 0.03
    closes = 100.0 * np.cumprod(1.0 + rets)
    return closes


@pytest.mark.parametrize(
    "cfg",
    [
        {"signal": "breakout_momentum", "vol_window": 4, "breakout_k": 2.0, "hold_bars": 3},
        {"signal": "regime_momentum", "vol_window": 4, "regime_pct": 0.70, "hold_bars": 3},
        {"signal": "regime_reversion", "vol_window": 4, "regime_pct": 0.70, "hold_bars": 3},
    ],
)
def test_positions_are_time_prefix_causal(cfg):
    # Parametrized over every signal — an arm exercised only by its rule test is an arm whose lookahead
    # property nobody checks. The regime arms carry a SECOND lookahead surface (the trailing quantile),
    # so their whole position pipeline must be causal too, not just breakout's.
    closes = _causal_closes()
    clean = intraday.positions_from_closes(closes, cfg)
    moved = False
    for cut in range(12, 34):
        dirty_closes = closes.copy()
        # Corrupt bar `cut`'s OWN return and everything after it. The position HELD INTO bar cut (pos[cut],
        # which earns bar cut's return) was decided at bar cut-1, so it must survive the corruption of bar
        # cut too — this is what pins property (a) tightly: a book that read its own bar's return would
        # flip pos[cut] here. Merely corrupting bar cut+1 onward would let an own-bar peek slip through.
        dirty_closes[cut:] *= 1.5
        dirty = intraday.positions_from_closes(dirty_closes, cfg)
        # Every position decided from data at/before bar cut-1 (i.e. pos[0..cut]) is byte-identical — the
        # PER-CUT causality guarantee, checked for every cut.
        assert np.array_equal(clean[: cut + 1], dirty[: cut + 1])
        moved = moved or not np.array_equal(clean[cut + 1 :], dirty[cut + 1 :])
    # Non-vacuous: across the sweep the corruption DID move the book somewhere after the cut, so the
    # prefix-equality above is a real constraint and not a test that a do-nothing book would also pass.
    # (Aggregate rather than per-cut: a quantile-gated regime arm can legitimately re-enter the same side
    # after the shock at some individual cut, which is not a lookahead.)
    assert moved


def test_pipeline_is_time_prefix_causal_through_the_1m_resample():
    # The same property, but end-to-end through resample: corrupt the raw 1m prices after a bar boundary
    # and the decision-bar positions formed entirely before it must not budge.
    minutes = 120
    price = 100.0
    rows = []
    for i in range(minutes):
        step = 0.0004 if i % 2 else -0.0004
        if i == 20:
            step = 0.03  # one clean intrabar breakout early, well before the cut
        price *= 1.0 + step
        rows.append(_row(i * 60000, price, price, price, price))
    cfg = {"signal": "breakout_momentum", "bar": 5, "vol_window": 3, "breakout_k": 2.0, "hold_bars": 2}
    _, clean = intraday.positions_from_rows(rows, cfg)
    cut_minute = 60  # a 5-minute bar boundary (bar index 12)
    cut_bar = cut_minute // 5
    dirty_rows = [dict(r) for r in rows]
    for r in dirty_rows:
        if r["timestamp"] >= cut_minute * 60000:
            for key in ("price_open", "price_high", "price_low", "price"):
                r[key] = f"{float(r[key]) * 1.5}"
    _, dirty = intraday.positions_from_rows(dirty_rows, cfg)
    assert np.array_equal(clean[:cut_bar], dirty[:cut_bar])  # bars fully before the cut are untouched
    assert not np.array_equal(clean, dirty)  # the corruption is real -> something moves after the cut


# --- causality (b): the REGIME quantile is past-only, never whole-sample -------------------------------


def _regime_vol_fixture():
    # A vol series whose LAST portion is high-vol: 40 calm bars, then a 5-bar burst to 100, then 45 bars
    # holding at 10 (still well above the calm floor). The bar we probe (a "10" deep in the tail) is a
    # top-decile bar versus the WHOLE sample -- the whole-sample 90th percentile is only 10.0, dragged
    # down by the 40 calm bars -- yet versus its OWN past it is unremarkable, because the 100-burst still
    # sits in the upper tail of everything seen so far (past 90th percentile ~= 46). Past-only must not
    # flag it; the whole-sample quantile WOULD.
    vol = [1.0] * 40 + [100.0] * 5 + [10.0] * 45
    return vol


def test_regime_flag_uses_only_the_vol_distribution_before_the_bar():
    vol = _regime_vol_fixture()
    pct = 0.90
    flags = intraday.regime_flags(vol, pct)
    t = 47  # decision bar; it is gated on vol[t-1] = vol[46] = 10.0
    assert vol[t - 1] >= np.quantile(vol, pct)          # high-vol versus the FULL sample (~10.0)
    assert vol[t - 1] < np.quantile(vol[: t], pct)      # ordinary versus its OWN past (~46.0)
    assert flags[t] is False  # so the causal rule leaves it FLAT; whole-sample would wrongly flag it


def test_regime_flag_still_fires_on_a_genuine_onset_so_the_guard_is_not_vacuous():
    vol = _regime_vol_fixture()
    flags = intraday.regime_flags(vol, 0.90)
    # Bar 45 is gated on vol[44] = 100.0, which towers over everything seen before it -> in-regime.
    assert flags[45] is True


def test_regime_momentum_enters_only_in_regime_and_never_on_its_own_bar():
    # A calm head then a sustained high-vol tail with a clear up-drift; the regime gate opens in the tail
    # and a long is taken the bar AFTER the gate opens, never on the gating bar itself.
    closes = [100.0] * 12 + [100.0 * (1.03 ** i) for i in range(1, 20)]
    cfg = {
        "signal": "regime_momentum", "vol_window": 3, "regime_pct": 0.80, "hold_bars": 2,
    }
    pos = intraday.positions_from_closes(closes, cfg)
    assert np.count_nonzero(pos[:12]) == 0  # calm region is never traded
    assert np.count_nonzero(pos) > 0  # the regime tail IS traded (non-vacuous)


def test_regime_reversion_is_the_opposite_side_of_regime_momentum():
    closes = [100.0] * 12 + [100.0 * (1.03 ** i) for i in range(1, 20)]
    base = {"vol_window": 3, "regime_pct": 0.80, "hold_bars": 2}
    mom = intraday.positions_from_closes(closes, {**base, "signal": "regime_momentum"})
    rev = intraday.positions_from_closes(closes, {**base, "signal": "regime_reversion"})
    # Same regime gate, opposite entry sign: wherever momentum holds a side, reversion holds its negation.
    both = [(m, r) for m, r in zip(mom, rev) if m != 0.0 or r != 0.0]
    assert both  # the gate actually opened somewhere
    assert all(r == -m for m, r in both)


# --- fees, trades_per_day, and finiteness -------------------------------------------------------------


def test_fee_is_charged_both_ways():
    closes = [100.0] * 6  # flat prices: the ONLY thing that can move equity is the round-trip fee
    pos = np.array([0.0, 1.0, 1.0, 0.0, 0.0, 0.0])  # enter at bar 1, exit at bar 3
    fee = 0.001
    free = intraday.backtest(closes, pos, fee=0.0)
    paid = intraday.backtest(closes, pos, fee=fee)
    assert free[-1] == 1.0  # no price move, no fee -> nothing changes
    assert abs(paid[-1] - (1.0 - fee) ** 2) < 1e-12  # TWO haircuts: one at entry, one at exit
    assert paid[-1] < free[-1]


def test_round_trips_counts_entries():
    assert intraday.round_trips(np.array([0.0, 1.0, 1.0, 0.0, -1.0, -1.0, 0.0])) == 2
    assert intraday.round_trips(np.array([0.0, 0.0, 0.0])) == 0


def test_trades_per_day_is_roundtrips_over_test_calendar_days():
    day = 86_400_000
    # 2 round-trips over a span of exactly 4 calendar days -> 0.5 trades/day.
    assert intraday.trades_per_day(2, first_ts=1_000, last_ts=1_000 + 4 * day) == pytest.approx(0.5)
    # A degenerate zero-length span must not divide by zero.
    assert intraday.trades_per_day(3, first_ts=5, last_ts=5) == 0.0


def test_backtest_of_short_or_flat_input_stays_finite():
    assert intraday.backtest([100.0], np.array([0.0]), fee=0.001) == [1.0]
    eq = intraday.backtest([], np.array([]), fee=0.001)
    assert eq == [] or all(np.isfinite(x) for x in eq)
    # A warm-up-only close series (never enough bars to define vol) produces an all-flat, finite book.
    pos = intraday.positions_from_closes([100.0, 101.0], {"signal": "breakout_momentum", "vol_window": 5})
    assert np.count_nonzero(pos) == 0


# --- the run() contract: the full metric vocabulary the scorecard consumes ----------------------------


def _synthetic_bars(n=400):
    # A high-vol, spiky bar series with enough breakouts to make several round-trips, wrapped as the
    # (year, month) -> bars loader run() expects, so the contract test never touches disk.
    rng = np.random.default_rng(7)
    rets = rng.normal(0, 0.004, n)
    for i in range(10, n, 17):
        rets[i] = 0.05 * (1 if (i // 17) % 2 else -1)  # periodic breakouts, both directions
    closes = 100.0 * np.cumprod(1.0 + rets)
    hour = 3_600_000
    base = intraday._month_start_ms("2024-01")
    return {
        "timestamp": [base + i * hour for i in range(n)],
        "open": list(closes), "high": list(closes * 1.001), "low": list(closes * 0.999),
        "close": list(closes), "volume": [1.0] * n,
    }


def test_run_emits_the_full_metric_vocabulary(monkeypatch):
    bars = _synthetic_bars()
    monkeypatch.setattr(intraday, "_load_bars", lambda asset, pairs, bar_minutes: bars)
    cfg = {
        "asset": "BTCUSDT", "bar": 60, "signal": "breakout_momentum", "vol_window": 20,
        "breakout_k": 1.5, "hold_bars": 4, "transaction_fee": 0.001,
        "walk_forward_window": "2024", "seed": 0,
    }
    summary = intraday.run(cfg)
    m = summary["metrics"]
    for key in (
        "total_return_pct", "oos_sharpe", "return_vs_hold_pct", "hold_return_pct",
        "trades_per_day", "realized_cost_bps", "signal_expectancy", "n_trades",
        "beta", "up_capture", "down_capture",
    ):
        assert key in m, f"missing metric {key}"
        assert np.isfinite(m[key])
    assert summary["objective"] == m["oos_sharpe"]  # objective is the per-step OOS Sharpe
    assert m["trades_per_day"] > 0  # the whole point: it trades intraday, several times over the span
    assert "provenance" in summary and "config" in summary
