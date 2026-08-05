"""Compute the trainer-standard RunSummary for one BlackSwan trading test run.

BlackSwan trades a FIXED stake (the account is reset to ``initial_balance`` after every close,
so a losing streak can't compound to ruin), so the honest performance figure is the SUM of
per-trade P&L, NOT the endpoint of the raw ``net_worths`` curve (which sawtooths back to the
initial balance after each trade and would only reflect the last/open position). This module
reconstructs the round-trips from the env's retained per-step arrays (read-side, no env change),
derives a REAL equity curve (initial + cumulative realized + current unrealized), and reports the
behavioural breakdown that makes a run explainable: exit reasons, per-regime performance, and a
trade ledger. Buy-and-hold is kept only as a display yardstick, never an objective.
"""

import bisect
import datetime
import math

import numpy as np

from trainer.fidelity import resolve_fidelity
from trainer.sharpe import sharpe_stats

_MAX_SERIES_POINTS = 200

# Trades a run must make to earn full credit for its return — the "trade often" bar. Below it the
# objective is QUADRATICALLY gated toward 0 (gate = (n_trades/MIN)**2), so a near-buy-and-hold run
# (e.g. 1 trade) scores ~0 no matter how far the underlying price moved, and under-trading is punished
# steeply. The aim is to trade OFTEN and WELL; a single trade is ~holding, not a strategy. Tune to the
# test-window length.
MIN_TRADES_FOR_FULL_CREDIT = 20
# At or below this many trades a run is effectively buy-and-hold, flagged degenerate (RL runs only —
# the hodl baseline trades once by design).
DEGENERATE_TRADE_COUNT = 2
# Trailing return over this fraction of the test window classifies a bar's regime as up/down (else
# flat) for the trend-based regime split. The band keeps a sideways drift from reading as a trend.
_REGIME_TREND_BAND = 0.01
_REGIME_TREND_WINDOW_DIVISOR = 50


def _finite(x, default=0.0):
    return float(x) if isinstance(x, (int, float)) and math.isfinite(x) else default


def _action_int(a):
    try:
        return int(np.asarray(a).reshape(-1)[0])
    except Exception:
        try:
            return int(a)
        except Exception:
            return 0


def _downsample(values, cap=_MAX_SERIES_POINTS):
    vals = [_finite(v) for v in values]
    if len(vals) <= cap:
        return vals
    stride = (len(vals) - 1) / (cap - 1)
    idx = sorted({round(i * stride) for i in range(cap)} | {len(vals) - 1})
    return [vals[i] for i in idx]


def _live(seq, lookback):
    values = list(seq)
    return values[lookback:] if len(values) > lookback else values


def _downsample_indexed(values, cap=_MAX_SERIES_POINTS):
    vals = [_finite(v) for v in values]
    if len(vals) <= cap:
        return vals, list(range(len(vals)))
    stride = (len(vals) - 1) / (cap - 1)
    idx = sorted({round(i * stride) for i in range(cap)} | {len(vals) - 1})
    return [vals[i] for i in idx], idx


def _marker_x(original_index, kept_indices):
    pos = bisect.bisect_left(kept_indices, original_index)
    if pos >= len(kept_indices):
        return len(kept_indices) - 1
    if pos > 0 and (kept_indices[pos] - original_index) > (original_index - kept_indices[pos - 1]):
        return pos - 1
    return pos


def _run_prices(env, n):
    provider = getattr(env, "data_provider", None)
    if provider is None:
        return []
    get_price = getattr(provider, "get_price", None)
    if callable(get_price):
        out = []
        for i in range(n):
            try:
                out.append(_finite(get_price(i)))
            except Exception:
                break
        if len(out) >= 2:
            return out
    prices = getattr(provider, "prices", None)
    if isinstance(prices, (list, tuple, np.ndarray)) and len(prices) >= 2:
        return [_finite(p) for p in list(prices)[:n]]
    return []


def _run_chart(env, lookback, trades):
    """A JSON-renderable price line + trade markers for the hub viewer, built from the SAME
    reconstructed round-trips as the ledger so the two never disagree.

    Marker types distinguish EXECUTED trades from ATTEMPTED (no-op) agent requests, and by side:
    executed opens ``buy`` (long) / ``short``; executed agent closes ``sell`` (long) / ``cover``
    (short); auto closes ``tp`` / ``trailing`` / ``sl``; no-op requests ``*_attempt``. ``counts`` is
    AUTHORITATIVE — tallied over the full series before the draw-only downsample dedup — so the legend
    can't under-count (the old chart collapsed e.g. 28 stop-losses to 21 on the 200-point grid)."""
    actions = [_action_int(a) for a in _live(getattr(env, "actions", []), lookback)]
    if len(actions) < 2:
        return None
    prices = _run_prices(env, len(actions))
    if len(prices) < 2:
        return None
    n = min(len(actions), len(prices))
    actions = actions[:n]
    prices = prices[:n]
    made = list(_live(getattr(env, "actions_made", []), lookback))[:n]

    ds_price, kept = _downsample_indexed(prices)
    open_type = {"long": "buy", "short": "short"}
    attempt_type = {1: "buy_attempt", 2: "sell_attempt", 3: "short_attempt", 4: "cover_attempt"}
    raw = []
    for t in trades:
        ei, xi = int(t["entry_step"]), int(t["exit_step"])
        if 0 <= ei < n:
            raw.append((ei, open_type.get(t["side"], "buy"), prices[ei]))
        if t["reason"] != "open" and 0 <= xi < n:
            raw.append((xi, t["reason"], prices[xi]))
    for i in range(n):
        if i < len(made) and not made[i] and actions[i] in attempt_type:
            raw.append((i, attempt_type[actions[i]], prices[i]))

    counts = {}
    for _, typ, _ in raw:
        counts[typ] = counts.get(typ, 0) + 1

    seen = set()
    markers = []
    for orig_i, typ, price in raw:
        x = _marker_x(orig_i, kept)
        if (x, typ) in seen:
            continue
        seen.add((x, typ))
        markers.append({"i": x, "type": typ, "price": _finite(price)})
    return {"price": ds_price, "markers": markers, "counts": counts}


def _signal_noise(env, lookback):
    """Out-of-position / redundant SIGNAL diagnostic: how much of the agent's non-hold output was a
    no-op — a buy emitted while already long, a sell emitted while flat — that ``take_action`` refused,
    so it never moved the book. A high ``blocked_signal_ratio`` means the raw action stream is unusable
    as a standalone signal even when the EXECUTED trades are good (the model spams entries/exits it
    cannot take). 'Executed' = the agent's own action moved the position (made AND not a forced TP/SL),
    matching ``combo_noop_penalty``'s no-op definition so the metric measures exactly what that lever
    penalizes."""
    actions = [_action_int(a) for a in _live(getattr(env, "actions", []), lookback)]
    if not actions:
        return None
    made = list(_live(getattr(env, "actions_made", []), lookback))
    forced = [_action_int(f) for f in _live(getattr(env, "forced_actions", []), lookback)]
    blocked = 0
    executed = 0
    for i, a in enumerate(actions):
        if a == 0:
            continue
        agent_executed = (i < len(made) and bool(made[i])) and (forced[i] if i < len(forced) else 0) == 0
        if agent_executed:
            executed += 1
        else:
            blocked += 1
    emitted = blocked + executed
    return {
        "blocked_signals": blocked,
        "executed_signals": executed,
        "blocked_signal_ratio": (blocked / emitted) if emitted else 0.0,
        "signal_noise_pct": 100.0 * blocked / len(actions),
    }


def _signal_expectancy(trades, prices, horizon):
    """Case-1 SIGNAL lens — score each AGENT buy/sell by the realised forward return over ``horizon`` bars,
    INDEPENDENT of position and P&L. Where the Case-2 metrics (traded_return, equity) measure a POSITION
    MANAGER, this asks the position-blind 'does every signal stand on its own?' question the B2 signal
    emitter would optimise: a long entry bets price RISES over the next H bars (edge = +forward return), a
    long exit bets it FALLS (edge = -forward return); shorts mirror. Forced TP/SL/trailing exits are NOT
    agent signals and are excluded. Returns {} when nothing is scorable (no trades / window too short)."""
    n = len(prices)
    if n < 2 or horizon < 1 or not trades:
        return {}
    edges = []
    agent_signals = 0
    for t in trades:
        long = t.get("side") == "long"
        ei = int(t["entry_step"])
        agent_signals += 1  # every reconstructed entry is an agent open
        if ei + horizon < n and prices[ei] > 0:
            fwd = prices[ei + horizon] / prices[ei] - 1.0
            edges.append(fwd if long else -fwd)
        if t.get("reason") in ("sell", "cover"):  # agent's OWN close, not a forced tp/sl/trailing/open
            xi = int(t["exit_step"])
            agent_signals += 1
            if xi + horizon < n and prices[xi] > 0:
                fwd = prices[xi + horizon] / prices[xi] - 1.0
                edges.append(-fwd if long else fwd)
    if not edges:
        return {}
    wins = sum(1 for e in edges if e > 0)
    return {
        "signal_expectancy": _finite(sum(edges) / len(edges) * 100),  # mean forward edge per signal, percent
        "signal_hit_rate": _finite(100.0 * wins / len(edges)),
        "signal_coverage": _finite(agent_signals / n),
        "signal_count": len(edges),
        "signal_horizon": int(horizon),
    }


def _benchmark(env, lookback):
    """Buy-and-hold control over the same live window — a display yardstick, NOT a reward target.

    Computed from the price series the run already saw (buy at the first live bar, hold to the last),
    so every run is self-describing against "just holding" without a separate hodl run. Charged the SAME
    per-trade fee the model pays — once on entry, once on exit — so a strategy that merely buys and holds
    scores ~0 against it instead of losing by its fee drag. The capital base differs from the fixed-stake
    strategy, so the delta is indicative, not exact.
    """
    actions = _live(getattr(env, "actions", []), lookback)
    prices = _run_prices(env, len(actions)) if len(actions) >= 2 else []
    prices = [p for p in prices if math.isfinite(p) and p > 0]
    if len(prices) < 2:
        return None
    fee = getattr(env, "transaction_fee_multiplier", 0.0)
    fee = fee if isinstance(fee, (int, float)) and math.isfinite(fee) else 0.0
    round_trip = (1.0 - fee) ** 2
    return {"hold_return_pct": (prices[-1] / prices[0] * round_trip - 1.0) * 100}


def _trade(entry, exit_step, exit_price, reason, pnl, initial):
    return {
        "entry_step": int(entry["step"]),
        "entry_price": _finite(entry["price"]),
        "exit_step": int(exit_step),
        "exit_price": _finite(exit_price),
        "side": entry["side"],
        "reason": reason,
        "pnl": _finite(pnl),
        "pnl_pct": _finite(pnl / initial * 100) if initial else 0.0,
        "bars_held": int(exit_step - entry["step"]),
    }


def _reconstruct_trades(env, lookback, initial):
    """Reconstruct round-trips from the env's OWN per-step close/open signals — actions_made, actions,
    forced_actions, tpsls — i.e. exactly how ``update_position_and_balance`` books a trade. This is
    robust to long/short and the swap env's overloaded action, and stays aligned with ``net_worths``.
    (The ``positions`` array is NOT used: ``take_action`` mutates it in place and the post-close reset
    lands one index off ``net_worths``, so transitions there mispair trades.) Each close's realized
    P&L is ``net_worth_at_close - initial`` (clean per-trade P&L thanks to the fixed-stake reset). A
    position still open at the last bar is closed at the last price (implied sell), reason ``open``.
    Returns (trades, real_equity_curve, live_prices)."""
    actions = [_action_int(a) for a in _live(getattr(env, "actions", []), lookback)]
    made = list(_live(getattr(env, "actions_made", []), lookback))
    forced = [_action_int(f) for f in _live(getattr(env, "forced_actions", []), lookback)]
    tpsls = [_action_int(t) for t in _live(getattr(env, "tpsls", []), lookback)]
    kinds = list(_live(getattr(env, "tpsl_kinds", []), lookback))
    nws = [_finite(x) for x in _live(getattr(env, "net_worths", []), lookback)]
    n = min(len(actions), len(nws))
    if n < 1:
        return [], [], []
    prices = _run_prices(env, n)
    n = min(n, len(prices))
    if n < 1:
        return [], [], prices

    trades = []
    entry = None
    holding = False
    in_position = [False] * n
    for i in range(n):
        is_made = bool(made[i]) if i < len(made) else False
        a = actions[i] if i < len(actions) else 0
        f = forced[i] if i < len(forced) else 0
        tp = tpsls[i] if i < len(tpsls) else 0
        opened = is_made and a in (1, 3) and f == 0
        closed = is_made and ((a in (2, 4) and f == 0) or f in (2, 4))
        if closed and entry is not None:
            kind = kinds[i] if i < len(kinds) else None
            if tp == 1:
                reason = "trailing" if kind == "trailing" else "tp"
            elif tp == -1:
                reason = "sl"
            else:
                reason = "sell" if entry["side"] == "long" else "cover"
            trades.append(_trade(entry, i, prices[i], reason, nws[i] - initial, initial))
            entry = None
            holding = False
        elif opened and entry is None:
            entry = {"step": i, "price": prices[i], "side": "long" if a == 1 else "short"}
            holding = True
        in_position[i] = holding
    if entry is not None:
        trades.append(_trade(entry, n - 1, prices[n - 1], "open", nws[n - 1] - initial, initial))

    equity = _equity_from_trades(trades, nws, in_position, initial, n)
    return trades, equity, prices


def _equity_from_trades(trades, nws, in_position, initial, n):
    """A real (non-sawtooth) equity curve for the fixed-stake account: at each step,
    ``initial + cumulative realized P&L of trades closed so far + current unrealized P&L``."""
    closed = sorted((t["exit_step"], t["pnl"]) for t in trades if t["reason"] != "open")
    equity = []
    realized = 0.0
    ci = 0
    for i in range(n):
        while ci < len(closed) and closed[ci][0] <= i:
            realized += closed[ci][1]
            ci += 1
        unrealized = (nws[i] - initial) if (i < len(in_position) and in_position[i]) else 0.0
        equity.append(initial + realized + unrealized)
    return equity


def _exit_breakdown(trades, initial):
    """Per exit-reason (sell/cover/tp/trailing/sl/open) count, win-rate and P&L — answers
    'does it decide to sell, or do the TP/trailing/SL rules close for it?'."""
    out = {}
    for t in trades:
        b = out.setdefault(t["reason"], {"count": 0, "wins": 0, "pnl": 0.0})
        b["count"] += 1
        if t["pnl"] >= 0:
            b["wins"] += 1
        b["pnl"] += t["pnl"]
    for b in out.values():
        b["win_pct"] = 100.0 * b["wins"] / b["count"] if b["count"] else 0.0
        b["total_pnl_pct"] = _finite(b["pnl"] / initial * 100) if initial else 0.0
        b["avg_pnl_pct"] = _finite(b["pnl"] / b["count"] / initial * 100) if (b["count"] and initial) else 0.0
    return out or None


def _regime_windows(trades, prices, initial, n_windows=4):
    """Skill-vs-luck by equal time sub-windows: each window's market move vs the model's realized
    trading P&L + win-rate. Profit that only appears where the market rose is beta (luck)."""
    n = len(prices)
    if n < 2 or not initial:
        return None
    k = max(1, min(n_windows, n - 1))
    size = n / k
    windows = []
    for w in range(k):
        lo = int(round(w * size))
        hi = (int(round((w + 1) * size)) if w < k - 1 else n) - 1
        if hi <= lo:
            continue
        market = (prices[hi] / prices[lo] - 1.0) * 100 if prices[lo] else 0.0
        wt = [t for t in trades if lo <= int(t["exit_step"]) <= hi]
        pnl = sum(t["pnl"] for t in wt)
        wins = sum(1 for t in wt if t["pnl"] >= 0)
        windows.append({
            "market_return_pct": _finite(market),
            "realized_pnl_pct": _finite(pnl / initial * 100),
            "n_trades": len(wt),
            "win_pct": 100.0 * wins / len(wt) if wt else 0.0,
        })
    return windows or None


def _regime_trend(trades, prices, initial):
    """Skill-vs-luck by market regime: classify each bar up/flat/down by its trailing trend, then
    report realized P&L + win-rate of trades ENTERED in each regime, plus how much of the window
    each regime occupied. Profit concentrated in 'up' is riding the market, not timing it."""
    n = len(prices)
    if n < 3 or not initial:
        return None
    w = max(2, n // _REGIME_TREND_WINDOW_DIVISOR)

    def label(i):
        j = i - w
        if j < 0 or prices[j] <= 0:
            return "flat"
        r = prices[i] / prices[j] - 1.0
        return "up" if r > _REGIME_TREND_BAND else ("down" if r < -_REGIME_TREND_BAND else "flat")

    buckets = {key: {"n_trades": 0, "wins": 0, "pnl": 0.0} for key in ("up", "flat", "down")}
    for t in trades:
        b = buckets[label(int(t["entry_step"]))]
        b["n_trades"] += 1
        if t["pnl"] >= 0:
            b["wins"] += 1
        b["pnl"] += t["pnl"]
    bar_counts = {"up": 0, "flat": 0, "down": 0}
    for i in range(n):
        bar_counts[label(i)] += 1
    out = {}
    for key, b in buckets.items():
        out[key] = {
            "n_trades": b["n_trades"],
            "win_pct": 100.0 * b["wins"] / b["n_trades"] if b["n_trades"] else 0.0,
            "realized_pnl_pct": _finite(b["pnl"] / initial * 100),
            "bars_pct": 100.0 * bar_counts[key] / n if n else 0.0,
        }
    return out


def _iso_from_ms(value):
    try:
        return datetime.datetime.fromtimestamp(
            float(value) / 1000.0, datetime.timezone.utc
        ).isoformat()
    except Exception:
        return None


def _dataset(env, cfg, fidelity, candles):
    fset_id, fspec = resolve_fidelity(cfg)
    dataset = {
        "asset": str(cfg.get("asset", "BTCUSDT")),
        "timeframe": fidelity,
        "candles": int(candles),
        "walk_forward_window": str(cfg.get("walk_forward_window", "2024")),
        "fidelity_set": fset_id,
        "layers": list(fspec["layers"]),
    }
    provider = getattr(env, "data_provider", None)
    timestamps = getattr(provider, "timestamps", None) if provider is not None else None
    if isinstance(timestamps, (list, tuple, np.ndarray)) and len(timestamps) >= 1:
        first, last = _iso_from_ms(timestamps[0]), _iso_from_ms(timestamps[-1])
        if first:
            dataset["from"] = first
        if last:
            dataset["to"] = last
    return dataset


def _lookback(env, cfg):
    provider = getattr(env, "data_provider", None)
    if provider is not None and hasattr(provider, "get_lookback_window"):
        try:
            return int(provider.get_lookback_window())
        except Exception:
            pass
    return int(cfg.get("lookback_window_size", 32))


def _trade_gate(n_trades, min_trades):
    """Map a trade count to the [0,1] objective multiplier: full credit at/above ``min_trades``, falling
    off quadratically below it so under-trading is punished steeply."""
    if min_trades <= 0:
        return 1.0
    return min(1.0, (n_trades / min_trades) ** 2)


# Calendar days one DECISION bar spans, keyed by the step timeframe — for time-normalising the trade count.
_BAR_DAYS = {"1m": 1.0 / 1440, "1h": 1.0 / 24, "1d": 1.0, "1w": 7.0}


def _bar_days(timeframe):
    return _BAR_DAYS.get(str(timeframe or "1d").lower(), 1.0)


def _trades_per_day(n_trades, n_bars, timeframe):
    """Round-trip trades per calendar DAY over the test window — a step-frequency-invariant trade rate.
    The scorecard's liveness gate reads this instead of raw ``n_trades`` (which is confounded by both the
    window length AND the step cadence). 0.0 when there are no test bars (never a divide-by-zero)."""
    days = n_bars * _bar_days(timeframe)
    return _finite(n_trades / days) if days > 0 else 0.0


def _dead_feature_flags(env, lookback):
    """L8 feature-health: sample the observation across the run and flag how many observation entries are
    CONSTANT (zero variance) post-warmup — a dead/broken feature (e.g. the known constant-0 z_score) that
    wastes obs dimensions and can mask a NaN-zeroing bug. Surfaced as a health flag, never a crash."""
    provider = getattr(env, "data_provider", None)
    if provider is None:
        return []
    try:
        n = int(provider.get_timesteps())
    except Exception:
        return []
    if n <= lookback + 4:
        return []
    rows = []
    for s in range(lookback, n, max(1, (n - lookback) // 50)):
        try:
            rows.append(np.asarray(provider.get_values(s), dtype=float).ravel())
        except Exception:
            return []
    rows = [r for r in rows if rows and r.shape == rows[0].shape]
    if len(rows) < 3:
        return []
    dead = int(np.sum(np.asarray(rows).var(axis=0) == 0.0))
    return [f"dead_features:{dead}"] if dead else []


def _health(env, state, is_rl, lookback):
    flags = []
    n_trades = _finite(state[17]) if len(state) > 17 else 0
    if any(not math.isfinite(_finite(state[i], float("nan"))) for i in (1, 2)):
        flags.append("nan_metrics")
    flags.extend(_dead_feature_flags(env, lookback))
    if is_rl:
        live_actions = [_action_int(a) for a in getattr(env, "actions", [])[lookback:]]
        if live_actions and len(set(live_actions)) <= 1:
            flags.append("degenerate_policy")
        if n_trades == 0:
            flags.append("zero_trades")
        elif n_trades <= DEGENERATE_TRADE_COUNT:
            flags.append("few_trades")
    return {"status": "degenerate" if flags else "ok", "flags": flags}


def _oos_stats(equity):
    """Per-step return distribution of the test-window equity curve — the Deflated-Sharpe inputs the
    Wave-2 verdict layer aggregates across runs (oos_sharpe + skew/kurtosis/n for the PSR/DSR). Empty
    when the curve is too short (<3 points) or degenerate, so callers can `metrics.update(...)` safely."""
    eq = [_finite(x) for x in equity]
    rets = [eq[i] / eq[i - 1] - 1.0 for i in range(1, len(eq)) if eq[i - 1] > 0]
    if len(rets) < 2:
        return {}
    s = sharpe_stats(rets)
    return {
        "oos_sharpe": s["sharpe"],
        "oos_n_obs": s["n_obs"],
        "oos_ret_skew": s["skew"],
        "oos_ret_kurt": s["kurtosis"],
    }


def _max_drawdown_pct(equity):
    """Worst peak-to-trough decline of the test-window equity curve, as a signed percent (<= 0, 0 when the
    curve never dips below a prior peak). The one risk metric surfaced — the Diagnosis tab's risk lens and the
    yardstick for the combo_drawdown_penalty experiment. Empty (skippable via metrics.update) when too short."""
    eq = [_finite(x) for x in equity]
    if len(eq) < 2:
        return {}
    peak = eq[0]
    mdd = 0.0
    for x in eq:
        if x > peak:
            peak = x
        if peak > 0:
            dd = x / peak - 1.0
            if dd < mdd:
                mdd = dd
    return {"max_drawdown_pct": _finite(mdd * 100)}


def _capture_stats(equity, prices):
    """Beta + up/down capture of the model vs the market — the A4.3 honesty gate's "not a closet-long" inputs.
    Market return = raw price change per step; model return = equity change per step; each step classified by
    the SIGN of its market return. up_capture / down_capture = the model's summed return over up / down market
    bars divided by the market's summed return there (down_capture < 1 = defensive in the bear; a closet-long
    sits near 1 both ways). beta = cov(model, market) / var(market). Empty (skippable via metrics.update, so
    the engine's capture gate SKIPS rather than reading a non-participating run as maximally 'defensive') when
    the two series don't align, are too short, the model never participated (flat equity), or the market has no
    variance; each side is omitted independently when its regime never occurred."""
    eq = [_finite(x) for x in equity]
    px = [_finite(x) for x in prices]
    if len(eq) != len(px) or len(px) < 3:
        return {}
    if max(eq) - min(eq) <= 0.0:  # flat equity = the model never participated — a do-nothing run, not defensive
        return {}
    mkt, mdl = [], []
    for i in range(1, len(px)):
        if px[i - 1] > 0 and eq[i - 1] > 0:
            mkt.append(px[i] / px[i - 1] - 1.0)
            mdl.append(eq[i] / eq[i - 1] - 1.0)
    if len(mkt) < 2:
        return {}
    out = {}
    up_k = sum(k for k in mkt if k > 0)
    if up_k > 0:
        out["up_capture"] = _finite(sum(m for m, k in zip(mdl, mkt) if k > 0) / up_k)
    down_k = sum(k for k in mkt if k < 0)
    if down_k < 0:
        out["down_capture"] = _finite(sum(m for m, k in zip(mdl, mkt) if k < 0) / down_k)
    mean_k = sum(mkt) / len(mkt)
    var_k = sum((k - mean_k) ** 2 for k in mkt) / len(mkt)
    if var_k > 0:
        mean_m = sum(mdl) / len(mdl)
        cov = sum((mdl[i] - mean_m) * (mkt[i] - mean_k) for i in range(len(mkt))) / len(mkt)
        out["beta"] = _finite(cov / var_k)
    return out


def _provenance_fingerprint(cfg, stored_cfg):
    """Reproducibility fingerprint (L5): stamp code + config + data + lib versions + the resolved train/test
    span so any result can be re-derived and audited. Every field is best-effort — a missing git or lib
    never breaks a run."""
    import hashlib
    import json
    import os
    import subprocess

    fp = {}
    try:
        fp["gitCommit"] = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
        )
        fp["gitDirty"] = bool(
            subprocess.check_output(["git", "status", "--porcelain"], stderr=subprocess.DEVNULL).decode().strip()
        )
    except Exception:
        pass
    try:
        fp["configHash"] = hashlib.sha256(
            json.dumps(stored_cfg, sort_keys=True, default=str).encode()
        ).hexdigest()[:16]
    except Exception:
        pass
    try:
        from trainer.walk_forward import resolve_walk_forward_window

        _, _, meta = resolve_walk_forward_window(cfg)
        fp["trainFrom"], fp["trainTo"] = meta.get("train_from"), meta.get("train_to")
        fp["testFrom"], fp["testTo"] = meta.get("test_from"), meta.get("test_to")
    except Exception:
        pass
    try:
        import trainer.config_builder as cb

        dc = cb.build_data_config(cfg)
        groups = list(dc.train_data_paths) + list(dc.test_data_paths)
        paths = [p for g in groups for p in (g if isinstance(g, (list, tuple)) else [g])]
        sig = sorted((os.path.basename(p), os.path.getsize(p)) for p in paths if os.path.exists(p))
        fp["dataVersion"] = hashlib.sha256(repr(sig).encode()).hexdigest()[:16]
        fp["dataFiles"] = len(sig)
    except Exception:
        pass
    try:
        import importlib.metadata as _im

        libs = {}
        for k in ("torch", "numpy", "pandas", "stable_baselines3", "sb3_contrib", "gymnasium"):
            try:
                libs[k] = _im.version(k)
            except Exception:
                pass
        if libs:
            fp["libVersions"] = libs
    except Exception:
        pass
    return fp


def build_summary(env, state, cfg, model, ran_at, is_rl):
    lookback = _lookback(env, cfg)
    fidelity = resolve_fidelity(cfg)[1]["fidelity_run"]
    initial = (
        _finite(getattr(env, "initial_net_worth", 0))
        or _finite(getattr(env, "initial_balance", 0))
        or 1.0
    )

    trades, equity, prices = _reconstruct_trades(env, lookback, initial)
    if len(equity) >= 2 and initial:
        total_return = (equity[-1] - initial) / initial
    else:
        total_return = _finite(state[2]) if len(state) > 2 else 0.0
        equity = [_finite(x) for x in _live(getattr(env, "net_worths", []), lookback)]

    n_trades = _finite(state[17]) if len(state) > 17 else 0.0
    trade_gate = _trade_gate(n_trades, MIN_TRADES_FOR_FULL_CREDIT)
    traded_return = total_return * 100 * trade_gate

    # Realized transaction-cost drag: the env appends each trade's fee (in $) to `fees`, so the total
    # over the run as basis points of the stake is the honest "how much did fees cost" figure.
    fees_paid = sum(_finite(f) for f in getattr(env, "fees", []))
    metrics = {
        # Diagnostic only (no longer the objective): the old trade-gated return = total_return_pct × the
        # quadratic trade_gate. Kept so a run's gating is still visible; the scorecard's trades_per_day gate
        # now carries the "trade often enough" constraint instead.
        "traded_return": traded_return,
        # The trivial-baseline reference (in objective units = total_return_pct) the exploration autopilot's
        # basin gate must be beaten by: a do-nothing agent earns 0 total_return_pct, so a region only qualifies
        # as a basin when it returns PROFITABLY above this floor.
        "baseline": 0.0,
        "total_return_pct": total_return * 100,
        "win_pct": _finite(state[7]) if len(state) > 7 else 0.0,
        "n_trades": n_trades,
        "trade_gate": trade_gate,
        "stop_losses": _finite(state[18]) if len(state) > 18 else 0.0,
        "final_net_worth": equity[-1] if equity else initial,
        "realized_cost_bps": _finite(fees_paid / initial * 10000) if initial else 0.0,
    }
    metrics.update(_oos_stats(equity))
    metrics.update(_max_drawdown_pct(equity))
    metrics.update(_capture_stats(equity, prices))
    # Time-normalised trade liveness (the scorecard's trades_per_day gate) — over the test bars (oos_n_obs
    # when available, else the equity length), scaled by the step cadence so 1d and 1h runs are comparable.
    n_bars_oos = metrics.get("oos_n_obs", max(len(equity) - 1, 0))
    metrics["trades_per_day"] = _trades_per_day(n_trades, n_bars_oos, cfg.get("timeframe"))
    # Case-1 signal lens (position-blind forward-return edge per buy/sell) — a SECOND read alongside the
    # Case-2 position-manager metrics; {} for non-trading runs, so it never adds noise to a do-nothing run.
    metrics.update(_signal_expectancy(trades, prices, int(cfg.get("signal_horizon", 5))))
    benchmark = _benchmark(env, lookback)
    if benchmark:
        metrics["hold_return_pct"] = benchmark["hold_return_pct"]
        metrics["return_vs_hold_pct"] = _finite(total_return * 100 - benchmark["hold_return_pct"])
        # Provenance flag: this run's hold benchmark already nets out the round-trip fee, so the
        # viewer's one-time migration knows not to re-adjust it.
        metrics["hold_net_of_fees"] = True

    # RL-only: the share of the agent's buy/sell output that was a no-op (a signal it couldn't act on).
    # The headline number for "can I trust the raw signal stream?" — see _signal_noise.
    if is_rl:
        noise = _signal_noise(env, lookback)
        if noise:
            metrics.update(noise)

    series = {"equity": _downsample(equity)}

    # Store the CONCRETE fidelity_set (never the "auto" synonym) so the run record shows + groups by the
    # actual value (e.g. "1h+1d") everywhere — "auto" stays a launch-form convenience only.
    stored_cfg = dict(cfg)
    stored_cfg["fidelity_set"] = resolve_fidelity(cfg)[0]

    summary = {
        # The objective is the HONEST post-fee portfolio return (== metrics.total_return_pct), not the
        # magic-20 trade-gated traded_return — trade frequency is a scorecard GATE (trades_per_day) now, so
        # the objective no longer suppresses a real return. traded_return is kept as a diagnostic metric.
        "objective": metrics["total_return_pct"],
        "metrics": metrics,
        "health": _health(env, state, is_rl, lookback),
        "config": stored_cfg,
        "provenance": {"ranAt": ran_at, **_provenance_fingerprint(cfg, stored_cfg)},
        "series": series,
        "dataset": _dataset(env, cfg, fidelity, len(equity)),
    }

    exits = _exit_breakdown(trades, initial)
    if exits:
        summary["exits"] = exits
    regimes = {}
    windows = _regime_windows(trades, prices, initial)
    if windows:
        regimes["windows"] = windows
    trend = _regime_trend(trades, prices, initial)
    if trend:
        regimes["trend"] = trend
    if regimes:
        summary["regimes"] = regimes
    if trades:
        summary["ledger"] = trades[:5000]

    artifacts = {}
    try:
        run_chart = _run_chart(env, lookback, trades)
    except Exception:
        run_chart = None
    if run_chart:
        artifacts["runChart"] = run_chart
    checkpoint = getattr(model, "id", None)
    # Only advertise the checkpoint artifact when one was actually persisted — with save_checkpoint off there is
    # no file, so recording the path would make --evaluate / replay fail trying to load it.
    saved_checkpoint = getattr(getattr(model, "config", None), "save_checkpoint", True)
    if getattr(model, "produces_checkpoint", lambda: False)() and checkpoint and saved_checkpoint:
        artifacts["checkpoint"] = f"checkpoints/{checkpoint}.zip"
        artifacts["best"] = False
    if artifacts:
        summary["artifacts"] = artifacts
    if benchmark:
        summary["benchmark"] = benchmark
    if "seed" in cfg:
        summary["seed"] = int(cfg["seed"])
        summary["provenance"]["seed"] = int(cfg["seed"])
    # Extra-train lineage: a continued run records the parent checkpoint it was seeded from, so the viewer
    # can render the parent → continued chain and judge it on the standardised test sets (not the parent).
    if cfg.get("continue_from"):
        summary["provenance"]["continuedFrom"] = str(cfg["continue_from"])
    return summary
