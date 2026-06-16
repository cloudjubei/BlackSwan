"""Compute the trainer-standard RunSummary for one BlackSwan trading test run.

The risk-adjusted objective (Sharpe / CAGR / max-drawdown) is computed HERE from
the env's retained per-step equity curve (``env.net_worths``, padded by the
provider's lookback window) and action history — the env's own risk methods are
abstract-unimplemented or price-based placeholders, so they are not used.
"""

import bisect
import datetime
import math
import statistics

import numpy as np

from trainer.fidelity import resolve_fidelity

_PERIODS_PER_YEAR = {"1m": 365.0 * 24 * 60, "5m": 365.0 * 24 * 12, "15m": 365.0 * 24 * 4,
                     "1h": 365.0 * 24, "4h": 365.0 * 6, "1d": 365.0}
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


def _run_chart(env, lookback):
    """A JSON-renderable price line + preserved buy/sell/TP/SL markers for the hub viewer.

    Re-creates the data the repo's interactive-only ``plot_actions_data`` draws, but as
    serialisable arrays: a downsampled close price with every trade marker mapped onto the
    downsampled grid so no trade is dropped by the downsample (1=buy, 2=sell; TP/SL from tpsls).
    """
    actions = [_action_int(a) for a in _live(getattr(env, "actions", []), lookback)]
    if len(actions) < 2:
        return None
    prices = _run_prices(env, len(actions))
    if len(prices) < 2:
        return None
    n = min(len(actions), len(prices))
    actions = actions[:n]
    prices = prices[:n]
    tpsls = [_action_int(t) for t in _live(getattr(env, "tpsls", []), lookback)[:n]]

    ds_price, kept = _downsample_indexed(prices)
    seen = set()
    markers = []
    for i in range(n):
        events = []
        if actions[i] == 1:
            events.append("buy")
        elif actions[i] == 2:
            events.append("sell")
        if i < len(tpsls) and tpsls[i] == 1:
            events.append("tp")
        elif i < len(tpsls) and tpsls[i] == -1:
            events.append("sl")
        if not events:
            continue
        x = _marker_x(i, kept)
        for kind in events:
            if (x, kind) in seen:
                continue
            seen.add((x, kind))
            markers.append({"i": x, "type": kind, "price": _finite(prices[i])})
    return {"price": ds_price, "markers": markers}


def _benchmark(env, lookback, periods):
    """Buy-and-hold control over the same live window — a display benchmark, NOT a reward target.

    Computed from the price series the run already saw (buy at the first live bar, hold to the
    last), so every run is self-describing against "just holding" without a separate hodl run.
    """
    actions = _live(getattr(env, "actions", []), lookback)
    prices = _run_prices(env, len(actions)) if len(actions) >= 2 else []
    prices = [p for p in prices if math.isfinite(p) and p > 0]
    if len(prices) < 2:
        return None
    curve = [p / prices[0] for p in prices]
    rets = [curve[i] / curve[i - 1] - 1.0 for i in range(1, len(curve)) if curve[i - 1] > 0]
    return {
        "hold_return_pct": (curve[-1] - 1.0) * 100,
        "hold_sharpe": _sharpe(rets, periods),
        "hold_max_drawdown_pct": _max_drawdown(curve) * 100,
    }


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


def _equity_curve(env, lookback):
    nw = [float(x) for x in getattr(env, "net_worths", [])]
    live = nw[lookback:] if len(nw) > lookback + 1 else nw
    return live if len(live) >= 2 else nw


def _sharpe(returns, periods_per_year):
    if len(returns) < 2:
        return 0.0
    sd = statistics.pstdev(returns)
    if sd <= 0:
        return 0.0
    return _finite((statistics.fmean(returns) / sd) * math.sqrt(periods_per_year))


def _max_drawdown(curve):
    peak = float("-inf")
    mdd = 0.0
    for x in curve:
        peak = max(peak, x)
        if peak > 0:
            mdd = min(mdd, x / peak - 1.0)
    return _finite(mdd)


def _cagr(curve, periods_per_year):
    if len(curve) < 2 or curve[0] <= 0:
        return 0.0
    years = max(len(curve), 1) / periods_per_year
    return _finite((curve[-1] / curve[0]) ** (1.0 / years) - 1.0)


def _trade_gate(n_trades, mode, min_trades):
    """Map a trade count to the [0,1] objective multiplier for the chosen gate mode. The gate is the
    research-named "churn band-aid" — realistic fees already regulate frequency — so it is a sweepable
    lever: none (ungated), linear, quadratic (the historical default, punishes under-trading steeply),
    or threshold (full credit at/above the bar, none below)."""
    if min_trades <= 0 or mode == "none":
        return 1.0
    ratio = n_trades / min_trades
    if mode == "linear":
        return min(1.0, ratio)
    if mode == "threshold":
        return 1.0 if n_trades >= min_trades else 0.0
    return min(1.0, ratio**2)


def _window_breakdown(curve, n_windows=4):
    """RB7: split the test equity curve into ``n_windows`` equal sub-periods and report robustness
    across them — a strategy that only profits in one sub-window (a single lucky regime) is fragile.
    Returns per-window return %, the worst window, and the fraction of windows that were profitable.
    Equal-size index chunks (bars are uniform within a fidelity) avoid timestamp-alignment fragility.
    """
    pts = [c for c in curve if isinstance(c, (int, float)) and math.isfinite(c) and c > 0]
    if len(pts) < 4:
        return None
    n = max(1, min(n_windows, len(pts) - 1))
    size = len(pts) / n
    returns = []
    for i in range(n):
        lo = int(round(i * size))
        hi = (int(round((i + 1) * size)) if i < n - 1 else len(pts)) - 1
        if hi > lo and pts[lo] > 0:
            returns.append((pts[hi] / pts[lo] - 1.0) * 100)
    if not returns:
        return None
    profitable = sum(1 for r in returns if r > 0)
    return {
        "window_returns_pct": [round(r, 4) for r in returns],
        "worst_window_return_pct": min(returns),
        "windows_profitable_pct": 100.0 * profitable / len(returns),
        "n_windows": len(returns),
    }


def _health(env, state, is_rl, lookback):
    flags = []
    n_trades = _finite(state[17]) if len(state) > 17 else 0
    if any(not math.isfinite(_finite(state[i], float("nan"))) for i in (1, 2)):
        flags.append("nan_metrics")
    if is_rl:
        live_actions = [_action_int(a) for a in getattr(env, "actions", [])[lookback:]]
        if live_actions and len(set(live_actions)) <= 1:
            flags.append("degenerate_policy")
        if n_trades == 0:
            flags.append("zero_trades")
        elif n_trades <= DEGENERATE_TRADE_COUNT:
            flags.append("few_trades")
    return {"status": "degenerate" if flags else "ok", "flags": flags}


def build_summary(env, state, cfg, model, ran_at, is_rl):
    lookback = _lookback(env, cfg)
    fidelity = resolve_fidelity(cfg)[1]["fidelity_run"]
    periods = _PERIODS_PER_YEAR.get(fidelity, 365.0)
    curve = _equity_curve(env, lookback)
    returns = [curve[i] / curve[i - 1] - 1.0 for i in range(1, len(curve)) if curve[i - 1] > 0]

    sharpe = _sharpe(returns, periods)
    benchmark = _benchmark(env, lookback, periods)
    # Total return from the post-fee equity curve (so it is CONSISTENT with final_net_worth = curve[-1]).
    # The env's state[2] is GROSS realized profit / initial — fees are never subtracted from it — which
    # let a fee-eaten run report a positive % while its final net worth sat BELOW the starting balance.
    # This also makes the traded_return objective fee-honest.
    total_return = (
        (curve[-1] / curve[0] - 1.0)
        if len(curve) >= 2 and curve[0]
        else (_finite(state[2]) if len(state) > 2 else 0.0)
    )
    n_trades = _finite(state[17]) if len(state) > 17 else 0.0
    # Trade-aware objective: total return (profit, NOT beat-hold) gated by trade frequency.
    trade_gate = _trade_gate(
        n_trades, str(cfg.get("trade_gate_mode", "quadratic")), MIN_TRADES_FOR_FULL_CREDIT
    )
    traded_return = total_return * 100 * trade_gate
    windows = _window_breakdown(curve)
    metrics = {
        "traded_return": traded_return,
        "total_return_pct": total_return * 100,
        "win_pct": _finite(state[7]) if len(state) > 7 else 0.0,
        "n_trades": n_trades,
        "trade_gate": trade_gate,
        "sharpe": sharpe,
        "max_drawdown_pct": _max_drawdown(curve) * 100,
        "cagr_pct": _cagr(curve, periods) * 100,
        "stop_losses": _finite(state[18]) if len(state) > 18 else 0.0,
        "final_net_worth": curve[-1] if curve else 0.0,
    }
    if windows:
        # RB7: robustness across sub-periods — the worst window's return + how many windows profited.
        metrics["worst_window_return_pct"] = windows["worst_window_return_pct"]
        metrics["windows_profitable_pct"] = windows["windows_profitable_pct"]
    if benchmark:
        metrics["sharpe_alpha"] = _finite(sharpe - _finite(benchmark.get("hold_sharpe")))
    series = {"equity": _downsample(curve)}
    if windows:
        series["window_returns_pct"] = windows["window_returns_pct"]
    summary = {
        "objective": traded_return,
        "metrics": metrics,
        "health": _health(env, state, is_rl, lookback),
        "config": dict(cfg),
        "provenance": {"ranAt": ran_at},
        "series": series,
        "dataset": _dataset(env, cfg, fidelity, len(curve)),
    }
    artifacts = {}
    try:
        run_chart = _run_chart(env, lookback)
    except Exception:
        run_chart = None
    if run_chart:
        artifacts["runChart"] = run_chart
    checkpoint = getattr(model, "id", None)
    if is_rl and checkpoint:
        artifacts["checkpoint"] = f"checkpoints/{checkpoint}.zip"
        artifacts["best"] = False
    if artifacts:
        summary["artifacts"] = artifacts
    if benchmark:
        summary["benchmark"] = benchmark
    if "seed" in cfg:
        summary["seed"] = int(cfg["seed"])
        summary["provenance"]["seed"] = int(cfg["seed"])
    return summary
