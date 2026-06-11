"""Compute the trainer-standard RunSummary for one BlackSwan trading test run.

The risk-adjusted objective (Sharpe / CAGR / max-drawdown) is computed HERE from
the env's retained per-step equity curve (``env.net_worths``, padded by the
provider's lookback window) and action history — the env's own risk methods are
abstract-unimplemented or price-based placeholders, so they are not used.
"""

import math
import statistics

import numpy as np

_PERIODS_PER_YEAR = {"1m": 365.0 * 24 * 60, "5m": 365.0 * 24 * 12, "15m": 365.0 * 24 * 4,
                     "1h": 365.0 * 24, "4h": 365.0 * 6, "1d": 365.0}
_MAX_SERIES_POINTS = 200


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
    return {"status": "degenerate" if flags else "ok", "flags": flags}


def build_summary(env, state, cfg, model, ran_at, is_rl):
    lookback = _lookback(env, cfg)
    fidelity = "1d" if str(cfg.get("timeframe", "1d")) == "1d" else "1h"
    periods = _PERIODS_PER_YEAR.get(fidelity, 365.0)
    curve = _equity_curve(env, lookback)
    returns = [curve[i] / curve[i - 1] - 1.0 for i in range(1, len(curve)) if curve[i - 1] > 0]

    sharpe = _sharpe(returns, periods)
    summary = {
        "objective": sharpe,
        "metrics": {
            "sharpe": sharpe,
            "total_return_pct": _finite(state[2]) * 100 if len(state) > 2 else 0.0,
            "max_drawdown_pct": _max_drawdown(curve) * 100,
            "cagr_pct": _cagr(curve, periods) * 100,
            "win_pct": _finite(state[7]) if len(state) > 7 else 0.0,
            "n_trades": _finite(state[17]) if len(state) > 17 else 0.0,
            "stop_losses": _finite(state[18]) if len(state) > 18 else 0.0,
            "final_net_worth": curve[-1] if curve else 0.0,
        },
        "health": _health(env, state, is_rl, lookback),
        "config": dict(cfg),
        "provenance": {"ranAt": ran_at},
        "series": {"equity": _downsample(curve)},
    }
    checkpoint = getattr(model, "id", None)
    if is_rl and checkpoint:
        summary["artifacts"] = {"checkpoint": f"checkpoints/{checkpoint}.zip", "best": False}
    if "seed" in cfg:
        summary["seed"] = int(cfg["seed"])
        summary["provenance"]["seed"] = int(cfg["seed"])
    return summary
