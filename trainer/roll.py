"""Commodity index-roll ("Goldman roll") front-running probe — the first NON-CRYPTO, non-price-derived probe.

Passive long-only commodity-index funds (S&P GSCI, Bloomberg BCOM) mechanically roll their front-month longs into
the next contract on a FIXED, PUBLISHED schedule (GSCI = business days 5-9 each month), price-INSENSITIVELY and
fully TELEGRAPHED. That makes them a genuinely NAMED forced counterparty that structurally cannot stop — the one
thing the "model Apple and predict its price" idea lacks. This probe asks the honest question: does anticipating
that flow — going SHORT the WTI M1-M2 calendar spread into the roll (roll_frontrun, Mou 2011's direction) or LONG
it (roll_fade, betting the effect over-arbitraged and reversed) — earn a risk-adjusted return net of cost, in
BOTH a 2006-2011 (reproduce Mou) and a 2016-2025 (refute-via-decay) window.

The trade is MARKET-NEUTRAL (a calendar spread), so there is no buy-and-hold benchmark and the honest decider is
the strategy's own per-trade Sharpe, DSR-deflated for the config sweep (the objective is oos_sharpe). It reuses
the trainer summary primitives (_oos_stats / _max_drawdown_pct / _track_record_stats), so every existing gate and
DSR aggregation reads a roll cell unchanged. The surface new here is (a) the two-contract spread and (b) the
business-day roll calendar, both carrying the leakage traps pinned in test_roll.py: the spread is built from RAW
per-contract settlements (M1 and M2 of the actual contracts), NEVER a back-adjusted continuous series (a
continuous roll bakes the effect in — a look-ahead machine); a month missing either leg on its entry/exit day is
DROPPED, never forward-filled; entry business-day < exit business-day is enforced; and each month's return reads
ONLY that month's entry+exit settlements. Deterministic and model-free (the seed is contract-only).

DATA: WTI contract settlements M1..M4, daily, from EIA (RCLC1..RCLC4), mined to eia-wti/<asset>.json as
{ "YYYY-MM-DD": {"m1": .., "m2": .., "m3": .., "m4": ..} }. See scripts/fetch_wti.py.
"""

import argparse
import json
import os

import numpy as np

from trainer.summary import (
    _finite,
    _max_drawdown_pct,
    _oos_stats,
    _provenance_fingerprint,
    _track_record_stats,
)

# roll_frontrun SHORTS the M1-M2 spread into the roll (Mou's direction); roll_fade LONGS it. An unknown value is
# refused so a typo can never file one arm's result under the other's label.
SIGNALS = ("roll_frontrun", "roll_fade")
DEFAULT_SIGNAL = "roll_frontrun"

DEFAULT_ASSET = "WTI"
DEFAULT_ENTRY_BDAY = 3        # enter before the GSCI day-5 roll start
DEFAULT_EXIT_BDAY = 10        # exit at/after the day-9 roll end
DEFAULT_ROLL_SCHEDULE = "gsci"  # descriptive: the GSCI window is bdays 5-9; entry/exit_bday are the actual levers
DEFAULT_FEE = 0.0003          # per-leg, charged on both legs at entry and again at exit

WTI_DIR = "eia-wti"

# window id -> (first_test_year, last_test_year). The roll strategy is a FIXED rule (no training leg), so a window
# is just the accounted test span. roll-2006-2011 reproduces Mou's era; roll-2016-2025 is the post-publication OOS.
ROLL_WINDOWS = {
    "roll-2006-2011": (2006, 2011),
    "roll-2016-2025": (2016, 2025),
}


# --- settlement loader: two distinct legs required, missing leg dropped (no forward-fill) ------------


def _load_table(asset):
    path = f"{WTI_DIR}/{asset}.json"
    if not os.path.exists(path):
        return {}
    with open(path) as fh:
        return json.load(fh)


def settlement_series_from_table(table):
    """Ordered [(date, m1, m2)] from a raw settlement table, keeping ONLY dates where BOTH the M1 and M2 legs are
    present and positive. A date missing either leg is dropped, never forward-filled from a neighbour — the trap
    that would fabricate a spread the market never quoted."""
    out = []
    for date in sorted(table):
        row = table[date] or {}
        m1, m2 = row.get("m1"), row.get("m2")
        if m1 is None or m2 is None:
            continue
        m1, m2 = float(m1), float(m2)
        if m1 > 0 and m2 > 0:
            out.append((date, m1, m2))
    return out


def settlement_series(asset):
    """The on-disk WTI M1/M2 settlement series for ``asset`` (empty when the file is absent)."""
    return settlement_series_from_table(_load_table(asset))


def months_from_series(series):
    """Group a settlement series into an ordered {``YYYY-MM``: [(date, m1, m2), ...]} map, each month's rows in
    date order — so business-day k is simply the k-th trading date of the month (present dates ARE trading days)."""
    out = {}
    for date, m1, m2 in series:
        out.setdefault(date[:7], []).append((date, m1, m2))
    for key in out:
        out[key].sort(key=lambda r: r[0])
    return out


def _trade_dates_for_month(rows, entry_bday, exit_bday):
    """(entry_row, exit_row) for a month's rows, entry/exit at the entry_bday/exit_bday-th trading date, or
    ``None`` when the month has fewer trading days than exit_bday. ``rows`` are assumed date-sorted."""
    if len(rows) < exit_bday:
        return None
    return rows[entry_bday - 1], rows[exit_bday - 1]


# --- the spread trade: direction, cost ---------------------------------------------------------------


def spread_trade_return(entry_row, exit_row, signal, fee):
    """(net, gross, cost) of one month's M1-M2 calendar-spread round-trip, as a fraction of the front notional.

    spread = M1 - M2; delta = (spread_out - spread_in)/M1_in. roll_frontrun SHORTS the spread (gross = -delta,
    profits when the nearby cheapens vs the deferred — the index-selling footprint Mou documented); roll_fade
    LONGS it (gross = +delta). Cost charges every leg it touches — both contracts, at entry and at exit. This is
    a CONSERVATIVE model (four legged crossings, ~4x the per-leg fee): an exchange-listed calendar spread trades
    as ONE instrument at ~2-4 bps round trip, so the true cost is ~3-4x lower — but the disproved verdict is
    robust to it (zeroing cost still leaves the best in-sample cell insignificant)."""
    if signal not in SIGNALS:
        raise SystemExit(f"unknown signal {signal!r}; choose one of {sorted(SIGNALS)}")
    _d_in, m1_in, m2_in = entry_row
    _d_out, m1_out, m2_out = exit_row
    spread_in = m1_in - m2_in
    spread_out = m1_out - m2_out
    delta = (spread_out - spread_in) / m1_in
    gross = delta if signal == "roll_fade" else -delta
    cost = float(fee) * (m1_in + m2_in + m1_out + m2_out) / m1_in
    return gross - cost, gross, cost


def roll_returns(series, signal, entry_bday, exit_bday, fee, months=None):
    """The per-trade ledger over ``months`` (default: every month in ``series``): one M1-M2 spread round-trip per
    month with enough trading days. Each entry is {month, entry_date, exit_date, net, gross, cost}. Enforces
    entry_bday < exit_bday (a spread cannot be entered after it is exited)."""
    entry_bday, exit_bday = int(entry_bday), int(exit_bday)
    if entry_bday < 1 or exit_bday <= entry_bday:
        raise SystemExit(f"need 1 <= entry_bday < exit_bday; got entry_bday={entry_bday}, exit_bday={exit_bday}")
    by_month = months_from_series(series)
    keys = list(months) if months is not None else sorted(by_month)
    out = []
    for key in keys:
        rows = by_month.get(key)
        if not rows:
            continue
        dates = _trade_dates_for_month(rows, entry_bday, exit_bday)
        if dates is None:
            continue
        entry_row, exit_row = dates
        net, gross, cost = spread_trade_return(entry_row, exit_row, signal, fee)
        out.append({
            "month": key, "entry_date": entry_row[0], "exit_date": exit_row[0],
            "net": net, "gross": gross, "cost": cost,
        })
    return out


def _window_months(window):
    lo, hi = ROLL_WINDOWS[window]
    return [f"{y:04d}-{m:02d}" for y in range(lo, hi + 1) for m in range(1, 13)]


# --- the run contract ---------------------------------------------------------------------------------


def run(cfg):
    """Run one roll cell and return a trainer-contract RunSummary. The objective is the per-trade OOS Sharpe of
    the market-neutral spread strategy; there is no buy-and-hold benchmark (a spread has none), so the DSR-
    deflated Sharpe is the honest decider. n_trades (one round-trip per traded month) is emitted so a thin count
    is visible."""
    asset = str(cfg.get("asset", DEFAULT_ASSET))
    signal = str(cfg.get("signal", DEFAULT_SIGNAL))
    if signal not in SIGNALS:
        raise SystemExit(f"unknown signal {signal!r}; choose one of {sorted(SIGNALS)}")
    window = str(cfg.get("walk_forward_window", next(iter(ROLL_WINDOWS))))
    if window not in ROLL_WINDOWS:
        raise SystemExit(f"unknown walk_forward_window {window!r}; choices: {sorted(ROLL_WINDOWS)}")
    entry_bday = int(cfg.get("entry_bday", DEFAULT_ENTRY_BDAY))
    exit_bday = int(cfg.get("exit_bday", DEFAULT_EXIT_BDAY))
    fee = float(cfg.get("transaction_cost", DEFAULT_FEE))

    series = settlement_series(asset)
    ledger = roll_returns(series, signal, entry_bday, exit_bday, fee, months=_window_months(window))
    nets = [t["net"] for t in ledger]
    costs = [t["cost"] for t in ledger]
    n_trades = len(nets)

    equity = list(np.cumprod([1.0 + r for r in nets])) if nets else []
    total_return_pct = (equity[-1] - 1.0) * 100.0 if equity else 0.0

    metrics = {
        "total_return_pct": _finite(total_return_pct),
        "baseline": 0.0,
        "n_trades": n_trades,
        "n_months_traded": n_trades,
        "final_net_worth": _finite(equity[-1]) if equity else 1.0,
        "realized_cost_bps": _finite(sum(costs) / n_trades * 10000) if n_trades else 0.0,
    }
    oos = _oos_stats(equity)
    metrics.update(oos)
    metrics.update(_track_record_stats(oos))
    metrics.update(_max_drawdown_pct(equity))
    if nets:
        metrics["signal_expectancy"] = _finite(sum(nets) / len(nets) * 100.0)
        metrics["signal_hit_rate"] = _finite(100.0 * sum(1 for r in nets if r > 0) / len(nets))
        metrics["signal_count"] = n_trades
    else:
        metrics["signal_expectancy"] = 0.0
        metrics["signal_hit_rate"] = 0.0

    objective = metrics.get("oos_sharpe", 0.0)
    stored_cfg = dict(cfg)
    summary = {
        "objective": _finite(objective),
        "metrics": metrics,
        "health": {"status": "ok" if n_trades else "degenerate", "flags": [] if n_trades else ["no_data"]},
        "config": stored_cfg,
        "dataset": {
            "asset": asset,
            "timeframe": "monthly-roll",
            "candles": n_trades,
            "walk_forward_window": window,
            "from": ledger[0]["entry_date"] if ledger else None,
            "to": ledger[-1]["exit_date"] if ledger else None,
        },
        "walk_forward_window": window,
    }
    try:
        summary["provenance"] = {"ranAt": cfg.get("ran_at"), **_provenance_fingerprint(cfg, stored_cfg)}
    except Exception:
        summary["provenance"] = {}
    if "seed" in cfg:
        summary["seed"] = int(cfg["seed"])
    return summary


def main():
    parser = argparse.ArgumentParser(description="Commodity index-roll (Goldman roll) calendar-spread backtest")
    parser.add_argument("--config-json", required=True)
    parser.add_argument("--summary-out", required=True)
    args = parser.parse_args()
    with open(args.config_json) as fh:
        cfg = json.load(fh)
    summary = run(cfg)
    with open(args.summary_out, "w") as fh:
        json.dump(summary, fh)
    m = summary["metrics"]
    print(
        f"objective(oos_sharpe)={summary['objective']:.4f} "
        f"n_trades={m.get('n_trades', 0)} signal_expectancy={m.get('signal_expectancy', 0.0):.4f} "
        f"total_return_pct={m.get('total_return_pct', 0.0):.4f} "
        f"realized_cost_bps={m.get('realized_cost_bps', 0.0):.2f} -> {args.summary_out}"
    )


if __name__ == "__main__":
    main()
