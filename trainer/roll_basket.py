"""Pooled energy-basket index-roll probe — the power-recoverable follow-up to the single-asset WTI roll null.

The WTI roll came back a scope-limited null: single-asset WTI discarded ~sqrt(24) of Mou (2011)'s POOLED
~24-commodity t-stat, so it could not adjudicate the basket. This probe trades the SAME monthly M1-M2 roll
spread across the EIA ENERGY complex (crude WTI + NY-Harbor heating oil + RBOB gasoline + Henry-Hub natural gas)
as an EQUAL-WEIGHT PORTFOLIO: each month is the mean of the per-leg spread-trade returns, so diversification
lowers the portfolio return variance and lifts the t-stat — the mechanism behind Mou's t~4-5. It is the honest
reproduction at a closer-to-unit-of-analysis scope, though STILL only the energy sub-basket (GSCI's largest
weight): effective power is bounded by sqrt(4) and reduced by the high cross-energy correlation, and the crash
super-contango tail that drove the WTI null (2008, 2020) is a COMMON energy factor pooling cannot diversify away.

It reuses trainer/roll.py's per-commodity spread machinery wholesale (settlement parsing, the business-day roll
calendar, the M1-M2 spread trade, and all four mutation-proven leakage guards). The only new surface is the
equal-weight aggregation: a leg with no trade in a given month is EXCLUDED from that month's average, never
zero-filled. Deterministic and model-free (the seed is contract-only).

DATA: eia-energy/<SYMBOL>.json for SYMBOL in the basket, mined by scripts/fetch_energy.py (needs EIA_API_KEY).
"""

import argparse
import json
import os

import numpy as np

from trainer import roll
from trainer.summary import (
    _finite,
    _max_drawdown_pct,
    _oos_stats,
    _provenance_fingerprint,
    _track_record_stats,
)

ENERGY_DIR = "eia-energy"
BASKETS = {"energy4": ["CRUDE", "HEATOIL", "RBOB", "NATGAS"]}
DEFAULT_BASKET = "energy4"


def _load_series(symbol):
    path = f"{ENERGY_DIR}/{symbol}.json"
    if not os.path.exists(path):
        return []
    with open(path) as fh:
        return roll.settlement_series_from_table(json.load(fh))


def basket_month_returns(series_by_sym, signal, entry_bday, exit_bday, fee, months=None):
    """The equal-weight portfolio monthly-return series: [{month, portfolio_ret, n_legs}]. Each leg's per-month
    spread-trade return is computed by roll.roll_returns (which carries the leakage guards); each month's
    portfolio return is the MEAN of the legs that traded that month (a leg absent that month is excluded, never
    zero-filled). ``months`` optionally restricts to a window; otherwise the union of all legs' traded months."""
    by_sym = {}
    all_months = set()
    for sym, series in series_by_sym.items():
        ledger = {t["month"]: t["net"] for t in roll.roll_returns(series, signal, entry_bday, exit_bday, fee, months)}
        by_sym[sym] = ledger
        all_months.update(ledger)
    keys = list(months) if months is not None else sorted(all_months)
    out = []
    for key in keys:
        nets = [by_sym[sym][key] for sym in series_by_sym if key in by_sym[sym]]
        if nets:
            out.append({"month": key, "portfolio_ret": sum(nets) / len(nets), "n_legs": len(nets)})
    return out


def run(cfg):
    """Run one pooled-basket cell and return a trainer-contract RunSummary. The objective is the per-month OOS
    Sharpe of the EQUAL-WEIGHT energy roll-spread PORTFOLIO; there is no buy-and-hold benchmark (a spread
    portfolio has none), so the DSR-deflated Sharpe is the decider. mean_legs (average legs traded per month) is
    emitted so thin coverage is visible."""
    signal = str(cfg.get("signal", roll.DEFAULT_SIGNAL))
    if signal not in roll.SIGNALS:
        raise SystemExit(f"unknown signal {signal!r}; choose one of {sorted(roll.SIGNALS)}")
    window = str(cfg.get("walk_forward_window", next(iter(roll.ROLL_WINDOWS))))
    if window not in roll.ROLL_WINDOWS:
        raise SystemExit(f"unknown walk_forward_window {window!r}; choices: {sorted(roll.ROLL_WINDOWS)}")
    basket = str(cfg.get("basket", DEFAULT_BASKET))
    if basket not in BASKETS:
        raise SystemExit(f"unknown basket {basket!r}; choices: {sorted(BASKETS)}")
    entry_bday = int(cfg.get("entry_bday", roll.DEFAULT_ENTRY_BDAY))
    exit_bday = int(cfg.get("exit_bday", roll.DEFAULT_EXIT_BDAY))
    fee = float(cfg.get("transaction_cost", roll.DEFAULT_FEE))

    symbols = BASKETS[basket]
    series_by_sym = {sym: _load_series(sym) for sym in symbols}
    months = roll._window_months(window)
    ledger = basket_month_returns(series_by_sym, signal, entry_bday, exit_bday, fee, months=months)
    rets = [t["portfolio_ret"] for t in ledger]
    legs = [t["n_legs"] for t in ledger]
    n_trades = len(rets)

    equity = list(np.cumprod([1.0 + r for r in rets])) if rets else []
    total_return_pct = (equity[-1] - 1.0) * 100.0 if equity else 0.0

    metrics = {
        "total_return_pct": _finite(total_return_pct),
        "baseline": 0.0,
        "n_trades": n_trades,
        "n_months_traded": n_trades,
        "mean_legs": _finite(sum(legs) / len(legs)) if legs else 0.0,
        "final_net_worth": _finite(equity[-1]) if equity else 1.0,
        # per-month portfolio cost drag (average legs charged 4 spread legs each) is captured inside each leg's
        # net; realized_cost_bps here is the per-month round-trip cost of one average leg for visibility.
        "realized_cost_bps": _finite(fee * 4 * 10000),
    }
    oos = _oos_stats(equity)
    metrics.update(oos)
    metrics.update(_track_record_stats(oos))
    metrics.update(_max_drawdown_pct(equity))
    if rets:
        metrics["signal_expectancy"] = _finite(sum(rets) / len(rets) * 100.0)
        metrics["signal_hit_rate"] = _finite(100.0 * sum(1 for r in rets if r > 0) / len(rets))
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
            "asset": "ENERGY4",
            "timeframe": "monthly-roll",
            "candles": n_trades,
            "walk_forward_window": window,
            "from": ledger[0]["month"] if ledger else None,
            "to": ledger[-1]["month"] if ledger else None,
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
    parser = argparse.ArgumentParser(description="Pooled energy-basket index-roll calendar-spread backtest")
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
        f"n_trades={m.get('n_trades', 0)} mean_legs={m.get('mean_legs', 0.0):.2f} "
        f"signal_expectancy={m.get('signal_expectancy', 0.0):.4f} "
        f"total_return_pct={m.get('total_return_pct', 0.0):.4f} -> {args.summary_out}"
    )


if __name__ == "__main__":
    main()
