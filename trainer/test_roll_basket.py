"""Direct gating tests for the POOLED energy-basket roll probe — the power-recoverable follow-up to the
single-asset WTI roll null. It trades the SAME monthly M1-M2 spread across the four EIA energy legs as an
equal-weight PORTFOLIO, so diversification lowers the portfolio return variance and lifts the t-stat.

It reuses trainer/roll.py's per-commodity spread machinery wholesale (the four leakage guards are already
mutation-proven there), so the ONE new surface pinned here is the portfolio aggregation: each month is the
EQUAL-WEIGHT MEAN of the legs that actually traded that month, and a leg with no data that month is EXCLUDED
from the average, never zero-filled (which would silently dilute the portfolio return toward zero).
"""

import json

import numpy as np
import pytest

from trainer import roll, roll_basket


def _write(tmp_path, monkeypatch, sym, table):
    (tmp_path / f"{sym}.json").write_text(json.dumps(table))
    monkeypatch.setattr(roll_basket, "ENERGY_DIR", str(tmp_path))


def _month(year, month, n_days, m1_of, m2_of, start_day=3):
    out = {}
    for i in range(n_days):
        out[f"{year:04d}-{month:02d}-{start_day + i:02d}"] = {"m1": m1_of(i), "m2": m2_of(i)}
    return out


# --- the new surface: equal-weight aggregation, absent legs EXCLUDED not zero-filled -----------------


def test_portfolio_is_equal_weight_mean_of_present_legs(monkeypatch):
    ledgers = {
        "A": [{"month": "2006-01", "net": 0.10}, {"month": "2006-02", "net": 0.20}],
        "B": [{"month": "2006-01", "net": 0.30}, {"month": "2006-02", "net": -0.40}],
    }
    monkeypatch.setattr(roll, "roll_returns", lambda series, *a, **k: ledgers[series])
    out = roll_basket.basket_month_returns({"A": "A", "B": "B"}, "roll_frontrun", 2, 9, 0.0003)
    by = {r["month"]: r for r in out}
    assert by["2006-01"]["portfolio_ret"] == pytest.approx(0.20)   # mean(0.10, 0.30)
    assert by["2006-02"]["portfolio_ret"] == pytest.approx(-0.10)  # mean(0.20, -0.40)
    assert by["2006-01"]["n_legs"] == 2


def test_absent_leg_is_excluded_not_zero_filled(monkeypatch):
    ledgers = {
        "A": [{"month": "2006-01", "net": 0.10}, {"month": "2006-02", "net": 0.20}],
        "B": [{"month": "2006-01", "net": 0.30}],  # B has NO 2006-02 trade
    }
    monkeypatch.setattr(roll, "roll_returns", lambda series, *a, **k: ledgers[series])
    out = roll_basket.basket_month_returns({"A": "A", "B": "B"}, "roll_frontrun", 2, 9, 0.0003)
    by = {r["month"]: r for r in out}
    # 2006-02 averages ONLY the present leg A -> 0.20, n_legs 1; NOT (0.20 + 0)/2 = 0.10
    assert by["2006-02"]["portfolio_ret"] == pytest.approx(0.20)
    assert by["2006-02"]["n_legs"] == 1


def test_months_filter_restricts_the_series(monkeypatch):
    ledgers = {"A": [{"month": "2006-01", "net": 0.1}, {"month": "2007-05", "net": 0.2}]}
    monkeypatch.setattr(roll, "roll_returns", lambda series, *a, **k: ledgers[series])
    out = roll_basket.basket_month_returns({"A": "A"}, "roll_frontrun", 2, 9, 0.0003, months=["2006-01"])
    assert [r["month"] for r in out] == ["2006-01"]


# --- the run() contract (delegates validation to roll) -----------------------------------------------


def _ten_month_table(base_m1, carry0):
    table = {}
    for mo in range(1, 11):
        table.update(_month(
            2006, mo, 12,
            lambda i, mo=mo: base_m1 + 0.5 * mo - 0.1 * i,
            lambda i, mo=mo: base_m1 + 0.5 * mo - 0.1 * i + (carry0 - 0.03 * mo),
        ))
    return table


def test_run_emits_the_full_metric_vocabulary(tmp_path, monkeypatch):
    monkeypatch.setattr(roll_basket, "ENERGY_DIR", str(tmp_path))
    for sym, (b, c) in {"CRUDE": (60.0, 1.0), "HEATOIL": (2.0, 0.05), "RBOB": (2.2, 0.04), "NATGAS": (5.0, 0.2)}.items():
        (tmp_path / f"{sym}.json").write_text(json.dumps(_ten_month_table(b, c)))
    cfg = {
        "signal": "roll_frontrun", "entry_bday": 2, "exit_bday": 9, "roll_schedule": "gsci",
        "basket": "energy4", "transaction_cost": 0.0003, "walk_forward_window": "roll-2006-2011", "seed": 0,
    }
    summary = roll_basket.run(cfg)
    m = summary["metrics"]
    for key in (
        "total_return_pct", "oos_sharpe", "oos_n_obs", "signal_expectancy", "signal_hit_rate",
        "n_trades", "realized_cost_bps", "max_drawdown_pct", "psr", "mean_legs",
    ):
        assert key in m, f"missing metric {key}"
        assert np.isfinite(m[key])
    assert summary["objective"] == m["oos_sharpe"]
    assert m["n_trades"] == 10            # one portfolio round-trip per month with data
    assert m["mean_legs"] == pytest.approx(4.0)  # all four legs present every month
    assert summary["dataset"]["asset"] == "ENERGY4"
    assert summary["dataset"]["walk_forward_window"] == "roll-2006-2011"


def test_run_refuses_unknown_signal(tmp_path, monkeypatch):
    monkeypatch.setattr(roll_basket, "ENERGY_DIR", str(tmp_path))
    (tmp_path / "CRUDE.json").write_text(json.dumps(_month(2006, 1, 12, lambda i: 60.0, lambda i: 61.0)))
    with pytest.raises(SystemExit):
        roll_basket.run({"signal": "roll_wat", "walk_forward_window": "roll-2006-2011"})


def test_run_refuses_unknown_window(tmp_path, monkeypatch):
    monkeypatch.setattr(roll_basket, "ENERGY_DIR", str(tmp_path))
    (tmp_path / "CRUDE.json").write_text(json.dumps(_month(2006, 1, 12, lambda i: 60.0, lambda i: 61.0)))
    with pytest.raises(SystemExit):
        roll_basket.run({"signal": "roll_frontrun", "walk_forward_window": "roll-1999-2000"})
