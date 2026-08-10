"""Direct gating tests for the commodity index-roll ("Goldman roll") probe — the first NON-CRYPTO, non-price-
derived probe, and the first to pass the counterparty test with a genuinely NAMED price-insensitive forced flow
(passive GSCI/BCOM index funds mechanically rolling front-month longs into the next contract on a fixed schedule).

The trade is a market-neutral WTI M1-M2 CALENDAR SPREAD held across the roll window each month, so the surface
new here is (a) the two-contract spread and (b) the business-day roll calendar — both carrying leakage traps
pinned below. Guards mutation-proven here:

* the spread uses TWO DISTINCT contract months (M1-M2 of RAW settlements), never a back-adjusted continuous
  series that would bake the roll in — a continuous roll is a look-ahead machine;
* a month whose entry/exit day is missing EITHER leg is DROPPED, never forward-filled from a neighbouring day;
* entry business-day < exit business-day is enforced (a spread can't be entered after it's exited);
* each month's return reads ONLY that month's entry+exit settlements — corrupting any other date cannot move it.
"""

import json

import numpy as np
import pytest

from trainer import roll


def _write_series(tmp_path, monkeypatch, asset, table):
    """table: {date_iso: {"m1": .., "m2": .., ...}} -> written to <tmp>/<asset>.json, dir monkeypatched in."""
    (tmp_path / f"{asset}.json").write_text(json.dumps(table))
    monkeypatch.setattr(roll, "WTI_DIR", str(tmp_path))


def _month(year, month, n_days, m1_of, m2_of, start_day=3):
    """A month of consecutive daily rows (present dates == trading/business days, as real settlement data has)."""
    out = {}
    for i in range(n_days):
        d = f"{year:04d}-{month:02d}-{start_day + i:02d}"
        out[d] = {"m1": m1_of(i), "m2": m2_of(i)}
    return out


# --- loader: two legs required, missing leg dropped (no forward-fill) --------------------------------


def test_settlement_series_drops_dates_missing_a_leg(tmp_path, monkeypatch):
    table = {
        "2006-01-03": {"m1": 60.0, "m2": 61.0},
        "2006-01-04": {"m1": 60.5},            # missing M2 leg -> dropped, NOT forward-filled
        "2006-01-05": {"m1": 61.0, "m2": 62.0},
        "2006-01-06": {"m1": 0.0, "m2": 62.5}, # non-positive M1 -> dropped
    }
    _write_series(tmp_path, monkeypatch, "WTI", table)
    series = roll.settlement_series("WTI")
    dates = [d for d, _m1, _m2 in series]
    assert dates == ["2006-01-03", "2006-01-05"]  # only both-legs-present, positive, sorted


def test_missing_file_is_empty_series(tmp_path, monkeypatch):
    monkeypatch.setattr(roll, "WTI_DIR", str(tmp_path))
    assert roll.settlement_series("NOPE") == []


# --- business-day roll calendar: entry/exit = k-th trading date of the month -------------------------


def test_business_day_mapping_picks_kth_trading_date():
    rows = [(f"2006-01-{d:02d}", 60.0, 61.0) for d in range(3, 15)]  # 12 trading dates
    entry_row, exit_row = roll._trade_dates_for_month(rows, entry_bday=2, exit_bday=9)
    assert entry_row[0] == "2006-01-04"  # 2nd business day
    assert exit_row[0] == "2006-01-11"   # 9th business day


def test_month_too_short_is_not_traded():
    rows = [(f"2006-01-{d:02d}", 60.0, 61.0) for d in range(3, 8)]  # only 5 days
    assert roll._trade_dates_for_month(rows, entry_bday=2, exit_bday=9) is None


def test_entry_before_exit_is_enforced():
    for bad in ((9, 5), (5, 5), (0, 4)):
        with pytest.raises(SystemExit):
            roll.roll_returns([], "roll_frontrun", entry_bday=bad[0], exit_bday=bad[1], fee=0.0)


# --- the spread: two distinct contracts, direction, cost --------------------------------------------


def test_spread_uses_two_distinct_contracts_not_a_flat_continuous():
    # contango DEEPENS (M1 falls further below M2): spread -1 -> -2. Short-spread (frontrun) profits.
    entry = ("2006-01-04", 60.0, 61.0)  # spread_in  = -1.0
    exit_ = ("2006-01-11", 59.0, 61.0)  # spread_out = -2.0
    net, gross, cost = roll.spread_trade_return(entry, exit_, "roll_frontrun", fee=0.0)
    # delta = (spread_out - spread_in)/m1_in = (-2 - -1)/60 = -1/60 ; frontrun gross = -delta = +1/60
    assert gross == pytest.approx(1.0 / 60.0, rel=1e-9)
    assert cost == 0.0 and net == pytest.approx(1.0 / 60.0, rel=1e-9)
    # A back-adjusted CONTINUOUS series would make both legs the same number -> spread==0 -> gross==0.
    # This asserts the return genuinely comes from the M1-M2 gap, so collapsing the legs would flip it red.
    assert gross != 0.0


def test_frontrun_and_fade_are_exact_mirrors():
    entry = ("2006-01-04", 60.0, 61.0)
    exit_ = ("2006-01-11", 59.0, 61.5)
    fee = 0.0005
    nf, gf, cf = roll.spread_trade_return(entry, exit_, "roll_frontrun", fee=fee)
    nl, gl, cl = roll.spread_trade_return(entry, exit_, "roll_fade", fee=fee)
    assert gf == pytest.approx(-gl, rel=1e-9)      # opposite gross legs
    assert cf == pytest.approx(cl, rel=1e-9)       # same cost on both arms
    assert cf > 0.0                                # and cost is actually charged
    assert (nf + nl) == pytest.approx(-2.0 * cf, rel=1e-9)  # gross cancels, both pay cost


def test_cost_charges_both_legs_at_entry_and_exit():
    entry = ("2006-01-04", 50.0, 50.0)
    exit_ = ("2006-01-11", 50.0, 50.0)
    fee = 0.001
    _net, _gross, cost = roll.spread_trade_return(entry, exit_, "roll_frontrun", fee=fee)
    # cost = fee*(m1_in + m2_in + m1_out + m2_out)/m1_in = 0.001*(200)/50 = 0.004 (four legs)
    assert cost == pytest.approx(0.004, rel=1e-9)


def test_unknown_signal_is_refused():
    with pytest.raises(SystemExit):
        roll.spread_trade_return(("d", 60.0, 61.0), ("d", 60.0, 61.0), "roll_wat", fee=0.0)


# --- causality: a month's return reads ONLY its own entry+exit settlements ---------------------------


def _three_month_series():
    table = {}
    for mo in (1, 2, 3):
        table.update(_month(2006, mo, 12, lambda i, mo=mo: 60.0 + mo - 0.1 * i, lambda i, mo=mo: 61.5 + mo - 0.05 * i))
    return roll.settlement_series_from_table(table)


def test_month_return_is_time_prefix_causal():
    series = _three_month_series()
    base = roll.roll_returns(series, "roll_frontrun", entry_bday=2, exit_bday=9, fee=0.0003)
    by_month = {t["month"]: t for t in base}
    m2 = by_month["2006-02"]
    entry_d, exit_d = m2["entry_date"], m2["exit_date"]

    def _perturb(target_date):
        s2 = [(d, (999.0 if d == target_date else m1), m2v) for d, m1, m2v in series]
        got = {t["month"]: t for t in roll.roll_returns(s2, "roll_frontrun", 2, 9, 0.0003)}
        return got["2006-02"]["net"]

    # a date in Jan (a different month), and a MIDDLE day of Feb that is neither entry nor exit -> no effect
    assert _perturb("2006-01-06") == pytest.approx(m2["net"], rel=1e-12)
    middle = next(d for d, _a, _b in series if d.startswith("2006-02") and d not in (entry_d, exit_d))
    assert _perturb(middle) == pytest.approx(m2["net"], rel=1e-12)
    # corrupting Feb's OWN exit settlement DOES move it (proves it reads exactly that day)
    assert _perturb(exit_d) != pytest.approx(m2["net"], rel=1e-9)


# --- the run() contract ------------------------------------------------------------------------------


def test_unknown_window_is_refused(tmp_path, monkeypatch):
    _write_series(tmp_path, monkeypatch, "WTI", _month(2006, 1, 12, lambda i: 60.0, lambda i: 61.0))
    with pytest.raises(SystemExit):
        roll.run({"signal": "roll_frontrun", "walk_forward_window": "roll-1999-2000"})


def _ten_month_table():
    table = {}
    for mo in range(1, 11):
        table.update(_month(
            2006, mo, 12,
            lambda i, mo=mo: 60.0 + 0.5 * mo - 0.1 * i,
            lambda i, mo=mo: 60.0 + 0.5 * mo - 0.1 * i + (1.0 - 0.03 * mo),  # contango that narrows across months
        ))
    return table


def test_run_emits_the_full_metric_vocabulary(tmp_path, monkeypatch):
    _write_series(tmp_path, monkeypatch, "WTI", _ten_month_table())
    cfg = {
        "asset": "WTI", "signal": "roll_frontrun", "entry_bday": 2, "exit_bday": 9,
        "roll_schedule": "gsci", "transaction_cost": 0.0003, "walk_forward_window": "roll-2006-2011", "seed": 0,
    }
    summary = roll.run(cfg)
    m = summary["metrics"]
    for key in (
        "total_return_pct", "oos_sharpe", "oos_n_obs", "signal_expectancy", "signal_hit_rate",
        "n_trades", "realized_cost_bps", "max_drawdown_pct", "psr",
    ):
        assert key in m, f"missing metric {key}"
        assert np.isfinite(m[key])
    assert summary["objective"] == m["oos_sharpe"]
    assert m["n_trades"] == 10  # one round-trip per month with data in the window
    assert summary["dataset"]["walk_forward_window"] == "roll-2006-2011"
    assert "provenance" in summary and "config" in summary


def test_run_refuses_unknown_signal(tmp_path, monkeypatch):
    _write_series(tmp_path, monkeypatch, "WTI", _month(2006, 1, 12, lambda i: 60.0, lambda i: 61.0))
    with pytest.raises(SystemExit):
        roll.run({"signal": "roll_wat", "walk_forward_window": "roll-2006-2011"})
