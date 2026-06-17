import pytest

from trainer import summary as summary_mod


class _FakeProvider:
    def __init__(self, prices, lookback=0, timestamps=None):
        self._prices = list(prices)
        self._lookback = lookback
        self.timestamps = list(timestamps) if timestamps is not None else []

    def get_price(self, i):
        return self._prices[i]

    def get_lookback_window(self):
        return self._lookback


class _FakeEnv:
    def __init__(self, net_worths, actions, prices, tpsls=None, positions=None,
                 tpsl_kinds=None, actions_made=None, forced_actions=None,
                 lookback=0, timestamps=None, initial=100000.0):
        self.net_worths = list(net_worths)
        self.actions = list(actions)
        self.tpsls = list(tpsls) if tpsls is not None else [0] * len(actions)
        self.positions = list(positions) if positions is not None else []
        self.tpsl_kinds = list(tpsl_kinds) if tpsl_kinds is not None else [None] * len(actions)
        self.actions_made = list(actions_made) if actions_made is not None else []
        self.forced_actions = list(forced_actions) if forced_actions is not None else [0] * len(actions)
        self.initial_net_worth = initial
        self.initial_balance = initial
        self.data_provider = _FakeProvider(prices, lookback, timestamps)


class _FakeModel:
    id = None


def _state(n_trades=25, win=50.0, total_profit=0.0, sls=0, initial=100000.0):
    s = [0.0] * 19
    s[1] = total_profit
    s[2] = total_profit / initial if initial else 0.0
    s[7] = win
    s[17] = n_trades
    s[18] = sls
    return s


def _build(cfg, net_worths=None, actions=None, prices=None, n_trades=25):
    if net_worths is None:
        net_worths = [100000, 101000, 100500, 102000, 101800, 103000]
    if actions is None:
        actions = [0, 1, 0, 2, 1, 0]
    if prices is None:
        prices = [100, 110, 105, 120, 115, 130]
    env = _FakeEnv(net_worths, actions, prices)
    return summary_mod.build_summary(
        env, _state(n_trades=n_trades), cfg, _FakeModel(), "2026-01-01T00:00:00Z", True
    )


# --- two closed long round-trips: an agent-sell win then a stop-loss, ending flat ---
_CFG = {"timeframe": "1d", "lookback_window_size": 0}


def _two_trade_env():
    return _FakeEnv(
        net_worths=[100000, 110000, 100000, 100000, 92000, 100000],
        actions=[1, 2, 0, 1, 0, 0],
        prices=[100, 110, 105, 90, 95, 100],
        tpsls=[0, 0, 0, 0, -1, 0],
        positions=[1, 0, 0, 1, 0, 0],
        tpsl_kinds=[None, None, None, None, "sl", None],
        actions_made=[True, True, False, True, True, False],
        forced_actions=[0, 0, 0, 0, 2, 0],
    )


def _two_trade_summary():
    env = _two_trade_env()
    state = _state(n_trades=2, win=50.0, total_profit=2000.0, sls=1)
    return summary_mod.build_summary(env, state, _CFG, _FakeModel(), "2026-01-01T00:00:00Z", True), env, state


def test_reconstructs_round_trips_with_reasons_and_pnl():
    out, _, _ = _two_trade_summary()
    ledger = out["ledger"]
    assert [t["reason"] for t in ledger] == ["sell", "sl"]
    assert [t["side"] for t in ledger] == ["long", "long"]
    assert ledger[0]["entry_step"] == 0 and ledger[0]["exit_step"] == 1
    assert ledger[0]["pnl"] == pytest.approx(10000.0)
    assert ledger[1]["pnl"] == pytest.approx(-8000.0)


def test_total_return_is_realized_sum_not_curve_endpoint():
    out, _, _ = _two_trade_summary()
    # +10000 then -8000 on a 100000 stake = +2% realized; the raw net_worths endpoint is 100000 (0%).
    assert out["metrics"]["total_return_pct"] == pytest.approx(2.0)
    assert out["series"]["equity"][-1] == pytest.approx(102000.0)
    assert out["metrics"]["final_net_worth"] == pytest.approx(102000.0)


def test_closed_trade_count_matches_env_n_trades():
    out, _, state = _two_trade_summary()
    closed = [t for t in out["ledger"] if t["reason"] != "open"]
    assert len(closed) == state[17]


def test_closed_pnl_matches_env_total_profit():
    out, _, state = _two_trade_summary()
    closed_pnl = sum(t["pnl"] for t in out["ledger"] if t["reason"] != "open")
    assert closed_pnl == pytest.approx(state[1])


def test_sl_count_matches_env_stop_losses():
    out, _, state = _two_trade_summary()
    sl = [t for t in out["ledger"] if t["reason"] == "sl"]
    assert len(sl) == state[18]


def test_realized_equals_equity_endpoint():
    out, _, _ = _two_trade_summary()
    initial = 100000.0
    realized = sum(t["pnl"] for t in out["ledger"])
    assert realized / initial * 100 == pytest.approx(out["metrics"]["total_return_pct"])


def test_exit_breakdown_counts_and_winrate():
    out, _, _ = _two_trade_summary()
    exits = out["exits"]
    assert exits["sell"]["count"] == 1 and exits["sell"]["win_pct"] == 100.0
    assert exits["sell"]["total_pnl_pct"] == pytest.approx(10.0)
    assert exits["sl"]["count"] == 1 and exits["sl"]["win_pct"] == 0.0
    assert exits["sl"]["total_pnl_pct"] == pytest.approx(-8.0)


def test_regime_windows_attribute_trades_by_exit_step():
    env = _two_trade_env()
    trades, _, prices = summary_mod._reconstruct_trades(env, 0, 100000.0)
    windows = summary_mod._regime_windows(trades, prices, 100000.0, n_windows=2)
    assert len(windows) == 2
    assert windows[0]["n_trades"] == 1 and windows[0]["realized_pnl_pct"] == pytest.approx(10.0)
    assert windows[1]["n_trades"] == 1 and windows[1]["realized_pnl_pct"] == pytest.approx(-8.0)


def test_regime_trend_buckets_present_and_sum_trades():
    env = _two_trade_env()
    trades, _, prices = summary_mod._reconstruct_trades(env, 0, 100000.0)
    trend = summary_mod._regime_trend(trades, prices, 100000.0)
    assert set(trend) == {"up", "flat", "down"}
    assert sum(trend[k]["n_trades"] for k in trend) == len(trades)


def test_open_position_at_end_is_implied_sold():
    # A short opened and never closed: marked to the last price as an implied close.
    env = _FakeEnv(
        net_worths=[100000, 100000, 120000],
        actions=[3, 0, 0],
        prices=[100, 100, 80],
        tpsls=[0, 0, 0],
        positions=[-1, -1, -1],
        tpsl_kinds=[None, None, None],
        actions_made=[True, False, False],
        forced_actions=[0, 0, 0],
    )
    state = _state(n_trades=0, win=0.0, total_profit=0.0, sls=0)
    out = summary_mod.build_summary(env, state, _CFG, _FakeModel(), "2026-01-01T00:00:00Z", True)
    ledger = out["ledger"]
    assert len(ledger) == 1
    assert ledger[0]["reason"] == "open" and ledger[0]["side"] == "short"
    assert ledger[0]["pnl"] == pytest.approx(20000.0)
    assert out["metrics"]["total_return_pct"] == pytest.approx(20.0)
    # The implied (never-closed) trade is excluded from the env's realized n_trades.
    assert len([t for t in ledger if t["reason"] != "open"]) == state[17]


def test_trailing_take_profit_is_its_own_bucket():
    env = _FakeEnv(
        net_worths=[100000, 120000, 128000],
        actions=[1, 0, 0],
        prices=[100, 120, 130],
        tpsls=[0, 0, 1],
        positions=[1, 1, 0],
        tpsl_kinds=[None, None, "trailing"],
        actions_made=[True, False, True],
        forced_actions=[0, 0, 2],
    )
    state = _state(n_trades=1, win=100.0, total_profit=28000.0, sls=0)
    out = summary_mod.build_summary(env, state, _CFG, _FakeModel(), "2026-01-01T00:00:00Z", True)
    assert [t["reason"] for t in out["ledger"]] == ["trailing"]
    assert "trailing" in out["exits"] and "tp" not in out["exits"]


def test_regular_take_profit_bucket_distinct_from_trailing():
    env = _FakeEnv(
        net_worths=[100000, 120000, 128000],
        actions=[1, 0, 0],
        prices=[100, 120, 130],
        tpsls=[0, 0, 1],
        positions=[1, 1, 0],
        tpsl_kinds=[None, None, "tp"],
        actions_made=[True, False, True],
        forced_actions=[0, 0, 2],
    )
    state = _state(n_trades=1, win=100.0, total_profit=28000.0, sls=0)
    out = summary_mod.build_summary(env, state, _CFG, _FakeModel(), "2026-01-01T00:00:00Z", True)
    assert [t["reason"] for t in out["ledger"]] == ["tp"]


def test_run_chart_counts_executed_and_attempts_authoritatively():
    # buy(open) @0, a no-op buy attempt while already long @1, agent sell(close) @2, hold @3.
    env = _FakeEnv(
        net_worths=[100000, 110000, 120000, 120000],
        actions=[1, 1, 2, 0],
        prices=[100, 110, 120, 115],
        tpsls=[0, 0, 0, 0],
        positions=[1, 1, 0, 0],
        tpsl_kinds=[None, None, None, None],
        actions_made=[True, False, True, False],
    )
    state = _state(n_trades=1, win=100.0, total_profit=20000.0, sls=0)
    out = summary_mod.build_summary(env, state, _CFG, _FakeModel(), "2026-01-01T00:00:00Z", True)
    counts = out["artifacts"]["runChart"]["counts"]
    assert counts.get("buy") == 1
    assert counts.get("sell") == 1
    assert counts.get("buy_attempt") == 1


def test_run_chart_marks_short_and_cover():
    env = _FakeEnv(
        net_worths=[100000, 100000, 120000],
        actions=[3, 0, 4],
        prices=[100, 90, 80],
        tpsls=[0, 0, 0],
        positions=[-1, -1, 0],
        tpsl_kinds=[None, None, None],
        actions_made=[True, False, True],
    )
    state = _state(n_trades=1, win=100.0, total_profit=20000.0, sls=0)
    out = summary_mod.build_summary(env, state, _CFG, _FakeModel(), "2026-01-01T00:00:00Z", True)
    counts = out["artifacts"]["runChart"]["counts"]
    assert counts.get("short") == 1
    assert counts.get("cover") == 1


def test_risk_metrics_are_dropped():
    out, _, _ = _two_trade_summary()
    for key in ("sharpe", "cagr_pct", "max_drawdown_pct", "sharpe_alpha",
                "worst_window_return_pct", "windows_profitable_pct"):
        assert key not in out["metrics"]


def test_benchmark_is_hold_return_only():
    out, _, _ = _two_trade_summary()
    assert "hold_return_pct" in out["benchmark"]
    assert "hold_sharpe" not in out["benchmark"]
    assert "hold_max_drawdown_pct" not in out["benchmark"]
    assert "hold_return_pct" in out["metrics"]


def test_dataset_stamps_the_walk_forward_window():
    out = _build({"timeframe": "1d", "walk_forward_window": "2023", "lookback_window_size": 0})
    assert out["dataset"]["walk_forward_window"] == "2023"


def test_dataset_window_defaults_to_2024_when_absent():
    out = _build({"timeframe": "1d", "lookback_window_size": 0})
    assert out["dataset"]["walk_forward_window"] == "2024"


def test_dataset_stamps_fidelity_set_and_layers():
    out = _build({"timeframe": "1h", "fidelity_set": "1h+1d+1w", "lookback_window_size": 0})
    assert out["dataset"]["fidelity_set"] == "1h+1d+1w"
    assert out["dataset"]["layers"] == ["1h", "1d", "1w"]
    assert out["dataset"]["timeframe"] == "1h"


def test_dataset_fidelity_defaults_from_timeframe():
    out = _build({"timeframe": "1d", "lookback_window_size": 0})
    assert out["dataset"]["fidelity_set"] == "1d"
    assert out["dataset"]["layers"] == ["1d"]


def test_trade_gate_modes():
    gate = summary_mod._trade_gate
    m = summary_mod.MIN_TRADES_FOR_FULL_CREDIT
    assert gate(m, "quadratic", m) == 1.0
    assert gate(m / 2, "quadratic", m) == pytest.approx(0.25)
    assert gate(m / 2, "linear", m) == pytest.approx(0.5)
    assert gate(m * 2, "linear", m) == 1.0
    assert gate(m - 1, "threshold", m) == 0.0
    assert gate(m, "threshold", m) == 1.0
    assert gate(0, "none", m) == 1.0
    assert gate(5, "quadratic", 0) == 1.0


def test_trade_gate_mode_none_ungates_a_low_trade_run():
    out = _build({"timeframe": "1d", "trade_gate_mode": "none", "lookback_window_size": 0}, n_trades=3)
    assert out["metrics"]["trade_gate"] == 1.0
    assert out["objective"] == pytest.approx(out["metrics"]["total_return_pct"])


def test_trade_gate_mode_defaults_to_quadratic():
    out = _build({"timeframe": "1d", "lookback_window_size": 0}, n_trades=10)
    assert out["metrics"]["trade_gate"] == pytest.approx((10 / summary_mod.MIN_TRADES_FOR_FULL_CREDIT) ** 2)
