import math
import types

import numpy as np
import pytest

from trainer import summary as summary_mod
from trainer.summary import _oos_stats


# --- _oos_stats: Deflated-Sharpe inputs from the equity curve ----------------

def test_oos_stats_from_rising_equity_curve():
    eq = [100, 101, 102, 101.5, 103, 104, 103, 105]
    s = _oos_stats(eq)
    assert s["oos_n_obs"] == len(eq) - 1  # one return per step transition
    assert math.isfinite(s["oos_sharpe"]) and s["oos_sharpe"] > 0  # net-rising -> positive Sharpe
    assert math.isfinite(s["oos_ret_skew"]) and math.isfinite(s["oos_ret_kurt"])


def test_oos_stats_empty_when_too_short():
    assert _oos_stats([100]) == {}
    assert _oos_stats([100, 100]) == {}  # only one return -> < 2 -> empty (safe to .update())


def test_oos_stats_flat_curve_is_zero_sharpe():
    s = _oos_stats([100.0] * 8)
    assert s["oos_sharpe"] == 0.0 and s["oos_n_obs"] == 7


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


def test_baseline_metric_is_the_do_nothing_traded_return_floor():
    # The exploration autopilot's basin threshold reads metrics.baseline as the trivial reference a region
    # must beat. For the traded_return objective a do-nothing / buy-and-hold agent earns 0 (the trade gate
    # zeroes it), so the honest baseline is 0.0 — tightening the basin gate to "profitable trading".
    out, _, _ = _two_trade_summary()
    assert out["metrics"]["baseline"] == 0.0


def test_realized_cost_bps_from_env_fees():
    env = _FakeEnv(
        net_worths=[100000, 101000],
        actions=[1, 2],
        prices=[100, 110],
        positions=[1, 0],
    )
    env.fees = [60.0, 40.0]  # $100 total fees on a 100000 stake -> 10 bps
    out = summary_mod.build_summary(
        env, _state(n_trades=1), _CFG, _FakeModel(), "2026-01-01T00:00:00Z", True
    )
    assert out["metrics"]["realized_cost_bps"] == pytest.approx(10.0)


def test_realized_cost_bps_zero_without_fees():
    out, _, _ = _two_trade_summary()
    assert out["metrics"]["realized_cost_bps"] == 0.0


def test_checkpoint_artifact_only_for_models_that_produce_one():
    env = _two_trade_env()
    state = _state(n_trades=2, win=50.0, total_profit=2000.0, sls=1)

    class _RLModel:
        id = "rl-123"

        def produces_checkpoint(self):
            return True

    class _SupervisedModel:
        id = "sup-123"

        def produces_checkpoint(self):
            return False

    rl_out = summary_mod.build_summary(env, state, _CFG, _RLModel(), "2026-01-01T00:00:00Z", True)
    sup_out = summary_mod.build_summary(env, state, _CFG, _SupervisedModel(), "2026-01-01T00:00:00Z", True)
    assert rl_out.get("artifacts", {}).get("checkpoint") == "checkpoints/rl-123.zip"
    assert "checkpoint" not in sup_out.get("artifacts", {})


def test_checkpoint_artifact_omitted_when_save_checkpoint_disabled():
    # With checkpoint saving disabled (config.save_checkpoint=False) no file is written, so the summary must
    # NOT advertise the checkpoint artifact — else --evaluate / replay would try to load a missing file.
    env = _two_trade_env()
    state = _state(n_trades=2, win=50.0, total_profit=2000.0, sls=1)

    class _Cfg:
        def __init__(self, save):
            self.save_checkpoint = save

    class _RLModel:
        def __init__(self, save):
            self.id = "rl-123"
            self.config = _Cfg(save)

        def produces_checkpoint(self):
            return True

    off = summary_mod.build_summary(env, state, _CFG, _RLModel(False), "2026-01-01T00:00:00Z", True)
    on = summary_mod.build_summary(env, state, _CFG, _RLModel(True), "2026-01-01T00:00:00Z", True)
    assert "checkpoint" not in off.get("artifacts", {})
    assert on.get("artifacts", {}).get("checkpoint") == "checkpoints/rl-123.zip"


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


def test_blocked_signal_ratio_flags_out_of_position_noise():
    # buy(open)@0, a no-op buy while already long@1, agent sell(close)@2, hold@3 → 2 executed, 1 blocked.
    env = _FakeEnv(
        net_worths=[100000, 110000, 120000, 120000],
        actions=[1, 1, 2, 0],
        prices=[100, 110, 120, 115],
        positions=[1, 1, 0, 0],
        actions_made=[True, False, True, False],
        forced_actions=[0, 0, 0, 0],
    )
    out = summary_mod.build_summary(
        env, _state(n_trades=1, win=100.0, total_profit=20000.0), _CFG, _FakeModel(), "2026-01-01T00:00:00Z", True
    )
    m = out["metrics"]
    assert m["blocked_signals"] == 1
    assert m["executed_signals"] == 2
    assert m["blocked_signal_ratio"] == pytest.approx(1 / 3)
    assert m["signal_noise_pct"] == pytest.approx(25.0)


def test_blocked_signal_counts_forced_tpsl_close_as_an_agent_noop():
    # buy@0 executes; @1 the agent emits a buy (no-op while long) but a forced TP/SL (2) does the close —
    # the agent's own action was still a no-op, matching combo_noop_penalty's definition.
    env = _FakeEnv(
        net_worths=[100000, 110000, 100000],
        actions=[1, 1, 0],
        prices=[100, 110, 105],
        positions=[1, 0, 0],
        actions_made=[True, True, False],
        forced_actions=[0, 2, 0],
    )
    out = summary_mod.build_summary(
        env, _state(n_trades=1, win=100.0, total_profit=10000.0), _CFG, _FakeModel(), "2026-01-01T00:00:00Z", True
    )
    m = out["metrics"]
    assert m["blocked_signals"] == 1
    assert m["executed_signals"] == 1
    assert m["blocked_signal_ratio"] == pytest.approx(0.5)


def test_signal_noise_metrics_only_emitted_for_rl_runs():
    env = _FakeEnv(
        net_worths=[100000, 110000],
        actions=[1, 2],
        prices=[100, 110],
        actions_made=[True, True],
        forced_actions=[0, 0],
    )
    out = summary_mod.build_summary(env, _state(n_trades=1), _CFG, _FakeModel(), "2026-01-01T00:00:00Z", False)
    assert "blocked_signal_ratio" not in out["metrics"]


def test_signal_noise_empty_actions_is_none():
    env = _FakeEnv(net_worths=[], actions=[], prices=[], actions_made=[], forced_actions=[])
    assert summary_mod._signal_noise(env, 0) is None


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
    for key in ("sharpe", "cagr_pct", "sharpe_alpha",
                "worst_window_return_pct", "windows_profitable_pct"):
        assert key not in out["metrics"]


def test_max_drawdown_pct_is_emitted():
    # The one risk metric kept: the worst peak-to-trough decline of the test-window equity curve, as a
    # signed percent (<= 0). Powers the Diagnosis tab's risk lens and the combo_drawdown_penalty experiment.
    out, _, _ = _two_trade_summary()
    assert "max_drawdown_pct" in out["metrics"]
    assert out["metrics"]["max_drawdown_pct"] <= 0


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
    # "auto"/default resolves to the concrete set id in BOTH the dataset and the stored config.
    assert out["dataset"]["fidelity_set"] == "1d"
    assert out["dataset"]["layers"] == ["1d"]
    assert out["config"]["fidelity_set"] == "1d"


def test_trade_gate_is_quadratic():
    gate = summary_mod._trade_gate
    m = summary_mod.MIN_TRADES_FOR_FULL_CREDIT
    assert gate(m, m) == 1.0  # full credit at the bar
    assert gate(m * 2, m) == 1.0  # capped at 1.0 above the bar
    assert gate(m / 2, m) == pytest.approx(0.25)  # quadratic falloff below
    assert gate(0, m) == 0.0
    assert gate(5, 0) == 1.0  # disabled gate


def test_trade_gate_quadratically_gates_a_low_trade_run():
    out = _build({"timeframe": "1d", "lookback_window_size": 0}, n_trades=10)
    assert out["metrics"]["trade_gate"] == pytest.approx((10 / summary_mod.MIN_TRADES_FOR_FULL_CREDIT) ** 2)


def test_objective_is_the_honest_total_return_pct_not_the_gated_traded_return():
    # The objective is now the raw post-fee portfolio return (== metrics.total_return_pct), NOT the magic-20
    # trade-gated traded_return. Trade FREQUENCY is enforced by the trades_per_day scorecard gate instead, so
    # the objective no longer suppresses a low-trade run's real return. traded_return stays as a diagnostic.
    out, _, _ = _two_trade_summary()  # n_trades=2 -> trade_gate=0.01, so traded_return is 100x below the return
    assert out["objective"] == pytest.approx(out["metrics"]["total_return_pct"])
    assert out["objective"] != pytest.approx(out["metrics"]["traded_return"])


# --- trades_per_day: a step-frequency-invariant trade rate for the scorecard's liveness gate ---


def test_trades_per_day_is_trades_over_test_days_at_daily_step():
    # 1d step: one decision bar == one calendar day, so trades/day = n_trades / oos_n_obs (test bars).
    out, _, _ = _two_trade_summary()
    m = out["metrics"]
    assert m["trades_per_day"] == pytest.approx(m["n_trades"] / m["oos_n_obs"])


def test_trades_per_day_scales_with_step_frequency():
    # The SAME equity curve stepped hourly spans 1/24 the calendar days, so the daily trade rate is 24×
    # the daily-step rate — the rate is normalised by cadence, not just by bar count.
    state = _state(n_trades=2, win=50.0, total_profit=2000.0, sls=1)
    daily = summary_mod.build_summary(_two_trade_env(), state, {"timeframe": "1d", "lookback_window_size": 0}, _FakeModel(), "t", True)
    hourly = summary_mod.build_summary(_two_trade_env(), state, {"timeframe": "1h", "lookback_window_size": 0}, _FakeModel(), "t", True)
    assert hourly["metrics"]["trades_per_day"] == pytest.approx(daily["metrics"]["trades_per_day"] * 24)


# --- _signal_expectancy: the Case-1 position-blind signal lens (forward-return edge per buy/sell) ---


def test_signal_expectancy_scores_a_good_long_entry():
    # A long entry at step 0; price rises over the horizon -> positive forward edge, 100% hit rate.
    trades = [{"entry_step": 0, "exit_step": 5, "side": "long", "reason": "open"}]
    prices = [100, 101, 102, 103, 104, 105]  # H=2: fwd at step 0 = 102/100 - 1 = +2%
    sig = summary_mod._signal_expectancy(trades, prices, 2)
    assert sig["signal_count"] == 1
    assert sig["signal_expectancy"] == pytest.approx(2.0)
    assert sig["signal_hit_rate"] == 100.0
    assert sig["signal_horizon"] == 2


def test_signal_expectancy_penalises_a_bad_long_entry():
    # Long entry, price FALLS over the horizon -> negative edge, 0% hit.
    trades = [{"entry_step": 0, "exit_step": 5, "side": "long", "reason": "open"}]
    prices = [100, 99, 98, 97, 96, 95]
    sig = summary_mod._signal_expectancy(trades, prices, 2)
    assert sig["signal_expectancy"] < 0
    assert sig["signal_hit_rate"] == 0.0


def test_signal_expectancy_credits_a_well_timed_agent_exit():
    # Long entry@0 (price rises after = good) + an AGENT sell@2 after which price falls (good exit = +edge).
    trades = [{"entry_step": 0, "exit_step": 2, "side": "long", "reason": "sell"}]
    prices = [100, 105, 110, 108, 106, 104]  # H=2: entry fwd +10%; exit@2 fwd=106/110-1<0 -> exit edge = -(neg) > 0
    sig = summary_mod._signal_expectancy(trades, prices, 2)
    assert sig["signal_count"] == 2
    assert sig["signal_hit_rate"] == 100.0


def test_signal_expectancy_excludes_forced_tpsl_exits():
    # A stop-loss exit is a FORCED close, not the agent's directional signal — only the entry is scored.
    trades = [{"entry_step": 0, "exit_step": 2, "side": "long", "reason": "sl"}]
    prices = [100, 101, 102, 103, 104, 105]
    assert summary_mod._signal_expectancy(trades, prices, 2)["signal_count"] == 1


def test_signal_expectancy_coverage_is_agent_signals_over_bars():
    # Entry(open) + agent sell = 2 agent signals over 6 bars -> coverage 2/6; the entry alone is scorable at H=2.
    trades = [{"entry_step": 0, "exit_step": 1, "side": "long", "reason": "sell"}]
    sig = summary_mod._signal_expectancy(trades, [100, 102, 104, 106, 108, 110], 2)
    assert sig["signal_coverage"] == pytest.approx(2 / 6)


def test_signal_expectancy_empty_without_scorable_signals():
    assert summary_mod._signal_expectancy([], [100, 101, 102], 1) == {}
    # A signal too close to the end (no H bars ahead) can't be scored -> {}.
    assert summary_mod._signal_expectancy([{"entry_step": 5, "exit_step": 5, "side": "long", "reason": "open"}], [100, 101], 2) == {}


def test_build_summary_emits_the_signal_lens_for_a_trading_run():
    out, _, _ = _two_trade_summary()
    m = out["metrics"]
    for k in ("signal_expectancy", "signal_hit_rate", "signal_coverage", "signal_count", "signal_horizon"):
        assert k in m


def test_trades_per_day_zero_when_no_test_bars():
    # A too-short equity curve has no OOS bars — the rate must be 0.0, never a divide-by-zero blow-up.
    class _Prov:
        prices = None

        def get_price(self, i):
            return [100.0][i]

    env = types.SimpleNamespace(
        net_worths=[100000], actions=[1], actions_made=[True], forced_actions=[0],
        tpsls=[0], tpsl_kinds=[None], fees=[], initial_net_worth=100000.0,
        initial_balance=100000.0, data_provider=_Prov(),
    )
    out = summary_mod.build_summary(env, _state(n_trades=1), _CFG, _FakeModel(), "t", True)
    assert out["metrics"]["trades_per_day"] == 0.0


# --- _finite: only finite ints/floats pass; NaN/inf/non-numeric fall back to the default ---


def test_finite_passes_real_numbers_and_falls_back_otherwise():
    assert summary_mod._finite(3) == 3.0
    assert summary_mod._finite(2.5) == pytest.approx(2.5)
    assert summary_mod._finite(float("nan")) == 0.0
    assert summary_mod._finite(float("inf")) == 0.0
    assert summary_mod._finite("x") == 0.0
    assert summary_mod._finite(None, default=7.0) == 7.0


# --- _action_int: vector -> first element; scalar fallback; unparseable -> 0 ---


def test_action_int_handles_vectors_scalars_and_garbage():
    assert summary_mod._action_int([2, 9]) == 2
    assert summary_mod._action_int(np.array([3])) == 3
    assert summary_mod._action_int(4) == 4
    # A non-array, non-int-coercible value exhausts both fallbacks and yields 0.
    assert summary_mod._action_int("nope") == 0
    assert summary_mod._action_int(None) == 0


# --- _downsample / _downsample_indexed: short series pass through; long series stride down to cap ---


def test_downsample_short_series_is_unchanged():
    assert summary_mod._downsample([1, 2, 3]) == [1.0, 2.0, 3.0]


def test_downsample_long_series_caps_and_keeps_endpoints():
    big = list(range(500))
    out = summary_mod._downsample(big, cap=10)
    assert len(out) <= 10
    assert out[0] == 0.0
    assert out[-1] == 499.0  # the last point is always retained


def test_downsample_indexed_returns_kept_indices():
    big = list(range(500))
    vals, idx = summary_mod._downsample_indexed(big, cap=10)
    assert len(vals) == len(idx)
    assert idx[0] == 0 and idx[-1] == 499
    # Returned values correspond to the kept original indices.
    assert vals == [float(i) for i in idx]


def test_downsample_indexed_short_series_indexes_one_to_one():
    vals, idx = summary_mod._downsample_indexed([5, 6])
    assert vals == [5.0, 6.0]
    assert idx == [0, 1]


# --- _marker_x: map an original index onto the downsampled grid, snapping to the nearest kept point ---


def test_marker_x_snaps_to_nearest_kept_index():
    kept = [0, 5, 10]
    assert summary_mod._marker_x(0, kept) == 0
    assert summary_mod._marker_x(5, kept) == 1  # exact hit
    assert summary_mod._marker_x(7, kept) == 1  # closer to 5 than to 10
    assert summary_mod._marker_x(9, kept) == 2  # closer to 10
    # An index past the last kept point clamps to the final grid position.
    assert summary_mod._marker_x(100, kept) == 2


# --- _run_prices: prefers get_price(i), falls back to a prices array, else empty ---


def test_run_prices_returns_empty_without_provider():
    env = types.SimpleNamespace(data_provider=None)
    assert summary_mod._run_prices(env, 3) == []


def test_run_prices_uses_prices_attr_when_no_get_price():
    provider = types.SimpleNamespace(prices=[1, 2, 3, 4], get_price=None)
    env = types.SimpleNamespace(data_provider=provider)
    assert summary_mod._run_prices(env, 3) == [1.0, 2.0, 3.0]


def test_run_prices_get_price_short_series_falls_through_to_empty():
    # get_price raises after index 0 (<2 collected), and there's no usable prices array -> [].
    class _Prov:
        prices = None

        def get_price(self, i):
            if i > 0:
                raise IndexError()
            return 5.0

    env = types.SimpleNamespace(data_provider=_Prov())
    assert summary_mod._run_prices(env, 5) == []


def test_run_prices_get_price_happy_path():
    class _Prov:
        prices = None

        def get_price(self, i):
            return [100.0, 110.0, 120.0][i]

    env = types.SimpleNamespace(data_provider=_Prov())
    assert summary_mod._run_prices(env, 3) == [100.0, 110.0, 120.0]


# --- _run_chart: needs >=2 live actions AND >=2 prices, else None ---


def test_run_chart_none_when_too_few_actions():
    env = _FakeEnv(net_worths=[1], actions=[1], prices=[100, 110])
    assert summary_mod._run_chart(env, 0, []) is None


def test_run_chart_none_when_too_few_prices():
    class _Prov:
        prices = None

        def get_price(self, i):
            return [100.0][i]  # only index 0 -> <2 prices

    env = types.SimpleNamespace(actions=[1, 2], data_provider=_Prov())
    assert summary_mod._run_chart(env, 0, []) is None


# --- _benchmark: None when fewer than 2 positive finite prices; else the hold return ---


def test_benchmark_none_with_too_few_prices():
    env = _FakeEnv(net_worths=[1], actions=[1], prices=[100, 110])
    assert summary_mod._benchmark(env, 0) is None


def test_benchmark_filters_nonpositive_prices_to_none():
    env = _FakeEnv(net_worths=[1, 2], actions=[1, 2], prices=[0.0, 0.0])
    assert summary_mod._benchmark(env, 0) is None


def test_benchmark_hold_return_first_to_last():
    # No fee multiplier on the fake env → fee-free hold (the absent-fee case).
    env = _FakeEnv(net_worths=[1, 2], actions=[1, 2], prices=[100, 200])
    assert summary_mod._benchmark(env, 0) == {"hold_return_pct": pytest.approx(100.0)}


def test_benchmark_charges_entry_and_exit_fee():
    # A fair buy-and-hold pays the same per-trade fee the model does, on entry AND exit: (1-fee)^2.
    env = _FakeEnv(net_worths=[1, 2], actions=[1, 2], prices=[100, 200])
    env.transaction_fee_multiplier = 0.001
    expected = (200 / 100 * (1 - 0.001) ** 2 - 1) * 100
    assert summary_mod._benchmark(env, 0) == {"hold_return_pct": pytest.approx(expected)}


def test_benchmark_none_fee_multiplier_treated_as_zero():
    env = _FakeEnv(net_worths=[1, 2], actions=[1, 2], prices=[100, 200])
    env.transaction_fee_multiplier = None
    assert summary_mod._benchmark(env, 0) == {"hold_return_pct": pytest.approx(100.0)}


def test_build_summary_marks_hold_net_of_fees():
    env = _FakeEnv(net_worths=[100000, 110000], actions=[1, 2], prices=[100, 110])
    env.transaction_fee_multiplier = 0.001
    out = summary_mod.build_summary(
        env, _state(n_trades=2), _CFG, _FakeModel(), "2026-01-01T00:00:00Z", True
    )
    assert out["metrics"]["hold_net_of_fees"] is True
    # the benchmark is charged the round-trip fee, so it's below the gross price move
    assert out["metrics"]["hold_return_pct"] < 10.0


# --- _reconstruct_trades: empty / price-less envs return empty structures ---


def test_reconstruct_trades_empty_env():
    env = _FakeEnv(net_worths=[], actions=[], prices=[])
    assert summary_mod._reconstruct_trades(env, 0, 100000.0) == ([], [], [])


def test_reconstruct_trades_no_prices_returns_empty_trades():
    # Actions/net_worths present but get_price raises for every index -> no prices -> no trades.
    class _Prov:
        prices = None

        def get_price(self, i):
            raise IndexError()

    env = types.SimpleNamespace(
        net_worths=[100000, 101000],
        actions=[1, 2],
        actions_made=[True, True],
        forced_actions=[0, 0],
        tpsls=[0, 0],
        tpsl_kinds=[None, None],
        data_provider=_Prov(),
    )
    trades, equity, prices = summary_mod._reconstruct_trades(env, 0, 100000.0)
    assert trades == [] and equity == [] and prices == []


# --- _regime_windows / _regime_trend: guard short / no-initial inputs ---


def test_regime_windows_none_for_short_or_no_initial():
    assert summary_mod._regime_windows([], [100], 100000.0) is None
    assert summary_mod._regime_windows([], [100, 110, 120], 0) is None


def test_regime_trend_none_for_short_or_no_initial():
    assert summary_mod._regime_trend([], [100, 110], 100000.0) is None
    assert summary_mod._regime_trend([], [100, 110, 120], 0) is None


# --- _iso_from_ms: ms-since-epoch -> ISO UTC, None on garbage ---


def test_iso_from_ms_converts_and_handles_garbage():
    assert summary_mod._iso_from_ms(1577836800000) == "2020-01-01T00:00:00+00:00"
    assert summary_mod._iso_from_ms("not-a-number") is None
    assert summary_mod._iso_from_ms(None) is None


# --- _dataset: stamps asset/timeframe/window + from/to from provider timestamps ---


def test_dataset_includes_from_to_from_timestamps():
    provider = types.SimpleNamespace(timestamps=[1577836800000, 1580515200000])
    env = types.SimpleNamespace(data_provider=provider)
    d = summary_mod._dataset(
        env, {"asset": "ETHUSDT", "walk_forward_window": "2023"}, "1d", 100
    )
    assert d["asset"] == "ETHUSDT"
    assert d["candles"] == 100
    assert d["walk_forward_window"] == "2023"
    assert d["from"] == "2020-01-01T00:00:00+00:00"
    assert d["to"] == "2020-02-01T00:00:00+00:00"


def test_dataset_without_provider_or_timestamps_omits_from_to():
    d_none = summary_mod._dataset(types.SimpleNamespace(data_provider=None), {}, "1d", 5)
    assert "from" not in d_none and "to" not in d_none
    assert d_none["asset"] == "BTCUSDT"  # default
    # Provider present but no timestamps -> still no from/to.
    env = types.SimpleNamespace(data_provider=types.SimpleNamespace(timestamps=[]))
    d_empty = summary_mod._dataset(env, {}, "1d", 5)
    assert "from" not in d_empty and "to" not in d_empty


# --- _lookback: provider.get_lookback_window() wins, else cfg, else default 32 ---


def test_lookback_prefers_provider():
    provider = types.SimpleNamespace(get_lookback_window=lambda: 7)
    env = types.SimpleNamespace(data_provider=provider)
    assert summary_mod._lookback(env, {"lookback_window_size": 99}) == 7


def test_lookback_falls_back_to_cfg_then_default():
    env = types.SimpleNamespace(data_provider=None)
    assert summary_mod._lookback(env, {"lookback_window_size": 5}) == 5
    assert summary_mod._lookback(env, {}) == 32


def test_lookback_provider_error_falls_back_to_cfg():
    class _Prov:
        def get_lookback_window(self):
            raise RuntimeError("boom")

    env = types.SimpleNamespace(data_provider=_Prov())
    assert summary_mod._lookback(env, {"lookback_window_size": 9}) == 9


# --- _health: nan-metric, degenerate-policy, zero/few-trade flags ---


def test_health_flags_nan_metrics_and_degenerate_policy():
    state = [0.0] * 19
    state[1] = float("nan")  # a NaN core metric
    state[17] = 5
    env = types.SimpleNamespace(actions=[1, 1, 1])  # single repeated action -> degenerate policy
    health = summary_mod._health(env, state, True, 0)
    assert health["status"] == "degenerate"
    assert "nan_metrics" in health["flags"]
    assert "degenerate_policy" in health["flags"]


def test_health_flags_zero_trades():
    state = [0.0] * 19
    state[17] = 0
    env = types.SimpleNamespace(actions=[0, 1, 2])
    assert "zero_trades" in summary_mod._health(env, state, True, 0)["flags"]


def test_health_flags_few_trades():
    state = [0.0] * 19
    state[17] = summary_mod.DEGENERATE_TRADE_COUNT
    env = types.SimpleNamespace(actions=[0, 1, 2])
    flags = summary_mod._health(env, state, True, 0)["flags"]
    assert "few_trades" in flags and "zero_trades" not in flags


def test_health_non_rl_skips_policy_and_trade_flags():
    state = [0.0] * 19
    state[17] = 0
    env = types.SimpleNamespace(actions=[1, 1, 1])
    health = summary_mod._health(env, state, False, 0)
    assert health == {"status": "ok", "flags": []}


def test_health_ok_for_healthy_rl_run():
    state = [0.0] * 19
    state[17] = 25
    env = types.SimpleNamespace(actions=[0, 1, 2, 0, 2])  # varied actions
    assert summary_mod._health(env, state, True, 0) == {"status": "ok", "flags": []}


# --- build_summary fallback: too-few reconstructed equity points -> state[2] return + raw net_worths ---


def test_build_summary_falls_back_to_state_return_when_equity_too_short():
    class _Prov:
        prices = None

        def get_price(self, i):
            return [100.0][i]  # only one price -> reconstruction can't form an equity curve

    env = types.SimpleNamespace(
        net_worths=[100000],
        actions=[1],
        actions_made=[True],
        forced_actions=[0],
        tpsls=[0],
        tpsl_kinds=[None],
        fees=[],
        initial_net_worth=100000.0,
        initial_balance=100000.0,
        data_provider=_Prov(),
    )
    state = _state(n_trades=1)
    state[2] = 0.05  # 5% return reported by the env state
    out = summary_mod.build_summary(
        env, state, _CFG, _FakeModel(), "2026-01-01T00:00:00Z", True
    )
    assert out["metrics"]["total_return_pct"] == pytest.approx(5.0)
    # Equity falls back to the raw (live) net_worths.
    assert out["series"]["equity"] == [100000.0]


# --- build_summary: seed is stamped onto the summary and provenance when present in cfg ---


def test_build_summary_stamps_seed():
    out = _build({"timeframe": "1d", "lookback_window_size": 0, "seed": 42})
    assert out["seed"] == 42
    assert out["provenance"]["seed"] == 42


def test_build_summary_omits_seed_when_absent():
    out = _build({"timeframe": "1d", "lookback_window_size": 0})
    assert "seed" not in out
    assert "seed" not in out["provenance"]


# --- build_summary: a run_chart that raises is swallowed (no runChart artifact, no crash) ---


def test_build_summary_swallows_run_chart_errors(monkeypatch):
    monkeypatch.setattr(
        summary_mod, "_run_chart", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    out, _, _ = _two_trade_summary()
    assert "runChart" not in out.get("artifacts", {})


def test_run_chart_counts_authoritative_while_markers_dedup_on_grid():
    # Two same-type no-op attempts at original indices 1 & 2 both downsample to the same grid cell, so
    # the drawn markers collapse to one — but `counts` is tallied over the full series before the
    # dedup, so it must still report 2 (the legend can't under-count). Exercises the marker dedup path.
    n = 400  # > _MAX_SERIES_POINTS forces the downsample grid
    actions = [0] * n
    made = [True] * n
    actions[1] = 1
    made[1] = False  # no-op buy attempt
    actions[2] = 1
    made[2] = False  # adjacent no-op buy attempt -> same grid cell as index 1
    prices = [100.0 + i for i in range(n)]
    provider = types.SimpleNamespace(get_price=lambda i: prices[i])
    env = types.SimpleNamespace(actions=actions, actions_made=made, data_provider=provider)
    chart = summary_mod._run_chart(env, 0, [])
    assert chart["counts"]["buy_attempt"] == 2
    drawn = [m for m in chart["markers"] if m["type"] == "buy_attempt"]
    assert len(drawn) == 1


# --- _capture_stats: beta + up/down capture (A4.3 honesty guardrail inputs) --------------------------
# The engine's beta/up-vs-down-capture gate (kill the closet-long) reads these per-run scalars. Market = raw
# price return per step, model = equity return per step, classified by the SIGN of the market return.

def test_capture_stats_up_down_and_beta_on_a_known_series():
    # prices: +10% (up bar) then -10% (down bar); model equity: +5% then -4%.
    out = summary_mod._capture_stats([100.0, 105.0, 100.8], [100.0, 110.0, 99.0])
    assert out["up_capture"] == pytest.approx(0.5)  # +0.05 / +0.10
    assert out["down_capture"] == pytest.approx(0.4)  # -0.04 / -0.10  (<1 ⇒ defensive in the bear)
    assert out["beta"] == pytest.approx(0.45)  # cov(model,market) / var(market) over the two steps


def test_capture_stats_empty_when_degenerate():
    assert summary_mod._capture_stats([100.0, 100.0], [100.0, 110.0]) == {}  # < 3 aligned points
    assert summary_mod._capture_stats([100.0, 105.0], [100.0, 110.0, 99.0]) == {}  # misaligned lengths
    assert summary_mod._capture_stats([100.0] * 4, [100.0, 110.0, 99.0, 120.0]) == {}  # flat equity = do-nothing
    assert summary_mod._capture_stats([100.0, 101.0, 102.0], [100.0, 100.0, 100.0]) == {}  # no market variance


def test_capture_stats_emits_only_the_reachable_side():
    # Two UP bars only ⇒ up_capture present; down_capture absent (no down bar to divide by), beta absent (no
    # market variance). A partial emission is still safe to metrics.update().
    out = summary_mod._capture_stats([100.0, 105.0, 110.0], [100.0, 110.0, 121.0])
    assert "up_capture" in out
    assert "down_capture" not in out


def test_build_summary_emits_capture_metrics_for_a_trading_run():
    out, _, _ = _two_trade_summary()
    m = out["metrics"]
    assert "up_capture" in m and "down_capture" in m and "beta" in m
    assert all(math.isfinite(m[k]) for k in ("up_capture", "down_capture", "beta"))


def test_build_summary_omits_capture_metrics_for_a_do_nothing_run():
    # Flat equity + no trades ⇒ the capture helper returns {} so the engine's capture gate SKIPS the run
    # rather than reading a non-participating agent as maximally 'defensive' (down_capture 0).
    out = _build(
        _CFG, net_worths=[100000] * 6, actions=[0] * 6, prices=[100, 110, 105, 120, 115, 130], n_trades=0
    )
    for key in ("up_capture", "down_capture", "beta"):
        assert key not in out["metrics"]


def test_dead_feature_flag_detects_constant_observation_column():
    # L8: a feature column that never changes across the run (e.g. the known constant-0 z_score) is flagged
    # so the judge/human sees a dead/broken feature instead of it silently wasting obs dimensions.
    class P:
        def get_timesteps(self):
            return 100

        def get_values(self, step):
            return np.array([float(step), 7.0])  # col 0 varies; col 1 is constant (dead)

    flags = summary_mod._dead_feature_flags(types.SimpleNamespace(data_provider=P()), lookback=1)
    dead = [f for f in flags if f.startswith("dead_features:")]
    assert dead and int(dead[0].split(":")[1]) >= 1


def test_dead_feature_flag_silent_when_every_column_varies():
    class P:
        def get_timesteps(self):
            return 100

        def get_values(self, step):
            return np.array([float(step), float(step * 2)])

    assert summary_mod._dead_feature_flags(types.SimpleNamespace(data_provider=P()), lookback=1) == []


def test_provenance_fingerprint_stable_and_config_sensitive():
    # L5: the reproducibility fingerprint records a deterministic config hash + the resolved train/test span
    # so a run can be audited and re-derived. The hash is stable across identical configs and MUST change on
    # any lever change; the resolved span is stamped.
    cfg = {"walk_forward_window": "2024", "asset": "BTCUSDT", "seed": 0}
    stored = dict(cfg)
    fp = summary_mod._provenance_fingerprint(cfg, stored)
    assert "configHash" in fp
    assert summary_mod._provenance_fingerprint(cfg, stored)["configHash"] == fp["configHash"]
    assert summary_mod._provenance_fingerprint(cfg, {**stored, "seed": 1})["configHash"] != fp["configHash"]
    assert fp.get("trainFrom") == "2020-01" and fp.get("testFrom") == "2024-01"
    # dataVersion is hashed over the resolved (nested OmegaConf ListConfig) file list — must be stamped
    assert isinstance(fp.get("dataVersion"), str) and fp["dataVersion"]
    assert fp.get("dataFiles", 0) >= 1
