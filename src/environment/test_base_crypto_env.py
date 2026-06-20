"""Direct tests for the additive ``tpsl_kinds`` exit-reason tag on BaseCryptoEnv.

These exercise ``resolve_tpsl`` / ``resolve_action`` in isolation via ``__new__`` (bypassing the
heavy data-provider __init__), so they assert ONLY the tag wiring without changing trading behaviour.
Plain asserts (no pytest dependency) so they run under the project venv, which has no pytest.
"""

import math
import os
import sys
import types

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.environment.base_crypto_env import BaseCryptoEnv, _concat_layer_grids


def _env(position, net_worth, *, entry=100.0, price=130.0, highest=140.0, lowest=0.0,
         take_profit=0.1, trailing=0.05, stop_loss=0.1):
    e = BaseCryptoEnv.__new__(BaseCryptoEnv)
    e.env_config = types.SimpleNamespace(
        take_profit=take_profit, trailing_take_profit=trailing, stop_loss=stop_loss
    )
    e.positions = [position]
    e.initial_net_worth = 100000.0
    e.current_step = 0
    e.current_price = price
    e.position_price_entry = entry
    e.position_price_highest = highest
    e.position_price_lowest = lowest
    e._calculate_net_worth = lambda step: net_worth
    return e


def test_resolve_tpsl_flat_returns_none_triplet():
    forced, tp, kind = _env(0, 100000.0).resolve_tpsl()
    assert (forced, tp, kind) == (0, None, None)


def test_resolve_tpsl_trailing_kind():
    # long, profit 5% (< take_profit 10% so not a plain TP); highest 140 >= activation 110 and
    # price 130 <= 140*(1-0.05)=133 -> trailing stop fires.
    forced, tp, kind = _env(10.0, 105000.0, highest=140.0, price=130.0).resolve_tpsl()
    assert forced == 2 and tp is True and kind == "trailing"


def test_resolve_tpsl_regular_take_profit_kind():
    # highest below activation so no trailing; net worth +12% >= take_profit 10% -> plain tp.
    forced, tp, kind = _env(10.0, 112000.0, highest=105.0).resolve_tpsl()
    assert forced == 2 and tp is True and kind == "tp"


def test_resolve_tpsl_stop_loss_kind():
    # no take-profit hit; loss 15% >= stop_loss 10% -> sl.
    forced, tp, kind = _env(10.0, 85000.0, highest=105.0).resolve_tpsl()
    assert forced == 2 and tp is False and kind == "sl"


def test_resolve_tpsl_short_close_action_is_4():
    # short position closes/covers with action 4; trailing mirrored on the lows.
    forced, tp, kind = _env(-10.0, 105000.0, entry=100.0, price=80.0, lowest=70.0).resolve_tpsl()
    assert forced == 4 and tp is True and kind == "trailing"


def _action_env():
    e = BaseCryptoEnv.__new__(BaseCryptoEnv)
    e.env_config = types.SimpleNamespace(take_profit=0.1, trailing_take_profit=0.05, stop_loss=0.1)
    e.positions = [10.0]
    e.initial_net_worth = 100000.0
    e.current_step = 0
    e.current_price = 130.0
    e.position_price_entry = 100.0
    e.position_price_highest = 140.0
    e.position_price_lowest = 0.0
    e._calculate_net_worth = lambda step: 105000.0
    e.tpsls, e.tpsl_kinds, e.actions, e.actions_made, e.forced_actions = [], [], [], [], []
    return e


def test_resolve_action_records_forced_trailing_kind():
    e = _action_env()
    e.take_action = lambda action: action in (2, 4)  # only a close executes
    e.resolve_action(0)  # agent holds; trailing stop forces a close
    assert e.tpsls == [1]
    assert e.tpsl_kinds == ["trailing"]
    assert e.forced_actions == [2]
    assert e.actions_made == [True]


def test_resolve_action_agent_close_records_none_kind():
    e = _action_env()
    e.take_action = lambda action: True  # the agent's own action executes
    e.resolve_action(2)  # agent closes the long itself; no forced TP/SL
    assert e.tpsls == [0]
    assert e.tpsl_kinds == [None]
    assert e.forced_actions == [0]


def test_concat_layer_grids_single_2d_returned_as_is():
    # SingleDataProvider with lookback>1 returns ONE [lookback, per_bar] grid; np.concatenate(_, axis=1)
    # would iterate it into 1-D rows and raise — the helper must return it unchanged.
    grid = np.arange(6).reshape(2, 3)
    out = _concat_layer_grids(grid)
    assert out.shape == (2, 3)
    assert np.array_equal(out, grid)


def test_concat_layer_grids_multi_3d_merges_layers():
    # MultiTimelineDataProvider returns a 3-D [n_layers, lookback, per_bar_layer] array.
    grids = np.arange(12).reshape(2, 2, 3)  # 2 layers, lookback 2, 3 features each
    out = _concat_layer_grids(grids)
    assert out.shape == (2, 6)  # layers merged onto the feature axis


def test_concat_layer_grids_list_of_grids_merges():
    out = _concat_layer_grids([np.zeros((2, 3)), np.ones((2, 4))])
    assert out.shape == (2, 7)


def test_update_reward_records_additive_components_summing_to_the_reward():
    e = BaseCryptoEnv.__new__(BaseCryptoEnv)
    e._calculate_reward = lambda: 5.0
    e._turnover_penalty = lambda: 0.5
    e._noop_penalty = lambda: 0.2
    e.total_reward = 0.0
    e.rewards = []
    e.rewards_history = []
    e.reward_components = []
    reward = e.update_reward()
    assert abs(reward - (5.0 - 0.5 - 0.2)) < 1e-9
    comp = e.reward_components[-1]
    assert comp == {"base": 5.0, "turnover_penalty": -0.5, "noop_penalty": -0.2}
    assert abs(sum(comp.values()) - reward) < 1e-9  # components are additive contributions


# ---------------------------------------------------------------------------
# Direct method tests against BaseCryptoEnv via __new__ bypass + SimpleNamespace
# fakes. These exercise the deterministic trade/PnL/reward/observation logic
# WITHOUT the heavy data-provider __init__, a model, or any training loop.
# pytest.approx is used for float comparisons (no fixtures, so _run_all stays valid).
# ---------------------------------------------------------------------------

import pytest


def _bare():
    """A BaseCryptoEnv with no __init__ run; attributes are set per-test."""
    return BaseCryptoEnv.__new__(BaseCryptoEnv)


# --- _calculate_net_worth (offset by lookback window) -----------------------


def test_calculate_net_worth_lookback1():
    # offset_step = step + lookback - 1 = 0; net = balance + position*price.
    e = _bare()
    e.data_provider = types.SimpleNamespace(get_lookback_window=lambda: 1)
    e.balances = [1000.0]
    e.positions = [2.0]
    e.get_price = lambda step: 50.0
    assert e._calculate_net_worth(0) == pytest.approx(1100.0)


def test_calculate_net_worth_uses_lookback_offset():
    # lookback 3, step 1 -> offset_step = 1 + 3 - 1 = 3 reads balances/positions[3].
    e = _bare()
    e.data_provider = types.SimpleNamespace(get_lookback_window=lambda: 3)
    e.balances = [10.0, 20.0, 30.0, 40.0]
    e.positions = [0.0, 0.0, 0.0, 5.0]
    e.get_price = lambda step: 2.0
    assert e._calculate_net_worth(1) == pytest.approx(40.0 + 5.0 * 2.0)


def test_calculate_net_worth_negative_position_is_short():
    # A short (negative position) loses value as price rises: net = balance - |pos|*price.
    e = _bare()
    e.data_provider = types.SimpleNamespace(get_lookback_window=lambda: 1)
    e.balances = [200.0]
    e.positions = [-3.0]
    e.get_price = lambda step: 50.0
    assert e._calculate_net_worth(0) == pytest.approx(200.0 - 150.0)


# --- _calculate_drawdown ----------------------------------------------------


def test_calculate_drawdown_zero_when_no_peak():
    e = _bare()
    e.drawdown_peak = 0
    e.drawdown_trough = 0
    assert e._calculate_drawdown() == 0


def test_calculate_drawdown_ratio_minus_one():
    e = _bare()
    e.drawdown_peak = 100.0
    e.drawdown_trough = 80.0
    assert e._calculate_drawdown() == pytest.approx(-0.2)


# --- update_drawdown (long peak/trough + short via inverse price) ------------


def test_update_drawdown_long_new_peak_resets_trough():
    e = _bare()
    e.positions = [5.0]
    e.drawdown_peak = 0
    e.drawdown_trough = 0
    e.drawdowns = []
    e.current_price = 100.0
    e.update_drawdown()
    assert e.drawdown_peak == 100.0 and e.drawdown_trough == 100.0
    assert e.drawdowns[-1] == 0


def test_update_drawdown_long_falls_to_trough():
    e = _bare()
    e.positions = [5.0]
    e.drawdown_peak = 100.0
    e.drawdown_trough = 100.0
    e.drawdowns = []
    e.current_price = 80.0  # adverse for a long -> trough drops
    e.update_drawdown()
    assert e.drawdown_trough == 80.0
    assert e.drawdowns[-1] == pytest.approx(-0.2)


def test_update_drawdown_short_uses_inverse_price():
    # For a short the adverse direction is price rising; tracked via inverse price.
    e = _bare()
    e.positions = [-5.0]
    e.drawdown_peak = 0
    e.drawdown_trough = 0
    e.drawdowns = []
    e.current_price = 100.0  # inv 0.01 -> peak
    e.update_drawdown()
    e.current_price = 50.0  # price fell (favourable): inv 0.02 > peak -> new peak, trough resets
    e.update_drawdown()
    assert e.drawdown_peak == pytest.approx(0.02)
    e.current_price = 200.0  # price rose (adverse): inv 0.005 < trough -> drawdown deepens (line 370)
    e.update_drawdown()
    assert e.drawdown_trough == pytest.approx(0.005)
    assert e.drawdowns[-1] == pytest.approx(0.005 / 0.02 - 1.0)


def test_update_drawdown_short_zero_price_inv_guard():
    # A zero current price must not divide-by-zero; inv falls back to 0.
    e = _bare()
    e.positions = [-5.0]
    e.drawdown_peak = 0.5
    e.drawdown_trough = 0.5
    e.drawdowns = []
    e.current_price = 0.0
    e.update_drawdown()  # inv = 0 < trough 0.5 -> trough = 0
    assert e.drawdown_trough == 0


def test_update_drawdown_flat_resets():
    e = _bare()
    e.positions = [0.0]
    e.drawdown_peak = 100.0
    e.drawdown_trough = 80.0
    e.drawdowns = []
    e.current_price = 123.0
    e.update_drawdown()
    assert e.drawdown_peak == 0 and e.drawdown_trough == 0
    assert e.drawdowns[-1] == 0


# --- update_position_prices -------------------------------------------------


def test_update_position_prices_on_open_seeds_all_three():
    e = _bare()
    e.positions = [5.0]
    e.actions_made = [True]
    e.actions = [1]  # just opened a long
    e.current_price = 100.0
    e.update_position_prices()
    assert (e.position_price_entry, e.position_price_highest, e.position_price_lowest) == (100.0, 100.0, 100.0)


def test_update_position_prices_tracks_new_high():
    e = _bare()
    e.positions = [5.0]
    e.actions_made = [False]
    e.actions = [0]
    e.position_price_entry = 100.0
    e.position_price_highest = 120.0
    e.position_price_lowest = 90.0
    e.current_price = 150.0
    e.update_position_prices()
    assert e.position_price_highest == 150.0


def test_update_position_prices_tracks_new_low():
    # Line 349: lowest tightens while holding (not just-opened).
    e = _bare()
    e.positions = [5.0]
    e.actions_made = [False]
    e.actions = [0]
    e.position_price_entry = 100.0
    e.position_price_highest = 120.0
    e.position_price_lowest = 90.0
    e.current_price = 80.0
    e.update_position_prices()
    assert e.position_price_lowest == 80.0


def test_update_position_prices_lowest_seeds_from_zero():
    # When lowest is still 0 (e.g. seeded short), the first price sets it (line 348 guard).
    e = _bare()
    e.positions = [5.0]
    e.actions_made = [False]
    e.actions = [0]
    e.position_price_entry = 100.0
    e.position_price_highest = 120.0
    e.position_price_lowest = 0.0
    e.current_price = 200.0
    e.update_position_prices()
    assert e.position_price_lowest == 200.0
    assert e.position_price_highest == 200.0


def test_update_position_prices_flat_resets():
    e = _bare()
    e.positions = [0.0]
    e.actions_made = [False]
    e.actions = [0]
    e.position_price_entry = 100.0
    e.position_price_highest = 120.0
    e.position_price_lowest = 90.0
    e.current_price = 50.0
    e.update_position_prices()
    assert (e.position_price_entry, e.position_price_highest, e.position_price_lowest) == (0, 0, 0)


# --- update_position_and_balance (close bookkeeping) ------------------------


def _close_env(net_worth, *, action, forced=0, tpsl=0):
    """Bare env wired for a single update_position_and_balance call (lookback 1)."""
    e = _bare()
    e.data_provider = types.SimpleNamespace(get_lookback_window=lambda: 1)
    e.current_step = 0
    # balances/positions[-1] reflect the post-take_action state; _calculate_net_worth reads them.
    e.balances = [net_worth]
    e.positions = [0.0]
    e.get_price = lambda s: 0.0  # so net = balances[0]
    e.initial_net_worth = 100.0
    e.initial_balance = 100.0
    e.total_profit = 0.0
    e.net_worths = [100.0]
    e.current_profits = []
    e.buys_count = []
    e.sells_count = []
    e.total_profits = []
    e.actions = [action]
    e.actions_made = [True]
    e.forced_actions = [forced]
    e.tpsls = [tpsl]
    e.buys = []
    e.sells = []
    e.trades_won = []
    e.trades_lost = []
    e.trades_tp = []
    e.trades_sl = []
    return e


def test_update_position_and_balance_close_win_records_sell_and_profit():
    e = _close_env(120.0, action=2)
    e.update_position_and_balance()
    assert e.sells == [120.0]
    assert e.trades_won == [pytest.approx(20.0)]
    assert e.trades_lost == []
    assert e.total_profit == pytest.approx(20.0)
    # RESET: position flat, balance back to initial.
    assert e.positions[-1] == 0
    assert e.balances[-1] == pytest.approx(100.0)
    assert e.buys_count[-1] == 0 and e.sells_count[-1] == 1


def test_update_position_and_balance_close_loss_records_trades_lost():
    e = _close_env(85.0, action=2)
    e.update_position_and_balance()
    assert e.trades_lost == [pytest.approx(-15.0)]
    assert e.trades_won == []


def test_update_position_and_balance_forced_tp_tags_trades_tp():
    # forced cover (action recorded 0 but forced_action 4) with tpsl flag 1 -> a TP trade.
    e = _close_env(108.0, action=0, forced=4, tpsl=1)
    e.update_position_and_balance()
    assert e.sells == [pytest.approx(108.0)]
    assert e.trades_tp == [pytest.approx(8.0)]
    assert e.trades_sl == []


def test_update_position_and_balance_forced_sl_tags_trades_sl():
    e = _close_env(92.0, action=0, forced=2, tpsl=-1)
    e.update_position_and_balance()
    assert e.trades_sl == [pytest.approx(-8.0)]
    assert e.trades_tp == []


def test_update_position_and_balance_open_records_buy_not_sell():
    e = _close_env(100.0, action=1)
    e.positions = [5.0]  # still holding after open
    e.update_position_and_balance()
    assert e.buys == [pytest.approx(100.0)]
    assert e.sells == []
    assert e.buys_count[-1] == 1 and e.sells_count[-1] == 0
    # an open does NOT reset the position.
    assert e.positions[-1] == 5.0


# --- _realized_vol / _position_size -----------------------------------------


def test_realized_vol_matches_numpy_std():
    prices = [100, 101, 102, 101, 103, 104, 102, 105, 106, 104, 107]
    e = _bare()
    e.env_config = types.SimpleNamespace(vol_window=10)
    e.current_step = 10
    e.get_price = lambda s: float(prices[s])
    sub = prices[0:11]
    rets = [sub[i] / sub[i - 1] - 1 for i in range(1, len(sub))]
    assert e._realized_vol() == pytest.approx(float(np.std(rets)))


def test_realized_vol_none_with_too_few_returns():
    # current_step 0 -> only one price -> zero returns -> None (line 454).
    e = _bare()
    e.env_config = types.SimpleNamespace(vol_window=10)
    e.current_step = 0
    e.get_price = lambda s: 100.0
    assert e._realized_vol() is None


def test_realized_vol_window_floored_at_two():
    # vol_window below 2 is clamped to 2 -> looks back 2 bars.
    e = _bare()
    e.env_config = types.SimpleNamespace(vol_window=0)
    e.current_step = 5
    prices = {3: 100.0, 4: 110.0, 5: 121.0}
    e.get_price = lambda s: prices.get(s, 100.0)
    # window=2 -> lo = 3 -> prices[3,4,5] -> rets [0.1, 0.1] -> std 0
    assert e._realized_vol() == pytest.approx(0.0)


def test_position_size_fixed_is_all_in():
    e = _bare()
    e.env_config = types.SimpleNamespace(position_sizing="fixed")
    assert e._position_size() == 1.0


def test_position_size_default_attr_missing_is_all_in():
    # getattr default keeps non-vol-target configs all-in.
    e = _bare()
    e.env_config = types.SimpleNamespace()
    assert e._position_size() == 1.0


def test_position_size_vol_target_zero_vol_is_all_in():
    e = _bare()
    e.env_config = types.SimpleNamespace(position_sizing="vol_target")
    e._realized_vol = lambda: 0.0
    assert e._position_size() == 1.0


def test_position_size_vol_target_none_vol_is_all_in():
    e = _bare()
    e.env_config = types.SimpleNamespace(position_sizing="vol_target")
    e._realized_vol = lambda: None
    assert e._position_size() == 1.0


def test_position_size_vol_target_scales_and_clamps_low():
    # target/vol = 0.02/0.10 = 0.2 (within [min,1]).
    e = _bare()
    e.env_config = types.SimpleNamespace(position_sizing="vol_target", vol_target=0.02, vol_target_min=0.1)
    e._realized_vol = lambda: 0.10
    assert e._position_size() == pytest.approx(0.2)


def test_position_size_vol_target_floored_at_min():
    # turbulent regime: target/vol tiny -> clamped up to vol_target_min.
    e = _bare()
    e.env_config = types.SimpleNamespace(position_sizing="vol_target", vol_target=0.02, vol_target_min=0.1)
    e._realized_vol = lambda: 1.0
    assert e._position_size() == pytest.approx(0.1)


def test_position_size_vol_target_capped_at_one():
    # calm regime: target/vol > 1 -> clamped down to 1.
    e = _bare()
    e.env_config = types.SimpleNamespace(position_sizing="vol_target", vol_target=0.5, vol_target_min=0.1)
    e._realized_vol = lambda: 0.1
    assert e._position_size() == pytest.approx(1.0)


# --- _turnover_penalty ------------------------------------------------------


def test_turnover_penalty_off_for_non_fee_model():
    e = _bare()
    e.reward_model = "combo_all"
    assert e._turnover_penalty() == 0.0


def test_turnover_penalty_zero_when_no_action_made():
    e = _bare()
    e.reward_model = "combo_all_fee"
    e.actions_made = [False]
    assert e._turnover_penalty() == 0.0


def test_turnover_penalty_zero_when_actions_made_empty():
    e = _bare()
    e.reward_model = "combo_all_fee"
    e.actions_made = []
    assert e._turnover_penalty() == 0.0


def test_turnover_penalty_scales_fee_by_weight():
    e = _bare()
    e.reward_model = "combo_all_fee"
    e.actions_made = [True]
    e.transaction_fee_multiplier = 0.001
    e.reward_multipliers = {"combo_fee_penalty": 2.0}
    assert e._turnover_penalty() == pytest.approx(0.002)


def test_turnover_penalty_default_weight_one():
    e = _bare()
    e.reward_model = "combo_all_fee"
    e.actions_made = [True]
    e.transaction_fee_multiplier = 0.0015
    e.reward_multipliers = {}
    assert e._turnover_penalty() == pytest.approx(0.0015)


# --- update_reward (sum + history bookkeeping) ------------------------------


def test_update_reward_subtracts_penalties_and_tracks_totals():
    e = _bare()
    e._calculate_reward = lambda: 1.0
    e._turnover_penalty = lambda: 0.2
    e._noop_penalty = lambda: 0.1
    e.total_reward = 5.0
    e.rewards = []
    e.rewards_history = []
    e.reward_components = []
    r = e.update_reward()
    assert r == pytest.approx(0.7)
    assert e.total_reward == pytest.approx(5.7)
    assert e.rewards[-1] == pytest.approx(5.7)  # cumulative
    assert e.rewards_history[-1] == pytest.approx(0.7)  # per-step


# --- _calculate_reward: profit_percentage_direct / differential_sharpe ------


def _reward_env(reward_model, multipliers=None):
    e = _bare()
    e.reward_model = reward_model
    e.reward_multipliers = multipliers or {}
    return e


def test_reward_direct_step_return():
    e = _reward_env("profit_percentage_direct")
    e.net_worths = [100.0, 110.0]
    assert e._calculate_reward() == pytest.approx(0.1)


def test_reward_direct_zero_when_too_few_networths():
    e = _reward_env("profit_percentage_direct")
    e.net_worths = [100.0]
    assert e._calculate_reward() == 0.0


def test_reward_direct_zero_when_prev_networth_nonpositive():
    e = _reward_env("profit_percentage_direct")
    e.net_worths = [0.0, 110.0]
    assert e._calculate_reward() == 0.0


def test_differential_sharpe_finite_and_updates_state():
    e = _reward_env("differential_sharpe")
    e.net_worths = [100.0, 110.0]
    e._ds_a = 0.0
    e._ds_b = 0.0
    dsr = e._calculate_reward()
    assert math.isfinite(dsr)
    # first step: denom = (b - a^2)^1.5 = 0 -> dsr 0; running estimates advance.
    assert dsr == 0.0
    assert e._ds_a == pytest.approx(0.01 * 0.1)
    assert e._ds_b == pytest.approx(0.01 * 0.01)


def test_differential_sharpe_nonzero_after_state_built_up():
    e = _reward_env("differential_sharpe")
    e.net_worths = [100.0, 110.0]
    e._ds_a = 0.05
    e._ds_b = 0.02  # b - a^2 = 0.0175 > 0 -> finite nonzero dsr
    dsr = e._calculate_reward()
    assert math.isfinite(dsr)
    assert dsr != 0.0


# --- _calculate_reward: combo ----------------------------------------------

_COMBO = {
    "combo_sell_profit": 2.0,
    "combo_sell_profit_prev": 3.0,
    "combo_sell_perfect": 5.0,
    "combo_sell_drawdown": 7.0,
    "combo_buy_profit": 2.0,
    "combo_buy_perfect": 5.0,
    "combo_buy_profitable_offset": 11.0,
    "combo_buy_profitable": 0.5,
    "combo_buy_drawdown": 4.0,
    "combo_hold_profit": 2.0,
    "combo_hold_drawdown": 3.0,
    "combo_wrongaction": -1.0,
}


def test_combo_sell_reward_sums_four_components():
    e = _reward_env("combo", _COMBO)
    e.actions_made = [True]
    e.actions = [2]
    e.tpsls = [0]
    e.initial_net_worth = 100.0
    e.sells = [110.0]
    e.net_worths = [110.0, 100.0]  # [-2] = 110 -> prev profit 0
    e.drawdowns = [-0.05]
    e.current_step = 0
    e.data_provider = types.SimpleNamespace(get_signal_buy_sell=lambda s: -2)
    # 0.10*2 (sell_profit) + 0*3 (sell_profit_prev) + 1*5 (perfect) + (-0.05)*7 (drawdown)
    assert e._calculate_reward() == pytest.approx(0.2 + 5.0 - 0.35)


def test_combo_sell_reward_fires_on_tp_flag_without_sell_action():
    # tpsls[-1] == 1 routes through the SELL branch even though action != 2.
    e = _reward_env("combo", _COMBO)
    e.actions_made = [True]
    e.actions = [0]
    e.tpsls = [1]
    e.initial_net_worth = 100.0
    e.sells = [105.0]
    e.net_worths = [105.0, 105.0]
    e.drawdowns = [0.0]
    e.current_step = 0
    e.data_provider = types.SimpleNamespace(get_signal_buy_sell=lambda s: 0)
    # 0.05*2 + 0*3 + 0 + 0
    assert e._calculate_reward() == pytest.approx(0.1)


def test_combo_buy_reward_sums_four_components():
    e = _reward_env("combo", _COMBO)
    e.actions_made = [True]
    e.actions = [1]
    e.tpsls = [0]
    e.initial_net_worth = 100.0
    e.buys = [99.0]
    e.current_step = 0
    e.data_provider = types.SimpleNamespace(
        get_signal_buy_sell=lambda s: 2,
        get_signal_buy_profitable=lambda s: 3,
        get_signal_buy_drawdown=lambda s: -0.1,
    )
    # (99/100-1)*2 + 1*5 + (11-3)*0.5 + (-0.1)*4
    assert e._calculate_reward() == pytest.approx(-0.02 + 5.0 + 4.0 - 0.4)


def test_combo_hold_long_rewards_price_up():
    e = _reward_env("combo", _COMBO)
    e.actions_made = [False]
    e.actions = [0]
    e.positions = [5.0]
    e.current_price = 100.0
    e.current_step = 0
    e.get_price = lambda s: 110.0 if s == 1 else 100.0
    e.drawdowns = [-0.05]
    # dir +1 * 0.1 * 2 + (-0.05)*3
    assert e._calculate_reward() == pytest.approx(0.2 - 0.15)


def test_combo_hold_short_rewards_price_down():
    e = _reward_env("combo", _COMBO)
    e.actions_made = [False]
    e.actions = [0]
    e.positions = [-5.0]
    e.current_price = 100.0
    e.current_step = 0
    e.get_price = lambda s: 110.0 if s == 1 else 100.0
    e.drawdowns = [-0.05]
    # dir -1 * 0.1 * 2 + (-0.05)*3
    assert e._calculate_reward() == pytest.approx(-0.2 - 0.15)


# --- _calculate_reward: combo_all family ------------------------------------

_COMBO_ALL = {
    "combo_sell": 2.0,
    "combo_buy": 3.0,
    "combo_positionprofitpercentage": 4.0,
    "combo_noaction": 5.0,
    "combo_wrongaction": -1.0,
}


@pytest.mark.parametrize("model", ["combo_all", "combo_all2", "combo_all_fee", "combo_all_noop"])
def test_combo_all_close_uses_sells(model):
    e = _reward_env(model, _COMBO_ALL)
    e.actions_made = [True]
    e.actions = [2]
    e.tpsls = [0]
    e.initial_net_worth = 100.0
    e.sells = [110.0]
    assert e._calculate_reward() == pytest.approx(0.2)


def test_combo_all_close_fires_on_tpsl_flag():
    e = _reward_env("combo_all", _COMBO_ALL)
    e.actions_made = [True]
    e.actions = [0]
    e.tpsls = [-1]  # SL flag still counts as a close
    e.initial_net_worth = 100.0
    e.sells = [90.0]
    assert e._calculate_reward() == pytest.approx(-0.2)


def test_combo_all_open_long_rewards_price_up():
    e = _reward_env("combo_all", _COMBO_ALL)
    e.actions_made = [True]
    e.actions = [1]
    e.tpsls = [0]
    e.current_price = 100.0
    e.current_step = 0
    e.get_price = lambda s: 105.0 if s == 1 else 100.0
    assert e._calculate_reward() == pytest.approx(0.05 * 3)


def test_combo_all_open_short_rewards_price_down():
    e = _reward_env("combo_all", _COMBO_ALL)
    e.actions_made = [True]
    e.actions = [3]
    e.tpsls = [0]
    e.current_price = 100.0
    e.current_step = 0
    e.get_price = lambda s: 105.0 if s == 1 else 100.0
    # open_direction -1 for a short -> rewarded only when price falls
    assert e._calculate_reward() == pytest.approx(-0.05 * 3)


def test_combo_all_in_position_profit_delta():
    e = _reward_env("combo_all", _COMBO_ALL)
    e.actions_made = [False]
    e.actions = [0]
    e.positions = [5.0, 5.0]
    e.initial_net_worth = 100.0
    e.net_worths = [100.0, 110.0]
    # (0.1 - 0.0) * ppp
    assert e._calculate_reward() == pytest.approx(0.4)


def test_combo_all2_open_while_in_position_adds_wrongaction():
    # Line 567: combo_all2 penalizes opening (1/3) while already in position.
    e = _reward_env("combo_all2", _COMBO_ALL)
    e.actions_made = [False]
    e.actions = [1]
    e.positions = [5.0, 5.0]
    e.initial_net_worth = 100.0
    e.net_worths = [100.0, 110.0]
    assert e._calculate_reward() == pytest.approx(0.4 + _COMBO_ALL["combo_wrongaction"])


def test_combo_all_flat_noaction_rewards_price_move():
    # Lines 571-579: flat, step>1 -> price_diff * combo_noaction.
    e = _reward_env("combo_all", _COMBO_ALL)
    e.actions_made = [False]
    e.actions = [0]
    e.positions = [0, 0]
    e.current_step = 2
    e.current_price = 102.0
    e.get_price = lambda s: 100.0 if s == 1 else 0.0
    assert e._calculate_reward() == pytest.approx(0.02 * 5)


def test_combo_all2_close_while_flat_adds_wrongaction():
    # Line 576-577: combo_all2 closing (2/4) while not in a position is a wrong action.
    e = _reward_env("combo_all2", _COMBO_ALL)
    e.actions_made = [False]
    e.actions = [2]
    e.positions = [0, 0]
    e.current_step = 2
    e.current_price = 102.0
    e.get_price = lambda s: 100.0 if s == 1 else 0.0
    assert e._calculate_reward() == pytest.approx(_COMBO_ALL["combo_wrongaction"])


def test_combo_all_flat_early_step_returns_zero():
    # current_step <= 1 and flat -> 0 (line 580).
    e = _reward_env("combo_all", _COMBO_ALL)
    e.actions_made = [False]
    e.actions = [0]
    e.positions = [0, 0]
    e.current_step = 1
    assert e._calculate_reward() == 0


# --- _calculate_reward: buy_sell_signal family ------------------------------


def test_buy_sell_signal_perfect_sell_scaled_110pct():
    e = _reward_env("buy_sell_signal", {"combo_positionprofitpercentage": 4.0})
    e.actions_made = [True]
    e.actions = [2]
    e.tpsls = [0]
    e.initial_net_worth = 100.0
    e.sells = [110.0]
    e.current_step = 0
    e.data_provider = types.SimpleNamespace(get_signal_buy_sell=lambda s: -2)
    # profit 0.1 * 1.1 * 100
    assert e._calculate_reward() == pytest.approx(11.0)


def test_buy_sell_signal_nonperfect_sell_scaled_100():
    e = _reward_env("buy_sell_signal", {"combo_positionprofitpercentage": 4.0})
    e.actions_made = [True]
    e.actions = [2]
    e.tpsls = [0]
    e.initial_net_worth = 100.0
    e.sells = [110.0]
    e.current_step = 0
    e.data_provider = types.SimpleNamespace(get_signal_buy_sell=lambda s: 0)
    assert e._calculate_reward() == pytest.approx(10.0)


def test_buy_sell_signal_perfect_buy_returns_one():
    e = _reward_env("buy_sell_signal", {"combo_positionprofitpercentage": 4.0})
    e.actions_made = [True]
    e.actions = [1]
    e.tpsls = [0]
    e.current_step = 0
    e.data_provider = types.SimpleNamespace(get_signal_buy_sell=lambda s: 2)
    assert e._calculate_reward() == 1


def test_buy_sell_signal_base_hold_returns_zero():
    e = _reward_env("buy_sell_signal", {"combo_positionprofitpercentage": 4.0})
    e.actions_made = [False]
    e.actions = [0]
    e.current_step = 0
    e.data_provider = types.SimpleNamespace(get_signal_buy_sell=lambda s: 0)
    assert e._calculate_reward() == 0


def test_buy_sell_signal2_hold_in_position_rewards_return():
    e = _reward_env("buy_sell_signal2", {"combo_positionprofitpercentage": 4.0})
    e.actions_made = [False]
    e.actions = [0]
    e.positions = [5.0]
    e.current_price = 100.0
    e.current_step = 0
    e.get_price = lambda s: 110.0 if s == 1 else 100.0
    assert e._calculate_reward() == pytest.approx(0.1 * 4)


def test_buy_sell_signal2_hold_flat_negates_return():
    # Flat -> rewarded for price falling (negated step return).
    e = _reward_env("buy_sell_signal2", {"combo_positionprofitpercentage": 4.0})
    e.actions_made = [False]
    e.actions = [0]
    e.positions = [0]
    e.current_price = 100.0
    e.current_step = 0
    e.get_price = lambda s: 110.0 if s == 1 else 100.0
    assert e._calculate_reward() == pytest.approx(-0.1 * 4)


def test_buy_sell_signal3_uses_drawdown_only():
    e = _reward_env("buy_sell_signal3", {"combo_positionprofitpercentage": 4.0})
    e.actions_made = [False]
    e.actions = [0]
    e.positions = [5.0]
    e.current_price = 100.0
    e.current_step = 0
    e.get_price = lambda s: 110.0 if s == 1 else 100.0
    e.drawdowns = [-0.05]
    assert e._calculate_reward() == pytest.approx(-0.05 * 4)


def test_buy_sell_signal4_combines_return_and_drawdown():
    e = _reward_env("buy_sell_signal4", {"combo_positionprofitpercentage": 4.0})
    e.actions_made = [False]
    e.actions = [0]
    e.positions = [5.0]
    e.current_price = 100.0
    e.current_step = 0
    e.get_price = lambda s: 110.0 if s == 1 else 100.0
    e.drawdowns = [-0.05]
    assert e._calculate_reward() == pytest.approx((0.1 - 0.05) * 4)


# --- _calculate_reward: profit_all family -----------------------------------


def test_profit_all_sell_returns_total_profit_pct():
    e = _reward_env("profit_all", {})
    e.actions_made = [True]
    e.actions = [2]
    e.tpsls = [0]
    e.initial_net_worth = 100.0
    e.sells = [110.0]
    assert e._calculate_reward() == pytest.approx(0.1)


def test_profit_all2_sell_adds_step_and_total():
    e = _reward_env("profit_all2", {})
    e.actions_made = [True]
    e.actions = [2]
    e.tpsls = [0]
    e.initial_net_worth = 100.0
    e.sells = [110.0]
    e.net_worths = [110.0, 100.0]  # [-2] = 110
    # 110/110 - 1 + (110/100 - 1) = 0 + 0.1
    assert e._calculate_reward() == pytest.approx(0.1)


def test_profit_all_buy_returns_buy_pct():
    e = _reward_env("profit_all", {})
    e.actions_made = [True]
    e.actions = [1]
    e.tpsls = [0]
    e.initial_net_worth = 100.0
    e.buys = [95.0]
    assert e._calculate_reward() == pytest.approx(-0.05)


def test_profit_all_hold_in_position_returns_step_pct():
    e = _reward_env("profit_all", {})
    e.actions_made = [False]
    e.actions = [0]
    e.positions = [5.0, 5.0]
    e.net_worths = [110.0, 100.0]  # [-1]=100, [-2]=110
    assert e._calculate_reward() == pytest.approx(100.0 / 110.0 - 1.0)


def test_profit_all_flat_after_flat_btc_ratio():
    # Lines 625-629: flat now AND flat previously -> BTC-denominated value change.
    e = _reward_env("profit_all", {})
    e.actions_made = [False]
    e.actions = [0]
    e.positions = [0, 0]
    e.net_worths = [100.0]
    e.current_price = 110.0
    e.current_step = 2
    e.get_price = lambda s: 100.0 if s == 1 else 0.0
    # (100/110) / (100/100) - 1
    assert e._calculate_reward() == pytest.approx((100.0 / 110.0) / 1.0 - 1.0)


def test_profit_all_flat_after_position_returns_zero():
    # positions[-2] != 0 (a just-closed sell) -> ignored, returns 0 (line 630).
    e = _reward_env("profit_all", {})
    e.actions_made = [False]
    e.actions = [0]
    e.positions = [5.0, 0]
    e.net_worths = [100.0, 100.0]
    assert e._calculate_reward() == 0


# --- _calculate_reward: profit_percentage* + fallthrough --------------------


@pytest.mark.parametrize("model", ["profit_percentage2", "profit_percentage3", "profit_percentage4"])
def test_profit_percentage_sell_scaled_by_combo_sell(model):
    e = _reward_env(model, {"combo_sell": 2.0, "combo_positionprofitpercentage": 4.0})
    e.actions_made = [True]
    e.actions = [2]
    e.tpsls = [0]
    e.net_worths = [110.0]
    e.initial_net_worth = 100.0
    assert e._calculate_reward() == pytest.approx(0.1 * 2)


def test_profit_percentage3_flat_uses_inverse_next_move():
    # Lines 640-641: flat -> (1 - price_next/price) * ppp.
    e = _reward_env("profit_percentage3", {"combo_sell": 2.0, "combo_positionprofitpercentage": 4.0})
    e.actions_made = [False]
    e.actions = [0]
    e.net_worths = [100.0]
    e.initial_net_worth = 100.0
    e.positions = [0]
    e.current_price = 100.0
    e.current_step = 0
    e.get_price = lambda s: 110.0 if s == 1 else 100.0
    assert e._calculate_reward() == pytest.approx((1 - 110.0 / 100.0) * 4)


def test_profit_percentage3_in_position_returns_total_pct():
    # Lines 639-640: pp3 in a position falls through to net/init - 1 (unchanged profit_percentage).
    e = _reward_env("profit_percentage3", {"combo_sell": 2.0, "combo_positionprofitpercentage": 4.0})
    e.actions_made = [False]
    e.actions = [0]
    e.net_worths = [105.0]
    e.initial_net_worth = 100.0
    e.positions = [5.0]
    e.current_price = 100.0
    e.current_step = 0
    e.get_price = lambda s: 110.0 if s == 1 else 100.0
    assert e._calculate_reward() == pytest.approx(0.05)


def test_profit_percentage4_in_position_uses_next_step_return():
    # Lines 642-645: pp4 in a position overrides with the raw next-step return.
    e = _reward_env("profit_percentage4", {"combo_sell": 2.0, "combo_positionprofitpercentage": 4.0})
    e.actions_made = [False]
    e.actions = [0]
    e.net_worths = [100.0]
    e.initial_net_worth = 100.0
    e.positions = [5.0]
    e.current_price = 100.0
    e.current_step = 0
    e.get_price = lambda s: 110.0 if s == 1 else 100.0
    assert e._calculate_reward() == pytest.approx(0.1)


def test_profit_percentage2_made_buy_falls_through_to_total_pct():
    # pp2 made a buy (not a sell) -> no special branch, returns net/init - 1.
    e = _reward_env("profit_percentage2", {"combo_sell": 2.0, "combo_positionprofitpercentage": 4.0})
    e.actions_made = [True]
    e.actions = [1]
    e.tpsls = [0]
    e.net_worths = [105.0]
    e.initial_net_worth = 100.0
    assert e._calculate_reward() == pytest.approx(0.05)


def test_unknown_reward_model_falls_through_to_total_profit_pct():
    e = _reward_env("something_else", {})
    e.actions_made = [False]
    e.actions = [0]
    e.net_worths = [120.0]
    e.initial_net_worth = 100.0
    assert e._calculate_reward() == pytest.approx(0.2)


# --- get_next_observation ---------------------------------------------------


def _obs_env(lookback, observations_contain, take_profit=None, stop_loss=None):
    e = _bare()
    e.env_config = types.SimpleNamespace(
        observations_contain=observations_contain, take_profit=take_profit, stop_loss=stop_loss
    )
    e.data_provider = types.SimpleNamespace(get_lookback_window=lambda: lookback)
    e.initial_net_worth = 100.0
    return e


def test_get_next_observation_lookback1_all_extras_in_position():
    e = _obs_env(1, ["networth_percent_this_trade", "drawdown", "in_position"], take_profit=0.1, stop_loss=0.1)
    e.data_provider.get_values = lambda s: np.array([1.0, 2.0], dtype=np.float32)
    e.current_step = 0
    e.net_worths = [112.0]
    e.buys_count = [1]
    e.sells_count = [0]  # in a position
    e.drawdowns = [-0.03]
    e.positions = [5.0]
    out = e.get_next_observation()
    # values, networth(+0.12), drawdown(-0.03), tp(0.12/0.1=1.2), sl(loss<=0 ->0), in_position(1)
    assert list(out) == pytest.approx([1.0, 2.0, 0.12, -0.03, 1.2, 0.0, 1.0])


def test_get_next_observation_lookback1_flat_zeros_position_extras():
    e = _obs_env(1, ["networth_percent_this_trade", "drawdown", "in_position"], take_profit=0.1, stop_loss=0.1)
    e.data_provider.get_values = lambda s: np.array([1.0, 2.0], dtype=np.float32)
    e.current_step = 0
    e.net_worths = [80.0]
    e.buys_count = [0]
    e.sells_count = [1]  # flat -> networth/tp/sl all 0
    e.drawdowns = [-0.01]
    e.positions = [0]
    out = e.get_next_observation()
    assert list(out) == pytest.approx([1.0, 2.0, 0.0, -0.01, 0.0, 0.0, 0.0])


def test_get_next_observation_lookback1_stop_loss_when_underwater():
    # In a position AND underwater -> SL feature = loss / stop_loss (and tp 0 since no profit).
    e = _obs_env(1, ["networth_percent_this_trade"], take_profit=0.1, stop_loss=0.1)
    e.data_provider.get_values = lambda s: np.array([0.0], dtype=np.float32)
    e.current_step = 0
    e.net_worths = [95.0]
    e.buys_count = [1]
    e.sells_count = [0]
    e.positions = [5.0]
    out = e.get_next_observation()
    # networth(-0.05), tp(profit<=0 ->0), sl(loss 0.05 / 0.1 = 0.5)
    assert list(out) == pytest.approx([0.0, -0.05, 0.0, 0.5])


def test_get_next_observation_lookback_gt1_merges_grid_and_extras():
    # lookback 2 single-layer 2-D grid; per-row extras appended onto the feature axis then flattened.
    e = _obs_env(2, ["in_position"], take_profit=None, stop_loss=None)
    e.data_provider.get_values = lambda s: np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    e.current_step = 1
    e.net_worths = [100.0, 110.0]
    e.buys_count = [0, 1]
    e.sells_count = [0, 0]
    e.drawdowns = [0.0, -0.01, 0.0]
    e.positions = [5.0, -3.0, 0.0]  # positions[1:3] = [-3, 0] -> in_position [0, 0]
    out = e.get_next_observation()
    assert list(out) == pytest.approx([1.0, 2.0, 0.0, 3.0, 4.0, 0.0])


def test_get_next_observation_lookback_gt1_no_extras_flattens_grid():
    e = _obs_env(2, [], take_profit=None, stop_loss=None)
    e.data_provider.get_values = lambda s: np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    e.current_step = 1
    e.net_worths = [100.0, 110.0]
    e.buys_count = [0, 0]
    e.sells_count = [0, 0]
    e.positions = [0, 0, 0]
    out = e.get_next_observation()
    assert list(out) == pytest.approx([1.0, 2.0, 3.0, 4.0])


# --- step (action coercion, bookkeeping, done) ------------------------------


def _step_env(prices, lookback=1):
    """A bare env wired to drive a single step() without a model or real data."""
    e = _bare()
    e._prices = [float(p) for p in prices]
    e.data_provider = types.SimpleNamespace(
        get_lookback_window=lambda: lookback,
        get_timesteps=lambda: len(prices) - 1,
        get_values=lambda s: np.zeros(1, dtype=np.float32),
    )
    e.get_price = lambda s: e._prices[min(max(int(s), 0), len(e._prices) - 1)]
    e.get_timesteps = lambda: len(prices) - 1
    e.env_config = types.SimpleNamespace(observations_contain=[], take_profit=None, stop_loss=None)
    e.initial_net_worth = 100000.0
    e.initial_balance = 100000.0
    e.transaction_fee_multiplier = 0.0
    e.reward_model = "combo_all"
    e.reward_multipliers = {
        "combo_sell": 1.0,
        "combo_buy": 1.0,
        "combo_positionprofitpercentage": 1.0,
        "combo_noaction": 0.0,
    }
    # Seed the rolling state the way reset() does for lookback bars.
    e.current_step = 0
    e.current_price = 0
    e.total_profit = 0
    e.balances = [e.initial_balance]
    e.positions = [0]
    e.net_worths = [e.initial_net_worth]
    e.drawdowns = [0]
    e.buys_count = [0]
    e.sells_count = [0]
    e.total_profits = [0]
    e.current_profits = [0]
    e.rewards_history = [0]
    e.reward_components = [{"base": 0.0, "turnover_penalty": 0.0, "noop_penalty": 0.0}]
    e.actions = [0]
    e.actions_made = [False]
    e.forced_actions = [0]
    e.rewards = [0]
    e.tpsls = [0]
    e.tpsl_kinds = [None]
    e.position_price_highest = 0
    e.position_price_lowest = 0
    e.position_price_entry = 0
    e.total_reward = 0
    e.drawdown_peak = 0
    e.drawdown_trough = 0
    e._ds_a = 0.0
    e._ds_b = 0.0
    e.fees = []
    e.buys = []
    e.sells = []
    e.trades_won = []
    e.trades_lost = []
    e.trades_tp = []
    e.trades_sl = []
    # take_action is abstract on the base; provide a trivial long-only stub.
    def _take(action):
        if action == 1 and e.positions[-1] == 0:
            e.positions[-1] = e.balances[-1] / e.current_price
            e.balances[-1] = 0
            return True
        if action == 2 and e.positions[-1] > 0:
            e.balances[-1] = e.balances[-1] + e.positions[-1] * e.current_price
            e.positions[-1] = 0
            return True
        return False
    e.take_action = _take
    return e


def test_step_coerces_numpy_array_action():
    e = _step_env([100, 110, 110])
    obs, reward, done, finished_early, info = e.step(np.array([1]))
    # A buy executed -> position opened, current_step advanced.
    assert e.actions[-1] == 1
    assert e.actions_made[-1] is True
    assert e.positions[-1] != 0
    assert e.current_step == 1
    assert done is False


def test_step_advances_and_reports_done_at_last_step():
    e = _step_env([100, 110])  # timesteps = 1 -> one step finishes the episode
    obs, reward, done, finished_early, info = e.step(0)
    assert e.current_step == 1
    assert done is True
    assert finished_early is False


def test_step_finished_early_on_blown_account():
    # Force net worth to collapse: a held long with price crashing toward 0.
    e = _step_env([100, 0.0001, 0.0001])
    e.step(1)  # long @100
    obs, reward, done, finished_early, info = e.step(0)  # price ~0 -> net worth <= 0.1
    assert finished_early is True
    assert done is True


def test_step_returns_five_tuple_and_updates_reward_history():
    e = _step_env([100, 101, 102, 103])
    result = e.step(0)
    assert len(result) == 5
    assert len(e.rewards_history) == 2  # one seed + one step
    assert e.last_obs is result[0]


# --- get_run_state ----------------------------------------------------------


def _run_state_env():
    e = _bare()
    e.trades_won = [10.0, 30.0]
    e.trades_lost = [-5.0, -15.0]
    e.fees = [1.0, 2.0]
    e.buys = [100.0]
    e.sells = [120.0]
    e.initial_net_worth = 100.0
    e.total_reward = 3.0
    e.total_profit = 20.0
    e.trades_sl = [-15.0]
    e.reward_multipliers = {"a": 1.5, "b": 2.5}
    return e


def test_get_run_state_aggregates_pnl_fields():
    s = _run_state_env().get_run_state()
    assert s[0] == 3.0  # total_reward
    assert s[1] == 20.0  # total_profit
    assert s[2] == pytest.approx(0.2)  # total_profit / initial
    assert s[3] == pytest.approx(20.0)  # total_won + total_lost = 40 - 20
    assert s[5] == 2 and s[6] == 2  # n won / n lost
    assert s[7] == pytest.approx(50.0)  # win ratio %
    assert s[8] == pytest.approx(20.0)  # avg won
    assert s[9] == 30.0 and s[10] == 10.0  # max / min won
    assert s[11] == pytest.approx(-10.0)  # avg lost
    assert s[12] == -5.0 and s[13] == -15.0  # max / min lost
    assert s[14] == pytest.approx(5.0)  # avg trade
    assert s[15] == pytest.approx(-3.0)  # -fees
    assert s[16] == pytest.approx(223.0)  # volume = buys + sells + fees
    assert s[17] == 4  # total trades
    assert s[18] == 1  # n SL
    assert s[19] == pytest.approx(-15.0)  # sum SL
    assert s[20] == "1.500;2.500;"  # multiplier string


def test_get_run_state_compound_return():
    # compound = ((1 - 5/100)(1 + 10/100)(...)... - 1) * initial, lost applied first.
    s = _run_state_env().get_run_state()
    expected = ((1 - 5 / 100) * (1 - 15 / 100) * (1 + 10 / 100) * (1 + 30 / 100) - 1) * 100.0
    assert s[4] == pytest.approx(expected)


def test_get_run_state_empty_trades_guards_divisions():
    e = _bare()
    e.trades_won = []
    e.trades_lost = []
    e.fees = []
    e.buys = []
    e.sells = []
    e.initial_net_worth = 100.0
    e.total_reward = 0
    e.total_profit = 0
    e.trades_sl = []
    e.reward_multipliers = {}
    s = e.get_run_state()
    assert s[7] == 0  # win ratio guarded
    assert s[8] == 0  # avg won guarded
    assert s[14] == 0  # avg trade guarded
    assert s[17] == 0  # zero trades
    assert s[20] == ""  # no multipliers


# --- render (smoke; matplotlib is import-time only here) --------------------


def test_render_prints_without_error():
    import contextlib
    import io

    e = _bare()
    e.current_step = 3
    e.net_worths = [100.0, 105.0]
    e.total_reward = 1.23
    e.total_profit = 5.0
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        e.render()
    out = buf.getvalue()
    assert "net_worth" in out and "105.0" in out


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
    return len(fns)


if __name__ == "__main__":
    print(f"{_run_all()} env tag tests passed")
