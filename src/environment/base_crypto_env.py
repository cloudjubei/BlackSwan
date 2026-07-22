
from src.data.abstract_dataprovider import AbstractDataProvider
from .abstract_env import AbstractEnv

from gymnasium import spaces
from typing import Any
import numpy as np
from src.conf.env_config import EnvConfig
import math
import matplotlib.pyplot as plt


# Flat per-no-op cost for the combo_all_noop reward (≈ one default fee); scaled by combo_noop_penalty.
_NOOP_PENALTY = 0.001


def _concat_layer_grids(values):
    """Merge per-fidelity-layer feature grids onto the feature axis into one ``[lookback, per_bar]``
    grid. The multi-timeline provider yields a 3-D ``[n_layers, lookback, per_bar_layer]`` array which
    ``np.concatenate(_, axis=1)`` correctly merges layer-by-layer; the SINGLE provider yields ONE 2-D
    ``[lookback, per_bar]`` grid, which that same call would instead iterate into 1-D rows and raise an
    AxisError — so a lone 2-D grid (single layer, lookback > 1) is returned as-is."""
    if isinstance(values, np.ndarray) and values.ndim == 2:
        return values
    return np.concatenate(values, axis=1)


class BaseCryptoEnv(AbstractEnv):
    """
    Base trading environment for reinforcement learning with crypto.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self, env_config: EnvConfig, data_provider: AbstractDataProvider, device: str):
        super(BaseCryptoEnv, self).__init__(env_config, data_provider, device)

        self.max_steps = self.data_provider.get_timesteps()
        self.initial_balance = self.env_config.initial_balance
        self.transaction_fee_multiplier = self.env_config.transaction_fee

        self.initial_price = self.get_price(0)
        self.initial_net_worth = self.initial_balance
        self.hodl_amount = self.initial_balance / self.initial_price

        self.action_space = self.create_action_space()

        obs, _ = self.reset()
        self.observation_space = self.create_observation_space(obs)

    def create_observation_space(self, initial_observation) -> spaces.Box:
        # print(f'initial_observation.shape: {initial_observation.shape}')
        return spaces.Box(
            # low=-np.inf, high=np.inf, shape=initial_observation.shape, dtype=np.float32
            low=-1, high=1, shape=initial_observation.shape, dtype=np.float32
        )

    def get_next_observation(self) -> np.ndarray:

        values = self.data_provider.get_values(self.current_step)
        
        lookback_window_size = self.data_provider.get_lookback_window()

        extra_values = []
        if 'networth_percent_this_trade' in self.env_config.observations_contain:
            vs = []
            for i in range(lookback_window_size-1, -1, -1):
                index = self.current_step - i
                buys_count = self.buys_count[index]
                sells_count = self.sells_count[index]
                if buys_count > sells_count:
                    net_worth = self.net_worths[index]
                    vs.append(net_worth/self.initial_net_worth - 1)
                else:
                    vs.append(0)
            if lookback_window_size > 1:
                extra_values.append(vs)
            else:
                extra_values += vs
        if 'drawdown' in self.env_config.observations_contain:
            vs = self.drawdowns[self.current_step : self.current_step + lookback_window_size]
            if lookback_window_size > 1:
                extra_values.append(vs)
            else:
                extra_values += vs
        if self.env_config.take_profit is not None:
            take_profit_threshold = self.env_config.take_profit
            
            vs = []
            for i in range(lookback_window_size-1, -1, -1):
                index = self.current_step - i
                buys_count = self.buys_count[index]
                sells_count = self.sells_count[index]
                if buys_count > sells_count:
                    net_worth = self.net_worths[index]
                    profit_percentage = (net_worth / self.initial_net_worth) - 1
                    if profit_percentage > 0:
                        vs.append(profit_percentage/take_profit_threshold)
                    else:
                        vs.append(0)
                else:
                    vs.append(0)
            if lookback_window_size > 1:
                extra_values.append(vs)
            else:
                extra_values += vs

        if self.env_config.stop_loss is not None:
            stop_loss_threshold = self.env_config.stop_loss

            vs = []
            for i in range(lookback_window_size-1, -1, -1):
                index = self.current_step - i
                buys_count = self.buys_count[index]
                sells_count = self.sells_count[index]
                if buys_count > sells_count:
                    net_worth = self.net_worths[index]
                    loss = 1 - (net_worth / self.initial_net_worth)
                    if loss > 0:
                        vs.append(loss/stop_loss_threshold)
                    else:
                        vs.append(0)
                else:
                    vs.append(0)
            if lookback_window_size > 1:
                extra_values.append(vs)
            else:
                extra_values += vs
        if 'in_position' in self.env_config.observations_contain:
            vs = self.positions[self.current_step : self.current_step + lookback_window_size]
            vs = [1 if v > 0 else 0 for v in vs]
            if lookback_window_size > 1:
                extra_values.append(vs)
            else:
                extra_values += vs

        if lookback_window_size > 1:
            out = _concat_layer_grids(values) # so [[1h, 1h, 1h...], [1d, 1d, 1d...]] = [[1h, 1d], [1h,1d], ...]
            if extra_values:
                extra_values_T = [list(row) for row in zip(*extra_values)]
                out = np.concatenate((out, extra_values_T), axis=1)
            out = out.flatten()
        else:
            out = np.concatenate((values, extra_values), axis=-1)

        return out

    def reset(self, seed: int = None, options: dict[str, Any] = None):
        super().reset(seed=seed)

        if not hasattr(self, 'actions'):
            self.actions_epochs = []
        else:
            self.actions_epochs.append(self.actions)

        self.current_step = 0
        self.current_price = 0
        self.total_profit = 0

        self.balances = []
        self.positions = []
        self.net_worths = []
        self.drawdowns = []
        self.buys_count = []
        self.sells_count = []
        self.total_profits = []
        self.current_profits = []
        self.rewards_history = []
        self.reward_components = []
        self.actions = []
        self.actions_made = []
        self.forced_actions = []
        self.rewards = []
        self.tpsls = []
        self.tpsl_kinds = []

        self.position_price_highest = 0
        self.position_price_lowest = 0
        self.position_price_entry = 0

        for _ in range(self.data_provider.get_lookback_window()):
            self.balances.append(self.initial_balance)
            self.positions.append(0)
            self.net_worths.append(self.initial_net_worth)
            self.drawdowns.append(0)
            self.buys_count.append(0)
            self.sells_count.append(0)
            self.total_profits.append(self.total_profit)
            self.current_profits.append(0)
            self.rewards_history.append(0)
            self.reward_components.append({"base": 0.0, "turnover_penalty": 0.0, "noop_penalty": 0.0})
            self.actions.append(0)
            self.actions_made.append(False)
            self.forced_actions.append(0)
            self.rewards.append(0)
            self.tpsls.append(0)
            self.tpsl_kinds.append(None)

        self.total_reward = 0
        self.drawdown_peak = 0
        self.drawdown_trough = 0

        self.fees = []
        self.buys = []
        self.sells = []
        self.trades_won = []
        self.trades_lost = []
        self.trades_tp = []
        self.trades_sl = []

        #TODO: REMOVE
        # values = self.data_provider.get_values(0)
        # print("values[0]:")
        # print(values)
        # values = self.data_provider.get_values(self.data_provider.get_timesteps()-1)
        # values = self.data_provider.get_values(self.data_provider.get_timesteps())
        
        # raise ValueError("TEST")

        self.last_obs = self.get_next_observation()

        return self.last_obs, {}
    
    def resolve_tpsl(self):
        position = self.positions[-1]
        if position == 0:
            return 0, None, None
        # Close action by side: long closes with 2 (sell), short covers with 4. TP/SL thresholds are
        # measured on net worth, which already reflects the position's sign — so the same profit/loss
        # checks serve both directions; only the trailing stop (price-anchored) is mirrored by side.
        close_action = 2 if position > 0 else 4
        net_worth = self._calculate_net_worth(self.current_step)
        profit_percentage = (net_worth / self.initial_net_worth) - 1.0
        if self.env_config.take_profit is not None:
            take_profit = self.env_config.take_profit
            if self.env_config.trailing_take_profit is not None:
                entry = self.position_price_entry
                trailing = self.env_config.trailing_take_profit
                if position > 0:
                    activation = entry * (1.0 + take_profit)
                    if self.position_price_highest >= activation:
                        if self.current_price <= self.position_price_highest * (1.0 - trailing):
                            return close_action, True, "trailing"
                else:
                    activation = entry * (1.0 - take_profit)
                    if 0 < self.position_price_lowest <= activation:
                        if self.current_price >= self.position_price_lowest * (1.0 + trailing):
                            return close_action, True, "trailing"
            if profit_percentage >= take_profit:
                return close_action, True, "tp"
        if self.env_config.stop_loss is not None:
            loss_percentage = 1.0 - (net_worth / self.initial_net_worth)
            if loss_percentage >= self.env_config.stop_loss:
                return close_action, False, "sl"
        return 0, None, None
    

    # static UpdateTakeProfit(trade: TradingSetupTradeModel, setup: TradingSetupModel, minAmount: string) : TradingSetupActionModel
    # {
    #     const takeProfit = setup.config.takeProfit
    #     if (takeProfit){
    #         const trailingStop = takeProfit.trailingStop
    #         if (trailingStop){
    #             const activationAmount = MathUtils.MultiplyNumbers(trade.entryPriceAmount, "" + (1.0 + takeProfit.percentage))
    #             if (MathUtils.IsGreaterThanOrEqualTo(trade.highestPriceAmount, activationAmount)){
    #                 const triggerAmount = MathUtils.MultiplyNumbers(trade.highestPriceAmount, "" + (1.0 - trailingStop.deltaPercentage))
    #                 if (MathUtils.IsLessThanOrEqualTo(setup.currentPriceAmount, triggerAmount)){
    #                     return new TradingSetupActionModel(TradingSetupActionType.TAKEPROFIT, -1)
    #                 }
    #             }
    #         }
    # }


    
    def resolve_action(self, action):
        made_action = self.take_action(action) if action is not None else False

        forced_action = 0
        if made_action:
            self.tpsls.append(0)
            self.tpsl_kinds.append(None)
        else:
            forced_action, tp_action, tpsl_kind = self.resolve_tpsl()
            if forced_action != 0:
                made_action = self.take_action(forced_action)
                self.tpsls.append(1 if tp_action else -1)
                self.tpsl_kinds.append(tpsl_kind)
            else:
                self.tpsls.append(0)
                self.tpsl_kinds.append(None)

        self.actions.append(action if action is not None else 0)
        self.actions_made.append(made_action)
        self.forced_actions.append(forced_action)

    def update_position_and_balance(self):

        net_worth = self._calculate_net_worth(self.current_step)

        self.net_worths.append(net_worth)
        self.positions.append(self.positions[-1])
        self.balances.append(self.balances[-1])
        
        current_profit = net_worth - self.initial_net_worth

        action = self.actions[-1]
        made_action = self.actions_made[-1]
        forced_action = self.forced_actions[-1]
        
        if made_action:
            tpsl = self.tpsls[-1]

            opened = action in (1, 3) and forced_action == 0
            closed = (action in (2, 4) and forced_action == 0) or (forced_action in (2, 4))
            if opened:
                self.buys.append(net_worth)
            elif closed:
                self.sells.append(net_worth)

                if current_profit >= 0:
                    self.trades_won.append(current_profit)
                else:
                    self.trades_lost.append(current_profit)
                if tpsl == 1:
                    self.trades_tp.append(current_profit)
                elif tpsl == -1:
                    self.trades_sl.append(current_profit)

                self.total_profit += current_profit
                # RESET
                self.positions[-1] = 0
                self.balances[-1] = self.initial_balance

        self.current_profits.append(current_profit)
        self.buys_count.append(len(self.buys))
        self.sells_count.append(len(self.sells))
        self.total_profits.append(self.total_profit)

    def update_position_prices(self):
        if self.positions[-1] != 0:
            if self.actions_made[-1] and self.actions[-1] in (1, 3): # just opened a position
                self.position_price_entry = self.current_price
                self.position_price_highest = self.current_price
                self.position_price_lowest = self.current_price
            if self.current_price > self.position_price_highest:
                self.position_price_highest = self.current_price
            if self.position_price_lowest == 0 or self.current_price < self.position_price_lowest:
                self.position_price_lowest = self.current_price
        else:
            self.position_price_entry = 0
            self.position_price_highest = 0
            self.position_price_lowest = 0

    def update_drawdown(self):
        # Drawdown tracks adverse price moves WHILE in a position; for a short the adverse direction
        # is price rising, so it mirrors via the inverse price (keeping the same peak/trough math).
        if self.positions[-1] > 0:
            if self.current_price > self.drawdown_peak:
                self.drawdown_peak = self.current_price
                self.drawdown_trough = self.drawdown_peak
            elif self.current_price < self.drawdown_trough:
                self.drawdown_trough = self.current_price
        elif self.positions[-1] < 0:
            inv = 1.0 / self.current_price if self.current_price > 0 else 0
            if inv > self.drawdown_peak:
                self.drawdown_peak = inv
                self.drawdown_trough = self.drawdown_peak
            elif inv < self.drawdown_trough:
                self.drawdown_trough = inv
        else:
            self.drawdown_peak = 0
            self.drawdown_trough = 0

        self.drawdowns.append(self._calculate_drawdown())

    def update_reward(self):
        # Decompose into named, additive contributions (base + the two penalties, stored as the negative
        # they contribute) so explainability can answer "why this reward". Read-only: the summed reward is
        # identical to before.
        base = self._calculate_reward()
        turnover_penalty = self._turnover_penalty()
        noop_penalty = self._noop_penalty()
        drawdown_penalty = self._drawdown_penalty()
        reward = base - turnover_penalty - noop_penalty - drawdown_penalty
        self.total_reward += reward
        self.rewards.append(self.total_reward)
        self.rewards_history.append(reward)
        self.reward_components.append(
            {
                "base": base,
                "turnover_penalty": -turnover_penalty,
                "noop_penalty": -noop_penalty,
                "drawdown_penalty": -drawdown_penalty,
            }
        )
        return reward

    def _noop_penalty(self):
        # combo_all_noop: penalize a "no-op" trade — the agent emits a buy/sell/short/cover that does
        # NOT change the position (buy while already long, sell while flat, etc.), so `take_action`
        # returned False. The combo rewards otherwise ignore these wasted/invalid decisions, leaving
        # the agent free to spam dead actions. A flat per-no-op cost (fee-independent — it's a
        # decision-quality signal, not a real cost) discourages that. Tunable via `combo_noop_penalty`.
        # combo_unified folds the no-op penalty in too, gated purely by the combo_noop_penalty weight
        # (default OFF so the unified reward with weight 0 ≡ combo_all). combo_all_noop keeps its _NOOP_PENALTY
        # default so existing runs reproduce byte-for-byte.
        if self.reward_model == "combo_all_noop":
            default = _NOOP_PENALTY
        elif self.reward_model == "combo_unified":
            default = 0.0
        else:
            return 0.0
        if not self.actions or self.actions[-1] == 0:
            return 0.0
        # The agent's OWN action executed only if something was made AND it wasn't a forced TP/SL — if a
        # forced exit did the work, the agent's buy/sell was still a no-op.
        agent_executed = bool(self.actions_made[-1]) and self.forced_actions[-1] == 0
        if agent_executed:
            return 0.0
        # The multiplier IS the per-no-op penalty value (combo_noop_penalty lever); 0 = off.
        return self.reward_multipliers.get("combo_noop_penalty", default)

    def _turnover_penalty(self):
        # RB4: turnover/fee-aware reward VARIANT. The combo rewards already see the realized fee
        # (it shrinks net worth), but a single trade's fee is tiny per step, so nothing discourages
        # churn — the "trade often" objective can be gamed by over-trading. The `*_fee` variants add
        # an explicit per-trade penalty scaled by the fee rate, so the agent is pushed to trade WELL,
        # not just often. Tunable via the `combo_fee_penalty` multiplier; zero for every other reward.
        # combo_unified folds the fee penalty in too, gated by the combo_fee_penalty weight (default OFF so
        # the unified reward with weight 0 ≡ combo_all). combo_all_fee keeps its 1.0 default for byte-for-byte
        # reproduction of existing runs.
        if self.reward_model == "combo_all_fee":
            default = 1.0
        elif self.reward_model == "combo_unified":
            default = 0.0
        else:
            return 0.0
        if not self.actions_made or not self.actions_made[-1]:
            return 0.0
        weight = self.reward_multipliers.get("combo_fee_penalty", default)
        return self.transaction_fee_multiplier * weight

    def _drawdown_penalty(self):
        # combo_unified only: cost of sitting in an open position that is under water. `self.drawdowns[-1]`
        # is the current open-position peak-to-trough return (<= 0, resets to 0 when flat), updated each step
        # BEFORE the reward (update_drawdown runs before update_reward in step). A `weight * |drawdown|` cost
        # pushes the agent to CUT a losing hold rather than sit in it — a lever to test whether drawdown
        # aversion makes hold-forever (zero-trade) policies actually exit and trade. Default 0 = off, so a
        # combo_unified run with the weight unset is byte-for-byte identical to before.
        if self.reward_model != "combo_unified":
            return 0.0
        weight = self.reward_multipliers.get("combo_drawdown_penalty", 0.0)
        if not weight or not self.drawdowns:
            return 0.0
        return weight * abs(self.drawdowns[-1])

    def step(self, action):
        # make sure action is one value
        if isinstance(action, np.ndarray) and len(action.shape) > 0:
            action = action.item()
            # action = action[0]

        self.current_price = self.get_price(self.current_step)

        # Execute one time step within the environment
        self.resolve_action(action)

        self.update_position_and_balance()
        self.update_position_prices()
        self.update_drawdown()
        reward = self.update_reward()

        self.current_step += 1

        finished_early = (self.net_worths[-1] <= 0.1)
        done = (self.current_step >= self.get_timesteps()) or finished_early

        self.last_obs = self.get_next_observation()

        return self.last_obs, reward, done, finished_early, {}

    def _calculate_net_worth(self, step: int):
        offset_step = step + self.data_provider.get_lookback_window() - 1
        return (self.balances[offset_step] + self.positions[offset_step] * self.get_price(step))
    def _calculate_drawdown(self):
        return 0 if self.drawdown_peak <= 0 else (self.drawdown_trough / self.drawdown_peak) - 1.0

    def _realized_vol(self):
        """Lookahead-free volatility: stdev of the last ``vol_window`` price returns up to now."""
        window = max(2, int(getattr(self.env_config, "vol_window", 10)))
        lo = max(0, self.current_step - window)
        prices = [self.get_price(s) for s in range(lo, self.current_step + 1)]
        rets = [prices[i] / prices[i - 1] - 1.0 for i in range(1, len(prices)) if prices[i - 1] > 0]
        if len(rets) < 2:
            return None
        return float(np.std(rets))

    def _position_size(self):
        """Fraction of balance to deploy on entry. ``fixed`` = all-in (1.0); ``vol_target`` scales
        toward a constant volatility (vol_target / realized_vol), clamped to [vol_target_min, 1]."""
        if getattr(self.env_config, "position_sizing", "fixed") != "vol_target":
            return 1.0
        vol = self._realized_vol()
        if not vol or vol <= 0:
            return 1.0
        target = float(getattr(self.env_config, "vol_target", 0.02))
        min_size = float(getattr(self.env_config, "vol_target_min", 0.1))
        return float(min(1.0, max(min_size, target / vol)))
    
    
    def _calculate_reward(self):
        # Direct-RL reward on the portfolio's per-step return (research-favoured over the combo shaping
        # family): profit_percentage_direct = the raw step return.
        if self.reward_model == "profit_percentage_direct":
            if len(self.net_worths) < 2 or self.net_worths[-2] <= 0:
                return 0.0
            return self.net_worths[-1] / self.net_worths[-2] - 1.0

        # The combo family is ONE per-step weighted sum of the same components; the named variants differ
        # only in which weights/penalties are on. `combo_unified` makes that explicit: the SAME base, with
        # the combo_wrongaction term added (gated by its weight) and the fee/no-op penalties gated by their
        # weights in update_reward — so e.g. combo_unified(combo_wrongaction=0, combo_fee_penalty=0,
        # combo_noop_penalty=0) ≡ combo_all. (combo_all2 REPLACES the no-action term with combo_wrongaction
        # at a wrong-close, so it only matches combo_unified when combo_noaction == 0 — see the equivalence
        # tests.) Keeping the named variants here too means existing runs reproduce byte-for-byte.
        if self.reward_model in ("combo_all", "combo_all2", "combo_all_fee", "combo_all_noop", "combo_unified"):
            # combo_unified folds the DIRECT per-step portfolio return in as one more weighted component
            # (combo_direct): with every other weight 0 it IS profit_percentage_direct, and it composes with
            # the shaping terms. 0 (the default, and for the named variants) leaves the reward unchanged.
            direct = 0.0
            if self.reward_model == "combo_unified":
                cd = self.reward_multipliers.get("combo_direct", 0.0)
                if cd and len(self.net_worths) >= 2 and self.net_worths[-2] > 0:
                    direct = cd * (self.net_worths[-1] / self.net_worths[-2] - 1.0)

            if self.actions_made[-1]: # just made an action
                if self.actions[-1] in (2, 4) or self.tpsls[-1] != 0: # closed a position (sell/cover) or SL/TP
                    sell_net_worth = self.sells[-1]
                    profit_percentage = sell_net_worth/self.initial_net_worth - 1
                    return profit_percentage * self.reward_multipliers["combo_sell"] + direct

                # opened a position — a long is rewarded for price rising, a short for price falling
                price = self.current_price
                price_next = self.get_price(self.current_step+1)
                price_diff = price_next/price - 1
                open_direction = 1 if self.actions[-1] == 1 else -1
                return open_direction * price_diff * self.reward_multipliers["combo_buy"] + direct

            if len(self.positions) > 1 and self.positions[-1] != 0 and self.positions[-2] != 0: # in position is diff than out of position
                net_worth = self.net_worths[-1]
                prev_net_worth = self.net_worths[-2]
                profit_percentage = net_worth/self.initial_net_worth - 1
                prev_profit_percentage = prev_net_worth/self.initial_net_worth - 1
                reward = (profit_percentage - prev_profit_percentage) * self.reward_multipliers["combo_positionprofitpercentage"]

                # opening while ALREADY in position is a wasted/wrong action — combo_all2 and the unified
                # reward add the (signed) combo_wrongaction weight here; combo_all/fee/noop leave it (weight 0).
                if self.actions[-1] in (1, 3) and self.reward_model in ("combo_all2", "combo_unified"):
                    reward += self.reward_multipliers["combo_wrongaction"]

                return reward + direct

            if self.current_step > 1:
                price_prev = self.get_price(self.current_step-1)
                price = self.current_price
                price_diff = price/price_prev - 1

                # closing while NOT in position is a wrong action. combo_all2 REPLACES the no-action term with
                # combo_wrongaction; combo_unified ADDS it (so with combo_noaction=0 it matches combo_all2, and
                # with combo_wrongaction=0 it matches combo_all/fee/noop — a true weighted sum).
                if self.actions[-1] in (2, 4) and self.reward_model == "combo_all2": # closing but not in position
                    return self.reward_multipliers["combo_wrongaction"]
                if self.actions[-1] in (2, 4) and self.reward_model == "combo_unified":
                    return price_diff * self.reward_multipliers["combo_noaction"] + self.reward_multipliers["combo_wrongaction"] + direct

                return price_diff * self.reward_multipliers["combo_noaction"] + direct
            return direct

        elif self.reward_model == "buy_sell_signal" or self.reward_model == "buy_sell_signal2" or self.reward_model == "buy_sell_signal3" or self.reward_model == "buy_sell_signal4":
            if self.actions_made[-1]: # just made an action
                signal = self.data_provider.get_signal_buy_sell(self.current_step)
                if self.actions[-1] == 2 or self.tpsls[-1] == -1 or self.tpsls[-1] == 1: # has made a sell action or SL/TP triggered
                    
                    sell_net_worth = self.sells[-1]
                    profit_percentage = sell_net_worth/self.initial_net_worth - 1
                    if signal < -1: # perfect sell
                        profit_percentage = profit_percentage * 1.1
                    return profit_percentage * 100
                else:
                    if signal > 1: # perfect buy
                        return 1
                    
            if self.reward_model == "buy_sell_signal2" or self.reward_model == "buy_sell_signal3" or self.reward_model == "buy_sell_signal4":
                price = self.current_price
                price_next = self.get_price(self.current_step+1)
                profit_percentage = price_next/price - 1
                if self.positions[-1] == 0:
                    profit_percentage = -profit_percentage 
                if self.reward_model == "buy_sell_signal3":
                    return self.drawdowns[-1] * self.reward_multipliers["combo_positionprofitpercentage"]
                if self.reward_model == "buy_sell_signal4":
                    return (profit_percentage + self.drawdowns[-1]) * self.reward_multipliers["combo_positionprofitpercentage"]
                return profit_percentage * self.reward_multipliers["combo_positionprofitpercentage"]
            return 0
        
        # Any reward_model not matched above falls through to the cumulative portfolio return.
        net_worth = self.net_worths[-1]
        profit_percentage = net_worth/self.initial_net_worth - 1
        return profit_percentage
    
    def get_run_state(self):        
        total_won = sum(self.trades_won)
        total_lost = sum(self.trades_lost)
        total_trades = len(self.trades_won) + len(self.trades_lost)
        fees = sum(self.fees)

        buy_amounts = sum(self.buys)
        sell_amounts = sum(self.sells)
        volume = buy_amounts + sell_amounts + fees

        compound_won = 1.0
        for i in range(0, len(self.trades_lost)):
            compound_won *= (1 + self.trades_lost[i]/self.initial_net_worth)
        for i in range(0, len(self.trades_won)):
            compound_won *= (1 + self.trades_won[i]/self.initial_net_worth)
        compound_won = (compound_won - 1.0) * self.initial_net_worth

        rewards = ""
        for value in self.reward_multipliers.values():
            rewards += f'{"%.3f" % value};'
        return [
            # self.net_worths[0], # ignore start $ for now,
            self.total_reward,
            self.total_profit,
            self.total_profit/self.initial_net_worth,
            total_won + total_lost,
            compound_won,
            len(self.trades_won),
            len(self.trades_lost),
            100 * (len(self.trades_won)/total_trades) if total_trades > 0 else 0,
            total_won/len(self.trades_won) if len(self.trades_won) > 0 else 0,
            max(self.trades_won) if len(self.trades_won) > 0 else 0,
            min(self.trades_won) if len(self.trades_won) > 0 else 0,
            total_lost/len(self.trades_lost) if len(self.trades_lost) > 0 else 0,
            max(self.trades_lost) if len(self.trades_lost) > 0 else 0,
            min(self.trades_lost) if len(self.trades_lost) > 0 else 0,
            (total_won + total_lost)/total_trades if total_trades > 0 else 0,
            -fees,
            volume,
            total_trades,
            # self.current_step-1,
            len(self.trades_sl),
            sum(self.trades_sl),
            rewards
        ]

    def render(self):
        print(f'ENV [{self.current_step}] net_worth: {self.net_worths[-1]} Total Reward: {self.total_reward} Total profit: {self.total_profit}')

    def render_profits(self):
        x_values = np.arange(self.current_step+1)
        series1 = self.total_profits
        series2 = self.current_profits
        series3 = self.net_worths

        # Plotting the data
        plt.plot(x_values, series1, marker='o', linestyle='-', color='r', label='total_profits')
        plt.plot(x_values, series2, marker='x', linestyle='--', color='g', label='current_profits')
        # plt.plot(x_values, series3, marker='s', linestyle='-.', color='b', label='net_worths')

        # Adding titles and labels
        plt.title('Profits')
        plt.xlabel('X Values')
        plt.ylabel('Y Values')
        plt.legend()
        plt.show()