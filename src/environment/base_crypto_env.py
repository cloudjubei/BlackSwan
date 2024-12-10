
from src.data.abstract_dataprovider import AbstractDataProvider
from .abstract_env import AbstractEnv

from gymnasium import spaces
from typing import Any
import numpy as np
from src.conf.env_config import EnvConfig
import math
import matplotlib.pyplot as plt


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
            extra_values_T = [list(row) for row in zip(*extra_values)]
            out = np.concatenate(values, axis=1) # so [[1h, 1h, 1h...], [1d, 1d, 1d...]] = [[1h, 1d], [1h,1d], ...]
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
        self.actions = []
        self.actions_made = []
        self.forced_actions = []
        self.rewards = []
        self.tpsls = []

        self.position_price_highest = 0
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
            self.actions.append(0)
            self.actions_made.append(False)
            self.forced_actions.append(0)
            self.rewards.append(0)
            self.tpsls.append(0)

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
        if self.positions[-1] > 0:
            if self.env_config.take_profit is not None:
                net_worth = self._calculate_net_worth(self.current_step)
                take_profit = self.env_config.take_profit
                
                if self.env_config.trailing_take_profit is not None:
                    entry_price_amount = self.position_price_entry
                    activation_amount = entry_price_amount * (1.0 + take_profit)
                    current_price_amount = self.current_price
                    highest_price_amount = self.position_price_highest
                    if highest_price_amount >= activation_amount:
                        trailing_take_profit = self.env_config.trailing_take_profit
                        trigger_amount = highest_price_amount * (1.0 - trailing_take_profit)
                        if current_price_amount <= trigger_amount:
                            return 2, True # TP TRAILING SELL

                profit_percentage = (net_worth / self.initial_net_worth) - 1.0
                if profit_percentage >= take_profit:
                    return 2, True # TP SELL
            if self.env_config.stop_loss is not None:
                net_worth = self._calculate_net_worth(self.current_step)
                stop_loss = self.env_config.stop_loss
                loss_percentage = 1.0 - (net_worth / self.initial_net_worth)
                if loss_percentage >= stop_loss:
                    return 2, False # SL SELL
        return 0, None
    

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
        else:
            forced_action, tp_action = self.resolve_tpsl()
            if forced_action != 0:
                made_action = self.take_action(forced_action)
                self.tpsls.append(1 if tp_action else -1)
            else:
                self.tpsls.append(0)

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

            if (action == 1 and forced_action == 0) or (forced_action == 1):
                self.buys.append(net_worth)
            elif (action == 2 and forced_action == 0) or (forced_action == 2):
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
        if self.positions[-1] > 0:
            if self.actions_made[-1]: # just made an action
                if self.actions[-1] == 1: # had made a buy action
                    self.position_price_entry = self.current_price
            if self.current_price > self.position_price_highest:
                self.position_price_highest = self.current_price
        else:
            self.position_price_entry = 0
            self.position_price_highest = 0

    def update_drawdown(self):
        if self.positions[-1] > 0:
            if self.current_price > self.drawdown_peak:
                self.drawdown_peak = self.current_price
                self.drawdown_trough = self.drawdown_peak
            elif self.current_price < self.drawdown_trough:
                self.drawdown_trough = self.current_price
        else:
            self.drawdown_peak = 0
            self.drawdown_trough = 0

        self.drawdowns.append(self._calculate_drawdown())

    def update_reward(self):
        reward = self._calculate_reward()
        self.total_reward += reward
        self.rewards.append(self.total_reward)
        self.rewards_history.append(reward)
        return reward

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
    
    
    def _calculate_reward(self):
        if self.reward_model == "combo":

            if self.actions_made[-1]: # just made an action
                # SELL
                if self.actions[-1] == 2 or self.tpsls[-1] == -1 or self.tpsls[-1] == 1: # SELL action or SL/TP triggered
                    sell_net_worth = self.sells[-1]
                    profit_percentage = sell_net_worth/self.initial_net_worth - 1
                    sell_reward1 = profit_percentage * self.reward_multipliers["combo_sell_profit"]

                    prev_net_worth = self.net_worths[-2]
                    profit_percentage_prev = sell_net_worth/prev_net_worth - 1
                    sell_reward2 = profit_percentage_prev * self.reward_multipliers["combo_sell_profit_prev"]

                    signal = self.data_provider.get_signal_buy_sell(self.current_step)
                    perfect_sell = 1 if signal < -1 else 0
                    sell_reward3 = perfect_sell * self.reward_multipliers["combo_sell_perfect"]

                    drawdown = self.drawdowns[-1]
                    sell_reward4 = drawdown * self.reward_multipliers["combo_sell_drawdown"]

                    return sell_reward1 + sell_reward2 + sell_reward3 + sell_reward4

                # BUY
                if self.actions[-1] == 1:
                    sell_net_worth = self.buys[-1]
                    profit_percentage = sell_net_worth/self.initial_net_worth - 1
                    buy_reward1 = profit_percentage * self.reward_multipliers["combo_buy_profit"]

                    buy_perfect = self.data_provider.get_signal_buy_sell(self.current_step)
                    buy_perfect_signal = 1 if buy_perfect > 1 else 0
                    buy_reward2 = buy_perfect_signal * self.reward_multipliers["combo_buy_perfect"]

                    buy_profitable = self.data_provider.get_signal_buy_profitable(self.current_step) # 1 means next step will be profitable, 10 means will be profitable in 10 steps (default profit threshold 0.004)
                    buy_profitable_signal = self.reward_multipliers["combo_buy_profitable_offset"] - buy_profitable
                    buy_reward3 = buy_profitable_signal * self.reward_multipliers["combo_buy_profitable"]

                    buy_drawdown = self.data_provider.get_signal_buy_drawdown(self.current_step) # highest negative profit percentage till position is profitable
                    buy_reward4 = buy_drawdown * self.reward_multipliers["combo_buy_drawdown"]

                    return buy_reward1 + buy_reward2 + buy_reward3 + buy_reward4

            # HOLD
            price = self.current_price
            price_next = self.get_price(self.current_step+1)
            profit_percentage = price_next/price - 1
            price_direction = 1 if self.positions[-1] > 0 else -1
            hold_reward1 = (price_direction * profit_percentage) * self.reward_multipliers["combo_hold_profit"]
            
            drawdown = self.drawdowns[-1]
            hold_reward2 = drawdown * self.reward_multipliers["combo_hold_drawdown"]
            wrong_action_reward = 0

            if self.actions[-1] != 0 and self.reward_model == "combo2": # acting but shouldn't
                wrong_action_reward = self.reward_multipliers["combo_wrongaction"]
            
            return hold_reward1 + hold_reward2 + wrong_action_reward

        if self.reward_model == "combo_all" or self.reward_model == "combo_all2":
            if self.actions_made[-1]: # just made an action
                if self.actions[-1] == 2 or self.tpsls[-1] == -1 or self.tpsls[-1] == 1: # has made a sell action or SL/TP triggered
                    sell_net_worth = self.sells[-1]
                    profit_percentage = sell_net_worth/self.initial_net_worth - 1
                    return profit_percentage * self.reward_multipliers["combo_sell"]
                
                price = self.current_price
                price_next = self.get_price(self.current_step+1)
                price_diff = price_next/price - 1
                return price_diff * self.reward_multipliers["combo_buy"]
            
            if len(self.positions) > 1 and self.positions[-1] > 0 and self.positions[-2] > 0: # in position is diff than out of position
                net_worth = self.net_worths[-1]
                prev_net_worth = self.net_worths[-2]
                profit_percentage = net_worth/self.initial_net_worth - 1
                prev_profit_percentage = prev_net_worth/self.initial_net_worth - 1

                if self.actions[-1] == 1 and self.reward_model == "combo_all2": # buying but already in position
                    return (profit_percentage - prev_profit_percentage) * self.reward_multipliers["combo_positionprofitpercentage"] + self.reward_multipliers["combo_wrongaction"]

                return (profit_percentage - prev_profit_percentage) * self.reward_multipliers["combo_positionprofitpercentage"]

            if self.current_step > 1:
                price_prev = self.get_price(self.current_step-1)
                price = self.current_price
                price_diff = price/price_prev - 1

                if self.actions[-1] == 2 and self.reward_model == "combo_all2": # selling but not in position
                    return self.reward_multipliers["combo_wrongaction"]
                
                return price_diff * self.reward_multipliers["combo_noaction"]
            return 0

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
        
        elif self.reward_model == "profit_all" or self.reward_model == "profit_all2":
            if self.actions_made[-1]: # just made an action
                if self.actions[-1] == 2 or self.tpsls[-1] == -1 or self.tpsls[-1] == 1: # has made a sell action or SL/TP triggered
                    net_worth = self.sells[-1] # in $
                    if self.reward_model == "profit_all2":
                        net_worth_prev = self.net_worths[-2]
                        return net_worth/net_worth_prev - 1 + (net_worth/self.initial_net_worth - 1)
                    return net_worth/self.initial_net_worth - 1
                if self.actions[-1] == 1:
                    net_worth = self.buys[-1] # in $
                    return net_worth/self.initial_net_worth - 1
            if self.positions[-1] > 0: # in position
                net_worth = self.net_worths[-1] # in $
                prev_net_worth = self.net_worths[-2] # in $
                return net_worth/prev_net_worth - 1
            
            if self.positions[-2] == 0: # if currently NOT in position but previous was in position, then it was a sell - ignore, hence 0 at end
                net_worth = self.net_worths[-1]
                net_worth_in_position = net_worth / self.current_price # in BTC
                net_worth_in_position_prev = net_worth / self.get_price(self.current_step-1) # in BTC
                return net_worth_in_position/net_worth_in_position_prev - 1 
            return 0



        net_worth = self.net_worths[-1]
        profit_percentage = net_worth/self.initial_net_worth - 1

        if (self.reward_model == "profit_percentage4" or self.reward_model == "profit_percentage3" or self.reward_model == "profit_percentage2") and self.actions_made[-1] and (self.actions[-1] == 2 or self.tpsls[-1] == -1):
            return profit_percentage * self.reward_multipliers["combo_sell"]
        if (self.reward_model == "profit_percentage4" or self.reward_model == "profit_percentage3"):
            if self.positions[-1] == 0:
                profit_percentage = (1 - self.get_price(self.current_step+1)/self.current_price) * self.reward_multipliers["combo_positionprofitpercentage"]
            elif self.reward_model == "profit_percentage4":
                price = self.current_price
                price_next = self.get_price(self.current_step+1)
                profit_percentage = price_next/price - 1

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
            (len(self.buys) + len(self.sells))/2,
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