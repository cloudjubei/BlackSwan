
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


    # Action space: 0 = Hold, 1 = Buy, 2 = Sell (BEWARE: by default the RL models return actions as [0..<n])
    def create_action_space(self) -> spaces.Discrete:
        return spaces.Discrete(3)

    def create_observation_space(self, initial_observation) -> spaces.Box:
        # print(f'initial_observation.shape: {initial_observation.shape}')
        return spaces.Box(
            # low=-np.inf, high=np.inf, shape=initial_observation.shape, dtype=np.float32
            low=-1, high=1, shape=initial_observation.shape, dtype=np.float32
        )
    
    def get_next_observation(self) -> np.ndarray:

        values = self.data_provider.get_values(self.current_step)
        
        lookback_window_size = self.data_provider.get_lookback_window()

        if lookback_window_size > 1:
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
                extra_values.append(vs)
            if 'drawdown' in self.env_config.observations_contain:
                vs = self.drawdowns[self.current_step : self.current_step + lookback_window_size]
                extra_values.append(vs)
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
                extra_values.append(vs)
            if 'in_position' in self.env_config.observations_contain:
                vs = self.positions[self.current_step : self.current_step + lookback_window_size]
                vs = [1 if v > 0 else 0 for v in vs]
                extra_values.append(vs)
# self.actions_made_count

            if self.data_provider.is_multilayered():
                if self.data_provider.get_lookback_window() > 1: 
                    extra_values = np.column_stack((extra_values))
                    extra_values = np.tile(extra_values, (values.shape[0], 1, 1))
                else:
                    extra_values = np.tile(extra_values, (values.shape[0], 1))
            else:
                if self.data_provider.get_lookback_window() > 1: 
                    extra_values = np.column_stack((extra_values))

            out = np.concatenate((values, extra_values), axis=-1)
            if self.data_provider.config.flat_lookback:
                if not self.data_provider.is_multilayered() or self.data_provider.config.flat_layers:
                    out = out.flatten()
                else:
                    new_out = []
                    for i in range(len(self.data_provider.config.layers)):
                        new_out.append(out[i].flatten())
                    out = np.array(new_out)
            elif self.data_provider.config.flat_layers and self.data_provider.is_multilayered():
                new_out = out[0]
                for i in range(len(self.data_provider.config.layers)-1):
                    new_out = np.concatenate((new_out, out[i+1]), axis=0)
                out = new_out

            return out
        
        extra_values = []

        if 'networth_percent_this_trade' in self.env_config.observations_contain:
            if len(self.buys) > len(self.sells):
                net_worth = self.net_worths[self.current_step]
                extra_values.append(net_worth/self.initial_net_worth - 1)
            else:
                extra_values.append(0)
        if 'drawdown' in self.env_config.observations_contain:
            extra_values.append(self.drawdowns[-1])
        if self.env_config.stop_loss is not None:
            stop_loss_threshold = self.env_config.stop_loss
            if len(self.buys) > len(self.sells):
                net_worth = self.net_worths[self.current_step]
                loss = 1 - (net_worth / self.initial_net_worth)
                if loss > 0:
                    extra_values.append(loss/stop_loss_threshold)
                else:
                    extra_values.append(0)
            else:
                extra_values.append(0)
        if 'in_position' in self.env_config.observations_contain:
            extra_values.append(1 if self.positions[-1] > 0 else 0)

        if self.data_provider.is_multilayered():
            extra_values = np.tile(extra_values, (values.shape[0], 1))
            if self.data_provider.config.flat_layers:
                return np.append(values, extra_values, axis=1).flatten()
            return np.append(values, extra_values, axis=1)
        return np.append(values, extra_values, axis=0)

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

        self.last_obs = self.get_next_observation()
        return self.last_obs, {}
    
    def resolve_tpsl(self):
        if self.positions[-1] > 0:
            if self.env_config.take_profit is not None:
                net_worth = self._calculate_net_worth(self.current_step)
                take_profit = self.env_config.take_profit
                profit_percentage = (net_worth / self.initial_net_worth) - 1.0
                if profit_percentage >= take_profit:
                    return 2, True # TP SELL
            if self.env_config.stop_loss is not None:
                net_worth = self._calculate_net_worth(self.current_step)
                stop_loss = self.env_config.stop_loss
                loss_percentage = 1.0 - (net_worth / self.initial_net_worth)
                if loss_percentage >= stop_loss:
                    # print("SL step: ", self.current_step, " loss: ", round(loss_percentage, 3), " net_worth: ", net_worth, " position: ", self.positions[-1], " price: ", self.get_price(self.current_step), " prev: ", self.net_worths[-1], " prev_price:", self.get_price(self.current_step-1))
                    return 2, False # SL SELL
        return 0, None
    
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
    def _calculate_net_worth2(self, step: int):
        return (self.balances[step] + self.positions[step] * self.get_price(step)) * (1 - self.transaction_fee_multiplier)
    def _calculate_cagr(self): #Compound Annual Growth Rate
        # CAGR % = {[(End of period price / Beginning of period price)^1/t] - 1} x 100
        # t = the amount of time in terms of years
        start_amount = self.net_worths[0]
        end_amount = self.net_worths[-1]
        time_out_of_year = 0.01
        return math.pow((end_amount / start_amount) , 1/time_out_of_year) # * 100 # TODO: need to find out time_out_of_year
    def _calculate_time_spent_in_market(self):
        return 0 # TODO:
    def _calculate_risk_adjusted_return(self):
        time = self._calculate_time_spent_in_market()
        return 0 if time <= 0 else self._calculate_cagr() / time
    def _calculate_drawdown(self):
        return 0 if self.drawdown_peak <= 0 else (self.drawdown_trough / self.drawdown_peak) - 1.0
    
    def _calculate_reward(self):
        if self.reward_model == "combo_all" or self.reward_model == "combo_all2":
            if self.actions_made[-1]: # just made an action
                if self.actions[-1] == 2 or self.tpsls[-1] == -1: # has made a sell action or SL triggered
                    sell_net_worth = self.sells[-1]
                    profit_percentage = sell_net_worth/self.initial_net_worth - 1
                    return profit_percentage * self.reward_multipliers["combo_sell"]
                
                price = self.current_price
                price_next = self.get_price(self.current_step+1)
                price_diff = price_next/price - 1
                return price_diff * self.reward_multipliers["combo_buy"] # price_diff wasn't here
            
            if len(self.positions) > 1 and self.positions[-1] > 0 and self.positions[-2] > 0: # in position is diff than out of position
                net_worth = self.net_worths[-1]
                prev_net_worth = self.net_worths[-2]
                profit_percentage = net_worth/self.initial_net_worth - 1
                prev_profit_percentage = prev_net_worth/self.initial_net_worth - 1

                if self.reward_model == "combo_all2":
                    return (profit_percentage - prev_profit_percentage + self.drawdowns[-1]) * self.reward_multipliers["combo_positionprofitpercentage"]

                return (profit_percentage - prev_profit_percentage) * self.reward_multipliers["combo_positionprofitpercentage"]

            if self.current_step > 1:
                price_prev = self.get_price(self.current_step-1)
                price = self.current_price
                price_diff = price/price_prev - 1
                return price_diff * self.reward_multipliers["combo_noaction"]
            return 0
        elif self.reward_model == "buy_sell_signal" or self.reward_model == "buy_sell_signal2" or self.reward_model == "buy_sell_signal3" or self.reward_model == "buy_sell_signal4":
            if self.actions_made[-1]: # just made an action
                signal = self.data_provider.get_buy_sell_signal(self.current_step)
                if self.actions[-1] == 2 or self.tpsls[-1] == -1: # has made a sell action or SL triggered
                    
                    sell_net_worth = self.sells[-1]
                    profit_percentage = sell_net_worth/self.initial_net_worth - 1
                    if signal < 0: # perfect sell
                        profit_percentage = profit_percentage * 1.1
                    return profit_percentage * 100
                else:
                    if signal > 0: # perfect buy
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
        elif self.reward_model == "combo_actions" or self.reward_model == "combo_actions2":

            if self.actions_made[-1]: # just made an action
                if self.actions[-1] == 2 or self.tpsls[-1] == -1: # has made a sell action or SL triggered
                    sell_net_worth = self.sells[-1]
                    profit_percentage = sell_net_worth/self.initial_net_worth - 1
                    return profit_percentage * 10
                
                buy_signal = self.data_provider.get_buy_signal(self.current_step)

                multiplier = buy_signal - 2

                if multiplier <= 0:
                    out = 0.01 * ((1-multiplier)/(2 + 1))
                    return out

                out = -0.0001 * multiplier 
                return out
                
            # signal = self.data_provider.get_hold_signal(self.current_step)
            if self.current_step > 0:
                if self.reward_model == "combo_actions2":
                    return self.drawdowns[-1] * 0.1

                multiplier = -1 if self.positions[-1] == 0 else 1
                price_prev = self.get_price(self.current_step-1)
                price = self.get_price(self.current_step)
                price_diff = price/price_prev - 1
                if self.reward_model == "combo_actions3":
                    return (multiplier * price_diff + self.drawdowns[-1]) * 0.1
                return multiplier * price_diff * 0.1
            # print(f'REWARD NOTHING step: {self.current_step} price: {self.get_price(self.current_step)}')
            return 0
        elif self.reward_model == "drawdown":
            if self.actions_made[-1]: # just made an action
                if self.actions[-1] == 2 or self.tpsls[-1] == -1: # has made a sell action or SL triggered
                    sell_net_worth = self.sells[-1]
                    profit_percentage = sell_net_worth/self.initial_net_worth - 1
                    return profit_percentage * self.reward_multipliers["combo_sell"]
                
            net_worth = self.net_worths[-1]
            profit_percentage = net_worth/self.initial_net_worth - 1

            return (profit_percentage + self.drawdowns[-1]) * self.reward_multipliers["combo_positionprofitpercentage"]
        
        net_worth = self.net_worths[-1]
        if self.actions_made[-1]: # just made an action
            if self.actions[-1] == 2 or self.tpsls[-1] == -1: # has made a sell action or SL triggered
                net_worth = self.sells[-1]

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