
from src.data.abstract_dataprovider import AbstractDataProvider
from .abstract_env import AbstractEnv

from gymnasium import spaces
from typing import Any
import numpy as np
from src.conf.env_config import EnvConfig
import math
import matplotlib.pyplot as plt


class DipPredictEnv(AbstractEnv):
    """
    Dip prediction environment for reinforcement learning with crypto.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self, env_config: EnvConfig, data_provider: AbstractDataProvider, device: str):
        super(DipPredictEnv, self).__init__(env_config, data_provider, device)

        self.max_steps = self.data_provider.get_timesteps()

        self.action_space = self.create_action_space()

        obs, _ = self.reset()
        self.observation_space = self.create_observation_space(obs)


    # Action space: 0 = Nothing, 1 = Dip (BEWARE: by default the RL models return actions as [0..<n])
    def create_action_space(self) -> spaces.Discrete:
        return spaces.Discrete(2)

    def create_observation_space(self, initial_observation) -> spaces.Box:
        # print(f'initial_observation.shape: {initial_observation.shape}')
        return spaces.Box(
            # low=-np.inf, high=np.inf, shape=initial_observation.shape, dtype=np.float32
            low=-1, high=1, shape=initial_observation.shape, dtype=np.float32
            # low=-10, high=10, shape=initial_observation.shape, dtype=np.float32
        )

    def get_next_observation(self) -> np.ndarray:

        out = self.data_provider.get_values(self.current_step)
        
        lookback_window_size = self.data_provider.get_lookback_window()

        if lookback_window_size > 1:
            out = out.flatten()

        return out

    def reset(self, seed: int = None, options: dict[str, Any] = None):
        super().reset(seed=seed)

        if not hasattr(self, 'actions'):
            self.actions_epochs = []
        else:
            self.actions_epochs.append(self.actions)

        self.current_step = 0
        self.current_price = 0
        self.current_streak = 0

        self.rewards_history = []
        self.actions = []
        self.rewards_totals = []

        self.recalls = []
        self.precisions = []
        self.negative_recalls = []
        self.accuracies = []
        self.streaks = []

        self.correct_dips = 0
        self.total_dips_seen = 0
        self.incorrect_dips = 0
        self.total_nondips_seen = 0

        self.total_reward = 0

        self.last_obs = self.get_next_observation()

        return self.last_obs, {}
    
    def resolve_action(self, action):
        self.actions.append(action)

    def update_reward(self, action):
        reward = self._calculate_reward(action)
        self.total_reward += reward
        self.rewards_totals.append(self.total_reward)
        self.rewards_history.append(reward)
        return reward

    def step(self, action):
        # make sure action is one value
        if isinstance(action, np.ndarray) and len(action.shape) > 0:
            action = action.item()

        self.current_price = self.get_price(self.current_step)

        self.resolve_action(action)
        reward = self.update_reward(action)

        self.current_step += 1

        finished_early = False
        done = (self.current_step >= self.get_timesteps()) or finished_early

        self._update_streaks(action, reward, done)

        self.last_obs = self.get_next_observation()

        return self.last_obs, reward, done, finished_early, {}

    #TODO: we can try to predict - whether it's a dip, reversal, the next candle color, good opp to buy (not just dip), trend length, price prediction
    def _calculate_reward(self, action):

                # return (profit_percentage - prev_profit_percentage) * self.reward_multipliers["combo_positionprofitpercentage"]
    
        profitable_steps = self.data_provider.get_signal_buy_profitable(self.current_step)
        # print(f'step: {self.current_step} price: {self.current_price} signal_buy_profitable: {profitable_steps} action: {action}')
        # drawdown = self.data_provider.get_signal_buy_drawdown(self.current_step)

        if profitable_steps <= self.data_provider.buyreward_maxwait:
            self.total_dips_seen += 1
            if action == 1:
                self.correct_dips += 1
                return self.reward_multipliers["combo_buy"]
            return self.reward_multipliers["combo_wrongaction"]
        
        self.total_nondips_seen += 1
        if action == 1:
            self.incorrect_dips += 1
            return self.reward_multipliers["combo_sell"]
        return self.reward_multipliers["combo_noaction"]
    


        # signal = self.data_provider.get_signal_buy_sell(self.current_step)

        # if signal > 1:
        #     prev_signal = self.data_provider.get_signal_buy_sell(self.current_step-1)
        #     if prev_signal < 1:
        #         self.total_dips_seen += 1
        #         if action == 1:
        #             self.correct_dips += 1
        #             return 1
        #         return -0.1
            
        # if action == 1:
        #     return -1
        # return 0
    
    def _update_streaks(self, action, reward, done):
        self.recalls.append(self._get_recall())
        self.precisions.append(self._get_precision())
        self.negative_recalls.append(self._get_negative_recall())
        self.accuracies.append(self._get_accuracy())

        if action == 1:
            if reward > 0:
                self.current_streak += 1
                if done:
                    self.streaks.append(self.current_streak)
            else:
                self.streaks.append(self.current_streak)
                self.current_streak = 0
        elif done:
            self.streaks.append(self.current_streak)

    def _get_recall(self):
        return self.correct_dips / self.total_dips_seen if self.total_dips_seen > 0 else 1
    def _get_precision(self):
        guessed_dips = self.correct_dips + self.incorrect_dips
        return self.correct_dips / guessed_dips if guessed_dips > 0 else 1
    def _get_negative_recall(self):
        return 1 - (self.incorrect_dips / self.total_nondips_seen) if self.total_nondips_seen > 0 else 1
    def _get_accuracy(self):
        recall = self._get_recall()
        negative_recall = self._get_negative_recall()
        return (recall + negative_recall)/2
    
    def get_run_state(self):        

        accuracy = self.accuracies[-1] if len(self.accuracies) > 0 else 0
        precision = self.precisions[-1] if len(self.precisions) > 0 else 0
        recall = self.recalls[-1] if len(self.recalls) > 0 else 0
        negative_recall = self.negative_recalls[-1] if len(self.negative_recalls) > 0 else 0
        dips_ratio = self.correct_dips/self.incorrect_dips if self.incorrect_dips > 0 else self.correct_dips

        f1_score = 2*precision*recall/(precision + recall) if precision + recall > 0 else 0

        avg_streak = sum(self.streaks) / len(self.streaks) if len(self.streaks) > 0 else 0
        max_streak = max(self.streaks) if len(self.streaks) > 0 else 0

        rewards = ""
        for value in self.reward_multipliers.values():
            rewards += f'{"%.3f" % value};'
        return [
            self.total_reward,
            f1_score,
            dips_ratio,
            accuracy,
            precision,
            recall,
            negative_recall,
            avg_streak,
            max_streak,
            f'[{self.correct_dips}/{self.total_dips_seen}]-[{self.incorrect_dips}/{self.total_nondips_seen}]',
            # self.current_step-1,
            rewards
        ]

    def render(self):
        print(f'ENV [{self.current_step}] accuracy: {self.accuracies[-1]} [{self.correct_dips}/{self.total_dips_seen}]-[{self.incorrect_dips}/{self.total_nondips_seen}] Total Reward: {self.total_reward}')

    def render_profits(self):
        x_values = np.arange(self.current_step)
        series1 = self.accuracies
        # series2 = self.actions

        # print(f'self.current_step: {self.current_step}')
        # print(f'len of rewards_totals: {len(self.rewards_totals)}')
        # print(f'len of actions: {len(self.actions)}')

        # Plotting the data
        plt.plot(x_values, series1, marker='o', linestyle='-', color='r', label='accuracies')
        # plt.plot(x_values, series2, marker='x', linestyle='--', color='g', label='actions')
        # plt.plot(x_values, series3, marker='s', linestyle='-.', color='b', label='net_worths')

        # Adding titles and labels
        plt.title('Accuracies')
        plt.xlabel('X Values')
        plt.ylabel('Y Values')
        plt.legend()
        plt.show()

# 1h  - 1-0.002  694|1442    7/3   0.020/2.333      
# 1h  - 2-0.002  992|1144   29/17  0.056/1.706
# 1h  - 1-0.005  305|1831    2/1   0.013/2.000
# TEST MIN     31706|96454  86/169 0.005/0.509
# 1h  - 2-0.005  550|1586    5/0   0.018/5.00
# TEST MIN     50083|78077 288/130 0.011/2.215   
# 1h  - 3-0.005 
# TEST MIN     61267|66893 405/246 0.013/1.646 
# 1h  - 4-0.01   421|1715   10/2   0.046/5.000
# TEST MIN     36213|91947  11/7   0.001/1.571

# 15m - 1-0.001  2727|5817 18/12-x/1.5 ;
# 15m - 2-0.002  2653|5891 9/4-x/2.25 ; 33/19-x/1.737
# 15m - 4-0.005  x|x
# TEST MIN     31706|96454  19/4   0.001/4.750 

# 10m - 1-0.001 3589|9227   13/17-0.007/0.765
# TEST MIN     55490|72670 160/156-0.006/1.026 
# 10m - 2-0.002 44364|83796  88/114-0.004/0.772
# + 2021                    122/55-0.005/2.218
# + 2021 reward_multiplier_combo_buy= [10]  23516/32431-0.469/0.725
# + 2020                  19740/26765-0.434/0.738  
# + 2019                    272/330-0.012/0.824 
# + 2018 bad
# + 2017                     46/14-0.002/3.286
# 10m - 3-0.003
# TEST MIN     36532|91628 67/100-0.004/0.670

# 5m  - 1-0.001 5492|20140 18/39-0.006/0.462       
# 5m  - 2-0.001 8815|16817 
# 5m  - 4-0.002 44364|83796
# TEST MIN                  6221/7176-0.215/0.867

# 1m  - 2*60-0.005 51806|76354 1959/1952-0.070/1.004    

# 0.001 => 10m ?
# 0.002 => 20m
# 0.003 => 30m
# 0.005 => 1h
# 0.01  => 4h ?



# 10m  - 2-0.002 TEST MIN + 2021 reward_multiplier_combo_buy= [2]
# 5m   - 4-0.002 TEST MIN reward_multiplier_combo_buy= [10,5,2]
# 
# 5m  - 4-0.002 TEST MIN + 2019      
# 5m  - 4-0.002 TEST MIN + 2020
# 5m  - 4-0.002 TEST MIN + 2021
# 5m  - 4-0.002 TEST MIN + 2017

# check lookback for one of these
# TODO: check 0.01 with 1h,15m,10m,5m
# TODO: check 0.005 with 1h,15m,10m,5m
# TODO: check 0.001 with 10m,5m
# TODO: check with 1m the above

# check if adding more data improves things
# TODO: lr x net_arch= [[8192,4096,512,256]],
# TODO: lr x custom_net_arch [[4096,4096,4096,4096,4096,4096]]
# indicators
# extra data
# trees
# making the data be fed 00:00->00:59 not 1h by 1h cos it's different to how humans see it

# multi asset data
# after, which commodity to invest in

# more data?




# learning_rate -> net_arch / model -> parameters
