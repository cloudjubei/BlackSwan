
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
        # drawdown = self.data_provider.get_signal_buy_drawdown(self.current_step)

        if profitable_steps <= self.data_provider.config.buyreward_maxwait:
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

        accuracy = self.accuracies[-1]
        precision = self.precisions[-1]
        recall = self.recalls[-1]
        negative_recall = self.negative_recalls[-1]

        f1_score = 2*precision*recall/(precision + recall) if precision + recall > 0 else 0

        avg_streak = sum(self.streaks) / len(self.streaks) if len(self.streaks) > 0 else 0
        max_streak = max(self.streaks)

        rewards = ""
        for value in self.reward_multipliers.values():
            rewards += f'{"%.3f" % value};'
        return [
            self.total_reward,
            f1_score,
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


# reward for not doing anything should be 0 - combo_noaction

# should add precision, recall, dips guessed etc as obs?
# need to rethink whether anything other than dip predict makes sense
# multi asset defo also makes sense

# 1h  - max_wait=4  buy_percent=0.01  -> [0/3]-[0/2133]
# 15m - max_wait=4  buy_percent=0.01  -> [2/101]-[143/8443]
# 15m - max_wait=16 buy_percent=0.01  -> [8/101]-[359/8443]
# 10m - max_wait=3  buy_percent=0.005 -> [9/90]-[736/12726] 0.022 
# 10m - max_wait=2  buy_percent=0.001 -> [1764/2080]-[7625/10736] 0.308 

# 1h  max_wait=24 buy_percent=0.01 1/2
# 1h  max_wait=12 buy_percent=0.01 1/2
# 1h  max_wait=4 buy_percent=0.005 0/2
# 10m max_wait=2 buy_percent=0.001 0/2
# 10m max_wait=2 buy_percent=0.002 0/2
# 10m max_wait=2 buy_percent=0.001 x BEST custom_net_arch 0/5
# 5@5m  max_wait=3 12/16
# 5@5m  max_wait=2 12/16


# 1m@1m max_wait=5 TODO
# 1m@1m max_wait=3 TODO