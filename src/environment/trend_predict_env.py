
from src.data.abstract_dataprovider import AbstractDataProvider
from .abstract_env import AbstractEnv

from gymnasium import spaces
from typing import Any
import numpy as np
from src.conf.env_config import EnvConfig
import math
import matplotlib.pyplot as plt


class TrendPredictEnv(AbstractEnv):
    """
    Trend prediction environment for reinforcement learning with crypto.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self, env_config: EnvConfig, data_provider: AbstractDataProvider, device: str):
        super(TrendPredictEnv, self).__init__(env_config, data_provider, device)

        self.max_steps = self.data_provider.get_timesteps()

        self.action_space = self.create_action_space()

        obs, _ = self.reset()
        self.observation_space = self.create_observation_space(obs)


    # Action space: 0 = Positive, 1 = Negative (BEWARE: by default the RL models return actions as [0..<n])
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
        self.accuracies = []
        self.streaks = []

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

        # prev_signal = self.data_provider.get_signal_buy_sell(self.current_step-1)
        signal = self.data_provider.get_signal_buy_sell(self.current_step)

        if action == 1:
            return 1 if signal < 0 else 0
        return 1 if signal > 0 else 0
    
    def _update_streaks(self, action, reward, done):
        self.accuracies.append(self._get_accuracy())

        if reward > 0:
            self.current_streak += 1
            if done:
                self.streaks.append(self.current_streak)
        else:
            self.streaks.append(self.current_streak)
            self.current_streak = 0

    def _get_accuracy(self):
        return self.total_reward / len(self.rewards_history)
    
    def get_run_state(self):        

        accuracy = self.accuracies[-1] if len(self.accuracies) > 0 else 0
        avg_streak = sum(self.streaks) / len(self.streaks) if len(self.streaks) > 0 else 0
        max_streak = max(self.streaks) if len(self.streaks) > 0 else 0


        rewards = ""
        for value in self.reward_multipliers.values():
            rewards += f'{"%.3f" % value};'
        return [
            self.total_reward,
            accuracy,
            avg_streak,
            max_streak,
            # self.current_step-1,
            rewards
        ]

    def render(self):
        print(f'ENV [{self.current_step}] accuracy: {self.accuracies[-1]} Total Reward: {self.total_reward}')

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

# 15m 
# rl_reppo-custom_combo_all_0~0001_20000_512_100...    87591 0.507      1.028         17
# 15m@15
# rl_reppo-custom_combo_0~0001_20000_512_1000000...     4397 0.515      1.060         14
# rl_duel-dqn-custom_combo_0~0001_20000_512_1000...     1112 0.521      1.086         10
# 1h
# rl_reppo-custom_combo_all_0~0001_20000_512_100...     1109 0.519      1.080         11

# 1 1m