
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


# reward for not doing anything should be 0 - combo_noaction

# should add precision, recall, dips guessed etc as obs?
# need to rethink whether anything other than dip predict makes sense
# multi asset defo also makes sense

# 1h  - max_wait=2  buy_percent=0.001 -> [2/384]-[1/1752] 0.010/2.00 ; [284/384]-[1092/1752] 0.323/0.260 ;
# 1h  - max_wait=2  buy_percent=0.002 -> [116/254]-[596/1882] 0.240/0.195 ; [11/254]-[17/1882] 0.078/0.647; [2/254]-[0/1882] 0.016/2.000 ; [5/254]-[4/1882] 0.038/1.250 ; [52/254]-[72/1882] 0.275/0.722 ;
# 1h  - max_wait=4  buy_percent=0.005 -> [1/26]-[75/2110] 0.013

# 10m - max_wait=2  buy_percent=0.001 -> [1764/2080]-[7625/10736] 0.308; [580/2080]-[2105/10736] 0.243 ; [705/2080]-[3008/10736] 0.243 ; [18/2080]-[16/10736] 0.017
# 10m - max_wait=2  buy_percent=0.001 ["trpo-custom"] -> [176/2080]-[217/10736] 0.142/0.811 ; [350/2080]-[748/10736] 0.220/0.468 ;
# 10m - max_wait=2  buy_percent=0.001 [8192,512] -> [4/2080]-[3/10736] 0.004/1.333 ;
# 10m - max_wait=2  buy_percent=0.001 custom_net_arch -> [1588/2080]-[7055/10736] 0.296/0.225 ;

# 10m - max_wait=2  buy_percent=0.002 -> [18/572]-[233/12244] 0.044

# 5m  - max_wait=3  buy_percent=0.001 -> [4788/5736]-[14248/19896] 0.387/0.336 ; [60/5736]-[79/19896] 0.020/0.759 ;
# 5m  - max_wait=2  buy_percent=0.001 -> [3584/4135]-[16201/21497] 0.300/0.221 ; [2051/4135]-[8372/21497] 0.282/0.245 ; [1258/4135]-[4244/21497] 0.261/0.296 ;

# 1h  - max_wait=2  buy_percent=0.001 trpo
# 1h  - max_wait=2  buy_percent=0.002 trpo
# 1h  - max_wait=4  buy_percent=0.005 trpo
# 1h  - max_wait=6  buy_percent=0.005 trpo
# 1h  - max_wait=2  buy_percent=0.001
# 1h  - max_wait=2  buy_percent=0.002
# 1h  - max_wait=4  buy_percent=0.005
# 1h  - max_wait=6  buy_percent=0.005
# 1h  - max_wait=2  buy_percent=0.002 duel-dqn

# 1h max_wait=4  buy_percent=0.005
# 1h max_wait=12  buy_percent=0.01
# 10m max_wait=3  buy_percent=0.005
# 1m@1m max_wait=5 TODO
# 1m@1m max_wait=3 TODO