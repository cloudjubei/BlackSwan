
from typing import Any
from src.data.abstract_dataprovider import AbstractDataProvider
from .abstract_env import AbstractEnv

from gymnasium import spaces
import numpy as np
from src.conf.env_config import EnvConfig
import torch
from torch.utils.data import DataLoader, TensorDataset

class   RegressionPredictEnv(AbstractEnv):
    """
    Regression prediction environment with crypto.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self, env_config: EnvConfig, data_provider: AbstractDataProvider, device: str):
        super(RegressionPredictEnv, self).__init__(env_config, data_provider, device)

        self.max_steps = self.data_provider.get_timesteps()

        self.action_space = self.create_action_space()

        x_values = []
        y_values = []
        for i in range(self.max_steps):
            xs = data_provider.get_values(i).flatten()
            x_values.append(xs)
            signal_buy_profitable = data_provider.get_signal_buy_profitable(i)
            y = 0 if signal_buy_profitable >= data_provider.buyreward_maxwait else 1
            y_values.append(y)


        self.n_positive = sum(1 for y in y_values if y == 1)
        self.n_negative = len(y_values) - self.n_positive

        # self.dataloader = DataLoader(TensorDataset(torch.from_numpy(np.stack(x_values)), torch.from_numpy(np.stack(y_values))), batch_size= env_config.batch_size, shuffle= False)
        self.dataloader = DataLoader(TensorDataset(torch.tensor(x_values, dtype=torch.float32, device= self.device), torch.tensor(y_values, dtype=torch.float32, device= self.device)), batch_size= env_config.batch_size, shuffle= False)
        # self.dataloader = DataLoader(TensorDataset(torch.tensor(np.ndarray(x_values), dtype=torch.float32, device= self.device), torch.tensor(np.ndarray(y_values), dtype=torch.float32, device= self.device)), batch_size= env_config.batch_size, shuffle= False)

        self.reset()

    def reset(self, seed: int = None, options: dict[str, Any] = None):
        super().reset(seed=seed)

        self.total_seen = 0
        self.total_correct = 0
        self.total_incorrect = 0
        self.correct_pos = 0
        self.incorrect_pos = 0
        self.correct_neg = 0 
        self.incorrect_neg = 0

        self.predictions = []
        self.actuals = []

        self.last_obs = self.get_next_observation()

        return self.last_obs, {}
    
    def create_action_space(self) -> spaces.Discrete:
        return spaces.Discrete(2)
    
    def get_next_observation(self) -> np.ndarray:

        out = self.data_provider.get_values(self.current_step)
        
        lookback_window_size = self.data_provider.get_lookback_window()

        if lookback_window_size > 1:
            out = out.flatten()

        return out
    
    def step(self, action):
        self.current_step += 1

        done = (self.current_step >= self.get_timesteps())

        return [], 0, done, False, {}

    def get_dataloader(self) -> DataLoader:
        return self.dataloader
    
    def store_result(self, prediction, actual):
        self.predictions += prediction
        self.actuals += actual

        actual_correct = (actual == 1)
        actual_incorrect = (actual == 0)
        prediction_correct = (prediction == 1)
        prediction_incorrect = (prediction == 0)

        self.total_seen += len(actual)
        self.total_correct += actual_correct.sum().item()
        self.total_incorrect += actual_incorrect.sum().item()

        self.correct_pos += (prediction_correct & actual_correct).sum().item()
        self.incorrect_pos += (prediction_correct & actual_incorrect).sum().item()
        self.correct_neg += (prediction_incorrect & actual_incorrect).sum().item()
        self.incorrect_neg += (prediction_incorrect & actual_correct).sum().item()
    
    def _get_recall(self):
        return self.correct_pos / self.total_correct if self.total_correct > 0 else 1
    def _get_precision(self):
        guessed = self.correct_pos + self.incorrect_pos
        return self.correct_pos / guessed if guessed > 0 else 1
    def _get_negative_recall(self):
        return 1 - (self.incorrect_pos / self.total_incorrect) if self.total_incorrect > 0 else 1
    def _get_accuracy(self):
        recall = self._get_recall()
        negative_recall = self._get_negative_recall()
        return (recall + negative_recall)/2
    
    def get_run_state(self):        
        accuracy = self._get_accuracy()
        precision = self._get_precision()
        recall = self._get_recall()
        negative_recall = self._get_negative_recall()
        simple_ratio = self.correct_pos/self.incorrect_pos if self.incorrect_pos > 0 else self.correct_pos

        f1_score = 2*precision*recall/(precision + recall) if precision + recall > 0 else 0
        
        return [
            f1_score,
            simple_ratio,
            accuracy,
            precision,
            recall,
            negative_recall,
            f'[{self.correct_pos}/{self.incorrect_pos}]-[{self.total_correct}/{self.total_incorrect}]'
        ]

    def render(self):
        return

    def render_profits(self):
        return
    

# 2021 -> 0.116  1.590    [2321/1460]-[36108/92052]
# 2020 -> 0.008 11.333       [136/12]-[36108/92052]
# 2019 -> 0.009  2.683       [161/60]-[36108/92052]
# 2018 -> 0.022  1.120      [412/368]-[36108/92052]
#      -> 0.010  1.255      [187/149]-[36108/92052]

# 15m + 0.01
# 15m + 0.005
# 10m + 0.01
# 10m + 0.005
# 10m + 0.003
# 10m + 0.002
# 5m + 0.002
# 5m + 0.005

# TODO:
# 5m + 0.003
# 5m + 0.01