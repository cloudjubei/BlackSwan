
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
            xs = data_provider.get_values(i)
            x_values.append(xs)
            signal_buy_profitable = data_provider.get_signal_buy_profitable(i)
            y = 0 if signal_buy_profitable >= data_provider.buyreward_maxwait else 1
            y_values.append(y)

        print("YS is: ")
        print(y_values)

        self.dataloader = DataLoader(TensorDataset(torch.tensor(x_values, device= self.device), torch.tensor(x_values, device= self.device)), batch_size= 32, shuffle= False)

    def reset(self, seed: int = None, options: dict[str, Any] = None):
        super().reset(seed=seed)

        self.total_seen = 0
        self.total_correct = 0
        self.total_incorrect = 0
        self.correct_pos = 0
        self.incorrect_pos = 0
        self.correct_neg = 0 
        self.incorrect_neg = 0

        return [], {}
    
    # Action space: 0 = Nothing, 1 = Dip
    def create_action_space(self) -> spaces.Discrete:
        return spaces.Discrete(2)

    def get_dataloader(self) -> DataLoader:
        return self.dataloader
    
    def store_result(self, prediction, actual):
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
            f'[{self.correct_pos}/{self.correct_neg}]-[{self.total_correct}/{self.total_incorrect}]'
        ]

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
        return

    def render_profits(self):
        return