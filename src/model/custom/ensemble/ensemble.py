from typing import Dict, Optional, Tuple, Union
import numpy as np
import torch
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm

class EnsembleModel(OffPolicyAlgorithm):
    def __init__(self, ensemble):
        self.ensemble = ensemble

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        """Predict an action from the input state."""

        predictions = [model(obs) for model in self.ensemble]
        
        return torch.mean(torch.stack(predictions), dim=0)

    def train(self):
        for model in self.ensemble:
            model.train()
