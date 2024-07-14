import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.buffers import ReplayBuffer, ReplayBufferSamples
from typing import Any, Dict, List, NamedTuple, Optional, Union
from stable_baselines3.common.vec_env import VecNormalize

class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        alpha: float = 0.6,
        beta: float = 0.4,
        epsilon: float = 1e-6
    ):
        super(PrioritizedReplayBuffer, self).__init__(
            buffer_size, observation_space, action_space, device, n_envs, optimize_memory_usage
        )
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        self.priorities = np.zeros((buffer_size,), dtype=np.float32)

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        max_priority = self.priorities.max() if self.buffer_size > 0 else 1.0
        super().add(obs, next_obs, action, reward, done, infos)
        self.priorities[self.pos] = max_priority


    def sample(self, batch_size: int, env: Optional[VecNormalize] = None):
        # Ensure priorities are not zero and normalize
        priorities = self.priorities[:self.buffer_size] if self.full else self.priorities[:self.pos] + 1e-6
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()

        indices = np.random.choice(len(probabilities), batch_size, p=probabilities)
        # indices = np.random.choice(len(priorities), batch_size, p=probabilities)

        weights = (len(probabilities) * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        samples = super()._get_samples(indices, env)
        return samples, th.tensor(weights).to(self.device), indices
    

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        self.priorities[indices] = priorities + self.epsilon
