from typing import Union
import torch
from torch import nn
from torch.nn import functional as F
from stable_baselines3 import DQN
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import TrainFreq, TrainFrequencyUnit
from gymnasium import spaces
import numpy as np

class PredictionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super(PredictionModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        return self.model(x)

class EpisodicMemory:
    def __init__(self):
        self.memory = set()

    def compute_novelty(self, obs):
        obs_tuple = tuple(obs.flatten())
        if obs_tuple in self.memory:
            return 0.0
        self.memory.add(obs_tuple)
        return 1.0

class CustomAgent57ReplayBuffer(ReplayBuffer):
    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[torch.device, str] = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
    ):
        super(CustomAgent57ReplayBuffer, self).__init__(
            buffer_size= buffer_size,
            observation_space= observation_space,
            action_space=action_space,
            device=device,
            n_envs=n_envs,
            optimize_memory_usage=optimize_memory_usage,
            handle_timeout_termination=handle_timeout_termination
        )
        self.intrinsic_rewards = np.zeros((self.buffer_size,))
        self.visit_counts = {}
        self.prediction_model = PredictionModel(observation_space.shape[0])
        self.memory_module = EpisodicMemory()
        self.prediction_optimizer = torch.optim.Adam(self.prediction_model.parameters(), lr=0.001)

    def add(self, obs, next_obs, action, reward, done, infos):
        super().add(obs, next_obs, action, reward, done, infos)
        intrinsic_reward = self.compute_intrinsic_reward(obs)
        self.intrinsic_rewards[self.pos - 1] = intrinsic_reward

    def compute_intrinsic_reward(self, obs):
        obs_tensor = torch.tensor(obs, dtype=torch.float32)
        obs_hash = hash(obs.tobytes())
        self.visit_counts[obs_hash] = self.visit_counts.get(obs_hash, 0) + 1

        # Count-based exploration bonus
        count_bonus = 1.0 / np.sqrt(self.visit_counts[obs_hash])

        # Prediction error bonus
        prediction_error_bonus = self.compute_prediction_error(obs_tensor)

        # Episodic memory novelty bonus
        episodic_memory_bonus = self.memory_module.compute_novelty(obs)

        # Weighted combination of bonuses
        return 0.5 * count_bonus + 0.3 * prediction_error_bonus + 0.2 * episodic_memory_bonus

    def compute_prediction_error(self, obs_tensor):
        self.prediction_optimizer.zero_grad()
        predicted_obs = self.prediction_model(obs_tensor)
        loss = torch.nn.functional.mse_loss(predicted_obs, obs_tensor)
        loss.backward()
        self.prediction_optimizer.step()
        return loss.item()

class Agent57(DQN):
    def __init__(
        self,
        policy,
        env,
        buffer_size: int = 1000000,
        learning_starts: int = 5000,
        train_freq: TrainFreq = TrainFreq(4, unit=TrainFrequencyUnit.STEP),
        **kwargs
    ):
        super(Agent57, self).__init__(
            policy,
            env,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            train_freq=train_freq,
            replay_buffer_class=CustomAgent57ReplayBuffer,
            **kwargs
        )

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update learning rate according to schedule
        self._update_learning_rate(self.policy.optimizer)

        losses = []
        for _ in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]

            with torch.no_grad():
                # Compute the next Q-values using the target network
                next_q_values = self.q_net_target(replay_data.next_observations)
                # Follow greedy policy: use the one with the highest value
                next_q_values, _ = next_q_values.max(dim=1)
                # Avoid potential broadcast issue
                next_q_values = next_q_values.reshape(-1, 1)

                intrinsic_rewards = self.compute_intrinsic_rewards(replay_data)
                total_rewards = replay_data.rewards + intrinsic_rewards
                target_q_values = total_rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            # Get current Q-values estimates
            current_q_values = self.q_net(replay_data.observations)

            # Retrieve the q-values for the actions from the replay buffer
            current_q_values = torch.gather(current_q_values, dim=1, index=replay_data.actions.long())

            # Compute Huber loss (less sensitive to outliers)
            loss = F.smooth_l1_loss(current_q_values, target_q_values)
            losses.append(loss.item())

            # Optimize the policy
            self.policy.optimizer.zero_grad()
            loss.backward()
            # Clip gradient norm
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        # Increase update counter
        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))

    def compute_intrinsic_rewards(self, replay_data):
        intrinsic_rewards = torch.tensor(
            self.replay_buffer.intrinsic_rewards[replay_data.indices], dtype=torch.float32
        )
        return intrinsic_rewards
