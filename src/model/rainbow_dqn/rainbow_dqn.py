from typing import Any, Dict, Optional, Tuple, Type, TypeVar, Union

import gym
import numpy as np
import torch.nn.functional as F
import torch as th
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.dqn.dqn import DQN

from src.model.rainbow_dqn.policies import RainbowPolicy
from src.model.rainbow_dqn.prioritized_replay_buffer import PrioritizedReplayBuffer

class RainbowDQN(DQN):
    def __init__(
        self,
        env: Union[gym.Env, str],
        learning_rate: Union[float, Schedule] = 1e-4,
        buffer_size: int = 1000000,
        learning_starts: int = 50000,
        batch_size: int = 32,
        tau: float = 1.0,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = 4,
        gradient_steps: int = 1,
        replay_buffer_kwargs: Dict[str, Any]= {"alpha": 0.6, "beta": 0.4},
        optimize_memory_usage: bool = False,
        target_update_interval: int = 10000,
        exploration_fraction: float = 0.1,
        exploration_initial_eps: float = 1.0,
        exploration_final_eps: float = 0.05,
        max_grad_norm: float = 10,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):
        super(RainbowDQN, self).__init__(
            RainbowPolicy,
            env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            replay_buffer_class= PrioritizedReplayBuffer,
            replay_buffer_kwargs= replay_buffer_kwargs,
            optimize_memory_usage=optimize_memory_usage,
            target_update_interval=target_update_interval,
            exploration_fraction=exploration_fraction,
            exploration_initial_eps=exploration_initial_eps,
            exploration_final_eps=exploration_final_eps,
            max_grad_norm=max_grad_norm,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=_init_setup_model,
        )

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        """
        Update policy using the given batch size and gradient steps
        """
        self.policy.train()
        self._update_learning_rate(self.policy.optimizer)

        losses = []
        for _ in range(gradient_steps):
            self._n_updates += 1
            replay_data, weights, indices = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            with th.no_grad():
                next_q_values = self.q_net_target(replay_data.next_observations)
                next_q_values, _ = next_q_values.max(dim=1)
                next_q_values = next_q_values.reshape(-1, 1)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            current_q_values = self.q_net(replay_data.observations)
            current_q_values = th.gather(current_q_values, dim=1, index=replay_data.actions.long().view(-1, 1))

            loss = (weights * F.mse_loss(current_q_values, target_q_values, reduction='none')).mean()

            self.replay_buffer.update_priorities(indices, th.abs(current_q_values - target_q_values).detach().cpu().numpy().squeeze(1))

            self.policy.optimizer.zero_grad()
            loss.backward()
            # Clip gradient norm
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        # Increase update counter
        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))
