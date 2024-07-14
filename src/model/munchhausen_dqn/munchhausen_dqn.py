import torch as th
import torch.nn.functional as F
from stable_baselines3 import DQN
import numpy as np

class MunchausenDQN(DQN):
    def __init__(self, env, policy, learning_rate=1e-4, buffer_size=100000, learning_starts=1000,
                 batch_size=32, tau=1.0, gamma=0.99, train_freq=1, gradient_steps=1, target_update_interval=1000,
                 exploration_fraction=0.1, exploration_initial_eps=1.0, exploration_final_eps=0.02, max_grad_norm=10,
                 munchausen_scale=0.9, munchausen_tau=0.03, verbose=0, tensorboard_log="", device="auto", **kwargs):
        super(MunchausenDQN, self).__init__(
            env=env,
            policy=policy,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            target_update_interval=target_update_interval,
            exploration_fraction=exploration_fraction,
            exploration_initial_eps=exploration_initial_eps,
            exploration_final_eps=exploration_final_eps,
            max_grad_norm=max_grad_norm,
            verbose=verbose,
            tensorboard_log=tensorboard_log,
            device=device,
            **kwargs
        )
        self.munchausen_scale = munchausen_scale
        self.munchausen_tau = munchausen_tau

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update learning rate according to schedule
        self._update_learning_rate(self.policy.optimizer)

        losses = []
        for _ in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]

            # Compute target Q-values
            with th.no_grad():
                next_q_values = self.q_net_target(replay_data.next_observations)
                next_actions = next_q_values.argmax(dim=1, keepdim=True)
                next_q_values, _ = next_q_values.max(dim=1)
                next_q_values = next_q_values.reshape(-1, 1)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

                # Compute Munchausen correction
                logits = th.log_softmax(self.q_net(replay_data.next_observations), dim=-1)
    
                log_pi_next = logits.gather(1, next_actions)
                log_pi_next = th.clamp(log_pi_next, min=-self.munchausen_tau)
                target_q_values += self.munchausen_scale * log_pi_next
                

            # Compute current Q-values
            current_q_values = self.q_net(replay_data.observations)
            current_q_values = th.gather(current_q_values, dim=1, index=replay_data.actions.long().view(-1, 1))

            # Compute loss
            loss = F.mse_loss(current_q_values, target_q_values)
            losses.append(loss.item())

            # Optimize the policy
            self.policy.optimizer.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

            # Update target network
            if self.num_timesteps % self.target_update_interval == 0:
                self.q_net_target.load_state_dict(self.q_net.state_dict())

        # Increase update counter
        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))
