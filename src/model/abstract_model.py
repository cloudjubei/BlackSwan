from abc import ABC, abstractmethod
from src.conf.model_config import ModelConfig
from src.environment.abstract_env import AbstractEnv
from src.conf.model_config import ModelConfig
from src.environment.abstract_env import AbstractEnv
import torch
import torch.nn as nn
import torch.optim as optim
import random
import time

class AbstractModel(ABC):
    def __init__(self, config: ModelConfig):
        super().__init__()

        self.config = config
        self.id = self.get_id(config)

    @abstractmethod
    def get_id(self, config: ModelConfig):
        pass
    
    @abstractmethod
    def train(self, env: AbstractEnv):
        pass

    @abstractmethod
    def test(self, env: AbstractEnv, deterministic: bool = True):
        pass

    def is_pretrained(self):
        return False

    def show_train_render(self) -> bool:
        return False
    
    def get_reward_model(self) -> str:
        return "percent_profit"
    
    def get_reward_multipliers(self):
        return {}
    
    def has_deterministic_test(self) -> bool:
        return True

class BaseRLModel(AbstractModel):
    def __init__(self, config: ModelConfig):
        super(BaseRLModel, self).__init__(config)

        self.rl_config = config.model_rl
        
    def get_id(self, config: ModelConfig):
        return f'{config.model_type}_{config.model_rl.model_name}_{config.model_rl.reward_model}_{config.model_rl.learning_rate}_{config.model_rl.learning_starts}_{config.model_rl.batch_size}_{config.model_rl.buffer_size}_{config.model_rl.gamma}_{config.model_rl.optimizer_class}_{config.model_rl.activation_fn}_{config.model_rl.net_arch}_{config.model_rl.episodes}_{time.time()}'
    
    def is_pretrained(self):
        return self.rl_config.checkpoint_to_load is not None
    
    def get_reward_model(self):
        return self.rl_config.reward_model
    
    def get_reward_multipliers(self):
        return {
            "combo_noaction": self.rl_config.reward_multiplier_combo_noaction,
            "combo_positionprofitpercentage": self.rl_config.reward_multiplier_combo_positionprofitpercentage,
            "combo_buy": self.rl_config.reward_multiplier_combo_buy,
            "combo_sell": self.rl_config.reward_multiplier_combo_sell,

            "buy_sell_sold": self.rl_config.reward_multiplier_buy_sell_sold,
            "buy_sell_bought": self.rl_config.reward_multiplier_buy_sell_bought,

            "combo_actions_sell_hit": self.rl_config.reward_multiplier_combo_actions_sell_hit,
            "combo_actions_sell_miss": self.rl_config.reward_multiplier_combo_actions_sell_miss,
            "combo_actions_buy_threshold": self.rl_config.reward_multiplier_combo_actions_buy_threshold,
            "combo_actions_buy_hit": self.rl_config.reward_multiplier_combo_actions_buy_hit,
            "combo_actions_buy_miss": self.rl_config.reward_multiplier_combo_actions_buy_miss,
            "combo_actions_hold_buy_threshold": self.rl_config.reward_multiplier_combo_actions_hold_buy_threshold,
            "combo_actions_hold_cash_hit": self.rl_config.reward_multiplier_combo_actions_hold_cash_hit,
            "combo_actions_hold_cash_miss": self.rl_config.reward_multiplier_combo_actions_hold_cash_miss,
            "combo_actions_hold_position_hit": self.rl_config.reward_multiplier_combo_actions_hold_position_hit,
            "combo_actions_hold_position_miss": self.rl_config.reward_multiplier_combo_actions_hold_position_miss
        }

class BaseDeepModel(AbstractModel):
    def __init__(self, config: ModelConfig):
        super(BaseDeepModel, self).__init__(config)

    @abstractmethod
    def get_model(self) -> nn.Module:
        pass
    @abstractmethod
    def get_model_args(self) -> nn.Module:
        pass
    @abstractmethod
    def get_target_model(self) -> nn.Module:
        pass
    @abstractmethod
    def get_loss_fn(self):
        pass
    @abstractmethod
    def get_optimizer(self) -> optim.Optimizer:
        pass
    @abstractmethod
    def get_optimizer_args(self):
        pass
    def get_episodes(self):
        return 1
    def get_checkpoints_path(self):
        return f'checkpoints/{self.id}'

    def train(self, env: AbstractEnv):
        epsilon = 1.0
        gamma = 0.99 #discount factor
        epsilon_decay = 0.995 #or 0.9
        epsilon_end = 0.01
        target_update_freq = 10
        episodes = self.get_episodes()

        model = self.get_model()
        model_args = self.get_model_args()
        target_model = self.get_target_model()
        loss_fn = self.get_loss_fn()
        optimizer = self.get_optimizer()
        optimizer_args = self.get_optimizer_args()
        checkpoints_path = self.get_checkpoints_path()

        best_reward = 0
        for episode in range(0, episodes):
            print(f"TRAINING EPISODE {episode+1}/{episodes}")

            model.train()

            obs, _ = env.reset()
            total_reward = 0

            steps = 0
            while True:
                # Choose an action using epsilon-greedy policy
                if random.random() < epsilon:
                    action = env.action_space.sample()  # Exploration
                else:
                    with torch.no_grad():
                        q_values = model(torch.tensor(obs, dtype=torch.float32))
                        action = torch.argmax(q_values).item()  # Exploitation

                next_obs, reward, done, finished_early, info = env.step(action)

                # Calculate the Q-value targets using the target network
                with torch.no_grad():
                    # print(f'obs.shape= {obs.shape}')
                    # tensor_sequence = tensor_sequence.unsqueeze(1)
                    # print(f'tensor_sequence unsqueezed')
                    # print(tensor_sequence)
                    target_q_values = target_model(torch.tensor(next_obs, dtype=torch.float32))
                    max_target_q_value = torch.max(target_q_values)
                    q_target = reward + gamma * max_target_q_value

                # Calculate the Q-value predictions and the loss
                q_values = model(torch.tensor(obs, dtype=torch.float32))
                q_value = q_values[action]
                loss = loss_fn(q_value, q_target)

                # Backpropagate and update the model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_reward += reward
                obs = next_obs

                steps += 1
                if done:
                    break

            if best_reward < total_reward:
                best_reward = total_reward

                torch.save(
                    obj={
                        'episode': episode,
                        'best_reward': best_reward,
                        'model_state_dict': model.state_dict(),
                        'target_model_state_dict': target_model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'model_args': model_args,
                        'optimizer_args': optimizer_args
                    },
                    f=checkpoints_path
                )  
                print(f"Saved {self.id} model to {checkpoints_path}")

            # Decay epsilon
            epsilon = max(epsilon_end, epsilon * epsilon_decay)

            # Update the target network periodically
            if (episode+1) % target_update_freq == 0:
                target_model.load_state_dict(model.state_dict())

    def test(self, env: AbstractEnv, deterministic: bool = True):
        model = self.get_model()
        model.eval()

        obs, _ = env.reset()

        while True:
            q_values = model(torch.tensor(obs, dtype=torch.float32))
            action = torch.argmax(q_values).item()

            obs, reward, done, finished_early, info = env.step(action)
            if done:
                break

class BaseLSTMModel(BaseDeepModel):
    def __init__(self, config: ModelConfig):
        super(BaseLSTMModel, self).__init__(config)

        self.lstm_config = config.model_lstm

    def get_id(self, config: ModelConfig):
        return f'{config.model_type}_{config.model_lstm.loss_fn}__{config.model_lstm.learning_rate}_{config.model_lstm.weight_decay}_{config.model_lstm.episodes}_{config.model_lstm.layers}_{config.model_lstm.lstm_dropout}_{config.model_lstm.extra_dropout}_{config.model_lstm.first_activation}_{config.model_lstm.last_activation}'
    
    def get_checkpoints_path(self):
        return f'{self.lstm_config.checkpoints_folder}{self.id}'

class BaseMLPModel(BaseDeepModel):
    def __init__(self, config: ModelConfig):
        super(BaseMLPModel, self).__init__(config)

        self.mlp_config = config.model_mlp

    def get_id(self, config: ModelConfig):
        return f'{config.model_type}_{config.model_mlp.loss_fn}__{config.model_mlp.learning_rate}_{config.model_mlp.weight_decay}_{config.model_mlp.episodes}_{config.model_mlp.hidden_dim1}_{config.model_mlp.hidden_dim2}_{config.model_mlp.first_activation}_{config.model_mlp.last_activation}'
    
    def get_checkpoints_path(self):
        return f'{self.mlp_config.checkpoints_folder}{self.id}'

class BaseStrategyModel(AbstractModel):
    
    def is_pretrained(self):
        return True
    
    def show_train_render(self):
        return True
    
    def train(self, env: AbstractEnv):
        obs, _ = env.reset()

        while True:
            action = self.get_action(env, obs)

            obs, reward, done, finished_early, info = env.step(action)
            if done:
                break

    def test(self, env: AbstractEnv, deterministic: bool = True):
        return self.train(env)
