import numpy as np
from src.model.abstract_model import BaseRLModel
from src.conf.model_config import ModelConfig, ModelRLConfig
from src.environment.abstract_env import AbstractEnv
from stable_baselines3.common.base_class import BaseAlgorithm
import os

# based on https://stable-baselines3.readthedocs.io/en/master/modules/base.html

class RLModel(BaseRLModel):
    def __init__(self, config: ModelConfig, rl_model: BaseAlgorithm):
        super(RLModel, self).__init__(config)

        self.rl_model = rl_model

    def train(self, env: AbstractEnv):
        timesteps = env.get_timesteps()
        
        for i in range(0, self.rl_config.episodes):
            print(f"TRAINING EPISODE {i+1}/{self.rl_config.episodes}")
            self.rl_model.learn(total_timesteps=timesteps, progress_bar=self.rl_config.progress_bar, log_interval=1000)

        path = os.path.join(self.rl_config.checkpoints_folder, self.id)
        self.rl_model.save(path)
        print(f"Saved RL model to {path}")

    def test(self, env: AbstractEnv, deterministic: bool = True):
        obs, _ = env.reset()

        if self.rl_config.model_name == "reppo":
            lstm_states = None
            num_envs = 1
            episode_starts = np.ones((num_envs,), dtype=bool)
            while True:
                action, lstm_states = self.rl_model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=deterministic)
                obs, rewards, done, finished_early, info = env.step(action)
                episode_starts = done
                if done:
                    break
        else:
            while True:
                (action, extra_info) = self.rl_model.predict(obs, deterministic=deterministic)
                obs, reward, done, finished_early, info = env.step(action)
                if done:
                    break

    def predict(self, env: AbstractEnv, deterministic: bool = True):
        (action, extra_info) = self.rl_model.predict(env.last_obs, deterministic=deterministic)
        obs, reward, done, finished_early, info = env.step(action)

        if isinstance(action, np.ndarray):
            action = action.item()
        return -1 if action == 2 else action
    
    def predictOnline(self, observation: np.ndarray, deterministic: bool = True):
        (action, extra_info) = self.rl_model.predict(observation, deterministic=deterministic)
        
        if isinstance(action, np.ndarray):
            action = action.item()
        return -1 if action == 2 else action