import numpy as np
import stable_baselines3
import torch
import tqdm
from src.model.abstract_model import AbstractModel
from src.conf.model_config import ModelConfig, ModelRLConfig
from src.environment.abstract_env import AbstractEnv
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import ProgressBarCallback
import os
import time

class RegressionModel(AbstractModel):
    def __init__(self, config: ModelConfig, model: AbstractModel, optimizer: torch.optim, loss_fn: torch.nn.Module):
        super(RegressionModel, self).__init__(config)

        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    def get_id(self, config: ModelConfig):
        custom_net_arch = ("-".join([s[:4] + s[-1] for s in config.model_regression.custom_net_arch])) if len(config.model_regression.custom_net_arch) > 0 and config.model_regression.custom_net_arch[0] != '' else ""
        return f'{config.model_type}_{config.model_regression.model_name}_{config.model_regression.learning_rate}_{config.model_regression.optimizer_class}_{config.model_regression.activation_fn}_{config.model_regression.loss_fn}_{config.model_regression.loss_fn_reduction}_{"|".join(str(s) for s in config.model_regression.net_arch)}_{custom_net_arch}_{config.model_regression.episodes}_{time.time()}'.replace('.', '~').replace('|', ']')
    
    def get_episodes(self):
        return self.config.model_regression.episodes
    
    def is_pretrained(self):
        return self.config.model_regression.checkpoint_to_load is not None

    def train(self, env: AbstractEnv):
        episodes = self.get_episodes()

        dataloader = env.get_dataloader()

        for episode in range(0, episodes):
            print(f"TRAINING EPISODE {episode+1}/{episodes}")

            self.optimizer.zero_grad()

            total_loss = 0
            for index, data in enumerate(tqdm.tqdm(dataloader)):
                X, y = data 

                # outputs = self.model(X).view(-1)  # OR predictions.squeeze()

                outputs = self.model(X).view(-1) 
                # print(f'outputs shape is: {outputs.shape}')
                # print(outputs)

                loss = self.loss_fn(outputs, y)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            env.total_reward = total_loss

            print(f"Ep [{episode}/{episodes}], Loss: {total_loss:.4f}")

            path = os.path.join(self.config.model_regression.checkpoints_folder, self.id)
            torch.save(self.model.state_dict(), path)
            print(f"Saved Regression model to {path}")


    def test(self, env: AbstractEnv, deterministic: bool = True):
        self.model.eval()

        dataloader = env.get_dataloader()

        for index, data in enumerate(tqdm.tqdm(dataloader)):
            X, y = data 

            # outputs = self.model(X)
            logits = self.model(X).view(-1)

            # probabilities = torch.softmax(probabilities, dim=1)
            # outputs = torch.argmax(probabilities, dim=1)
            probabilities = torch.sigmoid(logits)
            threshold = getattr(self.config.model_regression, "decision_threshold", 0.5)
            outputs = (probabilities > threshold).int()

            # print(probabilities)

            env.store_result(outputs, y)