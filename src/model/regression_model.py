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

class RegressionModel(AbstractModel):
    def __init__(self, config: ModelConfig, model: AbstractModel):
        super(RegressionModel, self).__init__(config)

        self.model = model

    def get_episodes(self):
        return self.config.model_regression.episodes

    def train(self, env: AbstractEnv):
        # epsilon = 1.0
        # gamma = 0.99 #discount factor
        # epsilon_decay = 0.995 #or 0.9
        # epsilon_end = 0.01
        # target_update_freq = 10
        episodes = self.get_episodes()

        # model = self.get_model()
        # model_args = self.get_model_args()
        # target_model = self.get_target_model()
        # loss_fn = self.get_loss_fn()
        # optimizer = self.get_optimizer()
        # optimizer_args = self.get_optimizer_args()
        # checkpoints_path = self.get_checkpoints_path()

        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        dataloader = env.get_dataloader()

        for episode in range(0, episodes):
            print(f"TRAINING EPISODE {episode+1}/{episodes}")

            optimizer.zero_grad()

            total_loss = 0
            for X, y in tqdm(dataloader):

                outputs = self.model(X)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            print(f"Ep [{episode}/{episodes}], Loss: {total_loss:.4f}")

            # path = os.path.join(self.config.model_regression.checkpoints_folder, self.id)
            # self.model.save(path)
            # print(f"Saved Regression model to {path}")

            # torch.save(
            #     obj={
            #         'episode': episode,
            #         'best_reward': best_reward,
            #         'model_state_dict': model.state_dict(),
            #         'target_model_state_dict': target_model.state_dict(),
            #         'optimizer_state_dict': optimizer.state_dict(),
            #         'model_args': model_args,
            #         'optimizer_args': optimizer_args
            #     },
            #     f=checkpoints_path
            # )  
            # print(f"Saved {self.id} model to {checkpoints_path}")



    def test(self, env: AbstractEnv, deterministic: bool = True):
        criterion = torch.nn.MSELoss()

        self.model.eval()

        dataloader = env.get_dataloader()


        for X, y in tqdm(dataloader):

            outputs = self.model(X)
            # loss = criterion(outputs, y)
            # total_loss += loss.item()

            env.store_result(outputs, y)

        predicted = self.model(X_test)#.detach().numpy().flatten()
        loss = criterion(predicted, y_test)