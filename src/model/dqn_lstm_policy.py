import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.dqn.policies import DQNPolicy

class LSTMFCE(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=16, lstm_hidden_size=256, lstm_num_layers=1):
        super(LSTMFCE, self).__init__(observation_space, features_dim=features_dim)
        
        self.lstm_hidden_size = lstm_hidden_size

        self.lstm = nn.LSTM(observation_space.shape[0], hidden_size=lstm_hidden_size, num_layers=lstm_num_layers, batch_first=True)
        self.linear = nn.Linear(lstm_hidden_size, features_dim)

    def forward(self, observations):
        lstm_out, _ = self.lstm(observations)
        features = self.linear(lstm_out)
        return features