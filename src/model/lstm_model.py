from src.model.abstract_model import BaseLSTMModel
from src.conf.model_config import ModelConfig
import torch
import torch.nn as nn
import torch.optim as optim
from src.common.losses import R2Loss, SquaredMeanLoss

class LSTMModel(BaseLSTMModel):
    def __init__(self, config: ModelConfig, input_size: int, output_size: int, device: str):
        super(LSTMModel, self).__init__(config)

        if self.lstm_config.checkpoint_to_load is not None:
            checkpoint_path = self.lstm_config.checkpoints_folder + self.lstm_config.checkpoint_to_load
            checkpoint = torch.load(checkpoint_path)

            self.model_args = checkpoint['model_args']
            self.model = LSTMLinear(**self.model_args)
            self.model.load_state_dict(checkpoint['model_state_dict'])

            self.target_model = LSTMLinear(**self.model_args)
            self.target_model.load_state_dict(checkpoint['target_model_state_dict'])
            self.target_model.load_state_dict(self.model.state_dict())
            self.target_model.eval()

            self.optimizer_args = checkpoint['optimizer_args']
            self.optimizer = optim.AdamW(**self.optimizer_args)
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        else:
            self.model_args = {
                'device': device,
                'input_size': input_size, 
                'hidden_size': self.lstm_config.hidden_size, 
                'num_layers': self.lstm_config.layers, 
                'output_size': output_size, 
                'lstm_dropout': self.lstm_config.lstm_dropout,
                'extra_dropout': self.lstm_config.extra_dropout,
                'first_activation': self.lstm_config.first_activation,
                'last_activation': self.lstm_config.last_activation,
            }
            self.model = LSTMLinear(**self.model_args)

            self.target_model = LSTMLinear(**self.model_args)
            self.target_model.load_state_dict(self.model.state_dict())
            self.target_model.eval()

            self.optimizer_args = { 'params': list(self.model.parameters()), 'lr': self.lstm_config.learning_rate, 'weight_decay': self.lstm_config.weight_decay }
            self.optimizer = optim.AdamW(**self.optimizer_args)
            self.loss_fn = nn.MSELoss()
            # self.loss_fn = nn.BCELoss() if self.lstm_config.loss_fn == 'bce' else (nn.MSELoss() if self.lstm_config.loss_fn == 'mse' else (R2Loss() if self.lstm_config.loss_fn == 'r2' else SquaredMeanLoss()))

    def get_model(self) -> nn.Module:
        return self.model
    def get_model_args(self) -> nn.Module:
        return self.model_args
    def get_target_model(self) -> nn.Module:
        return self.target_model
    def get_loss_fn(self):
        return self.loss_fn
    def get_optimizer(self) -> optim.Optimizer:
        return self.optimizer
    def get_optimizer_args(self):
        return self.optimizer_args
    def get_episodes(self):
        return self.lstm_config.episodes

class LSTMLinear(nn.Module):
    def __init__(self, device: str, input_size: int, hidden_size: int, num_layers: int, output_size: int, lstm_dropout: float, extra_dropout: float, first_activation: str, last_activation: str):
        super().__init__()

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=lstm_dropout, device= device)
        self.activation1 = nn.LeakyReLU() if first_activation == 'lrelu' else (nn.ReLU() if first_activation == 'relu' else nn.Sigmoid())
        self.dropout = nn.Dropout(extra_dropout)
        self.fc = nn.Linear(hidden_size, output_size, device= device)
        self.activation2 = nn.LeakyReLU() if last_activation == 'lrelu' else (nn.ReLU() if last_activation == 'relu' else nn.Sigmoid())

    def forward(self, x):
        if (len(x.shape) == 1):
            x = x.unsqueeze(0)
            
        _, (ht, _) = self.lstm(x)
        out = ht[-1]
        out = self.activation1(out)
        out = self.dropout(out)
        out = self.fc(out)
        out = self.activation2(out)

        if len(out.size()) > 1: 
            out = torch.squeeze(out, 1)
        return out