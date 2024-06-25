from src.model.abstract_model import BaseMLPModel
from src.conf.model_config import ModelConfig
import torch
import torch.nn as nn
import torch.optim as optim
from src.common.losses import R2Loss, SquaredMeanLoss

class MLPModel(BaseMLPModel):
    def __init__(self, config: ModelConfig, input_size: int, output_size: int, device: str):
        super(MLPModel, self).__init__(config)

        if self.mlp_config.checkpoint_to_load is not None:
            checkpoint_path = self.mlp_config.checkpoints_folder + self.mlp_config.checkpoint_to_load
            checkpoint = torch.load(checkpoint_path)

            self.model_args = checkpoint['model_args']
            self.model = MLPBasic(**self.model_args)
            self.model.load_state_dict(checkpoint['model_state_dict'])

            self.target_model = MLPBasic(**self.model_args)
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
                'hidden_dim1': self.mlp_config.hidden_dim1, 
                'hidden_dim2': self.mlp_config.hidden_dim2, 
                'output_size': output_size, 
                'first_activation': self.mlp_config.first_activation,
                'last_activation': self.mlp_config.last_activation,
            }
            self.model = MLPBasic(**self.model_args)

            self.target_model = MLPBasic(**self.model_args)
            self.target_model.load_state_dict(self.model.state_dict())
            self.target_model.eval()

            self.optimizer_args = { 'params': list(self.model.parameters()), 'lr': self.mlp_config.learning_rate, 'weight_decay': self.mlp_config.weight_decay }
            self.optimizer = optim.AdamW(**self.optimizer_args)
            self.loss_fn = nn.MSELoss()
            # self.loss_fn = nn.BCELoss() if self.mlp_config.loss_fn == 'bce' else (nn.MSELoss() if self.mlp_config.loss_fn == 'mse' else (R2Loss() if self.mlp_config.loss_fn == 'r2' else SquaredMeanLoss()))

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
        return self.mlp_config.episodes

class MLPBasic(nn.Module):
    def __init__(self, device: str, input_size: int, hidden_dim1: int, hidden_dim2: int, output_size: int, first_activation: str, last_activation: str):
        super().__init__()

        self.fc1 = nn.Linear(input_size, hidden_dim1, device= device)
        self.activation1 = nn.LeakyReLU() if first_activation == 'lrelu' else (nn.ReLU() if first_activation == 'relu' else nn.Sigmoid())
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2, device= device)
        # self.dropout = nn.Dropout(dropout)
        self.activation2 = nn.LeakyReLU() if last_activation == 'lrelu' else (nn.ReLU() if last_activation == 'relu' else nn.Sigmoid())
        self.fc3 = nn.Linear(hidden_dim2, output_size, device= device)

    def forward(self, x):
        # if (len(x.shape) == 1):
        #     x = x.unsqueeze(0)
            
        out = self.fc1(x)
        out = self.activation1(out)
        out = self.fc2(out)
        # out = self.dropout(out)
        out = self.activation2(out)
        out = self.fc3(out)
        return out

        # if len(out.size()) > 1: 
        #     out = torch.squeeze(out, 1)
        # return out