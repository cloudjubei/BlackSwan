
from torch import nn
import torch as th
import torch.nn.functional as F

class DenseBlock(nn.Module):
    def __init__(self, in_features, growth_rate, num_layers):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(nn.Linear(in_features + i * growth_rate, growth_rate))

    def forward(self, x):
        inputs = [x]
        for layer in self.layers:
            new_features = F.relu(layer(th.cat(inputs, dim=1)))
            inputs.append(new_features)
        return th.cat(inputs, dim=1)