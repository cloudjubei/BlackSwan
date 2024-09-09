from typing import Type
from torch import nn

class ResidualBlock(nn.Module):
    def __init__(self, in_features, activation_fn: Type[nn.Module] = nn.ReLU):
        super(ResidualBlock, self).__init__()
        self.linear = nn.Linear(in_features, in_features)
        self.activation = activation_fn()

    def forward(self, x):
        return x + self.activation(self.linear(x))