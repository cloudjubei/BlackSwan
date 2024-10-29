
from torch import nn
import torch
import torch.nn.functional as F

class DropConnectLinear(nn.Module):
    def __init__(self, in_features, out_features, p=0.5):
        super(DropConnectLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.p = p
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        if self.training:
            weight = self.linear.weight
            drop_mask = torch.rand(weight.size(), device=weight.device) >= self.p
            weight = weight * drop_mask.float()
            bias = self.linear.bias
            return F.linear(x, weight, bias)
        else:
            return self.linear(x)