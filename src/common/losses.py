import torch
from torch import Tensor
from torch.nn.modules.loss import _Loss

class R2Loss(_Loss):
    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super().__init__(size_average, reduce, reduction)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        ss_res = ((target - input)**2).sum()
        ss_tot = ((target - target.mean())**2).sum()
        return 1 - ss_res / ss_tot

class SquaredMeanLoss(_Loss):
    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super().__init__(size_average, reduce, reduction)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        score = ((target - input)/(target + 1e-8))**2
        return torch.mean(score)