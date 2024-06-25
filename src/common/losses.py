import torch
from torch import Tensor
from torch.nn.modules.loss import _Loss
# from torcheval.metrics.functional import r2_score
from sklearn.metrics import r2_score

class R2Loss(_Loss):
    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super().__init__(size_average, reduce, reduction)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return r2_score(input, target)
    
class SquaredMeanLoss(_Loss):
    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super().__init__(size_average, reduce, reduction)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        score = ((target - input)/target)**2
        return torch.mean(score)