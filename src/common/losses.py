import torch
from torch import Tensor
import torch.nn as nn
# from torcheval.metrics.functional import r2_score
from sklearn.metrics import r2_score

def R2Loss(input: Tensor, target: Tensor) -> Tensor:
    # print('input: %s, target: %s' % (input, target))
    # if len(input) < 2:
    #     return SquaredMeanLoss(input, target) 
    loss = r2_score(input, target)
    return loss

def SquaredMeanLoss(input: Tensor, target: Tensor) -> Tensor:
    score = ((target - input)/target)**2
    return torch.mean(score)