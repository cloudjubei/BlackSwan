
from typing import List, Type, Union
from stable_baselines3.common.torch_layers import MlpExtractor
from stable_baselines3.common.utils import get_device

import torch as th
from torch import nn

from src.model.custom.customqnetwork import CustomQNetwork

class CustomMlpExtractor(MlpExtractor):

    def __init__(
        self,
        feature_dim: int,
        net_arch: List[int],
        activation_fn: Type[nn.Module],
        device: Union[th.device, str] = "auto",
        custom_net_arch: List[str] = ["Linear"]
    ) -> None:
        super().__init__(
            feature_dim= feature_dim, 
            net_arch= net_arch,
            activation_fn= activation_fn,
            device= device
        )
        device = get_device(device)

        last_net_arch = net_arch[len(net_arch)-1]


        policy_net = CustomQNetwork.create_mlp_custom(feature_dim, last_net_arch, net_arch, activation_fn, custom_net_arch)
        policy_net = policy_net[:-1]
        self.policy_net = nn.Sequential(*policy_net).to(device)
        value_net = CustomQNetwork.create_mlp_custom(feature_dim, last_net_arch, net_arch, activation_fn, custom_net_arch)
        value_net = value_net[:-1]
        self.value_net = nn.Sequential(*value_net).to(device)
    
        # Save dim, used to create the distributions
        self.latent_dim_pi = last_net_arch
        self.latent_dim_vf = last_net_arch
