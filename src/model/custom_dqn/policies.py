from typing import Any, Dict, List, Optional, Type

import torch as th
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, CombinedExtractor, NatureCNN, FlattenExtractor
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.dqn.policies import DQNPolicy, QNetwork
from torch import nn
from torch.nn.utils import weight_norm, spectral_norm

from gymnasium import spaces

from src.model.custom_dqn.attention import AdditiveAttention, GlobalContextAttention, MultiHeadAttention, ScaledDotProductAttention, SelfAttention
from src.model.custom_dqn.denseblock import DenseBlock
from src.model.custom_dqn.dropconnect import DropConnectLinear
from src.model.custom_dqn.memorymodels import GRULocal, LSTMLocal
from src.model.dueling_dqn.policies import DuelingDQNPolicy

class CustomQNetwork(QNetwork):

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Discrete,
        features_extractor: BaseFeaturesExtractor,
        features_dim: int,
        net_arch: Optional[List[int]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
        custom_net_arch: List[str] = ["Linear"]
    ) -> None:
        super().__init__(
            observation_space,
            action_space,
            features_extractor,
            features_dim,
            net_arch,
            activation_fn,
            normalize_images,
        )

        action_dim = int(self.action_space.n)
        q_net = self.create_mlp_custom(self.features_dim, action_dim, self.net_arch, self.activation_fn, custom_net_arch)
        self.q_net = nn.Sequential(*q_net)
        print(self.q_net)

    # net_arch= [64, 64]
    # custom_net_arch= ["GlobalContextAttention", "activation_fn", "GlobalContextAttention"]

    # net_arch= [64, 64]
    # custom_net_arch= ["Linear", "BatchNorm1d", "activation_fn", "Linear"]

    # net_arch= [64, 64]
    # custom_net_arch= ["Linear", "activation_fn", "SelfAttention", "Linear"]

    def create_mlp_custom(
        self, 
        input_dim: int,
        output_dim: int,
        net_arch: List[int],
        activation_fn: Type[nn.Module] = nn.ReLU,
        custom_net_arch: List[str] = ["Linear"]
    ) -> List[nn.Module]:
        modules = []

        first_module_name = custom_net_arch[0]
        if first_module_name == "weight_norm":
            modules.append(weight_norm(nn.Linear(input_dim, net_arch[0], bias=True)))
        elif first_module_name == "spectral_norm":
            modules.append(spectral_norm(nn.Linear(input_dim, net_arch[0], bias=True)))
        elif first_module_name == "DropConnectLinear":
            modules.append(DropConnectLinear(input_dim, net_arch[0], p=0.5))
        elif first_module_name == "LSTMLocal":
            modules.append(LSTMLocal(input_dim, net_arch[0]))
        elif first_module_name == "LSTMLocalN":
            modules.append(LSTMLocal(input_dim, net_arch[0], with_hidden=False))
        elif first_module_name == "LSTFull":
            return [LSTMLocal(input_dim, net_arch[0], len(net_arch)), nn.Linear(net_arch[0], output_dim, bias=True)]
        elif first_module_name == "LSTFullN":
            return [LSTMLocal(input_dim, net_arch[0], len(net_arch), with_hidden=False), nn.Linear(net_arch[0], output_dim, bias=True)]
        elif first_module_name == "GRULocal":
            modules.append(GRULocal(input_dim, net_arch[0], with_hidden=False))
        elif first_module_name == "GRULocal2":
            modules.append(GRULocal(input_dim, net_arch[0], 2, with_hidden=False))
        elif first_module_name == "GRULocal4":
            modules.append(GRULocal(input_dim, net_arch[0], 4, with_hidden=False))
        elif first_module_name == "GRUFull":
            return [GRULocal(input_dim, net_arch[0], len(net_arch), with_hidden=False), nn.Linear(net_arch[0], output_dim, bias=True)]
        else:
            modules.append(nn.Linear(input_dim, net_arch[0], bias=True))

        idx = 0

        for module_idx in range(len(custom_net_arch)-2):
            module_name = custom_net_arch[module_idx+1]

            if module_name == "activation_fn":
                modules.append(activation_fn())
            elif module_name == "Linear":
                modules.append(nn.Linear(net_arch[idx], net_arch[idx + 1], bias=True))
                idx += 1 
            elif module_name == "weight_norm":
                modules.append(weight_norm(nn.Linear(net_arch[idx], net_arch[idx + 1], bias=True)))
                idx += 1 
            elif module_name == "spectral_norm":
                modules.append(spectral_norm(nn.Linear(net_arch[idx], net_arch[idx + 1], bias=True)))
                idx += 1 
            elif module_name == "DropConnectLinear":
                modules.append(DropConnectLinear(net_arch[idx], net_arch[idx + 1], p=0.5))
                idx += 1 
            elif module_name == "LSTMLocal":
                modules.append(LSTMLocal(net_arch[idx], net_arch[idx + 1]))
                idx += 1 
            elif module_name == "LSTMLocalN":
                modules.append(LSTMLocal(net_arch[idx], net_arch[idx + 1], with_hidden=False))
                idx += 1 
            elif module_name == "GRULocal":
                modules.append(GRULocal(net_arch[idx], net_arch[idx + 1], with_hidden=False))
                idx += 1 
            elif module_name == "GRULocal2":
                modules.append(GRULocal(net_arch[idx], net_arch[idx + 1], 2, with_hidden=False))
                idx += 1 
            elif module_name == "GRULocal4":
                modules.append(GRULocal(net_arch[idx], net_arch[idx + 1], 4, with_hidden=False))
                idx += 1 
            elif module_name == "BatchNorm1d":
                modules.append(nn.BatchNorm1d(net_arch[idx]))
            elif module_name == "Dropout":
                modules.append(nn.Dropout(p=0.5))
            elif module_name == "Dropout1":
                modules.append(nn.Dropout(p=0.1))
            elif module_name == "Dropout2":
                modules.append(nn.Dropout(p=0.05))
            elif module_name == "ResidualBlock":
                modules.append(ResidualBlock(net_arch[idx]))
            elif module_name == "LayerNorm":
                modules.append(nn.LayerNorm(net_arch[idx]))
            elif module_name == "DenseBlock":
                modules.append(DenseBlock(net_arch[idx], growth_rate=32, num_layers=2))
                modules.append(nn.Linear(net_arch[idx] + 2 * 32, net_arch[idx]))
            elif module_name == "DenseBlock4":
                modules.append(DenseBlock(net_arch[idx], growth_rate=32, num_layers=4))
                modules.append(nn.Linear(net_arch[idx] + 4 * 32, net_arch[idx]))
            elif module_name == "SelfAttention":
                modules.append(SelfAttention(net_arch[idx]))
            elif module_name == "ScaledDotProductAttention":
                modules.append(ScaledDotProductAttention(net_arch[idx]))
            elif module_name == "MultiHeadAttention":
                modules.append(MultiHeadAttention(net_arch[idx], num_heads=4))
            elif module_name == "AdditiveAttention":
                modules.append(AdditiveAttention(net_arch[idx]))
            elif module_name == "GlobalContextAttention":
                modules.append(GlobalContextAttention(net_arch[idx]))
            
        last_module_name = custom_net_arch[len(custom_net_arch)-1]
        if last_module_name == "weight_norm":
            modules.append(weight_norm(nn.Linear(net_arch[-1], output_dim, bias=True)))
        elif last_module_name == "spectral_norm":
            modules.append(spectral_norm(nn.Linear(net_arch[-1], output_dim, bias=True)))
        elif last_module_name == "DropConnectLinear":
            modules.append(DropConnectLinear(net_arch[-1], output_dim, p=0.5))
        elif last_module_name == "LSTMLocal":
            modules.append(LSTMLocal(net_arch[-1], output_dim))
        elif last_module_name == "LSTMLocalN":
            modules.append(LSTMLocal(net_arch[-1], output_dim, with_hidden=False))
        elif last_module_name == "GRULocal":
            modules.append(GRULocal(net_arch[-1], output_dim, with_hidden=False))
        elif last_module_name == "GRULocal2":
            modules.append(GRULocal(net_arch[-1], output_dim, 2, with_hidden=False))
        elif last_module_name == "GRULocal4":
            modules.append(GRULocal(net_arch[-1], output_dim, 4, with_hidden=False))
        else:
            modules.append(nn.Linear(net_arch[-1], output_dim, bias=True))

        return modules
    
class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        self.linear = nn.Linear(in_features, in_features)
        self.relu = nn.ReLU()

    def forward(self, x):
        return x + self.relu(self.linear(x))
    
class CustomDQNPolicy(DQNPolicy):

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Discrete,
        lr_schedule: Schedule,
        net_arch: Optional[List[int]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        custom_net_arch: List[str] = ["Linear"]
    ) -> None:
        self.custom_net_arch = custom_net_arch
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            normalize_images=normalize_images,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
        )

    def make_q_net(self) -> CustomQNetwork:
        # Make sure we always have separate networks for features extractors etc
        net_args = self._update_features_extractor(self.net_args, features_extractor=None)
        net_args["custom_net_arch"] = self.custom_net_arch
        return CustomQNetwork(**net_args).to(self.device)
    

class CustomDuelingDQNPolicy(DuelingDQNPolicy):

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Discrete,
        lr_schedule: Schedule,
        net_arch: Optional[List[int]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        custom_net_arch: List[str] = ["Linear"]
    ) -> None:
        self.custom_net_arch = custom_net_arch
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            normalize_images=normalize_images,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
        )

    def make_q_net(self) -> CustomQNetwork:
        # Make sure we always have separate networks for features extractors etc
        net_args = self._update_features_extractor(self.net_args, features_extractor=None)
        net_args["custom_net_arch"] = self.custom_net_arch
        return CustomQNetwork(**net_args).to(self.device)

