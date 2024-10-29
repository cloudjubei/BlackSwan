from typing import Any, Dict, List, Optional, Type, Union

import torch as th
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, FlattenExtractor
from stable_baselines3.common.type_aliases import Schedule
from torch import nn

from gymnasium import spaces

from src.model.custom.customqnetwork import CustomQNetwork
from src.model.iqn.policies import IQNPolicy, QuantileNetwork


class CustomQuantileNetwork(QuantileNetwork):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Discrete,
        features_extractor: BaseFeaturesExtractor,
        features_dim: int,
        n_quantiles: int = 64,
        num_cosine: int = 64,
        net_arch: Optional[List[int]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
        custom_net_arch: List[str] = ["Linear"]
    ):
        self.custom_net_arch = custom_net_arch
        super().__init__(
            observation_space= observation_space,
            action_space= action_space,
            features_extractor=features_extractor,
            features_dim= features_dim,
            n_quantiles= n_quantiles,
            num_cosine= num_cosine,
            net_arch= net_arch,
            activation_fn= activation_fn,
            normalize_images=normalize_images,
        )

        action_dim = int(action_space.n)  # number of actions
        quantile_net = CustomQNetwork.create_mlp_custom(features_dim, action_dim , net_arch, activation_fn, custom_net_arch)
        self.quantile_net = nn.Sequential(*quantile_net)

class CustomIQNPolicy(IQNPolicy):

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        n_quantiles: int = 64,
        num_cosine: int = 64,
        net_arch: Optional[List[int]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        custom_net_arch: List[str] = ["Linear"]
    ):
        self.custom_net_arch = custom_net_arch
        super().__init__(
            observation_space= observation_space,
            action_space= action_space,
            lr_schedule= lr_schedule,
            n_quantiles= n_quantiles,
            num_cosine= num_cosine,
            net_arch= net_arch,
            activation_fn= activation_fn,
            features_extractor_class= features_extractor_class,
            features_extractor_kwargs= features_extractor_kwargs,
            normalize_images=normalize_images,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
        )

    def make_quantile_net(self) -> QuantileNetwork:
        # Make sure we always have separate networks for features extractors etc
        net_args = self._update_features_extractor(self.net_args, features_extractor=None)
        net_args["custom_net_arch"] = self.custom_net_arch
        return CustomQuantileNetwork(**net_args).to(self.device)