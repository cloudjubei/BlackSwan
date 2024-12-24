from typing import Any, Dict, List, Optional, Type, Union

import torch as th
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, FlattenExtractor
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.dqn.policies import DQNPolicy
from stable_baselines3.common.policies import ActorCriticPolicy
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
from sb3_contrib.qrdqn.policies import QRDQNPolicy, QuantileNetwork
from torch import nn

from gymnasium import spaces

from src.model.custom.custommlpextractor import CustomMlpExtractor
from src.model.custom.customqnetwork import CustomQNetwork
from src.model.dueling_dqn.policies import DuelingDQNPolicy
from src.model.rainbow_dqn.policies import RainbowPolicy
from src.model.iqn.policies import IQNPolicy


class CustomRecurrentActorCriticPolicy(RecurrentActorCriticPolicy):

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        share_features_extractor: bool = True,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        lstm_hidden_size: int = 256,
        n_lstm_layers: int = 1,
        shared_lstm: bool = False,
        enable_critic_lstm: bool = True,
        lstm_kwargs: Optional[Dict[str, Any]] = None,
        custom_net_arch: List[str] = ["Linear"]
    ):
        self.custom_net_arch = custom_net_arch
        super().__init__(
            observation_space= observation_space,
            action_space= action_space,
            lr_schedule= lr_schedule,
            net_arch= net_arch,
            activation_fn= activation_fn,
            ortho_init= ortho_init,
            use_sde= use_sde,
            log_std_init= log_std_init,
            full_std= full_std,
            use_expln= use_expln,
            squash_output= squash_output,
            features_extractor_class= features_extractor_class,
            features_extractor_kwargs= features_extractor_kwargs,
            share_features_extractor= share_features_extractor,
            normalize_images= normalize_images,
            optimizer_class= optimizer_class,
            optimizer_kwargs= optimizer_kwargs,
            lstm_hidden_size= lstm_hidden_size,
            n_lstm_layers= n_lstm_layers,
            shared_lstm= shared_lstm,
            enable_critic_lstm= enable_critic_lstm,
            lstm_kwargs= lstm_kwargs
        )

    def _build_mlp_extractor(self) -> None:
        """
        Create the policy and value networks.
        Part of the layers can be shared.
        """
        # Note: If net_arch is None and some features extractor is used,
        #       net_arch here is an empty list and mlp_extractor does not
        #       really contain any layers (acts like an identity module).
        self.mlp_extractor = CustomMlpExtractor(
            feature_dim= self.lstm_output_dim,
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            device=self.device,
            custom_net_arch= self.custom_net_arch
        )

class CustomActorCriticPolicy(ActorCriticPolicy):

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        share_features_extractor: bool = True,
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
            net_arch= net_arch,
            activation_fn= activation_fn,
            ortho_init= ortho_init,
            use_sde= use_sde,
            log_std_init= log_std_init,
            full_std= full_std,
            use_expln= use_expln,
            squash_output= squash_output,
            features_extractor_class= features_extractor_class,
            features_extractor_kwargs= features_extractor_kwargs,
            share_features_extractor= share_features_extractor,
            normalize_images= normalize_images,
            optimizer_class= optimizer_class,
            optimizer_kwargs= optimizer_kwargs
        )

    def _build_mlp_extractor(self) -> None:
        """
        Create the policy and value networks.
        Part of the layers can be shared.
        """
        # Note: If net_arch is None and some features extractor is used,
        #       net_arch here is an empty list and mlp_extractor does not
        #       really contain any layers (acts like an identity module).
        self.mlp_extractor = CustomMlpExtractor(
            self.features_dim,
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            device=self.device,
            custom_net_arch= self.custom_net_arch
        )
    
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

class CustomRainbowPolicy(RainbowPolicy):

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
        n_quantiles: int = 200,
        noisy_net_sigma: float = 0.5,
        dueling: bool = True,
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
            n_quantiles=n_quantiles,
            noisy_net_sigma=noisy_net_sigma,
            dueling=dueling
        )

    def create_mlp(
        self, 
        input_dim: int,
        output_dim: int,
        net_arch: List[int],
        activation_fn: Type[nn.Module] = nn.ReLU,
        squash_output: bool = False,
    ) -> List[nn.Module]:

        return CustomQNetwork.create_mlp_custom(input_dim, output_dim, net_arch, activation_fn, self.custom_net_arch)#.to(self.device)

    def make_q_net(self) -> CustomQNetwork:
        # Make sure we always have separate networks for features extractors etc
        net_args = self._update_features_extractor(self.net_args, features_extractor=None)
        net_args["custom_net_arch"] = self.custom_net_arch
        return CustomQNetwork(**net_args).to(self.device)

class CustomQRDQNPolicy(QRDQNPolicy):

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Discrete,
        lr_schedule: Schedule,
        n_quantiles: int = 200,
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
    

class CustomQuantileNetwork(QuantileNetwork):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Discrete,
        features_extractor: BaseFeaturesExtractor,
        features_dim: int,
        n_quantiles: int = 200,
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
            net_arch= net_arch,
            activation_fn= activation_fn,
            normalize_images=normalize_images,
        )

        action_dim = int(self.action_space.n)  # number of actions
        quantile_net = CustomQNetwork.create_mlp_custom(self.features_dim, action_dim * self.n_quantiles, net_arch, activation_fn, self.custom_net_arch)
        self.quantile_net = nn.Sequential(*quantile_net)