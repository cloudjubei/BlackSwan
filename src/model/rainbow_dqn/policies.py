from typing import Any, Dict, List, Optional, Type

from gymnasium import spaces
import torch as th
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, FlattenExtractor
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.dqn.policies import DQNPolicy
from torch import nn
import torch.nn.functional as F

from src.model.custom.noisylinear import NoisyLinear

class RainbowPolicy(DQNPolicy):

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
    ) -> None:
        self.n_quantiles = n_quantiles
        self.noisy_net_sigma = noisy_net_sigma
        self.dueling = dueling
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

        action_dim = int(self.action_space.n)  # number of actions

        q_net = self.create_mlp(self.q_net.features_dim, self.n_quantiles, self.net_arch, self.activation_fn)
        q_net.append(NoisyLinear(self.n_quantiles, action_dim, std_init=self.noisy_net_sigma))
        self.q_net.q_net = nn.Sequential(*q_net)

        if self.dueling:
            self.v_net = self.make_q_net()
            v_net = self.create_mlp(self.v_net.features_dim, self.n_quantiles, self.net_arch, self.activation_fn)
            v_net.append(NoisyLinear(self.n_quantiles, 1, std_init=self.noisy_net_sigma))

            self.v_net.q_net = nn.Sequential(*v_net)

        q_net_target = self.create_mlp(self.q_net_target.features_dim, self.n_quantiles, self.net_arch, self.activation_fn)
        q_net_target.append(NoisyLinear(self.n_quantiles, action_dim, std_init=self.noisy_net_sigma))
        self.q_net_target.q_net = nn.Sequential(*q_net_target)
        self.q_net_target.load_state_dict(self.q_net.state_dict())
        self.q_net_target.set_training_mode(False)

    def create_mlp(
        self, 
        input_dim: int,
        output_dim: int,
        net_arch: List[int],
        activation_fn: Type[nn.Module] = nn.ReLU,
        squash_output: bool = False,
    ) -> List[nn.Module]:

        if len(net_arch) > 0:
            modules = [NoisyLinear(input_dim, net_arch[0]), activation_fn()]
        else:
            modules = []

        for idx in range(len(net_arch) - 1):
            modules.append(NoisyLinear(net_arch[idx], net_arch[idx + 1]))
            modules.append(activation_fn())

        if output_dim > 0:
            last_layer_dim = net_arch[-1] if len(net_arch) > 0 else input_dim
            modules.append(NoisyLinear(last_layer_dim, output_dim))
        if squash_output:
            modules.append(nn.Tanh())
        return modules

    def forward(self, obs: th.Tensor) -> th.Tensor:
        features = self.extract_features(obs)
        if self.dueling:
            q_values = self.q_net(features)
            v_values = self.v_net(features)
            q_values = v_values + (q_values - q_values.mean(dim=1, keepdim=True))
        else:
            q_values = self.q_net(features)
        return q_values
