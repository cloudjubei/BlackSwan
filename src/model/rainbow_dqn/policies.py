from typing import Any, Dict, List, Optional, Type

from gymnasium import spaces
import torch as th
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, FlattenExtractor, create_mlp
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.dqn.policies import DQNPolicy
from torch import nn
import torch.nn.functional as F

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
        super(RainbowPolicy, self).__init__(
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

        q_net = create_mlp(self.q_net.features_dim, self.n_quantiles, self.net_arch, self.activation_fn)
        q_net.append(NoisyLinear(self.n_quantiles, action_dim, sigma_init=self.noisy_net_sigma))
        self.q_net.q_net = nn.Sequential(*q_net)

        if self.dueling:
            self.v_net = self.make_q_net()
            v_net = create_mlp(self.v_net.features_dim, self.n_quantiles, self.net_arch, self.activation_fn)
            v_net.append(NoisyLinear(self.n_quantiles, 1, sigma_init=self.noisy_net_sigma))

            self.v_net.q_net = nn.Sequential(*v_net)

        q_net_target = create_mlp(self.q_net_target.features_dim, self.n_quantiles, self.net_arch, self.activation_fn)
        q_net_target.append(NoisyLinear(self.n_quantiles, action_dim, sigma_init=self.noisy_net_sigma))
        self.q_net_target.q_net = nn.Sequential(*q_net_target)
        self.q_net_target.load_state_dict(self.q_net.state_dict())
        self.q_net_target.set_training_mode(False)

    def create_mlp(
        input_dim: int,
        output_dim: int,
        net_arch: List[int],
        activation_fn: Type[nn.Module] = nn.ReLU,
        squash_output: bool = False,
        with_bias: bool = True,
    ) -> List[nn.Module]:

        if len(net_arch) > 0:
            modules = [NoisyLinear(input_dim, net_arch[0], bias=with_bias), activation_fn()]
        else:
            modules = []

        for idx in range(len(net_arch) - 1):
            modules.append(NoisyLinear(net_arch[idx], net_arch[idx + 1], bias=with_bias))
            modules.append(activation_fn())

        if output_dim > 0:
            last_layer_dim = net_arch[-1] if len(net_arch) > 0 else input_dim
            modules.append(NoisyLinear(last_layer_dim, output_dim, bias=with_bias))
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

class NoisyLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, sigma_init: float = 0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_init = sigma_init

        self.weight_mu = nn.Parameter(th.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(th.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', th.empty(out_features, in_features))

        self.bias_mu = nn.Parameter(th.empty(out_features))
        self.bias_sigma = nn.Parameter(th.empty(out_features))
        self.register_buffer('bias_epsilon', th.empty(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / th.sqrt(th.tensor(self.in_features, dtype=th.float32))
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init / th.sqrt(th.tensor(self.in_features, dtype=th.float32)))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma_init / th.sqrt(th.tensor(self.out_features, dtype=th.float32)))

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def _scale_noise(self, size: int):
        x = th.randn(size)
        return x.sign().mul_(x.abs().sqrt_())

    def forward(self, input: th.Tensor) -> th.Tensor:
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(input, weight, bias)
