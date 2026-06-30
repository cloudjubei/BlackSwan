"""Recurrent actor-critic policies whose recurrent core is a GRU or a diagonal SSM (S4D).

sb3-contrib's ``RecurrentActorCriticPolicy`` hard-wires ``nn.LSTM`` as the recurrent core, but all
of its forward / predict / evaluate machinery only ever touches the core through the ``nn.LSTM``
surface (``input_size`` / ``hidden_size`` / ``num_layers`` + the ``(input, (h, c)) -> (output,
(h, c))`` call). ``RecurrentCoreActorCriticPolicy`` rebuilds the parent's init in the same order
but plugs in a subclass-supplied core instead, decoupling the threaded-state width
(``core.hidden_size``) from the per-step output width (``lstm_output_dim``) — which the S4D core
needs, since its packed complex state is wider than its readout. Everything else (rollout buffer,
collect, train, eval state threading) is reused verbatim from the parent.

The LSTM levers carry over 1:1: ``lstm_hidden_size`` is the recurrent width (GRU hidden units / S4D
channels), and ``shared_lstm`` / ``enable_critic_lstm`` choose shared, separate, or no critic core.
"""

from typing import Any, Dict, List, Optional, Type, Union

import torch as th
from gymnasium import spaces
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, FlattenExtractor
from stable_baselines3.common.type_aliases import Schedule
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
from torch import nn

from src.model.custom.custommlpextractor import CustomMlpExtractor
from src.model.custom.recurrent_cores import GRURecurrentCore, S4DRecurrentCore, S4D_DEFAULT_STATE_DIM


class RecurrentCoreActorCriticPolicy(RecurrentActorCriticPolicy):
    """Recurrent actor-critic policy with a pluggable non-LSTM recurrent core.

    Subclasses implement ``_make_core(input_size) -> nn.Module`` returning a core that honours the
    ``nn.LSTM`` interface (see ``recurrent_cores``). The per-step output width fed to the MLP head
    is always ``lstm_hidden_size``; the threaded-state width is whatever the core reports.
    """

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
        custom_net_arch: List[str] = ["Linear"],
    ):
        self.recurrent_hidden_size = lstm_hidden_size
        self.n_recurrent_layers = n_lstm_layers
        self.custom_net_arch = custom_net_arch
        # Per-step output width feeding the MLP head; the state width is set later from the core.
        self.lstm_output_dim = lstm_hidden_size

        # Skip RecurrentActorCriticPolicy.__init__ (it builds nn.LSTM) and run the grandparent,
        # which builds the features extractor + MLP/value heads off self.lstm_output_dim.
        super(RecurrentActorCriticPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            ortho_init,
            use_sde,
            log_std_init,
            full_std,
            use_expln,
            squash_output,
            features_extractor_class,
            features_extractor_kwargs,
            share_features_extractor,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
        )

        self.lstm_kwargs = {}
        self.shared_lstm = shared_lstm
        self.enable_critic_lstm = enable_critic_lstm
        self.lstm_actor = self._make_core(self.features_dim)
        self.lstm_hidden_state_shape = (self.lstm_actor.num_layers, 1, self.lstm_actor.hidden_size)
        self.critic = None
        self.lstm_critic = None
        assert not (
            self.shared_lstm and self.enable_critic_lstm
        ), "You must choose between shared, separate or no recurrent core for the critic."
        assert not (
            self.shared_lstm and not self.share_features_extractor
        ), "If the features extractor is not shared, the recurrent core cannot be shared."

        if not (self.shared_lstm or self.enable_critic_lstm):
            self.critic = nn.Linear(self.features_dim, self.lstm_output_dim)
        if self.enable_critic_lstm:
            self.lstm_critic = self._make_core(self.features_dim)

        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def _make_core(self, input_size: int) -> nn.Module:
        raise NotImplementedError


class _CustomMlpHeadMixin:
    """Swap the policy/value head for BlackSwan's ``custom_net_arch``-driven CustomMlpExtractor."""

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = CustomMlpExtractor(
            feature_dim=self.lstm_output_dim,
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            device=self.device,
            custom_net_arch=self.custom_net_arch,
        )


class GRURecurrentActorCriticPolicy(RecurrentCoreActorCriticPolicy):
    """RecurrentPPO policy with a GRU recurrent core (default MLP head)."""

    def _make_core(self, input_size: int) -> nn.Module:
        return GRURecurrentCore(input_size, self.recurrent_hidden_size, self.n_recurrent_layers)


class CustomGRURecurrentActorCriticPolicy(_CustomMlpHeadMixin, GRURecurrentActorCriticPolicy):
    """GRU recurrent core with the custom_net_arch policy/value head."""


class S4DRecurrentActorCriticPolicy(RecurrentCoreActorCriticPolicy):
    """RecurrentPPO policy with a diagonal SSM (S4D) recurrent core (default MLP head)."""

    def __init__(self, *args, ssm_state_dim: int = S4D_DEFAULT_STATE_DIM, **kwargs):
        self.ssm_state_dim = ssm_state_dim
        super().__init__(*args, **kwargs)

    def _make_core(self, input_size: int) -> nn.Module:
        return S4DRecurrentCore(input_size, self.recurrent_hidden_size, state_dim=self.ssm_state_dim)


class CustomS4DRecurrentActorCriticPolicy(_CustomMlpHeadMixin, S4DRecurrentActorCriticPolicy):
    """S4D recurrent core with the custom_net_arch policy/value head."""
