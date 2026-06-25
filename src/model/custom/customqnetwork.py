from typing import List, Optional, Type

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.dqn.policies import QNetwork
from torch import nn

from gymnasium import spaces

from src.model.custom.attention import AdditiveAttention, GlobalContextAttention, MultiHeadAttention, ScaledDotProductAttention, SelfAttention
from src.model.custom.denseblock import DenseBlock
from src.model.custom.dropconnect import DropConnectLinear
from src.model.custom.memorymodels import GRULocal, LSTMLocal
from src.model.custom.noisylinear import NoisyLinear
from src.model.custom.residualblock import ResidualBlock

# Recipe block names that PRODUCE a Linear-like layer (each consumes one width from net_sizes). Everything
# else (BatchNorm1d, Dropout, activation_fn, ResidualBlock, attention, ...) decorates the current width
# without advancing it. A recipe needs exactly len(net_arch)+1 of these — one per width plus the output.
_LAYER_BLOCKS = frozenset({
    "Linear", "NoisyLinear",
    "weight_norm", "weight_norm2",
    "weight_norm_noisy", "weight_norm_noisy2", "weight_norm_noisy3",
    "weight_norm2_noisy", "weight_norm2_noisy2", "weight_norm2_noisy3",
    "spectral_norm", "spectral_norm2",
    "spectral_norm_noisy", "spectral_norm_noisy2", "spectral_norm_noisy3",
    "spectral_norm2_noisy", "spectral_norm2_noisy2", "spectral_norm2_noisy3",
    "DropConnectLinear",
    "LSTMLocal", "LSTMLocalN", "GRULocal", "GRULocal2", "GRULocal4",
})


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

    # net_arch= [64, 64]
    # custom_net_arch= ["GlobalContextAttention", "activation_fn", "GlobalContextAttention"]

    # net_arch= [64, 64]
    # custom_net_arch= ["Linear", "BatchNorm1d", "activation_fn", "Linear"]

    # net_arch= [64, 64]
    # custom_net_arch= ["Linear", "activation_fn", "SelfAttention", "Linear"]

    @staticmethod
    def _fit_recipe_to_depth(custom_net_arch: List[str], required: int) -> List[str]:
        """Tile a -custom recipe's HIDDEN block-pattern so it has exactly ``required`` layer-producing
        blocks (= len(net_arch)+1), letting ONE recipe build at any net_arch depth. The recipe is split
        into segments, each a run of decorations terminated by a layer-producing block; the LAST segment
        is the output, the rest are hidden units repeated (cycled) to reach the needed hidden count. A
        recipe that already matches is returned UNCHANGED (byte-identical build); one that cannot be
        tiled (no layer block, or no hidden unit to repeat) is returned as-is for the guard to reject."""
        segments: List[List[str]] = []
        current: List[str] = []
        for name in custom_net_arch:
            current.append(name)
            if name in _LAYER_BLOCKS:
                segments.append(current)
                current = []
        if current:
            # Trailing decorations after the last layer block belong to the output segment.
            if not segments:
                return custom_net_arch  # no layer-producing block at all — nothing to tile
            segments[-1] = segments[-1] + current
        if len(segments) == required:
            return custom_net_arch
        needed_hidden = required - 1
        hidden, output = segments[:-1], segments[-1]
        if needed_hidden < 0 or (needed_hidden > 0 and not hidden):
            return custom_net_arch  # un-adaptable — let the caller's guard raise a clear error
        tiled = [hidden[i % len(hidden)] for i in range(needed_hidden)]
        fitted: List[str] = []
        for segment in tiled + [output]:
            fitted.extend(segment)
        return fitted

    @staticmethod
    def create_mlp_custom(
        input_dim: int,
        output_dim: int,
        net_arch: List[int],
        activation_fn: Type[nn.Module] = nn.ReLU,
        custom_net_arch: List[str] = ["Linear"]
    ) -> List[nn.Module]:
        modules = []

        net_sizes = [input_dim] + net_arch + [output_dim]

        # No custom architecture supplied (e.g. a *-custom model whose custom_net_arch resolved empty):
        # fall back to a standard MLP — one Linear+activation per hidden layer plus the output Linear —
        # so the policy still builds instead of crashing on custom_net_arch[0].
        if not custom_net_arch:
            custom_net_arch = ["Linear", "activation_fn"] * len(net_arch) + ["Linear"]

        first_module_name = custom_net_arch[0]
        if first_module_name == "LSTFull":
            return [LSTMLocal(input_dim, net_arch[0], len(net_arch)), nn.Linear(net_arch[0], output_dim, bias=True)]
        elif first_module_name == "LSTFullN":
            return [LSTMLocal(input_dim, net_arch[0], len(net_arch), with_hidden=False), nn.Linear(net_arch[0], output_dim, bias=True)]
        elif first_module_name == "GRUFull":
            return [GRULocal(input_dim, net_arch[0], len(net_arch), with_hidden=False), nn.Linear(net_arch[0], output_dim, bias=True)]

        # A -custom recipe consumes one width per layer-producing block; it MUST define exactly
        # len(net_arch)+1 of them (one per width plus the output layer). Rather than force every net_arch
        # depth to ship its own hand-matched recipe, tile the recipe's hidden block-pattern to fit the
        # requested depth (so one recipe builds at any net_arch — see _fit_recipe_to_depth). A matching
        # recipe is unchanged. If it still can't be made to fit (e.g. a single-block recipe asked for
        # several hidden layers), fail fast here instead of crashing deep in the forward pass.
        required = len(net_sizes) - 1
        custom_net_arch = CustomQNetwork._fit_recipe_to_depth(custom_net_arch, required)
        produced = sum(1 for m in custom_net_arch if m in _LAYER_BLOCKS)
        if produced != required:
            raise ValueError(
                f"custom_net_arch defines {produced} layer-producing block(s) but net_arch={net_arch} "
                f"requires exactly {required} (= len(net_arch) + 1: one per width plus the output layer). "
                f"Fix the preset: use a net_arch with {produced - 1} width(s) for this recipe, or a recipe "
                f"with {required} Linear-like blocks. custom_net_arch={custom_net_arch}"
            )

        idx = 0

        for module_idx in range(len(custom_net_arch)):
            module_name = custom_net_arch[module_idx]

            if module_name == "activation_fn":
                modules.append(activation_fn())
            elif module_name == "Linear":
                modules.append(nn.Linear(net_sizes[idx], net_sizes[idx + 1], bias=True))
                idx += 1 
            elif module_name == "NoisyLinear":
                modules.append(NoisyLinear(net_sizes[idx], net_sizes[idx + 1], std_init=0.5))
                idx += 1 
            elif module_name == "weight_norm":
                modules.append(nn.utils.parametrizations.weight_norm(nn.Linear(net_sizes[idx], net_sizes[idx + 1], bias=True)))
                idx += 1 
            elif module_name == "weight_norm2":
                modules.append(nn.utils.parametrizations.weight_norm(nn.Linear(net_sizes[idx], net_sizes[idx + 1], bias=True), dim= None))
                idx += 1 
            elif module_name == "weight_norm_noisy": 
                modules.append(nn.utils.parametrizations.weight_norm(NoisyLinear(net_sizes[idx], net_sizes[idx + 1], std_init=0.5), "weight_epsilon"))
                idx += 1 
            elif module_name == "weight_norm_noisy2": 
                modules.append(nn.utils.parametrizations.weight_norm(NoisyLinear(net_sizes[idx], net_sizes[idx + 1], std_init=0.5), "weight_sigma"))
                idx += 1 
            elif module_name == "weight_norm_noisy3": 
                modules.append(nn.utils.parametrizations.weight_norm(NoisyLinear(net_sizes[idx], net_sizes[idx + 1], std_init=0.5), "weight_mu"))
                idx += 1 
            elif module_name == "weight_norm2_noisy":
                modules.append(nn.utils.parametrizations.weight_norm(NoisyLinear(net_sizes[idx], net_sizes[idx + 1], std_init=0.5), "weight_epsilon", dim= None))
                idx += 1 
            elif module_name == "weight_norm2_noisy2":
                modules.append(nn.utils.parametrizations.weight_norm(NoisyLinear(net_sizes[idx], net_sizes[idx + 1], std_init=0.5), "weight_sigma", dim= None))
                idx += 1 
            elif module_name == "weight_norm2_noisy3":
                modules.append(nn.utils.parametrizations.weight_norm(NoisyLinear(net_sizes[idx], net_sizes[idx + 1], std_init=0.5), "weight_mu", dim= None))
                idx += 1 
            elif module_name == "spectral_norm":
                modules.append(nn.utils.parametrizations.spectral_norm(nn.Linear(net_sizes[idx], net_sizes[idx + 1], bias=True)))
                idx += 1 
            elif module_name == "spectral_norm2":
                modules.append(nn.utils.parametrizations.spectral_norm(nn.Linear(net_sizes[idx], net_sizes[idx + 1], bias=True), dim= None))
                idx += 1 
            elif module_name == "spectral_norm_noisy":
                modules.append(nn.utils.parametrizations.spectral_norm(NoisyLinear(net_sizes[idx], net_sizes[idx + 1], std_init=0.5), "weight_epsilon"))
                idx += 1 
            elif module_name == "spectral_norm_noisy2":
                modules.append(nn.utils.parametrizations.spectral_norm(NoisyLinear(net_sizes[idx], net_sizes[idx + 1], std_init=0.5), "weight_sigma"))
                idx += 1 
            elif module_name == "spectral_norm_noisy3":
                modules.append(nn.utils.parametrizations.spectral_norm(NoisyLinear(net_sizes[idx], net_sizes[idx + 1], std_init=0.5), "weight_mu"))
                idx += 1 
            elif module_name == "spectral_norm2_noisy":
                modules.append(nn.utils.parametrizations.spectral_norm(NoisyLinear(net_sizes[idx], net_sizes[idx + 1], std_init=0.5), "weight_epsilon", dim= None))
                idx += 1 
            elif module_name == "spectral_norm2_noisy2":
                modules.append(nn.utils.parametrizations.spectral_norm(NoisyLinear(net_sizes[idx], net_sizes[idx + 1], std_init=0.5), "weight_sigma", dim= None))
                idx += 1 
            elif module_name == "spectral_norm2_noisy3":
                modules.append(nn.utils.parametrizations.spectral_norm(NoisyLinear(net_sizes[idx], net_sizes[idx + 1], std_init=0.5), "weight_mu", dim= None))
                idx += 1 
            elif module_name == "DropConnectLinear":
                modules.append(DropConnectLinear(net_sizes[idx], net_sizes[idx + 1], p=0.5))
                idx += 1 
            elif module_name == "LSTMLocal":
                modules.append(LSTMLocal(net_sizes[idx], net_sizes[idx + 1]))
                idx += 1 
            elif module_name == "LSTMLocalN":
                modules.append(LSTMLocal(net_sizes[idx], net_sizes[idx + 1], with_hidden=False))
                idx += 1 
            elif module_name == "GRULocal":
                modules.append(GRULocal(net_sizes[idx], net_sizes[idx + 1], with_hidden=False))
                idx += 1 
            elif module_name == "GRULocal2":
                modules.append(GRULocal(net_sizes[idx], net_sizes[idx + 1], 2, with_hidden=False))
                idx += 1 
            elif module_name == "GRULocal4":
                modules.append(GRULocal(net_sizes[idx], net_sizes[idx + 1], 4, with_hidden=False))
                idx += 1 
            elif module_name == "BatchNorm1d":
                modules.append(nn.BatchNorm1d(net_sizes[idx]))
            elif module_name == "Dropout":
                modules.append(nn.Dropout(p=0.5))
            elif module_name == "Dropout1":
                modules.append(nn.Dropout(p=0.1))
            elif module_name == "Dropout2":
                modules.append(nn.Dropout(p=0.2))
            elif module_name == "Dropout05":
                modules.append(nn.Dropout(p=0.05))
            elif module_name == "ResidualBlock":
                modules.append(ResidualBlock(net_sizes[idx], activation_fn= activation_fn))
            elif module_name == "LayerNorm":
                modules.append(nn.LayerNorm(net_sizes[idx]))
            elif module_name == "DenseBlock":
                modules.append(DenseBlock(net_sizes[idx], growth_rate=32, num_layers=2))
                modules.append(nn.Linear(net_sizes[idx] + 2 * 32, net_sizes[idx]))
            elif module_name == "DenseBlock4":
                modules.append(DenseBlock(net_sizes[idx], growth_rate=32, num_layers=4))
                modules.append(nn.Linear(net_sizes[idx] + 4 * 32, net_sizes[idx]))
            elif module_name == "SelfAttention":
                modules.append(SelfAttention(net_sizes[idx]))
            elif module_name == "ScaledDotProductAttention":
                modules.append(ScaledDotProductAttention(net_sizes[idx]))
            elif module_name == "MultiHeadAttention":
                modules.append(MultiHeadAttention(net_sizes[idx], num_heads=2))
            elif module_name == "MultiHeadAttention4":
                modules.append(MultiHeadAttention(net_sizes[idx], num_heads=4))
            elif module_name == "MultiHeadAttention8":
                modules.append(MultiHeadAttention(net_sizes[idx], num_heads=8))
            elif module_name == "AdditiveAttention":
                modules.append(AdditiveAttention(net_sizes[idx]))
            elif module_name == "GlobalContextAttention":
                modules.append(GlobalContextAttention(net_sizes[idx]))

        return modules