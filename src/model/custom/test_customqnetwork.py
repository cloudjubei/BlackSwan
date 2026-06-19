"""Tests for CustomQNetwork.create_mlp_custom — the static recipe -> nn.Module list builder.

This is the deterministic core of CustomQNetwork: it maps a list of layer names + widths onto a
module list. We exercise it directly (no QNetwork/policy init) so we cover the validation guard,
the empty-recipe fallback, the LSTM/GRU "*Full" shortcuts, the alternating Linear/activation build,
and that the built Sequential runs a forward to the right action dimension. Tiny dims, CPU only.
"""

import pytest
import torch as th
from torch import nn

from src.model.custom.customqnetwork import CustomQNetwork

build = CustomQNetwork.create_mlp_custom


# --- happy path: plain Linear MLP -------------------------------------------


def test_linear_mlp_builds_and_forwards_to_action_dim():
    mods = build(6, 3, [8, 8], nn.ReLU, ["Linear", "activation_fn", "Linear", "activation_fn", "Linear"])
    out = nn.Sequential(*mods)(th.zeros(4, 6))
    assert tuple(out.shape) == (4, 3)


def test_linear_widths_follow_net_sizes():
    # net_sizes = [6, 8, 3]; the two Linear layers must be 6->8 then 8->3.
    mods = build(6, 3, [8], nn.ReLU, ["Linear", "activation_fn", "Linear"])
    linears = [m for m in mods if isinstance(m, nn.Linear)]
    assert (linears[0].in_features, linears[0].out_features) == (6, 8)
    assert (linears[1].in_features, linears[1].out_features) == (8, 3)


def test_activation_fn_instances_inserted():
    mods = build(4, 2, [4], nn.Tanh, ["Linear", "activation_fn", "Linear"])
    assert any(isinstance(m, nn.Tanh) for m in mods)


# --- validation guard -------------------------------------------------------


def test_wrong_layer_count_raises_valueerror():
    # net_arch=[8,8] -> net_sizes has 3 hops, needs 3 layer-producing blocks; recipe has only 2.
    with pytest.raises(ValueError, match="layer-producing block"):
        build(6, 3, [8, 8], nn.ReLU, ["Linear", "Linear"])


def test_too_many_layers_raises_valueerror():
    with pytest.raises(ValueError):
        build(6, 3, [8], nn.ReLU, ["Linear", "Linear", "Linear"])


def test_valid_layer_count_does_not_raise():
    # exactly len(net_arch)+1 = 2 layer-producing blocks.
    mods = build(6, 3, [8], nn.ReLU, ["Linear", "NoisyLinear"])
    assert len(mods) == 2


# --- empty-recipe fallback --------------------------------------------------


def test_empty_recipe_falls_back_to_standard_mlp():
    # custom_net_arch=[] -> ["Linear","activation_fn"]*len(net_arch) + ["Linear"].
    mods = build(6, 3, [8], nn.ReLU, [])
    kinds = [type(m).__name__ for m in mods]
    assert kinds == ["Linear", "ReLU", "Linear"]
    out = nn.Sequential(*mods)(th.zeros(2, 6))
    assert tuple(out.shape) == (2, 3)


def test_empty_recipe_two_hidden_layers():
    mods = build(6, 3, [8, 16], nn.ReLU, [])
    kinds = [type(m).__name__ for m in mods]
    assert kinds == ["Linear", "ReLU", "Linear", "ReLU", "Linear"]


# --- first-module recurrent shortcuts ---------------------------------------


def test_lstfull_shortcut_returns_lstm_then_linear():
    mods = build(6, 3, [8, 8], nn.ReLU, ["LSTFull"])
    kinds = [type(m).__name__ for m in mods]
    assert kinds == ["LSTMLocal", "Linear"]
    # output Linear maps the hidden width (net_arch[0]) to the action dim.
    assert mods[1].in_features == 8 and mods[1].out_features == 3


def test_lstfulln_shortcut_returns_lstm_then_linear():
    mods = build(6, 3, [8], nn.ReLU, ["LSTFullN"])
    assert [type(m).__name__ for m in mods] == ["LSTMLocal", "Linear"]


def test_grufull_shortcut_returns_gru_then_linear():
    mods = build(6, 3, [8], nn.ReLU, ["GRUFull"])
    assert [type(m).__name__ for m in mods] == ["GRULocal", "Linear"]


# --- NoisyLinear path -------------------------------------------------------


def test_noisylinear_recipe_builds_and_forwards():
    from src.model.custom.noisylinear import NoisyLinear

    mods = build(6, 3, [8], nn.ReLU, ["NoisyLinear", "activation_fn", "NoisyLinear"])
    assert sum(isinstance(m, NoisyLinear) for m in mods) == 2
    out = nn.Sequential(*mods)(th.zeros(4, 6))
    assert tuple(out.shape) == (4, 3)


# --- weight/spectral norm wrappers (smoke: build + forward) -----------------


@pytest.mark.parametrize(
    "first",
    ["weight_norm", "weight_norm2", "spectral_norm", "spectral_norm2", "DropConnectLinear"],
)
def test_norm_wrapped_linear_recipes_build_and_forward(first):
    mods = build(6, 3, [8], nn.ReLU, [first, "activation_fn", "Linear"])
    out = nn.Sequential(*mods)(th.zeros(4, 6))
    assert tuple(out.shape) == (4, 3)


# --- non-layer-producing blocks (idx must NOT advance) ----------------------


def test_batchnorm_and_layernorm_do_not_consume_a_width():
    # BatchNorm1d/LayerNorm operate at net_sizes[idx] without advancing idx, so two Linears still
    # span 6->8->3 with a normaliser sandwiched between them.
    mods = build(6, 3, [8], nn.ReLU, ["Linear", "BatchNorm1d", "activation_fn", "Linear"])
    linears = [m for m in mods if isinstance(m, nn.Linear)]
    assert (linears[0].in_features, linears[0].out_features) == (6, 8)
    assert (linears[1].in_features, linears[1].out_features) == (8, 3)
    assert any(isinstance(m, nn.BatchNorm1d) for m in mods)
    out = nn.Sequential(*mods)(th.zeros(4, 6))
    assert tuple(out.shape) == (4, 3)


def test_dropout_inserted_without_consuming_width():
    mods = build(4, 2, [4], nn.ReLU, ["Linear", "Dropout", "Linear"])
    assert any(isinstance(m, nn.Dropout) for m in mods)
    assert sum(isinstance(m, nn.Linear) for m in mods) == 2


def test_selfattention_recipe_builds_and_forwards():
    # SelfAttention preserves the (batch, width) shape so it fits between two Linears.
    mods = build(6, 3, [8], nn.ReLU, ["Linear", "activation_fn", "SelfAttention", "Linear"])
    out = nn.Sequential(*mods)(th.zeros(4, 6))
    assert tuple(out.shape) == (4, 3)


def test_unknown_module_name_is_silently_skipped():
    # An unrecognised name appends nothing and does not advance idx; with the right number of real
    # layer blocks the build still succeeds and the unknown token is a no-op.
    mods = build(6, 3, [8], nn.ReLU, ["Linear", "TotallyUnknownLayer", "Linear"])
    assert sum(isinstance(m, nn.Linear) for m in mods) == 2
    out = nn.Sequential(*mods)(th.zeros(2, 6))
    assert tuple(out.shape) == (2, 3)
