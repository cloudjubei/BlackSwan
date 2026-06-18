import numpy as np
import pytest
import torch as th
from gymnasium import spaces

from src.model.custom.sequence_extractor import SequenceFeaturesExtractor


def _space(obs_dim):
    return spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)


@pytest.mark.parametrize("encoder", ["attn", "tcn"])
@pytest.mark.parametrize("lookback", [1, 32])
def test_forward_shape_is_features_dim(encoder, lookback):
    per_bar = 5
    obs_dim = lookback * per_bar
    ext = SequenceFeaturesExtractor(
        _space(obs_dim), lookback=lookback, encoder=encoder, features_dim=64
    )
    out = ext(th.zeros(4, obs_dim, dtype=th.float32))
    assert out.shape == (4, 64)
    assert th.isfinite(out).all()


def test_reshape_is_time_major_bar_grid():
    # The env flattens row-major TIME-MAJOR ([bar0 feats, bar1 feats, ...]); the extractor MUST
    # recover [B, lookback, per_bar] in that order or a TCN conv would see garbage (silent leakage).
    lookback, per_bar = 3, 2
    ext = SequenceFeaturesExtractor(_space(lookback * per_bar), lookback=lookback, encoder="attn")
    flat = th.arange(lookback * per_bar, dtype=th.float32).unsqueeze(0)  # [[0,1,2,3,4,5]]
    grid = ext._to_sequence(flat)  # [1, lookback, per_bar]
    assert grid.shape == (1, lookback, per_bar)
    assert grid[0].tolist() == [[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]]


def test_rejects_non_divisible_obs_dim():
    with pytest.raises(ValueError):
        SequenceFeaturesExtractor(_space(13), lookback=4, encoder="attn")


def test_rejects_unknown_encoder():
    with pytest.raises(ValueError):
        SequenceFeaturesExtractor(_space(10), lookback=2, encoder="nope")


class _Provider:
    """Minimal single-bar provider to stand up a real env for the build smoke (lookback 1)."""

    def __init__(self, features=4, steps=8):
        self._features = features
        self._steps = steps

    def get_timesteps(self):
        return self._steps

    def get_lookback_window(self):
        return 1

    def get_price(self, step):
        return 100.0

    def get_values(self, step):
        return np.zeros(self._features, dtype=np.float32)

    def get_signal_buy_sell(self, step):
        return 0

    def get_signal_buy_profitable(self, step):
        return 0

    def get_signal_buy_drawdown(self, step):
        return 0


@pytest.mark.parametrize("model_name", ["attn-ppo", "tcn-ppo"])
def test_create_model_builds_ppo_with_sequence_extractor(model_name):
    from src.conf.env_config import EnvConfig
    from src.environment.trade_all_crypto_env import TradeAllCryptoEnv
    from src.model.model_factory import create_model
    from trainer import config_builder

    env = TradeAllCryptoEnv(
        EnvConfig(
            type="trade_all",
            initial_balance=100000,
            transaction_fee=0.0,
            observations_contain=[],
            take_profit=None,
            stop_loss=None,
        ),
        _Provider(),
        "cpu",
    )
    config = config_builder.build_model_config({"model_name": model_name, "net_arch": "64,32"})
    model = create_model(config, env, "cpu")
    extractor = model.rl_model.policy.features_extractor
    assert isinstance(extractor, SequenceFeaturesExtractor)
    assert extractor.encoder == ("attn" if model_name == "attn-ppo" else "tcn")
