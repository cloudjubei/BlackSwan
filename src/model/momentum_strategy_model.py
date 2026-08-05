from src.model.abstract_model import BaseStrategyModel
from src.conf.model_config import ModelConfig
from src.environment.abstract_env import AbstractEnv

class MomentumStrategyModel(BaseStrategyModel):
    """Time-series-momentum baseline (Moskowitz/Ooi/Pedersen): be long when the trailing
    `lookback_periods`-bar return is positive, flat otherwise. Single-asset BTC spot, long/flat
    (shorting is a future extension). Deterministic; a non-RL control to compare every run against."""

    def __init__(self, config: ModelConfig):
        super(MomentumStrategyModel, self).__init__(config)

        self.momentum_config = config.model_momentum

    def get_id(self, config: ModelConfig):
        return f'{config.model_type}_{config.model_momentum.lookback_periods}'.replace('.', '~').replace('|', ']')

    def get_action(self, env: AbstractEnv, obs):
        lookback_periods = self.momentum_config.lookback_periods
        step = env.current_step

        if lookback_periods <= 0 or step < lookback_periods:
            return 0  # not enough history yet -> HOLD

        price_now = env.get_price(step)
        price_then = env.get_price(step - lookback_periods)

        if price_then <= 0 or price_now <= 0:
            return 0  # guard against missing/garbage reference price

        trailing_return = price_now / price_then - 1.0

        if trailing_return > 0:
            return 1  # positive momentum -> be long
        # non-positive momentum -> downtrend. Long/short envs SHORT it; long-only be flat.
        allow_shorting = getattr(getattr(env, "env_config", None), "allow_shorting", False)
        return 3 if allow_shorting else 2
