from src.model.abstract_model import BaseStrategyModel
from src.conf.model_config import ModelConfig
from src.environment.abstract_env import AbstractEnv


class MaCrossoverStrategyModel(BaseStrategyModel):
    """Variable-length moving-average crossover baseline (Brock/Lakonishok/LeBaron 1992; Grobys et al.
    2020). Compare a SHORT moving average to a LONG moving average of the trailing price window: short
    above long by more than an optional band -> be long (BUY=1); short below long by more than the band ->
    be flat (EXIT=2); inside the band -> HOLD=0 (keep the current position — the band is the classic
    whipsaw filter). short_window=1 makes the short MA the current price (the 1/N "variable MA" rule).
    Windows are in BARS, so a daily timeframe reproduces the papers' daily rules (e.g. 1/50, 1/150, 1/200,
    1/20). Single-asset BTC spot, long/flat; deterministic, a non-RL control to compare every run against.
    """

    def __init__(self, config: ModelConfig):
        super(MaCrossoverStrategyModel, self).__init__(config)

        self.ma_config = config.model_ma_crossover

    def get_id(self, config: ModelConfig):
        return f'{config.model_type}_{config.model_ma_crossover.short_window}_{config.model_ma_crossover.long_window}_{config.model_ma_crossover.band}'.replace('.', '~').replace('|', ']')

    def _mean(self, env: AbstractEnv, step: int, window: int) -> float:
        total = 0.0
        for i in range(window):
            total += env.get_price(step - i)
        return total / window

    def get_action(self, env: AbstractEnv, obs):
        short_window = int(self.ma_config.short_window)
        long_window = int(self.ma_config.long_window)
        band = float(self.ma_config.band)
        step = env.current_step

        # Need a full long window ending at the current bar (bars 0..step -> step+1 available).
        if short_window <= 0 or long_window <= 0 or step + 1 < long_window:
            return 0  # not enough history yet -> HOLD

        short_ma = self._mean(env, step, short_window)
        long_ma = self._mean(env, step, long_window)
        if long_ma <= 0:
            return 0  # guard against a zero/garbage window

        if short_ma > long_ma * (1.0 + band):
            return 1  # short above long -> be long
        if short_ma < long_ma * (1.0 - band):
            # short below long -> downtrend. Long/short envs SHORT it (profit the downtrend); long-only be flat.
            allow_shorting = getattr(getattr(env, "env_config", None), "allow_shorting", False)
            return 3 if allow_shorting else 2
        return 0      # inside the band -> keep the current position
