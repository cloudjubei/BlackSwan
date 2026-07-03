from src.model.abstract_model import BaseStrategyModel
from src.conf.model_config import ModelConfig
from src.environment.abstract_env import AbstractEnv


class BreakoutStrategyModel(BaseStrategyModel):
    """Trading-range breakout baseline (Brock/Lakonishok/LeBaron 1992). Compare the current price to the
    local resistance (max) and support (min) of the PRIOR `window` bars (excluding the current bar): a
    break above resistance by more than an optional band -> be long (BUY=1); a break below support by more
    than the band -> be flat (EXIT=2); inside the range -> HOLD=0 (keep the current position). The band
    filters marginal penetrations, as in the paper. `window` is in BARS, so a daily timeframe reproduces
    the paper's daily rules (50/150/200). Single-asset BTC spot, long/flat; deterministic, a non-RL
    control to compare every run against.
    """

    def __init__(self, config: ModelConfig):
        super(BreakoutStrategyModel, self).__init__(config)

        self.breakout_config = config.model_breakout

    def get_id(self, config: ModelConfig):
        return f'{config.model_type}_{config.model_breakout.window}_{config.model_breakout.band}'.replace('.', '~').replace('|', ']')

    def get_action(self, env: AbstractEnv, obs):
        window = int(self.breakout_config.window)
        band = float(self.breakout_config.band)
        step = env.current_step

        # Need `window` PRIOR bars (step-window .. step-1) to form the range.
        if window <= 0 or step < window:
            return 0  # not enough history yet -> HOLD

        upper = None
        lower = None
        for i in range(1, window + 1):
            p = env.get_price(step - i)
            if upper is None or p > upper:
                upper = p
            if lower is None or p < lower:
                lower = p

        price_now = env.get_price(step)
        if upper is not None and upper > 0 and price_now > upper * (1.0 + band):
            return 1  # broke above resistance -> be long
        if lower is not None and lower > 0 and price_now < lower * (1.0 - band):
            return 2  # broke below support -> be flat
        return 0      # inside the range -> keep the current position
