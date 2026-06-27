from src.model.abstract_model import BaseStrategyModel
from src.conf.model_config import ModelConfig
from src.environment.abstract_env import AbstractEnv


class TimeStrategyModel(BaseStrategyModel):
    """A deterministic time-of-day baseline: open a long at a fixed UTC hour (`time_buy`) and close it at a
    fixed UTC hour (`time_sell`), every day. With time_buy < time_sell it's an intraday round-trip (e.g.
    long the US session 14->21); with time_buy > time_sell it holds OVERNIGHT (e.g. buy 21, sell 14 next
    day). A control to BEAT, and a probe for any time-of-day / overnight seasonality. Needs INTRADAY data
    (timeframe=1h) so the hour-of-day actually varies; at a daily step every bar shares one hour and it
    never trades."""

    def __init__(self, config: ModelConfig):
        super(TimeStrategyModel, self).__init__(config)
        self.time_config = config.model_time

    def get_id(self, config: ModelConfig):
        return f'{config.model_type}_{config.model_time.time_buy}_{config.model_time.time_sell}'.replace('.', '~').replace('|', ']')

    def get_action(self, env: AbstractEnv, obs):
        # The current bar's UTC hour (0-23). Buy at the buy hour (opens a long iff flat), sell at the sell
        # hour (closes iff long); the env no-ops a redundant action, so this is a clean clock strategy.
        hour = env.data_provider.get_timestamp(env.current_step).hour
        if hour == int(self.time_config.time_buy):
            return 1  # open long
        if hour == int(self.time_config.time_sell):
            return 2  # close long
        return 0  # hold
