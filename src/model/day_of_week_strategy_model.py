from src.model.abstract_model import BaseStrategyModel
from src.conf.model_config import ModelConfig
from src.environment.abstract_env import AbstractEnv


class DayOfWeekStrategyModel(BaseStrategyModel):
    """A deterministic day-of-week baseline: open a long at the start of weekday ``day_buy`` (0=Mon..6=Sun)
    and close at weekday ``day_sell``, every week. day_buy < day_sell is an intra-week hold (e.g. long
    Mon->Fri); day_buy > day_sell holds across the WEEKEND (e.g. buy Fri 4, sell Mon 0 — crypto trades the
    weekend). A control to BEAT and a probe for weekly seasonality (the Monday / day-of-week effect). Runs at
    a DAILY step (timeframe=1d) so the weekday is unambiguous and each bar is exactly one day."""

    def __init__(self, config: ModelConfig):
        super(DayOfWeekStrategyModel, self).__init__(config)
        self.day_config = config.model_day

    def get_id(self, config: ModelConfig):
        return f'{config.model_type}_{config.model_day.day_buy}_{config.model_day.day_sell}'.replace('.', '~').replace('|', ']')

    def get_action(self, env: AbstractEnv, obs):
        # The current bar's weekday (0=Mon..6=Sun). Buy on day_buy (opens a long iff flat), sell on day_sell
        # (closes iff long); the env no-ops a redundant action, so this is a clean weekly clock.
        weekday = env.data_provider.get_timestamp(env.current_step).weekday()
        if weekday == int(self.day_config.day_buy):
            return 1  # open long
        if weekday == int(self.day_config.day_sell):
            return 2  # close long
        return 0  # hold
