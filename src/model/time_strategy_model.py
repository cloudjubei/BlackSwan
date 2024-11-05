from src.model.abstract_model import BaseStrategyModel
from src.conf.model_config import ModelConfig
from src.environment.abstract_env import AbstractEnv
from datetime import datetime, timedelta

class TimeStrategyModel(BaseStrategyModel):
    def __init__(self, config: ModelConfig):
        super(TimeStrategyModel, self).__init__(config)

        self.time_config = config.model_time

    def get_id(self, config: ModelConfig):
        return f'{config.model_type}_{config.model_time.time_buy}_{config.model_time.time_sell}'.replace('.', '~').replace('|', ']')

    def get_action(self, env: AbstractEnv, obs):

        # net_worth = obs[-1]
        
        lookback_window = env.env_config.lookback_window_size
        item_size = int((len(obs) - 1) / lookback_window)
        last_item_start = (lookback_window - 1) * item_size
        last_item = obs[last_item_start : last_item_start + item_size]
        
        # TODO: this no longer works for non-flat datas
        # timestamp_open = last_item[env.df.columns.get_loc("timestamp")]
        timestamp_close = last_item[env.df.columns.get_loc("timestamp_close")]

        date_close = datetime.utcfromtimestamp(timestamp_close)  
        date_close_offset = date_close + timedelta(seconds=1)

        buy_time = self.time_config.time_buy * 100
        sell_time = self.time_config.time_sell * 100

        time_close = date_close.hour * 10000 + date_close.minute * 100 + date_close.second
        time_close_offset = date_close_offset.hour * 10000 + date_close_offset.minute * 100 + date_close_offset.second

        if buy_time == time_close or buy_time == time_close_offset:
            return 1 #BUY
        if sell_time == time_close or sell_time == time_close_offset:
            return 2 #SELL
        return 0

