from src.model.abstract_model import BaseStrategyModel
from src.conf.model_config import ModelConfig
from src.environment.abstract_env import AbstractEnv
from datetime import datetime

class TechnicalStrategyModel(BaseStrategyModel):
    def __init__(self, config: ModelConfig):
        super(TechnicalStrategyModel, self).__init__(config)

        self.technical_config = config.model_technical

    def get_id(self, config: ModelConfig):
        return f'{config.model_technical.buy_indicator}_{config.model_technical.buy_amount_threshold}_{config.model_technical.sell_indicator if config.model_technical.buy_indicator != config.model_technical.sell_indicator else ""}_{config.model_technical.sell_amount_threshold}'.replace('.', '~').replace('|', ']')

    def get_action(self, env: AbstractEnv, obs):
        # net_worth = obs[-1]
        
        lookback_window = env.env_config.lookback_window_size
        item_size = int((len(obs) - 1) / lookback_window)
        last_item_start = (lookback_window - 1) * item_size
        last_item = obs[last_item_start : last_item_start + item_size]
        
        price = env.get_price(env.current_step)

        buy_indicator = last_item[env.df.columns.get_loc(self.technical_config.buy_indicator)]
        if self.technical_config.buy_amount_is_multiplier:
            buy_indicator = buy_indicator * self.technical_config.buy_amount_threshold

        if self.technical_config.buy_is_price_check:
            if self.technical_config.buy_is_down_check:
                if price <= buy_indicator:
                    return 1 #BUY
            else:
                if price >= buy_indicator:
                    return 1 #BUY
        else:
            if self.technical_config.buy_is_down_check:
                if buy_indicator <= self.technical_config.buy_amount_threshold:
                    return 1 #BUY
            else:
                if buy_indicator >= self.technical_config.buy_amount_threshold:
                    return 1 #BUY

        sell_indicator = last_item[env.df.columns.get_loc(self.technical_config.sell_indicator)]
        if self.technical_config.sell_amount_is_multiplier:
            sell_indicator = sell_indicator * self.technical_config.sell_amount_threshold

        if self.technical_config.sell_is_price_check:
            if self.technical_config.sell_is_up_check:
                if price >= sell_indicator:
                    return 2 #SELL
            else:
                if price <= sell_indicator:
                    return 2 #SELL
        else:
            if self.technical_config.sell_is_up_check:
                if sell_indicator >= self.technical_config.sell_amount_threshold:
                    return 2 #SELL
            else:
                if sell_indicator <= self.technical_config.sell_amount_threshold:
                    return 2 #SELL

        return 0