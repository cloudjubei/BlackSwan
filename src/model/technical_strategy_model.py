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
        # BUY (1) when the buy indicator crosses its threshold, else SELL (2) when the sell indicator does,
        # else HOLD (0). The indicator value at the current decision bar comes from the data provider's
        # look-ahead-safe named-feature accessor (NOT the flattened obs), so it stays correct across the
        # single/multi-fidelity providers and their stride mapping.
        step = env.current_step
        price = env.get_price(step)

        buy_indicator = env.data_provider.get_feature(step, self.technical_config.buy_indicator)
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

        sell_indicator = env.data_provider.get_feature(step, self.technical_config.sell_indicator)
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