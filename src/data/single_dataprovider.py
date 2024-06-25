from abc import abstractmethod
from typing import List

import numpy as np
from src.conf.data_config import DataConfig
from src.data.abstract_dataprovider import AbstractDataProvider

class SingleDataProvider(AbstractDataProvider):
    def __init__(self, config: DataConfig, paths: List[str]):
        super(SingleDataProvider, self).__init__(config)

        self.paths = paths

        df, prices, timestamps, buy_sells, rewards_hold, rewards_buy = self.get_data(self.paths, self.config.type, config.timestamp, config.indicator, config.percent_buysell)
        self.df = df
        self.prices = prices
        self.buy_sell_signals = buy_sells
        self.hold_signals = rewards_hold
        self.buy_signals = rewards_buy

    def get_timesteps(self) -> int:
        return self.df.shape[0] - self.get_start_index() - 1
    
    def get_start_index(self):
        return self.config.lookback_window_size - 1

    def get_price(self, step: int) -> float:
        return self.prices[step + self.get_start_index()]
    
    def get_values(self, step: int):
        offset = step + self.get_start_index()
        vs = self.df.loc[
            offset - (self.config.lookback_window_size - 1) : offset
        ].values

        if self.config.lookback_window_size <= 1:
            return np.array(vs.flatten())
        return np.array(vs)
    
    def get_buy_sell_signal(self, step: int) -> int:
        return self.buy_sell_signals[step + self.get_start_index()]
    
    def get_hold_signal(self, step: int) -> int:
        return self.hold_signals[step + self.get_start_index()]
    
    def get_buy_signal(self, step: int) -> int:
        return self.buy_signals[step + self.get_start_index()]
