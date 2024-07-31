from abc import abstractmethod
from typing import List

import numpy as np
from src.conf.data_config import DataConfig
from src.data.abstract_dataprovider import AbstractDataProvider

class SingleDataProvider(AbstractDataProvider):
    def __init__(self, config: DataConfig, paths: List[str]):
        super(SingleDataProvider, self).__init__(config)

        self.paths = paths

        df, prices, timestamps, buy_sells, rewards_buy_profitable, rewards_buy_drawdown = self.get_data(self.paths, self.config.type, config.timestamp, config.indicator, config.buyreward_percent, config.buyreward_maxwait)
        self.df = df
        self.prices = prices
        self.signals_buy_sell = buy_sells
        self.signals_buy_profitable = rewards_buy_profitable
        self.signals_buy_drawdown = rewards_buy_drawdown

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
    
    def get_signal_buy_sell(self, step: int) -> int:
        return self.signals_buy_sell[step + self.get_start_index()]
    
    def get_signal_buy_profitable(self, step: int) -> int:
        return self.signals_buy_profitable[step + self.get_start_index()]
    
    def get_signal_buy_drawdown(self, step: int) -> int:
        return self.signals_buy_drawdown[step + self.get_start_index()]
