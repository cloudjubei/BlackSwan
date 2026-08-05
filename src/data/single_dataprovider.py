from abc import abstractmethod
from typing import List

import numpy as np
import pandas as pd
from src.conf.data_config import DataConfig
from src.data.abstract_dataprovider import AbstractDataProvider

class SingleDataProvider(AbstractDataProvider):
    def __init__(self, config: DataConfig, paths: List[str]):
        super(SingleDataProvider, self).__init__(config)

        self.paths = paths

        df, prices, timestamps, buy_sells, rewards_buy_profitable, rewards_buy_drawdown = self.get_data(self.paths, self.config.type, config.timestamp, config.indicator, config.buyreward_percent, config.buyreward_maxwait)
        self.df = df
        self.prices = prices
        self.timestamps = timestamps
        self.signals_buy_sell = buy_sells
        self.signals_buy_profitable = rewards_buy_profitable
        self.signals_buy_drawdown = rewards_buy_drawdown
        self.buyreward_maxwait = config.buyreward_maxwait
        self.buyreward_percent = config.buyreward_percent

    def get_timesteps(self) -> int:
        return self.df.shape[0] - self.get_start_index() - 1

    def get_timestamp(self, step: int) -> pd.Timestamp:
        # `timestamps` are the bar CLOSE in ms + 1 (see process_df); subtract that 1 to recover the true
        # close, whose hour-of-day IS the bar's hour (an hourly 14:00-15:00 bar closes 14:59:59 -> hour 14).
        ms = int(self.timestamps[step + self.get_start_index()]) - 1
        return pd.to_datetime(ms, unit="ms", utc=True)

    def get_start_index(self):
        return self.config.lookback_window_size - 1

    def get_price(self, step: int) -> float:
        return self.prices[step + self.get_start_index()]

    def get_open(self, step: int) -> float:
        # The OPEN of decision bar `step`, aligned with get_price's close. self.prices come from the same
        # concatenated frame (process_df_simple reads price/price_open off the raw rows before any feature
        # work), so the raw price_open at row `step + start_index` is this bar's open. Lazily read once
        # (only when next_open fills are used) from the same cached raw build.
        if getattr(self, "_opens", None) is None:
            raw_df, *_ = self.get_raw_data(self.paths, self.config.timestamp)
            self._opens = raw_df["price_open"].to_numpy()
        i = min(max(step + self.get_start_index(), 0), len(self._opens) - 1)
        return float(self._opens[i])

    def get_values(self, step: int):
        offset = step + self.get_start_index()
        vs = self.df.loc[
            offset - (self.config.lookback_window_size - 1) : offset
        ].values

        if self.config.lookback_window_size <= 1:
            return np.array(vs.flatten())
        return np.array(vs)
    
    def get_feature(self, step: int, name: str) -> float:
        # Same decision-bar row get_values ends its window on (offset = step + start_index), read by
        # column name — clamped to the last row for the out-of-range edge step the env reads at `done`.
        offset = min(max(step + self.get_start_index(), 0), self.df.shape[0] - 1)
        return float(self.df.loc[offset, name])

    def get_signal_buy_sell(self, step: int) -> int:
        return self.signals_buy_sell[step + self.get_start_index()]
    
    def get_signal_buy_profitable(self, step: int) -> int:
        return self.signals_buy_profitable[step + self.get_start_index()]
    
    def get_signal_buy_drawdown(self, step: int) -> int:
        return self.signals_buy_drawdown[step + self.get_start_index()]
