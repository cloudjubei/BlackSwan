from abc import abstractmethod
from typing import List

import numpy as np
from src.conf.data_config import DataConfig
from src.data.abstract_dataprovider import AbstractDataProvider

# there needs to be three ways of handling this:
# 1:
# 1m [16:00 - 16:01 10.05]          ... [16:01 - 16:02 10.05]
# 1h [15:01 - 16:01 10.05]          ... [15:02 - 16:02 10.05]
# 1d [16:01 - 16:01 09.05-10.05]    ... [16:02 - 16:02 09.05-10.05]
# 2:
# 1m [16:00 - 16:01 10.05]          ... [16:01 - 16:02 10.05]
# 1h [16:00 - 16:01 10.05]          ... [16:00 - 16:02 10.05]
# 1d [00:00 - 16:01 10.05]          ... [00:00 - 16.02 10.05]
# 3:
# 1m [15:59 - 16:00 10.05][16:00 - 16:01 10.05]          ... [16:01 - 16:02 10.05]...[00:00 - 00:01 11.05]
# 1h [14:59 - 15:00 10.05][15:59 - 16:00 10.05]          ... [15:59 - 16:00 10.05]...[23:00 - 23:59 10.05]
# 1d [00:00 - 23:59 09.05][00:00 - 23:59 09.05]          ... [00:00 - 23:59 09.05]...[00:00 - 23:59 10.05]

# with lookback of 3, #1 is the same, but #2:
# 1m [15:59 - 16:00 10.05][16:00 - 16:01 10.05][16:01 - 16:02 10.05]
# 1h [14:00 - 14:59 10.05][15:00 - 15:59 10.05][16:00 - 16:02 10.05]
# 1d [00:00 - 23:59 08.05][00:00 - 23:59 09.05][00:00 - 16.02 10.05]

# and #3
# 1m [15:59 - 16:00 10.05][16:00 - 16:01 10.05][16:01 - 16:02 10.05]
# 1h [13:00 - 13:59 10.05][14:00 - 14:59 10.05][15:00 - 15:59 10.05]
# 1d [00:00 - 23:59 07.05][00:00 - 23:59 08.05][00:00 - 23:59 09.05]

# it seems #1 is better without lookback, but with lookback #2 seems good, #3 sees more of the past

class MultiTimelineDataProvider(AbstractDataProvider):
    def __init__(self, config: DataConfig, layer_paths: List[List[str]]):
        super(MultiTimelineDataProvider, self).__init__(config)

        self.is_resolved_from_fidelity = len(layer_paths) == 1
        self.paths = layer_paths

        self.dfs = []
        self.timestamps = []
        self.prices = []
        # self.XPRICES = []
        self.buy_sell_signals = []
        for i in range(len(self.paths)):
            path = self.paths[i]
            df, prices, timestamps, buy_sells, rewards_hold, rewards_buy = self.get_data(path, config.type, config.timestamp, config.indicator, config.percent_buysell)
            self.dfs.append(df)
            self.timestamps.append(timestamps)
            # self.XPRICES.append(prices)
            if i == 0:
                self.prices = prices
                self.buy_sell_signals = buy_sells
                self.hold_signals = rewards_hold
                self.buy_signals = rewards_buy

        self.index_mappings = []
        for j in range(len(self.timestamps) - 1):
            self.index_mappings.append([])

        base_timestamps = self.timestamps[0]
        for i in range(len(base_timestamps)):
            base_timestamp = base_timestamps[i]

            for j in range(len(self.timestamps) - 1):
                target_timestamps = self.timestamps[j+1]
                mappings = self.index_mappings[j]
                last_timestamp_index = 0 if len(mappings) <= 0 else mappings[-1]
                last_timestamp_index = (last_timestamp_index if last_timestamp_index >= 0 else 0)

                for k in range(len(target_timestamps)):
                    target_timestamp = target_timestamps[k + last_timestamp_index]
                    if base_timestamp < target_timestamp:
                        mappings.append(k + last_timestamp_index - 1)
                        break
                    elif base_timestamp == target_timestamp:
                        mappings.append(k + last_timestamp_index)
                        break


        self.starting_index = self.config.lookback_window_size - 1

        last_mappings = self.index_mappings[-1]
        for i in range(len(last_mappings)):
            index = last_mappings[i]

            if index >= self.starting_index:
                self.starting_index = i
                break

        # TODO: prepare the 1+ vals for each column
        if self.is_resolved_from_fidelity:
            self.multipliers = []
            for i in range(len(self.config.layers)):
                if self.config.fidelity == "1m":
                    if self.config.layers[i] == "1m":
                        self.multipliers.append(1)
                    elif self.config.layers[i] == "1h":
                        self.multipliers.append(60)
                    elif self.config.layers[i] == "1d":
                        self.multipliers.append(60*24)
                    elif self.config.layers[i] == "1w":
                        self.multipliers.append(60*24*7)
                elif self.config.fidelity == "1h":
                    if self.config.layers[i] == "1h":
                        self.multipliers.append(1)
                    elif self.config.layers[i] == "1d":
                        self.multipliers.append(24)
                    elif self.config.layers[i] == "1w":
                        self.multipliers.append(24*7)

    def get_timesteps(self) -> int:
        return self.dfs[0].shape[0] - self.get_start_index() - 1
    
    def get_start_index(self):
        return self.starting_index

    def get_price(self, step: int) -> float:
        return self.prices[step + self.get_start_index()]

    def get_values(self, step: int):
        out = []
        if self.is_resolved_from_fidelity:
            return out
        
        offset = step + self.get_start_index()
        
        v = self.dfs[0].loc[
            offset - (self.config.lookback_window_size - 1) : offset
        ].values
        if self.config.lookback_window_size <= 1:
            v = v.flatten()
        out.append(v)
        
        # print(f'len(self.index_mappings)= {len(self.index_mappings)} offset= {offset}')
        for i in range(len(self.index_mappings)):
            index = self.index_mappings[i][offset]
            v = self.dfs[i+1].loc[
                index - (self.config.lookback_window_size - 1) : index
            ].values

            if self.config.lookback_window_size <= 1:
                v = v.flatten()

            out.append(v)
        return np.array(out)

        # out = []
        # for i in len(self.config.layers):
        #     layer = self.config.layers[i]
        #     if layer == self.config.fidelity:
        #         v = self.dfs[i].loc[
        #             step : step + self.config.lookback_window_size - 1
        #         ].values
        #         out.append(v)
        #     else:
        #         v = []
        #         for j in len(self.multipliers[i]):
        #             TAKE FIRST VALUE (self.dfs[i].loc[step]) and multiply by the rolling window products
        #             offset = - self.multipliers[i] + j + 1
        #             values = self.dfs[i].loc[
        #                 step + offset : step + self.config.lookback_window_size - 1 + offset
        #             ].values
        #             price_window = dfs[i]['prices'] + 1
        #             products = price_window.rolling(window=self.multipliers[i]-1).apply(np.prod, raw=True)

        #         v = self.dfs[i].loc[
        #             step - self.multipliers[i] : step + self.config.lookback_window_size - 1
        #         ].values


        # return out
    
    def get_buy_sell_signal(self, step: int) -> int:
        return self.buy_sell_signals[step + self.get_start_index()]
    
    def get_hold_signal(self, step: int) -> int:
        return self.hold_signals[step + self.get_start_index()]
    
    def get_buy_signal(self, step: int) -> int:
        return self.buy_signals[step + self.get_start_index()]