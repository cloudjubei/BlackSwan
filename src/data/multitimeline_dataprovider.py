from abc import abstractmethod
from typing import List

import numpy as np
import pandas as pd
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
# 1m [15:57 - 15:58 10.05][15:58 - 15:59 10.05]          ... [16:01 - 16:02 10.05]...[00:00 - 00:01 11.05]
# 1h [14:00 - 14:59 10.05][15:00 - 15:59 10.05]          ... [15:00 - 15:59 10.05]...[23:00 - 23:59 10.05]
# 1d [00:00 - 23:59 09.05][00:00 - 23:59 09.05]          ... [00:00 - 23:59 09.05]...[00:00 - 23:59 10.05]

# with lookback of 3, #1 is the same, but #2:
# 1m [15:59 - 16:00 10.05][16:00 - 16:01 10.05][16:01 - 16:02 10.05]
# 1h [14:00 - 14:59 10.05][15:00 - 15:59 10.05][16:00 - 16:02 10.05]
# 1d [00:00 - 23:59 08.05][00:00 - 23:59 09.05][00:00 - 16:02 10.05]

# and #3
# 1m [15:59 - 16:00 10.05][16:00 - 16:01 10.05][16:01 - 16:02 10.05]
# 1h [13:00 - 13:59 10.05][14:00 - 14:59 10.05][15:00 - 15:59 10.05]
# 1d [00:00 - 23:59 07.05][00:00 - 23:59 08.05][00:00 - 23:59 09.05]

# it seems #1 is better without lookback, but with lookback #2 seems good, #3 sees more of the past

class MultiTimelineDataProvider(AbstractDataProvider):
    def __init__(self, config: DataConfig, layer_paths: List[List[str]], fidelity_input: str, fidelity_run: str, layers: List[str]):
        super(MultiTimelineDataProvider, self).__init__(config)

        self.is_resolved_from_fidelity = len(layer_paths) == 1
        self.paths = layer_paths
        self.fidelity_input = fidelity_input
        self.fidelity_run = fidelity_run
        self.layers = layers
        self.steps = 0

        self.dfs = []
        self.timestamps = []
        self.prices = []
        self.signals_buy_sell = []
        self.signals_buy_profitable = []
        self.signals_buy_drawdown = []
        self.starting_index = self.config.lookback_window_size - 1
        self.divider_run = 1

        if self.is_resolved_from_fidelity:
            print('RESOLVED FROM FIDELITY')
            path = layer_paths[0]

            self.fidelity_dfs = []
            self.fidelity_raw_dfs = []
            raw_df, df, prices, timestamps = self.get_raw_data(path, "day_of_week")
            self.dfs = [df]
            self.timestamps = timestamps
            self.prices = prices
            self.raw_df = raw_df

            if fidelity_input == "1m":
                if fidelity_run == "5m":
                    self.divider_run = 5
                elif fidelity_run == "10m":
                    self.divider_run = 10
                elif fidelity_run == "15m":
                    self.divider_run = 15
                elif fidelity_run == "30m":
                    self.divider_run = 30
                elif fidelity_run == "1h":
                    self.divider_run = 60
                # elif fidelity_run == "1d":
                #     self.divider_run = 60*24
                # elif fidelity_run == "1w":
                #     self.divider_run = 60*24*7
            
            self.fidelity_offset = 0
            self.multipliers = []
            for i in range(len(layers)):
                multiplier = 1
                if fidelity_input == "1m":
                    if layers[i] == "5m":
                        multiplier = 5
                    elif layers[i] == "10m":
                        multiplier = 10
                    elif layers[i] == "15m":
                        multiplier = 15
                    elif layers[i] == "30m":
                        multiplier = 30
                    elif layers[i] == "1h":
                        multiplier = 60
                    elif layers[i] == "1d":
                        multiplier = 60*24
                    elif layers[i] == "1w":
                        multiplier = 60*24*7
                elif fidelity_input == "1h":
                    if layers[i] == "1d":
                        multiplier = 24
                    elif layers[i] == "1w":
                        multiplier = 24*7

                self.fidelity_offset = multiplier * self.config.lookback_window_size
                self.multipliers.append(multiplier)

            self.starting_index = self.fidelity_offset - 1
            # self.steps = int((df.shape[0] - self.starting_index) / self.divider_run)

            for i in range(len(layers)):
                layer = layers[i]
                if layer == fidelity_input:
                    self.fidelity_dfs.append(self.dfs) # maybe should be self.fidelity_dfs.append(self.dfs)
                else:
                    multiplier_run = int(self.multipliers[i]/self.divider_run)
                    fidelity_offset = self.fidelity_offset - (self.multipliers[i] * self.config.lookback_window_size)
                    dfs, raw_dfs = self.process_fidelity(raw_df, layer, fidelity_offset, self.multipliers[i], fidelity_run, multiplier_run, self.divider_run, "day_of_week")
                    self.fidelity_dfs.append(dfs)
                    self.fidelity_raw_dfs.append(raw_dfs)

            self.steps = int((df.shape[0] - self.fidelity_offset)/self.divider_run)
        else:         
            for i in range(len(self.paths)):
                path = self.paths[i]
                df, prices, timestamps, buy_sells, rewards_buy_profitable, rewards_buy_drawdown = self.get_data(path, config.type, config.timestamp, config.indicator, config.buyreward_percent, config.buyreward_maxwait)
                self.dfs.append(df)
                self.timestamps.append(timestamps)
                if i == 0:
                    self.prices = prices
                    self.signals_buy_sell = buy_sells
                    self.signals_buy_profitable = rewards_buy_profitable
                    self.signals_buy_drawdown = rewards_buy_drawdown

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


            last_mappings = self.index_mappings[-1]
            for i in range(len(last_mappings)):
                index = last_mappings[i]

                if index >= self.starting_index:
                    self.starting_index = i
                    break

            self.steps = self.dfs[0].shape[0] - self.starting_index - 1

    def get_raw_df_for_plotting(self):
        print(self.raw_df.head())
        return self.raw_df.iloc[self.get_start_index()::self.divider_run]
    
    def get_timesteps(self) -> int:
        return self.steps
    
    def get_start_index(self):
        return self.starting_index

    def get_price(self, step: int) -> float:
        return self.prices[step*self.divider_run + self.get_start_index()]

    def get_values(self, step: int):

        out = []
        v = None

        if self.is_resolved_from_fidelity:
            for i in range(len(self.fidelity_dfs)):
                offset = step*self.divider_run
                mapping = self.get_current_mapping(self.raw_df, offset, self.layers[i], self.fidelity_run)
                df = self.fidelity_dfs[i][mapping]

                index = int((offset - mapping)/self.multipliers[i])
                v = df.loc[
                    index : index + (self.config.lookback_window_size-1)
                ].values
                # raw_test = self.fidelity_raw_dfs[i][mapping].loc[
                #     index : index + (self.config.lookback_window_size-1)
                # ]
                # if raw_test['timestamp_close'].values[len(raw_test)-1] != v_test['timestamp_close'].values[len(v_test)-1]:
                #     print('v_test:')
                #     print(v_test.head())
                #     print(v_test.tail())
                #     print('raw_test.head + tail step: ', step, ' i: ', i, ' mapping: ', mapping)
                # print(raw_test.head())
                # print(raw_test.tail())
                # if i == 1 and step > 27:
                #     raise ValueError('TEST')

                if self.config.lookback_window_size <= 1:
                    v = v.flatten()
                    if len(out) == 0:
                        out = v
                    else:
                        out += v
                else:
                    out.append(v)
        else:
            offset = step + self.get_start_index()
            v = self.dfs[0].loc[
                offset - (self.config.lookback_window_size - 1) : offset
            ].values
            # v_test = self.test_raw_df.loc[
            #     offset - (self.config.lookback_window_size - 1) : offset
            # ]

            if self.config.lookback_window_size <= 1:
                v = v.flatten()
                out = v
            else:
                out.append(v)

            # print(f'len(self.index_mappings)= {len(self.index_mappings)} offset= {offset}')
            for i in range(len(self.index_mappings)):
                index = self.index_mappings[i][offset]
                v = self.dfs[i+1].loc[
                    index - (self.config.lookback_window_size - 1) : index
                ].values

                if self.config.lookback_window_size <= 1:
                    v = v.flatten()
                    out += v
                else:
                    out.append(v)

        try:
            return np.array(out)
        except:
            print('GOT ERROR for np.array(out) self.config.lookback_window_size: ', self.config.lookback_window_size, ' step: ', step, ' start_index: ', self.get_start_index())
            print('v:')
            print(v)
            print('out: ')
            print(out)
        return np.array(out)
    
    def get_signal_buy_sell(self, step: int) -> int:
        return self.signals_buy_sell[step + self.get_start_index()]
    
    def get_signal_buy_profitable(self, step: int) -> int:
        return self.signals_buy_profitable[step + self.get_start_index()]
    
    def get_signal_buy_drawdown(self, step: int) -> int:
        return self.signals_buy_drawdown[step + self.get_start_index()]