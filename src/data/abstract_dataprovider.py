from abc import ABC, abstractmethod
from typing import List
from src.conf.data_config import DataConfig
import pandas as pd
import numpy as np
import calendar

from src.data.data_utils import plot_indicator
from src.util.plot import plot_timeseries

class AbstractDataProvider(ABC):
    def __init__(self, config: DataConfig):
        super().__init__()

        self.config = config
        self.id = self.get_id(config)
   
    def get_id(self, config: DataConfig):
        return f'{config.id}_{config.type}_{config.timestamp}_{config.indicator}_{config.fidelity}_{config.layers}_{config.lookback_window_size}_{1 if config.flat_lookback else 0}_{1 if config.flat_layers else 0}_{config.percent_buysell}'
    
    def is_multilayered(self) -> bool:
        return len(self.config.layers) > 1
    
    @abstractmethod
    def get_timesteps(self) -> int:
        pass
    
    @abstractmethod
    def get_price(self, step: int) -> float:
        pass

    @abstractmethod
    def get_buy_sell_signal(self, step: int) -> int:
        pass
    @abstractmethod
    def get_hold_signal(self, step: int) -> int:
        pass
    @abstractmethod
    def get_buy_signal(self, step: int) -> int:
        pass
    
    @abstractmethod
    def get_values(self, step: int):
        pass
    
    def get_lookback_window(self) -> int:
        return self.config.lookback_window_size

    def get_start_index(self):
        return 0

    def get_normalized_indicators(self):
        return ["lwti8", "lwti13", "lwti30", 
                "kandleCount2_30_green", "kandleCount2_30_red", "kandleCount2_30_high", "kandleCount2_30_mid", "kandleCount2_30_low", 
                "kandleCount3_30_green", "kandleCount3_30_red", "kandleCount3_30_high", "kandleCount3_30_mid", "kandleCount3_30_low", 
                "kandleCount5_90_green", "kandleCount5_90_red", "kandleCount5_90_high", "kandleCount5_90_mid", "kandleCount5_90_low", 
                "kandleCount2_10_green", "kandleCount2_10_red", "kandleCount2_10_high", "kandleCount2_10_mid", "kandleCount2_10_low", 
                "kandleCount3_90_green", "kandleCount3_90_red", "kandleCount3_90_high", "kandleCount3_90_mid", "kandleCount3_90_low", 
                "kandleCount5_300_green", "kandleCount5_300_red", "kandleCount5_300_high", "kandleCount5_300_mid", "kandleCount5_300_low", 
                "pump_dump10", "pump_dump90", "pump_dump300", 
                "candle_color", "candle_height10", "candle_height30", "candle_height90", "candle_height300", 
                "lwStrategyUp", "lwStrategyDown", "williams14","williams30"]
    
    def get_buy_sell(self, df):
        prices = df['price'].values

        count = 0
        buy_sell = []
        action = 0
        for i in range(len(prices) - 1):
            current = prices[i]
            next = prices[i+1]
            if current < next:
                if action == 1:
                    buy_sell.append(0)
                else:
                    action = 1
                    count += 1
                    buy_sell.append(count)
            else:
                if action == -1:
                    buy_sell.append(0)
                else:
                    action = -1
                    buy_sell.append(-count)

        buy_sell.append(0)
        # df['buy_sell'] = buy_sell
        # df['buy_price'] = df.apply(lambda x: x['price'] if x['buy_sell'] > 0 else None , axis=1)
        # df['sell_price'] = df.apply(lambda x: x['price'] if x['buy_sell'] < 0 else None , axis=1)
        # result_df = result_df.drop(columns=['buy_sell', 'buy_price', 'sell_price'])

        return buy_sell

    # sell is percent diff + perfect sell
    # buy is time to go profitable
    # hold is -1 or +1
    def get_rewards_action(self, df, percent_to_pass = 0.004, max_buy_wait = 10):
        prices = df['price'].values

        rewards_hold = []
        rewards_buy = []
        
        total_prices = len(prices) - 1
        for i in range(total_prices):
            current = prices[i]
            next = prices[i+1]
            
            rewards_hold.append(-1 if current > next else 1)

            profit_count = 0
            for j in range(min(total_prices - i, max_buy_wait)):
                next = prices[i+1 + j]

                percent_diff = next/current - 1

                if percent_to_pass <= percent_diff:
                    break
                profit_count += 1

            rewards_buy.append(profit_count)

        rewards_hold.append(0)
        rewards_buy.append(0)

        # plot_indicator(df, np.array(rewards_hold), "RewardsHold")
        # plot_indicator(df, np.array(rewards_buy) - 5, "RewardsBuy")

        return rewards_hold, rewards_buy


    def get_data(self, paths, type, timestamp, indicator, percent_buysell):
        dfs = []
        
        for path in paths:
            df = pd.read_json(path)
            dfs.append(df)

        result_df = pd.concat(dfs)
        return self.process_df(result_df, type, timestamp, indicator, percent_buysell)
    
    def process_df(self, result_df, type, timestamp, indicator, percent_buysell):

        prices = result_df["price"].values
        timestamps = ((pd.to_datetime(result_df["timestamp_close"]).astype('int64') // 10**6) + 1).to_numpy()
        buy_sells = self.get_buy_sell(result_df)
        rewards_hold, rewards_buy = self.get_rewards_action(result_df, percent_buysell)

        # rsi, williams, pump_dump just needs to be divided by 100
        # lwti just needs to be divided by 10
        # kandleCount_x_low, kandleCount_x_green, kandleCount_x_red, kandleCount_x_high, kandleCount_x_mid, candle_color, lwStrategyUp, lwStrategyDown - already [-1 : 1]
        # candle_height divided by 2
        if type != "standard" and type != "solo_price" and type != "only_price":
            exploded_df = result_df['indicators'].apply(pd.Series)

            result_df['price_percent'] = pd.to_numeric(result_df['price'], errors='coerce').astype(float).pct_change()
            
            if type != "solo_price_percent":
                result_df['price_high_percent'] = pd.to_numeric(result_df['price_high'], errors='coerce').astype(float).pct_change()
                result_df['price_low_percent'] = pd.to_numeric(result_df['price_low'], errors='coerce').astype(float).pct_change()

                if type != 'only_price_percent_sin_volume':
                    result_df['volume_percent'] = pd.to_numeric(result_df['volume'], errors='coerce').astype(float).pct_change()


        if type == "standard" or type == "all_percents" or type == "all_percents_without_candles":
            exploded_df = result_df['indicators'].apply(pd.Series)

            if type == "all_percents" or type == "all_percents_without_candles":
                normalized_indicators = self.get_normalized_indicators()
                for col in exploded_df.keys():
                    if type != "all_percents_without_candles" or (not col in normalized_indicators):
                        exploded_df[col] = pd.to_numeric(exploded_df[col], errors='coerce').astype(float)
                        exploded_df[col] = exploded_df[col].pct_change()

            result_df = pd.concat([result_df, exploded_df], axis=1)
        elif indicator != "none":
            exploded_df = result_df['indicators'].apply(pd.Series)

            indicators = ["rsi30", "sma8", "williams30"]
            if indicator == "rsi9":
                indicators = ["rsi9", "sma30", "williams30"]

            for i in range(3):
                ind = indicators[i]

                exploded_df[ind] = pd.to_numeric(exploded_df[ind], errors='coerce').astype(float)
                exploded_df[ind] = exploded_df[ind].pct_change()
                result_df[ind] = exploded_df[ind]


        if type == "solo_price" or type == "all_percents" or type == "solo_price_percent" or type == "only_price_percent" or type == "only_price_percent_sin_volume":
            result_df = result_df.drop(columns=['price', 'price_high', 'price_low', 'volume'])
        if type == "solo_price" or type == "all_percents" or type == "solo_price_percent" or type == "only_price_percent" or type == "only_price" or type == "only_price_percent_sin_volume":
            result_df = result_df.drop(columns=['timestamp_close'])

        if timestamp == "expanded":
            result_df['timestamp'] = pd.to_datetime(result_df['timestamp'], unit='ms')
            result_df['month'] = (result_df['timestamp'].dt.month - 1)/11
            result_df['day'] = result_df['timestamp'].dt.day - 1
            result_df['days_in_month'] = result_df.apply(days_in_month, axis=1) - 1
            result_df['day'] = result_df['day'] / result_df['days_in_month']
            result_df = result_df.drop(columns=['days_in_month'])
            result_df['time'] = (result_df['timestamp'].dt.hour * 60 + result_df['timestamp'].dt.minute)/1439
            result_df['day_of_week'] = result_df['timestamp'].dt.dayofweek/6
        elif timestamp == "day_of_week":
            result_df['timestamp'] = pd.to_datetime(result_df['timestamp'], unit='ms')
            result_df['day_of_week'] = result_df['timestamp'].dt.dayofweek/6

        result_df = result_df.drop(columns=['timestamp'])
        result_df = result_df.drop(columns=['price_open', 'indicators'])
        result_df = result_df.drop(columns=['asset_volume_quote', 'trades_number', 'asset_volume_taker_base', 'asset_volume_taker_quote']) # for now lets ignore these

        for col in result_df.keys():
            if col == 'timestamp_close' or col == 'timestamp':
                result_df[col] = pd.to_numeric(result_df[col]).astype(int) / 1000000000 # weird df formatting
            else:
                result_df[col] = pd.to_numeric(result_df[col], errors='coerce').astype(float)

        result_df = result_df.fillna(0)
        result_df = result_df.replace([np.inf, -np.inf], 0)
                
        # print('FINDING NANS:')
        # nan_df = result_df.isna()
        # nan_indices = []

        # for index, row in nan_df.iterrows():
        #     for col in nan_df.columns:
        #         if row[col]:
        #             print(f"{col}:{index} real: {result_df.loc[index][col]}")


        # print(result_df.head())
        # print(result_df.tail())

        result_df = result_df.reset_index(drop=True)

        return result_df, prices, timestamps, buy_sells, rewards_hold, rewards_buy
    
def days_in_month(row):
    year = row['timestamp'].year
    month = row['timestamp'].month
    return calendar.monthrange(year, month)[1]