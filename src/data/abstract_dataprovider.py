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
        return f'{config.id}_{config.type}_{config.timestamp}_{config.indicator}_{config.fidelity}_{"|".join(config.layers)}_{config.lookback_window_size}_{1 if config.flat_lookback else 0}_{1 if config.flat_layers else 0}_{config.buyreward_percent}_{config.buyreward_maxwait}'
    
    def is_multilayered(self) -> bool:
        return len(self.config.layers) > 1
    
    @abstractmethod
    def get_timesteps(self) -> int:
        pass
    
    @abstractmethod
    def get_price(self, step: int) -> float:
        pass

    @abstractmethod
    def get_signal_buy_sell(self, step: int) -> int:
        pass
    @abstractmethod
    def get_signal_buy_profitable(self, step: int) -> int:
        pass
    @abstractmethod
    def get_signal_buy_drawdown(self, step: int) -> int:
        pass
    
    @abstractmethod
    def get_values(self, step: int):
        pass
    
    def get_lookback_window(self) -> int:
        return self.config.lookback_window_size

    def get_start_index(self):
        return 0
    
    # buy_sell positive means buy, negative means sell for matching number of buy.
    # buy_sell [0,0,1,0,0,-1,2,-2] shows matching 1|-1 and matching 2|-2 buy|sell pairs
    def get_rewards_buy_sell(self, df):
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

    # buy is time to go profitable
    def get_rewards_buy(self, df, percent_to_pass = 0.004, max_buy_wait = 20):
        prices = df['price'].values

        rewards_profitable = []
        rewards_drawdown = []
        
        total_prices = len(prices) - 1
        for i in range(total_prices):
            current = prices[i]

            profit_count = 0
            drawdown = 0
            for j in range(min(total_prices - i, max_buy_wait)):
                next = prices[i+1 + j]

                percent_diff = next/current - 1

                if percent_diff < drawdown:
                    drawdown = percent_diff

                if percent_to_pass <= percent_diff:
                    break
                profit_count += 1

            rewards_profitable.append(profit_count)
            rewards_drawdown.append(drawdown)

        rewards_profitable.append(0)

        # plot_indicator(df, np.array(rewards_buy) - 5, "RewardsBuy")

        return rewards_profitable, rewards_drawdown


    def get_data(self, paths, type, timestamp, indicator, buyreward_percent, buyreward_maxwait):
        dfs = []
        
        for path in paths:
            df = pd.read_json(path)
            dfs.append(df)

        result_df = pd.concat(dfs)
        return self.process_df(result_df, type, timestamp, indicator, buyreward_percent, buyreward_maxwait)
    
    def process_df(self, result_df, type, timestamp, indicator, buyreward_percent, buyreward_maxwait):

        prices = result_df["price"].values
        timestamps = ((pd.to_datetime(result_df["timestamp_close"]).astype('int64') // 10**6) + 1).to_numpy()
        rewards_buysell = self.get_rewards_buy_sell(result_df)
        rewards_buy_profitable, rewards_buy_drawdown = self.get_rewards_buy(result_df, buyreward_percent, buyreward_maxwait)

        if type != "standard" and type != "solo_price" and type != "only_price":
            result_df['price_percent'] = pd.to_numeric(result_df['price'], errors='coerce').astype(float).pct_change()
            
            if type != "solo_price_percent":
                result_df['price_high_percent'] = pd.to_numeric(result_df['price_high']/result_df['price'] - 1, errors='coerce').astype(float)
                result_df['price_low_percent'] = pd.to_numeric(result_df['price_low']/result_df['price'] - 1, errors='coerce').astype(float)

                if type != 'only_price_percent_sin_volume':
                    result_df['volume_percent'] = pd.to_numeric(result_df['volume'], errors='coerce').astype(float).pct_change()


        if type == "standard" or type == "all_percents":
            exploded_df = result_df['indicators'].apply(pd.Series)

            if type == "all_percents":
                for col in exploded_df.keys():
                    exploded_df[col] = pd.to_numeric(exploded_df[col], errors='coerce').astype(float)
                    exploded_df[col] = exploded_df[col].pct_change()

            result_df = pd.concat([result_df, exploded_df], axis=1)
        elif indicator == "indicators1":
            exploded_df = result_df['indicators'].apply(pd.Series)
            indicators = ["kallman15", "timeseriesMomentum7", "closenessTo1000", "closenessTo10000", "meanReversion10", "meanReversion15", "rsi5", "rsi10", "rsi15"]
            for ind in indicators:
                result_df[ind] = pd.to_numeric(exploded_df[ind], errors='coerce').astype(float)
        elif indicator == "indicators2":
            exploded_df = result_df['indicators'].apply(pd.Series)
            indicators = ["timeseriesMomentum7", "closenessTo1000", "closenessTo10000", "meanReversion10", "meanReversion15", "rsi10", "choppiness30"]
            for ind in indicators:
                result_df[ind] = pd.to_numeric(exploded_df[ind], errors='coerce').astype(float)
        elif indicator == "indicators3":
            exploded_df = result_df['indicators'].apply(pd.Series)
            indicators = ["kallman15", "timeseriesMomentum7", "closenessTo1000", "closenessTo10000", "meanReversion10", "meanReversion15", "rsi10", "choppiness30"]
            for ind in indicators:
                result_df[ind] = pd.to_numeric(exploded_df[ind], errors='coerce').astype(float)
        elif indicator == "indicators4":
            exploded_df = result_df['indicators'].apply(pd.Series)
            indicators = ["kallman15", "timeseriesMomentum7", "closenessTo1000", "closenessTo10000", "meanReversion10", "meanReversion15", "choppiness30"]
            for ind in indicators:
                result_df[ind] = pd.to_numeric(exploded_df[ind], errors='coerce').astype(float)
        elif indicator == "indicators5":
            exploded_df = result_df['indicators'].apply(pd.Series)
            indicators = ["timeseriesMomentum7", "closenessTo1000", "closenessTo10000", "meanReversion10", "meanReversion15", "choppiness30"]
            for ind in indicators:
                result_df[ind] = pd.to_numeric(exploded_df[ind], errors='coerce').astype(float)       
        elif indicator == "indicators6":
            exploded_df = result_df['indicators'].apply(pd.Series)
            indicators = ["timeseriesMomentum7", "closenessTo1000", "closenessTo10000", "meanReversion10", "meanReversion15", "choppiness30", "cci10"]
            for ind in indicators:
                result_df[ind] = pd.to_numeric(exploded_df[ind], errors='coerce').astype(float)
        elif indicator == "indicators7":
            exploded_df = result_df['indicators'].apply(pd.Series)
            indicators = ["timeseriesMomentum7", "closenessTo1000", "closenessTo10000", "meanReversion10", "meanReversion15", "choppiness30", "disparityIndex7", "disparityIndex10"]
            for ind in indicators:
                result_df[ind] = pd.to_numeric(exploded_df[ind], errors='coerce').astype(float)
        elif indicator == "indicators8":
            exploded_df = result_df['indicators'].apply(pd.Series)
            indicators = ["timeseriesMomentum7", "closenessTo1000", "closenessTo10000", "meanReversion10", "meanReversion15", "choppiness30", "sortinoRatio30"]
            for ind in indicators:
                result_df[ind] = pd.to_numeric(exploded_df[ind], errors='coerce').astype(float)
        

        elif indicator == "indicators9":
            exploded_df = result_df['indicators'].apply(pd.Series)
            indicators = ["timeseriesMomentum7", "closenessTo1000", "closenessTo10000", "meanReversion10", "meanReversion15", "choppiness30", "volatilityVolume30", "volatilityVolume7"]
            for ind in indicators:
                result_df[ind] = pd.to_numeric(exploded_df[ind], errors='coerce').astype(float)
        elif indicator == "indicators10":
            exploded_df = result_df['indicators'].apply(pd.Series)
            indicators = ["timeseriesMomentum7", "closenessTo1000", "closenessTo10000", "meanReversion10", "meanReversion15", "choppiness30", "cci5", "cci7", "cci10", "turbulenceIndex10", "disparityIndex7", "disparityIndex10", "volatilityVolume30", "volatilityVolume7"]
            for ind in indicators:
                result_df[ind] = pd.to_numeric(exploded_df[ind], errors='coerce').astype(float)
        elif indicator == "indicators11":
            exploded_df = result_df['indicators'].apply(pd.Series)
            indicators = ["timeseriesMomentum7", "closenessTo1000", "closenessTo10000", "meanReversion10", "meanReversion15", "choppiness30", "williams5"]
            for ind in indicators:
                result_df[ind] = pd.to_numeric(exploded_df[ind], errors='coerce').astype(float)
        elif indicator == "indicators12":
            exploded_df = result_df['indicators'].apply(pd.Series)
            indicators = ["timeseriesMomentum7", "closenessTo1000", "closenessTo10000", "meanReversion10", "meanReversion15", "choppiness30", "williams10"]
            for ind in indicators:
                result_df[ind] = pd.to_numeric(exploded_df[ind], errors='coerce').astype(float)
        elif indicator == "indicators13":
            exploded_df = result_df['indicators'].apply(pd.Series)
            indicators = ["timeseriesMomentum7", "closenessTo1000", "closenessTo10000", "meanReversion10", "meanReversion15", "choppiness30", "stochasticOscillator5"]
            for ind in indicators:
                result_df[ind] = pd.to_numeric(exploded_df[ind], errors='coerce').astype(float)
        elif indicator == "indicators14":
            exploded_df = result_df['indicators'].apply(pd.Series)
            indicators = ["timeseriesMomentum7", "closenessTo1000", "closenessTo10000", "meanReversion10", "meanReversion15", "choppiness30", "stochasticOscillator10"]
            for ind in indicators:
                result_df[ind] = pd.to_numeric(exploded_df[ind], errors='coerce').astype(float)
        elif indicator == "indicators15":
            exploded_df = result_df['indicators'].apply(pd.Series)
            indicators = ["timeseriesMomentum7", "closenessTo1000", "closenessTo10000", "meanReversion10", "meanReversion15", "choppiness30", "sortinoRatio5"]
            for ind in indicators:
                result_df[ind] = pd.to_numeric(exploded_df[ind], errors='coerce').astype(float)
        
        elif indicator != "none":
            exploded_df = result_df['indicators'].apply(pd.Series)
            result_df[indicator] = pd.to_numeric(exploded_df[indicator], errors='coerce').astype(float)

        if type == "solo_price" or type == "all_percents" or type == "solo_price_percent" or type == "only_price_percent" or type == "only_price_percent_change" or type == "only_price_percent_sin_volume":
            result_df = result_df.drop(columns=['price', 'price_high', 'price_low', 'volume'])
        if type == "solo_price" or type == "all_percents" or type == "solo_price_percent" or type == "only_price_percent" or type == "only_price_percent_change" or type == "only_price" or type == "only_price_percent_sin_volume":
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

        return result_df, prices, timestamps, rewards_buysell, rewards_buy_profitable, rewards_buy_drawdown
    
def days_in_month(row):
    year = row['timestamp'].year
    month = row['timestamp'].month
    return calendar.monthrange(year, month)[1]