from abc import ABC, abstractmethod
from typing import List
from src.conf.data_config import DataConfig
import pandas as pd
import numpy as np
import calendar
import datetime

from src.data.data_utils import plot_indicator
from src.util.plot import plot_timeseries

class AbstractDataProvider(ABC):
    def __init__(self, config: DataConfig):
        super().__init__()

        self.config = config
        self.id = self.get_id(config)
   
    def get_id(self, config: DataConfig):
        return f'{config.id}_{config.type}_{config.timestamp}_{config.indicator}_{config.fidelity_input + "|" + config.fidelity_run}_{"|".join(config.layers)}_{config.fidelity_input_test + "|" + config.fidelity_run_test}_{"|".join(config.layers_test)}_{config.lookback_window_size}_{config.buyreward_percent}_{config.buyreward_maxwait}'.replace('.', '~').replace('|', ']')
    
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
    # buy_sell [-1,-1,2,1,1,-2,3,-3] shows matching 2|-2 and matching 3|-3 buy|sell pairs
    def get_rewards_buy_sell(self, df):
        prices = df['price'].values

        count = 1
        buy_sell = []
        action = -1
        for i in range(len(prices) - 1):
            current = prices[i]
            next = prices[i+1]
            if current < next:
                if action == 1:
                    buy_sell.append(action)
                else:
                    action = 1
                    count += 1
                    buy_sell.append(count)
            else:
                if action == -1:
                    buy_sell.append(action)
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

            profit_count = 1
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
        rewards_drawdown.append(0)

        # df_copy = df.copy()
        # df_copy['rewards_profitable'] = rewards_profitable
        # df_copy['rewards_drawdown'] = rewards_drawdown
        # df_copy.to_csv(f'all_data.csv', index=False)  

        # plot_indicator(df_copy, np.array(rewards_profitable) - 3, "RewardsBuy")

        return rewards_profitable, rewards_drawdown


    def get_data(self, paths, type, timestamp, indicator, buyreward_percent, buyreward_maxwait):
        dfs = []
        
        for path in paths:
            df = pd.read_json(path)
            dfs.append(df)

        result_df = pd.concat(dfs, ignore_index=True)
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
    
    def get_raw_data(self, paths, timestamp = "none", columns = ["timestamp","timestamp_close","price","price_open","price_high","price_low","volume","asset_volume_quote","trades_number"]):
        dfs = []

        # "asset_volume_taker_base":"616.24854100","asset_volume_taker_quote":"2678216.40060401"
        
        for path in paths:
            df = pd.read_json(path)
            dfs.append(df)

        result_df = pd.concat(dfs, ignore_index=True)

        raw_df = result_df[columns]
        result_df, prices, timestamps = self.process_df_simple(raw_df.copy(), timestamp, columns)
        return raw_df, result_df, prices, timestamps
    
    def process_df_simple(self, result_df, timestamp, columns):

        prices = result_df["price"].values
        timestamps = ((pd.to_datetime(result_df["timestamp_close"]).astype('int64') // 10**6) + 1).to_numpy()

        #1 Empty 5/15
        #2 Empty duel-dqn 2/15
        #3 total_volume 7/15
        #4 total_volume + volumes-max 7/15
        #5 total_volume + price-max|avg 7/15
        #6 total_volume + price-max|avg + price_z_score 1/3

        # TODO: check 1d vs 1y vs 1m vs 1w
        
        result_df['price_z_score_1d'] = (result_df['price'] - result_df['price'].rolling(1440).mean()) / result_df['price'].rolling(1440).std()
        result_df['price_z_score_1m'] = (result_df['price'] - result_df['price'].rolling(43200).mean()) / result_df['price'].rolling(43200).std()
        result_df['price_z_score_1y'] = (result_df['price'] - result_df['price'].rolling(525600).mean()) / result_df['price'].rolling(525600).std()
        
        result_df['price_to_max_1d'] = pd.to_numeric(result_df['price'] / result_df['price'].rolling(window=1440).max(), errors='coerce').astype(float)
        result_df['price_to_max_1m'] = pd.to_numeric(result_df['price'] / result_df['price'].rolling(window=43200).max(), errors='coerce').astype(float)
        result_df['price_to_max_1y'] = pd.to_numeric(result_df['price'] / result_df['price'].rolling(window=525600).max(), errors='coerce').astype(float)
        result_df['price_to_avg_1d'] = pd.to_numeric(result_df['price'] / result_df['price'].rolling(window=1440).mean(), errors='coerce').astype(float)
        result_df['price_to_avg_1m'] = pd.to_numeric(result_df['price'] / result_df['price'].rolling(window=43200).mean(), errors='coerce').astype(float)
        result_df['price_to_avg_1y'] = pd.to_numeric(result_df['price'] / result_df['price'].rolling(window=525600).mean(), errors='coerce').astype(float)

        # result_df['volume_avg_1d'] = result_df['volume'].rolling(window=1440).mean()
        # result_df['volume_avg_1m'] = result_df['volume'].rolling(window=43200).mean()
        # result_df['volume_avg_1y'] = result_df['volume'].rolling(window=525600).mean()
        # result_df['volume_max_1d'] = result_df['volume'].rolling(window=1440).max()
        # result_df['volume_max_1m'] = result_df['volume'].rolling(window=43200).max()
        # result_df['volume_max_1y'] = result_df['volume'].rolling(window=525600).max()

        # result_df['volume_quote_avg_1d'] = result_df['asset_volume_quote'].rolling(window=1440).mean()
        # result_df['volume_quote_avg_1m'] = result_df['asset_volume_quote'].rolling(window=43200).mean()
        # result_df['volume_quote_avg_1y'] = result_df['asset_volume_quote'].rolling(window=525600).mean()
        # result_df['volume_quote_max_1d'] = result_df['asset_volume_quote'].rolling(window=1440).max()
        # result_df['volume_quote_max_1m'] = result_df['asset_volume_quote'].rolling(window=43200).max()
        # result_df['volume_quote_max_1y'] = result_df['asset_volume_quote'].rolling(window=525600).max()

        result_df['total_volume'] = result_df['volume'] + result_df['asset_volume_quote']

        columns_to_drop = [] + columns
        columns_to_drop = columns_to_drop + ['total_volume'] 
        
        # result_df['volume_to_max_1d'] = pd.to_numeric(result_df['volume'] / result_df['volume_max_1d'], errors='coerce').astype(float)
        # result_df['volume_to_max_1m'] = pd.to_numeric(result_df['volume'] / result_df['volume_max_1m'], errors='coerce').astype(float)
        # result_df['volume_to_max_1y'] = pd.to_numeric(result_df['volume'] / result_df['volume_max_1y'], errors='coerce').astype(float)
        # result_df['volume_to_avg_1d'] = pd.to_numeric(result_df['volume'] / result_df['volume_avg_1d'], errors='coerce').astype(float)
        # result_df['volume_to_avg_1m'] = pd.to_numeric(result_df['volume'] / result_df['volume_avg_1m'], errors='coerce').astype(float)
        # result_df['volume_to_avg_1y'] = pd.to_numeric(result_df['volume'] / result_df['volume_avg_1y'], errors='coerce').astype(float)
        # result_df['volume_quote_to_max_1d'] = pd.to_numeric(result_df['asset_volume_quote'] / result_df['volume_quote_max_1d'], errors='coerce').astype(float)
        # result_df['volume_quote_to_max_1m'] = pd.to_numeric(result_df['asset_volume_quote'] / result_df['volume_quote_max_1m'], errors='coerce').astype(float)
        # result_df['volume_quote_to_max_1y'] = pd.to_numeric(result_df['asset_volume_quote'] / result_df['volume_quote_max_1y'], errors='coerce').astype(float)
        # result_df['volume_quote_to_avg_1d'] = pd.to_numeric(result_df['asset_volume_quote'] / result_df['volume_quote_avg_1d'], errors='coerce').astype(float)
        # result_df['volume_quote_to_avg_1m'] = pd.to_numeric(result_df['asset_volume_quote'] / result_df['volume_quote_avg_1m'], errors='coerce').astype(float)
        # result_df['volume_quote_to_avg_1y'] = pd.to_numeric(result_df['asset_volume_quote'] / result_df['volume_quote_avg_1y'], errors='coerce').astype(float)

        result_df['total_volume_percent'] = pd.to_numeric(result_df['total_volume'], errors='coerce').astype(float).pct_change()
        result_df['total_volume_to_max_1d'] = pd.to_numeric(result_df['total_volume'] / result_df['total_volume'].rolling(window=1440).max(), errors='coerce').astype(float)
        result_df['total_volume_to_max_1m'] = pd.to_numeric(result_df['total_volume'] / result_df['total_volume'].rolling(window=43200).max(), errors='coerce').astype(float)
        result_df['total_volume_to_max_1y'] = pd.to_numeric(result_df['total_volume'] / result_df['total_volume'].rolling(window=525600).max(), errors='coerce').astype(float)

        result_df['price_percent'] = pd.to_numeric(result_df['price'], errors='coerce').astype(float).pct_change()
        result_df['price_high_percent'] = pd.to_numeric(result_df['price_high']/result_df['price'] - 1, errors='coerce').astype(float)
        result_df['price_low_percent'] = pd.to_numeric(result_df['price_low']/result_df['price'] - 1, errors='coerce').astype(float)
        result_df['volume_percent'] = pd.to_numeric(result_df['volume'], errors='coerce').astype(float).pct_change()
        result_df['volume_quote_percent'] = pd.to_numeric(result_df['asset_volume_quote'], errors='coerce').astype(float).pct_change()
        result_df['trades_number_percent'] = pd.to_numeric(result_df['trades_number'], errors='coerce').astype(float).pct_change()

        if timestamp == "expanded":
            result_df['timestamp_close'] = pd.to_datetime(result_df['timestamp_close'], unit='ms')
            result_df['month'] = (result_df['timestamp_close'].dt.month - 1)/11
            result_df['day'] = result_df['timestamp_close'].dt.day - 1
            result_df['days_in_month'] = result_df.apply(days_in_month, axis=1) - 1
            result_df['day'] = result_df['day'] / result_df['days_in_month']
            result_df = result_df.drop(columns=['days_in_month'])
            result_df['time'] = (result_df['timestamp_close'].dt.hour * 60 + result_df['timestamp_close'].dt.minute)/1439
            result_df['day_of_week'] = result_df['timestamp_close'].dt.dayofweek/6
        elif timestamp == "day_of_week":
            result_df['timestamp_close'] = pd.to_datetime(result_df['timestamp_close'], unit='ms')
            result_df['day_of_week'] = result_df['timestamp_close'].dt.dayofweek/6
        elif timestamp == "normal":
            result_df['timestamp_new'] = pd.to_numeric(result_df['timestamp']).astype(int) / 1000000000
            result_df['timestamp_close_new'] = pd.to_numeric(result_df['timestamp_close']).astype(int) / 1000000000

        result_df = result_df.drop(columns=columns_to_drop).fillna(0).replace([np.inf, -np.inf], 0).reset_index(drop=True)

        return result_df, prices, timestamps
    
    def process_fidelity(self, df, layer, fidelity_offset, multiplier_input, fidelity_run, multiplier_run, multiplier_input_to_run, timestamp, columns = ["timestamp","timestamp_close","price","price_open","price_high","price_low","volume","asset_volume_quote","trades_number"]):
        steps = df.shape[0]

        dfs = []
        raw_dfs = []
        prices = []
        for i in range(0, multiplier_run):
            dfs.append(pd.DataFrame())
            raw_dfs.append(pd.DataFrame())
            prices.append([])
        for i in range(0, multiplier_run):

            values = {}
            for c in columns:
                values[c] = []

            mapping = self.get_current_mapping(df, i*multiplier_input_to_run, layer, fidelity_run)

            fidelity_steps = int((steps-fidelity_offset-i*multiplier_input_to_run)/multiplier_input)

            for j in range(0, fidelity_steps):
                pos_from = fidelity_offset + i*multiplier_input_to_run + (j*multiplier_input)
                part = df.iloc[pos_from:pos_from+multiplier_input].reset_index(drop=True)

                values["timestamp"].append(part['timestamp'].values[0])
                values["timestamp_close"].append(part['timestamp_close'].values[multiplier_input-1])
                values["price"].append(part['price'].values[multiplier_input-1])
                values["price_open"].append(part['price_open'].values[0])
                values["price_high"].append(part['price_high'].max())
                values["price_low"].append(part['price_low'].min())
                values["volume"].append(part['volume'].sum())
                values["asset_volume_quote"].append(part['asset_volume_quote'].sum())
                values["trades_number"].append(part['trades_number'].sum())
                
            raw_df = pd.DataFrame(values, columns=columns)
            result_df, ps, _ = self.process_df_simple(raw_df.copy(), timestamp, columns)
            dfs[mapping] = result_df
            raw_dfs[mapping] = raw_df
            prices[mapping] = ps

            # print(raw_df.head())
            # print(result_df.tail())

        return dfs, raw_dfs, prices

    def get_current_mapping(self, df, step, layer, fidelity):
        date_time = self.get_timestamp_datetime(df, step)

        if fidelity == "1m":
            if layer == "1w":
                return date_time.weekday *60*24 + date_time.hour *60 + date_time.minute
            if layer == "1d":
                return date_time.hour *60 + date_time.minute
            if layer == "1h":
                return date_time.minute
            if layer == "30m":
                return date_time.minute % 30
            if layer == "15m":
                return date_time.minute % 15
            if layer == "10m":
                return date_time.minute % 10
            if layer == "5m":
                return date_time.minute % 5
            return 0
        if fidelity == "30m":
            if layer == "1w":
                return int((date_time.weekday *60*24 + date_time.hour *60 + date_time.minute) / 30 % (2*24*7))
            if layer == "1d":
                return int((date_time.hour *60 + date_time.minute) / 30 % (2*24))
            if layer == "1h":
                return int((date_time.minute) / 30 % 2)
            return 0
        if fidelity == "15m":
            if layer == "1w":
                return int((date_time.weekday *60*24 + date_time.hour *60 + date_time.minute) / 15 % (4*24*7))
            if layer == "1d":
                return int((date_time.hour *60 + date_time.minute) / 15 % (4*24))
            if layer == "1h":
                return int((date_time.minute) / 15 % 4)
            if layer == "30m":
                return int((date_time.minute) / 15 % 2)
            return 0
        if fidelity == "10m":
            if layer == "1w":
                return int((date_time.weekday *60*24 + date_time.hour *60 + date_time.minute) / 10 % (6*24*7))
            if layer == "1d":
                return int((date_time.hour *60 + date_time.minute) / 10 % (6*24))
            if layer == "1h":
                return int((date_time.minute) / 10 % 6)
            if layer == "30m":
                return int((date_time.minute) / 10 % 3)
            return 0
        if fidelity == "5m":
            if layer == "1w":
                return int((date_time.weekday *60*24 + date_time.hour *60 + date_time.minute) / 5 % (12*24*7))
            if layer == "1d":
                return int((date_time.hour *60 + date_time.minute) / 5 % (12*24))
            if layer == "1h":
                return int((date_time.minute) / 5 % 12)
            if layer == "30m":
                return int((date_time.minute) / 5 % 6)
            if layer == "15m":
                return int((date_time.minute) / 5 % 3)
            if layer == "10m":
                return int((date_time.minute) / 5 % 2)
            return 0
        else:
            if layer == "1w":
                return date_time.weekday *24 + date_time.hour
            if layer == "1d":
                return date_time.hour % 24
            return 0
            

    def get_timestamp_datetime(self, df, i):
        return pd.to_datetime(df.iloc[i]["timestamp"])
        # return datetime.datetime.fromtimestamp(df.iloc[i]['timestamp'])



    
def days_in_month(row):
    year = row['timestamp_close'].year
    month = row['timestamp_close'].month
    return calendar.monthrange(year, month)[1]