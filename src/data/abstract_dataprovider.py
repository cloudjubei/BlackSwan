from abc import ABC, abstractmethod
from typing import List
from src.conf.data_config import DataConfig
import pandas as pd
import numpy as np
import calendar
import datetime
import hashlib

from src.data.data_utils import plot_indicator
from src.util.plot import plot_timeseries
from src.data import indicators as _indicators
from src.data import feature_cache
from trainer.context import context_columns, load_series_observations, resolve_context

# Bars needed before a time-scaled rolling horizon emits a value. A horizon longer than the available
# history (a 1-year window on a 1-year test split) would otherwise be all-NaN -> a constant fill -> a DEAD
# feature that occupies observation space and skews every feature-importance read. Degrading to a shorter
# trailing sample keeps the feature real and stays backward-looking (no lookahead).
MIN_ROLLING_OBS = 20

# The neutral value of a RATIO feature is parity (1.0 = at the reference), not 0.0 — a zero would assert
# "price is 0x its average", an extreme false reading injected at every warm-up bar.
RATIO_FEATURE_NEUTRAL = 1.0


def _infer_minutes_per_bar(closes):
    """Minutes between consecutive bar closes. Accepts datetime64 closes (what `read_json` yields) OR
    epoch-millisecond integers (what an aggregated/resampled frame carries) — reading ms as ns collapses
    the inference to 1 minute, which silently unscales every 1d/1m/1y horizon into a dead column."""
    series = pd.Series(closes)
    if not pd.api.types.is_datetime64_any_dtype(series):
        series = pd.to_datetime(pd.to_numeric(series, errors="coerce"), unit="ms", errors="coerce")
    span_ms = (series.astype("int64") // 10**6).diff().median()
    if not pd.notna(span_ms) or span_ms <= 0:
        return 1
    return max(1, int(round(span_ms / 60000.0)))


def _rolling(series, window):
    """A trailing window that degrades to a shorter sample rather than blanking the whole column."""
    return series.rolling(window, min_periods=min(window, MIN_ROLLING_OBS))


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

    def get_open(self, step: int) -> float:
        # The OPEN price of decision bar `step`. Default = the bar's close (get_price), a safe fallback for
        # providers that don't expose opens; the real providers override with the true open so next_open
        # fills execute at the next bar's OPEN. Never a look-ahead (open of `step` is known at `step`).
        return self.get_price(step)

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

    def get_feature(self, step: int, name: str) -> float:
        """Value of a named feature column at ``step``'s most-recent CLOSED decision bar. Deterministic
        indicator strategies (TechnicalStrategyModel) read their signal columns through this — it reuses
        each provider's OWN look-ahead-safe decision-bar mapping, so the strategy never re-derives the
        stride/fidelity math. Providers with no named feature frame don't implement it."""
        raise NotImplementedError(f"{type(self).__name__} does not expose named features")

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

    def _add_curated_indicators(self, result_df):
        if not getattr(self.config, "use_indicators", False):
            return result_df
        if not all(c in result_df.columns for c in ("price", "price_high", "price_low", "volume")):
            return result_df
        close = pd.to_numeric(result_df["price"], errors="coerce")
        high = pd.to_numeric(result_df["price_high"], errors="coerce")
        low = pd.to_numeric(result_df["price_low"], errors="coerce")
        volume = pd.to_numeric(result_df["volume"], errors="coerce")
        result_df["rsi10"] = _indicators.rsi(close, 10)
        result_df["williams10"] = _indicators.williams(close, 10)
        result_df["stochasticOscillator10"] = _indicators.stochastic(close, low, high, 10)
        result_df["choppiness30"] = _indicators.choppiness(close, low, high, 30)
        result_df["meanReversion10"] = np.tanh(_indicators.mean_reversion(close, 10) / 2.5)
        result_df["turbulenceIndex10"] = np.tanh(_indicators.turbulence(close, 10) / 4.0)
        result_df["obv10"] = np.tanh(_indicators.obv(close, volume, 10) / 1.5)
        # Regime/trend context (research bets): realized-volatility regime + deviation-from-trend, both
        # tanh-bounded into [-1,1]. Volatility tells the agent how turbulent the tape is; trend slope
        # tells it which way and how strongly price is moving relative to its recent mean.
        returns = close.pct_change()
        result_df["volRegime10"] = np.tanh(returns.rolling(10).std() * 20.0)
        result_df["trendSlope10"] = np.tanh((close / close.rolling(10).mean() - 1.0) * 20.0)
        return result_df

    def _add_calendar_channel(self, result_df):
        """Stage-1 SEASONALITY channel: day-of-week / day-of-month / month / turn-of-month, derived purely
        from the bar's own close timestamp. Nothing is read ahead of the bar and no external table is
        involved, so the channel is causal by construction. Scaled to [0,1] to sit inside the declared
        Box(-1,1). Off by default — `calendar_features` is the isolating lever."""
        if not getattr(self.config, "calendar_features", False):
            return result_df
        if 'timestamp_close' not in result_df.columns:
            return result_df
        closes = pd.to_datetime(result_df['timestamp_close'], unit='ms', errors='coerce') \
            if not pd.api.types.is_datetime64_any_dtype(result_df['timestamp_close']) \
            else pd.to_datetime(result_df['timestamp_close'])
        days_in = closes.dt.days_in_month
        result_df['cal_day_of_week'] = closes.dt.dayofweek / 6.0
        result_df['cal_day_of_month'] = (closes.dt.day - 1) / (days_in - 1).clip(lower=1)
        result_df['cal_month'] = (closes.dt.month - 1) / 11.0
        # Turn-of-month: the last 2 and first 3 calendar days, the window the documented turn-of-month
        # effect lives in. A pure calendar predicate — knowable in advance, never a peek at future prices.
        result_df['cal_turn_of_month'] = (
            (closes.dt.day <= 3) | (closes.dt.day >= days_in - 1)
        ).astype(float)
        return result_df

    def _add_regime_channel(self, result_df):
        """Stage-1 REGIME channel: realized-volatility level + deviation-from-trend, tanh-bounded — the same
        pair the curated bundle emits, available WITHOUT the rest of it so the lens can be attributed on its
        own. A no-op when the bundle already added them (never duplicated / overwritten)."""
        if not getattr(self.config, "regime", False):
            return result_df
        if 'volRegime10' in result_df.columns and 'trendSlope10' in result_df.columns:
            return result_df
        if 'price' not in result_df.columns:
            return result_df
        close = pd.to_numeric(result_df['price'], errors='coerce')
        returns = close.pct_change()
        result_df['volRegime10'] = np.tanh(returns.rolling(10).std() * 20.0)
        result_df['trendSlope10'] = np.tanh((close / close.rolling(10).mean() - 1.0) * 20.0)
        return result_df

    def _feature_cache_params(self, fn, timestamp, columns=None, **extra):
        """The cache key for a built feature frame. EVERY config knob that SHAPES the frame must appear
        here — a lever missing from the key makes each arm of a feature experiment silently reuse the
        baseline frame, voiding the comparison. Off-by-default channels are omitted when off, so
        pre-existing cache entries for default runs stay valid."""
        params = {
            "fn": fn,
            "timestamp": timestamp,
            "use_indicators": getattr(self.config, "use_indicators", False),
            "obs_squash": getattr(self.config, "obs_squash", "none"),
            **extra,
        }
        if columns is not None:
            params["columns"] = list(columns)
        ctx = self._context_cache_key()
        if ctx:
            params["context"] = ctx
        if getattr(self.config, "calendar_features", False):
            params["calendar_features"] = True
        if getattr(self.config, "regime", False):
            params["regime"] = True
        return params

    def get_data(self, paths, type, timestamp, indicator, buyreward_percent, buyreward_maxwait):
        params = self._feature_cache_params(
            "get_data",
            timestamp,
            type=type,
            indicator=indicator,
            buyreward_percent=buyreward_percent,
            buyreward_maxwait=buyreward_maxwait,
        )

        def _build():
            dfs = [pd.read_json(path) for path in paths]
            result_df = pd.concat(dfs, ignore_index=True)
            return self.process_df(result_df, type, timestamp, indicator, buyreward_percent, buyreward_maxwait)

        return feature_cache.load_or_build(paths, params, _build)
    
    def process_df(self, result_df, type, timestamp, indicator, buyreward_percent, buyreward_maxwait):

        prices = result_df["price"].values
        timestamps = ((pd.to_datetime(result_df["timestamp_close"]).astype('int64') // 10**6) + 1).to_numpy()
        rewards_buysell = self.get_rewards_buy_sell(result_df)
        rewards_buy_profitable, rewards_buy_drawdown = self.get_rewards_buy(result_df, buyreward_percent, buyreward_maxwait)

        # QW2: taker (aggressor) buy pressure — bounded [0,1], computed before the raw volume/taker
        # columns are dropped below so the order-flow signal survives into the feature set.
        result_df['taker_buy_ratio'] = (pd.to_numeric(result_df['asset_volume_taker_base'], errors='coerce') / pd.to_numeric(result_df['volume'], errors='coerce')).clip(lower=0, upper=1)

        result_df = self._add_curated_indicators(result_df)
        # The Stage-1 channels must be emitted on BOTH build paths — a default daily run goes through
        # SingleDataProvider/process_df, a stacked/resampled one through process_df_simple.
        result_df = self._add_regime_channel(result_df)
        result_df = self._add_calendar_channel(result_df)

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
            result_df['timestamp_close'] = pd.to_datetime(result_df['timestamp_close'], unit='ms')
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
        result_df = result_df.drop(columns=['price_open'])
        if 'indicators' in result_df.columns:
            result_df = result_df.drop(columns=['indicators'])
        result_df = result_df.drop(columns=['asset_volume_quote', 'trades_number', 'asset_volume_taker_base', 'asset_volume_taker_quote']) # raw cols dropped; taker_buy_ratio (above) retains the order-flow signal

        result_df = self._add_context_columns(result_df, timestamps)

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
    
    def get_raw_data(self, paths, timestamp = "none", columns = ["timestamp","timestamp_close","price","price_open","price_high","price_low","volume","asset_volume_quote","trades_number","asset_volume_taker_base"]):
        # asset_volume_taker_base carried through for the QW2 taker_buy_ratio feature. Indicators are
        # COMPUTED from this OHLCV at runtime (process_df_simple -> _add_curated_indicators), not read.

        params = self._feature_cache_params("get_raw_data", timestamp, columns=columns)

        def _build():
            frames = [pd.read_json(path) for path in paths]
            result_df = pd.concat(frames, ignore_index=True)
            raw_df = result_df[columns]
            processed, prices, timestamps = self.process_df_simple(raw_df.copy(), timestamp, columns)
            return raw_df, processed, prices, timestamps

        return feature_cache.load_or_build(paths, params, _build)
    
    def process_df_simple(self, result_df, timestamp, columns):

        prices = result_df["price"].values
        timestamps = ((pd.to_datetime(result_df["timestamp_close"]).astype('int64') // 10**6) + 1).to_numpy()

        result_df = self._add_curated_indicators(result_df)
        result_df = self._add_regime_channel(result_df)
        result_df = self._add_calendar_channel(result_df)

        # The _1d/_1m/_1y windows below are fixed time horizons (1 day / 1 month / 1 year), but
        # the bars reaching this method may be 1m, 1h or 1d (single-layer or process_fidelity
        # aggregates). Infer minutes-per-bar from the spacing between consecutive bar closes and
        # scale the windows so the horizons hold at any fidelity. Without this, e.g. rolling(1440)
        # on 1h bars spans 60 days, leaving every row NaN -> fillna(0) -> a signal-less zero column.
        minutes_per_bar = _infer_minutes_per_bar(result_df['timestamp_close'])
        w_1d = max(2, round(1440 / minutes_per_bar))
        w_1m = max(2, round(43200 / minutes_per_bar))
        w_1y = max(2, round(525600 / minutes_per_bar))

        result_df['price_z_score_1d'] = (result_df['price'] - _rolling(result_df['price'], w_1d).mean()) / _rolling(result_df['price'], w_1d).std()
        result_df['price_z_score_1m'] = (result_df['price'] - _rolling(result_df['price'], w_1m).mean()) / _rolling(result_df['price'], w_1m).std()
        result_df['price_z_score_1y'] = (result_df['price'] - _rolling(result_df['price'], w_1y).mean()) / _rolling(result_df['price'], w_1y).std()

        result_df['price_to_max_1d'] = pd.to_numeric(result_df['price'] / _rolling(result_df['price'], w_1d).max(), errors='coerce').astype(float)
        result_df['price_to_max_1m'] = pd.to_numeric(result_df['price'] / _rolling(result_df['price'], w_1m).max(), errors='coerce').astype(float)
        result_df['price_to_max_1y'] = pd.to_numeric(result_df['price'] / _rolling(result_df['price'], w_1y).max(), errors='coerce').astype(float)
        result_df['price_to_avg_1d'] = pd.to_numeric(result_df['price'] / _rolling(result_df['price'], w_1d).mean(), errors='coerce').astype(float)
        result_df['price_to_avg_1m'] = pd.to_numeric(result_df['price'] / _rolling(result_df['price'], w_1m).mean(), errors='coerce').astype(float)
        result_df['price_to_avg_1y'] = pd.to_numeric(result_df['price'] / _rolling(result_df['price'], w_1y).mean(), errors='coerce').astype(float)

        # adding data:
        # Pi_Cycle_Top_Signal  
        # Pi_Cycle_Bottom_Ratio
        # Pi_Cycle_Bottom
        # Pi_Cycle_Bottom_Signal

        # add indicators


        result_df['SMA111'] = result_df['price'].rolling(window=111).mean()
        result_df['SMA350'] = result_df['price'].rolling(window=350).mean()
        result_df['SMA350_x2'] = result_df['SMA350'] * 2
        result_df['Pi_Cycle_Top'] = result_df['SMA111'] > result_df['SMA350_x2']
        result_df['Pi_Cycle_Top_Ratio'] = (result_df['SMA111'] / result_df['SMA350_x2']).astype(float).fillna(RATIO_FEATURE_NEUTRAL)
        result_df['Pi_Cycle_Top_Signal'] = (result_df['Pi_Cycle_Top'] & ~result_df['Pi_Cycle_Top'].shift(1).fillna(False)).astype(int)

        result_df['SMA471'] = result_df['price'].rolling(window=471).mean()
        result_df['EMA150'] = result_df['price'].ewm(span=150, adjust=False).mean()
        result_df['SMA471_/2'] = result_df['SMA471'] / 2
        result_df['Pi_Cycle_Bottom'] = result_df['EMA150'] < result_df['SMA471_/2']
        result_df['Pi_Cycle_Bottom_Ratio'] = (result_df['SMA471_/2'] / result_df['EMA150']).astype(float).fillna(RATIO_FEATURE_NEUTRAL)
        result_df['Pi_Cycle_Bottom_Signal'] = (result_df['Pi_Cycle_Bottom'] & ~result_df['Pi_Cycle_Bottom'].shift(1).fillna(False)).astype(int)

        result_df['total_volume'] = result_df['volume'] + result_df['asset_volume_quote']

        columns_to_drop = [] + columns
        columns_to_drop = columns_to_drop + ['total_volume'] 
        # columns_to_drop = columns_to_drop + ['volume_avg_1d', 'volume_avg_1m', 'volume_avg_1y', 'volume_max_1d', 'volume_max_1m', 'volume_max_1y']
        # columns_to_drop = columns_to_drop + ['volume_quote_avg_1d', 'volume_quote_avg_1m', 'volume_quote_avg_1y', 'volume_quote_max_1d', 'volume_quote_max_1m', 'volume_quote_max_1y']
        columns_to_drop = columns_to_drop + ['SMA111', 'SMA350', 'SMA350_x2', 'Pi_Cycle_Top']
        columns_to_drop = columns_to_drop + ['SMA471', 'SMA471_/2', 'EMA150', 'Pi_Cycle_Bottom']


        result_df['total_volume_percent'] = pd.to_numeric(result_df['total_volume'], errors='coerce').astype(float).pct_change()
        result_df['total_volume_to_max_1d'] = pd.to_numeric(result_df['total_volume'] / _rolling(result_df['total_volume'], w_1d).max(), errors='coerce').astype(float)
        result_df['total_volume_to_max_1m'] = pd.to_numeric(result_df['total_volume'] / _rolling(result_df['total_volume'], w_1m).max(), errors='coerce').astype(float)
        result_df['total_volume_to_max_1y'] = pd.to_numeric(result_df['total_volume'] / _rolling(result_df['total_volume'], w_1y).max(), errors='coerce').astype(float)

        result_df['price_percent'] = pd.to_numeric(result_df['price'], errors='coerce').astype(float).pct_change()
        result_df['price_high_percent'] = pd.to_numeric(result_df['price_high']/result_df['price'] - 1, errors='coerce').astype(float)
        result_df['price_low_percent'] = pd.to_numeric(result_df['price_low']/result_df['price'] - 1, errors='coerce').astype(float)
        result_df['volume_percent'] = pd.to_numeric(result_df['volume'], errors='coerce').astype(float).pct_change()
        result_df['volume_quote_percent'] = pd.to_numeric(result_df['asset_volume_quote'], errors='coerce').astype(float).pct_change()
        result_df['trades_number_percent'] = pd.to_numeric(result_df['trades_number'], errors='coerce').astype(float).pct_change()

        # QW2: taker (aggressor) buy pressure — the fraction of base volume that was taker-buy. Already
        # present in every kline but previously dropped. Bounded [0,1] so it sits safely in the [-1,1]
        # observation space. Aggregates correctly: process_fidelity sums taker-base and volume, so the
        # ratio of the sums is the period's true taker-buy share. Guarded so a caller-supplied `columns`
        # list without the taker column still yields a consistent (neutral 0) feature instead of crashing.
        if 'asset_volume_taker_base' in result_df.columns:
            result_df['taker_buy_ratio'] = (pd.to_numeric(result_df['asset_volume_taker_base'], errors='coerce') / pd.to_numeric(result_df['volume'], errors='coerce')).clip(lower=0, upper=1)
        else:
            result_df['taker_buy_ratio'] = 0.0

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

        result_df = self._add_context_columns(result_df, timestamps)

        result_df = result_df.drop(columns=columns_to_drop)
        # Fill each family at its OWN neutral before the blanket zero: a ratio's neutral is parity, so a
        # warm-up bar reads "at the reference" instead of the extreme "zero times the reference". The
        # trailing fillna(0) stays as the safety net for the mean-centred families (z-scores, pct changes).
        ratio_cols = [c for c in result_df.columns if c.endswith(('_to_max_1d', '_to_max_1m', '_to_max_1y', '_to_avg_1d', '_to_avg_1m', '_to_avg_1y'))]
        if ratio_cols:
            result_df[ratio_cols] = result_df[ratio_cols].replace([np.inf, -np.inf], np.nan).fillna(RATIO_FEATURE_NEUTRAL)
        result_df = result_df.fillna(0).replace([np.inf, -np.inf], 0).reset_index(drop=True)
        result_df = self._squash_observation_features(result_df)

        return result_df, prices, timestamps

    def _squash_observation_features(self, result_df):
        """Apply the obs_squash experiment to the final feature frame: clip or tanh every numeric
        feature into the declared Box(-1,1) bound (SB3 does not clip), or leave as-is for "none"."""
        squash = getattr(self.config, "obs_squash", "none")
        if not squash or squash == "none":
            return result_df
        cols = result_df.select_dtypes(include=[np.number]).columns
        if squash == "clip":
            result_df[cols] = result_df[cols].clip(-1.0, 1.0)
        elif squash == "tanh":
            result_df[cols] = np.tanh(result_df[cols])
        return result_df

    def _add_context_columns(self, result_df, timestamps):
        """Fuse the run's GLOBAL context panel (mined macro series) onto the bar clock as raw-level
        columns, leakage-safe (each value visible only from its release instant, forward-filled). A no-op
        when the `context` config is 'none' (the default), so non-context runs are byte-identical."""
        panel = getattr(self.config, "context", "none")
        if not panel or panel == "none":
            return result_df
        _, specs = resolve_context({"context_set": panel})
        for name, values in context_columns(timestamps, specs, load_series_observations).items():
            result_df[name] = values
        return result_df

    def _context_cache_key(self):
        """The context panel for feature-cache keying, or None when 'none' — so non-context runs keep
        their EXISTING cache keys (no mass rebuild) while each context panel gets a distinct one."""
        panel = getattr(self.config, "context", "none")
        return panel if panel and panel != "none" else None

    def process_fidelity(self, df, layer, fidelity_offset, multiplier_input, fidelity_run, multiplier_run, multiplier_input_to_run, timestamp, columns = ["timestamp","timestamp_close","price","price_open","price_high","price_low","volume","asset_volume_quote","trades_number","asset_volume_taker_base"]):
        # Coarse-layer resampling is the dominant provider-build cost (~97% of a multi-year train build:
        # a per-bar pandas-slice loop). It is a pure function of the input frame's CONTENT + the resample
        # params + the feature config, so cache it keyed on a hash of those — every other seed/algo run on
        # the same window reuses it. The frame hash makes the key independent of the source-file path.
        frame_fp = hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values.tobytes()).hexdigest()
        _key_parts = [
            frame_fp, layer, fidelity_offset, multiplier_input, fidelity_run, multiplier_run,
            multiplier_input_to_run, timestamp, list(columns),
            getattr(self.config, "use_indicators", False), getattr(self.config, "obs_squash", "none"),
        ]
        _ctx = self._context_cache_key()
        if _ctx:
            _key_parts.append(("context", _ctx))
        key_src = repr(tuple(_key_parts))
        key = "fid_" + hashlib.sha256(key_src.encode()).hexdigest()[:28]
        return feature_cache.load_or_build_key(
            key,
            lambda: self._process_fidelity_uncached(
                df, layer, fidelity_offset, multiplier_input, fidelity_run, multiplier_run,
                multiplier_input_to_run, timestamp, columns,
            ),
        )

    def _process_fidelity_uncached(self, df, layer, fidelity_offset, multiplier_input, fidelity_run, multiplier_run, multiplier_input_to_run, timestamp, columns = ["timestamp","timestamp_close","price","price_open","price_high","price_low","volume","asset_volume_quote","trades_number","asset_volume_taker_base"]):
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
                values["asset_volume_taker_base"].append(part['asset_volume_taker_base'].sum())

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
                return date_time.weekday() *60*24 + date_time.hour *60 + date_time.minute
            if layer == "1d":
                return date_time.hour *60 + date_time.minute
            if layer == "4h":
                return (date_time.hour % 4) *60 + date_time.minute
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
                return int((date_time.weekday() *60*24 + date_time.hour *60 + date_time.minute) / 30 % (2*24*7))
            if layer == "1d":
                return int((date_time.hour *60 + date_time.minute) / 30 % (2*24))
            if layer == "4h":
                return int((((date_time.hour % 4) *60) + date_time.minute) / 30 % (2*4))
            if layer == "1h":
                return int((date_time.minute) / 30 % 2)
            return 0
        if fidelity == "15m":
            if layer == "1w":
                return int((date_time.weekday() *60*24 + date_time.hour *60 + date_time.minute) / 15 % (4*24*7))
            if layer == "1d":
                return int((date_time.hour *60 + date_time.minute) / 15 % (4*24))
            if layer == "4h":
                return int((((date_time.hour % 4) *60) + date_time.minute) / 15 % (4*4))
            if layer == "1h":
                return int((date_time.minute) / 15 % 4)
            if layer == "30m":
                return int((date_time.minute) / 15 % 2)
            return 0
        if fidelity == "10m":
            if layer == "1w":
                return int((date_time.weekday() *60*24 + date_time.hour *60 + date_time.minute) / 10 % (6*24*7))
            if layer == "1d":
                return int((date_time.hour *60 + date_time.minute) / 10 % (6*24))
            if layer == "4h":
                return int((((date_time.hour % 4) *60) + date_time.minute) / 10 % (6*4))
            if layer == "1h":
                return int((date_time.minute) / 10 % 6)
            if layer == "30m":
                return int((date_time.minute) / 10 % 3)
            return 0
        if fidelity == "5m":
            if layer == "1w":
                return int((date_time.weekday() *60*24 + date_time.hour *60 + date_time.minute) / 5 % (12*24*7))
            if layer == "1d":
                return int((date_time.hour *60 + date_time.minute) / 5 % (12*24))
            if layer == "4h":
                return int((((date_time.hour % 4) *60) + date_time.minute) / 5 % (12*4))
            if layer == "1h":
                return int((date_time.minute) / 5 % 12)
            if layer == "30m":
                return int((date_time.minute) / 5 % 6)
            if layer == "15m":
                return int((date_time.minute) / 5 % 3)
            if layer == "10m":
                return int((date_time.minute) / 5 % 2)
            return 0
        if fidelity == "1d":
            # DAILY step over a 1h base: bucket by DAY, not hour. A 1w layer has 7 day-phases (weekday);
            # the 1d layer and the 1h base each collapse to a single phase (0). The hourly else-branch
            # below would return hour-scaled phases that overrun the per-day substream count.
            if layer == "1w":
                return date_time.weekday()
            return 0
        else:
            if layer == "1w":
                return date_time.weekday() *24 + date_time.hour
            if layer == "1d":
                return date_time.hour % 24
            if layer == "4h":
                return date_time.hour % 4
            return 0
            

    def get_timestamp_datetime(self, df, i):
        return pd.to_datetime(df.iloc[i]["timestamp"])
        # return datetime.datetime.fromtimestamp(df.iloc[i]['timestamp'])



    
def days_in_month(row):
    year = row['timestamp_close'].year
    month = row['timestamp_close'].month
    return calendar.monthrange(year, month)[1]