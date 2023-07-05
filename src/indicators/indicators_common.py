import numpy as np
import pandas as pd

def calculateTrendDirection(df, open_name, close_name, low_name, high_name, trend_name, trend_change_amount):

    trend_name_change = f'{trend_name}_change'
    trend_name_consecutive = f'{trend_name}_consecutive'
    df[trend_name] = df[close_name] - df[open_name]
    df[trend_name_change] = 0
    df[trend_name_consecutive] = 0
    df[f'{trend_name}_swing'] = (df[high_name] - df[low_name])/trend_change_amount

    prev_high = True
    for i in range(len(df)):
        if df[trend_name].iloc[i] >= 0:
            if prev_high and i > 0:
                df[trend_name_consecutive].iloc[i] = df[trend_name_consecutive].iloc[i-1] + 1
            else:
                prev_high = True
                df[trend_name_change].iloc[i] = 1
                df[trend_name_consecutive].iloc[i] = 0
        else:
            if not prev_high and i > 0:
                df[trend_name_consecutive].iloc[i] = df[trend_name_consecutive].iloc[i-1] + 1
            else:
                prev_high = False
                df[trend_name_change].iloc[i] = 1
                df[trend_name_consecutive].iloc[i] = 0

        df[trend_name].iloc[i] = df[trend_name].iloc[i] / trend_change_amount
    

def calculateRSI(df, close_name, rsi_name, price_diff_name, window_size=14):
    price_diff = df[close_name].diff().dropna()
    gains = price_diff * (price_diff > 0)
    losses = -price_diff * (price_diff < 0)

    average_gain = gains.rolling(window_size).mean()
    average_loss = losses.rolling(window_size).mean()

    rs = average_gain / average_loss
    rsi = 100 - (100 / (1 + rs))

    df[price_diff_name] = price_diff
    df[rsi_name] = rsi

def calculateWilliams(df, price_inname, low_name, high_name, williams_name, period=14): #Williams %R
    wpr_values = []

    for i in range(period, len(df)):
        highest_high = max(df[high_name].iloc[i - period : i + 1])
        lowest_low = min(df[low_name].iloc[i - period : i + 1])
        closing_price = df[price_inname].iloc[i]
        wpr = ((highest_high - closing_price) / (highest_high - lowest_low)) #* -100
        wpr_values.append(wpr)

    df[williams_name] = pd.Series(wpr_values, index=df.index[period:])

def calculateSAR(df, low_name, high_name, sar_name, initial_sar=8, initial_ep=10, initial_af=0.02):
    sar = [initial_sar]  # List to store SAR values
    ep = initial_ep  # Initialize the extreme point
    af = initial_af  # Initialize the acceleration factor

    for i in range(1, len(df)):
        high = df[high_name].iloc[i]
        low = df[low_name].iloc[i]
        prev_sar = sar[i - 1]

        if prev_sar < high:  # Bullish SAR condition
            sar.append(prev_sar + af * (ep - prev_sar))
            if low < sar[i]:
                sar[i] = low
            if high > ep:
                ep = high
                af += initial_af  # Increase acceleration factor if new extreme point is observed

        else:  # Bearish SAR condition
            sar.append(prev_sar - af * (prev_sar - ep))
            if high > sar[i]:
                sar[i] = high
            if low < ep:
                ep = low
                af += initial_af  # Increase acceleration factor if new extreme point is observed

    df[sar_name] = pd.Series(sar, index=df.index)

def calculateEMA(df, price_name, ma_name, ema_name, period=10): # Moving Average Indicator
    df[ma_name] = df[price_name].rolling(window=period).mean().shift(period-1)
    df[ema_name] = df[price_name].ewm(span=period, adjust=False).mean().shift(period-1) # WRONG!!!


def calculateOBV(df, price_name, volume_name, obv_name):
    df[obv_name] = 0
    for i in range(1, len(df)):
        if df[price_name].iloc[i] > df[price_name].iloc[i-1]:
            df[obv_name].iloc[i] = df[obv_name].iloc[i-1] + df[volume_name].iloc[i]
        elif df[price_name].iloc[i] < df[price_name].iloc[i-1]:
            df[obv_name].iloc[i] = df[obv_name].iloc[i-1] - df[volume_name].iloc[i]
        else:
            df[obv_name].iloc[i] = df[obv_name].iloc[i-1]


def calculateADL(df, price_name, volume_name, high_name, low_name, adl_name): #Accumulation/Distribution line
    df[adl_name] = 0
    for i in range(1, len(df)):
        money_flow_multiplier = ((df[price_name].iloc[i] - df[low_name].iloc[i]) - (df[high_name].iloc[i] - df[price_name].iloc[i])) / (df[high_name].iloc[i] - df[low_name].iloc[i])
        money_flow_volume = money_flow_multiplier * df[volume_name].iloc[i]
        df[adl_name].iloc[i] = df[adl_name].iloc[i-1] + money_flow_volume

def calculateADX(df, price_name, low_name, high_name, adx_name, period=14): #Average Directional Index
    df[adx_name] = 0
    df['High-Low'] = df[high_name] - df[low_name]
    df['High-PrevClose'] = abs(df[high_name] - df[price_name].shift(1))
    df['Low-PrevClose'] = abs(df[low_name] - df[price_name].shift(1))
    df['TR'] = df[['High-Low', 'High-PrevClose', 'Low-PrevClose']].max(axis=1)
    df['UpMove'] = df[high_name] - df[high_name].shift(1)
    df['DownMove'] = df[low_name].shift(1) - df[low_name]
    df[adx_name + '_dm_pos'] = np.where((df['UpMove'] > df['DownMove']) & (df['UpMove'] > 0), df['UpMove'], 0)
    df[adx_name + '_dm_neg'] = np.where((df['DownMove'] > df['UpMove']) & (df['DownMove'] > 0), df['DownMove'], 0)
    df[adx_name + '_di_pos'] = 100 * (df[adx_name + '_dm_pos'].ewm(span=period, adjust=False).mean() / df['TR'].ewm(span=period, adjust=False).mean())
    df[adx_name + '_di_neg'] = 100 * (df[adx_name + '_dm_neg'].ewm(span=period, adjust=False).mean() / df['TR'].ewm(span=period, adjust=False).mean())
    df[adx_name + '_dx'] = 100 * (abs(df[adx_name + '_di_pos'] - df[adx_name + '_di_neg']) / (df[adx_name + '_di_pos'] + df[adx_name + '_di_neg']))
    df[adx_name] = df[adx_name + '_dx'].ewm(span=period, adjust=False).mean()

def caclulateAroon(df, low_name, high_name, aroon_name, lookback_period=14): #Aroon Oscillator
    df[aroon_name] = 0
    df['High_Rolling_Max'] = df[high_name].rolling(lookback_period).max()
    df['Low_Rolling_Min'] = df[low_name].rolling(lookback_period).min()
    df['Periods_Since_Highest'] = df[high_name].rolling(lookback_period).apply(lambda x: lookback_period - np.argmax(x), raw=True)
    df['Periods_Since_Lowest'] = df[low_name].rolling(lookback_period).apply(lambda x: lookback_period - np.argmin(x), raw=True)
    df['Aroon_Up'] = (lookback_period - df['Periods_Since_Highest']) / lookback_period * 100
    df['Aroon_Down'] = (lookback_period - df['Periods_Since_Lowest']) / lookback_period * 100
    df[aroon_name] = df['Aroon_Up'] - df['Aroon_Down']

def calculateMACD(df, price_name, macd_name): #Moving Average Convergence Divergence
    df[macd_name] = 0
    df['EMA_12'] = df[price_name].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df[price_name].ewm(span=26, adjust=False).mean()
    df['MACD_Line'] = df['EMA_12'] - df['EMA_26']
    df['MACD_Signal_Line'] = df['MACD_Line'].ewm(span=9, adjust=False).mean()
    df[macd_name] = df['MACD_Line'] - df['MACD_Signal_Line']

def calculateSO(df, price_name, low_name, high_name, so_name, lookback_period = 14): #Stochastic Oscillator
    df[so_name] = 0
    df['High_Rolling_Max'] = df[high_name].rolling(lookback_period).max()
    df['Low_Rolling_Min'] = df[low_name].rolling(lookback_period).min()
    df['%K'] = 100 * (df[price_name] - df['Low_Rolling_Min']) / (df['High_Rolling_Max'] - df['Low_Rolling_Min'])
    df[so_name] = df['%K'].rolling(window=3).mean()

def calculateBollingerBands(df, price_name, band_up_name, band_down_name, period=20):
    df['Middle Band'] = df[price_name].rolling(window=period).mean()
    df['Standard Deviation'] = df[price_name].rolling(window=period).std()
    df[band_up_name] = df['Middle Band'] + 2 * df['Standard Deviation']
    df[band_down_name] = df['Middle Band'] - 2 * df['Standard Deviation']

def calculateIchimoku(df, price_name, low_name, high_name, ichi_name, period = 9): #
    df[ichi_name + '_conversion'] = (df[high_name].rolling(window=period).max() + df[low_name].rolling(window=period).min()) / 2
    base_period = 3* period #given example was 26
    df[ichi_name + '_base'] = (df[high_name].rolling(window=base_period).max() + df[low_name].rolling(window=base_period).min()) / 2
    df[ichi_name + '_span_a'] = ((df[ichi_name + '_conversion'] + df[ichi_name + '_base']) / 2).shift(periods=base_period)
    senkou_period = base_period*2 #given example was 52
    df[ichi_name + '_span_b'] = ((df[high_name].rolling(window=senkou_period).max() + df[low_name].rolling(window=senkou_period).min()) / 2).shift(periods=base_period)
    df[ichi_name + '_span_lagging'] = df[price_name].shift(-base_period)

def calculateCCI(df, price_name, low_name, high_name, cci_name, period = 20): #Commodity Channel Index
    typical_prices = (df[high_name] + df[low_name] + df[price_name]) / 3
    sma = typical_prices.rolling(window=period).mean()
    mean_deviation = (typical_prices - sma).abs().rolling(window=period).mean()
    df[cci_name] = (typical_prices - sma) / (0.015 * mean_deviation)
