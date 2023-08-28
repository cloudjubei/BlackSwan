
import pandas as pd

from src.indicators.indicators_common import caclulateAroon, calculateADL, calculateADX, calculateBollingerBands, calculateCCI, calculateEMA, calculateIchimoku, calculateMACD, calculateOBV, calculateRSI, calculateSAR, calculateSO, calculateTrendDirection, calculateWilliams

def prepareIndicators(df: pd.DataFrame, trend_change_amount: float):
    calculateTrendDirection(df, 'price_open', 'price_close', 'price_low', 'price_high', 'trend_direction', trend_change_amount)
    calculateRSI(df, 'price_close', 'rsi', '24h_price_diff')    
    # calculateCCI(df, 'price_close', 'price_low', 'price_high', 'cci')
    # calculateSAR(df, 'price_low', 'price_high', 'sar')
    # calculateEMA(df, 'price_close', 'ma', 'ema')
    # calculateMACD(df, 'price_close', 'macd')

    df[['trend_direction', 'trend_direction_change', 'trend_direction_consecutive', 'trend_direction_swing', 'rsi', '24h_price_diff']] = df[['trend_direction', 'trend_direction_change', 'trend_direction_consecutive', 'trend_direction_swing', 'rsi', '24h_price_diff']].fillna(0)

def prepareAllIndicators(df: pd.DataFrame, trend_change_amount: float):
    calculateTrendDirection(df, 'price_open', 'price_close', 'price_low', 'price_high', 'trend_direction', trend_change_amount)
    calculateRSI(df, 'price_close', 'rsi', '24h_price_diff')    
    calculateCCI(df, 'price_close', 'price_low', 'price_high', 'cci')
    calculateWilliams(df, 'price_close', 'price_low', 'price_high', 'williams')
    calculateSAR(df, 'price_low', 'price_high', 'sar')
    calculateEMA(df, 'price_close', 'ma', 'ema')
    calculateOBV(df, 'price_close', 'volume', 'obv')
    calculateADL(df, 'price_close', 'volume', 'price_high', 'price_low', 'adl')
    calculateADX(df, 'price_close', 'price_low', 'price_high', 'adx')
    caclulateAroon(df, 'price_low', 'price_high', 'aroon')
    calculateMACD(df, 'price_close', 'macd')
    calculateSO(df, 'price_close', 'price_low', 'price_high', 'so')
    calculateBollingerBands(df, 'price_close', 'boll_up', 'boll_down')
    calculateIchimoku(df, 'price_close', 'price_low', 'price_high', 'ichi')

    cols = ['trend_direction', 'trend_direction_change', 'trend_direction_consecutive', 'rsi', '24h_price_diff', 'cci', 'williams', 'sar', 'ma', 'ema', 'obv', 'adl', 'adx', 'aroon', 'macd', 'so', 'boll_up', 'boll_down']
    df[cols] = df[cols].fillna(0)
    return df
