
import pandas as pd
import mplfinance as mpf

def plot_timeseries(dataframe: pd.DataFrame):
    df = dataframe.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

    df.set_index('timestamp', inplace=True)
    df.rename(columns={'price': 'close', 'price_open': 'open', 'price_high': 'high', 'price_low': 'low'}, inplace=True)

    buy_signals = mpf.make_addplot(df['buy_price'], type='scatter', markersize=100, marker='^', color='green')
    sell_signals = mpf.make_addplot(df['sell_price'], type='scatter', markersize=100, marker='v', color='red')

    mpf.plot(df, type='candle', style='charles', title='Candlestick Chart', ylabel='Price', addplot=[buy_signals, sell_signals])
