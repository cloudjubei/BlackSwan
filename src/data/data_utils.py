import numpy as np
import pandas as pd
import mplfinance as mpf

def plot_actions_data(paths, env):
    dfs = []
    
    for path in paths:
        df = pd.read_json(path)
        dfs.append(df)

    actions = env.actions[env.data_provider.get_lookback_window()-1:][:100]
    actions_made = env.actions_made[env.data_provider.get_lookback_window()-1:][:100]
    tpsls = env.tpsls[env.data_provider.get_lookback_window()-1:][:100]
    rewards = env.rewards_history[env.data_provider.get_lookback_window():][:100]
    buysignals_data = (np.array(env.data_provider.buy_signals[env.data_provider.get_start_index():-1])[:100] - 3) * -0.01
    result_df = pd.concat(dfs)

    start_index = env.data_provider.get_start_index()
    result_df = result_df.iloc[start_index:]
    
    result_df['open'] = result_df['price_open']
    result_df['close'] = result_df['price']
    result_df['low'] = result_df['price_low']
    result_df['high'] = result_df['price_high']
    result_df = result_df[['timestamp', 'open', 'close', 'low', 'high']].iloc[:-1].iloc[:100]
    result_df['timestamp'] = pd.to_datetime(result_df['timestamp'], unit='ms')
    result_df.set_index('timestamp', inplace=True)
    
    result_df['actions_made'] = np.array([1 if a else np.nan for a in actions_made]) * result_df['close']
    result_df['tpsls_sell_reverse'] = np.array([np.nan if a == -1 else 1 for a in tpsls])

    result_df['actions_buy'] = np.array([1 if a == 1 else np.nan for a in actions])
    result_df['actions_made_buy'] = result_df['actions_buy'] * result_df['actions_made'] * result_df['tpsls_sell_reverse']

    result_df['actions_sell'] = np.array([1 if a == 2 else np.nan for a in actions])
    result_df['actions_made_sell'] = result_df['actions_sell'] * result_df['actions_made'] #* result_df['tpsls_sell_reverse']

    result_df['tpsls_sell'] = np.array([1 if a == -1 else np.nan for a in tpsls])
    result_df['tpsls_made_sell'] = result_df['tpsls_sell'] * result_df['actions_made']

    # print(result_df.drop(columns=["actions_buy","actions_sell","tpsls_sell"]).head(40))

    buy_signals = mpf.make_addplot(result_df['actions_made_buy'], type='scatter', markersize=100, marker='^', color='blue', secondary_y=False)
    sell_signals = mpf.make_addplot(result_df['actions_made_sell'], type='scatter', markersize=100, marker='v', color='yellow', secondary_y=False)
    tpsl_sell_signals = mpf.make_addplot(result_df['tpsls_made_sell'], type='scatter', markersize=100, marker='v', color='orange', secondary_y=False)
    reward_signals = mpf.make_addplot(rewards, color='purple', ylabel='Rewards', secondary_y=True)
    buysignals_data_signals = mpf.make_addplot(buysignals_data, color='red', ylabel='Buy|Sell', type='scatter', secondary_y=True)

    mpf.plot(result_df, type='candle', style='charles', title='Buy Sell Signals', ylabel='Price', addplot=[buy_signals, sell_signals, tpsl_sell_signals, reward_signals, buysignals_data_signals])
    
    # env.render_profits()


def plot_indicator(df, indicator, name):
    result_df = df[['timestamp', 'price_open', 'price', 'price_low', 'price_high']].iloc[0:50]

    result_df['open'] = result_df['price_open']
    result_df['close'] = result_df['price']
    result_df['low'] = result_df['price_low']
    result_df['high'] = result_df['price_high']
    result_df['timestamp'] = pd.to_datetime(result_df['timestamp'], unit='ms')
    result_df.set_index('timestamp', inplace=True)
    
    indicator_signals = mpf.make_addplot(indicator[0:50], type='scatter', color='purple', ylabel=name, secondary_y=True)

    mpf.plot(result_df, type='candle', style='charles', title=f'{name} Signals', ylabel='Price', addplot=[indicator_signals])
    
    # env.render_profits()



