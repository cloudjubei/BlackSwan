import json
import os
import numpy as np
import pandas as pd
import mplfinance as mpf

def plot_actions_data(env):
    
    # dfs = []
    
    # for path in paths:
    #     df = pd.read_json(path)
    #     dfs.append(df)
    # result_df = pd.concat(dfs)

    end_point = 26 * 24 * 60
    # end_point = 3 * 26 * 24 * 60
    actions = env.actions[:end_point]
    actions_made = env.actions_made[:end_point]
    tpsls = env.tpsls[:end_point]
    rewards = env.rewards_history[:end_point]

    # actions = env.actions[env.data_provider.get_lookback_window()-1:][:end_point]
    # actions_made = env.actions_made[env.data_provider.get_lookback_window()-1:][:end_point]
    # tpsls = env.tpsls[env.data_provider.get_lookback_window()-1:][:end_point]
    # rewards = env.rewards_history[env.data_provider.get_lookback_window():][:end_point]
    # buysignals_data = (np.array(env.data_provider.buy_signals[env.data_provider.get_start_index():-1])[:end_point] - 3) * -0.01

    result_df = env.data_provider.get_raw_df_for_plotting()
    
    result_df['open'] = result_df['price_open']
    result_df['close'] = result_df['price']
    result_df['low'] = result_df['price_low']
    result_df['high'] = result_df['price_high']
    result_df = result_df[['timestamp', 'open', 'close', 'low', 'high']].iloc[:end_point]
    result_df['timestamp'] = pd.to_datetime(result_df['timestamp'], unit='ms')
    result_df.set_index('timestamp', inplace=True)
    
    result_df['actions_made'] = np.array([1 if a else np.nan for a in actions_made]) * result_df['close']
    result_df['tpsls_sell_reverse'] = np.array([np.nan if (a == -1 or a == 1) else 1 for a in tpsls])

    result_df['actions_buy'] = np.array([1 if a == 1 else np.nan for a in actions])
    result_df['actions_made_buy'] = result_df['actions_buy'] * result_df['actions_made'] * result_df['tpsls_sell_reverse']
    result_df['actions_buy_total'] = result_df['actions_buy'] * result_df['close']

    result_df['actions_sell'] = np.array([1 if a == 2 else np.nan for a in actions])
    result_df['actions_made_sell'] = result_df['actions_sell'] * result_df['actions_made'] * result_df['tpsls_sell_reverse']

    result_df['tpsls_sl'] = np.array([1 if a == -1 else np.nan for a in tpsls])
    result_df['tpsls_made_sl'] = result_df['tpsls_sl'] * result_df['actions_made']
    result_df['tpsls_tp'] = np.array([1 if a == 1 else np.nan for a in tpsls])
    result_df['tpsls_made_tp'] = result_df['tpsls_tp'] * result_df['actions_made']

    # print(result_df.drop(columns=["actions_buy","actions_sell","tpsls_sell"]).head(40))

    buy_signals = mpf.make_addplot(result_df['actions_made_buy'], type='scatter', markersize=100, marker='^', color='blue', secondary_y=False)
    sell_signals = mpf.make_addplot(result_df['actions_made_sell'], type='scatter', markersize=100, marker='v', color='yellow', secondary_y=False)
    tpsl_sl_signals = mpf.make_addplot(result_df['tpsls_made_sl'], type='scatter', markersize=100, marker='v', color='orange', secondary_y=False)
    tpsl_tp_signals = mpf.make_addplot(result_df['tpsls_made_tp'], type='scatter', markersize=100, marker='v', color='purple', secondary_y=False)
    reward_signals = mpf.make_addplot(rewards, color='purple', ylabel='Rewards', secondary_y=True)
    # buysignals_data_signals = mpf.make_addplot(buysignals_data, color='red', ylabel='Buy|Sell', type='scatter', secondary_y=True)
    buy_attempts = mpf.make_addplot(result_df['actions_buy_total'], type='scatter', markersize=60, marker='^', color='green', secondary_y=False)

    # mpf.plot(result_df, type='candle', style='charles', title='Buy Sell Signals', ylabel='Price', addplot=[buy_signals, sell_signals, tpsl_sl_signals, tpsl_tp_signals, reward_signals])
    mpf.plot(result_df, type='candle', style='charles', title='Buy Sell Signals', ylabel='Price', addplot=[buy_signals, sell_signals, tpsl_sl_signals, tpsl_tp_signals, reward_signals, buy_attempts])
    # mpf.plot(result_df, type='candle', style='charles', title='Buy Sell Signals', ylabel='Price', addplot=[buy_signals, sell_signals, tpsl_sell_signals, reward_signals, buysignals_data_signals])
    
    # env.render_profits()

def plot_actions_data_non_trade(env):
    
    end_point = 26 * 24 * 60
    # end_point = 3 * 26 * 24 * 60
    actions = env.actions[:end_point]
    rewards = env.rewards_history[:end_point]
    accuracies = env.accuracies[:end_point]

    result_df = env.data_provider.get_raw_df_for_plotting()
    # print(result_df.head())
    
    result_df['open'] = result_df['price_open']
    result_df['close'] = result_df['price']
    result_df['low'] = result_df['price_low']
    result_df['high'] = result_df['price_high']
    result_df = result_df[['timestamp', 'open', 'close', 'low', 'high', 'volume']].iloc[:-1].iloc[env.data_provider.get_lookback_window()-1:end_point]
    result_df['timestamp'] = pd.to_datetime(result_df['timestamp'], unit='ms')
    result_df.set_index('timestamp', inplace=True)
    
    print('len of results:', len(result_df), ' len of actions: ', len(actions))



    result_df['rewards_pos'] = np.array([1 if a > 0 else np.nan for a in rewards])
    result_df['rewards_neg'] = np.array([1 if a < 0 else np.nan for a in rewards])
    result_df['actions_ok'] = np.array([1 if a > 0 else np.nan for a in actions]) * result_df['rewards_pos'] * result_df['low']
    result_df['actions_bad'] = np.array([1 if a > 0 else np.nan for a in actions]) * result_df['rewards_neg'] * result_df['high']

    # result_df['actions_ok'] = np.array([1 if a > 0 else np.nan for a in actions]) * result_df['low']

    action_signals_ok = mpf.make_addplot(result_df['actions_ok'], type='scatter', markersize=100, marker='^', color='blue', secondary_y=False)
    action_signals_bad = mpf.make_addplot(result_df['actions_bad'], type='scatter', markersize=100, marker='v', color='red', secondary_y=False)
    # reward_signals = mpf.make_addplot(rewards, color='purple', ylabel='Rewards', secondary_y=True)
    # accuracies_signals = mpf.make_addplot(accuracies, color='yellow', ylabel='Rewards', secondary_y=True)

    mpf.plot(result_df, type='candle', style='charles', title='Action Signals', ylabel='Price', addplot=[action_signals_ok, action_signals_bad])
    # mpf.plot(result_df, type='candle', style='charles', title='Action Signals', ylabel='Price', addplot=[action_signals, reward_signals, accuracies_signals])

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

def fill_missing_timestamps(folder='binance_new'):
    new_folder = 'binance2'
    for filename in os.listdir(new_folder):
        if filename.endswith(".json"):
            file_path = os.path.join(new_folder, filename)
            with open(file_path, 'r') as file:
                data = json.load(file)

                new_filepath = os.path.join(folder, "SOLUSDT-1m-" + filename)
                with open(new_filepath, 'w') as file2:
                    json.dump(data, file2, default=str, separators=(',', ':'))

    for filename in os.listdir(folder):
        if filename.endswith(".json"):
            file_path = os.path.join(folder, filename)
            with open(file_path, 'r') as file:
                data = json.load(file)

            time_diff = data[0]['timestamp_close']-data[0]['timestamp']+1
            copy = data.copy()
            added = 0
            for i in range(len(data)-1):
                if data[i]['timestamp_close'] + 1 != data[i+1]['timestamp']:
                    data[i]['timestamp_close'] = data[i]['timestamp']+ time_diff-1
                    copy[i+added]['timestamp_close'] = copy[i+added]['timestamp']+ time_diff-1
                    total_diff = data[i+1]['timestamp'] - (data[i]['timestamp_close'] + 1)
                    missing_items = int(total_diff / time_diff)
                    print('FOUND INEQUALITY at: ', i, ' missing_items: ', missing_items, ' filename: ', filename)
                    for j in range(0, missing_items):
                        new_item = data[i].copy()
                        new_item['timestamp'] = int(data[i]['timestamp_close']) + 1 + int(time_diff*j)
                        new_item['timestamp_close'] = int(new_item['timestamp']) + time_diff-1
                        new_item['price_low'] = new_item['price']
                        new_item['price_high'] = new_item['price']
                        new_item['volume'] = 0
                        added += 1
                        copy.insert(i+added, new_item)
                elif i == len(data)-1:
                    fullday = 1000*60*60*24
                    timestamp = int(data[i]['timestamp_close'] + 1)
                    while timestamp % fullday != 0:
                        print('FOUND NON-ENDING timestamp at: ', i,' filename: ', filename)
                        new_item = data[i].copy()
                        new_item['timestamp'] = timestamp
                        timestamp = timestamp + time_diff
                        new_item['timestamp_close'] = timestamp - 1
                        new_item['price_low'] = new_item['price']
                        new_item['price_high'] = new_item['price']
                        new_item['volume'] = 0
                        added += 1
                        copy.insert(i+added, new_item)



            with open(file_path, 'w') as file:
                json.dump(copy, file, default=str, separators=(',', ':'))