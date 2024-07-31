import asyncio

import numpy as np
import pandas as pd

from src.conf.data_config import DataConfig
from src.conf.env_config import EnvConfig
from src.conf.model_config import ModelConfig, ModelRLConfig
from src.environment.server_crypto_env import ServerCryptoEnv
from src.model.model_factory import create_model
from src.data.server_dataprovider import ServerDataProvider
from src.server.websocket.WebsocketServer import WebsocketServer
from src.server.websocket.WebsocketClient import WebsocketClient
import argparse

parser = argparse.ArgumentParser("run_server_model")
parser.add_argument("identifier", help="The identifier to use for outputting signals", type=str)
parser.add_argument("token_pair", help="The token pair to use for price input", type=str)
parser.add_argument("interval", help="The base interval", type=str)
parser.add_argument("lookback_window", help="The lookback window", type=int)
parser.add_argument("check_interval", help="The check interval", type=int)
parser.add_argument("port", help="The port to use", type=int)
args = parser.parse_args()


pd.options.display.float_format = '{:.3f}'.format
np.set_printoptions(formatter={'float_kind':'{:.4f}'.format})

identifier = args.identifier
tokenPair = args.token_pair
interval = args.interval
lookback_window = args.lookback_window
seconds_to_wait = args.check_interval
emitter_port = args.port

latest_price = "0"
# model_path = "trials/rl_dqn_combo_all_0.01_0.995_Adamax_CELU_[64, 64]_1_1716187086.833314"
model_path = "checkpoints/rl_dqn_combo_all_0.005_1000_4_1000000_0.995_Adamax_CELU_[64, 64, 64, 64]_1_1716768803.222076"
device = 'cpu'

env_config = EnvConfig(observations_contain= ["networth_percent_this_trade", "in_position", "drawdown"], stop_loss=0.02)

data_config = DataConfig(id="latest", train_data_paths= [], test_data_paths= [], layers= ["1h", "1d"])
data_provider = ServerDataProvider(data_config)
env = ServerCryptoEnv(env_config, data_provider, device)

model_config = ModelConfig(
    model_type= "rl",
    model_rl= ModelRLConfig(
        model_name= "dqn",
        reward_model= "combo_all",
        learning_rate= 0.01,
        batch_size= 128,
        buffer_size = 1_000,
        gamma = 0.995,
        tau = 0,
        exploration_fraction = 0.05,
        exploration_final_eps = 0.1,
        learning_starts= 10_000,
        train_freq= 8,
        gradient_steps= -1,
        target_update_interval= 10_000,
        max_grad_norm= 1000,
        optimizer_class = 'Adamax',
        optimizer_eps = 0.001,
        optimizer_weight_decay = 0.00001,
        optimizer_centered = True,
        optimizer_alpha = 0.9,
        optimizer_momentum = 0.0001, 
        activation_fn= 'CELU',
        net_arch= [64,64],

        episodes= 1,

        reward_multiplier_combo_noaction= 0.1,
        reward_multiplier_combo_positionprofitpercentage= 0.001,
        reward_multiplier_combo_buy= 1,
        reward_multiplier_combo_sell= 100,

        reward_multiplier_combo_sell_profit= 100,
        reward_multiplier_combo_sell_profit_prev= 100,
        reward_multiplier_combo_sell_perfect= 0.01,
        reward_multiplier_combo_sell_drawdown= 0.01,
        reward_multiplier_combo_buy_profit= 100,
        reward_multiplier_combo_buy_perfect= 0.01,
        reward_multiplier_combo_buy_profitable_offset= 5,
        reward_multiplier_combo_buy_profitable= 0.01,
        reward_multiplier_combo_buy_drawdown= 0.01,
        reward_multiplier_combo_hold_profit= 100,
        reward_multiplier_combo_hold_drawdown= 0.01,

        checkpoints_folder='trials/',
        checkpoint_to_load='rl_dqn_combo_all_0.01_0.995_Adamax_CELU_[64, 64]_1_1716187086.833314'
    )
)
rl_model = create_model(model_config, env, device)

client = WebsocketClient()
server = WebsocketServer(identifier, interval, emitter_port)

async def process_signal(signals_data):
    observation = env.create_observation(signals_data)
    if observation:
        action = rl_model.predictOnline(observation)
        await server.emit_signal(tokenPair, action)

async def periodic_emit(seconds_to_wait: int = 5):
    while True:
        if client.is_connected:
            await client.ask_signal(tokenPair, interval, lookback_window, callback= process_signal)
            # await client.ask_price(tokenPair, interval, callback= process_signal)
        print(f'Emit waiting: {seconds_to_wait}s')
        await asyncio.sleep(seconds_to_wait)

def store_price(price):
    print(f"Price of {tokenPair} : {price}")
    data_provider.store_price(price)

async def main():
    server_task = asyncio.ensure_future(server.start())
    client_task = asyncio.ensure_future(client.start())

    # TODO: just peridocailly ask for price/signals
    # client.listen_to_price(tokenPair, store_price)

    asyncio.create_task(periodic_emit(seconds_to_wait))

    await asyncio.gather(server_task, client_task)

if __name__ == "__main__":
    asyncio.run(main())