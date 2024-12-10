from dataclasses import dataclass, field
from typing import List

@dataclass
class EnvConfig:
    type: str = "trade_all" # possible: ["swap", "trade_all", "trade_percent", "trade_position", "trade_amount"]
    initial_balance: int = 100000
    amount: float = 1
    observations_contain: List[str] = field(default_factory=[]) # possible ["networth_percent_this_trade", "in_position", "drawdown", "actions_made", "trades_won", "trades_lost", "win_ratio"]
    transaction_fee: float = 0.001
    take_profit: float | None = None
    trailing_take_profit: float | None = None
    stop_loss: float | None = None
    no_sell_action: bool = False
    
env_swap_all = EnvConfig(
    type= "swap",
    observations_contain= []
)

env_trade_all = EnvConfig(
    type= "trade_all",
    observations_contain= []
)
env_trade_all_sl = EnvConfig(
    type= "trade_all",
    observations_contain= ["networth_percent_this_trade", "in_position", "drawdown"], #removing drawdown decreases scores, in_position is weird, because even with position it can get a good score
    stop_loss=0.02
)
env_trade_all_tpsl = EnvConfig(
    type= "trade_all",
    observations_contain= ["networth_percent_this_trade", "in_position", "drawdown"],
    take_profit=0.02,
    stop_loss=0.02,
    no_sell_action=True
)
env_trend_predict = EnvConfig(
    type= "trend_predict",
    observations_contain= []
)
env_dip_predict = EnvConfig(
    type= "dip_predict",
    observations_contain= []
)
# PROMISING
# 0.02|0.02|0.01
# 0.02|0.02|0.005
# 0.02|0.02|0.001 
# 0.01|0.01|0.005

# BAD
# 0.02|0.01|0.005
# 0.01|0.01|0.001

env_trade_all_tpsl_trailing0 = EnvConfig(type= env_trade_all_tpsl.type, observations_contain= env_trade_all_tpsl.observations_contain, take_profit= 0.01, stop_loss= 0.02, no_sell_action= env_trade_all_tpsl.no_sell_action, trailing_take_profit= 0.002)
env_trade_all_tpsl_trailing1 = EnvConfig(type= env_trade_all_tpsl.type, observations_contain= env_trade_all_tpsl.observations_contain, take_profit= 0.01, stop_loss= 0.02, no_sell_action= env_trade_all_tpsl.no_sell_action, trailing_take_profit= 0.001)
env_trade_all_tpsl_trailing2 = EnvConfig(type= env_trade_all_tpsl.type, observations_contain= env_trade_all_tpsl.observations_contain, take_profit= 0.01, stop_loss= 0.01, no_sell_action= env_trade_all_tpsl.no_sell_action, trailing_take_profit= 0.001)
env_trade_all_tpsl_trailing3 = EnvConfig(type= env_trade_all_tpsl.type, observations_contain= env_trade_all_tpsl.observations_contain, take_profit= 0.005, stop_loss= 0.01, no_sell_action= env_trade_all_tpsl.no_sell_action, trailing_take_profit= 0.001)
env_trade_all_tpsl_trailing4 = EnvConfig(type= env_trade_all_tpsl.type, observations_contain= env_trade_all_tpsl.observations_contain, take_profit= 0.002, stop_loss= 0.01, no_sell_action= env_trade_all_tpsl.no_sell_action, trailing_take_profit= 0.001)


env_trade_percent_10 = EnvConfig(
    type= "trade_percent",
    amount= 0.1,
    observations_contain= []
)
env_trade_position_100 = EnvConfig(
    type= "trade_position",
    amount= 100,
    observations_contain= []
)
env_trade_amount_1 = EnvConfig(
    type= "trade_amount",
    amount= 1,
    observations_contain= []
)

def get_envs_simple():
    # return [env_trend_predict]
    return [env_dip_predict]

    # return [env_trade_all_tpsl_trailing0]
    # return [env_trade_all_tpsl_trailing0, env_trade_all_tpsl_trailing1, env_trade_all_tpsl_trailing2, env_trade_all_tpsl_trailing3, env_trade_all_tpsl_trailing4]
    
def get_envs_swaps():
    return [env_swap_all]

def get_envs_all():
    return [env_swap_all, env_trade_all, env_trade_percent_10, env_trade_position_100, env_trade_amount_1]
