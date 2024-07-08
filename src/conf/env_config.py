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
    stop_loss: float | None = None
    
env_swap_all = EnvConfig(
    type= "swap",
    observations_contain= []
)

env_trade_all_no_fee = EnvConfig(
    type= "trade_all",
    observations_contain= ["networth_percent_this_trade", "in_position", "drawdown"],
    transaction_fee= 0 #this obviously makes the model trade more, and for 1h can double profits
)

env_trade_all_fee_002 = EnvConfig(
    type= "trade_all",
    observations_contain= ["networth_percent_this_trade", "in_position", "drawdown"],
    transaction_fee= 0.002
)
env_trade_all_fee_003 = EnvConfig(
    type= "trade_all",
    observations_contain= ["networth_percent_this_trade", "in_position", "drawdown"],
    transaction_fee= 0.003
)
env_trade_all_fee_004 = EnvConfig(
    type= "trade_all",
    observations_contain= ["networth_percent_this_trade", "in_position", "drawdown"],
    transaction_fee= 0.004
)
env_trade_all_fee_005 = EnvConfig(
    type= "trade_all",
    observations_contain= ["networth_percent_this_trade", "in_position", "drawdown"],
    transaction_fee= 0.005
)
env_trade_all_fee_01 = EnvConfig(
    type= "trade_all",
    observations_contain= ["networth_percent_this_trade", "in_position", "drawdown"],
    transaction_fee= 0.01
)

env_trade_all = EnvConfig(
    type= "trade_all",
    observations_contain= ["networth_percent_this_trade", "in_position", "drawdown"], #removing drawdown decreases scores, in_position is weird, because even with position it can get a good score
)
env_trade_all_sl = EnvConfig(
    type= "trade_all",
    observations_contain= ["networth_percent_this_trade", "in_position", "drawdown"],
    stop_loss=0.02
)
env_trade_all_sl_0005 = EnvConfig(
    type= "trade_all",
    observations_contain= ["networth_percent_this_trade", "in_position", "drawdown"],
    stop_loss=0.005
)
env_trade_all_sl_001 = EnvConfig(
    type= "trade_all",
    observations_contain= ["networth_percent_this_trade", "in_position", "drawdown"],
    stop_loss=0.01
)
env_trade_all_sl_002 = EnvConfig(
    type= "trade_all",
    observations_contain= ["networth_percent_this_trade", "in_position", "drawdown"],
    stop_loss=0.02
)
env_trade_all_sl_003 = EnvConfig(
    type= "trade_all",
    observations_contain= ["networth_percent_this_trade", "in_position", "drawdown"],
    stop_loss=0.03
)
env_trade_all_sl_004 = EnvConfig(
    type= "trade_all",
    observations_contain= ["networth_percent_this_trade", "in_position", "drawdown"],
    stop_loss=0.04
)
env_trade_all_sl_005 = EnvConfig(
    type= "trade_all",
    observations_contain= ["networth_percent_this_trade", "in_position", "drawdown"],
    stop_loss=0.05
)
env_trade_all_sl_010 = EnvConfig(
    type= "trade_all",
    observations_contain= ["networth_percent_this_trade", "in_position", "drawdown"],
    stop_loss=0.1
)
env_trade_all_tpsl = EnvConfig(
    type= "trade_all",
    observations_contain= [],
    take_profit=0.02,
    stop_loss=0.01
)
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
    return [env_trade_all_sl]
    # return [env_trade_all, env_trade_all_sl_005]
    
    # return [env_trade_all_no_fee, env_trade_all, env_trade_all_fee_002, env_trade_all_fee_003, env_trade_all_fee_004, env_trade_all_fee_005, env_trade_all_fee_01]
    # return [env_trade_all, env_trade_all_sl_0005, env_trade_all_sl_001, env_trade_all_sl_002, env_trade_all_sl_003, env_trade_all_sl_004, env_trade_all_sl_005, env_trade_all_sl_010]
    
def get_envs_simple2():
    return [env_trade_all_tpsl]

def get_envs_simple3():
    return [env_trade_all_no_fee, env_trade_all, env_trade_all_tpsl]

def get_envs_swaps():
    return [env_swap_all]

def get_envs_all():
    return [env_swap_all, env_trade_all, env_trade_percent_10, env_trade_position_100, env_trade_amount_1]
