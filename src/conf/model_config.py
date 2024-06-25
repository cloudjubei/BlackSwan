from dataclasses import dataclass, field
from typing import List

@dataclass
class ModelRLConfigSearch:
    model_name: List[str] = field(default_factory=[]), # possible ["ppo", "a2c", "dqn"]
    reward_model: List[str] = field(default_factory=[]) # possible ["combo_all", "state_reflective", "win_loss_trade", "trade_percentprofit"],
    learning_rate: List[float] = field(default_factory=[]) 
    batch_size: List[int] = field(default_factory=[]) 
    buffer_size: List[int] = field(default_factory=[]) 
    gamma: List[float] = field(default_factory=[]) 
    tau: List[float] = field(default_factory=[])
    exploration_fraction: List[float] = field(default_factory=[]) 
    exploration_final_eps: List[float] = field(default_factory=[]) 
    learning_starts: List[int] = field(default_factory=[]) 
    train_freq: List[int] = field(default_factory=[]) 
    gradient_steps: List[int] = field(default_factory=[]) 
    target_update_interval: List[int] = field(default_factory=[]) 
    max_grad_norm: List[float] = field(default_factory=[]) 
    optimizer_class: List[str] = field(default_factory=[])
    optimizer_eps: List[float] = field(default_factory=[]) 
    optimizer_weight_decay: List[float] = field(default_factory=[]) 
    optimizer_centered: List[bool] = field(default_factory=[])
    optimizer_alpha: List[float] = field(default_factory=[]) 
    optimizer_momentum: List[float] = field(default_factory=[]) 
    activation_fn: List[str] = field(default_factory=[])
    net_arch: List[List[int]] = field(default_factory=[])

    episodes: List[int] = field(default_factory=[])

    reward_multiplier_combo_noaction: List[float] = field(default_factory=[])
    reward_multiplier_combo_positionprofitpercentage: List[float] = field(default_factory=[])
    reward_multiplier_combo_buy: List[float] = field(default_factory=[])
    reward_multiplier_combo_sell: List[float] = field(default_factory=[])
    reward_multiplier_buy_sell_sold: List[float] = field(default_factory=[])
    reward_multiplier_buy_sell_bought: List[float] = field(default_factory=[])
    reward_multiplier_combo_actions_sell_hit: List[float] = field(default_factory=[])
    reward_multiplier_combo_actions_sell_miss: List[float] = field(default_factory=[])
    reward_multiplier_combo_actions_buy_threshold: List[float] = field(default_factory=[])
    reward_multiplier_combo_actions_buy_hit: List[float] = field(default_factory=[])
    reward_multiplier_combo_actions_buy_miss: List[float] = field(default_factory=[])
    reward_multiplier_combo_actions_hold_buy_threshold: List[float] = field(default_factory=[])
    reward_multiplier_combo_actions_hold_cash_hit: List[float] = field(default_factory=[])
    reward_multiplier_combo_actions_hold_cash_miss: List[float] = field(default_factory=[])
    reward_multiplier_combo_actions_hold_position_hit: List[float] = field(default_factory=[])
    reward_multiplier_combo_actions_hold_position_miss: List[float] = field(default_factory=[])

    progress_bar: bool = True
    checkpoints_folder: str = 'checkpoints/'
    checkpoint_to_load: str | None = None
@dataclass
class ModelRLConfig:
    model_name: str
    reward_model: str
    learning_rate: float = 0.0001
    batch_size: int = 32
    buffer_size: int = 1_000_000
    gamma: float = 0.99
    tau: float = 1.0
    exploration_fraction: float = 0.1
    exploration_final_eps: float = 0.05
    learning_starts: int = 50000
    train_freq: int = 4
    gradient_steps: int = 1
    target_update_interval: int = 10000
    max_grad_norm: float = 10
    optimizer_class: str = 'Adam'
    optimizer_eps: float = 0.00000001
    optimizer_weight_decay: float = 0
    optimizer_centered: bool = False
    optimizer_alpha: float = 0.99
    optimizer_momentum: float = 0
    activation_fn: str = 'ReLU'
    net_arch: List[int] = field(default_factory=[])

    episodes: int = 1

    reward_multiplier_combo_noaction: float = 0
    reward_multiplier_combo_positionprofitpercentage: float = 0
    reward_multiplier_combo_buy: float = 0
    reward_multiplier_combo_sell: float = 0
    reward_multiplier_buy_sell_sold: float = 1
    reward_multiplier_buy_sell_bought: float = 0
    reward_multiplier_combo_actions_sell_hit: float = 0
    reward_multiplier_combo_actions_sell_miss: float = 0
    reward_multiplier_combo_actions_buy_threshold: float = 0
    reward_multiplier_combo_actions_buy_hit: float = 0
    reward_multiplier_combo_actions_buy_miss: float = 0
    reward_multiplier_combo_actions_hold_buy_threshold: float = 0
    reward_multiplier_combo_actions_hold_cash_hit: float = 0
    reward_multiplier_combo_actions_hold_cash_miss: float = 0
    reward_multiplier_combo_actions_hold_position_hit: float = 0
    reward_multiplier_combo_actions_hold_position_miss: float = 0

    progress_bar: bool = True
    checkpoints_folder: str = 'checkpoints/'
    checkpoint_to_load: str | None = None
@dataclass
class ModelLSTMConfigSearch:
    # loss_fn: List[str] = field(default_factory=[]), # possible ["bce", "mse", "r2", "sqm"]
    learning_rate: List[float] = field(default_factory=[]) 
    weight_decay: List[float] = field(default_factory=[]) 
    episodes: List[int] = field(default_factory=[])

    hidden_size: List[int] = field(default_factory=[])
    layers: List[int] = field(default_factory=[])
    lstm_dropout: List[float] = field(default_factory=[]) 
    extra_dropout: List[float] = field(default_factory=[]) 
    first_activation: List[str] = field(default_factory=[]), # possible ["relu", "lrelu", "sigmoid"]
    last_activation: List[str] = field(default_factory=[]), # possible ["relu", "lrelu", "sigmoid"]
@dataclass
class ModelLSTMConfig:
    loss_fn: str = 'mse'
    learning_rate: float = 0.0001
    weight_decay: float = 0.0001
    episodes: int = 1

    hidden_size: int = 16
    layers: int = 1
    lstm_dropout: float = 0.5
    extra_dropout: float = 0.5
    first_activation: str = 'lrelu'
    last_activation: str = 'sigmoid'

    progress_bar: bool = True
    checkpoints_folder: str = 'checkpoints/'
    checkpoint_to_load: str | None = None
@dataclass
class ModelMLPConfigSearch:
    # loss_fn: List[str] = field(default_factory=[]), # possible ["bce", "mse", "r2", "sqm"]
    learning_rate: List[float] = field(default_factory=[]) 
    weight_decay: List[float] = field(default_factory=[]) 
    episodes: List[int] = field(default_factory=[])

    hidden_dim1: List[int] = field(default_factory=[])
    hidden_dim2: List[int] = field(default_factory=[])
    first_activation: List[str] = field(default_factory=[]), # possible ["relu", "lrelu", "sigmoid"]
    last_activation: List[str] = field(default_factory=[]), # possible ["relu", "lrelu", "sigmoid"]
@dataclass
class ModelMLPConfig:
    loss_fn: str = 'mse'
    learning_rate: float = 0.0001
    weight_decay: float = 0.0001
    episodes: int = 1

    hidden_dim1: int = 64
    hidden_dim2: int = 32
    first_activation: str = 'relu'
    last_activation: str = 'relu'

    progress_bar: bool = True
    checkpoints_folder: str = 'checkpoints/'
    checkpoint_to_load: str | None = None
@dataclass
class ModelTimeConfigSearch:
    time_buy: List[int] = field(default_factory=[1200])
    time_sell: List[int] = field(default_factory=[1400])
@dataclass
class ModelTimeConfig:
    time_buy: int
    time_sell: int
@dataclass
class ModelTechnicalConfigSearch:
    buy_indicator: List[str] = field(default_factory=[])
    buy_amount_threshold: List[float] = field(default_factory=[])
    sell_indicator: List[str] = field(default_factory=[])
    sell_amount_threshold: List[float] = field(default_factory=[])
    buy_amount_is_multiplier: bool = False
    buy_is_price_check: bool = False
    buy_is_down_check: bool = True
    sell_amount_is_multiplier: bool = False
    sell_is_price_check: bool = False
    sell_is_up_check: bool = True
@dataclass
class ModelTechnicalConfig:
    buy_indicator: str
    buy_amount_threshold: float
    sell_indicator: str
    sell_amount_threshold: float
    buy_amount_is_multiplier: bool = False
    buy_is_price_check: bool = False
    buy_is_down_check: bool = True
    sell_amount_is_multiplier: bool = False
    sell_is_price_check: bool = False
    sell_is_up_check: bool = True
@dataclass
class ModelConfigSearch:
    model_type: str # possible ["hodl", "rl", "lstm", "mlp", "technical", "time"]
    model_rl: ModelRLConfigSearch | None = None
    model_lstm: ModelLSTMConfigSearch | None = None
    model_mlp: ModelMLPConfigSearch | None = None
    model_technical: ModelTechnicalConfigSearch | None = None
    model_time: ModelTimeConfigSearch | None = None
@dataclass
class ModelConfig:
    model_type: str # possible ["hodl", "rl", "lstm", "mlp", "technical", "time"]
    iterations_to_pick_best: int = 1
    model_rl: ModelRLConfig | None = None
    model_lstm: ModelLSTMConfig | None = None
    model_mlp: ModelMLPConfig | None = None
    model_technical: ModelTechnicalConfig | None = None
    model_time: ModelTimeConfig | None = None
    
    def is_deep(self) -> bool:
        return self.model_type == "rl" or self.model_type == "lstm" or self.model_type == "mlp"
    def is_hodl(self) -> bool:
        return self.model_type == "hodl"
    
# only_price_percent is the best
# DQN lr - between 0.005, 0.01
# it seems DQN > QRDQN > A2C
# A2C becomes better with more steps (on 1h, managed to get 69k compound, but 5/1 trades, 32 lookback)

# COMBO_ALL
# 0.1	0	0.1	    100
# 0.01	0	0.01	100
# 0.1	0	0.01	100
# 0.01	0	1	    1 <-- 69k
# 0.1	0	1	    10 <--- 99/4 WIN ratio

# data with timestamp none - best; day_of_week also gives positive results, but doesn't improve
# data H with additional D seems to improve results
# lookback makes everything better, but not too much of it
# flattening data doesn't impact anything
# min+h only 2023 can get even 57k profit (70k compounded), but makes very few trades (4-7). HAS to use small LR=0.000001
# not having volume - only_price_percent_sin_volume - can yield perhaps better results

# RMSprop x Relu6 [64,64,64,64] => 1H

# 1min 2023vs2024
# Adamax x Relu6 [64,64] => 1min - 16813 profit, 37713 Trade$, 45739 Compound, 209/0
# RMSprop x Relu6 [64,64] => 1min
# Adamax x PReLU [64,64] => 1min
# RMSprop x PReLU [64,64] => 1min
# Adamax x PReLU [64,64,64,64] => 1min 274/0

# Indicators that seem to help:
# RSI9, SMA30, Williams30, Bollinger20SD, saar005_03 
# percent_buysell -> 0.001, 0.003, 0.01  

# duel_dqn seems to get much better rewards than dqn, yet much lower profits on combo_actions_diff
# drawdown + profit percentage seems to be an ok indicator -> the model definitely learns something
# profit_percentage2 + profit_percentage3 + profit_percentage4 -> are all ok indicators, the model learns from them
# buy_sell_signal + buy_sell_signal2 -> are ok


model_rl_h = ModelConfigSearch(
    model_type= "rl",
    model_rl= ModelRLConfigSearch(
        # model_name= ["ppo", "reppo", "trpo", "a2c", "ars", "ars-mlp", "qrdqn", "dqn-sbx", "dqn"],
        # model_name= ["dqn"],
        # model_name= ["rainbow-dqn", "dqn-lstm", "iqn", "dqn"], # very slow, but can work very well
        model_name= ["duel-dqn"],

        # 5.3.1
        # model_name= ["ppo", "a2c", "qrdqn", "dqn"],

        # reward_model= ["combo_all", "combo_actions", "buy_sell_signal", "drawdown", "profit_percentage", "profit_percentage3"],
        # reward_model= ["combo_all"],
        reward_model= ["buy_sell_signal2"],
        # reward_model= ["buy_sell_signal2"],
        # reward_model= ["buy_sell_signal", "buy_sell_signal2", "buy_sell_signal3", "buy_sell_signal4"],
        # reward_model= ["drawdown"],
        # reward_model= ["profit_percentage"],
        # reward_model= ["profit_percentage3"],
        # reward_model= ["profit_percentage", "profit_percentage2", "profit_percentage3", "profit_percentag4"],
        # reward_model= ["combo_all", "combo_all2", "buy_sell_signal", "buy_sell_signal2", "buy_sell_signal3", "buy_sell_signal4", "combo_actions", "combo_actions2", "combo_actions3"],
        
        # learning_rate= [0.0001, 0.05],
        learning_rate= [0.0001],
        # batch_size= [4, 256],
        batch_size= [256],
        # buffer_size = [10, 100_000, 1_000_000],
        buffer_size = [1_000_000],
        # gamma = [0.9, 0.95],
        gamma = [0.9],
        # tau = [0, 0.9],
        tau = [0.9],
        # exploration_fraction = [0.05, 0.9],
        exploration_fraction = [0.05],
        # exploration_final_eps = [0.01, 0.05, 0.1],
        exploration_final_eps = [0.1],
        # learning_starts= [1, 1_000, 10_000, 20_000, 40_000, 50_000],
        learning_starts= [10_000],
        # train_freq= [2, 4, 8],
        train_freq= [8],
        # gradient_steps= [-1, 1, 2],
        gradient_steps= [-1],
        target_update_interval= [10_000],
        # max_grad_norm= [1, 1000],
        max_grad_norm= [1000],

        # 5.3.2
        # optimizer_class = ['Adam', 'Adagrad', 'Adamax', 'RMSprop', 'SGD', 'LBFGS'],

        # optimizer_class = ['Adamax', 'RMSprop'],
        optimizer_class = ['Adamax'],
        # optimizer_eps = [0.000001, 0.1, 0.2, 0.5],
        optimizer_eps = [0.5],
        # optimizer_weight_decay = [0.00000001, 0.0001],
        optimizer_weight_decay = [0.00000001],
        # optimizer_centered = [True, False],
        optimizer_centered = [True],
        optimizer_alpha = [0.9],
        # optimizer_momentum = [0.0001, 0.1, 0.000001],
        optimizer_momentum = [0.0001], 

        # 5.3.3
        # activation_fn = ['ReLU', 'LeakyReLU', 'CELU', 'Softsign', 'Sigmoid', 'Tanh'],

        # activation_fn = ['LogSigmoid', 'CELU', 'PReLU', 'ReLU6', 'Softsign'],
        # activation_fn = ['PReLU', 'ReLU6'],
        activation_fn= ['CELU'],
        # net_arch= [[256,256], [1024,1024], [512,512,512,512], [64,64,64,64,64,64]],
        # net_arch= [[1024,1024,1024,1024,1024], [512,512,512,512,512]],
        net_arch= [[64,64,64,64]],
        # net_arch= [[64,64]],

        # episodes= [1, 3, 5],
        episodes= [1],

        # reward_multiplier_combo_noaction= [-1, -10, -100],
        reward_multiplier_combo_noaction= [-1],
        # reward_multiplier_combo_positionprofitpercentage= [0.001, 0.01, 10],
        # reward_multiplier_combo_positionprofitpercentage= [0.01, 10], <--- profit_percentage3
        reward_multiplier_combo_positionprofitpercentage= [10],
        # reward_multiplier_combo_buy= [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
        # reward_multiplier_combo_buy= [0.01, 0.1, 1, 10, 100],
        reward_multiplier_combo_buy= [0.1],
        # reward_multiplier_combo_sell= [0.1, 1, 10, 100, 1000, 10000],
        reward_multiplier_combo_sell= [1000],

        # reward_multiplier_buy_sell_sold= [0.1, 1, 10, 100, 1000, 10000],
        reward_multiplier_buy_sell_sold= [100],
        # reward_multiplier_buy_sell_bought= [0.01, 0.1, 1, 10],
        reward_multiplier_buy_sell_bought= [0.1],

        # reward_multiplier_combo_actions_sell_hit= [0.1, 0.5, 1],
        reward_multiplier_combo_actions_sell_hit= [10],
        reward_multiplier_combo_actions_sell_miss= [-1],
        # reward_multiplier_combo_actions_buy_threshold = [0, 3, 5], # no diff
        reward_multiplier_combo_actions_buy_threshold= [2],
        # reward_multiplier_combo_actions_buy_hit= [0.6, 1],
        reward_multiplier_combo_actions_buy_hit= [0.01],
        # reward_multiplier_combo_actions_buy_miss= [-0.0001],
        reward_multiplier_combo_actions_buy_miss= [-0.0001],
        # reward_multiplier_combo_actions_hold_buy_threshold = [1, 5],
        reward_multiplier_combo_actions_hold_buy_threshold = [2],
        # reward_multiplier_combo_actions_hold_cash_hit= [0.0001, 0.001],
        reward_multiplier_combo_actions_hold_cash_hit= [0.1],
        # reward_multiplier_combo_actions_hold_cash_miss= [-0.001, -0.01],
        reward_multiplier_combo_actions_hold_cash_miss= [1],
        # reward_multiplier_combo_actions_hold_position_hit= [0.0001],
        reward_multiplier_combo_actions_hold_position_hit= [0.1],
        # reward_multiplier_combo_actions_hold_position_miss= [-0.01, -10],
        reward_multiplier_combo_actions_hold_position_miss= [-0.1],

        # checkpoints_folder='checkpoints/',
        # checkpoint_to_load='rl_dqn_combo_actions_diff_0.0001_10000_256_1000000_0.9_Adamax_CELU_[64, 64, 64, 64]_1_1717946749.513737'
    )
)

# [MIN] ["duel-dqn"] x lr[0.0001] x day_of_week x rewards 1/10
# [MIN] ["duel-dqn"] x lr[0.0000001] x day_of_week x rewards 1/10
# [MIN] ["duel-dqn"] x lr[0.0001] x [1m_1h_1d vs 1m_1h vs 1m] x combo_all2 3/6

# TODO:
# [MIN] ["duel-dqn"] x lr[0.0000001] x rewards_chosen -> day_of_week vs non 1/10

# day_of_week + lookback [7,64]
model_rl_min = ModelConfigSearch(
    model_type= "rl",
    model_rl= ModelRLConfigSearch(
        # model_name= ["ppo", "a2c", "dqn", "qrdqn", "dqn-sbx"],
        # model_name= ["duel-dqn", "dqn-sbx", "dqn"],
        # model_name= ["duel-dqn"],
        model_name= ["iqn"],
        # model_name= ["dqn"],
        # model_name= ["dqn-sbx"],

        # reward_model= ["profit_percentage2", "profit_percentage4", "combo_all", "combo_all2", "combo_actions", "combo_actions2", "combo_actions3", "buy_sell_signal2", "buy_sell_signal4"],
        reward_model= ["combo_all2"],
        # reward_model= ["combo_all"],
        # reward_model= ["combo_actions"],
        # reward_model= ["buy_sell_signal"],
        # reward_model= ["drawdown"],
        # reward_model= ["profit_percentage"],
        # reward_model= ["profit_percentage2"],
        # reward_model= ["profit_percentage4"],

        # learning_rate= [0.0000001, 0.0001],
        learning_rate= [0.0001],
        batch_size= [256],
        buffer_size = [1_000_000],
        gamma = [0.9],
        tau = [0.9],
        exploration_fraction = [0.05],
        exploration_final_eps = [0.1],
        learning_starts= [10_000],
        train_freq= [8],
        gradient_steps= [-1],
        target_update_interval= [10_000],
        max_grad_norm= [1000],
        optimizer_class = ['Adamax'],
        optimizer_eps = [0.5],
        optimizer_weight_decay = [0.00000001],
        optimizer_centered = [True],
        optimizer_alpha = [0.9],
        optimizer_momentum = [0.000001],
        activation_fn= ['CELU'],
        net_arch= [[64,64,64,64]], 

        episodes= [1],

        reward_multiplier_combo_noaction= [-1],
        reward_multiplier_combo_positionprofitpercentage= [10],
        reward_multiplier_combo_buy= [0.1],
        reward_multiplier_combo_sell= [1000],
        reward_multiplier_buy_sell_sold= [100],
        reward_multiplier_buy_sell_bought= [10],
        reward_multiplier_combo_actions_sell_hit= [10],
        reward_multiplier_combo_actions_sell_miss= [-1],
        reward_multiplier_combo_actions_buy_threshold= [2],
        reward_multiplier_combo_actions_buy_hit= [0.01],
        reward_multiplier_combo_actions_buy_miss= [-0.0001],
        reward_multiplier_combo_actions_hold_buy_threshold = [2],
        reward_multiplier_combo_actions_hold_cash_hit= [0.1],
        reward_multiplier_combo_actions_hold_cash_miss= [1],
        reward_multiplier_combo_actions_hold_position_hit= [0.1],
        reward_multiplier_combo_actions_hold_position_miss= [-0.1],
    )
)


model_lstm = ModelConfigSearch(
    model_type= "lstm",
    model_lstm= ModelLSTMConfigSearch(
        # learning_rate= [0.1, 0.01, 0.001, 0.0001],
        learning_rate= [0.001],
        # weight_decay= [0.0001, 0.001],
        weight_decay= [0.0001],
        # episodes= [1, 10, 20],
        episodes= [1],

        # hidden_size= [8, 16, 32, 64, 128],
        hidden_size= [128],
        # hidden_size= [16],
        # layers= [1, 2, 5, 10],
        layers= [10],
        # lstm_dropout= [0.0, 0.1, 0.5],
        lstm_dropout= [0.5],
        # extra_dropout= [0.0, 0.1, 0.5],
        extra_dropout= [0.5],
        # first_activation= ["relu", "lrelu", "sigmoid"],
        first_activation= ["relu"],
        # last_activation= ["relu", "lrelu", "sigmoid"],
        last_activation= ["relu"]
    )
)

model_mlp = ModelConfigSearch(
    model_type= "mlp",
    model_mlp= ModelMLPConfigSearch(
        # learning_rate= [0.001, 0.0001, 0.00001],
        learning_rate= [0.0001],
        # weight_decay= [0.00001, 0.0001],
        weight_decay= [0.0001],
        episodes= [1],
        # episodes= [1],

        # hidden_dim1= [32, 64, 512],
        hidden_dim1= [64],
        # hidden_dim2= [32, 64, 512],
        hidden_dim2= [64],
        # first_activation= ["relu", "lrelu", "sigmoid"],
        first_activation= ["relu"],
        # last_activation= ["relu", "lrelu", "sigmoid"],
        last_activation= ["relu"]
    )
)

model_hodl = ModelConfigSearch(
    model_type= "hodl"
)
model_time = ModelConfigSearch(
    model_type= "time",
    model_time= ModelTimeConfigSearch(
        time_buy=[600, 700, 800, 900, 1000, 1100, 1200, 1300],
        time_sell=[1400, 1500, 1600, 1700, 1800, 1900, 2000]
    )
)
model_time2 = ModelConfigSearch(
    model_type= "time",
    model_time= ModelTimeConfigSearch(
        time_buy=[1000,1100,1200,1300,1350],
        time_sell=[1400,1401,1405,1410,1430,1500,1505,1600]
    )
)
model_technical_bollinger= ModelConfigSearch(
    model_type= "technical",
    model_technical= ModelTechnicalConfigSearch(
        buy_indicator= ["bollinger20Low"],
        buy_amount_threshold= [1.01, 1.05, 1.1, 1.2, 1.3, 1.5],
        buy_amount_is_multiplier= True,
        buy_is_price_check= True,
        sell_indicator= ["bollinger20High"],
        sell_amount_threshold= [0.5, 0.7, 0.8, 0.9, 0.95, 0.99],
        sell_amount_is_multiplier= True,
        sell_is_price_check= True,
    )
)
model_technical_kallmanfilter5 = ModelConfigSearch(
    model_type= "technical",
    model_technical= ModelTechnicalConfigSearch(
        buy_indicator= ["kallman5Momentum"],
        buy_amount_threshold= [0.01, 0.1, 0.6, 1.0, 5.0],
        sell_indicator= ["kallman5Momentum"],
        sell_amount_threshold= [-0.01, -0.1, -0.6, -1.0, -5.0],
    )
)
model_technical_kallmanfilter14 = ModelConfigSearch(
    model_type= "technical",
    model_technical= ModelTechnicalConfigSearch(
        buy_indicator= ["kallman14Momentum"],
        buy_amount_threshold= [0.01, 0.1, 0.6, 1.0, 5.0],
        sell_indicator= ["kallman14Momentum"],
        sell_amount_threshold= [-0.01, -0.1, -0.6, -1.0, -5.0],
    )
)
model_technical_kallmanfilter20 = ModelConfigSearch(
    model_type= "technical",
    model_technical= ModelTechnicalConfigSearch(
        buy_indicator= ["kallman20Momentum"],
        buy_amount_threshold= [0.01, 0.1, 0.6, 1.0, 5.0],
        sell_indicator= ["kallman20Momentum"],
        sell_amount_threshold= [-0.01, -0.1, -0.6, -1.0, -5.0],
    )
)

model_time_test = ModelConfigSearch(
    model_type= "time",
    model_time= ModelTimeConfigSearch(
        time_buy=[1200],
        time_sell=[1400]
    )
)
model_technical_kallmanfilter_test = ModelConfigSearch(
    model_type= "technical",
    model_technical= ModelTechnicalConfigSearch(
        buy_indicator= ["kallman20Momentum"],
        buy_amount_threshold= [0.01],
        sell_indicator= ["kallman20Momentum"],
        sell_amount_threshold= [-0.01],
    )
)
model_technical_bollinger_test = ModelConfigSearch(
    model_type= "technical",
    model_technical= ModelTechnicalConfigSearch(
        buy_indicator= ["bollinger20Low"],
        buy_amount_threshold= [1.3],
        buy_amount_is_multiplier= True,
        buy_is_price_check= True,
        sell_indicator= ["bollinger20High"],
        sell_amount_threshold= [0.7],
        sell_amount_is_multiplier= True,
        sell_is_price_check= True,
    )
)

model_rl_h_win = ModelConfigSearch(
    model_type= "rl",
    model_rl= ModelRLConfigSearch(
        model_name= ["dqn"],
        reward_model= ["combo_all"],
        # learning_rate= [0.01, 0.05],
        learning_rate= [0.01],
        # batch_size= [4, 32, 64, 128],
        batch_size= [128],
        # buffer_size = [1_000, 10_000],
        buffer_size = [10_000],
        # gamma = [0.9, 0.95, 0.995],
        gamma = [0.995],
        # tau = [0, 0.25],
        tau = [0],
        # exploration_fraction = [0.05, 0.2],
        exploration_fraction = [0.05],
        # exploration_final_eps = [0, 0.1, 0.2],
        exploration_final_eps = [0.1],
        learning_starts= [1, 1_000, 10_000],
        # learning_starts= [1_000],
        # train_freq= [2, 4, 8],
        train_freq= [8],
        # gradient_steps= [-1, 1, 2],
        gradient_steps= [-1],
        target_update_interval= [10_000],
        # max_grad_norm= [1, 1000],
        max_grad_norm= [1000],
        optimizer_class = ['RMSprop'],
        # optimizer_eps = [0.001, 0.0001, 0.01, 0.0000000001],
        optimizer_eps = [0.001],
        # optimizer_weight_decay = [0.00001, 0.0000000001],
        optimizer_weight_decay = [0.00001],
        # optimizer_centered = [True, False],
        optimizer_centered = [True],
        optimizer_alpha = [0.9],
        # optimizer_momentum = [0.0001, 0.1, 0.000001],
        optimizer_momentum = [0.0001], 
        # activation_fn = ['LogSigmoid', 'CELU', 'PReLU', 'ReLU6', 'Softsign'],
        activation_fn= ['Softsign'],
        # net_arch= [[64,64], [64,64,64,64]],
        net_arch= [[64,64]],

        # episodes= [1, 3],
        episodes= [1],

        reward_multiplier_combo_noaction= [0.1],
        # reward_multiplier_combo_noaction= [0.01, 0.1],
        reward_multiplier_combo_positionprofitpercentage= [0.001],
        # reward_multiplier_combo_positionprofitpercentage= [0, 0.001, 0.01, 0.1, 1],
        reward_multiplier_combo_buy= [1],
        # reward_multiplier_combo_buy= [0.01, 0.1, 1],
        reward_multiplier_combo_sell= [100],
        # reward_multiplier_combo_sell= [1, 10, 100, 1000]
        reward_multiplier_buy_sell_sold= [1],
        reward_multiplier_buy_sell_bought= [0],
        reward_multiplier_combo_actions_sell_hit= [0],
        reward_multiplier_combo_actions_sell_miss= [0],
        reward_multiplier_combo_actions_buy_threshold= [0],
        reward_multiplier_combo_actions_buy_hit= [0],
        reward_multiplier_combo_actions_buy_miss= [0],
        reward_multiplier_combo_actions_hold_buy_threshold = [0],
        reward_multiplier_combo_actions_hold_cash_hit= [0],
        reward_multiplier_combo_actions_hold_cash_miss= [0],
        reward_multiplier_combo_actions_hold_position_hit= [0],
        reward_multiplier_combo_actions_hold_position_miss= [0],
    )
)

model_rl_comboall_adamax = ModelConfigSearch(
    model_type= "rl",
    model_rl= ModelRLConfigSearch(
        model_name= ["dqn"],
        reward_model= ["combo_all"],
        learning_rate= [0.01],
        batch_size= [128],
        buffer_size = [1_000],
        gamma = [0.995],
        tau = [0],
        exploration_fraction = [0.05],
        exploration_final_eps = [0.1],
        learning_starts= [10_000],
        train_freq= [8],
        gradient_steps= [-1],
        target_update_interval= [10_000],
        max_grad_norm= [1000],
        optimizer_class = ['Adamax'],
        optimizer_eps = [0.001],
        optimizer_weight_decay = [0.00001],
        optimizer_centered = [True],
        optimizer_alpha = [0.9],
        optimizer_momentum = [0.0001], 
        activation_fn= ['CELU'],
        net_arch= [[64,64]],

        episodes= [1],

        reward_multiplier_combo_noaction= [0.1],
        reward_multiplier_combo_positionprofitpercentage= [0.001],
        reward_multiplier_combo_buy= [1],
        reward_multiplier_combo_sell= [100],
        reward_multiplier_buy_sell_sold= [1],
        reward_multiplier_buy_sell_bought= [0],
        reward_multiplier_combo_actions_sell_hit= [0],
        reward_multiplier_combo_actions_sell_miss= [0],
        reward_multiplier_combo_actions_buy_threshold= [0],
        reward_multiplier_combo_actions_buy_hit= [0],
        reward_multiplier_combo_actions_buy_miss= [0],
        reward_multiplier_combo_actions_hold_buy_threshold = [0],
        reward_multiplier_combo_actions_hold_cash_hit= [0],
        reward_multiplier_combo_actions_hold_cash_miss= [0],
        reward_multiplier_combo_actions_hold_position_hit= [0],
        reward_multiplier_combo_actions_hold_position_miss= [0],

        # checkpoints_folder='trials/',
        # checkpoint_to_load='rl_dqn_combo_all_0.01_0.995_Adamax_CELU_[64, 64]_1_1716187086.833314'
        checkpoints_folder='checkpoints/',
        checkpoint_to_load='rl_dqn_combo_all_0.005_1000_4_1000000_0.995_Adamax_CELU_[64, 64, 64, 64]_1_1716768803.222076'
    )
)

model_rl_comboall_rmsprop = ModelConfigSearch(
    model_type= "rl",
    model_rl= ModelRLConfigSearch(
        model_name= ["dqn"],
        reward_model= ["combo_all"],
        learning_rate= [0.01],
        batch_size= [128],
        buffer_size = [1_000],
        gamma = [0.995],
        tau = [0],
        exploration_fraction = [0.05],
        exploration_final_eps = [0.1],
        learning_starts= [10_000],
        train_freq= [8],
        gradient_steps= [-1],
        target_update_interval= [10_000],
        max_grad_norm= [1000],
        optimizer_class = ['RMSprop'],
        optimizer_eps = [0.001],
        optimizer_weight_decay = [0.00001],
        optimizer_centered = [True],
        optimizer_alpha = [0.9],
        optimizer_momentum = [0.0001], 
        activation_fn= ['CELU'],
        net_arch= [[64,64]],

        episodes= [1],

        reward_multiplier_combo_noaction= [0.1],
        reward_multiplier_combo_positionprofitpercentage= [0.001],
        reward_multiplier_combo_buy= [1],
        reward_multiplier_combo_sell= [100],
        reward_multiplier_buy_sell_sold= [1],
        reward_multiplier_buy_sell_bought= [0],
        reward_multiplier_combo_actions_sell_hit= [0],
        reward_multiplier_combo_actions_sell_miss= [0],
        reward_multiplier_combo_actions_buy_threshold= [0],
        reward_multiplier_combo_actions_buy_hit= [0],
        reward_multiplier_combo_actions_buy_miss= [0],
        reward_multiplier_combo_actions_hold_buy_threshold = [0],
        reward_multiplier_combo_actions_hold_cash_hit= [0],
        reward_multiplier_combo_actions_hold_cash_miss= [0],
        reward_multiplier_combo_actions_hold_position_hit= [0],
        reward_multiplier_combo_actions_hold_position_miss= [0],

        checkpoints_folder='trials/',
        checkpoint_to_load='rl_dqn_combo_all_0.01_0.995_RMSprop_CELU_[64, 64]_1_1716135233.756086'
    )
)

model_rl_comboall_adamax_min = ModelConfigSearch(
    model_type= "rl",
    model_rl= ModelRLConfigSearch(
        model_name= ["dqn"],
        reward_model= ["combo_all"],
        learning_rate= [0.01],
        batch_size= [128],
        buffer_size = [1_000],
        gamma = [0.995],
        tau = [0],
        exploration_fraction = [0.05],
        exploration_final_eps = [0.1],
        learning_starts= [10_000],
        train_freq= [8],
        gradient_steps= [-1],
        target_update_interval= [10_000],
        max_grad_norm= [1000],
        optimizer_class = ['Adamax'],
        optimizer_eps = [0.001],
        optimizer_weight_decay = [0.00001],
        optimizer_centered = [True],
        optimizer_alpha = [0.9],
        optimizer_momentum = [0.0001], 
        activation_fn= ['PReLU'],
        net_arch= [[64,64]],

        episodes= [1],

        reward_multiplier_combo_noaction= [0.1],
        reward_multiplier_combo_positionprofitpercentage= [0.001],
        reward_multiplier_combo_buy= [1],
        reward_multiplier_combo_sell= [100],
        reward_multiplier_buy_sell_sold= [1],
        reward_multiplier_buy_sell_bought= [0],
        reward_multiplier_combo_actions_sell_hit= [0],
        reward_multiplier_combo_actions_sell_miss= [0],
        reward_multiplier_combo_actions_buy_threshold= [0],
        reward_multiplier_combo_actions_buy_hit= [0],
        reward_multiplier_combo_actions_buy_miss= [0],
        reward_multiplier_combo_actions_hold_buy_threshold = [0],
        reward_multiplier_combo_actions_hold_cash_hit= [0],
        reward_multiplier_combo_actions_hold_cash_miss= [0],
        reward_multiplier_combo_actions_hold_position_hit= [0],
        reward_multiplier_combo_actions_hold_position_miss= [0],

        checkpoints_folder='trials/',
        checkpoint_to_load='rl_dqn_combo_all_0.01_0.995_Adamax_PReLU_[64, 64]_1_1716165787.3580859'
    )
)

@dataclass
class ModelRLConfig:
    model_name: str
    reward_model: str
    learning_rate: float = 0.0001
    batch_size: int = 32
    buffer_size: int = 1_000_000
    gamma: float = 0.99
    tau: float = 1.0
    exploration_fraction: float = 0.1
    exploration_final_eps: float = 0.05
    learning_starts: int = 50000
    train_freq: int = 4
    gradient_steps: int = 1
    target_update_interval: int = 10000
    max_grad_norm: float = 10
    optimizer_class: str = 'Adam'
    optimizer_eps: float = 0.00000001
    optimizer_weight_decay: float = 0
    optimizer_centered: bool = False
    optimizer_alpha: float = 0.99
    optimizer_momentum: float = 0
    activation_fn: str = 'ReLU'
    net_arch: List[int] = field(default_factory=[])

    episodes: int = 1

    reward_multiplier_combo_noaction: float = 0
    reward_multiplier_combo_positionprofitpercentage: float = 0
    reward_multiplier_combo_buy: float = 0
    reward_multiplier_combo_sell: float = 0
    reward_multiplier_buy_sell_sold: float = 1
    reward_multiplier_buy_sell_bought: float = 0
    reward_multiplier_combo_actions_sell_hit: float = 0
    reward_multiplier_combo_actions_sell_miss: float = 0
    reward_multiplier_combo_actions_buy_threshold: float = 0
    reward_multiplier_combo_actions_buy_hit: float = 0
    reward_multiplier_combo_actions_buy_miss: float = 0
    reward_multiplier_combo_actions_hold_buy_threshold: float = 0
    reward_multiplier_combo_actions_hold_cash_hit: float = 0
    reward_multiplier_combo_actions_hold_cash_miss: float = 0
    reward_multiplier_combo_actions_hold_position_hit: float = 0
    reward_multiplier_combo_actions_hold_position_miss: float = 0

    progress_bar: bool = True
    checkpoints_folder: str = 'checkpoints/'
    checkpoint_to_load: str | None = None

def get_models_simple():
    # return [model_hodl, model_time, model_technical_bollinger, model_technical_kallmanfilter5, model_technical_kallmanfilter14, model_technical_kallmanfilter20]
    return [model_hodl, model_time_test, model_technical_kallmanfilter_test, model_technical_bollinger_test]

def get_models_rl_h():
    # return [model_rl_comboall_adamax]
    return [model_hodl, model_rl_h]
def get_models_rl_min():
    return [model_hodl, model_rl_min]

def get_models_all():
    # return [model_hodl]
    return [model_hodl, model_lstm] # currently LSTM doesn't trade and MLP errors
    # return [model_hodl, model_lstm, model_mlp]