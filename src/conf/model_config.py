from dataclasses import dataclass, field
from typing import List

@dataclass
class ModelRLConfigSearch:
    model_name: List[str] = field(default_factory=[]), # possible ["ppo", "a2c", "dqn"]
    reward_model: List[str] = field(default_factory=[]) # possible ["combo_all", "combo_actions", "combo_old"],
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
    custom_net_arch: List[List[str]] = field(default_factory=[])

    episodes: List[int] = field(default_factory=[])

    reward_multiplier_combo_noaction: List[float] = field(default_factory=[])
    reward_multiplier_combo_positionprofitpercentage: List[float] = field(default_factory=[])
    reward_multiplier_combo_buy: List[float] = field(default_factory=[])
    reward_multiplier_combo_sell: List[float] = field(default_factory=[])

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
    custom_net_arch: List[str] = field(default_factory=[])

    episodes: int = 1

    reward_multiplier_combo_noaction: float = 0
    reward_multiplier_combo_positionprofitpercentage: float = 0
    reward_multiplier_combo_buy: float = 0
    reward_multiplier_combo_sell: float = 0

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
    iterations_to_pick_best: int = 10
    model_rl: ModelRLConfig | None = None
    model_lstm: ModelLSTMConfig | None = None
    model_mlp: ModelMLPConfig | None = None
    model_technical: ModelTechnicalConfig | None = None
    model_time: ModelTimeConfig | None = None
    
    def is_deep(self) -> bool:
        return self.model_type == "rl" or self.model_type == "lstm" or self.model_type == "mlp"
    def is_hodl(self) -> bool:
        return self.model_type == "hodl"
    
# duel_dqn seems to get much better rewards than dqn, yet much lower profits on combo_actions_diff
# drawdown + profit percentage seems to be an ok indicator -> the model definitely learns something
# profit_percentage2 + profit_percentage3 + profit_percentage4 -> are all ok indicators, the model learns from them
# buy_sell_signal + buy_sell_signal2 -> are ok

# indicators1 - ["kallman15", "timeseriesMomentum7", "closenessTo1000", "closenessTo10000", "meanReversion10", "meanReversion15", "rsi5", "rsi10", "rsi15"]
# indicators2 - ["timeseriesMomentum7", "closenessTo1000", "closenessTo10000", "meanReversion10", "meanReversion15", "rsi10", "choppiness30"]
# indicators3 - ["kallman15", "timeseriesMomentum7", "closenessTo1000", "closenessTo10000", "meanReversion10", "meanReversion15", "rsi10", "choppiness30"]
# indicators4 - ["kallman15", "timeseriesMomentum7", "closenessTo1000", "closenessTo10000", "meanReversion10", "meanReversion15", "choppiness30", "bollinger15Low"]
# indicators5 - ["kallman15", "timeseriesMomentum7", "closenessTo1000", "closenessTo10000", "meanReversion10", "meanReversion15", "rsi10", "choppiness30", "bollinger10Mid"]
# indicators6 - ["kallman15", "timeseriesMomentum7", "closenessTo1000", "closenessTo10000", "meanReversion10", "meanReversion15", "choppiness30", "bollinger10Mid"]
# indicators7 - ["kallman15", "timeseriesMomentum7", "closenessTo1000", "closenessTo10000", "meanReversion10", "meanReversion15", "choppiness30", "bollinger10Mid", "bollinger15Low"]


# BAD -> ["kallman15", "timeseriesMomentum7", "closenessTo1000", "closenessTo10000", "meanReversion10", "meanReversion15", "rsi5", "rsi15", "choppiness30"] -> extra rsi5
# BAD -> ["kallman15", "timeseriesMomentum7", "closenessTo1000", "closenessTo10000", "meanReversion10", "meanReversion15", "rsi5", "rsi10", "rsi15", "choppiness30"] -> extra rsi15
# BAD -> ["timeseriesMomentum7", "closenessTo1000", "closenessTo10000", "meanReversion10", "meanReversion15", "rsi10", "rsi15", "choppiness30"] -> extra rsi15
# BAD -> ["kallman15", "kallman30", "timeseriesMomentum7", "closenessTo1000", "closenessTo10000", "meanReversion10", "meanReversion15", "rsi10", "choppiness30"] -> extra kallman30
# BAD -> ["timeseriesMomentum7", "closenessTo1000", "closenessTo10000", "meanReversion10", "meanReversion15", "rsi10"] -> no choppiness30
# BAD -> ["kallman15", "timeseriesMomentum7", "closenessTo1000", "closenessTo10000", "meanReversion10", "meanReversion15", "rsi10", "choppiness30", "bollinger15Low"] -> extra bollinger15Low
# BAD -> ["kallman15", "timeseriesMomentum7", "closenessTo1000", "closenessTo10000", "meanReversion10", "meanReversion15", "rsi10", "choppiness30", "donchianChannels5Low"] -> extra donchianChannels5Low
# BAD -> ["kallman15", "timeseriesMomentum7", "closenessTo1000", "closenessTo10000", "meanReversion10", "meanReversion15", "rsi10", "choppiness30", "donchianChannels10High"] -> extra donchianChannels10High

# BatchNorm1d single with single dropout improves things a lot
# Dropout0.5 seems too strong, but had single great results -> experiment more later
# DropConnectLinear got some great results, placing it at the beginning can lead to no learning, but also great results, at the beginning it seems to improve things -> experiment more
# LayerNorm seems to improve slightly, just 1 is good enough (at the beginning rather than end)
# ResidualBlock seems to improve slightly, just 1 is good enough (at the beginning rather than end)
# Dense+DenseBlock4, can get some single great results -> experiment more later
# GRUFull seems to improve
# GRULocal improves
# LSTFull can have some good results, but very few moves - retry
# LSTMLocalN not sure
# ScaledDotProductAttention improves
# AdditiveAttention improves
# MultiHeadAttention doesn't learn
# SelfAttention doesn't learn

# BAD -> weight_norm
# BAD -> spectral_norm
# BAD -> LSTFullN
# BAD -> GRULocal2+4

# ["duel-dqn", "dqn"] 1/3 -> policy='MlpPolicy', activation_fn, normalize_images?, optimizer_class (has to be good like RMSprop)

# duel-dqn x [net_arch] 10/11
# dqn x [net_arch] 3/11
# ["dqn"] x [indicators] 2/16
# ["duel-dqn"] x [indicators] 7/16

# TODO: DGWO optim

model_rl_h = ModelConfigSearch(
    model_type= "rl",
    model_rl= ModelRLConfigSearch(
        # model_name= ["ppo", "reppo", "trpo", "a2c", "ars", "ars-mlp", "qrdqn", "dqn-sbx", "dqn"],
        # model_name= ["r-dqn", "dqn-lstm", "iqn", "dqn"], # very slow (12it/s, 30it/s, 10it/s), but can work very well
        # model_name= ["dqn-custom", "duel-dqn-custom"],
        # model_name= ["duel-dqn"],
        # model_name= ["duel-dqn-custom"],
        # model_name= ["duel-dqn2", "duel-dqn"],
        # model_name= ["dqn-sbx", "duel-dqn", "dqn"],
        model_name= ["dqn"],
        # model_name= ["duel-dqn"],

        # model_name= ["dqn-lstm"],

        # reward_model= ["combo_all", "combo_actions", "combo_old"],
        reward_model= ["combo_actions2"],

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

        # optimizer_class = ['AdamW', 'NAdam', 'RMSprop'], # TODO check with finetuning 
        optimizer_class = ['RMSprop'],
        # optimizer_class = ['default'],
        # optimizer_eps = [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 0.2],
        optimizer_eps = [0.5],
        # optimizer_weight_decay = [0, 0.00000001, 0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1],
        optimizer_weight_decay = [0.00000001],
        # optimizer_centered = [True, False],
        optimizer_centered = [True],
        # optimizer_alpha = [0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 0.99],
        optimizer_alpha = [0.9],
        # optimizer_momentum = [0.0001, 0.1, 0.000001],
        optimizer_momentum = [0.0001], 

        # activation_fn = ['LogSigmoid', 'CELU', 'PReLU', 'ReLU6', 'Softsign'],
        activation_fn= ['CELU'],
        net_arch= [[256,256]], # worth it
        # net_arch= [[512,256,64,256,512]], # worth it
        # net_arch= [[64,64], [256,256], [256, 64, 32, 16], [256, 64, 32, 16, 8], [512, 256, 64, 32], [512, 256, 64, 32, 16], [64,64,64], [256,256,256], [64,64,64,64], [256,256,256,256]], #<--- these are promising

        # net_arch= [[64,64,64,64]],
        custom_net_arch= [[""]],

        # custom_net_arch= [
        #     ["LSTMLocalN", "activation_fn", "Linear", "activation_fn", "Linear", "activation_fn", "Linear", "activation_fn", "Linear"],
        # ],

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


        # checkpoints_folder='checkpoints/',
        # checkpoint_to_load='rl_dqn_combo_actions_diff_0.0001_10000_256_1000000_0.9_Adamax_CELU_[64, 64, 64, 64]_1_1717946749.513737'
    )
)

# combo_actions, combo_actions2, combo_actions3

# [MIN 2022] ["duel-dqn"] x lr[0.00000001, 0.000001] x combo_actions2 x [32lookback] -> 1/3
# [MIN 2017] ["duel-dqn"] x lr[0.0000001] x combo_actions2 x [32lookback] -> none

model_rl_min = ModelConfigSearch(
    model_type= "rl",
    model_rl= ModelRLConfigSearch(
        # model_name= ["ppo", "a2c", "dqn", "qrdqn", "dqn-sbx"],
        # model_name= ["duel-dqn", "dqn-sbx", "dqn"],
        model_name= ["duel-dqn"],
        # model_name= ["dqn-sbx"],
        # model_name= ["rainbow-dqn", "dqn-lstm"],

        # reward_model= ["combo_all", "combo_actions2", "combo_old"],
        reward_model= ["combo_actions2"],

        # learning_rate= [0.00000001, 0.000001],
        learning_rate= [0.0000001],
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
        custom_net_arch= [[""]],

        episodes= [1],

        reward_multiplier_combo_noaction= [-1],
        reward_multiplier_combo_positionprofitpercentage= [10],
        reward_multiplier_combo_buy= [0.1],
        reward_multiplier_combo_sell= [1000],
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
        custom_net_arch= [[""]],

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
        custom_net_arch= [[""]],

        episodes= [1],

        reward_multiplier_combo_noaction= [0.1],
        reward_multiplier_combo_positionprofitpercentage= [0.001],
        reward_multiplier_combo_buy= [1],
        reward_multiplier_combo_sell= [100],

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
        custom_net_arch= [[""]],

        episodes= [1],

        reward_multiplier_combo_noaction= [0.1],
        reward_multiplier_combo_positionprofitpercentage= [0.001],
        reward_multiplier_combo_buy= [1],
        reward_multiplier_combo_sell= [100],

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
        custom_net_arch= [[""]],

        episodes= [1],

        reward_multiplier_combo_noaction= [0.1],
        reward_multiplier_combo_positionprofitpercentage= [0.001],
        reward_multiplier_combo_buy= [1],
        reward_multiplier_combo_sell= [100],

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
    custom_net_arch: List[str] = field(default_factory=[])

    episodes: int = 1

    reward_multiplier_combo_noaction: float = 0
    reward_multiplier_combo_positionprofitpercentage: float = 0
    reward_multiplier_combo_buy: float = 0
    reward_multiplier_combo_sell: float = 0

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